"""
StyleGAN2 Discriminator with R1 Regularization
Optimized for RTX 3050 (4GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EqualizedConv2d(nn.Module):
    """Conv2d with equalized learning rate"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
    
    def forward(self, x):
        out = F.conv2d(x, self.weight * self.scale, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate"""
    def __init__(self, in_features, out_features, bias=True, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        
        self.lr_mul = lr_mul
        self.scale = (1 / math.sqrt(in_features)) * lr_mul
    
    def forward(self, x):
        out = F.linear(x, self.weight * self.scale)
        if self.bias is not None:
            out = out + self.bias * self.lr_mul
        return out


class MinibatchStdDev(nn.Module):
    """Minibatch standard deviation layer for better diversity"""
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        group_size = min(batch_size, self.group_size)
        
        # Reshape for group statistics
        y = x.reshape(group_size, -1, channels, height, width)
        
        # Calculate std dev across group
        y = y - y.mean(0, keepdim=True)
        y = torch.sqrt(y.pow(2).mean(0) + 1e-8)
        
        # Average across channels and spatial dimensions
        y = y.mean([1, 2, 3], keepdim=True)
        y = y.repeat(group_size, 1, height, width)
        
        # Concatenate with input
        return torch.cat([x, y], dim=1)


class DiscriminatorBlock(nn.Module):
    """Discriminator residual block with downsampling"""
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.downsample = downsample
        
        self.conv1 = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2)
        
        if downsample:
            self.downsample_layer = nn.AvgPool2d(2)
        
        # Residual connection
        self.skip = EqualizedConv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        residual = self.skip(x)
        
        # Main path
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.lrelu(out)
        
        # Add residual
        out = (out + residual) / math.sqrt(2)
        
        # Downsample
        if self.downsample:
            out = self.downsample_layer(out)
        
        return out


class StyleGAN2Discriminator(nn.Module):
    """
    StyleGAN2 Discriminator
    Takes 256x256 RGB images and outputs real/fake score
    
    Uses R1 regularization instead of gradient penalty
    """
    def __init__(self, img_resolution=256, img_channels=3):
        super().__init__()
        
        self.img_resolution = img_resolution
        
        # From RGB layer
        self.from_rgb = EqualizedConv2d(img_channels, 32, kernel_size=1)
        
        # Discriminator blocks (progressive downsampling)
        # 256x256 -> 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(32, 64, downsample=True),    # 256 -> 128
            DiscriminatorBlock(64, 128, downsample=True),   # 128 -> 64
            DiscriminatorBlock(128, 256, downsample=True),  # 64 -> 32
            DiscriminatorBlock(256, 512, downsample=True),  # 32 -> 16
            DiscriminatorBlock(512, 512, downsample=True),  # 16 -> 8
            DiscriminatorBlock(512, 512, downsample=True),  # 8 -> 4
        ])
        
        # Minibatch std dev for final block
        self.mbstd = MinibatchStdDev(group_size=4)
        
        # Final layers
        self.final_conv = EqualizedConv2d(513, 512, kernel_size=3, padding=1)  # +1 for mbstd
        self.final_linear = EqualizedLinear(512 * 4 * 4, 512)
        self.output_linear = EqualizedLinear(512, 1)
        
        self.lrelu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        """
        Args:
            x: Input images [batch, 3, 256, 256] in range [-1, 1]
        Returns:
            scores: Real/fake scores [batch, 1]
        """
        # From RGB
        x = self.from_rgb(x)
        x = self.lrelu(x)
        
        # Progressive downsampling through blocks
        for block in self.blocks:
            x = block(x)
        
        # Minibatch std dev
        x = self.mbstd(x)
        
        # Final conv
        x = self.final_conv(x)
        x = self.lrelu(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.final_linear(x)
        x = self.lrelu(x)
        x = self.output_linear(x)
        
        return x


def r1_regularization(real_images, discriminator):
    """
    R1 regularization for StyleGAN2
    Penalizes discriminator gradient on real images
    
    Args:
        real_images: Real images with requires_grad=True
        discriminator: Discriminator model
    
    Returns:
        r1_penalty: Gradient penalty value
    """
    real_scores = discriminator(real_images)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=real_scores.sum(),
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # R1 penalty: ||∇D(x)||²
    r1_penalty = gradients.pow(2).reshape(gradients.size(0), -1).sum(1).mean()
    
    return r1_penalty


if __name__ == '__main__':
    # Test discriminator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    discriminator = StyleGAN2Discriminator().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256, device=device)
    scores = discriminator(x)
    print(f"Output shape: {scores.shape}")  # Should be [2, 1]
    
    # Test R1 regularization
    x.requires_grad_(True)
    r1_loss = r1_regularization(x, discriminator)
    print(f"R1 penalty: {r1_loss.item():.4f}")
