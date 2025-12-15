"""
StyleGAN2 Generator with Adaptive Instance Normalization (AdaIN)
Optimized for RTX 3050 (4GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class MappingNetwork(nn.Module):
    """Maps latent z to intermediate latent w"""
    def __init__(self, z_dim=512, w_dim=512, n_layers=8):
        super().__init__()
        
        layers = []
        for i in range(n_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.append(EqualizedLinear(in_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, z):
        # Normalize input
        z = F.normalize(z, dim=1)
        w = self.mapping(z)
        return w


class AdaptiveInstanceNorm(nn.Module):
    """Adaptive Instance Normalization (AdaIN)"""
    def __init__(self, num_features, w_dim=512):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        
        # Style modulation
        self.style_scale = EqualizedLinear(w_dim, num_features, bias=True)
        self.style_bias = EqualizedLinear(w_dim, num_features, bias=True)
    
    def forward(self, x, w):
        # Normalize
        x = self.norm(x)
        
        # Apply style
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        
        return x * (1 + style_scale) + style_bias


class NoiseInjection(nn.Module):
    """Inject random noise for stochastic variation"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)
        
        return x + self.weight * noise


class StyleBlock(nn.Module):
    """Style-based synthesis block"""
    def __init__(self, in_channels, out_channels, w_dim=512, upsample=True):
        super().__init__()
        self.upsample = upsample
        
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.conv1 = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.noise1 = NoiseInjection()
        self.adain1 = AdaptiveInstanceNorm(out_channels, w_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)
        
        self.conv2 = EqualizedConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.noise2 = NoiseInjection()
        self.adain2 = AdaptiveInstanceNorm(out_channels, w_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)
    
    def forward(self, x, w, noise1=None, noise2=None):
        if self.upsample:
            x = self.upsample_layer(x)
        
        # First conv block
        x = self.conv1(x)
        x = self.noise1(x, noise1)
        x = self.adain1(x, w)
        x = self.lrelu1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.noise2(x, noise2)
        x = self.adain2(x, w)
        x = self.lrelu2(x)
        
        return x


class ToRGB(nn.Module):
    """Convert features to RGB image"""
    def __init__(self, in_channels, w_dim=512):
        super().__init__()
        self.conv = EqualizedConv2d(in_channels, 3, kernel_size=1)
        self.adain = AdaptiveInstanceNorm(in_channels, w_dim)
    
    def forward(self, x, w):
        x = self.adain(x, w)
        x = self.conv(x)
        return x


class StyleGAN2Generator(nn.Module):
    """
    StyleGAN2 Generator
    Generates 256x256 RGB images from 512D latent code
    
    Architecture optimized for RTX 3050 (4GB VRAM)
    """
    def __init__(self, z_dim=512, w_dim=512, img_resolution=256, img_channels=3):
        super().__init__()
        
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        
        # Mapping network: z -> w
        self.mapping = MappingNetwork(z_dim, w_dim, n_layers=8)
        
        # Constant input (4x4)
        self.const_input = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        # Synthesis blocks
        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        self.blocks = nn.ModuleList([
            StyleBlock(512, 512, w_dim, upsample=False),  # 4x4 (constant input)
            StyleBlock(512, 512, w_dim, upsample=True),   # 8x8
            StyleBlock(512, 512, w_dim, upsample=True),   # 16x16
            StyleBlock(512, 256, w_dim, upsample=True),   # 32x32
            StyleBlock(256, 128, w_dim, upsample=True),   # 64x64
            StyleBlock(128, 64, w_dim, upsample=True),    # 128x128
            StyleBlock(64, 32, w_dim, upsample=True),     # 256x256
        ])
        
        # RGB output layers for each resolution (for progressive training support)
        self.to_rgb = nn.ModuleList([
            ToRGB(512, w_dim),
            ToRGB(512, w_dim),
            ToRGB(512, w_dim),
            ToRGB(256, w_dim),
            ToRGB(128, w_dim),
            ToRGB(64, w_dim),
            ToRGB(32, w_dim),
        ])
    
    def forward(self, z, truncation_psi=1.0, noise_mode='random'):
        """
        Args:
            z: Latent code [batch, z_dim]
            truncation_psi: Truncation trick for controlling diversity (0.5-1.0)
            noise_mode: 'random' or 'const'
        """
        # Map to W space
        w = self.mapping(z)
        
        # Truncation trick for better quality
        if truncation_psi < 1.0:
            w = truncation_psi * w
        
        # Start with constant input
        batch_size = z.shape[0]
        x = self.const_input.repeat(batch_size, 1, 1, 1)
        
        # Generate through blocks
        for i, (block, to_rgb_layer) in enumerate(zip(self.blocks, self.to_rgb)):
            x = block(x, w)
            
            # Convert to RGB at final layer
            if i == len(self.blocks) - 1:
                img = to_rgb_layer(x, w)
        
        # Tanh activation for [-1, 1] range
        img = torch.tanh(img)
        
        return img
    
    def generate(self, batch_size, device, truncation_psi=0.7):
        """Generate images from random latent codes"""
        z = torch.randn(batch_size, self.z_dim, device=device)
        return self.forward(z, truncation_psi=truncation_psi)


if __name__ == '__main__':
    # Test generator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = StyleGAN2Generator(z_dim=512, w_dim=512).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator parameters: {total_params:,}")
    
    # Test forward pass
    z = torch.randn(2, 512, device=device)
    img = generator(z)
    print(f"Output shape: {img.shape}")  # Should be [2, 3, 256, 256]
    print(f"Output range: [{img.min():.2f}, {img.max():.2f}]")  # Should be ~[-1, 1]
