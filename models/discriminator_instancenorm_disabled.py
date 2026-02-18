"""
Discriminator with InstanceNorm DISABLED in forward pass
Loads original checkpoint, but skips InstanceNorm during training
This fixes WGAN-GP without changing layer indices
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features_d=16):
        super(Discriminator, self).__init__()

        # Keep SAME architecture as original (for loading checkpoint)
        # But we'll skip InstanceNorm in forward pass
        self.conv1 = spectral_norm(nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1))
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1))
        self.norm2 = nn.InstanceNorm2d(64, affine=True)  # Keep for loading, skip in forward
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        self.norm3 = nn.InstanceNorm2d(128, affine=True)  # Keep for loading, skip in forward
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.norm4 = nn.InstanceNorm2d(256, affine=True)  # Keep for loading, skip in forward
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1))
        self.norm5 = nn.InstanceNorm2d(512, affine=True)  # Keep for loading, skip in forward
        self.lrelu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv6 = spectral_norm(nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1))
        self.norm6 = nn.InstanceNorm2d(1024, affine=True)  # Keep for loading, skip in forward
        self.lrelu6 = nn.LeakyReLU(0.2, inplace=True)

        self.conv7 = spectral_norm(nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0))

    def forward(self, img):
        # Forward pass WITHOUT InstanceNorm (skip norm layers)
        x = self.lrelu1(self.conv1(img))
        x = self.lrelu2(self.conv2(x))  # ← Skip norm2
        x = self.lrelu3(self.conv3(x))  # ← Skip norm3
        x = self.lrelu4(self.conv4(x))  # ← Skip norm4
        x = self.lrelu5(self.conv5(x))  # ← Skip norm5
        x = self.lrelu6(self.conv6(x))  # ← Skip norm6
        x = self.conv7(x)
        return x.view(x.size(0), -1)

if __name__ == '__main__':
    # Test
    batch_size = 4
    disc = Discriminator()
    img = torch.randn(batch_size, 3, 256, 256)
    output = disc(img)
    print(f"Output shape: {output.shape}")
    print(f"✓ Discriminator with disabled InstanceNorm test passed!")
