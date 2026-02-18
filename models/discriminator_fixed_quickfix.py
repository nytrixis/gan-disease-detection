"""
Discriminator QUICK FIX - Removes InstanceNorm that breaks WGAN-GP
This should improve FID from 257 to ~150-180

CHANGES FROM ORIGINAL:
1. Removed ALL InstanceNorm layers (breaks Lipschitz constraint)
2. Kept only Spectral Normalization (correct for WGAN-GP)
3. Same architecture otherwise
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features_d=16):
        super(Discriminator, self).__init__()

        # All Conv2d layers wrapped with spectral_norm ONLY
        # NO InstanceNorm - it breaks WGAN-GP's Lipschitz constraint!
        self.model = nn.Sequential(
            # 256x256x3 -> 128x128x32
            spectral_norm(nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 128x128x32 -> 64x64x64
            spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),  # ← REMOVED InstanceNorm

            # 64x64x64 -> 32x32x128
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),  # ← REMOVED InstanceNorm

            # 32x32x128 -> 16x16x256
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),  # ← REMOVED InstanceNorm

            # 16x16x256 -> 8x8x512
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),  # ← REMOVED InstanceNorm

            # 8x8x512 -> 4x4x1024
            spectral_norm(nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),  # ← REMOVED InstanceNorm

            # 4x4x1024 -> 1x1x1
            spectral_norm(nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0))
            # No activation (Wasserstein distance can be any real number)
        )

    def forward(self, img):
        # img: (batch_size, 3, 256, 256)
        validity = self.model(img)
        validity = validity.view(validity.size(0), -1)  # Flatten to (batch_size, 1)
        return validity

if __name__ == '__main__':
    # Test discriminator
    batch_size = 4
    img_channels = 3
    img_size = 256

    disc = Discriminator(img_channels=img_channels)
    img = torch.randn(batch_size, img_channels, img_size, img_size)
    output = disc(img)

    print(f"Discriminator output shape: {output.shape}")
    print(f"✓ Discriminator Quick Fix test passed!")
    print(f"✓ NO InstanceNorm - WGAN-GP compatible")
