"""
Discriminator/Critic architecture for WGAN-GP
Takes 256x256x3 images and outputs single scalar (Wasserstein distance estimate)
"""

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features_d=16):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # 256x256x3 -> 128x128x32
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 128x128x32 -> 64x64x64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64x64 -> 32x32x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32x128 -> 16x16x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16x256 -> 8x8x512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8x512 -> 4x4x1024
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4x1024 -> 1x1x1
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0)
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
    print(f"✓ Discriminator test passed!")
