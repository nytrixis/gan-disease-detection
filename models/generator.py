"""
Generator architecture for WGAN-GP
Generates 256x256x3 images from 100D latent vectors
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, features_g=16):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim

        # Initial projection: 100D -> 4x4x1024
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 1024),
            nn.BatchNorm1d(4 * 4 * 1024),
            nn.ReLU(True)
        )

        # Transposed convolutions to upsample
        self.model = nn.Sequential(
            # 4x4x1024 -> 8x8x512
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 8x8x512 -> 16x16x256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 16x16x256 -> 32x32x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 32x32x128 -> 64x64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64x64x64 -> 128x128x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 128x128x32 -> 256x256x3
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, z):
        # z: (batch_size, latent_dim)
        x = self.initial(z)
        x = x.view(x.size(0), 1024, 4, 4)
        x = self.model(x)
        return x

if __name__ == '__main__':
    # Test generator
    batch_size = 4
    latent_dim = 100

    gen = Generator(latent_dim=latent_dim)
    z = torch.randn(batch_size, latent_dim)
    fake_img = gen(z)

    print(f"Generator output shape: {fake_img.shape}")
    print(f"Output range: [{fake_img.min():.3f}, {fake_img.max():.3f}]")
    print(f"✓ Generator test passed!")
