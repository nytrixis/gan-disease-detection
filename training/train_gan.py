"""
WGAN-GP Training Script
Optimized for RTX 3050 (4GB VRAM)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.generator import Generator
from models.discriminator import Discriminator
from preprocessing.data_loader import get_dataloader, set_seed

# Hyperparameters (optimized for RTX 3050)
HYPERPARAMETERS = {
    'latent_dim': 100,
    'batch_size': 8,  # Reduced for RTX 3050
    'lr_generator': 0.0001,
    'lr_critic': 0.0001,
    'beta1': 0.5,
    'beta2': 0.999,
    'lambda_gp': 10,  # Gradient penalty coefficient
    'critic_iterations': 5,  # Train critic 5 times per generator iteration
    'num_epochs': 200,  # Reduced from 240
    'checkpoint_interval': 2000,
    'sample_interval': 500,
    'seed': 42
}

def gradient_penalty(critic, real, fake, device):
    """Calculate gradient penalty for WGAN-GP"""
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated)

    # Compute gradients
    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

def train_wgan_gp():
    """Train WGAN-GP on Dermatofibroma images"""

    print("="*60)
    print("WGAN-GP Training - Dermatofibroma Synthesis")
    print("="*60)

    # Set seed
    set_seed(HYPERPARAMETERS['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create output directories
    Path('data/synthetic/raw').mkdir(parents=True, exist_ok=True)
    Path('results/checkpoints/wgan_gp').mkdir(parents=True, exist_ok=True)
    Path('results/samples').mkdir(parents=True, exist_ok=True)
    Path('logs/wgan_gp').mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter('logs/wgan_gp')
    print("✓ TensorBoard logging enabled: logs/wgan_gp")

    # Initialize models
    generator = Generator(latent_dim=HYPERPARAMETERS['latent_dim']).to(device)
    critic = Discriminator().to(device)

    # Optimizers
    optimizer_gen = optim.Adam(
        generator.parameters(),
        lr=HYPERPARAMETERS['lr_generator'],
        betas=(HYPERPARAMETERS['beta1'], HYPERPARAMETERS['beta2'])
    )
    optimizer_critic = optim.Adam(
        critic.parameters(),
        lr=HYPERPARAMETERS['lr_critic'],
        betas=(HYPERPARAMETERS['beta1'], HYPERPARAMETERS['beta2'])
    )

    # DataLoader (only Dermatofibroma images)
    dataloader = get_dataloader(
        split='train',
        class_filter='dermatofibroma',
        batch_size=HYPERPARAMETERS['batch_size'],
        shuffle=True,
        for_gan=True,
        num_workers=2
    )

    print(f"\nTraining samples: {len(dataloader.dataset)}")
    print(f"Batch size: {HYPERPARAMETERS['batch_size']}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Training loop
    global_step = 0
    fixed_noise = torch.randn(25, HYPERPARAMETERS['latent_dim']).to(device)

    print("\nStarting training...")
    print("="*60)

    for epoch in range(HYPERPARAMETERS['num_epochs']):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{HYPERPARAMETERS['num_epochs']}")

        for batch_idx, (real_images, _) in enumerate(pbar):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train Critic
            critic_losses = []
            for _ in range(HYPERPARAMETERS['critic_iterations']):
                noise = torch.randn(batch_size, HYPERPARAMETERS['latent_dim']).to(device)
                fake = generator(noise)

                critic_real = critic(real_images).reshape(-1)
                critic_fake = critic(fake.detach()).reshape(-1)
                gp = gradient_penalty(critic, real_images, fake, device)

                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + HYPERPARAMETERS['lambda_gp'] * gp
                )

                critic.zero_grad()
                loss_critic.backward()
                optimizer_critic.step()

                critic_losses.append(loss_critic.item())

            # Train Generator
            noise = torch.randn(batch_size, HYPERPARAMETERS['latent_dim']).to(device)
            fake = generator(noise)
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)

            generator.zero_grad()
            loss_gen.backward()
            optimizer_gen.step()

            # Update progress bar
            wasserstein_distance = torch.mean(critic_real) - torch.mean(critic_fake)
            pbar.set_postfix({
                'C_loss': f"{np.mean(critic_losses):.4f}",
                'G_loss': f"{loss_gen.item():.4f}",
                'W_dist': f"{wasserstein_distance.item():.4f}"
            })
            
            # Log to TensorBoard
            writer.add_scalar('Loss/Critic', np.mean(critic_losses), global_step)
            writer.add_scalar('Loss/Generator', loss_gen.item(), global_step)
            writer.add_scalar('Wasserstein_Distance', wasserstein_distance.item(), global_step)

            # Save samples
            if global_step % HYPERPARAMETERS['sample_interval'] == 0:
                with torch.no_grad():
                    fake_samples = generator(fixed_noise)
                    fake_samples = (fake_samples + 1) / 2  # Denormalize to [0, 1]
                    save_image(
                        fake_samples,
                        f"results/samples/step_{global_step:06d}.png",
                        nrow=5,
                        normalize=False
                    )

            # Save checkpoint
            if global_step % HYPERPARAMETERS['checkpoint_interval'] == 0 and global_step > 0:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'generator_state_dict': generator.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'optimizer_gen_state_dict': optimizer_gen.state_dict(),
                    'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                }, f'results/checkpoints/wgan_gp/checkpoint_step_{global_step}.pth')

            global_step += 1

    # Save final model
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'critic_state_dict': critic.state_dict(),
    }, 'results/checkpoints/wgan_gp/final_model.pth')
    
    # Close TensorBoard writer
    writer.close()

    print("\n" + "="*60)
    print("✓ Training complete!")
    print(f"✓ Final model saved to results/checkpoints/wgan_gp/final_model.pth")
    print("="*60)

    return generator

if __name__ == '__main__':
    import numpy as np
    train_wgan_gp()
