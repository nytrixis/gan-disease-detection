"""
WGAN-GP Training Script
Optimized for RTX 3050 (4GB VRAM)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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

# Hyperparameters (optimized for RTX 3050 + 545 training samples)
HYPERPARAMETERS = {
    'latent_dim': 100,
    'batch_size': 16,  # Increased from 8 - critical for stability
    'lr_generator': 0.0004,  # TTUR: 4x faster than critic
    'lr_critic': 0.0001,     # TTUR: slower to prevent domination
    'beta1': 0.0,            # TTUR: Use 0.0 instead of 0.5
    'beta2': 0.999,
    'lambda_gp_start': 10,   # Adaptive GP: start value
    'lambda_gp_end': 20,     # Adaptive GP: end value (epoch 100)
    'critic_iterations': 2,  # Reduced from 3 to prevent discriminator domination
    'num_epochs': 100,       # Reduced from 200 - plateau happens by epoch 60
    'checkpoint_interval': 5,  # Save every 5 epochs for FID tracking
    'sample_interval': 500,
    'fid_interval': 5,       # Calculate FID every 5 epochs
    'early_stopping_patience': 3,  # Stop if FID doesn't improve for 3 checks
    'resume_checkpoint': None,  # Path to resume from (e.g., 'results/checkpoints/wgan_gp/best_fid_model.pth')
    'reset_schedulers': True,  # Reset LR schedulers when resuming (fresh learning rates)
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

def get_adaptive_lambda_gp(epoch, total_epochs, lambda_start, lambda_end):
    """Calculate adaptive gradient penalty coefficient
    Increases from lambda_start to lambda_end over training
    Early training: loose constraints for exploration
    Late training: tight constraints for stability
    """
    progress = epoch / total_epochs
    lambda_gp = lambda_start + (lambda_end - lambda_start) * progress
    return lambda_gp

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

    # Optimizers with TTUR (Two Time-Scale Update Rule)
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
    
    # Learning rate schedulers (Cosine Annealing)
    scheduler_gen = CosineAnnealingLR(optimizer_gen, T_max=HYPERPARAMETERS['num_epochs'], eta_min=1e-6)
    scheduler_critic = CosineAnnealingLR(optimizer_critic, T_max=HYPERPARAMETERS['num_epochs'], eta_min=1e-6)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if HYPERPARAMETERS['resume_checkpoint'] and os.path.exists(HYPERPARAMETERS['resume_checkpoint']):
        print(f"\n{'='*60}")
        print(f"Resuming from checkpoint: {HYPERPARAMETERS['resume_checkpoint']}")
        checkpoint = torch.load(HYPERPARAMETERS['resume_checkpoint'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        
        if not HYPERPARAMETERS['reset_schedulers']:
            optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
            optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
            if 'scheduler_gen_state_dict' in checkpoint:
                scheduler_gen.load_state_dict(checkpoint['scheduler_gen_state_dict'])
                scheduler_critic.load_state_dict(checkpoint['scheduler_critic_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"✓ Resumed from epoch {start_epoch}")
            print(f"✓ Loaded optimizer and scheduler states")
        else:
            print(f"✓ Loaded model weights from epoch {checkpoint.get('epoch', 0)}")
            print(f"✓ Reset optimizers and schedulers (fresh learning rates)")
            start_epoch = 0  # Start epoch numbering from 0 with new training phase
        
        print(f"{'='*60}\n")

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
    print(f"\n✓ TTUR enabled: LR_G={HYPERPARAMETERS['lr_generator']}, LR_C={HYPERPARAMETERS['lr_critic']}")
    print(f"✓ n_critic reduced to {HYPERPARAMETERS['critic_iterations']} for small dataset")
    print(f"✓ Adaptive λ_GP: {HYPERPARAMETERS['lambda_gp_start']} → {HYPERPARAMETERS['lambda_gp_end']}")
    print(f"✓ FID calculation every {HYPERPARAMETERS['fid_interval']} epochs")
    print(f"✓ Early stopping patience: {HYPERPARAMETERS['early_stopping_patience']} checks")
    print(f"✓ Spectral Normalization enabled in discriminator")

    # Training loop
    global_step = 0
    fixed_noise = torch.randn(25, HYPERPARAMETERS['latent_dim']).to(device)
    
    # For FID calculation and early stopping
    best_fid = float('inf')
    fid_scores = {}
    patience_counter = 0
    best_epoch = 0

    print("\nStarting training...")
    print("="*60)

    for epoch in range(start_epoch, start_epoch + HYPERPARAMETERS['num_epochs']):
        # Calculate adaptive lambda for gradient penalty
        current_lambda_gp = get_adaptive_lambda_gp(
            epoch - start_epoch,
            HYPERPARAMETERS['num_epochs'],
            HYPERPARAMETERS['lambda_gp_start'],
            HYPERPARAMETERS['lambda_gp_end']
        )
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{start_epoch + HYPERPARAMETERS['num_epochs']}")

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
                    + current_lambda_gp * gp
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
            d_real_mean = torch.mean(critic_real).item()
            d_fake_mean = torch.mean(critic_fake).item()
            
            pbar.set_postfix({
                'C_loss': f"{np.mean(critic_losses):.4f}",
                'G_loss': f"{loss_gen.item():.4f}",
                'W_dist': f"{wasserstein_distance.item():.4f}",
                'λ_GP': f"{current_lambda_gp:.1f}"
            })
            
            # Log to TensorBoard
            writer.add_scalar('Loss/Critic', np.mean(critic_losses), global_step)
            writer.add_scalar('Loss/Generator', loss_gen.item(), global_step)
            writer.add_scalar('Wasserstein_Distance', wasserstein_distance.item(), global_step)
            writer.add_scalar('Gradient_Penalty', gp.item(), global_step)
            writer.add_scalar('Lambda_GP', current_lambda_gp, global_step)
            writer.add_scalar('D_real_mean', d_real_mean, global_step)
            writer.add_scalar('D_fake_mean', d_fake_mean, global_step)
            writer.add_scalar('LR/Generator', optimizer_gen.param_groups[0]['lr'], global_step)
            writer.add_scalar('LR/Critic', optimizer_critic.param_groups[0]['lr'], global_step)

            # Save samples (with unique naming to preserve existing samples)
            if global_step % HYPERPARAMETERS['sample_interval'] == 0:
                with torch.no_grad():
                    fake_samples = generator(fixed_noise)
                    fake_samples = (fake_samples + 1) / 2  # Denormalize to [0, 1]
                    save_image(
                        fake_samples,
                        f"results/samples/epoch_{epoch+1:03d}_step_{global_step:06d}.png",
                        nrow=5,
                        normalize=False
                    )

            global_step += 1
        
        # End of epoch: save checkpoint and calculate FID
        if (epoch + 1) % HYPERPARAMETERS['checkpoint_interval'] == 0:
            checkpoint_path = f'results/checkpoints/wgan_gp/epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_gen_state_dict': optimizer_gen.state_dict(),
                'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                'scheduler_gen_state_dict': scheduler_gen.state_dict(),
                'scheduler_critic_state_dict': scheduler_critic.state_dict(),
            }, checkpoint_path)
        
        # Calculate FID every N epochs
        if (epoch + 1) % HYPERPARAMETERS['fid_interval'] == 0:
            print(f"\n{'='*60}")
            print(f"Calculating FID at epoch {epoch+1}...")
            print(f"{'='*60}")
            
            # Generate temporary samples for FID
            temp_fid_dir = Path('data/synthetic/temp_fid')
            temp_fid_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear previous temp images
            for img in temp_fid_dir.glob('*.jpg'):
                img.unlink()
            
            # Generate 500 samples
            generator.eval()
            with torch.no_grad():
                for i in range(0, 500, HYPERPARAMETERS['batch_size']):
                    batch_size_fid = min(HYPERPARAMETERS['batch_size'], 500 - i)
                    noise = torch.randn(batch_size_fid, HYPERPARAMETERS['latent_dim']).to(device)
                    fake_imgs = generator(noise)
                    fake_imgs = (fake_imgs + 1) / 2  # Denormalize to [0, 1]
                    
                    for j in range(batch_size_fid):
                        save_image(fake_imgs[j], temp_fid_dir / f'sample_{i+j:04d}.jpg')
            generator.train()
            
            # Calculate FID
            try:
                from evaluation.calculate_fid import calculate_fid
                fid_score = calculate_fid(
                    real_images_path='data/splits/train/dermatofibroma',
                    generated_images_path=str(temp_fid_dir),
                    batch_size=32,
                    device=str(device)
                )
                
                fid_scores[epoch + 1] = fid_score
                writer.add_scalar('FID_Score', fid_score, epoch + 1)
                
                print(f"\nEpoch {epoch+1} FID Score: {fid_score:.2f}")
                
                if fid_score < best_fid:
                    best_fid = fid_score
                    best_epoch = epoch + 1
                    patience_counter = 0
                    print(f"✓ New best FID: {best_fid:.2f}")
                    # Save best model
                    torch.save({
                        'epoch': epoch + 1,
                        'fid_score': fid_score,
                        'generator_state_dict': generator.state_dict(),
                        'critic_state_dict': critic.state_dict(),
                        'optimizer_gen_state_dict': optimizer_gen.state_dict(),
                        'optimizer_critic_state_dict': optimizer_critic.state_dict(),
                        'scheduler_gen_state_dict': scheduler_gen.state_dict(),
                        'scheduler_critic_state_dict': scheduler_critic.state_dict(),
                    }, 'results/checkpoints/wgan_gp/best_fid_model.pth')
                else:
                    patience_counter += 1
                    print(f"⚠ FID did not improve ({patience_counter}/{HYPERPARAMETERS['early_stopping_patience']})")
                    print(f"  Best FID: {best_fid:.2f} at epoch {best_epoch}")
                    
                    # Early stopping check
                    if patience_counter >= HYPERPARAMETERS['early_stopping_patience']:
                        print(f"\n{'='*60}")
                        print(f"🛑 EARLY STOPPING at epoch {epoch+1}")
                        print(f"   FID has not improved for {HYPERPARAMETERS['early_stopping_patience']} consecutive checks")
                        print(f"   Best FID: {best_fid:.2f} at epoch {best_epoch}")
                        print(f"   Using best model from epoch {best_epoch}")
                        print(f"{'='*60}")
                        break
                
                print(f"{'='*60}\n")
            except Exception as e:
                print(f"Warning: FID calculation failed: {e}")
            
            # Cleanup temp directory
            for img in temp_fid_dir.glob('*.jpg'):
                img.unlink()
        
        # Check if early stopping triggered
        if patience_counter >= HYPERPARAMETERS['early_stopping_patience']:
            break
        
        # Step schedulers
        scheduler_gen.step()
        scheduler_critic.step()

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
