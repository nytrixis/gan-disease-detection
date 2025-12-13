"""Generate synthetic Dermatofibroma images using trained WGAN-GP"""

import torch
from torchvision.utils import save_image
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from models.generator import Generator

def generate_synthetic_images(checkpoint_path, num_images=1500, output_dir='data/synthetic/raw'):
    """Generate synthetic images from trained generator"""

    print("="*60)
    print("Generating Synthetic Dermatofibroma Images")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load generator
    generator = Generator(latent_dim=100).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    print(f"✓ Loaded generator from {checkpoint_path}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate images
    print(f"\nGenerating {num_images} synthetic images...")

    with torch.no_grad():
        for i in tqdm(range(num_images)):
            noise = torch.randn(1, 100).to(device)
            fake_image = generator(noise)
            fake_image = (fake_image + 1) / 2  # Denormalize from [-1,1] to [0,1]
            save_image(fake_image, f'{output_dir}/generated_{i:04d}.png')

    print(f"\n✓ Generated {num_images} images")
    print(f"✓ Saved to {output_dir}/")
    print("="*60)

if __name__ == '__main__':
    generate_synthetic_images(
        checkpoint_path='results/checkpoints/wgan_gp/final_model.pth',
        num_images=1500,
        output_dir='data/synthetic/raw'
    )
