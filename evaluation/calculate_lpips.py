"""
LPIPS (Learned Perceptual Image Patch Similarity) Calculation
Measures perceptual similarity between real and generated images
Lower LPIPS = more similar (better quality)
"""

import torch
import lpips
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import random


class ImagePairDataset(Dataset):
    """Dataset for loading pairs of real and generated images"""
    
    def __init__(self, real_dir, gen_dir, transform=None, max_pairs=None):
        """
        Args:
            real_dir: Directory containing real images
            gen_dir: Directory containing generated images
            transform: Image transformations
            max_pairs: Maximum number of pairs to load (None = all)
        """
        self.real_dir = Path(real_dir)
        self.gen_dir = Path(gen_dir)
        self.transform = transform
        
        # Get all image paths
        self.real_images = sorted(list(self.real_dir.glob('*.png')) + 
                                  list(self.real_dir.glob('*.jpg')) + 
                                  list(self.real_dir.glob('*.jpeg')))
        self.gen_images = sorted(list(self.gen_dir.glob('*.png')) + 
                                list(self.gen_dir.glob('*.jpg')) + 
                                list(self.gen_dir.glob('*.jpeg')))
        
        # Limit pairs if specified
        if max_pairs:
            self.real_images = random.sample(self.real_images, min(max_pairs, len(self.real_images)))
            self.gen_images = random.sample(self.gen_images, min(max_pairs, len(self.gen_images)))
        
        # Use minimum length for pairing
        self.num_pairs = min(len(self.real_images), len(self.gen_images))
        
    def __len__(self):
        return self.num_pairs
    
    def __getitem__(self, idx):
        # Load images
        real_img = Image.open(self.real_images[idx]).convert('RGB')
        gen_img = Image.open(self.gen_images[idx]).convert('RGB')
        
        if self.transform:
            real_img = self.transform(real_img)
            gen_img = self.transform(gen_img)
        
        return real_img, gen_img


def calculate_lpips(
    real_images_path,
    generated_images_path,
    batch_size=32,
    device='cuda',
    net='alex',
    max_pairs=1000
):
    """
    Calculate LPIPS score between real and generated images
    
    Args:
        real_images_path: Path to directory with real images
        generated_images_path: Path to directory with generated images
        batch_size: Batch size for processing
        device: Device (cuda or cpu)
        net: Network architecture ('alex', 'vgg', or 'squeeze')
        max_pairs: Maximum number of image pairs to compare
    
    Returns:
        float: Average LPIPS score (lower is better)
    """
    print(f"\n{'=' * 80}")
    print(f"CALCULATING LPIPS SCORE")
    print(f"{'=' * 80}")
    print(f"Real images: {real_images_path}")
    print(f"Generated images: {generated_images_path}")
    print(f"Network: {net}")
    print(f"Device: {device}")
    print(f"Max pairs: {max_pairs}\n")
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Initialize LPIPS model
    # net can be 'alex', 'vgg', or 'squeeze'
    lpips_model = lpips.LPIPS(net=net).to(device)
    lpips_model.eval()
    
    # Image transformations (LPIPS expects [-1, 1] normalized images)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset and dataloader
    dataset = ImagePairDataset(
        real_dir=real_images_path,
        gen_dir=generated_images_path,
        transform=transform,
        max_pairs=max_pairs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Number of image pairs: {len(dataset)}\n")
    
    # Calculate LPIPS scores
    lpips_scores = []
    
    with torch.no_grad():
        for real_imgs, gen_imgs in tqdm(dataloader, desc="Calculating LPIPS"):
            real_imgs = real_imgs.to(device)
            gen_imgs = gen_imgs.to(device)
            
            # Calculate LPIPS for batch
            distances = lpips_model(real_imgs, gen_imgs)
            lpips_scores.extend(distances.cpu().numpy().flatten())
    
    # Calculate statistics
    lpips_scores = np.array(lpips_scores)
    mean_lpips = lpips_scores.mean()
    std_lpips = lpips_scores.std()
    median_lpips = np.median(lpips_scores)
    min_lpips = lpips_scores.min()
    max_lpips = lpips_scores.max()
    
    print(f"\n{'=' * 80}")
    print(f"LPIPS RESULTS")
    print(f"{'=' * 80}")
    print(f"Mean LPIPS:   {mean_lpips:.4f} ± {std_lpips:.4f}")
    print(f"Median LPIPS: {median_lpips:.4f}")
    print(f"Min LPIPS:    {min_lpips:.4f}")
    print(f"Max LPIPS:    {max_lpips:.4f}")
    print(f"{'=' * 80}")
    
    print(f"\nInterpretation:")
    print(f"  < 0.10: Excellent perceptual similarity")
    print(f"  < 0.20: Good perceptual similarity")
    print(f"  < 0.30: Acceptable perceptual similarity")
    print(f"  < 0.40: Fair perceptual similarity")
    print(f"  > 0.40: Poor perceptual similarity")
    
    return {
        'mean': float(mean_lpips),
        'std': float(std_lpips),
        'median': float(median_lpips),
        'min': float(min_lpips),
        'max': float(max_lpips),
        'num_pairs': len(dataset)
    }


def calculate_lpips_progression(
    real_images_path='data/raw/dermatofibroma',
    checkpoints_dir='results/checkpoints/acgan',
    generator_checkpoint_pattern='acgan_epoch_*.pth',
    output_json='results/tables/acgan_lpips_progression.json',
    batch_size=32,
    device='cuda',
    net='alex',
    max_pairs=500
):
    """
    Calculate LPIPS progression across multiple checkpoints
    
    Args:
        real_images_path: Path to real images directory
        checkpoints_dir: Directory containing model checkpoints
        generator_checkpoint_pattern: Pattern for checkpoint files
        output_json: Output JSON file path
        batch_size: Batch size for processing
        device: Device (cuda or cpu)
        net: LPIPS network ('alex', 'vgg', or 'squeeze')
        max_pairs: Maximum pairs per checkpoint
    """
    import tempfile
    import shutil
    from training.generate_acgan_synthetic import generate_synthetic_images
    
    print(f"\n{'=' * 80}")
    print(f"CALCULATING LPIPS PROGRESSION")
    print(f"{'=' * 80}\n")
    
    checkpoints_path = Path(checkpoints_dir)
    checkpoint_files = sorted(checkpoints_path.glob(generator_checkpoint_pattern))
    
    if not checkpoint_files:
        print(f"⚠️  No checkpoints found matching: {generator_checkpoint_pattern}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints\n")
    
    results = []
    
    # Create temporary directory for generated images
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        for checkpoint_path in checkpoint_files:
            # Extract epoch number from filename
            epoch_str = checkpoint_path.stem.split('_')[-1]
            
            print(f"\n{'=' * 80}")
            print(f"Processing: {checkpoint_path.name} (Epoch {epoch_str})")
            print(f"{'=' * 80}\n")
            
            # Generate temporary synthetic images
            temp_gen_dir = temp_dir / f'epoch_{epoch_str}'
            temp_gen_dir.mkdir(exist_ok=True)
            
            print(f"Generating {max_pairs} images for LPIPS calculation...")
            generate_synthetic_images(
                checkpoint_path=str(checkpoint_path),
                output_dir=str(temp_gen_dir),
                num_images=max_pairs,
                target_class=0,
                batch_size=batch_size,
                apply_filtering=False
            )
            
            # Calculate LPIPS
            lpips_result = calculate_lpips(
                real_images_path=real_images_path,
                generated_images_path=str(temp_gen_dir),
                batch_size=batch_size,
                device=device,
                net=net,
                max_pairs=max_pairs
            )
            
            results.append({
                'epoch': int(epoch_str),
                'checkpoint': checkpoint_path.name,
                'lpips_mean': lpips_result['mean'],
                'lpips_std': lpips_result['std'],
                'lpips_median': lpips_result['median']
            })
            
            # Clean up temporary images
            shutil.rmtree(temp_gen_dir)
    
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)
    
    # Save results
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{'=' * 80}")
    print(f"LPIPS PROGRESSION SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Epoch':<10} {'LPIPS Mean':<15} {'LPIPS Std':<15}")
    print(f"{'-' * 80}")
    
    for result in results:
        print(f"{result['epoch']:<10} {result['lpips_mean']:<15.4f} {result['lpips_std']:<15.4f}")
    
    print(f"{'=' * 80}")
    print(f"✓ Results saved to: {output_path}\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate LPIPS score for AC-GAN')
    parser.add_argument('--real-path', type=str, 
                        default='data/raw/dermatofibroma',
                        help='Path to real images directory')
    parser.add_argument('--gen-path', type=str, 
                        default='data/synthetic_acgan/filtered',
                        help='Path to generated images directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--net', type=str, default='alex',
                        choices=['alex', 'vgg', 'squeeze'],
                        help='Network for LPIPS (alex=AlexNet, vgg=VGG, squeeze=SqueezeNet)')
    parser.add_argument('--max-pairs', type=int, default=1000,
                        help='Maximum number of image pairs to compare')
    parser.add_argument('--progression', action='store_true',
                        help='Calculate LPIPS progression across checkpoints')
    parser.add_argument('--output-json', type=str, 
                        default='results/tables/acgan_lpips.json',
                        help='Output JSON file path')
    
    args = parser.parse_args()
    
    if args.progression:
        calculate_lpips_progression(
            real_images_path=args.real_path,
            batch_size=args.batch_size,
            device=args.device,
            net=args.net,
            max_pairs=args.max_pairs,
            output_json=args.output_json
        )
    else:
        lpips_result = calculate_lpips(
            real_images_path=args.real_path,
            generated_images_path=args.gen_path,
            batch_size=args.batch_size,
            device=args.device,
            net=args.net,
            max_pairs=args.max_pairs
        )
        
        # Save single result
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(lpips_result, f, indent=4)
        
        print(f"\n✓ Results saved to: {output_path}")
