"""
FID (Fréchet Inception Distance) Score Calculation for AC-GAN
Evaluates quality of generated images using InceptionV3 features
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import numpy as np
from scipy import linalg
from tqdm import tqdm
import json


class ImageFolderDataset(Dataset):
    """Simple dataset for loading images from a folder"""
    
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.image_paths = list(self.folder_path.glob('*.png')) + \
                          list(self.folder_path.glob('*.jpg')) + \
                          list(self.folder_path.glob('*.jpeg'))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


class InceptionV3FeatureExtractor(nn.Module):
    """
    InceptionV3 feature extractor for FID calculation
    Extracts features from the last average pooling layer
    """
    
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        
        # Load pretrained InceptionV3
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.eval()
        
        # Extract all layers except final FC and dropout
        # InceptionV3 structure: features -> avgpool -> dropout -> fc
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = inception.maxpool1
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = inception.maxpool2
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        self.avgpool = inception.avgpool
        
        # Set to evaluation mode
        self.eval()
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Extract features from images
        
        Args:
            x: (batch_size, 3, 299, 299) - images normalized for InceptionV3
            
        Returns:
            (batch_size, 2048) - feature vectors
        """
        with torch.no_grad():
            # N x 3 x 299 x 299
            x = self.Conv2d_1a_3x3(x)
            # N x 32 x 149 x 149
            x = self.Conv2d_2a_3x3(x)
            # N x 32 x 147 x 147
            x = self.Conv2d_2b_3x3(x)
            # N x 64 x 147 x 147
            x = self.maxpool1(x)
            # N x 64 x 73 x 73
            x = self.Conv2d_3b_1x1(x)
            # N x 80 x 73 x 73
            x = self.Conv2d_4a_3x3(x)
            # N x 192 x 71 x 71
            x = self.maxpool2(x)
            # N x 192 x 35 x 35
            x = self.Mixed_5b(x)
            # N x 256 x 35 x 35
            x = self.Mixed_5c(x)
            # N x 288 x 35 x 35
            x = self.Mixed_5d(x)
            # N x 288 x 35 x 35
            x = self.Mixed_6a(x)
            # N x 768 x 17 x 17
            x = self.Mixed_6b(x)
            # N x 768 x 17 x 17
            x = self.Mixed_6c(x)
            # N x 768 x 17 x 17
            x = self.Mixed_6d(x)
            # N x 768 x 17 x 17
            x = self.Mixed_6e(x)
            # N x 768 x 17 x 17
            x = self.Mixed_7a(x)
            # N x 1280 x 8 x 8
            x = self.Mixed_7b(x)
            # N x 2048 x 8 x 8
            x = self.Mixed_7c(x)
            # N x 2048 x 8 x 8
            x = self.avgpool(x)
            # N x 2048 x 1 x 1
            x = x.squeeze(-1).squeeze(-1)
            # N x 2048
        
        return x


def extract_features(data_loader, feature_extractor, device):
    """
    Extract InceptionV3 features from all images in dataloader
    
    Args:
        data_loader: DataLoader containing images
        feature_extractor: InceptionV3FeatureExtractor model
        device: torch device
        
    Returns:
        numpy array of shape (num_images, 2048)
    """
    feature_extractor.to(device)
    feature_extractor.eval()
    
    features_list = []
    
    with torch.no_grad():
        for images in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            features = feature_extractor(images)
            features_list.append(features.cpu().numpy())
    
    # Concatenate all features
    all_features = np.concatenate(features_list, axis=0)
    
    return all_features


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Fréchet distance between two multivariate Gaussians
    
    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    
    Args:
        mu1: Mean of first distribution
        sigma1: Covariance matrix of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance matrix of second distribution
        eps: Small constant for numerical stability
        
    Returns:
        float: Fréchet distance
    """
    # Ensure mu arrays are 1D
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    # Ensure covariance matrices are 2D
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"
    
    # Calculate difference between means
    diff = mu1 - mu2
    
    # Calculate product of covariance matrices
    # Using sqrtm which can handle numerical issues
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Check for imaginary components (numerical errors)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"Warning: Imaginary component {m} in covariance product")
        covmean = covmean.real
    
    # Calculate trace
    tr_covmean = np.trace(covmean)
    
    # Calculate FID
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return fid


def calculate_statistics(features):
    """
    Calculate mean and covariance of feature vectors
    
    Args:
        features: numpy array of shape (num_samples, feature_dim)
        
    Returns:
        tuple of (mean, covariance)
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma


def calculate_fid(
    real_images_path='data/raw/dermatofibroma',
    generated_images_path='data/synthetic_acgan/filtered',
    batch_size=32,
    num_workers=4,
    device='cuda'
):
    """
    Calculate FID score between real and generated images
    
    Args:
        real_images_path: Path to folder containing real images
        generated_images_path: Path to folder containing generated images
        batch_size: Batch size for feature extraction
        num_workers: Number of dataloader workers
        device: torch device
        
    Returns:
        float: FID score (lower is better)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # InceptionV3 expects images normalized with these statistics
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print(f"\nLoading real images from: {real_images_path}")
    real_dataset = ImageFolderDataset(real_images_path, transform=transform)
    print(f"Real images: {len(real_dataset)}")
    
    print(f"\nLoading generated images from: {generated_images_path}")
    gen_dataset = ImageFolderDataset(generated_images_path, transform=transform)
    print(f"Generated images: {len(gen_dataset)}")
    
    if len(real_dataset) == 0 or len(gen_dataset) == 0:
        raise ValueError("One or both image folders are empty!")
    
    # Create dataloaders
    real_loader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    gen_loader = DataLoader(
        gen_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create feature extractor
    print("\nLoading InceptionV3 feature extractor...")
    feature_extractor = InceptionV3FeatureExtractor()
    
    # Extract features from real images
    print("\nExtracting features from real images...")
    real_features = extract_features(real_loader, feature_extractor, device)
    print(f"Real features shape: {real_features.shape}")
    
    # Extract features from generated images
    print("\nExtracting features from generated images...")
    gen_features = extract_features(gen_loader, feature_extractor, device)
    print(f"Generated features shape: {gen_features.shape}")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    mu_real, sigma_real = calculate_statistics(real_features)
    mu_gen, sigma_gen = calculate_statistics(gen_features)
    
    # Calculate FID
    print("Calculating FID score...")
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return fid_score


def calculate_fid_progression(
    real_images_path='data/raw/dermatofibroma',
    checkpoints_dir='results/checkpoints/acgan',
    output_json='results/acgan_fid_progression.json',
    temp_gen_dir='data/synthetic_acgan/temp_fid',
    num_samples=1000,
    batch_size=32
):
    """
    Calculate FID score progression across multiple checkpoints
    Shows how quality improves during training
    
    Args:
        real_images_path: Path to real images
        checkpoints_dir: Directory containing checkpoints
        output_json: Path to save FID progression results
        temp_gen_dir: Temporary directory for generated images
        num_samples: Number of samples to generate per checkpoint
        batch_size: Batch size for generation and FID calculation
    """
    from training.generate_acgan_synthetic import generate_synthetic_images
    import yaml
    
    checkpoints_path = Path(checkpoints_dir)
    temp_gen_path = Path(temp_gen_dir)
    temp_gen_path.mkdir(parents=True, exist_ok=True)
    
    # Find all checkpoints
    checkpoint_files = sorted(checkpoints_path.glob('acgan_epoch_*.pth'))
    
    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoints_dir}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    fid_results = {}
    
    for checkpoint_file in checkpoint_files:
        epoch_num = int(checkpoint_file.stem.split('_')[-1])
        
        print(f"\n{'=' * 80}")
        print(f"Evaluating checkpoint: {checkpoint_file.name} (Epoch {epoch_num})")
        print(f"{'=' * 80}")
        
        # Clear temp directory
        for f in temp_gen_path.glob('*'):
            f.unlink()
        
        # Generate samples
        print(f"Generating {num_samples} samples...")
        generate_synthetic_images(
            checkpoint_path=str(checkpoint_file),
            config_path='configs/acgan_config.yaml',
            output_dir=str(temp_gen_path),
            num_images=num_samples,
            target_class=0,
            batch_size=batch_size,
            apply_filtering=False
        )
        
        # Calculate FID
        fid = calculate_fid(
            real_images_path=real_images_path,
            generated_images_path=str(temp_gen_path),
            batch_size=batch_size
        )
        
        fid_results[f'epoch_{epoch_num}'] = {
            'epoch': epoch_num,
            'fid_score': float(fid),
            'checkpoint': checkpoint_file.name
        }
        
        print(f"\nEpoch {epoch_num} FID Score: {fid:.2f}")
    
    # Save results
    with open(output_json, 'w') as f:
        json.dump(fid_results, f, indent=4)
    
    print(f"\n{'=' * 80}")
    print(f"FID Progression Results:")
    print(f"{'=' * 80}")
    for key, val in sorted(fid_results.items(), key=lambda x: x[1]['epoch']):
        print(f"  Epoch {val['epoch']:3d}: FID = {val['fid_score']:6.2f}")
    
    print(f"\nResults saved to: {output_json}")
    
    # Cleanup temp directory
    for f in temp_gen_path.glob('*'):
        f.unlink()
    temp_gen_path.rmdir()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate FID score for AC-GAN')
    parser.add_argument('--real-path', type=str, 
                        default='data/raw/dermatofibroma',
                        help='Path to real images')
    parser.add_argument('--gen-path', type=str, 
                        default='data/synthetic_acgan/filtered',
                        help='Path to generated images')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--progression', action='store_true',
                        help='Calculate FID progression across all checkpoints')
    parser.add_argument('--output-json', type=str, 
                        default='results/acgan_fid_progression.json',
                        help='Output JSON file for progression results')
    
    args = parser.parse_args()
    
    if args.progression:
        calculate_fid_progression(
            real_images_path=args.real_path,
            output_json=args.output_json,
            batch_size=args.batch_size
        )
    else:
        fid = calculate_fid(
            real_images_path=args.real_path,
            generated_images_path=args.gen_path,
            batch_size=args.batch_size,
            device=args.device
        )
        
        print(f"\n{'=' * 80}")
        print(f"FID Score: {fid:.2f}")
        print(f"{'=' * 80}")
        print(f"\nInterpretation:")
        print(f"  < 100: Good quality")
        print(f"  < 150: Acceptable quality")
        print(f"  < 200: Fair quality")
        print(f"  > 200: Poor quality")
