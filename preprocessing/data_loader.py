"""
Data loading and preprocessing for ISIC 2019 dataset
Optimized for RTX 3050 (4GB VRAM)
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

# Set random seed for reproducibility
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

class ISICDataset(Dataset):
    """ISIC Dataset for Dermatofibroma vs Nevus classification"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def create_splits(raw_data_path='data/raw', output_path='data/splits', seed=42):
    """
    Create stratified 90/10/0 train/val splits (no test split - using separate test set)
    - Dermatofibroma: 606 total -> 545 train, 61 val, 0 test
    - Nevus: 12875 total -> 11588 train, 1287 val, 0 test
    """
    print("Creating train/val splits (90/10, no test split)...")

    set_seed(seed)

    # Create output directories
    for split in ['train', 'val', 'test']:
        Path(f'{output_path}/{split}/dermatofibroma').mkdir(parents=True, exist_ok=True)
        Path(f'{output_path}/{split}/nevus').mkdir(parents=True, exist_ok=True)

    # Process each class
    metadata = []

    for class_name in ['dermatofibroma', 'nevus']:
        source_dir = f'{raw_data_path}/{class_name}'
        image_files = list(Path(source_dir).glob('*.jpg'))

        # Shuffle
        np.random.shuffle(image_files)

        # Calculate split sizes (90/10/0 - no test split)
        n_total = len(image_files)
        n_train = int(0.90 * n_total)
        n_val = n_total - n_train
        n_test = 0

        print(f"\n{class_name.capitalize()}:")
        print(f"  Total: {n_total}")
        print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

        # Split files (90/10, no test)
        train_files = image_files[:n_train]
        val_files = image_files[n_train:]
        test_files = []

        # Label encoding: dermatofibroma=0, nevus=1
        label = 0 if class_name == 'dermatofibroma' else 1

        # Copy files to split directories
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            for img_file in tqdm(files, desc=f"Copying {class_name} to {split}"):
                dst = f'{output_path}/{split}/{class_name}/{img_file.name}'
                shutil.copy2(img_file, dst)
                metadata.append({
                    'image_path': dst,
                    'class': class_name,
                    'label': label,
                    'split': split
                })

    # Save metadata
    df = pd.DataFrame(metadata)
    df.to_csv(f'{output_path}/metadata.csv', index=False)
    print(f"\n✓ Metadata saved to {output_path}/metadata.csv")
    print(f"✓ Total images: {len(df)}")

    return df

def get_transforms(size=256):
    """
    Get standard transforms for GAN training (normalize to [-1, 1])
    Compatible with both WGAN-GP and StyleGAN2
    
    Args:
        size: Image size (default 256)
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # to [-1, 1]
    ])

def get_gan_transforms():
    """Transforms for GAN training (normalize to [-1, 1])"""
    return get_transforms(size=256)

def get_classifier_transforms(augment=False):
    """Transforms for classifier training (ImageNet normalization)"""
    if augment:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(degrees=30),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_dataloader(split='train', class_filter=None, batch_size=16, shuffle=True,
                  for_gan=False, augment=False, num_workers=2):
    """
    Get DataLoader for specified split

    Args:
        split: 'train', 'val', or 'test'
        class_filter: 'dermatofibroma', 'nevus', 'actinic_keratosis', or None for all
        batch_size: batch size (reduced for RTX 3050)
        shuffle: whether to shuffle
        for_gan: use GAN transforms if True, else classifier transforms
        augment: use augmentation for classifier
        num_workers: number of workers (reduced for laptop)
    """
    # Try metadata first (for old datasets - dermatofibroma and nevus only)
    metadata_path = 'data/splits/metadata.csv'

    # Check if we should use metadata CSV
    use_metadata = False
    if os.path.exists(metadata_path):
        # Only use metadata if class_filter is DF, nevus, or None
        if class_filter in ['dermatofibroma', 'nevus', None]:
            df = pd.read_csv(metadata_path)
            # Check if the requested class exists in metadata
            if class_filter is None or class_filter in df['class'].unique():
                use_metadata = True

    if use_metadata:
        # Use metadata CSV (old method)
        df = pd.read_csv(metadata_path)
        df = df[df['split'] == split]

        if class_filter:
            df = df[df['class'] == class_filter]

        image_paths = df['image_path'].tolist()
        labels = df['label'].tolist()
    else:
        # Read directly from splits directory (for new datasets like actinic_keratosis)
        split_dir = Path(f'data/splits/{split}')

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        image_paths = []
        labels = []

        if class_filter:
            # Single class
            class_dir = split_dir / class_filter
            if not class_dir.exists():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")

            for img_path in class_dir.glob('*.jpg'):
                image_paths.append(str(img_path))
                labels.append(0)  # Single class, label = 0
        else:
            # All classes
            for class_idx, class_name in enumerate(['dermatofibroma', 'nevus', 'actinic_keratosis']):
                class_dir = split_dir / class_name
                if class_dir.exists():
                    for img_path in class_dir.glob('*.jpg'):
                        image_paths.append(str(img_path))
                        labels.append(class_idx)

    # Choose transforms
    if for_gan:
        transform = get_gan_transforms()
    else:
        transform = get_classifier_transforms(augment=augment)

    dataset = ISICDataset(image_paths, labels, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

def get_gan_augmented_dataloader(split='train', synthetic_dir='data/synthetic/curated',
                                 batch_size=16, shuffle=True, num_workers=2):
    """
    Get DataLoader with GAN-augmented dataset (real + synthetic images)
    
    Args:
        split: 'train', 'val', or 'test'
        synthetic_dir: Directory containing curated synthetic images
        batch_size: batch size
        shuffle: whether to shuffle
        num_workers: number of workers
    
    Returns:
        DataLoader with mixed real and synthetic images
    """
    metadata_path = 'data/splits/metadata.csv'
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Metadata not found. Run create_splits() first.")
    
    # Load real images
    df = pd.read_csv(metadata_path)
    df = df[df['split'] == split]
    
    image_paths = df['image_path'].tolist()
    labels = df['label'].tolist()
    
    # Add synthetic dermatofibroma images (only for training split)
    if split == 'train' and os.path.exists(synthetic_dir):
        synthetic_images = list(Path(synthetic_dir).glob('*.png'))
        print(f"Found {len(synthetic_images)} synthetic images in {synthetic_dir}")
        
        for syn_img in synthetic_images:
            image_paths.append(str(syn_img))
            labels.append(1)  # Dermatofibroma label
        
        print(f"✓ Added {len(synthetic_images)} synthetic dermatofibroma images")
        
        # Calculate new class distribution
        df_count = sum(1 for l in labels if l == 1)
        nv_count = sum(1 for l in labels if l == 0)
        imbalance_ratio = nv_count / df_count if df_count > 0 else 0
        print(f"✓ New class distribution: Nevus={nv_count}, Dermatofibroma={df_count}")
        print(f"✓ New imbalance ratio: {imbalance_ratio:.1f}:1 (from 56.7:1)\n")
    
    # Use classifier transforms (ImageNet normalization)
    transform = get_classifier_transforms(augment=False)
    
    dataset = ISICDataset(image_paths, labels, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

if __name__ == '__main__':
    # Create splits
    print("="*60)
    print("ISIC 2019 Dataset Preprocessing")
    print("="*60)

    metadata = create_splits()

    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(metadata.groupby(['split', 'class']).size().unstack(fill_value=0))

    print("\n✓ Preprocessing complete!")
