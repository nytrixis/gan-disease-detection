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
    Create stratified 70/15/15 train/val/test splits
    - Dermatofibroma: 239 total -> 167 train, 36 val, 36 test
    - Nevus: 12875 total -> 9012 train, 1932 val, 1931 test
    """
    print("Creating train/val/test splits...")

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

        # Calculate split sizes (70/15/15)
        n_total = len(image_files)
        n_train = int(0.70 * n_total)
        n_val = int(0.15 * n_total)
        n_test = n_total - n_train - n_val

        print(f"\n{class_name.capitalize()}:")
        print(f"  Total: {n_total}")
        print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

        # Split files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train+n_val]
        test_files = image_files[n_train+n_val:]

        # Label encoding: dermatofibroma=1, nevus=0
        label = 1 if class_name == 'dermatofibroma' else 0

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

def get_gan_transforms():
    """Transforms for GAN training (normalize to [-1, 1])"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # to [-1, 1]
    ])

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
        class_filter: 'dermatofibroma', 'nevus', or None for both
        batch_size: batch size (reduced for RTX 3050)
        shuffle: whether to shuffle
        for_gan: use GAN transforms if True, else classifier transforms
        augment: use augmentation for classifier
        num_workers: number of workers (reduced for laptop)
    """
    metadata_path = 'data/splits/metadata.csv'

    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Metadata not found. Run create_splits() first.")

    df = pd.read_csv(metadata_path)
    df = df[df['split'] == split]

    if class_filter:
        df = df[df['class'] == class_filter]

    image_paths = df['image_path'].tolist()
    labels = df['label'].tolist()

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
