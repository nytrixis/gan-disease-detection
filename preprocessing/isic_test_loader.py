"""
ISIC 2019 Test Set Loader
Uses actual held-out test set (8238 images, 91 DF, 2495 NV, etc.)
NOT the train/val/test splits from training data
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

class ISICTestDataset(Dataset):
    """ISIC 2019 Test Dataset with ground truth labels"""

    def __init__(self, test_images_dir, ground_truth_csv, transform=None, binary_classification=True):
        """
        Args:
            test_images_dir: Path to ISIC_2019_Test_Input folder
            ground_truth_csv: Path to ISIC_2019_Test_GroundTruth.csv
            transform: Image transforms
            binary_classification: If True, only load DF (label=1) and NV (label=0) images
        """
        self.test_images_dir = Path(test_images_dir)
        self.transform = transform

        # Read ground truth
        df = pd.read_csv(ground_truth_csv)

        if binary_classification:
            # Filter for only DF and NV (binary classification)
            # DF=1, NV=0 to match training setup
            df_images = df[df['DF'] == 1.0][['image']].copy()
            df_images['label'] = 1  # Dermatofibroma

            nv_images = df[df['NV'] == 1.0][['image']].copy()
            nv_images['label'] = 0  # Nevus

            self.data = pd.concat([df_images, nv_images], ignore_index=True)

            print(f"Binary classification test set:")
            print(f"  Nevus (NV): {len(nv_images)} images (label=0)")
            print(f"  Dermatofibroma (DF): {len(df_images)} images (label=1)")
            print(f"  Total: {len(self.data)} images")
        else:
            # Multi-class (all 9 classes)
            self.data = df
            print(f"Multi-class test set: {len(self.data)} images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row['image']
        label = row['label']

        # Find image (nested folder structure)
        img_path = self.test_images_dir / 'ISIC_2019_Test_Input' / f'{img_name}.jpg'

        if not img_path.exists():
            # Try without nested folder
            img_path = self.test_images_dir / f'{img_name}.jpg'

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def get_isic_test_loader(
    test_images_dir='ISIC_2019_Test_Input',
    ground_truth_csv='ISIC_2019_Test_GroundTruth.csv',
    batch_size=16,
    num_workers=2,
    binary_classification=True
):
    """
    Get DataLoader for ISIC 2019 test set

    Returns:
        DataLoader with actual test images (not training splits!)
    """
    # ImageNet normalization (same as training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ISICTestDataset(
        test_images_dir=test_images_dir,
        ground_truth_csv=ground_truth_csv,
        transform=transform,
        binary_classification=binary_classification
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return loader

if __name__ == '__main__':
    # Test
    print("Testing ISIC 2019 Test Set Loader...\n")
    test_loader = get_isic_test_loader()
    print(f"\n✓ Test loader created successfully")
    print(f"  Total batches: {len(test_loader)}")
    print(f"  Total samples: {len(test_loader.dataset)}")

    # Check first batch
    images, labels = next(iter(test_loader))
    print(f"\n  First batch shape: {images.shape}")
    print(f"  First batch labels: {labels[:5].tolist()}")
    print(f"\n✓ Test passed!")
