"""
Organize ISIC 2019 images by disease category
Keep only Dermatofibroma and Nevus, delete the rest
"""

import pandas as pd
import shutil
import os
from pathlib import Path
from tqdm import tqdm

print("Starting ISIC 2019 dataset organization...")

# Paths
csv_path = "archive (2)/ISIC_2019_Training_GroundTruth.csv"
images_dir = "archive (2)/ISIC_2019_Training_Input/ISIC_2019_Training_Input/"

# Create output directories
Path("data/raw/dermatofibroma").mkdir(parents=True, exist_ok=True)
Path("data/raw/nevus").mkdir(parents=True, exist_ok=True)

# Load ground truth CSV
print(f"\nReading CSV file: {csv_path}")
df = pd.read_csv(csv_path)

print(f"Total images in dataset: {len(df)}")
print("\nDisease distribution:")
print(df.iloc[:, 1:].sum().sort_values(ascending=False))

# Find which column names correspond to our diseases
# The CSV has one-hot encoded columns for each disease
print("\nColumn names in CSV:")
print(df.columns.tolist())

# Count images for each disease we need
dermatofibroma_count = 0
nevus_count = 0
other_count = 0

# Identify dermatofibroma and nevus images
dermatofibroma_images = []
nevus_images = []

for idx, row in df.iterrows():
    image_name = row['image']

    # Check which disease columns exist (different datasets use different names)
    # Common variations: 'DF', 'dermatofibroma', 'NV', 'nevus', 'MEL', 'melanoma', etc.

    # Get all disease columns (skip first column which is image name)
    disease_cols = df.columns[1:]

    # Find which disease this image has (one-hot encoded, so value = 1.0)
    diseases = row[disease_cols]
    disease_name = diseases[diseases == 1.0].index

    if len(disease_name) > 0:
        disease = disease_name[0].lower()

        # Check if it's dermatofibroma
        if 'df' in disease or 'dermatofibroma' in disease:
            dermatofibroma_images.append(image_name)
            dermatofibroma_count += 1

        # Check if it's nevus
        elif 'nv' in disease or 'nevus' in disease:
            nevus_images.append(image_name)
            nevus_count += 1

        else:
            other_count += 1

print(f"\nFound:")
print(f"  Dermatofibroma (DF): {dermatofibroma_count} images")
print(f"  Nevus (NV): {nevus_count} images")
print(f"  Other diseases: {other_count} images (will be ignored)")

# Copy dermatofibroma images
print(f"\nCopying {dermatofibroma_count} Dermatofibroma images...")
for img_name in tqdm(dermatofibroma_images):
    # Try different file extensions
    for ext in ['.jpg', '.JPG', '_downsampled.jpg']:
        src = os.path.join(images_dir, img_name + ext)
        if os.path.exists(src):
            dst = f"data/raw/dermatofibroma/{img_name}.jpg"
            shutil.copy2(src, dst)
            break

# Copy nevus images
print(f"\nCopying {nevus_count} Nevus images...")
for img_name in tqdm(nevus_images):
    # Try different file extensions
    for ext in ['.jpg', '.JPG', '_downsampled.jpg']:
        src = os.path.join(images_dir, img_name + ext)
        if os.path.exists(src):
            dst = f"data/raw/nevus/{img_name}.jpg"
            shutil.copy2(src, dst)
            break

# Verify counts
df_copied = len(os.listdir("data/raw/dermatofibroma"))
nv_copied = len(os.listdir("data/raw/nevus"))

print("\n" + "="*60)
print("ORGANIZATION COMPLETE!")
print("="*60)
print(f"Dermatofibroma images: {df_copied}")
print(f"Nevus images: {nv_copied}")
print(f"\nYou can now DELETE the original extracted folder to save space:")
print(f"  rm -rf \"archive (2)\"/")
print(f"\nThis will free up ~7GB of disk space!")
print("="*60)
