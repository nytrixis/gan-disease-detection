#!/bin/bash
# Dataset Download Script

echo "Downloading ISIC 2020 Dataset..."

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "Installing Kaggle CLI..."
    pip install kaggle
fi

# Check if kaggle credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: Kaggle credentials not found."
    echo "Please download kaggle.json from https://www.kaggle.com/settings"
    echo "and place it in ~/.kaggle/"
    exit 1
fi

# Set permissions
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
echo "Downloading dataset (this may take a while)..."
kaggle datasets download -d sumaiyabinteshahid/isic-challenge-dataset-2020

# Unzip dataset
echo "Extracting dataset..."
unzip -q isic-challenge-dataset-2020.zip -d data/raw/

# Clean up zip file
rm isic-challenge-dataset-2020.zip

echo "✓ Dataset download complete!"
echo "Dataset location: data/raw/"
