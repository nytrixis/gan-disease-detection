"""
Complete GAN-based Disease Detection Pipeline
Runs all steps in sequence
"""

import os
import sys
from pathlib import Path

print("="*70)
print(" GAN-BASED DISEASE DETECTION - COMPLETE PIPELINE")
print("="*70)

# Step 1: Create data splits
print("\n[STEP 1/5] Creating train/val/test splits...")
print("-"*70)
from preprocessing.data_loader import create_splits
create_splits()

# Step 2: Train WGAN-GP (this will take several hours)
print("\n[STEP 2/5] Training WGAN-GP...")
print("-"*70)
print("NOTE: This will take 6-12 hours on RTX 3050")
print("You can monitor progress in results/samples/")
response = input("\nProceed with GAN training? (y/n): ")
if response.lower() == 'y':
    from training.train_gan import train_wgan_gp
    train_wgan_gp()
else:
    print("Skipped GAN training. Using existing checkpoint if available.")

# Step 3: Generate synthetic images
print("\n[STEP 3/5] Generating synthetic images...")
print("-"*70)
from training.generate_synthetic import generate_synthetic_images
if Path('results/checkpoints/wgan_gp/final_model.pth').exists():
    generate_synthetic_images(
        checkpoint_path='results/checkpoints/wgan_gp/final_model.pth',
        num_images=1500
    )
else:
    print("ERROR: No trained GAN model found. Please train GAN first.")
    sys.exit(1)

# Step 4: Train classifiers
print("\n[STEP 4/5] Training CNN classifiers...")
print("-"*70)
print("Training 3 models: Baseline, Traditional Aug, GAN Aug")
print("This will take 3-6 hours total")

# Baseline
print("\nTraining BASELINE model...")
os.system('python training/train_classifier.py --mode baseline --architecture resnet50')

# Traditional augmentation
print("\nTraining TRADITIONAL AUGMENTATION model...")
os.system('python training/train_classifier.py --mode traditional_aug --architecture resnet50')

# GAN augmentation
print("\nTraining GAN AUGMENTATION model...")
os.system('python training/train_classifier.py --mode gan_aug --architecture resnet50')

# Step 5: Evaluation
print("\n[STEP 5/5] Evaluating models and generating reports...")
print("-"*70)
os.system('python evaluation/evaluate_all.py')

print("\n" + "="*70)
print(" PIPELINE COMPLETE!")
print("="*70)
print("\nResults saved to:")
print("  - results/checkpoints/ (trained models)")
print("  - results/figures/ (plots and visualizations)")
print("  - results/tables/ (metrics and statistics)")
print("="*70)
