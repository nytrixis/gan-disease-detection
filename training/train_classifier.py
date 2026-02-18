"""
CNN Classifier Training Script
Supports: Baseline, Traditional Aug, GAN Aug
Optimized for RTX 3050
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
import json

sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.data_loader import get_dataloader, set_seed

def get_model(architecture='resnet50'):
    """Load pretrained model"""
    if architecture == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif architecture == 'densenet121':
        model = models.densenet121(weights='IMAGENET1K_V1')
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif architecture == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model

def train_classifier(mode='baseline', architecture='resnet50', epochs=50, synthetic_data_dir=None):
    """
    Train CNN classifier

    Args:
        mode: 'baseline', 'traditional_aug', 'gan_aug', or 'acgan_aug'
        architecture: 'resnet50', 'densenet121', or 'efficientnet_b0'
        epochs: number of training epochs
        synthetic_data_dir: Custom directory for synthetic images (for gan_aug mode)
    """

    print("="*60)
    print(f"Training {architecture.upper()} - {mode.upper()} mode")
    print("="*60)

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Model
    model = get_model(architecture).to(device)

    # Loss with class weights (imbalance handling)
    # Updated for 545 DF samples: 12875/545 = 23.6
    class_weights = torch.tensor([1.0, 23.6]).to(device)  # [nevus, dermatofibroma]
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # DataLoaders
    augment = (mode == 'traditional_aug')
    
    if mode == 'gan_aug':
        # Load dataset with GAN-augmented images (WGAN-GP or WGAN-GP Revised)
        from preprocessing.data_loader import get_gan_augmented_dataloader
        synthetic_dir = synthetic_data_dir or 'data/synthetic/curated'
        train_loader = get_gan_augmented_dataloader(
            'train', 
            synthetic_dir=synthetic_dir,
            batch_size=16, 
            shuffle=True
        )
        print(f"✓ Using GAN-augmented dataset")
        print(f"  Synthetic images from: {synthetic_dir}")
    elif mode == 'acgan_aug':
        # Load dataset with AC-GAN augmented images
        from preprocessing.data_loader import get_gan_augmented_dataloader
        train_loader = get_gan_augmented_dataloader(
            'train', 
            synthetic_dir='data/synthetic_acgan/filtered',
            batch_size=16, 
            shuffle=True
        )
        print(f"✓ Using AC-GAN augmented dataset (class-conditional synthetic images)")
    else:
        train_loader = get_dataloader('train', batch_size=16, shuffle=True, augment=augment)
    
    val_loader = get_dataloader('val', batch_size=16, shuffle=False, augment=False)

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}\n")

    # Training loop
    best_f1 = 0
    patience_counter = 0
    max_patience = 15

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_tp = val_fp = val_fn = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)

                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # Calculate TP, FP, FN for F1 (class 1 = dermatofibroma)
                for pred, true in zip(predicted, labels):
                    if pred == 1 and true == 1:
                        val_tp += 1
                    elif pred == 1 and true == 0:
                        val_fp += 1
                    elif pred == 0 and true == 1:
                        val_fn += 1

        val_acc = 100. * val_correct / val_total
        precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\nEpoch {epoch+1}: Val Acc={val_acc:.2f}%, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            Path('results/checkpoints/classifier').mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': f1,
            }, f'results/checkpoints/classifier/{mode}_{architecture}_best.pth')
            print(f"✓ Saved new best model (F1={f1:.3f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(f"\nEarly stopping triggered (patience={max_patience})")
            break

    print("\n" + "="*60)
    print(f"✓ Training complete! Best F1: {best_f1:.3f}")
    print(f"✓ Model saved to results/checkpoints/classifier/{mode}_{architecture}_best.pth")
    print("="*60)

    return best_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                       choices=['baseline', 'traditional_aug', 'gan_aug'])
    parser.add_argument('--architecture', type=str, default='resnet50',
                       choices=['resnet50', 'densenet121', 'efficientnet_b0'])
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()

    train_classifier(mode=args.mode, architecture=args.architecture, epochs=args.epochs)
