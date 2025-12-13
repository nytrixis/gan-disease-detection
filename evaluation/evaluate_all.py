"""
Comprehensive evaluation of all trained models
"""

import torch
import torch.nn as nn
from torchvision import models
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))
from preprocessing.data_loader import get_dataloader

def load_model(checkpoint_path, architecture='resnet50'):
    """Load trained model from checkpoint"""
    if architecture == 'resnet50':
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif architecture == 'densenet121':
        model = models.densenet121()
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif architecture == 'efficientnet_b0':
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'recall': recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'f1_score': f1_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }

    return metrics, all_preds, all_labels

def main():
    print("="*70)
    print(" MODEL EVALUATION")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Test DataLoader
    test_loader = get_dataloader('test', batch_size=16, shuffle=False)
    print(f"Test samples: {len(test_loader.dataset)}\n")

    # Models to evaluate
    models_to_eval = [
        ('baseline_resnet50', 'resnet50'),
        ('traditional_aug_resnet50', 'resnet50'),
        ('gan_aug_resnet50', 'resnet50'),
    ]

    results = {}

    for model_name, architecture in models_to_eval:
        checkpoint_path = f'results/checkpoints/classifier/{model_name}_best.pth'

        if not Path(checkpoint_path).exists():
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print(f"   Skipping {model_name}\n")
            continue

        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*70}")

        # Load and evaluate
        model = load_model(checkpoint_path, architecture).to(device)
        metrics, preds, labels = evaluate_model(model, test_loader, device)

        # Print results
        print(f"\nResults:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"  [[TN={cm[0][0]}, FP={cm[0][1]}],")
        print(f"   [FN={cm[1][0]}, TP={cm[1][1]}]]")

        results[model_name] = metrics

    # Save results
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    with open('results/tables/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Print comparison table
    print("\n" + "="*70)
    print(" COMPARISON TABLE")
    print("="*70)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)

    for model_name in results:
        m = results[model_name]
        print(f"{model_name:<25} {m['accuracy']:<12.4f} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1_score']:<12.4f}")

    print("="*70)
    print(f"\n✓ Results saved to results/tables/evaluation_results.json")

if __name__ == '__main__':
    main()
