"""
Visualization utilities for GAN training and evaluation
Comprehensive plotting functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def plot_gan_training_curves(losses_critic, losses_generator, wasserstein_distances, 
                             save_path='results/figures/gan_training_curves.png'):
    """
    Plot GAN training curves: Critic Loss, Generator Loss, Wasserstein Distance
    
    Args:
        losses_critic: List of critic losses
        losses_generator: List of generator losses
        wasserstein_distances: List of Wasserstein distances
        save_path: Where to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Critic Loss
    axes[0].plot(losses_critic, color='#2E86C1', linewidth=1.5, alpha=0.8)
    axes[0].set_title('Critic Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Generator Loss
    axes[1].plot(losses_generator, color='#E74C3C', linewidth=1.5, alpha=0.8)
    axes[1].set_title('Generator Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    
    # Wasserstein Distance
    axes[2].plot(wasserstein_distances, color='#27AE60', linewidth=1.5, alpha=0.8)
    axes[2].set_title('Wasserstein Distance', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Distance')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ GAN training curves saved to {save_path}")

def plot_generated_samples(generator, latent_dim=100, num_samples=25, 
                           save_path='results/figures/generated_samples_grid.png',
                           device='cuda'):
    """
    Generate and plot grid of synthetic images
    
    Args:
        generator: Trained generator model
        latent_dim: Latent dimension size
        num_samples: Number of samples (should be perfect square)
        save_path: Where to save the figure
        device: 'cuda' or 'cpu'
    """
    generator.eval()
    
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i in range(num_samples):
            noise = torch.randn(1, latent_dim).to(device)
            fake = generator(noise).detach().cpu()
            fake = (fake + 1) / 2  # Denormalize to [0, 1]
            
            img = fake.squeeze().permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].axis('off')
    
    plt.suptitle('Generated Synthetic Images', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated samples grid saved to {save_path}")

def plot_real_vs_synthetic_comparison(real_images_dir, synthetic_images_dir, 
                                     n_pairs=5,
                                     save_path='results/figures/real_vs_synthetic.png'):
    """
    Side-by-side comparison of real and synthetic images
    
    Args:
        real_images_dir: Directory with real images
        synthetic_images_dir: Directory with synthetic images
        n_pairs: Number of image pairs to show
        save_path: Where to save the figure
    """
    real_paths = list(Path(real_images_dir).glob('*.jpg'))[:n_pairs]
    synthetic_paths = list(Path(synthetic_images_dir).glob('*.png'))[:n_pairs]
    
    fig, axes = plt.subplots(2, n_pairs, figsize=(3*n_pairs, 6))
    
    for i in range(n_pairs):
        # Real image
        real_img = Image.open(real_paths[i]).convert('RGB')
        axes[0, i].imshow(real_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Real Images', fontsize=12, fontweight='bold')
        
        # Synthetic image
        syn_img = Image.open(synthetic_paths[i]).convert('RGB')
        axes[1, i].imshow(syn_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Synthetic Images', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Real vs synthetic comparison saved to {save_path}")

def plot_classifier_training_curves(train_losses, val_losses, train_accs, val_accs,
                                    save_path='results/figures/classifier_training_curves.png'):
    """
    Plot classifier training curves: Loss and Accuracy
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Where to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curves
    axes[0].plot(epochs, train_losses, label='Training', color='#2E86C1', linewidth=2, marker='o', markersize=4)
    axes[0].plot(epochs, val_losses, label='Validation', color='#E74C3C', linewidth=2, marker='s', markersize=4)
    axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, train_accs, label='Training', color='#2E86C1', linewidth=2, marker='o', markersize=4)
    axes[1].plot(epochs, val_accs, label='Validation', color='#E74C3C', linewidth=2, marker='s', markersize=4)
    axes[1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Classifier training curves saved to {save_path}")

def plot_confusion_matrix(cm, class_names=['Nevus', 'Dermatofibroma'],
                          save_path='results/figures/confusion_matrix.png',
                          title='Confusion Matrix'):
    """
    Plot confusion matrix heatmap
    
    Args:
        cm: Confusion matrix (2x2 numpy array or list)
        class_names: Names of classes
        save_path: Where to save the figure
        title: Plot title
    """
    cm = np.array(cm)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=ax, square=True,
                annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add accuracy in title
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    ax.text(0.5, 1.08, f'Accuracy: {accuracy:.2%}', 
            transform=ax.transAxes, ha='center', fontsize=11)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")

def plot_precision_recall_curve(precision, recall, thresholds=None,
                                save_path='results/figures/precision_recall_curve.png'):
    """
    Plot Precision-Recall curve
    
    Args:
        precision: Array of precision values
        recall: Array of recall values
        thresholds: Array of threshold values (optional)
        save_path: Where to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(recall, precision, linewidth=2.5, color='#8E44AD')
    ax.fill_between(recall, precision, alpha=0.2, color='#8E44AD')
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Precision-Recall curve saved to {save_path}")

def plot_roc_curve(fpr, tpr, roc_auc,
                   save_path='results/figures/roc_curve.png'):
    """
    Plot ROC curve
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        roc_auc: Area under ROC curve
        save_path: Where to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(fpr, tpr, linewidth=2.5, color='#E67E22', 
            label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.2, color='#E67E22')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curve saved to {save_path}")

def create_ablation_results_table(results_dict, 
                                  save_path='results/figures/ablation_study.png'):
    """
    Create visual table for ablation study results
    
    Args:
        results_dict: Dictionary of {experiment_name: metrics}
        save_path: Where to save the figure
    """
    # Prepare data
    experiments = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    data = []
    for exp in experiments:
        row = [results_dict[exp].get(m, 0) for m in metrics]
        data.append(row)
    
    data = np.array(data)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(experiments))
    width = 0.2
    
    colors = ['#3498DB', '#E74C3C', '#F39C12', '#27AE60']
    
    for i, metric in enumerate(metrics):
        offset = width * (i - 1.5)
        ax.bar(x + offset, data[:, i], width, label=metric.replace('_', ' ').title(),
               color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Ablation study table saved to {save_path}")

def plot_metrics_comparison(results_dict, 
                           save_path='results/figures/metrics_comparison.png'):
    """
    Create comprehensive comparison plot of all models
    
    Args:
        results_dict: Dictionary of {model_name: {metrics}}
        save_path: Where to save the figure
    """
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = sns.color_palette("husl", len(models))
    
    for idx, metric in enumerate(metrics):
        values = [results_dict[m].get(metric, 0) for m in models]
        
        bars = axes[idx].bar(models, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Score', fontsize=10)
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Rotate x-labels if needed
        if len(models) > 3:
            axes[idx].set_xticklabels(models, rotation=15, ha='right')
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics comparison saved to {save_path}")

def plot_class_distribution(train_counts, test_counts, 
                           class_names=['Nevus', 'Dermatofibroma'],
                           save_path='results/figures/class_distribution.png'):
    """
    Plot class distribution in train and test sets
    
    Args:
        train_counts: List of counts for each class in training
        test_counts: List of counts for each class in test
        class_names: Names of classes
        save_path: Where to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#3498DB', '#E74C3C']
    
    # Training set
    axes[0].bar(class_names, train_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_title('Training Set Distribution', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, count in enumerate(train_counts):
        axes[0].text(i, count, str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Test set
    axes[1].bar(class_names, test_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_title('Test Set Distribution', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, count in enumerate(test_counts):
        axes[1].text(i, count, str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Class distribution plot saved to {save_path}")

if __name__ == '__main__':
    print("Visualization utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_gan_training_curves()")
    print("  - plot_generated_samples()")
    print("  - plot_real_vs_synthetic_comparison()")
    print("  - plot_classifier_training_curves()")
    print("  - plot_confusion_matrix()")
    print("  - plot_precision_recall_curve()")
    print("  - plot_roc_curve()")
    print("  - create_ablation_results_table()")
    print("  - plot_metrics_comparison()")
    print("  - plot_class_distribution()")
