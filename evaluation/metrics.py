"""
Comprehensive metrics for GAN and Classifier evaluation
Includes: FID, IS, LPIPS, SSIM, Classification metrics
Optimized for RTX 3050 (4GB VRAM)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import lpips
from skimage.metrics import structural_similarity as ssim
import cv2
from scipy import linalg
from torchvision import models, transforms
from torch.nn.functional import adaptive_avg_pool2d

# ============================================================================
# GAN Quality Metrics
# ============================================================================

def calculate_fid(real_images_path, synthetic_images_path, batch_size=50, device='cuda'):
    """
    Calculate Fréchet Inception Distance (FID)
    
    Lower is better. Target: < 40
    
    Args:
        real_images_path: Path to real images directory
        synthetic_images_path: Path to synthetic images directory
        batch_size: Batch size for processing (reduced for RTX 3050)
        device: 'cuda' or 'cpu'
    
    Returns:
        FID score (float)
    """
    print("Calculating FID score...")
    
    if not torch.cuda.is_available():
        device = 'cpu'
        print("⚠️  CUDA not available, using CPU (slower)")
    
    # Load InceptionV3 model
    from torchvision.models import inception_v3
    
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = nn.Identity()  # Remove final layer
    inception_model = inception_model.to(device)
    inception_model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def get_activations(image_paths, model, batch_size, device):
        """Extract features from images"""
        activations = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
                batch_paths = image_paths[i:i+batch_size]
                batch_images = []
                
                for img_path in batch_paths:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = transform(img)
                        batch_images.append(img)
                    except:
                        continue
                
                if len(batch_images) > 0:
                    batch_tensor = torch.stack(batch_images).to(device)
                    features = model(batch_tensor)
                    activations.append(features.cpu().numpy())
        
        return np.concatenate(activations, axis=0)
    
    # Get image paths
    real_paths = list(Path(real_images_path).glob('*.jpg')) + list(Path(real_images_path).glob('*.png'))
    synthetic_paths = list(Path(synthetic_images_path).glob('*.png')) + list(Path(synthetic_images_path).glob('*.jpg'))
    
    print(f"Real images: {len(real_paths)}")
    print(f"Synthetic images: {len(synthetic_paths)}")
    
    # Extract features
    real_features = get_activations(real_paths, inception_model, batch_size, device)
    synthetic_features = get_activations(synthetic_paths, inception_model, batch_size, device)
    
    # Calculate statistics
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_synthetic = np.mean(synthetic_features, axis=0)
    sigma_synthetic = np.cov(synthetic_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_synthetic
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_synthetic), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_synthetic - 2 * covmean)
    
    print(f"✓ FID Score: {fid:.2f}")
    return float(fid)

def calculate_inception_score(images_path, batch_size=32, splits=10, device='cuda'):
    """
    Calculate Inception Score (IS)
    
    Higher is better. Target: > 2.5
    
    Args:
        images_path: Path to images directory
        batch_size: Batch size for processing
        splits: Number of splits for computing std
        device: 'cuda' or 'cpu'
    
    Returns:
        (mean, std) of Inception Score
    """
    print("Calculating Inception Score...")
    
    if not torch.cuda.is_available():
        device = 'cpu'
    
    from torchvision.models import inception_v3
    
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model = inception_model.to(device)
    inception_model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get predictions
    image_paths = list(Path(images_path).glob('*.png')) + list(Path(images_path).glob('*.jpg'))
    preds = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Computing predictions"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = transform(img)
                    batch_images.append(img)
                except:
                    continue
            
            if len(batch_images) > 0:
                batch_tensor = torch.stack(batch_images).to(device)
                pred = inception_model(batch_tensor)
                pred = torch.softmax(pred, dim=1)
                preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # Calculate IS
    split_scores = []
    
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.sum(pyx * np.log(pyx / py + 1e-10)))
        split_scores.append(np.exp(np.mean(scores)))
    
    is_mean = np.mean(split_scores)
    is_std = np.std(split_scores)
    
    print(f"✓ Inception Score: {is_mean:.3f} ± {is_std:.3f}")
    return is_mean, is_std

def calculate_lpips_diversity(images_path, n_samples=100, device='cuda'):
    """
    Calculate average LPIPS distance between random image pairs
    Measures diversity of generated images
    
    Target: ~0.42 (similar to real image diversity)
    
    Args:
        images_path: Path to images directory
        n_samples: Number of random pairs to compare
        device: 'cuda' or 'cpu'
    
    Returns:
        Mean LPIPS distance
    """
    print("Calculating LPIPS diversity...")
    
    if not torch.cuda.is_available():
        device = 'cpu'
    
    # Use AlexNet for speed on RTX 3050
    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()
    
    image_paths = list(Path(images_path).glob('*.png')) + list(Path(images_path).glob('*.jpg'))
    
    if len(image_paths) < 2:
        print("⚠️  Not enough images for LPIPS calculation")
        return 0.0
    
    distances = []
    
    with torch.no_grad():
        for _ in tqdm(range(n_samples), desc="Computing LPIPS"):
            # Sample two random images
            idx1, idx2 = np.random.choice(len(image_paths), 2, replace=False)
            
            try:
                img1 = Image.open(image_paths[idx1]).convert('RGB').resize((256, 256))
                img2 = Image.open(image_paths[idx2]).convert('RGB').resize((256, 256))
                
                img1_tensor = torch.from_numpy(np.array(img1)).permute(2, 0, 1).float() / 127.5 - 1.0
                img2_tensor = torch.from_numpy(np.array(img2)).permute(2, 0, 1).float() / 127.5 - 1.0
                
                img1_tensor = img1_tensor.unsqueeze(0).to(device)
                img2_tensor = img2_tensor.unsqueeze(0).to(device)
                
                distance = loss_fn(img1_tensor, img2_tensor).item()
                distances.append(distance)
            except:
                continue
    
    mean_lpips = np.mean(distances)
    std_lpips = np.std(distances)
    
    print(f"✓ LPIPS Diversity: {mean_lpips:.3f} ± {std_lpips:.3f}")
    return mean_lpips, std_lpips

def calculate_ssim_similarity(real_images_path, synthetic_images_path, n_samples=100):
    """
    Calculate average SSIM between synthetic and real images
    Measures structural similarity
    
    Target: ~0.68
    
    Args:
        real_images_path: Path to real images
        synthetic_images_path: Path to synthetic images
        n_samples: Number of random pairs to compare
    
    Returns:
        Mean SSIM score
    """
    print("Calculating SSIM similarity...")
    
    real_paths = list(Path(real_images_path).glob('*.jpg')) + list(Path(real_images_path).glob('*.png'))
    synthetic_paths = list(Path(synthetic_images_path).glob('*.png')) + list(Path(synthetic_images_path).glob('*.jpg'))
    
    ssim_scores = []
    
    for _ in tqdm(range(n_samples), desc="Computing SSIM"):
        # Sample random real and synthetic image
        real_path = np.random.choice(real_paths)
        synthetic_path = np.random.choice(synthetic_paths)
        
        try:
            real_img = cv2.imread(str(real_path), cv2.IMREAD_GRAYSCALE)
            synthetic_img = cv2.imread(str(synthetic_path), cv2.IMREAD_GRAYSCALE)
            
            real_img = cv2.resize(real_img, (256, 256))
            synthetic_img = cv2.resize(synthetic_img, (256, 256))
            
            score = ssim(real_img, synthetic_img)
            ssim_scores.append(score)
        except:
            continue
    
    mean_ssim = np.mean(ssim_scores)
    std_ssim = np.std(ssim_scores)
    
    print(f"✓ SSIM Similarity: {mean_ssim:.3f} ± {std_ssim:.3f}")
    return mean_ssim, std_ssim

# ============================================================================
# Classification Metrics
# ============================================================================

def calculate_classification_metrics(y_true, y_pred, y_probs=None):
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities (optional, for ROC-AUC)
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                 f1_score, confusion_matrix, roc_auc_score,
                                 classification_report)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_probs is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs)
        except:
            metrics['roc_auc'] = None
    
    # Calculate specificity and sensitivity
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics

def evaluate_gan_quality(real_images_path, synthetic_images_path, device='cuda'):
    """
    Comprehensive GAN quality evaluation
    
    Args:
        real_images_path: Path to real images
        synthetic_images_path: Path to synthetic images
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with all GAN metrics
    """
    print("\n" + "="*70)
    print(" GAN QUALITY METRICS")
    print("="*70 + "\n")
    
    results = {}
    
    # FID Score
    try:
        results['fid'] = calculate_fid(real_images_path, synthetic_images_path, 
                                       batch_size=32, device=device)
    except Exception as e:
        print(f"⚠️  FID calculation failed: {e}")
        results['fid'] = None
    
    # Inception Score
    try:
        is_mean, is_std = calculate_inception_score(synthetic_images_path, 
                                                     batch_size=32, device=device)
        results['inception_score_mean'] = is_mean
        results['inception_score_std'] = is_std
    except Exception as e:
        print(f"⚠️  IS calculation failed: {e}")
        results['inception_score_mean'] = None
        results['inception_score_std'] = None
    
    # LPIPS Diversity
    try:
        lpips_mean, lpips_std = calculate_lpips_diversity(synthetic_images_path, 
                                                          n_samples=100, device=device)
        results['lpips_mean'] = lpips_mean
        results['lpips_std'] = lpips_std
    except Exception as e:
        print(f"⚠️  LPIPS calculation failed: {e}")
        results['lpips_mean'] = None
        results['lpips_std'] = None
    
    # SSIM Similarity
    try:
        ssim_mean, ssim_std = calculate_ssim_similarity(real_images_path, 
                                                        synthetic_images_path, 
                                                        n_samples=100)
        results['ssim_mean'] = ssim_mean
        results['ssim_std'] = ssim_std
    except Exception as e:
        print(f"⚠️  SSIM calculation failed: {e}")
        results['ssim_mean'] = None
        results['ssim_std'] = None
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"FID Score:        {results['fid']:.2f} (target: < 40)" if results['fid'] else "FID Score:        N/A")
    print(f"Inception Score:  {results['inception_score_mean']:.3f} ± {results['inception_score_std']:.3f} (target: > 2.5)" if results['inception_score_mean'] else "Inception Score:  N/A")
    print(f"LPIPS Diversity:  {results['lpips_mean']:.3f} ± {results['lpips_std']:.3f} (target: ~0.42)" if results['lpips_mean'] else "LPIPS Diversity:  N/A")
    print(f"SSIM Similarity:  {results['ssim_mean']:.3f} ± {results['ssim_std']:.3f} (target: ~0.68)" if results['ssim_mean'] else "SSIM Similarity:  N/A")
    print("="*70 + "\n")
    
    return results

if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        real_path = sys.argv[1] if len(sys.argv) > 1 else 'data/splits/train/dermatofibroma'
        synthetic_path = sys.argv[2] if len(sys.argv) > 2 else 'data/synthetic/curated'
        
        results = evaluate_gan_quality(real_path, synthetic_path)
        
        # Save results
        import json
        from pathlib import Path
        
        Path('results/tables').mkdir(parents=True, exist_ok=True)
        with open('results/tables/gan_metrics.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print("✓ Results saved to results/tables/gan_metrics.json")
    else:
        print("Usage: python metrics.py <real_images_path> <synthetic_images_path>")
