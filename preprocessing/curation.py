"""
Synthetic Image Curation Pipeline
3-stage filtering: Blur Detection → SSIM → LPIPS
Optimized for RTX 3050 (4GB VRAM)
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import lpips
from skimage.metrics import structural_similarity as ssim
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import shutil

def detect_blur(image_path, threshold=100):
    """
    Filter 1: Blur Detection using Laplacian Variance
    
    Args:
        image_path: Path to image
        threshold: Minimum variance (default 100)
    
    Returns:
        True if image is sharp enough, False if blurry
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False
    
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var > threshold

def filter_by_ssim(synthetic_image_path, real_images_dir, min_ssim=0.3, max_ssim=0.95):
    """
    Filter 2: Structural Similarity Index (SSIM)
    
    Keep if 0.3 < max(SSIM scores) < 0.95
    This filters out exact copies (SSIM~1.0) and extreme outliers (SSIM<0.3)
    
    Args:
        synthetic_image_path: Path to synthetic image
        real_images_dir: Directory containing real training images
        min_ssim: Minimum SSIM threshold
        max_ssim: Maximum SSIM threshold
    
    Returns:
        True if image passes SSIM filter
    """
    # Load synthetic image
    synthetic = cv2.imread(str(synthetic_image_path), cv2.IMREAD_GRAYSCALE)
    if synthetic is None:
        return False
    
    synthetic = cv2.resize(synthetic, (256, 256))
    
    # Sample random real images for comparison (to save time)
    real_image_paths = list(Path(real_images_dir).glob('*.jpg'))
    
    # For efficiency on RTX 3050, compare with only 50 random real images
    if len(real_image_paths) > 50:
        np.random.seed(42)
        real_image_paths = np.random.choice(real_image_paths, 50, replace=False)
    
    max_similarity = 0
    
    for real_path in real_image_paths:
        real = cv2.imread(str(real_path), cv2.IMREAD_GRAYSCALE)
        if real is None:
            continue
        
        real = cv2.resize(real, (256, 256))
        
        try:
            similarity = ssim(synthetic, real)
            max_similarity = max(max_similarity, similarity)
        except:
            continue
    
    return min_ssim < max_similarity < max_ssim

def filter_by_lpips(synthetic_image_path, real_images_dir, min_lpips=0.15, device='cuda'):
    """
    Filter 3: Learned Perceptual Image Patch Similarity (LPIPS)
    
    LPIPS uses VGG features to measure perceptual distance
    Keep if LPIPS > 0.15 (ensures diversity)
    
    Args:
        synthetic_image_path: Path to synthetic image
        real_images_dir: Directory containing real training images
        min_lpips: Minimum LPIPS distance threshold
        device: 'cuda' or 'cpu'
    
    Returns:
        True if image passes LPIPS filter
    """
    # Initialize LPIPS model
    if not hasattr(filter_by_lpips, 'loss_fn'):
        if not torch.cuda.is_available():
            device = 'cpu'
        filter_by_lpips.loss_fn = lpips.LPIPS(net='alex').to(device)  # Use AlexNet for speed on RTX 3050
        filter_by_lpips.loss_fn.eval()
    
    loss_fn = filter_by_lpips.loss_fn
    
    # Load synthetic image
    synthetic_img = Image.open(synthetic_image_path).convert('RGB')
    synthetic_img = synthetic_img.resize((256, 256))
    synthetic_tensor = torch.from_numpy(np.array(synthetic_img)).permute(2, 0, 1).float() / 127.5 - 1.0
    synthetic_tensor = synthetic_tensor.unsqueeze(0).to(device)
    
    # Sample random real images for comparison
    real_image_paths = list(Path(real_images_dir).glob('*.jpg'))
    
    # For RTX 3050: compare with only 30 random real images
    if len(real_image_paths) > 30:
        np.random.seed(42)
        real_image_paths = np.random.choice(real_image_paths, 30, replace=False)
    
    min_distance = float('inf')
    
    with torch.no_grad():
        for real_path in real_image_paths:
            try:
                real_img = Image.open(real_path).convert('RGB')
                real_img = real_img.resize((256, 256))
                real_tensor = torch.from_numpy(np.array(real_img)).permute(2, 0, 1).float() / 127.5 - 1.0
                real_tensor = real_tensor.unsqueeze(0).to(device)
                
                distance = loss_fn(synthetic_tensor, real_tensor).item()
                min_distance = min(min_distance, distance)
            except:
                continue
    
    return min_distance > min_lpips

def automated_curation(synthetic_images_dir='data/synthetic/raw',
                       real_images_dir='data/splits/train/dermatofibroma',
                       output_dir='data/synthetic/filtered',
                       device='cuda'):
    """
    Apply 3-stage automated curation pipeline
    
    Pipeline:
    Stage 1: Blur detection (1500 -> ~1353)
    Stage 2: SSIM filtering (1353 -> ~1264)
    Stage 3: LPIPS filtering (1264 -> ~1140)
    
    Args:
        synthetic_images_dir: Directory with generated images
        real_images_dir: Directory with real training images
        output_dir: Output directory for filtered images
        device: 'cuda' or 'cpu'
    """
    print("="*70)
    print(" AUTOMATED CURATION PIPELINE")
    print("="*70)
    
    if not torch.cuda.is_available():
        device = 'cpu'
        print("⚠️  CUDA not available. Using CPU (will be slower)")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all synthetic images
    synthetic_images = list(Path(synthetic_images_dir).glob('*.png'))
    total_images = len(synthetic_images)
    
    print(f"\nTotal synthetic images: {total_images}")
    print(f"Real images directory: {real_images_dir}\n")
    
    # Stage 1: Blur Detection
    print("Stage 1/3: Blur Detection (Laplacian Variance > 100)")
    print("-"*70)
    stage1_passed = []
    
    for img_path in tqdm(synthetic_images, desc="Blur detection"):
        if detect_blur(img_path, threshold=100):
            stage1_passed.append(img_path)
    
    print(f"✓ Stage 1 complete: {len(stage1_passed)}/{total_images} passed ({len(stage1_passed)/total_images*100:.1f}%)")
    
    # Stage 2: SSIM Filtering
    print(f"\nStage 2/3: SSIM Filtering (0.3 < SSIM < 0.95)")
    print("-"*70)
    stage2_passed = []
    
    for img_path in tqdm(stage1_passed, desc="SSIM filtering"):
        if filter_by_ssim(img_path, real_images_dir, min_ssim=0.3, max_ssim=0.95):
            stage2_passed.append(img_path)
    
    print(f"✓ Stage 2 complete: {len(stage2_passed)}/{len(stage1_passed)} passed ({len(stage2_passed)/len(stage1_passed)*100:.1f}%)")
    
    # Stage 3: LPIPS Filtering
    print(f"\nStage 3/3: LPIPS Filtering (LPIPS > 0.15)")
    print("-"*70)
    stage3_passed = []
    
    for img_path in tqdm(stage2_passed, desc="LPIPS filtering"):
        if filter_by_lpips(img_path, real_images_dir, min_lpips=0.15, device=device):
            stage3_passed.append(img_path)
    
    print(f"✓ Stage 3 complete: {len(stage3_passed)}/{len(stage2_passed)} passed ({len(stage3_passed)/len(stage2_passed)*100:.1f}%)")
    
    # Copy filtered images to output directory
    print(f"\nCopying {len(stage3_passed)} filtered images to {output_dir}/")
    output_path = Path(output_dir)
    for img_path in tqdm(stage3_passed, desc="Copying"):
        shutil.copy2(img_path, output_path / img_path.name)
    
    # Summary
    print("\n" + "="*70)
    print(" CURATION SUMMARY")
    print("="*70)
    print(f"Initial images:           {total_images}")
    print(f"After blur detection:     {len(stage1_passed)} ({len(stage1_passed)/total_images*100:.1f}%)")
    print(f"After SSIM filtering:     {len(stage2_passed)} ({len(stage2_passed)/total_images*100:.1f}%)")
    print(f"After LPIPS filtering:    {len(stage3_passed)} ({len(stage3_passed)/total_images*100:.1f}%)")
    print(f"\n✓ Filtered images saved to: {output_dir}/")
    print("="*70)
    
    return stage3_passed

def diversity_analysis(filtered_images_dir='data/synthetic/filtered',
                      output_dir='data/synthetic/curated',
                      n_clusters=8,
                      min_per_cluster=50,
                      target_total=600):
    """
    Perform t-SNE clustering and select diverse subset
    
    Ensures balanced representation across 8 clusters
    Final output: ~600 high-quality, diverse images
    
    Args:
        filtered_images_dir: Directory with filtered images
        output_dir: Output directory for final curated set
        n_clusters: Number of clusters for diversity
        min_per_cluster: Minimum images per cluster
        target_total: Target total images
    """
    print("\n" + "="*70)
    print(" DIVERSITY ANALYSIS (t-SNE Clustering)")
    print("="*70)
    
    from torchvision import models, transforms
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained ResNet-50 for feature extraction
    print("Loading ResNet-50 for feature extraction...")
    resnet = models.resnet50(weights='IMAGENET1K_V1')
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove classification layer
    resnet = resnet.to(device)
    resnet.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Extract features
    image_paths = list(Path(filtered_images_dir).glob('*.png'))
    features = []
    valid_paths = []
    
    print(f"\nExtracting features from {len(image_paths)} images...")
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Feature extraction"):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                feat = resnet(img_tensor).squeeze().cpu().numpy()
                features.append(feat)
                valid_paths.append(img_path)
            except:
                continue
    
    features = np.array(features)
    print(f"✓ Extracted features: {features.shape}")
    
    # t-SNE dimensionality reduction
    print("\nPerforming t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_features = tsne.fit_transform(features)
    
    # K-means clustering
    print(f"Performing K-means clustering (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tsne_features)
    
    # Visualize clusters
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], 
                         c=clusters, cmap='tab10', alpha=0.6, s=50)
    plt.colorbar(scatter, label='Cluster')
    plt.title('t-SNE Visualization of Synthetic Images (8 Clusters)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    plt.savefig('results/figures/tsne_clusters.png', dpi=300, bbox_inches='tight')
    print(f"✓ t-SNE visualization saved to results/figures/tsne_clusters.png")
    
    # Select balanced subset
    print(f"\nSelecting balanced subset (target: {target_total} images)...")
    selected_paths = []
    
    images_per_cluster = target_total // n_clusters
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_size = len(cluster_indices)
        
        # Select images from this cluster
        n_select = min(images_per_cluster, cluster_size)
        selected_indices = np.random.choice(cluster_indices, n_select, replace=False)
        
        for idx in selected_indices:
            selected_paths.append(valid_paths[idx])
        
        print(f"  Cluster {cluster_id}: {cluster_size} images -> selected {n_select}")
    
    # If we're under target, add more from larger clusters
    if len(selected_paths) < target_total:
        remaining = target_total - len(selected_paths)
        all_indices = set(range(len(valid_paths)))
        selected_indices = set([valid_paths.index(p) for p in selected_paths])
        available_indices = list(all_indices - selected_indices)
        
        additional = np.random.choice(available_indices, 
                                     min(remaining, len(available_indices)), 
                                     replace=False)
        for idx in additional:
            selected_paths.append(valid_paths[idx])
    
    # Copy selected images to curated directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nCopying {len(selected_paths)} curated images to {output_dir}/")
    for i, img_path in enumerate(tqdm(selected_paths, desc="Copying")):
        shutil.copy2(img_path, Path(output_dir) / f"curated_{i:04d}.png")
    
    # Summary
    print("\n" + "="*70)
    print(" DIVERSITY ANALYSIS SUMMARY")
    print("="*70)
    print(f"Filtered images:          {len(image_paths)}")
    print(f"Curated images:           {len(selected_paths)}")
    print(f"Number of clusters:       {n_clusters}")
    print(f"Images per cluster:       ~{len(selected_paths)//n_clusters}")
    print(f"\n✓ Final curated set saved to: {output_dir}/")
    print("="*70)
    
    return selected_paths

if __name__ == '__main__':
    # Run full curation pipeline
    print("Running complete curation pipeline...\n")
    
    # Stage 1-3: Automated filtering
    filtered_images = automated_curation(
        synthetic_images_dir='data/synthetic/raw',
        real_images_dir='data/splits/train/dermatofibroma',
        output_dir='data/synthetic/filtered'
    )
    
    # Stage 4: Diversity analysis and final selection
    if len(filtered_images) > 0:
        curated_images = diversity_analysis(
            filtered_images_dir='data/synthetic/filtered',
            output_dir='data/synthetic/curated',
            n_clusters=8,
            target_total=600
        )
        
        print("\n✅ CURATION PIPELINE COMPLETE!")
        print(f"✅ {len(curated_images)} high-quality, diverse synthetic images ready for training")
    else:
        print("\n⚠️  No images passed filtering. Check GAN quality and adjust thresholds.")
