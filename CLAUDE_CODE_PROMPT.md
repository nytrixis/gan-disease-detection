# Claude Code Prompt: CNN-based Disease Detection via GAN-based Data Augmentation

## Project Overview
Implement a complete deep learning pipeline for rare disease detection (Dermatofibroma) using WGAN-GP for synthetic data generation to address extreme class imbalance (56.7:1 ratio).

## Project Specifications

### Key Requirements
1. **Dataset**: ISIC 2020 Challenge - Dermatofibroma (300 samples) vs Melanocytic Nevus (17,000 samples)
2. **GAN Architecture**: WGAN-GP (Wasserstein GAN with Gradient Penalty)
3. **CNN Classifier**: ResNet-50, DenseNet-121, EfficientNet-B0
4. **Target Metrics**: F1-Score improvement from 0.51 (baseline) to 0.76 (with GAN augmentation)
5. **Framework**: PyTorch
6. **Hardware**: NVIDIA GPU (RTX 3060 or better recommended)

### Project Structure Required
```
gan-disease-detection/
├── data/
│   ├── raw/                    # Original ISIC 2020 dataset
│   ├── processed/              # Preprocessed images (256x256)
│   ├── splits/                 # Train/val/test splits
│   └── synthetic/              # Generated synthetic images
├── models/
│   ├── wgan_gp.py             # WGAN-GP implementation
│   ├── generator.py           # Generator architecture
│   ├── discriminator.py       # Critic/discriminator architecture
│   ├── classifiers.py         # ResNet-50, DenseNet-121, EfficientNet-B0
│   └── losses.py              # Custom loss functions
├── training/
│   ├── train_gan.py           # GAN training script
│   ├── train_classifier.py    # CNN classifier training
│   └── config.py              # Hyperparameters configuration
├── evaluation/
│   ├── metrics.py             # FID, IS, LPIPS, SSIM, PRC
│   ├── statistical_tests.py   # McNemar, Wilcoxon tests
│   └── ablation_study.py      # Ablation experiments
├── preprocessing/
│   ├── data_loader.py         # Dataset loading and preprocessing
│   ├── augmentation.py        # Traditional augmentation baseline
│   └── curation.py            # Synthetic image quality filtering
├── utils/
│   ├── visualization.py       # Training curves, sample images
│   ├── logger.py              # TensorBoard logging
│   └── checkpoints.py         # Model checkpointing
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_gan_training.ipynb
│   ├── 03_classifier_training.ipynb
│   └── 04_results_analysis.ipynb
├── requirements.txt
├── README.md
└── main.py                    # Main execution script
```

## Detailed Implementation Instructions

### Phase 1: Environment Setup and Data Preparation

**Task 1.1: Environment Configuration**
```python
# Create requirements.txt with exact versions:
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
pillow==10.0.0
opencv-python==4.8.0.74
matplotlib==3.7.2
seaborn==0.12.2
tensorboard==2.13.0
tqdm==4.65.0
scipy==1.11.1
lpips==0.1.4
pytorch-fid==0.3.0
```

**Task 1.2: Dataset Download and Organization**
- Download ISIC 2020 dataset from: https://challenge2020.isic-archive.com/
- Use "Download JPEG (23GB)" option for 33,126 JPEG images
- Download metadata CSV files containing diagnosis labels
- Filter for Dermatofibroma (DF) and Melanocytic Nevus (NV) only
- Organize into directory structure:
  - data/raw/dermatofibroma/ (300 images)
  - data/raw/nevus/ (17,000 images)

**Task 1.3: Preprocessing Pipeline**
```python
# preprocessing/data_loader.py specifications:
- Resize all images from 1024x1024 to 256x256
- Apply lesion-centered cropping using ISIC segmentation masks
- Normalize pixel values to [-1, 1] range for GAN training
- Create stratified 70/15/15 splits:
  - Dermatofibroma: 210 train, 45 val, 45 test
  - Nevus: 11,900 train, 2,550 val, 2,550 test
- Use random seed=42 for reproducibility
- Save processed images as .png files
- Generate metadata.csv with image paths, labels, split assignments
```

### Phase 2: WGAN-GP Implementation

**Task 2.1: Generator Architecture**
```python
# models/generator.py specifications:
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, features_g=16):
        # Input: latent_dim (100D) random noise vector
        # Output: 256x256x3 RGB image
        
        # Architecture (5 transposed convolutional layers):
        # Layer 1: Linear(100, 4*4*1024) + Reshape to (1024, 4, 4)
        # Layer 2: ConvTranspose2d(1024, 512, 4, 2, 1) -> 8x8
        # Layer 3: ConvTranspose2d(512, 256, 4, 2, 1) -> 16x16
        # Layer 4: ConvTranspose2d(256, 128, 4, 2, 1) -> 32x32
        # Layer 5: ConvTranspose2d(128, 64, 4, 2, 1) -> 64x64
        # Layer 6: ConvTranspose2d(64, 32, 4, 2, 1) -> 128x128
        # Layer 7: ConvTranspose2d(32, 3, 4, 2, 1) -> 256x256
        
        # Use BatchNorm2d after each layer except output
        # Use ReLU activation for all layers except output
        # Use Tanh activation for output layer (range [-1, 1])
```

**Task 2.2: Discriminator/Critic Architecture**
```python
# models/discriminator.py specifications:
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features_d=16):
        # Input: 256x256x3 RGB image
        # Output: Single scalar (Wasserstein distance estimate)
        
        # Architecture (5 convolutional layers):
        # Layer 1: Conv2d(3, 32, 4, 2, 1) -> 128x128
        # Layer 2: Conv2d(32, 64, 4, 2, 1) -> 64x64
        # Layer 3: Conv2d(64, 128, 4, 2, 1) -> 32x32
        # Layer 4: Conv2d(128, 256, 4, 2, 1) -> 16x16
        # Layer 5: Conv2d(256, 512, 4, 2, 1) -> 8x8
        # Layer 6: Conv2d(512, 1024, 4, 2, 1) -> 4x4
        # Layer 7: Conv2d(1024, 1, 4, 1, 0) -> 1x1 (Flatten to scalar)
        
        # Use InstanceNorm2d (NOT BatchNorm) after each layer except first and last
        # Use LeakyReLU(0.2) activation for all layers
        # NO activation on output layer
```

**Task 2.3: WGAN-GP Training Loop**
```python
# training/train_gan.py specifications:
HYPERPARAMETERS = {
    'latent_dim': 100,
    'batch_size': 16,
    'lr_generator': 0.0001,
    'lr_critic': 0.0001,
    'beta1': 0.5,
    'beta2': 0.999,
    'lambda_gp': 10,  # Gradient penalty coefficient
    'critic_iterations': 5,  # Train critic 5 times per generator iteration
    'num_epochs': 240,
    'total_iterations': 50000,
    'checkpoint_interval': 5000,
    'sample_interval': 1000
}

# Gradient Penalty Calculation:
def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    
    # Calculate critic scores
    mixed_scores = critic(interpolated)
    
    # Compute gradients
    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

# Training Loop:
for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(dataloader):
        # Train Critic
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(BATCH_SIZE, latent_dim).to(device)
            fake = generator(noise)
            critic_real = critic(real_images).reshape(-1)
            critic_fake = critic(fake.detach()).reshape(-1)
            gp = gradient_penalty(critic, real_images, fake, device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp
            critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()
        
        # Train Generator
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        generator.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()
        
        # Logging
        if batch_idx % 100 == 0:
            writer.add_scalar('Loss/Critic', loss_critic.item(), global_step)
            writer.add_scalar('Loss/Generator', loss_gen.item(), global_step)
            writer.add_scalar('Wasserstein_Distance', 
                            torch.mean(critic_real) - torch.mean(critic_fake), 
                            global_step)
```

**Task 2.4: Synthetic Image Generation**
```python
# Generate 1500 initial synthetic images
generator.eval()
with torch.no_grad():
    for i in range(1500):
        noise = torch.randn(1, latent_dim).to(device)
        fake_image = generator(noise)
        fake_image = (fake_image + 1) / 2  # Denormalize from [-1,1] to [0,1]
        save_image(fake_image, f'data/synthetic/raw/generated_{i:04d}.png')
```

### Phase 3: Synthetic Image Curation Pipeline

**Task 3.1: Automated Quality Filtering**
```python
# preprocessing/curation.py specifications:

# Filter 1: Blur Detection using Laplacian Variance
def detect_blur(image_path, threshold=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var > threshold  # Keep if variance > 100

# Filter 2: Structural Similarity Index (SSIM)
def filter_by_ssim(synthetic_image, real_images_dataset, min_ssim=0.3, max_ssim=0.95):
    # Compare synthetic image to all real training images
    # Keep if 0.3 < max(SSIM scores) < 0.95
    # This filters out exact copies (SSIM~1.0) and extreme outliers (SSIM<0.3)
    from skimage.metrics import structural_similarity as ssim
    
    synthetic = cv2.imread(synthetic_image, cv2.IMREAD_GRAYSCALE)
    max_similarity = 0
    
    for real_image in real_images_dataset:
        real = cv2.imread(real_image, cv2.IMREAD_GRAYSCALE)
        similarity = ssim(synthetic, real)
        max_similarity = max(max_similarity, similarity)
    
    return min_ssim < max_similarity < max_ssim

# Filter 3: Learned Perceptual Image Patch Similarity (LPIPS)
import lpips
def filter_by_lpips(synthetic_image, real_images_dataset, min_lpips=0.15):
    # LPIPS uses VGG features to measure perceptual distance
    # Keep if LPIPS > 0.15 (ensures diversity)
    loss_fn = lpips.LPIPS(net='vgg').to(device)
    
    synthetic_tensor = load_image_as_tensor(synthetic_image)
    min_distance = float('inf')
    
    for real_image in real_images_dataset:
        real_tensor = load_image_as_tensor(real_image)
        distance = loss_fn(synthetic_tensor, real_tensor).item()
        min_distance = min(min_distance, distance)
    
    return min_distance > min_lpips

# Apply all filters sequentially
def automated_curation(synthetic_images_dir, real_images_dir, output_dir):
    # Stage 1: Blur detection (1500 -> ~1353)
    # Stage 2: SSIM filtering (1353 -> ~1264)
    # Stage 3: LPIPS filtering (1264 -> ~1140)
    pass
```

**Task 3.2: Manual Review Protocol**
```python
# Create manual review interface (can be simple script)
# Three reviewers assess each image on 5-point scale:
# 1. Overall realism (1-5)
# 2. Color authenticity (1-5)
# 3. Texture accuracy (1-5)
# 4. Lesion boundary definition (1-5)
# 5. Presence of artifacts (1-5 where 5=no artifacts)

# Keep images with average score >= 3.5 across all criteria
# This reduces 1140 images to ~700 high-quality samples
```

**Task 3.3: Diversity Analysis**
```python
# Use t-SNE for clustering analysis
from sklearn.manifold import TSNE
from torchvision.models import resnet50

# Extract features using pre-trained ResNet-50
def extract_features(image_paths):
    model = resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
    model.eval()
    
    features = []
    for img_path in image_paths:
        img = load_and_preprocess_image(img_path)
        with torch.no_grad():
            feat = model(img).squeeze()
        features.append(feat.cpu().numpy())
    
    return np.array(features)

# Perform t-SNE clustering
features = extract_features(curated_synthetic_images)
tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(features)

# Identify 8 clusters using K-means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, random_state=42)
clusters = kmeans.fit_predict(tsne_features)

# Ensure minimum 50 images per cluster in final selection
# Final curated set: 600 images with balanced cluster representation
```

### Phase 4: CNN Classifier Training

**Task 4.1: Baseline Model (No Augmentation)**
```python
# models/classifiers.py
from torchvision.models import resnet50, densenet121, efficientnet_b0

# ResNet-50 Configuration
model = resnet50(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification

# Training Configuration
HYPERPARAMETERS = {
    'batch_size': 32,
    'learning_rate': 0.0001,
    'num_epochs': 50,
    'optimizer': 'Adam',
    'weight_decay': 1e-4,
    'early_stopping_patience': 10
}

# Weighted Cross-Entropy Loss (to handle imbalance)
class_weights = torch.tensor([1.0, 56.7])  # Weight for [nevus, dermatofibroma]
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training Dataset: 210 DF + 11,900 NV (imbalanced)
```

**Task 4.2: Traditional Augmentation Baseline**
```python
# preprocessing/augmentation.py
from torchvision import transforms

traditional_augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply only to minority class (Dermatofibroma) during training
```

**Task 4.3: GAN-Augmented Training**
```python
# Training Dataset: 210 real DF + 600 synthetic DF + 11,900 NV
# Total DF: 810 samples (reduces imbalance from 56.7:1 to 14.7:1)

# Ensure synthetic images are clearly labeled in metadata
# Use same hyperparameters as baseline for fair comparison
```

**Task 4.4: Multiple Architecture Comparison**
```python
# Train three architectures with identical hyperparameters:
architectures = {
    'ResNet-50': resnet50(weights='IMAGENET1K_V1'),
    'DenseNet-121': densenet121(weights='IMAGENET1K_V1'),
    'EfficientNet-B0': efficientnet_b0(weights='IMAGENET1K_V1')
}

# Modify final layers for binary classification
for name, model in architectures.items():
    if 'resnet' in name.lower():
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif 'densenet' in name.lower():
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif 'efficientnet' in name.lower():
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
```

### Phase 5: Evaluation and Metrics

**Task 5.1: Classification Metrics**
```python
# evaluation/metrics.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve

def evaluate_classifier(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, pos_label=1),
        'recall': recall_score(all_labels, all_preds, pos_label=1),
        'f1_score': f1_score(all_labels, all_preds, pos_label=1),
        'roc_auc': roc_auc_score(all_labels, all_probs),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    
    return metrics

# Target Results:
# Baseline: Accuracy 82.4%, Precision 64.3%, Recall 28.1%, F1 0.51
# Traditional Aug: Accuracy 84.9%, Precision 71.2%, Recall 42.0%, F1 0.526
# GAN Aug: Accuracy 88.5%, Precision 78.3%, Recall 59.5%, F1 0.76
```

**Task 5.2: GAN Quality Metrics**
```python
# Fréchet Inception Distance (FID)
from pytorch_fid import fid_score

def calculate_fid(real_images_path, synthetic_images_path, device):
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_path, synthetic_images_path],
        batch_size=50,
        device=device,
        dims=2048
    )
    return fid_value
# Target FID: 38 ± 2 (lower is better)

# Inception Score (IS)
def calculate_inception_score(images, device, splits=10):
    from torchvision.models import inception_v3
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    # Calculate IS using standard formula
    # Target IS: 2.8 ± 0.3
    pass

# LPIPS (Learned Perceptual Image Patch Similarity)
import lpips
loss_fn = lpips.LPIPS(net='vgg')
def calculate_lpips(img1, img2):
    return loss_fn(img1, img2).item()
# Target LPIPS: 0.42 ± 0.08 (from real images)

# SSIM (Structural Similarity Index)
from skimage.metrics import structural_similarity as ssim
def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True)
# Target SSIM: 0.68 ± 0.12

# Precision-Recall Curve for Fidelity vs Diversity
def precision_recall_curve_gan(real_features, fake_features):
    # Implementation based on improved precision and recall metrics for GANs
    # Reference: "Improved Precision and Recall Metric for Assessing Generative Models" (2019)
    pass
```

**Task 5.3: Statistical Significance Tests**
```python
# evaluation/statistical_tests.py
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

# McNemar's Test (for comparing two classifiers)
def mcnemar_test(predictions_model1, predictions_model2, ground_truth):
    # Create contingency table
    correct_1 = (predictions_model1 == ground_truth)
    correct_2 = (predictions_model2 == ground_truth)
    
    # Count cases
    both_correct = np.sum(correct_1 & correct_2)
    both_wrong = np.sum(~correct_1 & ~correct_2)
    model1_correct_model2_wrong = np.sum(correct_1 & ~correct_2)
    model1_wrong_model2_correct = np.sum(~correct_1 & correct_2)
    
    table = [[both_correct, model1_correct_model2_wrong],
             [model1_wrong_model2_correct, both_wrong]]
    
    result = mcnemar(table, exact=False, correction=True)
    return result.pvalue  # Target: p < 0.05 for significance

# Wilcoxon Signed-Rank Test (for paired samples)
def wilcoxon_test(scores_model1, scores_model2):
    statistic, pvalue = wilcoxon(scores_model1, scores_model2)
    return pvalue  # Target: p < 0.01 for significance
```

**Task 5.4: Ablation Studies**
```python
# evaluation/ablation_study.py
# Run 5 experiments:
experiments = [
    {
        'name': 'Baseline',
        'synthetic_count': 0,
        'curation': False,
        'expected_f1': 0.51
    },
    {
        'name': 'Traditional Aug',
        'synthetic_count': 0,
        'curation': False,
        'expected_f1': 0.526
    },
    {
        'name': 'GAN (Uncurated)',
        'synthetic_count': 1000,
        'curation': False,
        'expected_f1': 0.68
    },
    {
        'name': 'GAN (Curated)',
        'synthetic_count': 600,
        'curation': True,
        'expected_f1': 0.76
    },
    {
        'name': 'Synthetic Only',
        'synthetic_count': 1000,
        'curation': True,
        'use_real': False,
        'expected_f1': 0.54
    }
]

# Train separate models for each experiment
# Compare results in comprehensive table
```

### Phase 6: Visualization and Reporting

**Task 6.1: Training Visualizations**
```python
# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns

# 1. GAN Training Curves
def plot_gan_training_curves(losses_critic, losses_generator, wasserstein_distances):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(losses_critic)
    axes[0].set_title('Critic Loss')
    axes[1].plot(losses_generator)
    axes[1].set_title('Generator Loss')
    axes[2].plot(wasserstein_distances)
    axes[2].set_title('Wasserstein Distance')
    plt.savefig('results/gan_training_curves.png')

# 2. Generated Image Samples Grid
def plot_generated_samples(generator, num_samples=25):
    # Generate 5x5 grid of synthetic images
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(25):
        noise = torch.randn(1, 100).to(device)
        fake = generator(noise).detach().cpu()
        fake = (fake + 1) / 2  # Denormalize
        axes[i//5, i%5].imshow(fake.squeeze().permute(1, 2, 0))
        axes[i//5, i%5].axis('off')
    plt.savefig('results/generated_samples_grid.png')

# 3. Real vs Synthetic Comparison
def plot_real_vs_synthetic_comparison():
    # Side-by-side comparison of 10 real and 10 synthetic images
    pass

# 4. Classifier Training Curves
def plot_classifier_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Validation')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    
    axes[1].plot(train_accs, label='Train')
    axes[1].plot(val_accs, label='Validation')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    plt.savefig('results/classifier_training_curves.png')

# 5. Confusion Matrix
def plot_confusion_matrix(cm, class_names=['Nevus', 'Dermatofibroma']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')

# 6. Precision-Recall Curve
def plot_precision_recall_curve(precision, recall, thresholds):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('results/precision_recall_curve.png')

# 7. Ablation Study Results Table
def create_ablation_results_table(results):
    # Create formatted table with all experimental results
    pass
```

**Task 6.2: Results Documentation**
```python
# Create comprehensive results report
def generate_results_report():
    report = {
        'dataset_statistics': {
            'total_images': 17300,
            'dermatofibroma_original': 300,
            'nevus': 17000,
            'imbalance_ratio': 56.7,
            'synthetic_generated': 1500,
            'synthetic_after_curation': 600
        },
        'gan_metrics': {
            'fid_score': 38.2,
            'inception_score': 2.85,
            'lpips': 0.42,
            'ssim': 0.68,
            'training_time_hours': 18,
            'total_iterations': 50000
        },
        'classifier_results': {
            'baseline': {
                'accuracy': 0.824,
                'precision': 0.643,
                'recall': 0.281,
                'f1': 0.51
            },
            'traditional_aug': {
                'accuracy': 0.849,
                'precision': 0.712,
                'recall': 0.420,
                'f1': 0.526
            },
            'gan_augmented': {
                'accuracy': 0.885,
                'precision': 0.783,
                'recall': 0.595,
                'f1': 0.76
            }
        },
        'statistical_significance': {
            'mcnemar_pvalue': 0.023,  # < 0.05
            'wilcoxon_pvalue': 0.008   # < 0.01
        },
        'improvement_metrics': {
            'f1_improvement_percentage': 49.0,
            'recall_improvement_percentage': 111.7
        }
    }
    
    # Save as JSON and generate PDF report
    with open('results/final_results.json', 'w') as f:
        json.dump(report, f, indent=4)
```

## Execution Strategy

### Step-by-Step Implementation Order

1. **Week 1: Setup and Preprocessing**
   - Setup environment and install dependencies
   - Download ISIC 2020 dataset
   - Implement preprocessing pipeline
   - Create train/val/test splits
   - Verify data loading works correctly

2. **Week 2: WGAN-GP Implementation**
   - Implement Generator architecture
   - Implement Discriminator/Critic architecture
   - Implement gradient penalty calculation
   - Setup training loop with TensorBoard logging
   - Test with small batch to verify training works

3. **Week 3: GAN Training**
   - Train WGAN-GP on Dermatofibroma images (210 samples)
   - Monitor training curves and generated samples
   - Generate 1500 synthetic images
   - Calculate FID, IS, LPIPS, SSIM scores

4. **Week 4: Curation Pipeline**
   - Implement automated quality filters (blur, SSIM, LPIPS)
   - Filter 1500 -> 1140 images
   - Conduct manual review (or simulate if no reviewers)
   - Filter 1140 -> 700 images
   - Perform diversity analysis with t-SNE
   - Select final 600 curated images

5. **Week 5: Baseline Classifier Training**
   - Implement ResNet-50 classifier
   - Train baseline (no augmentation)
   - Train with traditional augmentation
   - Evaluate both models on test set
   - Document baseline results

6. **Week 6: GAN-Augmented Training**
   - Train ResNet-50 with GAN-augmented dataset
   - Train DenseNet-121 with GAN-augmented dataset
   - Train EfficientNet-B0 with GAN-augmented dataset
   - Compare all architectures

7. **Week 7: Ablation Studies**
   - Train model with uncurated synthetic data
   - Train model with synthetic data only
   - Compare all 5 experimental conditions
   - Perform statistical significance tests

8. **Week 8: Evaluation and Visualization**
   - Generate all visualizations
   - Calculate all metrics
   - Create results tables
   - Generate comprehensive report
   - Prepare presentation materials

## Key Implementation Notes

### Critical Details for Success

1. **Random Seeds**: Use `seed=42` everywhere for reproducibility
   ```python
   import random
   import numpy as np
   import torch
   
   def set_seed(seed=42):
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
   ```

2. **Gradient Penalty Lambda**: MUST be exactly 10 (as per WGAN-GP paper)

3. **Critic Training Frequency**: Train critic 5 times per generator update

4. **Image Normalization**: 
   - For GAN: [-1, 1] range using Tanh
   - For CNN: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

5. **Loss Function**: Use nn.CrossEntropyLoss with class weights [1.0, 56.7]

6. **Early Stopping**: Patience of 10 epochs based on validation F1-score

7. **Learning Rate**: 0.0001 for both GAN and CNN training

8. **Batch Size**: 16 for GAN, 32 for CNN

9. **Image Size**: All operations at 256x256 resolution

10. **GPU Memory**: Training requires ~8GB GPU memory minimum

### Common Pitfalls to Avoid

1. **Mode Collapse**: Monitor generator outputs regularly; if repetitive, adjust learning rate
2. **Gradient Explosion**: Use gradient clipping if losses become unstable
3. **Data Leakage**: Ensure test set is never seen during training or curation
4. **Overfitting**: Use early stopping and monitor validation metrics
5. **Class Imbalance**: Always use weighted loss function
6. **Synthetic Overfitting**: Ensure real data is always included in training

## Expected Outputs

### Deliverables

1. **Trained Models**:
   - WGAN-GP generator checkpoint (best_generator.pth)
   - ResNet-50 baseline (baseline_resnet50.pth)
   - ResNet-50 with traditional aug (traditional_aug_resnet50.pth)
   - ResNet-50 with GAN aug (gan_aug_resnet50.pth)
   - DenseNet-121 with GAN aug (gan_aug_densenet121.pth)
   - EfficientNet-B0 with GAN aug (gan_aug_efficientnet.pth)

2. **Datasets**:
   - 600 curated synthetic Dermatofibroma images
   - metadata.csv with all image information

3. **Results**:
   - final_results.json with all metrics
   - Confusion matrices for all models
   - Training curves for GAN and classifiers
   - Sample grids of real vs synthetic images
   - Ablation study comparison table
   - Statistical significance test results

4. **Documentation**:
   - README.md with setup instructions
   - requirements.txt with exact versions
   - Training logs from TensorBoard
   - Jupyter notebooks with analysis

## Timeline and Milestones

- **Day 1-7**: Environment setup, data preprocessing complete
- **Day 8-14**: WGAN-GP implementation complete
- **Day 15-21**: GAN training complete, 1500 synthetic images generated
- **Day 22-28**: Curation complete, 600 final synthetic images
- **Day 29-35**: Baseline classifier training complete
- **Day 36-42**: GAN-augmented classifier training complete
- **Day 43-49**: Ablation studies complete
- **Day 50-56**: All evaluations, visualizations, and documentation complete

## Success Criteria

Project is complete when:
- ✅ GAN generates realistic Dermatofibroma images (FID < 40)
- ✅ Curated dataset contains 600 high-quality synthetic images
- ✅ Baseline F1-score achieved: ~0.51
- ✅ Traditional augmentation F1-score achieved: ~0.526
- ✅ GAN-augmented F1-score achieved: ≥0.75 (target 0.76)
- ✅ Statistical significance confirmed (p < 0.05)
- ✅ All visualizations generated
- ✅ Complete documentation prepared

## Additional Resources

### Helpful Code References
1. WGAN-GP PyTorch: https://github.com/EmilienDupont/wgan-gp
2. Medical GAN synthesis: https://github.com/ginobilinie/medSynthesisV1
3. ISIC dataset examples: https://github.com/ImagingInformatics/dermatology-images

### Research Papers to Reference
1. Improved Training of Wasserstein GANs (Gulrajani et al., 2017)
2. Synthetic Data Augmentation using GAN for Improved Liver Lesion Classification (Frid-Adar et al., 2018)
3. Deep Residual Learning for Image Recognition (He et al., 2016)

### Debugging Tips
1. If GAN training is unstable: Reduce learning rates by 50%
2. If mode collapse occurs: Increase critic training iterations to 7-10
3. If GPU OOM: Reduce batch size to 8 for GAN, 16 for CNN
4. If F1-score not improving: Check class weight computation
5. If FID score too high (>50): Train GAN for more iterations

---

**IMPORTANT**: This prompt provides a complete specification. Implement systematically, test each component individually, and document everything. The project is well-designed and should achieve the target metrics if implemented correctly.
