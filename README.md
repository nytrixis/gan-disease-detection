# GAN-based Disease Detection: Dermatofibroma Classification

Complete deep learning pipeline for rare disease detection using WGAN-GP synthetic data augmentation.

## 📊 Project Overview

- **Dataset**: ISIC 2019 (Dermatofibroma: 239 samples, Nevus: 12,875 samples)
- **Imbalance Ratio**: ~54:1
- **Method**: WGAN-GP for synthetic data generation
- **Target**: F1-score improvement from 0.51 → 0.76+
- **Hardware**: Optimized for RTX 3050 (4GB VRAM)

## ✅ Prerequisites

- ✅ Dataset organized in `data/raw/dermatofibroma/` (239 images) and `data/raw/nevus/` (12,875 images)
- ✅ Conda environment `gan-disease` activated
- ✅ NVIDIA GPU with CUDA support

## 🚀 Execution Steps

### Step 1: Create Data Splits (2 minutes)

```bash
conda activate gan-disease
python preprocessing/data_loader.py
```

**Output**: Creates train/val/test splits in `data/splits/`
- Dermatofibroma: 167 train, 36 val, 36 test
- Nevus: 9,012 train, 1,932 val, 1,931 test

---

### Step 2: Train WGAN-GP (6-12 hours)

```bash
python training/train_gan.py
```

**What happens**:
- Trains generator to create realistic Dermatofibroma images
- Saves checkpoints every 2000 steps to `results/checkpoints/wgan_gp/`
- Saves sample images every 500 steps to `results/samples/` (monitor here!)
- Final model: `results/checkpoints/wgan_gp/final_model.pth`

**Time**: 6-12 hours on RTX 3050
**Can stop/resume**: Yes, from last checkpoint

---

### Step 3: Generate Synthetic Images (10 minutes)

```bash
python training/generate_synthetic.py
```

**Output**: 1500 synthetic Dermatofibroma images in `data/synthetic/raw/`

---

### Step 4: Train Classifiers (3-6 hours total)

Train 3 models for comparison:

```bash
# Model 1: Baseline (no augmentation) - 1-2 hours
python training/train_classifier.py --mode baseline --architecture resnet50

# Model 2: Traditional augmentation - 1-2 hours
python training/train_classifier.py --mode traditional_aug --architecture resnet50

# Model 3: GAN augmentation - 1-2 hours
python training/train_classifier.py --mode gan_aug --architecture resnet50
```

**Output**: Trained models in `results/checkpoints/classifier/`

---

### Step 5: Evaluate & Generate Report (5 minutes)

```bash
python evaluation/evaluate_all.py
```

**Output**:
- Performance comparison table
- Confusion matrices
- ROC curves
- Results in `results/tables/` and `results/figures/`

---

## 📦 Alternative: Run Complete Pipeline

```bash
python run_complete_pipeline.py
```

Runs all 5 steps automatically (takes 12-24 hours total).

---

## 📁 Project Structure

```
gan-disease-detection/
├── data/
│   ├── raw/                    # Original images (you have this)
│   ├── splits/                 # Created in Step 1
│   └── synthetic/              # Created in Step 3
├── models/
│   ├── generator.py           # ✅ Implemented
│   ├── discriminator.py       # ✅ Implemented
│   └── __init__.py
├── training/
│   ├── train_gan.py           # ✅ Implemented
│   ├── train_classifier.py    # ⚠️ Need to implement
│   └── generate_synthetic.py  # ✅ Implemented
├── preprocessing/
│   └── data_loader.py         # ✅ Implemented
├── evaluation/
│   └── evaluate_all.py        # ⚠️ Need to implement
├── results/
│   ├── checkpoints/           # Saved models
│   ├── samples/               # Generated images
│   ├── figures/               # Plots
│   └── tables/                # Metrics
└── run_complete_pipeline.py   # ✅ Implemented
```

---

## 🎯 Expected Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Baseline | 82.4% | 64.3% | 28.1% | **0.51** |
| Traditional Aug | 84.9% | 71.2% | 42.0% | **0.53** |
| **GAN Aug (Ours)** | **88.5%** | **78.3%** | **59.5%** | **0.76** |

**Improvement**: +49% F1-score vs baseline

---

## 💾 Disk Space Requirements

- Raw dataset: ~3GB
- After splits: ~6GB (copies data)
- Synthetic images: ~500MB
- Checkpoints: ~2GB
- **Total**: ~12GB

---

## ⚡ Performance Tips for RTX 3050

1. **Batch sizes are optimized**: GAN uses batch_size=8, Classifier uses 16
2. **Reduced epochs**: 200 epochs for GAN (vs 240 in paper)
3. **Gradient checkpointing**: Automatically enabled
4. **Close other GPU applications**: Chrome, games, etc.
5. **Monitor VRAM**: `nvidia-smi` in another terminal

---

## 🐛 Troubleshooting

**Out of memory error**:
```bash
# Reduce batch size in training/train_gan.py
HYPERPARAMETERS['batch_size'] = 4  # Line 23
```

**Training too slow**:
- Normal on RTX 3050: ~4-6 it/s for GAN
- Check GPU usage: `nvidia-smi`

**Missing modules**:
```bash
conda activate gan-disease
pip install -r requirements.txt
```

---

## 📝 Citation

```bibtex
@misc{gan-disease-detection-2024,
  title={GAN-based Data Augmentation for Rare Disease Detection},
  year={2024}
}
```

---

## 👤 Author

Implementation optimized for RTX 3050 GPUs based on ISIC 2019 dataset.
