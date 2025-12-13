# 🚀 QUICK START GUIDE

## Your Current Status

✅ Dataset ready: 239 Dermatofibroma + 12,875 Nevus images
✅ Environment: `gan-disease` conda environment
✅ Hardware: RTX 3050 (4GB VRAM)
✅ Code: Complete implementation ready

---

## Run the Project (Choose One)

### Option A: Automated Pipeline (Recommended)

```bash
conda activate gan-disease
python run_complete_pipeline.py
```

**Total Time**: 12-24 hours
- Creates splits (2 min)
- Trains GAN (6-12 hours)
- Generates synthetic images (10 min)
- Trains 3 classifiers (3-6 hours)
- Evaluates and reports (5 min)

---

### Option B: Step-by-Step (Manual Control)

#### 1️⃣ Create Data Splits (2 minutes)

```bash
conda activate gan-disease
python preprocessing/data_loader.py
```

**Verify**:
```bash
ls data/splits/train/
# Should see: dermatofibroma/ nevus/
```

---

#### 2️⃣ Train WGAN-GP (6-12 hours)

```bash
python training/train_gan.py
```

**Monitor Progress**:
- Open another terminal: `watch -n 5 ls -lt results/samples/`
- Check latest images in `results/samples/`

**Stop/Resume**: Press Ctrl+C to stop. Resumes from last checkpoint automatically.

---

#### 3️⃣ Generate Synthetic Images (10 minutes)

```bash
python training/generate_synthetic.py
```

**Verify**:
```bash
ls data/synthetic/raw/ | wc -l
# Should output: 1500
```

---

#### 4️⃣ Train Classifiers (3-6 hours total)

```bash
# Baseline (1-2 hours)
python training/train_classifier.py --mode baseline --architecture resnet50

# Traditional augmentation (1-2 hours)
python training/train_classifier.py --mode traditional_aug --architecture resnet50

# GAN augmentation (1-2 hours)
python training/train_classifier.py --mode gan_aug --architecture resnet50
```

---

#### 5️⃣ Evaluate Results (5 minutes)

```bash
python evaluation/evaluate_all.py
```

**View Results**:
```bash
cat results/tables/evaluation_results.json
```

---

## Expected Timeline

| Step | Task | Time (RTX 3050) |
|------|------|-----------------|
| 1 | Data splits | 2 min |
| 2 | Train GAN | 6-12 hours |
| 3 | Generate images | 10 min |
| 4 | Train classifiers (×3) | 3-6 hours |
| 5 | Evaluate | 5 min |
| **TOTAL** | **Complete pipeline** | **12-24 hours** |

---

## What You'll Get

### Trained Models
- `results/checkpoints/wgan_gp/final_model.pth` - GAN generator
- `results/checkpoints/classifier/baseline_resnet50_best.pth`
- `results/checkpoints/classifier/traditional_aug_resnet50_best.pth`
- `results/checkpoints/classifier/gan_aug_resnet50_best.pth`

### Synthetic Data
- `data/synthetic/raw/` - 1500 generated Dermatofibroma images

### Results
- `results/tables/evaluation_results.json` - All metrics
- `results/samples/` - GAN training progress images

---

## Monitoring Tips

### Check GPU Usage
```bash
nvidia-smi
# Should show ~90-100% GPU utilization during training
```

### Check Training Progress (GAN)
```bash
ls -lt results/samples/ | head -5
# Shows latest generated images
```

### Check Disk Space
```bash
df -h .
# Need ~12GB free
```

---

## Troubleshooting

### Out of Memory
Edit `training/train_gan.py` line 23:
```python
'batch_size': 4,  # Reduced from 8
```

### Training Too Slow
```bash
nvidia-smi
# Check if GPU is being used
# Close Chrome, Discord, other GPU apps
```

### Missing Dependencies
```bash
conda activate gan-disease
pip install torch torchvision tqdm scikit-learn pillow
```

---

## Next Steps After Completion

1. Check `results/tables/evaluation_results.json` for metrics
2. Compare F1-scores: baseline vs GAN-augmented
3. View generated images in `results/samples/`
4. (Optional) Train DenseNet or EfficientNet variants

---

## Ready to Start?

```bash
conda activate gan-disease
python run_complete_pipeline.py
```

**Go grab coffee ☕ - this will take 12-24 hours!**
