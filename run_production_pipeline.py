"""
Production-Ready Complete Pipeline Runner
Runs all steps with error handling and progress tracking
Optimized for RTX 3050 (4GB VRAM)
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import torch

def check_environment():
    """Check if environment is properly set up"""
    print("\n" + "="*70)
    print(" ENVIRONMENT CHECK")
    print("="*70)
    
    issues = []
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.2f} GB")
        if vram < 4:
            print("  ⚠️  Warning: <4GB VRAM, may need to reduce batch sizes")
    else:
        print("⚠️  CUDA not available - training will be VERY slow on CPU")
        issues.append("CUDA recommended for training")
    
    # Check directories
    required_dirs = ['data/raw/dermatofibroma', 'data/raw/nevus']
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            issues.append(f"Missing directory: {dir_path}")
        else:
            n_files = len(list(Path(dir_path).glob('*.jpg')))
            print(f"✓ {dir_path}: {n_files} images")
    
    print("="*70)
    
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("\n✅ Environment check passed!")
    return True

def run_pipeline():
    """Run complete pipeline with error handling"""
    
    print("\n" + "="*70)
    print(" GAN-BASED DISEASE DETECTION - PRODUCTION PIPELINE")
    print(" Optimized for RTX 3050 (4GB VRAM)")
    print("="*70)
    
    # Check environment
    if not check_environment():
        print("\n❌ Please fix environment issues before continuing")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Pipeline steps
    steps = [
        {
            'name': 'Data Preprocessing',
            'description': 'Create train/val/test splits (70/15/15)',
            'command': 'python main.py --mode preprocess',
            'time_estimate': '2-5 minutes',
            'skip_if_exists': 'data/splits/metadata.csv'
        },
        {
            'name': 'WGAN-GP Training',
            'description': 'Train generator to create synthetic dermatofibroma images',
            'command': 'python main.py --mode train_gan',
            'time_estimate': '6-12 hours on RTX 3050',
            'skip_if_exists': 'results/checkpoints/wgan_gp/final_model.pth'
        },
        {
            'name': 'Generate Synthetic Images',
            'description': 'Generate 1500 synthetic images from trained GAN',
            'command': 'python main.py --mode generate',
            'time_estimate': '5-10 minutes',
            'skip_if_exists': 'data/synthetic/raw'
        },
        {
            'name': 'Curate Synthetic Images',
            'description': '3-stage curation: blur, SSIM, LPIPS + diversity (1500→600)',
            'command': 'python main.py --mode curate',
            'time_estimate': '30-60 minutes',
            'skip_if_exists': 'data/synthetic/curated'
        },
        {
            'name': 'Train Baseline Classifier',
            'description': 'ResNet-50 without augmentation (baseline)',
            'command': 'python main.py --mode train_classifier --classifier-mode baseline',
            'time_estimate': '1-2 hours',
            'skip_if_exists': 'results/checkpoints/classifier/baseline_resnet50_best.pth'
        },
        {
            'name': 'Train Traditional Aug Classifier',
            'description': 'ResNet-50 with traditional augmentation',
            'command': 'python main.py --mode train_classifier --classifier-mode traditional_aug',
            'time_estimate': '1-2 hours',
            'skip_if_exists': 'results/checkpoints/classifier/traditional_aug_resnet50_best.pth'
        },
        {
            'name': 'Train GAN-Augmented Classifier',
            'description': 'ResNet-50 with GAN-augmented data (BEST)',
            'command': 'python main.py --mode train_classifier --classifier-mode gan_aug',
            'time_estimate': '1-2 hours',
            'skip_if_exists': 'results/checkpoints/classifier/gan_aug_resnet50_best.pth'
        },
        {
            'name': 'Evaluate All Models',
            'description': 'Compare baseline vs traditional vs GAN augmentation',
            'command': 'python main.py --mode evaluate',
            'time_estimate': '5-10 minutes',
            'skip_if_exists': 'results/tables/evaluation_results.json'
        },
        {
            'name': 'Calculate GAN Metrics',
            'description': 'Compute FID, IS, LPIPS, SSIM scores',
            'command': 'python main.py --mode metrics',
            'time_estimate': '10-20 minutes',
            'skip_if_exists': 'results/tables/gan_metrics.json'
        }
    ]
    
    # Track progress
    start_time = datetime.now()
    completed_steps = []
    failed_steps = []
    skipped_steps = []
    
    print(f"\nPipeline has {len(steps)} steps")
    print(f"Total estimated time: 10-20 hours (mostly GAN training)\n")
    
    for i, step in enumerate(steps, 1):
        print("\n" + "="*70)
        print(f" STEP {i}/{len(steps)}: {step['name']}")
        print("="*70)
        print(f"Description: {step['description']}")
        print(f"Estimated time: {step['time_estimate']}")
        
        # Check if can skip
        if step.get('skip_if_exists') and Path(step['skip_if_exists']).exists():
            print(f"\n⏭️  Skipping: {step['skip_if_exists']} already exists")
            response = input("Skip this step? (y/n/q to quit): ")
            if response.lower() == 'q':
                print("\n🛑 Pipeline stopped by user")
                break
            elif response.lower() == 'y':
                skipped_steps.append(step['name'])
                continue
        
        # Confirm before running
        print(f"\nCommand: {step['command']}")
        response = input("\nRun this step? (y/n/q to quit): ")
        
        if response.lower() == 'q':
            print("\n🛑 Pipeline stopped by user")
            break
        elif response.lower() != 'y':
            skipped_steps.append(step['name'])
            print(f"⏭️  Skipped: {step['name']}")
            continue
        
        # Run step
        print(f"\n▶️  Running: {step['name']}...")
        print("="*70)
        
        try:
            result = os.system(step['command'])
            if result == 0:
                completed_steps.append(step['name'])
                print(f"\n✅ Completed: {step['name']}")
            else:
                failed_steps.append(step['name'])
                print(f"\n❌ Failed: {step['name']} (exit code: {result})")
                response = input("\nContinue with next step? (y/n): ")
                if response.lower() != 'y':
                    break
        except Exception as e:
            failed_steps.append(step['name'])
            print(f"\n❌ Error in {step['name']}: {e}")
            response = input("\nContinue with next step? (y/n): ")
            if response.lower() != 'y':
                break
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print(" PIPELINE SUMMARY")
    print("="*70)
    print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print()
    print(f"✅ Completed: {len(completed_steps)}/{len(steps)}")
    print(f"⏭️  Skipped:   {len(skipped_steps)}/{len(steps)}")
    print(f"❌ Failed:    {len(failed_steps)}/{len(steps)}")
    
    if completed_steps:
        print("\nCompleted steps:")
        for step in completed_steps:
            print(f"  ✓ {step}")
    
    if skipped_steps:
        print("\nSkipped steps:")
        for step in skipped_steps:
            print(f"  ⏭ {step}")
    
    if failed_steps:
        print("\nFailed steps:")
        for step in failed_steps:
            print(f"  ✗ {step}")
    
    print("\n" + "="*70)
    
    # Save summary
    summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'completed': completed_steps,
        'skipped': skipped_steps,
        'failed': failed_steps
    }
    
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    with open('results/tables/pipeline_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n✓ Summary saved to results/tables/pipeline_summary.json")
    
    if len(failed_steps) == 0 and len(completed_steps) > 0:
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("\nResults available in:")
        print("  - results/checkpoints/  (trained models)")
        print("  - results/figures/      (plots)")
        print("  - results/tables/       (metrics)")
        print("\nTo view results:")
        print("  tensorboard --logdir=logs")
        print("  python -m http.server 8000  # View figures in browser")
    else:
        print("\n⚠️  Pipeline completed with some issues")
        print("Check the summary above for details")

if __name__ == '__main__':
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\n🛑 Pipeline interrupted by user (Ctrl+C)")
        print("Progress has been saved. You can resume later.")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
