"""
Main execution script for GAN-based Disease Detection project.
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='GAN-based Disease Detection')
    parser.add_argument('--mode', type=str, required=True,
                      choices=['preprocess', 'train_gan', 'generate', 
                              'curate', 'train_classifier', 'evaluate', 'metrics',
                              'train_acgan', 'generate_acgan', 'evaluate_acgan', 'compare_acgan',
                              'train_wgan_revised', 'generate_wgan_revised', 'evaluate_wgan_revised'],
                      help='Execution mode')
    parser.add_argument('--classifier-mode', type=str, default='baseline',
                      choices=['baseline', 'traditional_aug', 'gan_aug', 'acgan_aug'],
                      help='Classifier training mode')
    parser.add_argument('--architecture', type=str, default='resnet50',
                      choices=['resnet50', 'densenet121', 'efficientnet_b0'],
                      help='Classifier architecture')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs for classifier training')
    
    args = parser.parse_args()
    
    if args.mode == 'preprocess':
        print("Running data preprocessing...")
        from preprocessing.data_loader import create_splits
        create_splits()
        
    elif args.mode == 'train_gan':
        print("Training WGAN-GP...")
        from training.train_gan import train_wgan_gp
        train_wgan_gp()
        
    elif args.mode == 'generate':
        print("Generating synthetic images...")
        from training.generate_synthetic import generate_synthetic_images
        generate_synthetic_images(
            checkpoint_path='results/checkpoints/wgan_gp/final_model.pth',
            num_images=1500,
            output_dir='data/synthetic/raw'
        )
        
    elif args.mode == 'curate':
        print("Curating synthetic images...")
        from preprocessing.curation import automated_curation, diversity_analysis
        
        # Stage 1-3: Automated filtering
        filtered_images = automated_curation(
            synthetic_images_dir='data/synthetic/raw',
            real_images_dir='data/splits/train/dermatofibroma',
            output_dir='data/synthetic/filtered'
        )
        
        # Stage 4: Diversity analysis
        if len(filtered_images) > 0:
            diversity_analysis(
                filtered_images_dir='data/synthetic/filtered',
                output_dir='data/synthetic/curated',
                n_clusters=8,
                target_total=600
            )
        
    elif args.mode == 'train_classifier':
        print(f"Training CNN classifier ({args.classifier_mode} mode)...")
        from training.train_classifier import train_classifier
        train_classifier(
            mode=args.classifier_mode,
            architecture=args.architecture,
            epochs=args.epochs
        )
        
    elif args.mode == 'evaluate':
        print("Evaluating models...")
        from evaluation.evaluate_all import main as evaluate_main
        evaluate_main()
    
    elif args.mode == 'metrics':
        print("Calculating GAN quality metrics...")
        from evaluation.metrics import evaluate_gan_quality
        results = evaluate_gan_quality(
            real_images_path='data/splits/train/dermatofibroma',
            synthetic_images_path='data/synthetic/curated'
        )
        
        import json
        import numpy as np
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        results = convert_to_native(results)
        
        Path('results/tables').mkdir(parents=True, exist_ok=True)
        with open('results/tables/gan_metrics.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("✓ Results saved to results/tables/gan_metrics.json")
    
    elif args.mode == 'train_acgan':
        print("Training AC-GAN...")
        from training.train_acgan import train_acgan
        train_acgan(config_path='configs/acgan_config.yaml')
    
    elif args.mode == 'generate_acgan':
        print("Generating AC-GAN synthetic images...")
        from training.generate_acgan_synthetic import generate_synthetic_images
        generated, filtered = generate_synthetic_images(
            checkpoint_path='results/checkpoints/acgan/acgan_final.pth',
            config_path='configs/acgan_config.yaml',
            output_dir='data/synthetic_acgan/raw',
            num_images=6000,
            target_class=0,
            apply_filtering=True,
            laplacian_threshold=100
        )
        print(f"✓ Generated {generated} images, filtered to {filtered} high-quality images")
    
    elif args.mode == 'evaluate_acgan':
        print("Calculating AC-GAN FID score...")
        from evaluation.calculate_fid import calculate_fid
        fid = calculate_fid(
            real_images_path='data/raw/dermatofibroma',
            generated_images_path='data/synthetic_acgan/filtered',
            batch_size=32
        )
        print(f"✓ FID Score: {fid:.2f}")
        
        # Save results
        Path('results/tables').mkdir(parents=True, exist_ok=True)
        with open('results/tables/acgan_fid.json', 'w') as f:
            json.dump({'fid_score': float(fid)}, f, indent=4)
    
    elif args.mode == 'compare_acgan':
        print("Comparing classifiers with AC-GAN augmentation...")
        from training.train_acgan_classifiers import compare_classifiers
        results = compare_classifiers(
            real_train_dir='data/splits/train',
            val_dir='data/splits/val',
            test_dir='data/splits/test',
            acgan_synthetic_dir='data/synthetic_acgan/filtered',
            epochs=30
        )
        print("✓ Classifier comparison completed")
    
    elif args.mode == 'train_wgan_revised':
        print("Training WGAN-GP Revised (SOTA Architecture)...")
        from training.train_wgan_gp_revised import train_wgan_gp_revised
        train_wgan_gp_revised(config_path='configs/wgan_gp_revised_config.yaml')
    
    elif args.mode == 'generate_wgan_revised':
        print("Generating synthetic images with WGAN-GP Revised...")
        from training.generate_wgan_gp_revised_synthetic import generate_synthetic_images
        generate_synthetic_images(
            checkpoint_path='results/checkpoints/wgan_gp_revised/best_model.pth',
            num_images=1500,
            output_dir='data/synthetic/wgan_gp_revised/raw'
        )
    
    elif args.mode == 'evaluate_wgan_revised':
        print("Evaluating WGAN-GP Revised...")
        from evaluation.calculate_fid import calculate_fid
        fid = calculate_fid(
            real_images_path='data/raw/dermatofibroma',
            generated_images_path='data/synthetic/wgan_gp_revised/raw',
            batch_size=32
        )
        print(f"✓ FID Score: {fid:.2f}")
        
        # Save results
        import json
        Path('results/tables').mkdir(parents=True, exist_ok=True)
        with open('results/tables/wgan_revised_fid.json', 'w') as f:
            json.dump({'fid_score': float(fid)}, f, indent=4)
    
    print(f"\n✓ {args.mode} completed successfully!")

if __name__ == '__main__':
    main()
