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
                              'curate', 'train_classifier', 'evaluate', 'metrics'],
                      help='Execution mode')
    parser.add_argument('--classifier-mode', type=str, default='baseline',
                      choices=['baseline', 'traditional_aug', 'gan_aug'],
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
        Path('results/tables').mkdir(parents=True, exist_ok=True)
        with open('results/tables/gan_metrics.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("✓ Results saved to results/tables/gan_metrics.json")
    
    print(f"\n✓ {args.mode} completed successfully!")

if __name__ == '__main__':
    main()
