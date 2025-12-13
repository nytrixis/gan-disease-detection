"""
Verification script to check if all dependencies and data are ready.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 8:
        print("✓ Python version OK")
        return True
    else:
        print("✗ Python version should be 3.8 or higher")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'cv2',
        'sklearn',
        'matplotlib',
        'lpips',
        'pytorch_fid'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} not found")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠ CUDA not available. Training will be slow on CPU.")
            return False
    except:
        return False

def check_directory_structure():
    """Check if directory structure is correct"""
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/splits',
        'data/synthetic',
        'models',
        'training',
        'evaluation',
        'preprocessing',
        'utils',
        'notebooks',
        'results',
        'configs'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} not found")
            all_exist = False
    
    return all_exist

def check_dataset():
    """Check if dataset is downloaded"""
    data_path = Path('data/raw')
    
    if not data_path.exists():
        print("✗ Dataset directory not found")
        return False
    
    # Check for some expected files
    files = list(data_path.rglob('*.jpg')) + list(data_path.rglob('*.jpeg'))
    
    if len(files) > 0:
        print(f"✓ Found {len(files)} images in dataset")
        return True
    else:
        print("⚠ No images found in data/raw/")
        print("  Run: ./download_dataset.sh")
        return False

def main():
    print("=" * 60)
    print("  GAN Disease Detection - Setup Verification")
    print("=" * 60)
    print()
    
    results = []
    
    print("Checking Python version...")
    results.append(check_python_version())
    print()
    
    print("Checking dependencies...")
    results.append(check_dependencies())
    print()
    
    print("Checking CUDA...")
    results.append(check_cuda())
    print()
    
    print("Checking directory structure...")
    results.append(check_directory_structure())
    print()
    
    print("Checking dataset...")
    results.append(check_dataset())
    print()
    
    print("=" * 60)
    if all(results):
        print("✓ All checks passed! Ready to start development.")
    else:
        print("⚠ Some checks failed. Please resolve issues before proceeding.")
    print("=" * 60)

if __name__ == '__main__':
    main()
