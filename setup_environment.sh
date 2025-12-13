#!/bin/bash
# Environment Setup Script

echo "Setting up GAN Disease Detection environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda not found. Please install Miniconda/Anaconda."
    exit 1
fi

# Accept conda TOS
echo "Accepting Conda Terms of Service..."
conda config --set channel_priority flexible
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 2>/dev/null || true

# Create environment
echo "Creating conda environment..."
conda create -n gan-disease python=3.10 -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gan-disease

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "✓ Environment setup complete!"
echo "To activate the environment, run: conda activate gan-disease"
