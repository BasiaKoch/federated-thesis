#!/bin/bash
#! ==============================================================
#!  HPC Environment Setup Script for Federated Learning
#!  Run this ONCE to set up your environment on the HPC
#! ==============================================================

echo "=============================================="
echo "Setting up Federated Learning Environment"
echo "=============================================="

#! ======= Configuration =======
PROJECT_DIR="$HOME/federated-thesis"
VENV_DIR="${PROJECT_DIR}/venv"

#! ======= Load modules =======
. /etc/profile.d/modules.sh
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn
module load python/3.11

echo "Loaded modules:"
module list

#! ======= Create project directory =======
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Creating project directory: $PROJECT_DIR"
    mkdir -p "$PROJECT_DIR"
fi

#! ======= Create virtual environment =======
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

#! ======= Activate and install packages =======
source "${VENV_DIR}/bin/activate"

echo "Installing packages..."
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install NVFlare and other dependencies
pip install nvflare>=2.4.0 numpy matplotlib

#! ======= Verify installation =======
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import nvflare; print('NVFlare:', nvflare.__version__)"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Upload your project files to: $PROJECT_DIR"
echo "   scp -r ./* <username>@login.hpc.cam.ac.uk:${PROJECT_DIR}/"
echo ""
echo "2. Make sure mnist.npz is in: ${PROJECT_DIR}/data/"
echo ""
echo "3. Submit job with:"
echo "   sbatch run_hpc.sh fedprox 50"
echo ""
