#!/bin/bash
#! ==============================================================
#!  SBATCH job submission script for MNIST 2-Digit Baselines
#!  Centralized + Local-Only (upper/lower bounds)
#! ==============================================================

#SBATCH -J baseline_mnist                 # Job name
#SBATCH -A FERGUSSON-SL3-GPU              # Your GPU project account
#SBATCH -p ampere                         # Ampere GPU partition
#SBATCH --nodes=1                         # One node only
#SBATCH --ntasks=1                        # One task
#SBATCH --gres=gpu:1                      # One GPU (A100)
#SBATCH --cpus-per-task=3                 # <=3 CPUs per GPU per CSD3 rules
#SBATCH --time=01:00:00                   # 1 hour (adjust as needed)
#SBATCH --output=baseline_mnist_%j.out    # %j = job ID
#SBATCH --error=baseline_mnist_%j.err     # Error log
#SBATCH --qos=INTR

#! ======= Configuration (modify these) =======
BASELINE="${1:-both}"                     # centralized, local, or both
EPOCHS="${2:-30}"                         # Number of epochs
LEARNING_RATE="${3:-0.01}"                # Learning rate
SEED="${4:-42}"                           # Random seed

#! ======= Paths =======
PROJECT_DIR="$HOME/federated/federated-thesis"
RESULTS_DIR="${PROJECT_DIR}/results/flower_mnist_2digits"
SRC_DIR="${PROJECT_DIR}/src"

#! ======= Load required environment modules =======
if [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
    module purge 2>/dev/null || true
    module load rhel8/default-amp 2>/dev/null || true
fi

#! ======= Activate conda environment =======
source ~/.bashrc
conda activate fed || {
    echo "ERROR: Conda environment 'fed' not found"
    exit 1
}

#! ======= Fix libstdc++ if needed =======
if [ -f /usr/lib64/libstdc++.so.6 ]; then
    export LD_PRELOAD="/usr/lib64/libstdc++.so.6"
fi

#! ======= Diagnostics =======
echo "=============================================="
echo "MNIST 2-Digit Baseline Experiments"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Baseline: $BASELINE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Seed: $SEED"
echo "=============================================="
echo "Python: $(which python)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "=============================================="

#! ======= Create results directory =======
mkdir -p "$RESULTS_DIR"

#! ======= Run experiment =======
echo ""
echo "Starting baseline experiments at $(date)"
echo ""

cd "$PROJECT_DIR"

python "${SRC_DIR}/baseline_mnist_2digits.py" \
    --baseline "$BASELINE" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --seed "$SEED" \
    --use_cuda \
    --output_dir "${RESULTS_DIR}"

#! ======= Done =======
echo ""
echo "=============================================="
echo "Baseline Experiments Complete!"
echo "=============================================="
echo "End time: $(date)"
echo "Results saved in: ${RESULTS_DIR}"
echo ""
