#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job submission script for MNIST 2-Digit Baselines
#!  Centralized + Local-Only (upper/lower bounds)
#! ==============================================================

#SBATCH -J baseline_mnist
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=01:00:00
#SBATCH --output=baseline_mnist_%j.out
#SBATCH --error=baseline_mnist_%j.err
#SBATCH --qos=INTR

#! ======= Configuration (match FL experiments for fair comparison) =======
BASELINE="both"          # centralized, local, or both
EPOCHS=30                # comparable to FL rounds
LEARNING_RATE=0.05       # match FL experiments
SEED=42

PROJECT_DIR="$HOME/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/src/baseline_mnist_2digits.py"
RESULTS_DIR="${PROJECT_DIR}/results/flower_mnist_2digits"

#! ======= Load required environment modules =======
. /etc/profile.d/modules.sh
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn openmpi/gcc/9.3/4.0.4

#! ======= Activate environment (DO NOT source ~/.bashrc) =======
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null
conda activate fed

#! ======= Diagnostics =======
echo "=============================================="
echo "MNIST 2-Digit Baseline Experiments"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "Python: $(which python)"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
nvidia-smi
echo "=============================================="
echo "Baseline: ${BASELINE}"
echo "Epochs: ${EPOCHS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Seed: ${SEED}"
echo "Results dir: ${RESULTS_DIR}"
echo "=============================================="

#! ======= Create results directory =======
mkdir -p "${RESULTS_DIR}"
cd "${PROJECT_DIR}"

echo "Starting baseline experiments..."
python "${SRC_FILE}" \
  --baseline "${BASELINE}" \
  --epochs "${EPOCHS}" \
  --lr "${LEARNING_RATE}" \
  --seed "${SEED}" \
  --use_cuda \
  --output_dir "${RESULTS_DIR}"

echo "✅ Baseline experiments completed!"
