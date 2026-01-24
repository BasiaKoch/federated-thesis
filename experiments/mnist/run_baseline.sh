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

#! ======= Paths =======
PROJECT_DIR="$HOME/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/experiments/mnist/baseline.py"
CONFIG_FILE="${PROJECT_DIR}/experiments/mnist/configs/baseline.yaml"
RESULTS_DIR="${PROJECT_DIR}/results/mnist"

#! ======= Load required environment modules =======
. /etc/profile.d/modules.sh
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn openmpi/gcc/9.3/4.0.4

#! ======= Activate environment =======
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null
conda activate fed

#! ======= Diagnostics =======
echo "=============================================="
echo "MNIST 2-Digit Baseline Experiments"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "Config: ${CONFIG_FILE}"
echo "Python: $(which python)"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
nvidia-smi
echo "=============================================="

#! ======= Create results directory =======
mkdir -p "${RESULTS_DIR}"
cd "${PROJECT_DIR}"

echo "Starting baseline experiments..."
python "${SRC_FILE}" --config "${CONFIG_FILE}"

echo "Baseline experiments completed!"
