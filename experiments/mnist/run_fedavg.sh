#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job submission script for Flower MNIST 2-Digit
#!  FedAvg (standard federated averaging, no proximal term)
#! ==============================================================

#SBATCH -J flower_fedavg
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=01:00:00
#SBATCH --output=flower_fedavg_%j.out
#SBATCH --error=flower_fedavg_%j.err
#SBATCH --qos=INTR

#! ======= Paths =======
PROJECT_DIR="$HOME/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/experiments/mnist/train.py"
CONFIG_FILE="${PROJECT_DIR}/experiments/mnist/configs/fedavg.yaml"
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
echo "Flower MNIST 2-Digit - FedAvg"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "Config: ${CONFIG_FILE}"
echo "Python: $(which python)"
python -c "import sys; print('Python', sys.version)"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
python -c "import flwr; print('Flower', flwr.__version__)"
nvidia-smi
echo "=============================================="

#! ======= Create results directory =======
mkdir -p "${RESULTS_DIR}"
cd "${PROJECT_DIR}"

echo "Starting Flower FedAvg run..."
python "${SRC_FILE}" --config "${CONFIG_FILE}"

echo "Job completed!"
