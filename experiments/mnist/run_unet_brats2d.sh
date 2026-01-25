#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job submission script
#!  Centralized 2D U-Net BraTS baseline (npz slices)
#! ==============================================================

#SBATCH -J brats_unet2d
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=06:00:00
#SBATCH --output=/home/bk489/federated/federated-thesis/unet/logs/brats_unet2d_%j.out
#SBATCH --error=/home/bk489/federated/federated-thesis/unet/logs/brats_unet2d_%j.err
#SBATCH --qos=INTR

set -euo pipefail

PROJECT_DIR="/home/bk489/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/unet/unet_new.py"
DATA_DIR="${PROJECT_DIR}/data/brats2020_top10_slices_split_npz"
LOG_DIR="${PROJECT_DIR}/unet/logs"

# Ensure log dir exists (also create it before sbatch ideally)
mkdir -p "${LOG_DIR}"

. /etc/profile.d/modules.sh
module load rhel8/default-amp
module load gcc/9

# If your conda env already provides torch+cuda, you usually don't need cuda/cudnn modules.
# module load cuda/12.1 cudnn

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null
conda activate fed

cd "${PROJECT_DIR}"

echo "=============================================="
echo "BraTS 2D U-Net (centralized) baseline"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "Workdir: $(pwd)"
echo "Script: ${SRC_FILE}"
echo "Data:   ${DATA_DIR}"
echo "Python: $(which python)"
python -c "import sys; print('Python', sys.version)"
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device count:', torch.cuda.device_count())"
nvidia-smi
echo "=============================================="

if [ ! -d "${DATA_DIR}" ]; then
  echo "ERROR: DATA_DIR not found: ${DATA_DIR}"
  exit 1
fi

echo "Num npz files:"
find "${DATA_DIR}" -name "*.npz" | wc -l
echo "Example files:"
find "${DATA_DIR}" -name "*.npz" | head -n 5

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export BRATS_DATA_DIR="${DATA_DIR}"

echo "Starting U-Net training..."
echo "BRATS_DATA_DIR=${BRATS_DATA_DIR}"
python -u "${SRC_FILE}"

echo "Job completed!"
