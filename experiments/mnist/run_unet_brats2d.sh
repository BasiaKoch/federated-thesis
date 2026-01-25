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
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=brats_unet2d_%j.out
#SBATCH --error=brats_unet2d_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/home/bk489/federated/federated-thesis/unet/logs/brats_unet2d_%j.out
#SBATCH --error=/home/bk489/federated/federated-thesis/unet/logs/brats_unet2d_%j.err
#SBATCH --qos=INTR

#! ======= Paths =======
PROJECT_DIR="$HOME/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/unet/unet_new.py"
DATA_DIR="${PROJECT_DIR}/data/brats2020_top10_slices_split_npz"
RESULTS_DIR="${PROJECT_DIR}/unet/runs_unet_brats2d"
LOG_DIR="${PROJECT_DIR}/unet/logs"

#! ======= Load required environment modules =======. /etc/profile.d/modules.sh
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn

#! ======= Activate environment =======
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null
conda activate fed

#! ======= Diagnostics =======
echo "=============================================="
echo "BraTS 2D U-Net (centralized) baseline"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "Script: ${SRC_FILE}"
echo "Data:   ${DATA_DIR}"
echo "Python: $(which python)"
python -c "import sys; print('Python', sys.version)"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'CUDA:', torch.version.cuda)"
nvidia-smi
echo "=============================================="

#! ======= Create output dirs =======
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

#! ======= Quick data check =======
if [ ! -d "${DATA_DIR}" ]; then
  echo "ERROR: DATA_DIR not found: ${DATA_DIR}"
  exit 1
fi

echo "Num npz files:"
find "${DATA_DIR}" -name "*.npz" | wc -l

echo "Example files:"
# Avoid premature exit due to SIGPIPE issues by capturing then printing
find "${DATA_DIR}" -name "*.npz" | head -n 5 || true

<<<<<<< HEAD
#! ======= Run (unbuffered so epoch/Dice prints appear immediately) =======
=======
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTHONUNBUFFERED=1
>>>>>>> 901f1ef6c21ee475e67743b7bc3773bf33c499f8
export BRATS_DATA_DIR="${DATA_DIR}"
export PYTHONUNBUFFERED=1

echo "Starting BraTS U-Net training..."
python -u "${SRC_FILE}"

echo "Job completed!"
