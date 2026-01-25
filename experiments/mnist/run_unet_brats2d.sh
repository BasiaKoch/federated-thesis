#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job submission script for BraTS2020 2D U-Net
#!  Train + Validate + Test on split PNG dataset
#! ==============================================================

#SBATCH -J unet_brats2d
#SBATCH -A fergusson-sl3-gpu
#SBATCH -p ampere
#SBATCH --qos=gpu2
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=unet_brats2d_%j.out
#SBATCH --error=unet_brats2d_%j.err


#! ======= Paths =======
PROJECT_DIR="$HOME/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/unet/train_unet_brats2d.py"

TARBALL="$HOME/BraTS2020_2D_png_split.tar.gz"
DATA_ROOT="$HOME/data/BraTS2020_2D_png_split"

CKPT_DIR="${PROJECT_DIR}/results/brats2d_unet"
CKPT_FILE="${CKPT_DIR}/unet_brats2d_best.pt"

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
echo "BraTS2020 2D U-Net"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "Tarball: ${TARBALL}"
echo "Data:    ${DATA_ROOT}"
echo "Python:  $(which python)"
python -c "import sys; print('Python', sys.version)"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
nvidia-smi
echo "=============================================="

#! ======= Prepare data (extract once) =======
mkdir -p "$HOME/data"
if [ ! -d "${DATA_ROOT}/train" ]; then
  echo "Extracting dataset to ${DATA_ROOT} ..."
  tar -xzf "${TARBALL}" -C "$HOME/data"
else
  echo "Dataset already extracted."
fi

#! ======= Create results directory =======
mkdir -p "${CKPT_DIR}"
cd "${PROJECT_DIR}"

echo "Starting BraTS 2D U-Net training..."
python "${SRC_FILE}" \
  --split_root "${DATA_ROOT}" \
  --device cuda \
  --amp \
  --image_size 240 \
  --epochs 30 \
  --batch_size 8 \
  --lr 3e-4 \
  --base 32 \
  --num_workers 4 \
  --min_fg_train 50 \
  --keep_empty_prob 0.05 \
  --ckpt "${CKPT_FILE}"

echo "Job completed!"
