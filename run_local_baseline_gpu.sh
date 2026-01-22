#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job submission script
#!  Local MNIST baseline (single-node PyTorch)
#! ==============================================================

#SBATCH -J mnist_local_baseline
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=00:30:00
#SBATCH --output=local_baseline_%j.out
#SBATCH --error=local_baseline_%j.err
#SBATCH --qos=INTR

#! ======= Paths =======
PROJECT_DIR="$HOME/federated/federated-thesis"
DATA_DIR="${PROJECT_DIR}/data"
OUT_DIR="${PROJECT_DIR}/results/local_baseline"

#! ======= Environment =======
. /etc/profile.d/modules.sh

# Same stack that WORKED before
module load rhel8/default-amp
module load intel-mkl-2017.4-gcc-5.4.0-2tzpyn7
module load gcc/9
module load cuda/12.1
module load cudnn
module load openmpi/gcc/9.3/4.0.4

# Activate your venv
source "${PROJECT_DIR}/fed/bin/activate"

# Sanity check (keep this!)
python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# Run baseline
python src/local_baseline.py \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUT_DIR}" \
  --epochs 50
