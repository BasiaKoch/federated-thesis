#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job submission script
#!  Federated BraTS 2D U-Net (Flower) - FedAvg / FedProx via YAML config
#! ==============================================================

#SBATCH -J brats_fl_unet
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=/home/bk489/federated/federated-thesis/federated_unet/unet/logs/brats_fl_unet_%j.out
#SBATCH --error=/home/bk489/federated/federated-thesis/federated_unet/unet/logs/brats_fl_unet_%j.err
#SBATCH --qos=INTR

set -euo pipefail

#! ======= Paths =======
PROJECT_DIR="$HOME/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/federated_unet/unet/unet_flower_train_10clients.py"

# Pick config: you can pass it as arg1, otherwise default to FedAvg config
CONFIG_FILE="${1:-${PROJECT_DIR}/federated_unet/unet/configs/unet_fedavg.yaml}"

# Data paths (used for quick checks + env)
DATA_DIR="${PROJECT_DIR}/data/brats2020_top10_slices_split_npz"
PARTITIONS_DIR="${PROJECT_DIR}/data/partitions/federated_clients_10_mixed/client_data"

LOG_DIR="${PROJECT_DIR}/federated_unet/unet/logs"
RESULTS_DIR="${PROJECT_DIR}/results/unet_flower"

#! ======= Load required environment modules =======
. /etc/profile.d/modules.sh
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn

#! ======= Activate environment =======
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null
conda activate fed

#! ======= Performance knobs =======
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTHONUNBUFFERED=1

#! ======= Create output dirs =======
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

cd "${PROJECT_DIR}"

#! ======= Diagnostics =======
echo "=============================================="
echo "BraTS 2D U-Net (Flower Federated)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "Workdir: $(pwd)"
echo "Script:  ${SRC_FILE}"
echo "Config:  ${CONFIG_FILE}"
echo "Data:    ${DATA_DIR}"
echo "Parts:   ${PARTITIONS_DIR}"
echo "Python:  $(which python)"
python -c "import sys; print('Python', sys.version)"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'CUDA:', torch.version.cuda); print('Device count:', torch.cuda.device_count())"
python -c "import flwr; print('Flower', flwr.__version__)"
nvidia-smi
echo "=============================================="

#! ======= Fail-fast checks =======
if [ ! -f "${SRC_FILE}" ]; then
  echo "ERROR: SRC_FILE not found: ${SRC_FILE}"
  exit 1
fi
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "ERROR: CONFIG_FILE not found: ${CONFIG_FILE}"
  exit 1
fi
if [ ! -d "${DATA_DIR}" ]; then
  echo "ERROR: DATA_DIR not found: ${DATA_DIR}"
  exit 1
fi
if [ ! -d "${PARTITIONS_DIR}" ]; then
  echo "ERROR: PARTITIONS_DIR not found: ${PARTITIONS_DIR}"
  exit 1
fi

echo "Num global npz files:"
find "${DATA_DIR}" -name "*.npz" | wc -l

echo "Example global npz files:"
find "${DATA_DIR}" -name "*.npz" | head -n 5 || true

echo "Client partition sanity check:"
for i in 0 1 2 3 4 5 6 7 8 9; do
  cdir="${PARTITIONS_DIR}/client_${i}/train"
  if [ ! -d "${cdir}" ]; then
    echo "ERROR: missing ${cdir}"
    exit 1
  fi
  n=$(find "${cdir}" -name "*.npz" | wc -l)
  echo "  client_${i}: ${n} slices"
done

#! ======= Run =======
# (Your python reads paths from the YAML, but we also export env for consistency)
export BRATS_DATA_DIR="${DATA_DIR}"

echo "Starting federated U-Net training..."
python -u "${SRC_FILE}" --config "${CONFIG_FILE}"

echo "Job completed!"
