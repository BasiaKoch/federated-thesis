#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job: Local (non-federated) BraTS 2D U-Net
#!  Runs train_local_70_30.py for one client_id (0 or 1)
#! ==============================================================

#SBATCH -J brats_local_unet
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=/home/bk489/federated/federated-thesis/federated_unet/unet/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated/federated-thesis/federated_unet/unet/logs/%x_%j.err
#SBATCH --qos=INTR

set -euo pipefail

# -------- Paths --------
PROJECT_DIR="$HOME/federated/federated-thesis"
SCRIPT="${PROJECT_DIR}/federated_unet/unet/train_local_70_30.py"
PARTITION_DIR="${PROJECT_DIR}/data/partitions/brats2d_one_slice_per_patient_clients"
OUT_BASE="${PROJECT_DIR}/results/local_unet"
LOG_DIR="${PROJECT_DIR}/federated_unet/unet/logs"

# -------- Hyperparams (override via sbatch --export=...) --------
CLIENT_ID="${CLIENT_ID:-0}"     # 0 or 1
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"

# -------- Modules --------
. /etc/profile.d/modules.sh
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn

# -------- Conda --------
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null
conda activate fed

# -------- Perf --------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTHONUNBUFFERED=1

mkdir -p "${LOG_DIR}" "${OUT_BASE}"
cd "${PROJECT_DIR}"

echo "=============================================="
echo "Local BraTS 2D U-Net (non-federated)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node(s):   ${SLURM_NODELIST}"
echo "Workdir:   $(pwd)"
echo "Script:    ${SCRIPT}"
echo "Partitions:${PARTITION_DIR}"
echo "Out base:  ${OUT_BASE}"
echo "Client ID: ${CLIENT_ID}"
echo "Epochs:    ${EPOCHS}"
echo "Batch:     ${BATCH_SIZE} | LR: ${LR} | Seed: ${SEED}"
echo "Python:    $(which python)"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
nvidia-smi || true
echo "=============================================="

# -------- Fail-fast checks --------
[ -f "${SCRIPT}" ] || { echo "ERROR: missing SCRIPT: ${SCRIPT}"; exit 1; }
[ -d "${PARTITION_DIR}" ] || { echo "ERROR: missing PARTITION_DIR: ${PARTITION_DIR}"; exit 1; }
if [[ "${CLIENT_ID}" != "0" && "${CLIENT_ID}" != "1" ]]; then
  echo "ERROR: CLIENT_ID must be 0 or 1 (got ${CLIENT_ID})"
  exit 1
fi
for s in train val test; do
  [ -d "${PARTITION_DIR}/client_${CLIENT_ID}/${s}" ] || { echo "ERROR: missing split dir: ${PARTITION_DIR}/client_${CLIENT_ID}/${s}"; exit 1; }
done

echo "Sanity counts:"
for s in train val test; do
  n=$(find "${PARTITION_DIR}/client_${CLIENT_ID}/${s}" -name "*.npz" | wc -l)
  echo "  client_${CLIENT_ID} ${s}: ${n}"
done

echo "Starting local training..."
python -u "${SCRIPT}" \
  --partition_dir "${PARTITION_DIR}" \
  --client_id "${CLIENT_ID}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --use_cuda \
  --out_dir "${OUT_BASE}"

echo "Done. Results in: ${OUT_BASE}"
