#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job: Federated U-Net (2 clients, 70/30)
#!  Trains global model using FedAvg/FedProx, evaluates per-client
#! ==============================================================

#SBATCH -J brats_fed_70_30
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=/home/bk489/federated/federated-thesis/federated_unet/unet/logs/brats_fed_70_30_%j.out
#SBATCH --error=/home/bk489/federated/federated-thesis/federated_unet/unet/logs/brats_fed_70_30_%j.err
#SBATCH --qos=INTR

set -euo pipefail

# ======= Paths =======
PROJECT_DIR="$HOME/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/federated_unet/unet/unet_flower_train_70_30.py"

PARTITION_DIR="${PROJECT_DIR}/data/partitions/federated_clients_2_70_30/client_data"

OUT_BASE="${PROJECT_DIR}/results/federated/clients_2_70_30"

LOG_DIR="${PROJECT_DIR}/federated_unet/unet/logs"

# Hyperparams (override by exporting before sbatch if desired)
STRATEGY="${STRATEGY:-fedavg}"       # fedavg or fedprox
MU="${MU:-0.01}"                     # FedProx mu (ignored if fedavg)
ROUNDS="${ROUNDS:-10}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"

# ======= Modules =======
. /etc/profile.d/modules.sh
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn

# ======= Conda =======
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null
conda activate fed

# ======= Performance =======
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTHONUNBUFFERED=1

# ======= Create dirs =======
mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_BASE}"
cd "${PROJECT_DIR}"

# ======= Diagnostics =======
echo "=============================================="
echo "BraTS 2D U-Net FEDERATED (70/30) training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "Workdir: $(pwd)"
echo "Script:  ${SRC_FILE}"
echo "Parts:   ${PARTITION_DIR}"
echo "Out:     ${OUT_BASE}"
echo "Strategy: ${STRATEGY} | MU: ${MU}"
echo "Rounds: ${ROUNDS} | LocalEpochs: ${LOCAL_EPOCHS}"
echo "Batch: ${BATCH_SIZE} | LR: ${LR} | Seed: ${SEED}"
echo "Python:  $(which python)"
python -c "import sys; print('Python', sys.version)"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'CUDA:', torch.version.cuda); print('Device count:', torch.cuda.device_count())"
python -c "import flwr; print('Flower', flwr.__version__)"
nvidia-smi
echo "=============================================="

# ======= Fail-fast checks =======
if [ ! -f "${SRC_FILE}" ]; then
  echo "ERROR: SRC_FILE not found: ${SRC_FILE}"
  exit 1
fi
if [ ! -d "${PARTITION_DIR}" ]; then
  echo "ERROR: PARTITION_DIR not found: ${PARTITION_DIR}"
  exit 1
fi

echo "Client partition sanity check (train/val/test must exist):"
for i in 0 1; do
  for s in train val test; do
    cdir="${PARTITION_DIR}/client_${i}/${s}"
    if [ ! -d "${cdir}" ]; then
      echo "ERROR: missing ${cdir}"
      exit 1
    fi
  done
  ntr=$(find "${PARTITION_DIR}/client_${i}/train" -name "*.npz" | wc -l)
  nva=$(find "${PARTITION_DIR}/client_${i}/val"   -name "*.npz" | wc -l)
  nte=$(find "${PARTITION_DIR}/client_${i}/test"  -name "*.npz" | wc -l)
  echo "  client_${i}: train=${ntr} val=${nva} test=${nte} slices"
done

# ======= Run Federated Training =======
echo ""
echo "----------------------------------------------"
echo "Running Federated Training (${STRATEGY})"
echo "----------------------------------------------"
python -u "${SRC_FILE}" \
  --partition_dir "${PARTITION_DIR}" \
  --strategy "${STRATEGY}" \
  --mu "${MU}" \
  --rounds "${ROUNDS}" \
  --local_epochs "${LOCAL_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --use_cuda \
  --out_dir "${OUT_BASE}"

echo ""
echo "Job completed!"
echo "Results saved to: ${OUT_BASE}"
