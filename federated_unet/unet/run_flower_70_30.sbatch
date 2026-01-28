#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job: Federated U-Net (N clients)
#!  FedAvg / FedProx (set STRATEGY=fedavg|fedprox; MU used only for fedprox)
#!  Supports any number of clients via NUM_CLIENTS parameter
#! ==============================================================

#SBATCH -J brats_fed_Nclient
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/home/bk489/federated/federated-thesis/federated_unet/unet/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated/federated-thesis/federated_unet/unet/logs/%x_%j.err
#SBATCH --qos=INTR

set -euo pipefail

# ======= Paths =======
PROJECT_DIR="$HOME/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/federated_unet/unet/unet_flower_any_Nclients.py"

# Choose partition (uncomment the one you want):
# 2-client ET-skewed partition:
# PARTITION_DIR="${PROJECT_DIR}/data/partitions/brats2d_7030_et_skewed/client_data"
# 4-client quartile partition:
PARTITION_DIR="${PROJECT_DIR}/data/partitions/brats2d_4client_quartile/client_data"

OUT_BASE="${PROJECT_DIR}/results/federated/brats2d_4client_quartile"
LOG_DIR="${PROJECT_DIR}/federated_unet/unet/logs"

# ======= Hyperparams (override by exporting before sbatch) =======
NUM_CLIENTS="${NUM_CLIENTS:-4}"   # number of clients
STRATEGY="${STRATEGY:-fedavg}"    # fedavg or fedprox
MU="${MU:-0.01}"                  # only used if STRATEGY=fedprox
ROUNDS="${ROUNDS:-30}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"     # smaller batch for more clients (less data per client)
LR="${LR:-0.01}"
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

# ======= Create dirs / cd =======
mkdir -p "${LOG_DIR}" "${OUT_BASE}"
cd "${PROJECT_DIR}"

# ======= Diagnostics =======
echo "=============================================="
echo "BraTS 2D U-Net FEDERATED (${NUM_CLIENTS} clients)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node(s):   ${SLURM_NODELIST}"
echo "Workdir:   $(pwd)"
echo "Script:    ${SRC_FILE}"
echo "Partitions:${PARTITION_DIR}"
echo "Out base:  ${OUT_BASE}"
echo "Clients:   ${NUM_CLIENTS}"
echo "Strategy:  ${STRATEGY} (mu=${MU})"
echo "Rounds:    ${ROUNDS} | LocalEpochs: ${LOCAL_EPOCHS}"
echo "Batch:     ${BATCH_SIZE} | LR: ${LR} | Seed: ${SEED}"
echo "Python:    $(which python)"
python -c "import sys; print('Python', sys.version)"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available(), 'Device count:', torch.cuda.device_count())"
python -c "import flwr; print('Flower', flwr.__version__)"
nvidia-smi || true
echo "=============================================="

# ======= Fail-fast checks =======
[ -f "${SRC_FILE}" ] || { echo "ERROR: missing SRC_FILE: ${SRC_FILE}"; exit 1; }
[ -d "${PARTITION_DIR}" ] || { echo "ERROR: missing PARTITION_DIR: ${PARTITION_DIR}"; exit 1; }

# Validate strategy
if [[ "${STRATEGY}" != "fedavg" && "${STRATEGY}" != "fedprox" ]]; then
  echo "ERROR: STRATEGY must be 'fedavg' or 'fedprox' (got '${STRATEGY}')"
  exit 1
fi

echo "Client partition sanity check:"
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
  for s in train val test; do
    cdir="${PARTITION_DIR}/client_${i}/${s}"
    [ -d "${cdir}" ] || { echo "ERROR: missing ${cdir}"; exit 1; }
  done
  ntr=$(find "${PARTITION_DIR}/client_${i}/train" -name "*.npz" | wc -l)
  nva=$(find "${PARTITION_DIR}/client_${i}/val"   -name "*.npz" | wc -l)
  nte=$(find "${PARTITION_DIR}/client_${i}/test"  -name "*.npz" | wc -l)
  echo "  client_${i}: train=${ntr} val=${nva} test=${nte}"
done

# ======= Run =======
echo "Starting federated training with ${NUM_CLIENTS} clients..."
python -u "${SRC_FILE}" \
  --partition_dir "${PARTITION_DIR}" \
  --num_clients "${NUM_CLIENTS}" \
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

echo "Done. Results in: ${OUT_BASE}"
