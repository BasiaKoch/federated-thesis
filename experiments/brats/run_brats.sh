#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job: BraTS 2D Federated Learning (N clients)
#!  FedAvg / FedProx — pure PyTorch (no Flower)
#!
#!  Usage:
#!    sbatch run_brats.sh                             # defaults: both strategies
#!    STRATEGY=fedprox MU=1.0 sbatch run_brats.sh     # FedProx only
#!    STRATEGY=fedavg sbatch run_brats.sh              # FedAvg only
#!    PARTITION_DIR=path/to/client_data sbatch run_brats.sh
#! ==============================================================

#SBATCH -J brats_fed_N
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/bk489/federated/federated-thesis/experiments/brats/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated/federated-thesis/experiments/brats/logs/%x_%j.err
#SBATCH --qos=INTR

set -euo pipefail

# ======= Paths =======
PROJECT_DIR="$HOME/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/experiments/brats/brats_n_clients.py"
PARTITION_DIR="${PARTITION_DIR:-${PROJECT_DIR}/data/partitions/brats2d_4client_dirichlet_balanced_a1p0/client_data}"
RESULTS_DIR="${PROJECT_DIR}/results/brats"
LOG_DIR="${PROJECT_DIR}/experiments/brats/logs"

# ======= Hyperparams (override by exporting before sbatch) =======
STRATEGY="${STRATEGY:-both}"               # fedavg, fedprox, or both
ROUNDS="${ROUNDS:-30}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-0.01}"
FRACTION_FIT="${FRACTION_FIT:-0.75}"
MU="${MU:-1.0}"                            # only used if STRATEGY=fedprox or both
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
DROP_PERCENT="${DROP_PERCENT:-0.5}"        # fraction of stragglers
MODEL_BASE="${MODEL_BASE:-32}"             # UNet2D base filters
NUM_WORKERS="${NUM_WORKERS:-2}"
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

# ======= Create dirs / cd =======
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
cd "${PROJECT_DIR}"

# ======= Diagnostics =======
echo "=============================================="
echo "BraTS 2D Federated Learning"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node(s):      ${SLURM_NODELIST}"
echo "Workdir:      $(pwd)"
echo "Script:       ${SRC_FILE}"
echo "Partition dir:${PARTITION_DIR}"
echo "Results dir:  ${RESULTS_DIR}"
echo "Strategy:     ${STRATEGY} (mu=${MU})"
echo "Model:        UNet2D base=${MODEL_BASE}"
echo "Fraction:     ${FRACTION_FIT} | DropPercent: ${DROP_PERCENT}"
echo "Rounds:       ${ROUNDS} | LocalEpochs: ${LOCAL_EPOCHS}"
echo "Batch:        ${BATCH_SIZE} | LR: ${LR} | Seed: ${SEED}"
echo "WeightDecay:  ${WEIGHT_DECAY} | NumWorkers: ${NUM_WORKERS}"
echo "Python:       $(which python)"
python -c "import sys; print('Python', sys.version)"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available(), 'Device count:', torch.cuda.device_count())"
nvidia-smi || true
echo "=============================================="

# ======= Fail-fast checks =======
[ -f "${SRC_FILE}" ] || { echo "ERROR: missing script: ${SRC_FILE}"; exit 1; }
[ -d "${PARTITION_DIR}" ] || { echo "ERROR: missing partition dir: ${PARTITION_DIR}"; exit 1; }

# Validate strategy
if [[ "${STRATEGY}" != "fedavg" && "${STRATEGY}" != "fedprox" && "${STRATEGY}" != "both" ]]; then
    echo "ERROR: STRATEGY must be 'fedavg', 'fedprox', or 'both' (got '${STRATEGY}')"
    exit 1
fi

# ======= Common args =======
COMMON_ARGS=(
    --partition_dir "${PARTITION_DIR}"
    --rounds "${ROUNDS}"
    --local_epochs "${LOCAL_EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --drop_percent "${DROP_PERCENT}"
    --fraction_fit "${FRACTION_FIT}"
    --base "${MODEL_BASE}"
    --num_workers "${NUM_WORKERS}"
    --seed "${SEED}"
    --use_cuda
    --output_dir "${RESULTS_DIR}"
)

# ======= Run =======
run_strategy() {
    local strat=$1
    local mu_val=$2
    echo ""
    echo ">>> Running ${strat^^} (mu=${mu_val}) ..."
    echo ""
    python -u "${SRC_FILE}" \
        --strategy "${strat}" \
        --mu "${mu_val}" \
        "${COMMON_ARGS[@]}"
    echo ""
    echo ">>> ${strat^^} done."
}

if [[ "${STRATEGY}" == "fedavg" ]]; then
    run_strategy fedavg 0.0
elif [[ "${STRATEGY}" == "fedprox" ]]; then
    run_strategy fedprox "${MU}"
elif [[ "${STRATEGY}" == "both" ]]; then
    run_strategy fedavg 0.0
    run_strategy fedprox "${MU}"
fi

echo ""
echo "All done. Results in: ${RESULTS_DIR}"
