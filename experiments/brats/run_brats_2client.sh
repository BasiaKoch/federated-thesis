#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job: BraTS 2D Federated Learning (2 clients)
#!  FedAvg / FedProx — equal data, corruption-only heterogeneity
#!
#!  Client 0: clean         (good hospital)
#!  Client 1: extreme noise (degraded scanner — C7-level corruption)
#!
#!  Usage:
#!    sbatch run_brats_2client.sh                            # FedAvg + FedProx mu=0.3
#!    STRATEGY=fedavg sbatch run_brats_2client.sh            # FedAvg only
#!    STRATEGY=fedprox MU=0.3 sbatch run_brats_2client.sh   # FedProx only
#!
#!  Mu sweep (submit as separate jobs):
#!    STRATEGY=fedprox MU=0.1 sbatch run_brats_2client.sh
#!    STRATEGY=fedprox MU=0.3 sbatch run_brats_2client.sh
#!    STRATEGY=fedprox MU=0.5 sbatch run_brats_2client.sh
#! ==============================================================

#SBATCH -J brats_2c
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/bk489/federated/federated-thesis/experiments/brats/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated/federated-thesis/experiments/brats/logs/%x_%j.err

# ======= Paths =======
PROJECT_DIR="$HOME/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/experiments/brats/brats_n_clients.py"
PARTITION_DIR="${PARTITION_DIR:-${PROJECT_DIR}/data/partitions/brats2d_2client_extreme/client_data}"
GLOBAL_TEST_DIR="${GLOBAL_TEST_DIR:-${PROJECT_DIR}/data/partitions/brats2d_2client_extreme/global_test}"
RESULTS_DIR="${PROJECT_DIR}/results/brats_2client"
LOG_DIR="${PROJECT_DIR}/experiments/brats/logs"

# ======= Hyperparams =======
#
#  2-client setup: equal data sizes, corruption-only heterogeneity.
#  The corrupted client has 50% aggregation weight, so its divergent
#  gradients strongly destabilize FedAvg.  FedProx's proximal term
#  constrains both clients, keeping the global model stable.
#
#  MU=0.3: higher than 8-client (0.1) because with only 2 clients
#  there is no cancellation of diverse updates — drift is more extreme.
#
STRATEGY="${STRATEGY:-both}"
ROUNDS="${ROUNDS:-50}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-0.01}"
FRACTION_FIT="${FRACTION_FIT:-1.0}"        # both clients every round
MU="${MU:-0.3}"                            # higher mu for 2-client extreme drift
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"        # no weight decay — FedProx is the only regulariser
DROP_PERCENT="${DROP_PERCENT:-0.0}"         # no stragglers — corruption heterogeneity only
MODEL_BASE="${MODEL_BASE:-16}"
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
echo "BraTS 2D Federated Learning — 2-Client Setup"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node(s):      ${SLURM_NODELIST}"
echo "Workdir:      $(pwd)"
echo "Script:       ${SRC_FILE}"
echo "Partition dir:${PARTITION_DIR}"
echo "Global test:  ${GLOBAL_TEST_DIR}"
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

if [[ -n "${GLOBAL_TEST_DIR}" ]]; then
    COMMON_ARGS+=(--global_test_dir "${GLOBAL_TEST_DIR}")
fi

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
