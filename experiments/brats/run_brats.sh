#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job: BraTS 2D Federated Learning (N clients)
#!  FedAvg / FedProx — pure PyTorch (no Flower)
#!
#!  DEFAULT: runs FedAvg then FedProx under straggler conditions
#!  (E=5, drop_percent=0.3) — the setting where FedProx shines.
#!
#!  Usage:
#!    sbatch run_brats.sh                                  # both strategies, straggler setting
#!    STRATEGY=fedprox MU=0.5 sbatch run_brats.sh          # FedProx only, custom mu
#!    STRATEGY=fedavg sbatch run_brats.sh                   # FedAvg only
#!
#!  Ablation examples (submit as separate jobs):
#!    # --- No stragglers (data heterogeneity only) ---
#!    DROP_PERCENT=0.0 sbatch run_brats.sh
#!
#!    # --- Heavier stragglers ---
#!    DROP_PERCENT=0.5 sbatch run_brats.sh
#!
#!    # --- Sweep mu values ---
#!    STRATEGY=fedprox MU=0.01 sbatch run_brats.sh
#!    STRATEGY=fedprox MU=0.1  sbatch run_brats.sh
#!    STRATEGY=fedprox MU=0.5  sbatch run_brats.sh
#!    STRATEGY=fedprox MU=1.0  sbatch run_brats.sh
#! ==============================================================

#SBATCH -J brats_fed_N
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/home/bk489/federated/federated-thesis/experiments/brats/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated/federated-thesis/experiments/brats/logs/%x_%j.err




# ======= Paths =======
PROJECT_DIR="$HOME/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/experiments/brats/brats_n_clients.py"
PARTITION_DIR="${PARTITION_DIR:-${PROJECT_DIR}/data/partitions/brats2d_8client_heterogeneous/client_data}"
GLOBAL_TEST_DIR="${GLOBAL_TEST_DIR:-${PROJECT_DIR}/data/partitions/brats2d_8client_heterogeneous/global_test}"
RESULTS_DIR="${PROJECT_DIR}/results/brats"
LOG_DIR="${PROJECT_DIR}/experiments/brats/logs"

# ======= Hyperparams (override by exporting before sbatch) =======
#
#  Key changes from previous runs (E=30, drop=0.0, mu=0.4):
#
#  1. LOCAL_EPOCHS=5  (was 30)
#     With 30 local epochs, clients drift so far from the global model
#     that the proximal term (a simple L2 penalty) is overwhelmed by
#     the accumulated gradient updates.  With E=5, the proximal term
#     has proportionally ~6x more influence, which is the regime where
#     FedProx is designed to operate (Li et al., 2020 use E=1-20).
#
#  2. DROP_PERCENT=0.3  (was 0.0)
#     Straggler simulation: 30% of selected clients each round receive
#     a *random* number of local epochs in [1, E).  This creates the
#     systems heterogeneity that FedProx was explicitly designed for.
#     FedAvg treats all updates equally regardless of how many epochs
#     a client ran; FedProx's proximal term keeps partial updates
#     aligned with the global model.
#
#  3. MU=0.1  (was 0.4)
#     With E=5 the per-round client drift is ~6x smaller, so a smaller
#     mu suffices.  mu=0.1 provides gentle regularisation without
#     starving clients of local learning capacity.
#
STRATEGY="${STRATEGY:-both}"               # fedavg, fedprox, or both
ROUNDS="${ROUNDS:-50}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"          # high drift amplifies FedProx benefit
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-0.01}"
FRACTION_FIT="${FRACTION_FIT:-1.0}"        # all 8 clients every round
MU="${MU:-0.1}"                            # proximal coefficient, tuned for E=5
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"        # no weight decay — let FedProx be the only drift control
DROP_PERCENT="${DROP_PERCENT:-0.0}"         # 0.0 = data heterogeneity only (set 0.3 for systems hetero ablation)
MODEL_BASE="${MODEL_BASE:-16}"             # UNet2D base filters
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
echo "Global test:  ${GLOBAL_TEST_DIR:-<per-client test pooled>}"
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

# Append global_test_dir if set
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
