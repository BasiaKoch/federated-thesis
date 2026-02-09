#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job: MNIST Federated Learning (N clients)
#!  FedAvg / FedProx — pure PyTorch (no Flower)
#!
#!  Usage:
#!    sbatch run_mnist.sh                             # defaults: FedAvg, 10 clients, shard
#!    STRATEGY=fedprox MU=0.1 sbatch run_mnist.sh     # FedProx
#!    NUM_CLIENTS=20 PARTITION=dirichlet ALPHA=0.5 sbatch run_mnist.sh
#!    STRATEGY=both sbatch run_mnist.sh               # run FedAvg then FedProx
#! ==============================================================

#SBATCH -J mnist_fed_N
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/bk489/federated/federated-thesis/experiments/mnist/logs/%x_%j.out
#SBATCH --error=/home/bk489/federated/federated-thesis/experiments/mnist/logs/%x_%j.err
#SBATCH --qos=INTR

set -euo pipefail

# ======= Paths =======
PROJECT_DIR="$HOME/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/experiments/mnist/mnist_n_clients.py"
DATA_DIR="${DATA_DIR:-$HOME/federated/federated-thesis/data/mnistdataset}"
RESULTS_DIR="${PROJECT_DIR}/results/mnist"
LOG_DIR="${PROJECT_DIR}/experiments/mnist/logs"

# ======= Hyperparams (override by exporting before sbatch) =======
STRATEGY="${STRATEGY:-fedavg}"          # fedavg, fedprox, or both
MODEL="${MODEL:-mclr}"                  # cnn or mclr (FedProx paper uses mclr)
NUM_CLIENTS="${NUM_CLIENTS:-10}"
ROUNDS="${ROUNDS:-30}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-10}"
LR="${LR:-0.01}"
FRACTION_FIT="${FRACTION_FIT:-0.5}"
MU="${MU:-1}"                            # only used if STRATEGY=fedprox or both
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"   # L2 regularization
DROP_PERCENT="${DROP_PERCENT:-0.5}"     # fraction of stragglers
PARTITION="${PARTITION:-niid}"           # iid, shard, dirichlet, or niid (FedProx paper)
ALPHA="${ALPHA:-0.1}"                   # Dirichlet alpha (only if PARTITION=dirichlet)
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
echo "MNIST Federated Learning (${NUM_CLIENTS} clients)"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node(s):     ${SLURM_NODELIST}"
echo "Workdir:     $(pwd)"
echo "Script:      ${SRC_FILE}"
echo "Data dir:    ${DATA_DIR}"
echo "Results dir: ${RESULTS_DIR}"
echo "Strategy:    ${STRATEGY} (mu=${MU})"
echo "Model:       ${MODEL}"
echo "Clients:     ${NUM_CLIENTS} | Fraction: ${FRACTION_FIT}"
echo "Rounds:      ${ROUNDS} | LocalEpochs: ${LOCAL_EPOCHS}"
echo "Batch:       ${BATCH_SIZE} | LR: ${LR} | Seed: ${SEED}"
echo "WeightDecay: ${WEIGHT_DECAY} | DropPercent: ${DROP_PERCENT}"
echo "Partition:   ${PARTITION} (alpha=${ALPHA})"
echo "Python:      $(which python)"
python -c "import sys; print('Python', sys.version)"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available(), 'Device count:', torch.cuda.device_count())"
nvidia-smi || true
echo "=============================================="

# ======= Fail-fast checks =======
[ -f "${SRC_FILE}" ] || { echo "ERROR: missing script: ${SRC_FILE}"; exit 1; }
[ -d "${DATA_DIR}" ]  || { echo "ERROR: missing data dir: ${DATA_DIR}"; exit 1; }

# Validate strategy
if [[ "${STRATEGY}" != "fedavg" && "${STRATEGY}" != "fedprox" && "${STRATEGY}" != "both" ]]; then
    echo "ERROR: STRATEGY must be 'fedavg', 'fedprox', or 'both' (got '${STRATEGY}')"
    exit 1
fi

# ======= Common args =======
COMMON_ARGS=(
    --data_dir "${DATA_DIR}"
    --model "${MODEL}"
    --num_clients "${NUM_CLIENTS}"
    --rounds "${ROUNDS}"
    --local_epochs "${LOCAL_EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --drop_percent "${DROP_PERCENT}"
    --fraction_fit "${FRACTION_FIT}"
    --partition "${PARTITION}"
    --alpha "${ALPHA}"
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
