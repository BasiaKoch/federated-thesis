#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job submission script for Federated Learning
#!  MNIST with NVFlare (FedProx/FedAvg)
#! ==============================================================

#SBATCH -J fl_mnist                       # Job name
#SBATCH -A FERGUSSON-SL3-GPU              # Your GPU project account
#SBATCH -p ampere                         # Ampere GPU partition
#SBATCH --nodes=1                         # One node only
#SBATCH --ntasks=1                        # One task
#SBATCH --gres=gpu:1                      # One GPU (A100)
#SBATCH --cpus-per-task=12                # More CPUs for NVFlare clients (10 parallel)
#SBATCH --mem=32G                         # Memory for 30 client data
#SBATCH --time=02:00:00                   # 2 hours (50 rounds with 30 clients)
#SBATCH --output=fl_mnist_%j.out          # %j = job ID
#SBATCH --error=fl_mnist_%j.err           # Error log
#SBATCH --qos=INTR

#! ======= Configuration (modify these) =======
ALGORITHM="${1:-fedprox}"                 # fedavg or fedprox
NUM_ROUNDS="${2:-50}"                     # Number of federation rounds
NUM_CLIENTS=30                            # Total simulated clients
MIN_CLIENTS=10                            # Clients per round

#! ======= Paths (modify for your HPC setup) =======
PROJECT_DIR="$HOME/federated-thesis"      # Your project directory on HPC
DATA_DIR="${PROJECT_DIR}/data"
JOBS_DIR="${PROJECT_DIR}/jobs"
WORKSPACE="${PROJECT_DIR}/workspace"

#! ======= Load required environment modules =======
. /etc/profile.d/modules.sh
module load rhel8/default-amp
module load gcc/9 cuda/12.1 cudnn
module load python/3.11                   # Or your preferred Python version

#! ======= Activate your environment =======
# Option 1: Use existing venv (you need to create this on HPC first)
# source ~/fl_env/bin/activate

# Option 2: Use conda
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate fl_env

# For now, assuming venv at project directory
if [ -d "${PROJECT_DIR}/venv" ]; then
    source "${PROJECT_DIR}/venv/bin/activate"
else
    echo "ERROR: Virtual environment not found at ${PROJECT_DIR}/venv"
    echo "Please create it first with: python -m venv ${PROJECT_DIR}/venv"
    echo "Then install requirements: pip install -r ${PROJECT_DIR}/requirements.txt"
    exit 1
fi

#! ======= Diagnostics =======
echo "=============================================="
echo "Federated Learning - MNIST - HPC Job"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Algorithm: $ALGORITHM"
echo "Rounds: $NUM_ROUNDS"
echo "Clients: $NUM_CLIENTS total, $MIN_CLIENTS per round"
echo "=============================================="
echo "Python: $(which python)"
python -c "import sys; print('Python version:', sys.version)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import nvflare; print('NVFlare:', nvflare.__version__)"
nvidia-smi
echo "=============================================="

#! ======= Validate inputs =======
if [[ "$ALGORITHM" != "fedavg" && "$ALGORITHM" != "fedprox" ]]; then
    echo "ERROR: Algorithm must be 'fedavg' or 'fedprox'"
    exit 1
fi

JOB_NAME="mnist_${ALGORITHM}"
JOB_DIR="${JOBS_DIR}/${JOB_NAME}"

#! ======= Step 1: Create data partition if needed =======
PARTITION_FILE="${DATA_DIR}/partitions/mnist_noniid_partition.json"
if [[ ! -f "$PARTITION_FILE" ]]; then
    echo ""
    echo "Step 1: Creating non-IID data partition..."
    cd "${DATA_DIR}"
    python partition_mnist.py \
        --num_clients ${NUM_CLIENTS} \
        --data_file ./mnist.npz \
        --output_dir ./partitions \
        --alpha 1.5 \
        --seed 42
    echo "Data partition created."
else
    echo ""
    echo "Step 1: Data partition exists. Skipping..."
fi

#! ======= Step 2: Update job config with num_rounds =======
echo ""
echo "Step 2: Configuring job for ${NUM_ROUNDS} rounds..."
SERVER_CONFIG="${JOB_DIR}/app/config/config_fed_server.json"
if [[ -f "$SERVER_CONFIG" ]]; then
    python3 - <<EOF
import json
with open("${SERVER_CONFIG}", "r") as f:
    config = json.load(f)
config["num_rounds"] = ${NUM_ROUNDS}
for workflow in config.get("workflows", []):
    if "scatter_and_gather" in workflow.get("id", ""):
        workflow["args"]["num_rounds"] = ${NUM_ROUNDS}
with open("${SERVER_CONFIG}", "w") as f:
    json.dump(config, f, indent=2)
print(f"Updated server config to ${NUM_ROUNDS} rounds")
EOF
fi

#! ======= Step 3: Clean previous workspace =======
echo ""
echo "Step 3: Preparing workspace..."
WORK_PATH="${WORKSPACE}/${JOB_NAME}"
if [[ -d "$WORK_PATH" ]]; then
    echo "Cleaning previous workspace..."
    rm -rf "$WORK_PATH"
fi

#! ======= Step 4: Run NVFlare simulator =======
echo ""
echo "Step 4: Starting NVFlare simulator..."
echo "Workspace: ${WORK_PATH}"
echo "Start time: $(date)"
echo ""

nvflare simulator \
    -w "${WORK_PATH}" \
    -n ${NUM_CLIENTS} \
    -t ${MIN_CLIENTS} \
    "${JOB_DIR}"

#! ======= Done =======
echo ""
echo "=============================================="
echo "Simulation Complete!"
echo "=============================================="
echo "End time: $(date)"
echo "Results saved in: ${WORK_PATH}"
echo ""

#! ======= Optional: Copy results to a safe location =======
# Uncomment to backup results
# RESULTS_BACKUP="$HOME/fl_results/${JOB_NAME}_${SLURM_JOB_ID}"
# mkdir -p "$RESULTS_BACKUP"
# cp -r "${WORK_PATH}/simulate_job" "$RESULTS_BACKUP/"
# echo "Results backed up to: $RESULTS_BACKUP"
