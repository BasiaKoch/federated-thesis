#!/bin/bash
# ==============================================================
# Local runner for Flower MNIST 2-Digit experiments
# Use this for testing before submitting to HPC
# ==============================================================
#
# Usage:
#   ./run_flower_local.sh [fedavg|fedprox] [num_rounds] [mu] [local_epochs]
#
# Examples:
#   ./run_flower_local.sh fedavg 30           # FedAvg for 30 rounds
#   ./run_flower_local.sh fedprox 30 0.01     # FedProx with mu=0.01
#   ./run_flower_local.sh fedprox 30 0.1 2    # FedProx with mu=0.1, 2 local epochs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
RESULTS_DIR="${SCRIPT_DIR}/results/flower_mnist_2digits"

# Default parameters
STRATEGY="${1:-fedavg}"
NUM_ROUNDS="${2:-30}"
MU="${3:-0.01}"
LOCAL_EPOCHS="${4:-1}"
LEARNING_RATE="${5:-0.01}"
SEED="${6:-42}"

# Validate strategy
if [[ "$STRATEGY" != "fedavg" && "$STRATEGY" != "fedprox" ]]; then
    echo "Error: Strategy must be 'fedavg' or 'fedprox'"
    echo "Usage: $0 [fedavg|fedprox] [num_rounds] [mu] [local_epochs] [lr] [seed]"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "Flower MNIST 2-Digit - Local Runner"
echo "=============================================="
echo "Strategy: $(echo $STRATEGY | tr '[:lower:]' '[:upper:]')"
if [[ "$STRATEGY" == "fedprox" ]]; then
    echo "Proximal mu: $MU"
fi
echo "Rounds: $NUM_ROUNDS"
echo "Clients: 10 (fixed for 2-digit pairing)"
echo "Local Epochs: $LOCAL_EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Seed: $SEED"
echo "=============================================="

# Check for CUDA
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    USE_CUDA="--use_cuda"
    echo "CUDA: Available - using GPU"
else
    USE_CUDA=""
    echo "CUDA: Not available - using CPU"
fi
echo "=============================================="
echo ""

cd "$SCRIPT_DIR"

# Build run name
if [[ "$STRATEGY" == "fedavg" ]]; then
    RUN_NAME="fedavg_r${NUM_ROUNDS}_e${LOCAL_EPOCHS}_lr${LEARNING_RATE}_s${SEED}"
else
    RUN_NAME="fedprox_r${NUM_ROUNDS}_mu${MU}_e${LOCAL_EPOCHS}_lr${LEARNING_RATE}_s${SEED}"
fi

# Run experiment
python "${SRC_DIR}/flower_mnist_2digits.py" \
    --strategy "$STRATEGY" \
    --mu "$MU" \
    --num_clients 10 \
    --rounds "$NUM_ROUNDS" \
    --local_epochs "$LOCAL_EPOCHS" \
    --lr "$LEARNING_RATE" \
    --seed "$SEED" \
    $USE_CUDA \
    --output_dir "${RESULTS_DIR}" \
    --run_name "$RUN_NAME"

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo "Results saved in: ${RESULTS_DIR}/${RUN_NAME}_results.json"
echo ""
echo "To compare results, run:"
echo "  python src/analyze_flower_results.py --results_dir ${RESULTS_DIR}"
echo ""
