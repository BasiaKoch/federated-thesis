#!/bin/bash
# ==============================================================
# Run baseline experiments for MNIST 2-Digit comparison
# ==============================================================
#
# Usage:
#   ./run_baseline.sh [centralized|local|both] [epochs]
#
# Examples:
#   ./run_baseline.sh both 30        # Run both baselines for 30 epochs
#   ./run_baseline.sh centralized 30 # Run only centralized baseline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
RESULTS_DIR="${SCRIPT_DIR}/results/flower_mnist_2digits"

# Default parameters
BASELINE="${1:-both}"
EPOCHS="${2:-30}"
LEARNING_RATE="${3:-0.01}"
SEED="${4:-42}"

# Validate baseline choice
if [[ "$BASELINE" != "centralized" && "$BASELINE" != "local" && "$BASELINE" != "both" ]]; then
    echo "Error: Baseline must be 'centralized', 'local', or 'both'"
    echo "Usage: $0 [centralized|local|both] [epochs] [lr] [seed]"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "MNIST 2-Digit Baseline Experiments"
echo "=============================================="
echo "Baseline: $BASELINE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
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

python "${SRC_DIR}/baseline_mnist_2digits.py" \
    --baseline "$BASELINE" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --seed "$SEED" \
    $USE_CUDA \
    --output_dir "${RESULTS_DIR}"

echo ""
echo "=============================================="
echo "Baseline Experiments Complete!"
echo "=============================================="
echo "Results saved in: ${RESULTS_DIR}"
echo ""
