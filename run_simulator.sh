#!/bin/bash
# Run NVFlare MNIST Federated Learning Simulation
#
# Usage:
#   ./run_simulator.sh [fedavg|fedprox] [num_rounds]
#
# Examples:
#   ./run_simulator.sh fedprox 50    # Run FedProx for 50 rounds
#   ./run_simulator.sh fedavg 30     # Run FedAvg for 30 rounds

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
JOBS_DIR="${SCRIPT_DIR}/jobs"
WORKSPACE="${SCRIPT_DIR}/workspace"

# Default parameters
ALGORITHM="${1:-fedprox}"
NUM_ROUNDS="${2:-50}"
NUM_CLIENTS=30
MIN_CLIENTS=10

# Validate algorithm choice
if [[ "$ALGORITHM" != "fedavg" && "$ALGORITHM" != "fedprox" ]]; then
    echo "Error: Algorithm must be 'fedavg' or 'fedprox'"
    echo "Usage: $0 [fedavg|fedprox] [num_rounds]"
    exit 1
fi

JOB_NAME="mnist_${ALGORITHM}"
JOB_DIR="${JOBS_DIR}/${JOB_NAME}"

echo "=============================================="
echo "NVFlare MNIST Federated Learning Simulator"
echo "=============================================="
ALGORITHM_UPPER=$(echo "$ALGORITHM" | tr '[:lower:]' '[:upper:]')
echo "Algorithm: ${ALGORITHM_UPPER}"
echo "Clients: ${NUM_CLIENTS} total, ${MIN_CLIENTS} per round"
echo "Rounds: ${NUM_ROUNDS}"
echo "Job: ${JOB_NAME}"
echo "=============================================="

# Step 1: Check if data partition exists
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
    echo "Data partition created successfully."
else
    echo ""
    echo "Step 1: Data partition already exists. Skipping..."
fi

# Step 2: Update job configuration with specified number of rounds
echo ""
echo "Step 2: Configuring job for ${NUM_ROUNDS} rounds..."

# Update server config
SERVER_CONFIG="${JOB_DIR}/app/config/config_fed_server.json"
if [[ -f "$SERVER_CONFIG" ]]; then
    # Use Python to update JSON (more reliable than sed for JSON)
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

print(f"Updated server config to {${NUM_ROUNDS}} rounds")
EOF
fi

# Step 3: Generate client site names
echo ""
echo "Step 3: Generating client site configuration..."
SITES=""
for i in $(seq 1 ${NUM_CLIENTS}); do
    if [[ -z "$SITES" ]]; then
        SITES="site-${i}"
    else
        SITES="${SITES},site-${i}"
    fi
done

# Step 4: Run the simulator
echo ""
echo "Step 4: Starting NVFlare simulator..."
echo "Workspace: ${WORKSPACE}/${JOB_NAME}"
echo ""

# Clean previous workspace if exists
if [[ -d "${WORKSPACE}/${JOB_NAME}" ]]; then
    echo "Cleaning previous workspace..."
    rm -rf "${WORKSPACE}/${JOB_NAME}"
fi

# Set data root for clients
export DATA_ROOT="${DATA_DIR}"

# Run NVFlare simulator
nvflare simulator \
    -w "${WORKSPACE}/${JOB_NAME}" \
    -n ${NUM_CLIENTS} \
    -t ${MIN_CLIENTS} \
    "${JOB_DIR}"

echo ""
echo "=============================================="
echo "Simulation Complete!"
echo "=============================================="
echo "Results saved in: ${WORKSPACE}/${JOB_NAME}"
echo ""
echo "To analyze results, run:"
echo "  python analyze_results.py --workspace ${WORKSPACE}/${JOB_NAME}"
echo ""
