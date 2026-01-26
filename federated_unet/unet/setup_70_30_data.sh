#!/bin/bash
# Setup 70/30 federated data partition on cluster
# Run this ONCE before running training experiments

set -euo pipefail

PROJECT_DIR="${HOME}/federated/federated-thesis"
BRATS_ROOT="${PROJECT_DIR}/data/brats2020_top10_slices_split_npz"
PARTITION_DIR="${PROJECT_DIR}/data/partitions/federated_clients_2_70_30"
SCRIPTS_DIR="${PROJECT_DIR}/federated_unet/unet"

# Load conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "${HOME}/anaconda3/etc/profile.d/conda.sh" 2>/dev/null
conda activate fed

echo "=============================================="
echo "Setting up 70/30 Federated Data Partition"
echo "=============================================="
echo "BraTS root: ${BRATS_ROOT}"
echo "Partition dir: ${PARTITION_DIR}"
echo ""

# Step 1: Create 70/30 partition (case assignments)
echo "Step 1: Creating 70/30 partition..."
python "${SCRIPTS_DIR}/create_70_30_partition.py" \
    --brats_root "${BRATS_ROOT}" \
    --output_dir "${PARTITION_DIR}" \
    --split 0.7 0.3 \
    --seed 42

# Step 2: Build client data with symlinks and train/val/test splits
echo ""
echo "Step 2: Building client data with symlinks..."
python "${SCRIPTS_DIR}/rebuild_client_data.py" \
    --partition_dir "${PARTITION_DIR}" \
    --brats_root "${BRATS_ROOT}" \
    --val_frac 0.10 \
    --test_frac 0.10 \
    --seed 42 \
    --rebuild

# Step 3: Verify the setup
echo ""
echo "=============================================="
echo "Verification"
echo "=============================================="

CLIENT_DATA="${PARTITION_DIR}/client_data"

echo ""
echo "Client 0 data:"
for split in train val test; do
    count=$(find "${CLIENT_DATA}/client_0/${split}" -name "*.npz" 2>/dev/null | wc -l)
    echo "  ${split}: ${count} slices"
done

echo ""
echo "Client 1 data:"
for split in train val test; do
    count=$(find "${CLIENT_DATA}/client_1/${split}" -name "*.npz" 2>/dev/null | wc -l)
    echo "  ${split}: ${count} slices"
done

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Now you can run training with:"
echo "  sbatch ${SCRIPTS_DIR}/run_flower_70_30.sbatch              # FedAvg"
echo "  STRATEGY=fedprox sbatch ${SCRIPTS_DIR}/run_flower_70_30.sbatch  # FedProx"
