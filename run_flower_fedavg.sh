#!/bin/bash
# ==============================================================
# SBATCH: Flower MNIST 2-Digit (FedProx)
# Simple + robust (few moving parts)
# ==============================================================

#SBATCH -J flower_fedprox
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=01:00:00
#SBATCH --output=flower_fedprox_%j.out
#SBATCH --error=flower_fedprox_%j.err
#SBATCH --qos=INTR

# ---- Simple config (edit only these if needed) ----
NUM_ROUNDS=30
MU=1.0
FRACTION_FIT=0.5
LOCAL_EPOCHS=5
LEARNING_RATE=0.05
SEED=42
NUM_CLIENTS=10

# ---- Paths ----
PROJECT_DIR="$HOME/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/src/flower_mnist_2digits.py"
RESULTS_DIR="${PROJECT_DIR}/results/flower_mnist_2digits"

# ---- Environment ----
source ~/.bashrc
conda activate fed

# Avoid CPU oversubscription (safe on HPC)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"

mkdir -p "${RESULTS_DIR}"
cd "${PROJECT_DIR}" || exit 1

RUN_NAME="fedprox_r${NUM_ROUNDS}_mu${MU}_ff${FRACTION_FIT}_e${LOCAL_EPOCHS}_lr${LEARNING_RATE}_s${SEED}"

echo "=============================================="
echo "Flower MNIST 2-Digit - FedProx"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Rounds: ${NUM_ROUNDS}"
echo "Clients: ${NUM_CLIENTS}"
echo "mu: ${MU}"
echo "fraction_fit: ${FRACTION_FIT}"
echo "local_epochs: ${LOCAL_EPOCHS}"
echo "lr: ${LEARNING_RATE}"
echo "seed: ${SEED}"
echo "Results dir: ${RESULTS_DIR}"
echo "Run name: ${RUN_NAME}"
echo "=============================================="

python "${SRC_FILE}" \
  --strategy fedprox \
  --mu "${MU}" \
  --num_clients "${NUM_CLIENTS}" \
  --rounds "${NUM_ROUNDS}" \
  --fraction_fit "${FRACTION_FIT}" \
  --local_epochs "${LOCAL_EPOCHS}" \
  --lr "${LEARNING_RATE}" \
  --seed "${SEED}" \
  --use_cuda \
  --output_dir "${RESULTS_DIR}" \
  --run_name "${RUN_NAME}"
