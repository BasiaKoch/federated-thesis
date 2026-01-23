#!/bin/bash
#! ==============================================================
#!  CSD3 Ampere GPU job submission script for Flower MNIST 2-Digit
#!  FedAvg (standard federated averaging, no proximal term)
#! ==============================================================

#SBATCH -J flower_fedavg
#SBATCH -A FERGUSSON-SL3-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=01:00:00
#SBATCH --output=flower_fedavg_%j.out
#SBATCH --error=flower_fedavg_%j.err
#SBATCH --qos=INTR

#! ======= Simple configuration (MUST MATCH FedProx for fair comparison) =======
NUM_ROUNDS=30
FRACTION_FIT=0.5
LOCAL_EPOCHS=5
LEARNING_RATE=0.05
SEED=42
NUM_CLIENTS=10

PROJECT_DIR="$HOME/federated/federated-thesis"
SRC_FILE="${PROJECT_DIR}/src/flower_mnist_2digits.py"
RESULTS_DIR="${PROJECT_DIR}/results/flower_mnist_2digits"

#! ======= Load required environment modules =======
. /etc/profile.d/modules.sh
module load rhel8/default-amp
# If your cluster requires these for CUDA/PyTorch, keep them (match your working template)
module load gcc/9 cuda/12.1 cudnn openmpi/gcc/9.3/4.0.4

#! ======= Activate environment (DO NOT source ~/.bashrc) =======
# Option A: venv (like your mnist_env_gpu_crazy)
# source ~/mnist_env_gpu_crazy/bin/activate

# Option B: conda (recommended if your env is conda-based)
# Use conda.sh directly so ~/.bashrc is NOT needed
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null
conda activate fed

#! ======= Diagnostics =======
echo "=============================================="
echo "Flower MNIST 2-Digit - FedAvg"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "Python: $(which python)"
python -c "import sys; print('Python', sys.version)"
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
python -c "import flwr; print('Flower', flwr.__version__)"
nvidia-smi
echo "=============================================="
echo "Rounds: ${NUM_ROUNDS}"
echo "Clients: ${NUM_CLIENTS}"
echo "fraction_fit: ${FRACTION_FIT}"
echo "local_epochs: ${LOCAL_EPOCHS}"
echo "lr: ${LEARNING_RATE}"
echo "seed: ${SEED}"
echo "Results dir: ${RESULTS_DIR}"
echo "=============================================="

#! ======= Create results directory =======
mkdir -p "${RESULTS_DIR}"
cd "${PROJECT_DIR}"

RUN_NAME="fedavg_r${NUM_ROUNDS}_ff${FRACTION_FIT}_e${LOCAL_EPOCHS}_lr${LEARNING_RATE}_s${SEED}"

echo "Starting Flower FedAvg run: ${RUN_NAME}"
python "${SRC_FILE}" \
  --strategy fedavg \
  --num_clients "${NUM_CLIENTS}" \
  --rounds "${NUM_ROUNDS}" \
  --fraction_fit "${FRACTION_FIT}" \
  --local_epochs "${LOCAL_EPOCHS}" \
  --lr "${LEARNING_RATE}" \
  --seed "${SEED}" \
  --use_cuda \
  --output_dir "${RESULTS_DIR}" \
  --run_name "${RUN_NAME}"

echo "✅ Job completed successfully!"
