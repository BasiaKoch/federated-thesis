#!/bin/bash
#! ==============================================================
#!  SBATCH job submission script for Flower MNIST 2-Digit
#!  FedProx Strategy - 10 Clients
#! ==============================================================

#SBATCH -J flower_fedprox                 # Job name
#SBATCH -A FERGUSSON-SL3-GPU              # Your GPU project account
#SBATCH -p ampere                         # Ampere GPU partition
#SBATCH --nodes=1                         # One node only
#SBATCH --ntasks=1                        # One task
#SBATCH --gres=gpu:1                      # One GPU (A100)
#SBATCH --cpus-per-task=3                 # <=3 CPUs per GPU per CSD3 rules
#SBATCH --time=01:00:00                   # 1 hour (adjust as needed)
#SBATCH --output=flower_fedprox_%j.out    # %j = job ID
#SBATCH --error=flower_fedprox_%j.err     # Error log
#SBATCH --qos=INTR

#! ======= Configuration (modify these) =======
NUM_ROUNDS="${1:-30}"                     # Number of federation rounds
MU="${2:-0.01}"                           # FedProx proximal term coefficient
NUM_CLIENTS=10                            # Fixed: 10 clients for 2-digit pairing
LOCAL_EPOCHS="${3:-1}"                    # Local training epochs per round
LEARNING_RATE="${4:-0.01}"                # Learning rate
SEED="${5:-42}"                           # Random seed

#! ======= Paths =======
PROJECT_DIR="$HOME/federated/federated-thesis"
RESULTS_DIR="${PROJECT_DIR}/results/flower_mnist_2digits"
SRC_DIR="${PROJECT_DIR}/src"

#! ======= Load required environment modules =======
if [ -f /etc/profile.d/modules.sh ]; then
    . /etc/profile.d/modules.sh
    module purge 2>/dev/null || true
    module load rhel8/default-amp 2>/dev/null || true
fi

#! ======= Activate conda environment =======
source ~/.bashrc
conda activate fed || {
    echo "ERROR: Conda environment 'fed' not found"
    echo "Create it with: conda create -n fed python=3.9 && conda activate fed"
    echo "Then install: pip install flwr torch torchvision"
    exit 1
}

#! ======= Fix libstdc++ if needed =======
if [ -f /usr/lib64/libstdc++.so.6 ]; then
    export LD_PRELOAD="/usr/lib64/libstdc++.so.6"
fi

#! ======= Diagnostics =======
echo "=============================================="
echo "Flower MNIST 2-Digit - FedProx"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Strategy: FedProx (mu=$MU)"
echo "Rounds: $NUM_ROUNDS"
echo "Clients: $NUM_CLIENTS"
echo "Local Epochs: $LOCAL_EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Seed: $SEED"
echo "=============================================="
echo "Python: $(which python)"
python -c "import sys; print('Python version:', sys.version)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import flwr; print('Flower:', flwr.__version__)"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "=============================================="

#! ======= Create results directory =======
mkdir -p "$RESULTS_DIR"

#! ======= Run experiment =======
echo ""
echo "Starting FedProx experiment at $(date)"
echo ""

cd "$PROJECT_DIR"

python "${SRC_DIR}/flower_mnist_2digits.py" \
    --strategy fedprox \
    --mu ${MU} \
    --num_clients ${NUM_CLIENTS} \
    --rounds ${NUM_ROUNDS} \
    --local_epochs ${LOCAL_EPOCHS} \
    --lr ${LEARNING_RATE} \
    --seed ${SEED} \
    --use_cuda \
    --output_dir "${RESULTS_DIR}" \
    --run_name "fedprox_r${NUM_ROUNDS}_mu${MU}_e${LOCAL_EPOCHS}_lr${LEARNING_RATE}_s${SEED}"

#! ======= Done =======
echo ""
echo "=============================================="
echo "FedProx Experiment Complete!"
echo "=============================================="
echo "End time: $(date)"
echo "Results saved in: ${RESULTS_DIR}"
echo ""
