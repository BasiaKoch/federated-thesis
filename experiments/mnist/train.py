import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import flwr as fl
import numpy as np
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -----------------------
# Model (small CNN)
# -----------------------
class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)  # 64 * 12 * 12
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    # NumpyClient expects numpy arrays (CPU)
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Safer than torch.tensor(v):
    - Preserves dtype of existing model tensors (usually float32)
    - Preserves device (CPU/GPU)
    """
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(keys) != len(parameters):
        raise ValueError(f"Parameter length mismatch: {len(keys)} vs {len(parameters)}")

    new_state = {}
    for k, arr in zip(keys, parameters):
        old_t = state_dict[k]
        t = torch.from_numpy(arr)
        # Move + cast to match the model tensor
        t = t.to(device=old_t.device, dtype=old_t.dtype)
        new_state[k] = t

    model.load_state_dict(new_state, strict=True)


# -----------------------
# Data partitioning: 10 clients, 2 digits each, disjoint samples
# Each client i gets digits: (i, (i+5)%10)
# Each digit appears in exactly 2 clients, and its samples are split between them.
# -----------------------
def make_two_digit_partitions(
    y: np.ndarray,
    num_clients: int = 10,
    seed: int = 42,
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    assert num_clients == 10, "This partitioner assumes 10 clients for the neat (i, i+5) pairing."
    rng = np.random.default_rng(seed)

    client_digits: List[Tuple[int, int]] = [(i, (i + 5) % 10) for i in range(num_clients)]
    digit_owners: Dict[int, Tuple[int, int]] = {d: (d, (d - 5) % 10) for d in range(10)}

    indices_by_digit: Dict[int, List[int]] = {d: [] for d in range(10)}
    for idx, label in enumerate(y):
        indices_by_digit[int(label)].append(int(idx))

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for d in range(10):
        idxs = np.array(indices_by_digit[d], dtype=np.int64)
        rng.shuffle(idxs)
        c1, c2 = digit_owners[d]
        half = len(idxs) // 2
        client_indices[c1].extend(idxs[:half].tolist())
        client_indices[c2].extend(idxs[half:].tolist())

    for c in range(num_clients):
        rng.shuffle(client_indices[c])

    return client_indices, client_digits


# -----------------------
# Train / eval
# -----------------------
# Following the reference implementation: github.com/litian96/FedProx
# FedAvg: Standard SGD local training (McMahan et al., 2017)
# FedProx: SGD with proximal term (Li et al., MLSys 2020)
#
# Key insight from reference:
#   FedProx objective: min F_k(w) + (mu/2) * ||w - w^t||^2
#   where w^t is the global model at round t (frozen during local training)
#
# The gradient becomes: grad F_k(w) + mu * (w - w^t)
# which is equivalent to the PerturbedGradientDescent optimizer in the reference.
# -----------------------


def train_local_epochs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    mu: float,
    global_params: Optional[List[torch.Tensor]],
) -> Dict[str, float]:
    """
    Local training for FedAvg (mu=0) or FedProx (mu>0).

    Following the reference implementation (github.com/litian96/FedProx):
    - Uses SGD without momentum (matching the original FedProx paper)
    - Creates optimizer ONCE per round (not per epoch) for efficiency
    - Applies proximal term: (mu/2) * ||w - w_global||^2
    - The gradient of the proximal term is: mu * (w - w_global)

    Args:
        model: The neural network model
        loader: DataLoader for training data
        device: torch device (cpu/cuda)
        lr: Learning rate
        epochs: Number of local epochs (E in FedAvg/FedProx papers)
        mu: Proximal term coefficient (0 for FedAvg, >0 for FedProx)
        global_params: Frozen snapshot of global model parameters (required if mu > 0)

    Returns:
        Dictionary with training metrics (final loss, avg loss)
    """
    model.train()

    # Create optimizer ONCE per round (following reference implementation)
    # No momentum, as per the original FedProx paper
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_batches = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            # FedProx: Add proximal term to prevent client drift
            # Reference: Li et al., "Federated Optimization in Heterogeneous Networks"
            # Objective: F_k(w) + (mu/2) * ||w - w^t||^2
            if mu > 0.0 and global_params is not None:
                prox_term = 0.0
                for p, p_global in zip(model.parameters(), global_params):
                    # p_global is detached (frozen), so gradient only flows through p
                    prox_term = prox_term + torch.sum((p - p_global) ** 2)
                loss = loss + (mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_batches += 1

        total_loss += epoch_loss
        num_batches += epoch_batches

    avg_loss = total_loss / max(num_batches, 1)
    return {"avg_loss": avg_loss, "num_batches": num_batches}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model on a dataset.

    Following reference implementation (github.com/litian96/FedProx):
    - Returns (loss, accuracy) tuple
    - Loss is average per-sample loss
    - Accuracy is fraction of correct predictions

    Args:
        model: The neural network model
        loader: DataLoader for evaluation data
        device: torch device (cpu/cuda)

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction='sum')  # Sum for proper averaging
        total_loss += float(loss.item())
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(x.size(0))
    return total_loss / max(total, 1), correct / max(total, 1)


# -----------------------
# Flower client
# -----------------------
# Following the reference implementation (github.com/litian96/FedProx):
# - Each client maintains its own model
# - Receives global parameters from server at start of each round
# - Performs local training with FedAvg or FedProx objective
# - Returns updated parameters weighted by dataset size
# -----------------------


class MnistClient(fl.client.NumPyClient):
    """
    Flower client for MNIST federated learning.

    Implements both FedAvg and FedProx based on mu parameter:
    - mu = 0: FedAvg (standard SGD)
    - mu > 0: FedProx (SGD with proximal term)

    Reference: github.com/litian96/FedProx
    """

    def __init__(
        self,
        cid: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        lr: float,
        local_epochs: int,
        mu: float = 0.0,  # Proximal term coefficient
    ):
        self.cid = cid
        self.model = SmallCNN().to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.mu = mu  # 0 for FedAvg, >0 for FedProx

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        """
        Local training round.

        Following reference implementation (github.com/litian96/FedProx):
        1. Load global (server) weights into local model
        2. Snapshot global params as frozen reference for proximal term
        3. Perform E epochs of local training with FedAvg or FedProx objective
        4. Return updated weights and dataset size for weighted aggregation
        """
        # Step 1: Load global (server) weights
        set_parameters(self.model, parameters)

        # Step 2: Snapshot global params BEFORE training (frozen reference)
        # These are used as w^t in the proximal term: (mu/2) * ||w - w^t||^2
        # Must be detached so gradients don't flow through them
        global_params = None
        if self.mu > 0.0:
            global_params = [p.detach().clone() for p in self.model.parameters()]

        # DEBUG: Print mu for client 0 to verify FedProx is active
        if self.cid == "0":
            print(f"  [Client 0] mu={self.mu:.4f}, lr={self.lr}, local_epochs={self.local_epochs}")

        # Step 3: Local training (unified function handles both FedAvg and FedProx)
        train_metrics = train_local_epochs(
            model=self.model,
            loader=self.train_loader,
            device=self.device,
            lr=self.lr,
            epochs=self.local_epochs,
            mu=self.mu,
            global_params=global_params,
        )

        # DEBUG: Measure drift (how much weights moved from global)
        # FedProx should result in smaller drift compared to FedAvg
        if self.cid == "0" and global_params is not None:
            with torch.no_grad():
                drift = sum(
                    torch.sum((p - p0) ** 2).item()
                    for p, p0 in zip(self.model.parameters(), global_params)
                )
            print(f"  [Client 0] drift={drift:.6f} (smaller = less client drift)")

        # Step 4: Return updated weights and dataset size
        # Server uses dataset size for weighted averaging (FedAvg aggregation)
        return (
            get_parameters(self.model),
            len(self.train_loader.dataset),
            {"cid": self.cid, "mu": self.mu, "avg_loss": train_metrics["avg_loss"]},
        )

    def evaluate(self, parameters, config):
        """Evaluate global model on local test data."""
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.test_loader, self.device)
        return float(loss), len(self.test_loader.dataset), {"local_acc": float(acc)}


# -----------------------
# Metrics tracking utilities
# -----------------------
def compute_convergence_round(accuracies: List[float], threshold: float = 0.95, window: int = 3) -> int:
    if len(accuracies) < window:
        return -1
    for i in range(len(accuracies) - window + 1):
        if all(acc >= threshold for acc in accuracies[i : i + window]):
            return i
    return -1


def compute_stability_metrics(values: List[float]) -> Dict[str, float]:
    if len(values) < 2:
        return {"variance": 0.0, "max_oscillation": 0.0, "smoothness": 1.0, "mean_change_per_round": 0.0}

    arr = np.array(values, dtype=np.float64)
    variance = float(np.var(arr))
    diffs = np.abs(np.diff(arr))
    max_oscillation = float(np.max(diffs)) if len(diffs) > 0 else 0.0
    mean_diff = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    smoothness = 1.0 / (1.0 + mean_diff)

    return {
        "variance": variance,
        "max_oscillation": max_oscillation,
        "smoothness": smoothness,
        "mean_change_per_round": mean_diff,
    }


# -----------------------
# Config loading
# -----------------------
def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """Merge YAML config with command-line args. CLI args take precedence."""
    # Map config keys to argparse attribute names
    key_mapping = {
        "learning_rate": "lr",
        "use_cuda": "use_cuda",
    }

    for key, value in config.items():
        arg_key = key_mapping.get(key, key)
        # Only set from config if not explicitly provided via CLI
        if hasattr(args, arg_key):
            cli_default = getattr(args, f"_default_{arg_key}", None)
            current_val = getattr(args, arg_key)
            # If current value equals the default, use config value
            if current_val == cli_default and value is not None:
                setattr(args, arg_key, value)
    return args


# -----------------------
# Main
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None,
                    help="Path to YAML config file")
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--fraction_fit", type=float, default=0.5,
                    help="Fraction of clients per round (0.5 helps show FedProx benefit)")
    ap.add_argument("--local_epochs", type=int, default=5,
                    help="Local epochs per round (>1 needed to see FedProx effect)")
    ap.add_argument("--lr", type=float, default=0.05,
                    help="Learning rate (higher makes drift more visible)")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for training")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    ap.add_argument("--mu", type=float, default=0.01,
                    help="FedProx proximal term (try 0.01-0.3 for non-IID)")
    ap.add_argument("--output_dir", type=str, default="./results/mnist",
                    help="Directory to save results and metrics")
    ap.add_argument("--run_name", type=str, default=None,
                    help="Optional run name for the experiment")

    # Parse and store defaults for merge logic
    args = ap.parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Store defaults before merging
        for key in ["num_clients", "rounds", "fraction_fit", "local_epochs",
                    "lr", "batch_size", "seed", "strategy", "mu", "output_dir", "run_name"]:
            if hasattr(args, key):
                setattr(args, f"_default_{key}", ap.get_default(key))
        args = merge_config_with_args(config, args)
        # Handle use_cuda from config (since it's a store_true action)
        if "use_cuda" in config and config["use_cuda"]:
            args.use_cuda = True

    if args.num_clients != 10:
        raise ValueError("This script currently implements the neat 10-client pairing (i, i+5). Use 10 clients.")

    # Reproducibility setup (peer-reviewed practice)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # CUDA determinism for reproducible results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mu_tag = f"mu{args.mu}" if args.strategy == "fedprox" else "muNA"
        args.run_name = f"{args.strategy}_rounds{args.rounds}_{mu_tag}_{timestamp}"

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    print(f"\n{'='*60}")
    print("Flower MNIST 2-Digit Federated Learning Experiment")
    print(f"{'='*60}")
    if args.config:
        print(f"Config: {args.config}")
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Proximal mu: {args.mu}" if args.strategy == "fedprox" else "Proximal mu: N/A (FedAvg)")
    print(f"Clients: {args.num_clients}")
    print(f"Rounds: {args.rounds}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {device}")
    print(f"Output dir: {args.output_dir}")
    print(f"Run name: {args.run_name}")
    print(f"{'='*60}\n")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST(root="./data/mnist", train=True, download=True, transform=tfm)
    testset = datasets.MNIST(root="./data/mnist", train=False, download=True, transform=tfm)

    y_train = np.array(trainset.targets)
    y_test = np.array(testset.targets)

    train_partitions, train_digits = make_two_digit_partitions(y_train, num_clients=10, seed=args.seed)
    test_partitions, _ = make_two_digit_partitions(y_test, num_clients=10, seed=args.seed)

    global_test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    def evaluate_fn(server_round: int, parameters, config):
        """
        Server-side evaluation function.

        Following reference implementation (github.com/litian96/FedProx):
        - Evaluates global model on test set after each round
        - Returns loss and metrics for tracking
        """
        model = SmallCNN().to(device)
        set_parameters(model, parameters)
        loss, acc = evaluate(model, global_test_loader, device)
        print(f"  [Round {server_round}] Global Test - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return float(loss), {"global_acc": float(acc), "global_loss": float(loss), "round": int(server_round)}

    def client_fn(cid: str):
        """
        Client factory function.

        Following reference implementation (github.com/litian96/FedProx):
        - Each client has its own data partition (non-IID in this case)
        - mu=0 for FedAvg, mu>0 for FedProx
        """
        c = int(cid)
        train_subset = Subset(trainset, train_partitions[c])
        test_subset = Subset(testset, test_partitions[c])

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=256,
            shuffle=False,
            num_workers=0,
        )

        # FedAvg: mu=0, FedProx: mu>0
        mu = float(args.mu) if args.strategy == "fedprox" else 0.0

        return MnistClient(
            cid=cid,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            lr=args.lr,
            local_epochs=args.local_epochs,
            mu=mu,
        )

    # -----------------------
    # Server Strategy
    # -----------------------
    # IMPORTANT: Both FedAvg and FedProx use the SAME server-side aggregation!
    # Following reference implementation (github.com/litian96/FedProx):
    # - Server performs weighted averaging: w_global = sum(n_k / N) * w_k
    # - The ONLY difference is the client-side objective function
    # - FedAvg: min F_k(w)
    # - FedProx: min F_k(w) + (mu/2) * ||w - w^t||^2
    # -----------------------
    min_fit = max(1, int(args.num_clients * args.fraction_fit))

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.fraction_fit,
        min_fit_clients=min_fit,
        min_available_clients=args.num_clients,
        evaluate_fn=evaluate_fn,
    )

    start_time = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1 if device.type == "cuda" else 0},
    )
    total_time = time.time() - start_time

    # Extract metrics from history
    rounds_list: List[int] = []
    losses_list: List[float] = []
    accuracies_list: List[float] = []

    if history.losses_centralized:
        for rnd, loss in history.losses_centralized:
            rounds_list.append(int(rnd))
            losses_list.append(float(loss))

    if history.metrics_centralized and "global_acc" in history.metrics_centralized:
        # Ensure ordering by round (some histories can be unordered)
        acc_pairs = [(int(rnd), float(acc)) for rnd, acc in history.metrics_centralized["global_acc"]]
        acc_pairs.sort(key=lambda x: x[0])

        # Align accuracies to rounds_list if possible; otherwise build separately
        if rounds_list:
            round_to_idx = {r: i for i, r in enumerate(rounds_list)}
            accuracies_list = [0.0] * len(rounds_list)
            for r, acc in acc_pairs:
                if r in round_to_idx:
                    accuracies_list[round_to_idx[r]] = acc
        else:
            rounds_list = [r for r, _ in acc_pairs]
            accuracies_list = [acc for _, acc in acc_pairs]

    convergence_90 = compute_convergence_round(accuracies_list, threshold=0.90, window=3)
    convergence_95 = compute_convergence_round(accuracies_list, threshold=0.95, window=3)
    convergence_98 = compute_convergence_round(accuracies_list, threshold=0.98, window=3)

    stability_acc = compute_stability_metrics(accuracies_list)
    stability_loss = compute_stability_metrics(losses_list)

    final_acc = accuracies_list[-1] if accuracies_list else 0.0
    final_loss = losses_list[-1] if losses_list else float("inf")
    best_acc = max(accuracies_list) if accuracies_list else 0.0
    best_loss = min(losses_list) if losses_list else float("inf")

    results = {
        "experiment": {
            "config_file": args.config,
            "strategy": args.strategy,
            "mu": args.mu if args.strategy == "fedprox" else None,
            "num_clients": args.num_clients,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "fraction_fit": args.fraction_fit,
            "seed": args.seed,
            "device": str(device),
            "run_name": args.run_name,
            "timestamp": datetime.now().isoformat(),
        },
        "per_round_metrics": {
            "rounds": rounds_list,
            "global_test_loss": losses_list,
            "global_test_accuracy": accuracies_list,
        },
        "final_metrics": {
            "final_accuracy": final_acc,
            "final_loss": final_loss,
            "best_accuracy": best_acc,
            "best_loss": best_loss,
            "total_time_seconds": total_time,
            "avg_time_per_round": total_time / args.rounds if args.rounds > 0 else 0.0,
        },
        "convergence": {
            "round_to_90_acc": convergence_90,
            "round_to_95_acc": convergence_95,
            "round_to_98_acc": convergence_98,
        },
        "stability": {
            "accuracy": stability_acc,
            "loss": stability_loss,
        },
        "client_data_distribution": {
            f"client_{i}": {"digits": list(pair)} for i, pair in enumerate(train_digits)
        },
    }

    results_file = os.path.join(args.output_dir, f"{args.run_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Strategy: {args.strategy.upper()}" + (f" (mu={args.mu})" if args.strategy == "fedprox" else ""))
    print("\n--- Accuracy & Loss ---")
    print(f"Final Accuracy: {final_acc:.4f}")
    print(f"Best Accuracy:  {best_acc:.4f}")
    print(f"Final Loss:     {final_loss:.4f}")
    print(f"Best Loss:      {best_loss:.4f}")
    print("\n--- Convergence (rounds to reach threshold, -1 if never) ---")
    print(f"Rounds to 90% accuracy: {convergence_90}")
    print(f"Rounds to 95% accuracy: {convergence_95}")
    print(f"Rounds to 98% accuracy: {convergence_98}")
    print("\n--- Stability (accuracy) ---")
    print(f"Variance:        {stability_acc['variance']:.6f}")
    print(f"Max Oscillation: {stability_acc['max_oscillation']:.4f}")
    print(f"Smoothness:      {stability_acc['smoothness']:.4f}")
    print("\n--- Timing ---")
    print(f"Total Time:     {total_time:.2f}s")
    print(f"Avg per Round:  {total_time / args.rounds:.2f}s")
    print(f"{'='*60}")

    print("\nClient digit pairs (train):")
    for i, pair in enumerate(train_digits):
        print(f"  client {i}: digits={pair}")


if __name__ == "__main__":
    main()
