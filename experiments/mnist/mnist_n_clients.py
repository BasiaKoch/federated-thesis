#!/usr/bin/env python3
"""
Federated Learning on MNIST with N clients: FedAvg and FedProx.

Pure-PyTorch simulation (no Flower). Loads MNIST from raw IDX files.

Usage:
    python mnist_n_clients.py --strategy fedavg --num_clients 10 --rounds 30
    python mnist_n_clients.py --strategy fedprox --num_clients 10 --rounds 30 --mu 0.01
    python mnist_n_clients.py --strategy fedprox --num_clients 20 --rounds 50 --mu 0.1 --partition dirichlet --alpha 0.5
"""

import argparse
import copy
import gzip
import json
import os
import struct
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -----------------------
# IDX file loading
# -----------------------
def read_idx_images(path: str) -> np.ndarray:
    """Read IDX image file (gzipped or raw) into numpy array [N, 28, 28]."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols).copy()


def read_idx_labels(path: str) -> np.ndarray:
    """Read IDX label file (gzipped or raw) into numpy array [N]."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.copy()


def load_mnist(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST from raw IDX files. Tries both naming conventions."""
    data_dir = os.path.abspath(data_dir)

    # Try different naming conventions (dot-variant first — dash-variant may be a dir on macOS)
    train_img_candidates = [
        "train-images.idx3-ubyte", "train-images-idx3-ubyte",
        "train-images-idx3-ubyte.gz",
    ]
    train_lbl_candidates = [
        "train-labels.idx1-ubyte", "train-labels-idx1-ubyte",
        "train-labels-idx1-ubyte.gz",
    ]
    test_img_candidates = [
        "t10k-images.idx3-ubyte", "t10k-images-idx3-ubyte",
        "t10k-images-idx3-ubyte.gz",
    ]
    test_lbl_candidates = [
        "t10k-labels.idx1-ubyte", "t10k-labels-idx1-ubyte",
        "t10k-labels-idx1-ubyte.gz",
    ]

    # Handle nested directory (e.g. scp -r creating mnistdataset/mnistdataset/)
    nested = os.path.join(data_dir, "mnistdataset")
    if os.path.isdir(nested):
        data_dir = nested

    def find_file(candidates):
        for name in candidates:
            # Check as a regular file
            full = os.path.join(data_dir, name)
            if os.path.isfile(full):
                return full
            # macOS extracts gzip into a directory with the file inside
            inner = os.path.join(data_dir, name, name)
            if os.path.isfile(inner):
                return inner
        contents = os.listdir(data_dir) if os.path.isdir(data_dir) else []
        raise FileNotFoundError(
            f"Could not find any of {candidates} in {data_dir}\n"
            f"Directory contents: {contents}"
        )

    train_images = read_idx_images(find_file(train_img_candidates))
    train_labels = read_idx_labels(find_file(train_lbl_candidates))
    test_images = read_idx_images(find_file(test_img_candidates))
    test_labels = read_idx_labels(find_file(test_lbl_candidates))

    return train_images, train_labels, test_images, test_labels


def make_dataloader(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader from numpy arrays. Normalizes to MNIST stats."""
    x = torch.from_numpy(images).float().unsqueeze(1) / 255.0
    x = (x - 0.1307) / 0.3081
    y = torch.from_numpy(labels).long()
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


# -----------------------
# Data partitioning
# -----------------------
def partition_iid(
    labels: np.ndarray, num_clients: int, seed: int = 42,
) -> List[np.ndarray]:
    """IID partitioning: randomly shuffle and split equally."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    return [shard for shard in np.array_split(indices, num_clients)]


def partition_niid_powerlaw(
    labels: np.ndarray, num_clients: int, seed: int = 42,
) -> List[np.ndarray]:
    """
    FedProx paper non-IID partition (Li et al., MLSys 2020).

    Each client i gets 2 digit classes: (i % 10) and ((i+1) % 10).
    First assigns a small fixed base per class, then distributes remaining
    samples via power law (lognormal), creating both label heterogeneity
    and quantity imbalance across clients.

    Reference: github.com/litian96/FedProx/blob/master/data/mnist/generate_niid.py
    """
    rng = np.random.default_rng(seed)
    num_classes = 10

    # Group indices by digit
    class_indices: List[np.ndarray] = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0].copy()
        rng.shuffle(idx)
        class_indices.append(idx)

    # Pointer into each class's index array
    ptr = np.zeros(num_classes, dtype=np.int64)

    # Number of groups of 10 clients (for the lognormal props table)
    num_groups = max(1, (num_clients + 9) // 10)

    # Phase 1: Assign a small fixed base (5 samples per class) to each client
    base_per_class = 5
    client_idx: List[List[int]] = [[] for _ in range(num_clients)]
    for k in range(num_clients):
        for j in range(2):
            c = (k + j) % num_classes
            end = min(ptr[c] + base_per_class, len(class_indices[c]))
            client_idx[k].extend(class_indices[c][ptr[c]:end].tolist())
            ptr[c] = end

    # Phase 2: Distribute remaining samples via power law (lognormal)
    # Shape: (num_classes, num_groups, 2) — proportions for each (class, group, j)
    props = rng.lognormal(0, 2.0, (num_classes, num_groups, 2))
    # Scale so that proportions for each class sum to the remaining samples
    remaining = np.array([len(ci) - ptr[c] for c, ci in enumerate(class_indices)],
                         dtype=np.float64)
    for c in range(num_classes):
        if remaining[c] > 0:
            props[c] = remaining[c] * props[c] / props[c].sum()

    for k in range(num_clients):
        group = k // 10
        within = k % 10
        for j in range(2):
            c = (within + j) % num_classes
            if group < num_groups:
                n_samples = int(props[c, group, j])
            else:
                n_samples = 0
            end = min(ptr[c] + n_samples, len(class_indices[c]))
            client_idx[k].extend(class_indices[c][ptr[c]:end].tolist())
            ptr[c] = end

    # Shuffle each client's data
    for k in range(num_clients):
        arr = np.array(client_idx[k])
        rng.shuffle(arr)
        client_idx[k] = arr.tolist()

    return [np.array(idx) for idx in client_idx]

class MCLR(nn.Module):
    """Multinomial logistic regression (FedProx paper model)."""
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        return self.fc(x)

# -----------------------
# Local training
# -----------------------
def train_local(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    mu: float,
    global_params: Optional[List[torch.Tensor]],
    weight_decay: float = 0.0,
) -> float:
    """
    Local training for FedAvg (mu=0) or FedProx (mu>0).

    FedProx objective: F_k(w) + (mu/2) * ||w - w^t||^2
    Reference: Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0,
                                weight_decay=weight_decay)

    total_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            if mu > 0.0 and global_params is not None:
                prox_term = 0.0
                for p, p_global in zip(model.parameters(), global_params):
                    prox_term = prox_term + torch.sum((p - p_global) ** 2)
                loss = loss + (mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> Tuple[float, float]:
    """Return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


# -----------------------
# FedAvg / FedProx aggregation
# -----------------------
def federated_averaging(
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_sizes: List[int],
) -> None:
    """
    Weighted averaging of client models into global_model.

    w_global = sum(n_k / N) * w_k  where N = sum(n_k)
    Both FedAvg and FedProx use the same server-side aggregation;
    the difference is only in the client-side local objective.
    """
    total_samples = sum(client_sizes)
    global_dict = global_model.state_dict()

    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
        for model, n_k in zip(client_models, client_sizes):
            weight = n_k / total_samples
            global_dict[key] += weight * model.state_dict()[key].float()

    global_model.load_state_dict(global_dict)


# -----------------------
# Main simulation loop
# -----------------------
def run_federated(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    args: argparse.Namespace,
) -> Dict:
    device = torch.device(
        "cuda" if (args.use_cuda and torch.cuda.is_available()) else
        "mps" if (args.use_mps and torch.backends.mps.is_available()) else
        "cpu"
    )

    print(f"\n{'='*60}")
    print(f"Federated Learning — {args.strategy.upper()}")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Clients: {args.num_clients}")
    print(f"Rounds: {args.rounds}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Fraction fit: {args.fraction_fit}")
    print(f"Partition: {args.partition}")
    if args.partition == "dirichlet":
        print(f"Dirichlet alpha: {args.alpha}")
    if args.strategy == "fedprox":
        print(f"Proximal mu: {args.mu}")
    print(f"Drop percent (stragglers): {args.drop_percent}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Partition training data
    if args.partition == "iid":
        client_indices = partition_iid(train_labels, args.num_clients, args.seed)
    elif args.partition == "niid":
        client_indices = partition_niid_powerlaw(
            train_labels, args.num_clients, seed=args.seed
        )
    else:
        raise ValueError(f"Unknown partition: {args.partition}")

    # Print data distribution summary
    print("Client data distribution:")
    for k in range(args.num_clients):
        idx = client_indices[k]
        counts = np.bincount(train_labels[idx], minlength=10)
        nonzero = np.where(counts > 0)[0]
        print(f"  Client {k:2d}: {len(idx):5d} samples, digits={list(nonzero)}")
    print()

    # Build client dataloaders
    client_loaders = []
    client_sizes = []
    for k in range(args.num_clients):
        idx = client_indices[k]
        loader = make_dataloader(
            train_images[idx], train_labels[idx], args.batch_size, shuffle=True
        )
        client_loaders.append(loader)
        client_sizes.append(len(idx))

    test_loader = make_dataloader(test_images, test_labels, 256, shuffle=False)

    # Initialize global model
    torch.manual_seed(args.seed)
    if args.model == "mclr":
        global_model = MCLR().to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    num_params = sum(p.numel() for p in global_model.parameters())
    print(f"Model: {args.model} ({num_params:,} parameters)\n")
  
    mu = args.mu if args.strategy == "fedprox" else 0.0

    # Tracking
    round_metrics = {
        "rounds": [],
        "global_test_loss": [],
        "global_test_accuracy": [],
        "avg_client_loss": [],
    }

    start_time = time.time()

    for rnd in range(1, args.rounds + 1):
        # Select fraction of clients
        num_selected = max(1, int(args.num_clients * args.fraction_fit))
        rng = np.random.default_rng(args.seed + rnd)
        selected = rng.choice(args.num_clients, size=num_selected, replace=False)

        # Local training
        client_models = []
        selected_sizes = []
        client_losses = []

        # Straggler simulation — both strategies train the SAME clients
        # with the SAME reduced epochs. The only difference is mu (proximal term).
        # This ensures a controlled comparison isolating FedProx's benefit.
        num_stragglers = int(len(selected) * args.drop_percent)
        straggler_set = set(rng.choice(selected, size=num_stragglers, replace=False)) if num_stragglers > 0 else set()

        for k in selected:
            is_straggler = k in straggler_set

            local_model = copy.deepcopy(global_model)

            global_params = None
            if mu > 0.0:
                global_params = [p.detach().clone() for p in local_model.parameters()]

            if is_straggler:
                local_ep = int(rng.integers(1, args.local_epochs))
            else:
                local_ep = args.local_epochs

            avg_loss = train_local(
                model=local_model,
                loader=client_loaders[k],
                device=device,
                lr=args.lr,
                epochs=local_ep,
                mu=mu,
                global_params=global_params,
                weight_decay=args.weight_decay,
            )

            if rnd <= 3 or rnd % 50 == 0:
                tag = f" [straggler, {local_ep} ep]" if is_straggler else ""
                print(f"    Client {k:2d}: loss={avg_loss:.4f}, epochs={local_ep}{tag}")

            client_models.append(local_model)
            selected_sizes.append(client_sizes[k])
            client_losses.append(avg_loss)

        # Aggregate
        federated_averaging(global_model, client_models, selected_sizes)

        # Evaluate global model
        test_loss, test_acc = evaluate(global_model, test_loader, device)
        avg_client_loss = float(np.mean(client_losses))

        round_metrics["rounds"].append(rnd)
        round_metrics["global_test_loss"].append(test_loss)
        round_metrics["global_test_accuracy"].append(test_acc)
        round_metrics["avg_client_loss"].append(avg_client_loss)

        print(
            f"  Round {rnd:3d}/{args.rounds}: "
            f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}, "
            f"Avg Client Loss={avg_client_loss:.4f}"
        )

    total_time = time.time() - start_time

    # Convergence metrics
    accs = round_metrics["global_test_accuracy"]
    convergence_90 = _convergence_round(accs, 0.90)
    convergence_95 = _convergence_round(accs, 0.95)
    convergence_98 = _convergence_round(accs, 0.98)

    final_acc = accs[-1] if accs else 0.0
    best_acc = max(accs) if accs else 0.0
    final_loss = round_metrics["global_test_loss"][-1] if accs else float("inf")

    results = {
        "experiment": {
            "strategy": args.strategy,
            "mu": args.mu if args.strategy == "fedprox" else None,
            "model": args.model,
            "num_clients": args.num_clients,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "fraction_fit": args.fraction_fit,
            "partition": args.partition,
            "alpha": args.alpha if args.partition == "dirichlet" else None,
            "drop_percent": args.drop_percent,
            "seed": args.seed,
            "device": str(device),
            "timestamp": datetime.now().isoformat(),
        },
        "per_round_metrics": round_metrics,
        "final_metrics": {
            "final_accuracy": final_acc,
            "best_accuracy": best_acc,
            "final_loss": final_loss,
            "best_loss": min(round_metrics["global_test_loss"]) if accs else float("inf"),
            "total_time_seconds": total_time,
        },
        "convergence": {
            "round_to_90_acc": convergence_90,
            "round_to_95_acc": convergence_95,
            "round_to_98_acc": convergence_98,
        },
        "client_data_distribution": {
            f"client_{k}": {
                "samples": int(client_sizes[k]),
                "digit_counts": np.bincount(train_labels[client_indices[k]], minlength=10).tolist(),
            }
            for k in range(args.num_clients)
        },
    }

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Strategy:       {args.strategy.upper()}" + (f" (mu={args.mu})" if args.strategy == "fedprox" else ""))
    print(f"Final Accuracy: {final_acc:.4f}")
    print(f"Best Accuracy:  {best_acc:.4f}")
    print(f"Final Loss:     {final_loss:.4f}")
    print(f"Convergence 90%: round {convergence_90}")
    print(f"Convergence 95%: round {convergence_95}")
    print(f"Convergence 98%: round {convergence_98}")
    print(f"Total Time:     {total_time:.2f}s")
    print(f"{'='*60}")

    return results


def _convergence_round(
    accuracies: List[float], threshold: float, window: int = 3,
) -> int:
    """Return first round where accuracy stays >= threshold for `window` rounds. -1 if never."""
    if len(accuracies) < window:
        return -1
    for i in range(len(accuracies) - window + 1):
        if all(a >= threshold for a in accuracies[i : i + window]):
            return i + 1  # 1-indexed round
    return -1


# -----------------------
# Entry point
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Federated MNIST with N clients (FedAvg / FedProx)"
    )
    ap.add_argument("--data_dir", type=str, default="./data/mnistdataset",
                    help="Path to directory with raw MNIST IDX files")
    ap.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    ap.add_argument("--model", choices=["cnn", "mclr"], default="mclr",
                    help="Model architecture: cnn (SmallCNN 1.2M) or mclr (logistic regression 7.8K)")
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--fraction_fit", type=float, default=0.5,
                    help="Fraction of clients selected each round")
    ap.add_argument("--local_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--mu", type=float, default=0.01,
                    help="FedProx proximal term coefficient (ignored for FedAvg)")
    ap.add_argument("--weight_decay", type=float, default=0.001,
                    help="L2 regularization (weight decay) for SGD")
    ap.add_argument("--drop_percent", type=float, default=0.5,
                    help="Fraction of selected clients that are stragglers (0=none, 0.5=50%%)")
    ap.add_argument("--partition", choices=["iid", "dirichlet", "shard", "niid"], default="niid",
                    help="Data partitioning strategy (niid = FedProx paper power-law)")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Dirichlet concentration parameter (lower = more non-IID)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--use_mps", action="store_true",
                    help="Use Apple MPS if available")
    ap.add_argument("--output_dir", type=str, default="./results/mnist",
                    help="Directory to save results JSON")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading MNIST from {args.data_dir} ...")
    train_images, train_labels, test_images, test_labels = load_mnist(args.data_dir)
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")

    results = run_federated(train_images, train_labels, test_images, test_labels, args)

    os.makedirs(args.output_dir, exist_ok=True)
    mu_tag = f"_mu{args.mu}" if args.strategy == "fedprox" else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{args.strategy}_n{args.num_clients}_{args.partition}"
        f"{mu_tag}_r{args.rounds}_{timestamp}_results.json"
    )
    output_path = os.path.join(args.output_dir, filename)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
