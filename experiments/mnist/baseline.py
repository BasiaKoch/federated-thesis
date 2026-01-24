#!/usr/bin/env python3
"""
Baseline experiments for MNIST 2-Digit Federated Learning comparison.

This script runs two baselines:
1. Centralized: Train on ALL data combined (upper bound)
2. Local-only: Each client trains independently, no federation (lower bound)

Usage:
    python baseline_mnist_2digits.py --baseline centralized --epochs 30
    python baseline_mnist_2digits.py --baseline local --epochs 30
    python baseline_mnist_2digits.py --baseline both --epochs 30
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -----------------------
# Model (same SmallCNN as federated experiments)
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


# -----------------------
# Data partitioning (same as federated)
# -----------------------
def make_two_digit_partitions(
    y: np.ndarray,
    num_clients: int = 10,
    seed: int = 42,
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    """Create 2-digit partitions: client i gets digits (i, (i+5)%10)."""
    assert num_clients == 10
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
# Training and evaluation
# -----------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    """Train for one epoch, return (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model, return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)


# -----------------------
# Centralized baseline
# -----------------------
def run_centralized_baseline(
    trainset,
    testset,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> Dict:
    """Train a single model on ALL data (upper bound baseline)."""
    print("\n" + "=" * 60)
    print("CENTRALIZED BASELINE")
    print("=" * 60)
    print(f"Training on full dataset: {len(trainset)} samples")
    print(f"Epochs: {epochs}, LR: {lr}, Batch size: {batch_size}")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    # Track per-epoch metrics
    epoch_metrics = {
        "epochs": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, device)

        epoch_metrics["epochs"].append(epoch)
        epoch_metrics["train_loss"].append(train_loss)
        epoch_metrics["train_accuracy"].append(train_acc)
        epoch_metrics["test_loss"].append(test_loss)
        epoch_metrics["test_accuracy"].append(test_acc)

        print(f"  Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

    total_time = time.time() - start_time

    results = {
        "baseline_type": "centralized",
        "experiment": {
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "seed": seed,
            "device": str(device),
            "train_samples": len(trainset),
            "test_samples": len(testset),
            "timestamp": datetime.now().isoformat(),
        },
        "per_epoch_metrics": epoch_metrics,
        "final_metrics": {
            "final_train_accuracy": epoch_metrics["train_accuracy"][-1],
            "final_test_accuracy": epoch_metrics["test_accuracy"][-1],
            "final_train_loss": epoch_metrics["train_loss"][-1],
            "final_test_loss": epoch_metrics["test_loss"][-1],
            "best_test_accuracy": max(epoch_metrics["test_accuracy"]),
            "best_test_loss": min(epoch_metrics["test_loss"]),
            "total_time_seconds": total_time,
        },
    }

    print(f"\n--- Centralized Final Results ---")
    print(f"Final Test Accuracy: {results['final_metrics']['final_test_accuracy']:.4f}")
    print(f"Best Test Accuracy:  {results['final_metrics']['best_test_accuracy']:.4f}")
    print(f"Total Time: {total_time:.2f}s")

    return results


# -----------------------
# Local-only baseline
# -----------------------
def run_local_only_baseline(
    trainset,
    testset,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> Dict:
    """Train 10 independent models, each on their 2-digit partition (lower bound baseline)."""
    print("\n" + "=" * 60)
    print("LOCAL-ONLY BASELINE (No Federation)")
    print("=" * 60)
    print("Training 10 independent models, each on 2 digits")
    print(f"Epochs per client: {epochs}, LR: {lr}")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    y_train = np.array(trainset.targets)
    y_test = np.array(testset.targets)

    train_partitions, client_digits = make_two_digit_partitions(y_train, num_clients=10, seed=seed)
    test_partitions, _ = make_two_digit_partitions(y_test, num_clients=10, seed=seed)

    # Global test loader for evaluating how well local models generalize
    global_test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    client_results = []
    all_epoch_metrics = {
        "epochs": list(range(epochs)),
        "avg_local_test_accuracy": [],
        "avg_global_test_accuracy": [],
        "weighted_global_test_accuracy": [],
    }

    # Initialize per-epoch tracking
    per_epoch_local_acc = [[] for _ in range(epochs)]
    per_epoch_global_acc = [[] for _ in range(epochs)]
    per_epoch_weights = [[] for _ in range(epochs)]

    start_time = time.time()

    for client_id in range(10):
        print(f"\n--- Client {client_id}: digits {client_digits[client_id]} ---")

        train_subset = Subset(trainset, train_partitions[client_id])
        test_subset = Subset(testset, test_partitions[client_id])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        local_test_loader = DataLoader(test_subset, batch_size=256, shuffle=False, num_workers=0)

        model = SmallCNN().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

        client_epoch_metrics = {
            "epochs": [],
            "train_loss": [],
            "train_accuracy": [],
            "local_test_loss": [],
            "local_test_accuracy": [],
            "global_test_loss": [],
            "global_test_accuracy": [],
        }

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer)
            local_test_loss, local_test_acc = evaluate(model, local_test_loader, device)
            global_test_loss, global_test_acc = evaluate(model, global_test_loader, device)

            client_epoch_metrics["epochs"].append(epoch)
            client_epoch_metrics["train_loss"].append(train_loss)
            client_epoch_metrics["train_accuracy"].append(train_acc)
            client_epoch_metrics["local_test_loss"].append(local_test_loss)
            client_epoch_metrics["local_test_accuracy"].append(local_test_acc)
            client_epoch_metrics["global_test_loss"].append(global_test_loss)
            client_epoch_metrics["global_test_accuracy"].append(global_test_acc)

            # Track for averaging
            per_epoch_local_acc[epoch].append(local_test_acc)
            per_epoch_global_acc[epoch].append(global_test_acc)
            per_epoch_weights[epoch].append(len(train_subset))

        print(f"  Final: Local Acc={local_test_acc:.4f}, Global Acc={global_test_acc:.4f}")

        client_results.append({
            "client_id": client_id,
            "digits": list(client_digits[client_id]),
            "train_samples": len(train_subset),
            "test_samples": len(test_subset),
            "per_epoch_metrics": client_epoch_metrics,
            "final_local_test_accuracy": client_epoch_metrics["local_test_accuracy"][-1],
            "final_global_test_accuracy": client_epoch_metrics["global_test_accuracy"][-1],
        })

    total_time = time.time() - start_time

    # Compute aggregate metrics per epoch
    for epoch in range(epochs):
        avg_local = np.mean(per_epoch_local_acc[epoch])
        avg_global = np.mean(per_epoch_global_acc[epoch])
        weights = np.array(per_epoch_weights[epoch])
        weighted_global = np.average(per_epoch_global_acc[epoch], weights=weights)

        all_epoch_metrics["avg_local_test_accuracy"].append(float(avg_local))
        all_epoch_metrics["avg_global_test_accuracy"].append(float(avg_global))
        all_epoch_metrics["weighted_global_test_accuracy"].append(float(weighted_global))

    # Final aggregated metrics
    final_local_accs = [c["final_local_test_accuracy"] for c in client_results]
    final_global_accs = [c["final_global_test_accuracy"] for c in client_results]
    train_samples = [c["train_samples"] for c in client_results]

    results = {
        "baseline_type": "local_only",
        "experiment": {
            "num_clients": 10,
            "epochs_per_client": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "seed": seed,
            "device": str(device),
            "timestamp": datetime.now().isoformat(),
        },
        "per_epoch_metrics": all_epoch_metrics,
        "client_results": client_results,
        "final_metrics": {
            # Average across clients (unweighted)
            "avg_local_test_accuracy": float(np.mean(final_local_accs)),
            "avg_global_test_accuracy": float(np.mean(final_global_accs)),
            # Weighted by number of training samples
            "weighted_global_test_accuracy": float(np.average(final_global_accs, weights=train_samples)),
            # Best among clients
            "best_client_global_accuracy": float(max(final_global_accs)),
            "worst_client_global_accuracy": float(min(final_global_accs)),
            # Variance shows how different clients perform
            "client_accuracy_variance": float(np.var(final_global_accs)),
            "total_time_seconds": total_time,
        },
        "client_data_distribution": {
            f"client_{i}": {"digits": list(client_digits[i]), "samples": train_samples[i]}
            for i in range(10)
        },
    }

    print(f"\n--- Local-Only Aggregate Results ---")
    print(f"Avg Local Test Accuracy:    {results['final_metrics']['avg_local_test_accuracy']:.4f}")
    print(f"Avg Global Test Accuracy:   {results['final_metrics']['avg_global_test_accuracy']:.4f}")
    print(f"Weighted Global Accuracy:   {results['final_metrics']['weighted_global_test_accuracy']:.4f}")
    print(f"Client Accuracy Range:      [{results['final_metrics']['worst_client_global_accuracy']:.4f}, "
          f"{results['final_metrics']['best_client_global_accuracy']:.4f}]")
    print(f"Total Time: {total_time:.2f}s")

    return results


# -----------------------
# Config loading
# -----------------------
def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """Merge YAML config with command-line args. CLI args take precedence."""
    key_mapping = {
        "learning_rate": "lr",
    }

    for key, value in config.items():
        arg_key = key_mapping.get(key, key)
        if hasattr(args, arg_key):
            cli_default = getattr(args, f"_default_{arg_key}", None)
            current_val = getattr(args, arg_key)
            if current_val == cli_default and value is not None:
                setattr(args, arg_key, value)
    return args


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Baseline experiments for MNIST 2-Digit FL")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file")
    parser.add_argument("--baseline", choices=["centralized", "local", "both"], default="both",
                       help="Which baseline to run")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of epochs (comparable to FL rounds)")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--output_dir", type=str, default="./results/mnist",
                       help="Output directory for results")
    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        for key in ["baseline", "epochs", "lr", "batch_size", "seed", "output_dir"]:
            if hasattr(args, key):
                setattr(args, f"_default_{key}", parser.get_default(key))
        args = merge_config_with_args(config, args)
        if "use_cuda" in config and config["use_cuda"]:
            args.use_cuda = True

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    print(f"\n{'='*60}")
    print("MNIST 2-Digit Baseline Experiments")
    print(f"{'='*60}")
    if args.config:
        print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Baseline(s): {args.baseline}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}")

    # Load MNIST
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST(root="./data/mnist", train=True, download=True, transform=tfm)
    testset = datasets.MNIST(root="./data/mnist", train=False, download=True, transform=tfm)

    os.makedirs(args.output_dir, exist_ok=True)

    results_all = {}

    # Run centralized baseline
    if args.baseline in ["centralized", "both"]:
        centralized_results = run_centralized_baseline(
            trainset, testset, device,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed
        )
        results_all["centralized"] = centralized_results

        output_file = os.path.join(args.output_dir, f"baseline_centralized_e{args.epochs}_results.json")
        with open(output_file, "w") as f:
            json.dump(centralized_results, f, indent=2)
        print(f"\nSaved centralized results to: {output_file}")

    # Run local-only baseline
    if args.baseline in ["local", "both"]:
        local_results = run_local_only_baseline(
            trainset, testset, device,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed
        )
        results_all["local_only"] = local_results

        output_file = os.path.join(args.output_dir, f"baseline_local_e{args.epochs}_results.json")
        with open(output_file, "w") as f:
            json.dump(local_results, f, indent=2)
        print(f"\nSaved local-only results to: {output_file}")

    # Print comparison summary
    if args.baseline == "both":
        print(f"\n{'='*60}")
        print("BASELINE COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Metric':<40} {'Centralized':>15} {'Local-Only':>15}")
        print("-" * 70)
        print(f"{'Final Test Accuracy':<40} "
              f"{centralized_results['final_metrics']['final_test_accuracy']:>15.4f} "
              f"{local_results['final_metrics']['weighted_global_test_accuracy']:>15.4f}")
        print(f"{'Best Test Accuracy':<40} "
              f"{centralized_results['final_metrics']['best_test_accuracy']:>15.4f} "
              f"{local_results['final_metrics']['best_client_global_accuracy']:>15.4f}")
        print("-" * 70)
        print("\nInterpretation:")
        print(f"  - Centralized = Upper bound (best possible with all data)")
        print(f"  - Local-Only  = Lower bound (no collaboration benefit)")
        print(f"  - Gap = {centralized_results['final_metrics']['final_test_accuracy'] - local_results['final_metrics']['weighted_global_test_accuracy']:.4f}")
        print(f"\nFederated learning should fall between these bounds.")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
