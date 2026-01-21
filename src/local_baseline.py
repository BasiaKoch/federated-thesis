"""
Local-Only Baseline Training for MNIST Federated Learning

This script trains 30 separate models (one per client) WITHOUT federation.
Used as a baseline to demonstrate the benefit of federated learning.

The comparison shows:
1. Local-only: each client trains on its own data only
2. FedAvg: clients collaborate via federated averaging
3. FedProx: clients collaborate with proximal term regularization

Usage:
    python local_baseline.py --data_dir ./data --output_dir ./results/local_baseline
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from model import create_model, get_model_params_count


class MNISTDataset(Dataset):
    """Custom Dataset for MNIST loaded from npz file."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: list = None):
        if indices is not None:
            self.images = images[indices]
            self.labels = labels[indices]
        else:
            self.images = images
            self.labels = labels

        # Normalize: MNIST mean=0.1307, std=0.3081
        self.images = self.images.astype(np.float32) / 255.0
        self.images = (self.images - 0.1307) / 0.3081

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST data from npz file."""
    data_path = Path(data_dir) / "mnist.npz"
    data = np.load(data_path)
    return data['x_train'], data['y_train'], data['x_test'], data['y_test']


def load_partition(data_dir: str) -> Dict:
    """Load partition file."""
    partition_path = Path(data_dir) / "partitions" / "mnist_noniid_partition.json"
    with open(partition_path, 'r') as f:
        return json.load(f)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    return total_loss / total, correct / total


def train_local_model(
    client_id: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    train_indices: List[int],
    test_indices: List[int],
    model_type: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device
) -> Dict:
    """Train a single local model for one client."""
    # Create datasets
    train_dataset = MNISTDataset(x_train, y_train, train_indices)
    local_test_dataset = MNISTDataset(x_test, y_test, test_indices)
    global_test_dataset = MNISTDataset(x_test, y_test)  # Full test set

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    local_test_loader = DataLoader(local_test_dataset, batch_size=batch_size)
    global_test_loader = DataLoader(global_test_dataset, batch_size=batch_size)

    # Create model
    model = create_model(model_type).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'local_test_loss': [],
        'local_test_acc': [],
    }

    # Train
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        local_test_loss, local_test_acc = evaluate(
            model, local_test_loader, criterion, device
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['local_test_loss'].append(local_test_loss)
        history['local_test_acc'].append(local_test_acc)

    # Final evaluation
    final_local_loss, final_local_acc = evaluate(model, local_test_loader, criterion, device)
    final_global_loss, final_global_acc = evaluate(model, global_test_loader, criterion, device)

    return {
        'client_id': client_id,
        'train_samples': len(train_indices),
        'local_test_samples': len(test_indices),
        'final_local_test_loss': final_local_loss,
        'final_local_test_acc': final_local_acc,
        'final_global_test_loss': final_global_loss,
        'final_global_test_acc': final_global_acc,
        'history': history,
        'model_state': model.state_dict()
    }


def run_local_baseline(
    data_dir: str,
    output_dir: str,
    model_type: str = "logistic",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.01
):
    """Run local-only baseline training for all clients."""
    print("=" * 60)
    print("LOCAL-ONLY BASELINE TRAINING")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model type: {model_type}")
    print(f"Epochs per client: {epochs}")

    # Load data
    print("\nLoading data...")
    x_train, y_train, x_test, y_test = load_data(data_dir)
    partition = load_partition(data_dir)

    num_clients = partition['num_clients']
    print(f"Number of clients: {num_clients}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Train each client
    results = []
    for client_id in range(num_clients):
        client_key = str(client_id)
        train_indices = partition['train_partition'][client_key]
        test_indices = partition['test_partition'][client_key]
        digits = partition['client_digits'][client_key]

        print(f"\nClient {client_id}: digits {digits}, "
              f"train={len(train_indices)}, test={len(test_indices)}")

        result = train_local_model(
            client_id=client_id,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            train_indices=train_indices,
            test_indices=test_indices,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device
        )

        # Add digit info
        result['digits'] = digits

        # Save model
        model_path = output_path / f"client_{client_id}_model.pt"
        torch.save(result['model_state'], model_path)

        # Remove model state from results (too large for JSON)
        del result['model_state']
        results.append(result)

        print(f"  Local test acc: {result['final_local_test_acc']:.4f}")
        print(f"  Global test acc: {result['final_global_test_acc']:.4f}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    local_accs = [r['final_local_test_acc'] for r in results]
    global_accs = [r['final_global_test_acc'] for r in results]

    print(f"\nLocal Test Accuracy (on client's own distribution):")
    print(f"  Mean: {np.mean(local_accs):.4f}")
    print(f"  Std:  {np.std(local_accs):.4f}")
    print(f"  Min:  {np.min(local_accs):.4f} (Client {np.argmin(local_accs)})")
    print(f"  Max:  {np.max(local_accs):.4f} (Client {np.argmax(local_accs)})")

    print(f"\nGlobal Test Accuracy (on full MNIST test set):")
    print(f"  Mean: {np.mean(global_accs):.4f}")
    print(f"  Std:  {np.std(global_accs):.4f}")
    print(f"  Min:  {np.min(global_accs):.4f} (Client {np.argmin(global_accs)})")
    print(f"  Max:  {np.max(global_accs):.4f} (Client {np.argmax(global_accs)})")

    # Save results
    summary = {
        'config': {
            'model_type': model_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'num_clients': num_clients
        },
        'summary': {
            'local_test_acc_mean': float(np.mean(local_accs)),
            'local_test_acc_std': float(np.std(local_accs)),
            'local_test_acc_min': float(np.min(local_accs)),
            'local_test_acc_max': float(np.max(local_accs)),
            'global_test_acc_mean': float(np.mean(global_accs)),
            'global_test_acc_std': float(np.std(global_accs)),
            'global_test_acc_min': float(np.min(global_accs)),
            'global_test_acc_max': float(np.max(global_accs)),
        },
        'per_client_results': results
    }

    results_path = output_path / "local_baseline_results.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train local-only baseline models for MNIST FL comparison"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data",
        help="Directory containing mnist.npz and partitions/"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/local_baseline",
        help="Directory to save results"
    )
    parser.add_argument(
        "--model_type", type=str, default="logistic",
        choices=["logistic", "cnn"],
        help="Model architecture"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs per client"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Learning rate"
    )

    args = parser.parse_args()

    run_local_baseline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == "__main__":
    main()
