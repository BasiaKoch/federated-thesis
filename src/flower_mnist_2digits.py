import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
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
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(keys) != len(parameters):
        raise ValueError(f"Parameter length mismatch: {len(keys)} vs {len(parameters)}")
    new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    model.load_state_dict(new_state, strict=True)


# -----------------------
# Data partitioning: 10 clients, 2 digits each, disjoint samples
# Each client i gets digits: (i, (i+5)%10)
# This makes every digit appear in exactly 2 clients, and we split that digit's samples between those 2 clients.
# -----------------------
def make_two_digit_partitions(
    y: np.ndarray,
    num_clients: int = 10,
    seed: int = 42,
) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    assert num_clients == 10, "This partitioner assumes 10 clients for the neat (i, i+5) pairing."
    rng = np.random.default_rng(seed)

    # client -> its two digits
    client_digits: List[Tuple[int, int]] = [(i, (i + 5) % 10) for i in range(num_clients)]
    # which two clients own each digit d? -> clients: d (as first digit) and (d-5)%10 (as second digit)
    digit_owners: Dict[int, Tuple[int, int]] = {d: (d, (d - 5) % 10) for d in range(10)}

    # collect indices per digit
    indices_by_digit: Dict[int, List[int]] = {d: [] for d in range(10)}
    for idx, label in enumerate(y):
        indices_by_digit[int(label)].append(int(idx))

    # shuffle and split each digit's indices across its 2 owning clients
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for d in range(10):
        idxs = np.array(indices_by_digit[d], dtype=np.int64)
        rng.shuffle(idxs)
        c1, c2 = digit_owners[d]
        half = len(idxs) // 2
        client_indices[c1].extend(idxs[:half].tolist())
        client_indices[c2].extend(idxs[half:].tolist())

    # final shuffle within each client for good measure
    for c in range(num_clients):
        rng.shuffle(client_indices[c])

    return client_indices, client_digits


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, lr: float) -> None:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        correct += int((logits.argmax(1) == y).sum().item())
        total += int(bs)
    return total_loss / max(total, 1), correct / max(total, 1)


# -----------------------
# Flower client
# -----------------------
class MnistClient(fl.client.NumPyClient):
    def __init__(self, cid: str, train_loader: DataLoader, test_loader: DataLoader, device: torch.device, lr: float, local_epochs: int):
        self.cid = cid
        self.model = SmallCNN().to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        for _ in range(self.local_epochs):
            train_one_epoch(self.model, self.train_loader, self.device, lr=self.lr)
        return get_parameters(self.model), len(self.train_loader.dataset), {"cid": self.cid}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = evaluate(self.model, self.test_loader, self.device)
        return float(loss), len(self.test_loader.dataset), {"local_acc": float(acc)}


# -----------------------
# Metrics tracking utilities
# -----------------------
def compute_convergence_round(accuracies: List[float], threshold: float = 0.95, window: int = 3) -> int:
    """
    Find the first round where accuracy stays above threshold for `window` consecutive rounds.
    Returns -1 if never converged.
    """
    if len(accuracies) < window:
        return -1
    for i in range(len(accuracies) - window + 1):
        if all(acc >= threshold for acc in accuracies[i : i + window]):
            return i
    return -1


def compute_stability_metrics(values: List[float]) -> Dict[str, float]:
    """
    Compute stability metrics: variance, max oscillation, smoothness score.
    Lower variance and oscillation = more stable convergence.
    """
    if len(values) < 2:
        return {"variance": 0.0, "max_oscillation": 0.0, "smoothness": 1.0}

    arr = np.array(values)
    variance = float(np.var(arr))

    # Max oscillation: max absolute change between consecutive rounds
    diffs = np.abs(np.diff(arr))
    max_oscillation = float(np.max(diffs)) if len(diffs) > 0 else 0.0

    # Smoothness: 1 / (1 + mean absolute diff) - higher is smoother
    mean_diff = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    smoothness = 1.0 / (1.0 + mean_diff)

    return {
        "variance": variance,
        "max_oscillation": max_oscillation,
        "smoothness": smoothness,
        "mean_change_per_round": mean_diff,
    }


# -----------------------
# Main: build data, simulate, compare FedAvg vs FedProx
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_clients", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--fraction_fit", type=float, default=1.0)   # participate all clients each round (stable)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    ap.add_argument("--mu", type=float, default=0.01)  # used only for fedprox
    ap.add_argument("--output_dir", type=str, default="./results/flower_mnist_2digits",
                    help="Directory to save results and metrics")
    ap.add_argument("--run_name", type=str, default=None,
                    help="Optional run name for the experiment")
    args = ap.parse_args()

    if args.num_clients != 10:
        raise ValueError("This script currently implements the neat 10-client pairing (i, i+5). Use 10 clients.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate run name if not provided
    if args.run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.strategy}_rounds{args.rounds}_mu{args.mu}_{timestamp}"

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print(f"\n{'='*60}")
    print(f"Flower MNIST 2-Digit Federated Learning Experiment")
    print(f"{'='*60}")
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Proximal mu: {args.mu}" if args.strategy == "fedprox" else "Proximal mu: N/A (FedAvg)")
    print(f"Clients: {args.num_clients}")
    print(f"Rounds: {args.rounds}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {device}")
    print(f"Output dir: {args.output_dir}")
    print(f"Run name: {args.run_name}")
    print(f"{'='*60}\n")

    # MNIST transforms
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    y_train = np.array(trainset.targets)
    y_test = np.array(testset.targets)

    train_partitions, train_digits = make_two_digit_partitions(y_train, num_clients=10, seed=args.seed)
    test_partitions, _ = make_two_digit_partitions(y_test, num_clients=10, seed=args.seed)

    # Server-side (global) evaluation on full test set
    global_test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)

    # Track per-round timing
    round_start_times: Dict[int, float] = {}
    round_end_times: Dict[int, float] = {}

    def evaluate_fn(server_round: int, parameters, config):
        round_end_times[server_round] = time.time()
        model = SmallCNN().to(device)
        set_parameters(model, parameters)
        loss, acc = evaluate(model, global_test_loader, device)
        print(f"  [Round {server_round}] Global Test - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return float(loss), {"global_acc": float(acc), "global_loss": float(loss), "round": int(server_round)}

    # Build client_fn
    def client_fn(cid: str):
        c = int(cid)
        train_subset = Subset(trainset, train_partitions[c])
        test_subset = Subset(testset, test_partitions[c])

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_subset, batch_size=256, shuffle=False, num_workers=0)

        return MnistClient(cid=cid, train_loader=train_loader, test_loader=test_loader, device=device, lr=args.lr, local_epochs=args.local_epochs)

    # Choose strategy
    if args.strategy == "fedavg":
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=args.fraction_fit,
            min_fit_clients=int(args.num_clients * args.fraction_fit),
            min_available_clients=args.num_clients,
            evaluate_fn=evaluate_fn,
        )
    else:
        strategy = fl.server.strategy.FedProx(
            fraction_fit=args.fraction_fit,
            min_fit_clients=int(args.num_clients * args.fraction_fit),
            min_available_clients=args.num_clients,
            evaluate_fn=evaluate_fn,
            proximal_mu=float(args.mu),
        )

    # Run simulation with timing
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
            rounds_list.append(rnd)
            losses_list.append(loss)

    if history.metrics_centralized and "global_acc" in history.metrics_centralized:
        for rnd, acc in history.metrics_centralized["global_acc"]:
            if rnd not in rounds_list:
                rounds_list.append(rnd)
            # Match index
            idx = rounds_list.index(rnd)
            while len(accuracies_list) <= idx:
                accuracies_list.append(0.0)
            accuracies_list[idx] = acc

    # Compute analysis metrics
    convergence_90 = compute_convergence_round(accuracies_list, threshold=0.90, window=3)
    convergence_95 = compute_convergence_round(accuracies_list, threshold=0.95, window=3)
    convergence_98 = compute_convergence_round(accuracies_list, threshold=0.98, window=3)

    stability_acc = compute_stability_metrics(accuracies_list)
    stability_loss = compute_stability_metrics(losses_list)

    final_acc = accuracies_list[-1] if accuracies_list else 0.0
    final_loss = losses_list[-1] if losses_list else float("inf")
    best_acc = max(accuracies_list) if accuracies_list else 0.0
    best_loss = min(losses_list) if losses_list else float("inf")

    # Build comprehensive results dict
    results = {
        "experiment": {
            "strategy": args.strategy,
            "mu": args.mu if args.strategy == "fedprox" else None,
            "num_clients": args.num_clients,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
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
            "avg_time_per_round": total_time / args.rounds if args.rounds > 0 else 0,
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

    # Save results to JSON
    results_file = os.path.join(args.output_dir, f"{args.run_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Strategy: {args.strategy.upper()}" + (f" (mu={args.mu})" if args.strategy == "fedprox" else ""))
    print(f"\n--- Accuracy & Loss ---")
    print(f"Final Accuracy: {final_acc:.4f}")
    print(f"Best Accuracy:  {best_acc:.4f}")
    print(f"Final Loss:     {final_loss:.4f}")
    print(f"Best Loss:      {best_loss:.4f}")
    print(f"\n--- Convergence (rounds to reach threshold, -1 if never) ---")
    print(f"Rounds to 90% accuracy: {convergence_90}")
    print(f"Rounds to 95% accuracy: {convergence_95}")
    print(f"Rounds to 98% accuracy: {convergence_98}")
    print(f"\n--- Stability (accuracy) ---")
    print(f"Variance:       {stability_acc['variance']:.6f}")
    print(f"Max Oscillation:{stability_acc['max_oscillation']:.4f}")
    print(f"Smoothness:     {stability_acc['smoothness']:.4f}")
    print(f"\n--- Timing ---")
    print(f"Total Time:     {total_time:.2f}s")
    print(f"Avg per Round:  {total_time / args.rounds:.2f}s")
    print(f"{'='*60}")

    # Print client digit assignments
    print("\nClient digit pairs (train):")
    for i, pair in enumerate(train_digits):
        print(f"  client {i}: digits={pair}")


if __name__ == "__main__":
    main()
