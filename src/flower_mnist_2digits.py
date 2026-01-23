import argparse
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
    args = ap.parse_args()

    if args.num_clients != 10:
        raise ValueError("This script currently implements the neat 10-client pairing (i, i+5). Use 10 clients.")

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

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

    def evaluate_fn(server_round: int, parameters, config):
        model = SmallCNN().to(device)
        set_parameters(model, parameters)
        loss, acc = evaluate(model, global_test_loader, device)
        # Flower logs this in History; we return (loss, metrics)
        return float(loss), {"global_acc": float(acc), "round": int(server_round)}

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

    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1 if device.type == "cuda" else 0},
    )

    # Print final global accuracy
    # history.metrics_centralized is a dict: metric_name -> list of (round, value)
    if history.metrics_centralized and "global_acc" in history.metrics_centralized:
        last_round, last_acc = history.metrics_centralized["global_acc"][-1]
        print(f"\nDONE: strategy={args.strategy} final_global_acc={last_acc:.4f} at round={last_round}")
    else:
        print("\nDONE: (No centralized metrics found—check Flower version / evaluate_fn)")

    # Print client digit assignments
    print("\nClient digit pairs (train):")
    for i, pair in enumerate(train_digits):
        print(f"  client {i}: digits={pair}")


if __name__ == "__main__":
    main()
