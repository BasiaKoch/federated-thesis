#!/usr/bin/env python3
"""
Flower federated simulation for BraTS2020 2D U-Net (NPZ slices).

- Uses the same U-Net + loss + Dice metrics as your centralized script.
- Simulates 5 clients from your partition folder.
- FedAvg vs FedProx:
    * Aggregation is FedAvg in both cases
    * FedProx difference is client-side proximal term (mu)

Config usage (YAML like MNIST):
    python -u unet_flower_train.py --config path/to/unet_fedavg.yaml
    python -u unet_flower_train.py --config path/to/unet_fedprox.yaml

Expected partition layout:
  partitions_dir/
    client_0/train/<case_id>/*.npz
    client_1/train/<case_id>/*.npz
    ...
"""

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
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader


# =========================
# Dataset + metrics utils
# =========================

def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def remap_brats_labels(y: np.ndarray) -> np.ndarray:
    """Map BraTS labels {0,1,2,4} -> {0,1,2,3} (4 -> 3)."""
    y2 = y.copy()
    y2[y2 == 4] = 3
    return y2


def compute_regions_from_labels(y: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    y: (H,W) int64 with values {0,1,2,3}
    Regions:
      WT = 1 or 2 or 3
      TC = 1 or 3
      ET = 3
    """
    wt = (y > 0)
    tc = (y == 1) | (y == 3)
    et = (y == 3)
    return {"WT": wt, "TC": tc, "ET": et}


def dice_binary(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = pred.bool()
    target = target.bool()
    inter = (pred & target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    return float((2.0 * inter + eps) / (denom + eps))


@torch.no_grad()
def dice_per_class_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    """
    logits: (B,C,H,W)
    y_true: (B,H,W) int64
    Uses argmax prediction, reports WT/TC/ET + Mean.
    """
    y_pred = torch.argmax(logits, dim=1)  # (B,H,W)
    dices = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
    b = y_true.shape[0]
    for i in range(b):
        regs_t = compute_regions_from_labels(y_true[i])
        regs_p = compute_regions_from_labels(y_pred[i])
        for k in dices:
            dices[k] += dice_binary(regs_p[k], regs_t[k])
    for k in dices:
        dices[k] /= max(b, 1)
    dices["Mean"] = (dices["WT"] + dices["TC"] + dices["ET"]) / 3.0
    return dices


class BratsNPZDataset(Dataset):
    """
    Reads .npz files saved as:
      x: (4,H,W) float32
      y: (H,W) uint8 labels {0,1,2,4} -> remapped to {0,1,2,3}
    """
    def __init__(self, split_dir: Path, augment: bool = False):
        self.split_dir = split_dir
        self.augment = augment
        self.files = sorted(split_dir.rglob("*.npz"))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found under: {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _augment(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (C,H,W), y: (H,W)
        if np.random.rand() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
        if np.random.rand() < 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[0])
        k = int(np.random.randint(0, 4))
        if k > 0:
            x = torch.rot90(x, k=k, dims=[1, 2])
            y = torch.rot90(y, k=k, dims=[0, 1])
        return x, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.files[idx]
        d = np.load(p, allow_pickle=False)
        x = d["x"].astype(np.float32)
        y = d["y"].astype(np.uint8)
        y = remap_brats_labels(y).astype(np.int64)
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        if self.augment:
            x_t, y_t = self._augment(x_t, y_t)
        return x_t, y_t


# =========================
# U-Net (same as centralized)
# =========================

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 4, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# =========================
# Loss (same as centralized)
# =========================

def soft_dice_loss(logits: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)  # (B,C,H,W)
    b, c, h, w = probs.shape
    y_onehot = F.one_hot(y_true, num_classes=c).permute(0, 3, 1, 2).float()
    probs_fg = probs[:, 1:, :, :]
    y_fg = y_onehot[:, 1:, :, :]
    dims = (0, 2, 3)
    inter = torch.sum(probs_fg * y_fg, dims)
    denom = torch.sum(probs_fg + y_fg, dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def combined_loss(logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y_true) + soft_dice_loss(logits, y_true)


# =========================
# Flower parameter helpers
# =========================

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(keys) != len(parameters):
        raise ValueError(f"Parameter length mismatch: {len(keys)} vs {len(parameters)}")

    new_state = {}
    for k, arr in zip(keys, parameters):
        old_t = state_dict[k]
        t = torch.from_numpy(arr).to(device=old_t.device, dtype=old_t.dtype)
        new_state[k] = t

    model.load_state_dict(new_state, strict=True)


def _get_mu_from_config(config: Dict[str, Any]) -> float:
    for k in ("proximal-mu", "proximal_mu", "mu"):
        if k in config:
            try:
                return float(config[k])
            except Exception:
                pass
    return 0.0


# =========================
# Train / eval loops
# =========================

def train_one_epoch_standard(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_d = {"WT": 0.0, "TC": 0.0, "ET": 0.0, "Mean": 0.0}
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = combined_loss(logits, y)
        loss.backward()
        optimizer.step()

        d = dice_per_class_from_logits(logits, y)
        total_loss += float(loss.item())
        for k in total_d:
            total_d[k] += float(d[k])
        n_batches += 1

    total_loss /= max(n_batches, 1)
    for k in total_d:
        total_d[k] /= max(n_batches, 1)

    return total_loss, total_d


def train_one_epoch_fedprox(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    mu: float,
    global_params: List[torch.Tensor],
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_d = {"WT": 0.0, "TC": 0.0, "ET": 0.0, "Mean": 0.0}
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = combined_loss(logits, y)

        # proximal term
        prox = 0.0
        for p, p0 in zip(model.parameters(), global_params):
            prox = prox + torch.sum((p - p0) ** 2)
        loss = loss + (mu / 2.0) * prox

        loss.backward()
        optimizer.step()

        d = dice_per_class_from_logits(logits, y)
        total_loss += float(loss.item())
        for k in total_d:
            total_d[k] += float(d[k])
        n_batches += 1

    total_loss /= max(n_batches, 1)
    for k in total_d:
        total_d[k] /= max(n_batches, 1)

    return total_loss, total_d


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_d = {"WT": 0.0, "TC": 0.0, "ET": 0.0, "Mean": 0.0}
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long()
        logits = model(x)
        loss = combined_loss(logits, y)

        d = dice_per_class_from_logits(logits, y)
        total_loss += float(loss.item())
        for k in total_d:
            total_d[k] += float(d[k])
        n_batches += 1

    total_loss /= max(n_batches, 1)
    for k in total_d:
        total_d[k] /= max(n_batches, 1)

    return total_loss, total_d


# =========================
# Flower client
# =========================

class UNetClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        train_dir: Path,
        device: torch.device,
        lr: float,
        batch_size: int,
        local_epochs: int,
        num_workers: int,
        force_mu: Optional[float] = None,
    ):
        self.cid = cid
        self.device = device
        self.model = UNet2D(in_channels=4, num_classes=4, base=32).to(device)

        pin_memory = (device.type == "cuda")
        self.train_ds = BratsNPZDataset(train_dir, augment=True)
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.lr = lr
        self.local_epochs = local_epochs
        self.force_mu = force_mu

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        mu = float(self.force_mu) if self.force_mu is not None else _get_mu_from_config(config)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        # snapshot global params once per round (FedProx reference)
        global_params = [p.detach().clone() for p in self.model.parameters()]

        last_loss, last_d = 0.0, {"Mean": 0.0, "WT": 0.0, "TC": 0.0, "ET": 0.0}
        for _ in range(self.local_epochs):
            if mu > 0.0:
                last_loss, last_d = train_one_epoch_fedprox(
                    self.model, self.train_loader, self.device, opt, mu, global_params
                )
            else:
                last_loss, last_d = train_one_epoch_standard(
                    self.model, self.train_loader, self.device, opt
                )

        return get_parameters(self.model), len(self.train_ds), {
            "cid": self.cid,
            "mu": float(mu),
            "train_loss": float(last_loss),
            "train_meanDice": float(last_d["Mean"]),
        }

    def evaluate(self, parameters, config):
        # Keep it simple; global eval is done on the server each round.
        set_parameters(self.model, parameters)
        return 0.0, 0, {}


# =========================
# Strategy that keeps latest parameters (for final test eval)
# =========================

class FedAvgKeepParams(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters: Optional[fl.common.Parameters] = None

    def aggregate_fit(self, server_round, results, failures):
        out = super().aggregate_fit(server_round, results, failures)
        if out is not None:
            params, metrics = out
            self.latest_parameters = params
        return out


# =========================
# Config loading (MNIST-style, clean)
# =========================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", type=str, default=None, help="Path to YAML config")

    # Data
    ap.add_argument("--partitions_dir", type=str,
                    default="/home/bk489/federated/federated-thesis/data/partitions/federated_clients_5_lgg_hgg/client_data",
                    help="Folder containing client_0/train/... client_1/train/... etc.")
    ap.add_argument("--data_root", type=str,
                    default="/home/bk489/federated/federated-thesis/data/brats2020_top10_slices_split_npz",
                    help="Global dataset root containing val/ and test/ for centralized evaluation")

    # FL
    ap.add_argument("--num_clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--fraction_fit", type=float, default=1.0)
    ap.add_argument("--local_epochs", type=int, default=1)

    # Optim
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)

    # Strategy
    ap.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    ap.add_argument("--mu", type=float, default=0.01)

    # Runtime
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cuda", action="store_true")

    # Eval/output
    ap.add_argument("--eval_split", choices=["val", "test"], default="val",
                    help="Which split to evaluate each round (centralized).")
    ap.add_argument("--output_dir", type=str,
                    default="/home/bk489/federated/federated-thesis/results/unet_flower")
    ap.add_argument("--run_name", type=str, default=None)

    return ap


def parse_args_with_yaml() -> argparse.Namespace:
    ap = build_parser()

    # Stage 1: read --config only
    args0, _ = ap.parse_known_args()

    # If config is provided, load it and set defaults before final parse
    if args0.config:
        cfg_path = Path(args0.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        if not isinstance(cfg, dict):
            raise ValueError("YAML config must be a mapping/dict at top level")

        # Set parser defaults from YAML
        ap.set_defaults(**cfg)

    # Stage 2: parse full args (CLI overrides YAML defaults)
    args = ap.parse_args()
    return args


# =========================
# Main
# =========================

def main() -> None:
    args = parse_args_with_yaml()

    seed_everything(args.seed)

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mu_tag = f"mu{args.mu}" if args.strategy == "fedprox" else "muNA"
        args.run_name = f"{args.strategy}_clients{args.num_clients}_rounds{args.rounds}_{mu_tag}_{ts}"

    print("\n" + "=" * 70)
    print("Flower BraTS 2D U-Net Federated Simulation")
    print("=" * 70)
    if args.config:
        print("Config:", args.config)
    print("Partitions:", args.partitions_dir)
    print("Global data:", args.data_root)
    print("Strategy:", args.strategy.upper(), (f"(mu={args.mu})" if args.strategy == "fedprox" else ""))
    print("Clients:", args.num_clients, "| Rounds:", args.rounds, "| Fraction fit:", args.fraction_fit)
    print("Local epochs:", args.local_epochs, "| Batch:", args.batch_size, "| LR:", args.lr)
    print("Device:", device)
    print("Eval split per round:", args.eval_split)
    print("Output:", out_dir)
    print("Run name:", args.run_name)
    print("=" * 70 + "\n")

    # Centralized evaluation loader
    data_root = Path(args.data_root)
    eval_dir = data_root / args.eval_split
    pin_memory = (device.type == "cuda")
    eval_ds = BratsNPZDataset(eval_dir, augment=False)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    print(f"Centralized {args.eval_split} slices: {len(eval_ds)}")

    def evaluate_fn(server_round: int, parameters: List[np.ndarray], config: Dict[str, Any]):
        model = UNet2D(in_channels=4, num_classes=4, base=32).to(device)
        set_parameters(model, parameters)
        loss, d = evaluate_model(model, eval_loader, device)
        print(
            f"[Round {server_round:02d}] global-{args.eval_split} "
            f"loss {loss:.4f} meanDice {d['Mean']:.4f} "
            f"(WT {d['WT']:.4f} TC {d['TC']:.4f} ET {d['ET']:.4f})"
        )
        return float(loss), {
            f"global_{args.eval_split}_loss": float(loss),
            f"global_{args.eval_split}_meanDice": float(d["Mean"]),
            f"global_{args.eval_split}_WT": float(d["WT"]),
            f"global_{args.eval_split}_TC": float(d["TC"]),
            f"global_{args.eval_split}_ET": float(d["ET"]),
        }

    def fit_config_fn(server_round: int) -> Dict[str, float]:
        if args.strategy == "fedprox":
            return {"proximal-mu": float(args.mu), "proximal_mu": float(args.mu), "mu": float(args.mu)}
        return {"proximal-mu": 0.0, "proximal_mu": 0.0, "mu": 0.0}

    partitions = Path(args.partitions_dir)

    # Flower newer API prefers client_fn(Context). We'll support both.
    try:
        from flwr.common import Context
    except Exception:
        Context = None  # type: ignore

    def _train_dir_for_cid(cid_str: str) -> Path:
        return partitions / f"client_{cid_str}" / "train"

    def client_fn_legacy(cid: str):
        train_dir = _train_dir_for_cid(cid)
        force_mu = float(args.mu) if args.strategy == "fedprox" else None
        return UNetClient(
            cid=cid,
            train_dir=train_dir,
            device=device,
            lr=args.lr,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            num_workers=args.num_workers,
            force_mu=force_mu,
        )

    def client_fn_context(context):
        # Most Flower versions provide partition id in node_config["partition-id"]
        cid = None
        if hasattr(context, "node_config") and isinstance(context.node_config, dict):
            if "partition-id" in context.node_config:
                cid = str(context.node_config["partition-id"])
        if cid is None and hasattr(context, "cid"):
            cid = str(context.cid)
        if cid is None:
            cid = "0"
        return client_fn_legacy(cid)

    client_fn = client_fn_context if Context is not None else client_fn_legacy

    min_fit = max(1, int(args.num_clients * args.fraction_fit))

    strategy = FedAvgKeepParams(
        fraction_fit=float(args.fraction_fit),
        min_fit_clients=min_fit,
        min_available_clients=int(args.num_clients),
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=fit_config_fn,
    )

    start_time = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=int(args.num_clients),
        config=fl.server.ServerConfig(num_rounds=int(args.rounds)),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1 if device.type == "cuda" else 0},
    )
    total_time = time.time() - start_time

    # Prepare results payload (MNIST-like)
    results = {
        "experiment": {
            "config_file": args.config,
            "run_name": args.run_name,
            "strategy": args.strategy,
            "mu": float(args.mu) if args.strategy == "fedprox" else None,
            "num_clients": int(args.num_clients),
            "rounds": int(args.rounds),
            "fraction_fit": float(args.fraction_fit),
            "local_epochs": int(args.local_epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "num_workers": int(args.num_workers),
            "seed": int(args.seed),
            "device": str(device),
            "eval_split": args.eval_split,
            "partitions_dir": str(args.partitions_dir),
            "data_root": str(args.data_root),
            "total_time_sec": float(total_time),
            "timestamp": datetime.now().isoformat(),
        },
        "history": {
            "losses_centralized": [(int(r), float(l)) for r, l in (history.losses_centralized or [])],
            "metrics_centralized": {
                k: [(int(r), float(v)) for r, v in vals]
                for k, vals in (history.metrics_centralized or {}).items()
            },
        },
    }

    # Final test evaluation (always on "test" split)
    if strategy.latest_parameters is not None:
        test_dir = Path(args.data_root) / "test"
        test_ds = BratsNPZDataset(test_dir, augment=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        model = UNet2D(in_channels=4, num_classes=4, base=32).to(device)
        nds = fl.common.parameters_to_ndarrays(strategy.latest_parameters)
        set_parameters(model, nds)
        te_loss, te_d = evaluate_model(model, test_loader, device)
        results["final_test"] = {
            "loss": float(te_loss),
            "meanDice": float(te_d["Mean"]),
            "WT": float(te_d["WT"]),
            "TC": float(te_d["TC"]),
            "ET": float(te_d["ET"]),
            "n_slices": int(len(test_ds)),
        }

    out_path = out_dir / f"{args.run_name}_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to: {out_path}")
    print(f"Total simulation time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
