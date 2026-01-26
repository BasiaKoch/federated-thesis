#!/usr/bin/env python3
import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# -------------------------
# Dataset: loads .npz slices from client_root/{train,val,test}/CASE/*.npz
# Robust to common key names for image/mask.
# -------------------------
def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(str(path), allow_pickle=False)

    # Try common keys
    img_keys = ["image", "img", "x", "X"]
    msk_keys = ["mask", "y", "Y", "seg", "label", "labels"]

    img = None
    msk = None

    for k in img_keys:
        if k in d:
            img = d[k]
            break
    for k in msk_keys:
        if k in d:
            msk = d[k]
            break

    # Fallback: heuristics (last resort)
    if img is None or msk is None:
        keys = list(d.keys())
        # pick 2 arrays with different dtypes/shapes; prefer 2D mask
        arrays = [(k, d[k]) for k in keys]
        # mask tends to be integer
        arrays.sort(key=lambda kv: (kv[1].ndim, kv[1].dtype.kind != "i"))
        # try select one 2D int as mask
        for k, arr in arrays:
            if arr.ndim in (2, 3) and arr.dtype.kind in ("i", "u"):
                msk = arr
                break
        # image tends to be float with 3 dims (C,H,W) or (H,W,C)
        for k, arr in arrays:
            if arr.dtype.kind == "f" and arr.ndim in (2, 3):
                img = arr
                break

    if img is None or msk is None:
        raise KeyError(f"Could not infer image/mask keys in {path}. Keys={list(d.keys())}")

    return img, msk


def _to_chw(img: np.ndarray) -> np.ndarray:
    # Accept: (H,W), (H,W,C), (C,H,W)
    if img.ndim == 2:
        img = img[None, ...]  # (1,H,W)
    elif img.ndim == 3:
        # If last dim looks like channels (<=8), assume HWC
        if img.shape[-1] <= 8 and img.shape[0] != img.shape[-1]:
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        # else assume already CHW
    else:
        raise ValueError(f"Unexpected image ndim={img.ndim}")
    return img.astype(np.float32)


def _mask_to_wt_tc_et(mask: np.ndarray) -> np.ndarray:
    """
    BraTS labels: {0,1,2,4}
    WT = {1,2,4}
    TC = {1,4}
    ET = {4}
    Output shape: (3,H,W) float32 in {0,1}
    """
    if mask.ndim == 3:
        # if (1,H,W) or (H,W,1)
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            # if already one-hot, user should adapt; we keep simplest assumption
            mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Unexpected mask shape {mask.shape}")

    m = mask.astype(np.int32)
    wt = (m > 0).astype(np.float32)
    tc = np.isin(m, [1, 4]).astype(np.float32)
    et = (m == 4).astype(np.float32)
    y = np.stack([wt, tc, et], axis=0)
    return y


class BratsNPZSliceDataset(Dataset):
    def __init__(self, split_dir: Path):
        self.files = sorted([p for p in split_dir.rglob("*.npz") if p.is_file()])
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files found under: {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img, mask = _load_npz(self.files[idx])
        x = _to_chw(img)
        y = _mask_to_wt_tc_et(mask)
        return torch.from_numpy(x), torch.from_numpy(y)


# -------------------------
# Small-ish 2D U-Net
# -------------------------
def _conv_block(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 3, base: int = 32):
        super().__init__()
        self.enc1 = _conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _conv_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = _conv_block(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)

        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        return self.head(d1)  # logits (B,3,H,W)


# -------------------------
# Loss + metrics
# -------------------------
def dice_per_channel_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    # (B,C,H,W) -> (C,)
    dims = (0, 2, 3)
    inter = torch.sum(probs * targets, dim=dims)
    denom = torch.sum(probs + targets, dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return dice


def loss_bce_dice(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_per_channel_from_logits(logits, targets)
    dice_loss = 1.0 - dice.mean()
    return bce + dice_loss


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    dices = []
    losses = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        losses.append(float(loss_bce_dice(logits, y).item()))
        dices.append(dice_per_channel_from_logits(logits, y).detach().cpu())
    d = torch.stack(dices, dim=0).mean(dim=0)  # (3,)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "WT": float(d[0].item()),
        "TC": float(d[1].item()),
        "ET": float(d[2].item()),
        "Mean": float(d.mean().item()),
    }


def train_epochs_standard(model: nn.Module, loader: DataLoader, device: torch.device, lr: float, epochs: int) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_bce_dice(logits, y)
            loss.backward()
            opt.step()


def train_epochs_fedprox(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    mu: float,
    global_params: List[torch.Tensor],
) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_bce_dice(logits, y)
            if mu > 0.0:
                prox = 0.0
                for p, p0 in zip(model.parameters(), global_params):
                    prox = prox + torch.sum((p - p0) ** 2)
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            opt.step()


# -------------------------
# Flower parameter helpers
# -------------------------
def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state = model.state_dict()
    keys = list(state.keys())
    if len(keys) != len(parameters):
        raise ValueError(f"Parameter mismatch: {len(keys)} vs {len(parameters)}")
    new_state = {}
    for k, arr in zip(keys, parameters):
        t = torch.from_numpy(arr).to(device=state[k].device, dtype=state[k].dtype)
        new_state[k] = t
    model.load_state_dict(new_state, strict=True)


# -------------------------
# Flower client (2 clients)
# -------------------------
class BratsClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        client_root: Path,
        device: torch.device,
        lr: float,
        local_epochs: int,
        batch_size: int,
        num_workers: int,
        mu: float,
    ):
        self.cid = cid
        self.client_root = client_root
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mu = mu

        # infer input channels from first sample in train split
        train_ds = BratsNPZSliceDataset(client_root / "train")
        x0, _ = train_ds[0]
        in_ch = int(x0.shape[0])

        self.model = UNet2D(in_ch=in_ch, out_ch=3, base=32).to(device)

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=(device.type == "cuda"))
        self.val_loader = DataLoader(BratsNPZSliceDataset(client_root / "val"), batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=(device.type == "cuda"))
        self.test_loader = DataLoader(BratsNPZSliceDataset(client_root / "test"), batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=(device.type == "cuda"))

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        # snapshot global params once per round
        global_params = [p.detach().clone() for p in self.model.parameters()]

        if self.mu > 0:
            train_epochs_fedprox(self.model, self.train_loader, self.device, self.lr, self.local_epochs, self.mu, global_params)
        else:
            train_epochs_standard(self.model, self.train_loader, self.device, self.lr, self.local_epochs)

        # optional: return client-val metrics for debugging
        va = evaluate_model(self.model, self.val_loader, self.device)
        return get_parameters(self.model), len(self.train_loader.dataset), {
            "cid": self.cid,
            "mu": float(self.mu),
            "val_meanDice": float(va["Mean"]),
            "val_WT": float(va["WT"]),
            "val_TC": float(va["TC"]),
            "val_ET": float(va["ET"]),
        }

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        te = evaluate_model(self.model, self.test_loader, self.device)
        # Flower expects (loss, num_examples, metrics)
        return float(te["loss"]), len(self.test_loader.dataset), {
            "test_meanDice": float(te["Mean"]),
            "test_WT": float(te["WT"]),
            "test_TC": float(te["TC"]),
            "test_ET": float(te["ET"]),
        }


# -------------------------
# Run + save results
# -------------------------
@dataclass
class RunCfg:
    strategy: str
    mu: float
    rounds: int
    local_epochs: int
    lr: float
    batch_size: int
    seed: int
    partition_dir: str
    out_dir: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition_dir", required=True, help=".../client_data (contains client_0, client_1)")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--out_dir", default="./results/unet_flower_2clients")
    args = ap.parse_args()

    # Repro
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    partition_dir = Path(args.partition_dir)

    # For FedAvg, force mu=0
    mu = float(args.mu) if args.strategy == "fedprox" else 0.0

    run_name = f"{args.strategy}_mu{mu}_R{args.rounds}_E{args.local_epochs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunCfg(
        strategy=args.strategy,
        mu=mu,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        partition_dir=str(partition_dir),
        out_dir=str(out_dir),
    )

    # pooled GLOBAL test loader (client0 test + client1 test)
    pooled_files = []
    for cid in [0, 1]:
        pooled_files.extend(sorted((partition_dir / f"client_{cid}" / "test").rglob("*.npz")))
    if len(pooled_files) == 0:
        raise FileNotFoundError("No pooled test .npz found. Check partition_dir structure.")

    class PooledTest(Dataset):
        def __len__(self): return len(pooled_files)
        def __getitem__(self, i):
            img, mask = _load_npz(pooled_files[i])
            return torch.from_numpy(_to_chw(img)), torch.from_numpy(_mask_to_wt_tc_et(mask))

    global_test_loader = DataLoader(
        PooledTest(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    def evaluate_fn(server_round: int, parameters, config):
        # Build a fresh model with correct in_ch (infer from client_0 train)
        ds0 = BratsNPZSliceDataset(partition_dir / "client_0" / "train")
        x0, _ = ds0[0]
        in_ch = int(x0.shape[0])
        model = UNet2D(in_ch=in_ch, out_ch=3, base=32).to(device)
        set_parameters(model, parameters)
        te = evaluate_model(model, global_test_loader, device)
        print(f"[Round {server_round}] GLOBAL pooled test meanDice={te['Mean']:.4f} (WT={te['WT']:.4f} TC={te['TC']:.4f} ET={te['ET']:.4f})")
        return float(te["loss"]), {
            "global_meanDice": float(te["Mean"]),
            "global_WT": float(te["WT"]),
            "global_TC": float(te["TC"]),
            "global_ET": float(te["ET"]),
        }

    def client_fn(cid: str):
        client_root = partition_dir / f"client_{cid}"
        return BratsClient(
            cid=cid,
            client_root=client_root,
            device=device,
            lr=args.lr,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mu=mu,
        )

    # 2 clients, always fit both
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        evaluate_fn=evaluate_fn,
    )

    t0 = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        # IMPORTANT: if you have only 1 GPU, requesting 1 per client forces sequential execution
        client_resources={"num_cpus": 1, "num_gpus": 1 if device.type == "cuda" else 0},
    )
    total = time.time() - t0

    # Extract centralized metrics
    rounds = []
    global_mean = []
    global_wt = []
    global_tc = []
    global_et = []

    if history.metrics_centralized:
        for key, store in history.metrics_centralized.items():
            if key == "global_meanDice":
                rounds = [int(r) for r, _ in store]
                global_mean = [float(v) for _, v in store]
            elif key == "global_WT":
                global_wt = [float(v) for _, v in store]
            elif key == "global_TC":
                global_tc = [float(v) for _, v in store]
            elif key == "global_ET":
                global_et = [float(v) for _, v in store]

    result = {
        "config": asdict(cfg),
        "timing": {"total_seconds": total, "seconds_per_round": total / max(args.rounds, 1)},
        "per_round": {
            "rounds": rounds,
            "global_meanDice": global_mean,
            "global_WT": global_wt,
            "global_TC": global_tc,
            "global_ET": global_et,
        },
        "final": {
            "last_global_meanDice": global_mean[-1] if global_mean else None,
            "best_global_meanDice": max(global_mean) if global_mean else None,
        },
    }

    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
