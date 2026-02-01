#!/usr/bin/env python3
import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        arrays = [(k, d[k]) for k in keys]
        arrays.sort(key=lambda kv: (kv[1].ndim, kv[1].dtype.kind != "i"))
        for k, arr in arrays:
            if arr.ndim in (2, 3) and arr.dtype.kind in ("i", "u"):
                msk = arr
                break
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
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
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
# GroupNorm is often more stable than BatchNorm under non-IID FL because it has
# no running stats that become client-specific.
# -------------------------
def _conv_block(in_ch: int, out_ch: int, use_groupnorm: bool = True) -> nn.Module:
    if use_groupnorm:
        num_groups = min(32, out_ch)
        if out_ch % num_groups != 0:
            num_groups = 8 if out_ch % 8 == 0 else 4 if out_ch % 4 == 0 else 1
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class UNet2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 3, base: int = 32, use_groupnorm: bool = True):
        super().__init__()
        self.enc1 = _conv_block(in_ch, base, use_groupnorm)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _conv_block(base, base * 2, use_groupnorm)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _conv_block(base * 2, base * 4, use_groupnorm)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = _conv_block(base * 4, base * 8, use_groupnorm)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4, use_groupnorm)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2, use_groupnorm)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base, use_groupnorm)

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
# Loss + Dice metrics (FIXED)
# -------------------------
def loss_bce_dice(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Training loss: BCE + soft Dice (common and smooth for optimization).
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = torch.sum(probs * targets, dim=dims)
    denom = torch.sum(probs + targets, dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    dice_loss = 1.0 - dice.mean()
    return bce + dice_loss


def dice_macro_per_sample_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Macro Dice: compute Dice per-sample (reduce over H,W), then mean over samples.
    Returns: (C,) macro Dice over the batch.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(targets.dtype)

    # per-sample, per-channel dice: reduce only over spatial dims
    inter = torch.sum(preds * targets, dim=(2, 3))          # (B,C)
    denom = torch.sum(preds + targets, dim=(2, 3))          # (B,C)
    dice = (2.0 * inter + eps) / (denom + eps)              # (B,C)

    return dice.mean(dim=0)  # (C,)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dice_mode: str = "macro",   # "macro" or "micro"
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Evaluate with a *well-defined* Dice.

    dice_mode:
      - "macro": per-sample Dice averaged across samples (recommended for defensible reporting)
      - "micro": global Dice computed from accumulated intersections/denominators

    Loss is sample-weighted to avoid last-batch bias.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    if dice_mode == "micro":
        # accumulate globally across the whole loader (true micro)
        inter_sum = torch.zeros(3, dtype=torch.float64)
        denom_sum = torch.zeros(3, dtype=torch.float64)
    elif dice_mode == "macro":
        # accumulate per-sample macro dice (sum of per-batch macro * batch_size)
        dice_sum = torch.zeros(3, dtype=torch.float64)
    else:
        raise ValueError(f"Invalid dice_mode='{dice_mode}', expected 'macro' or 'micro'")

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bsz = int(x.size(0))

        logits = model(x)

        # sample-weighted loss
        batch_loss = float(loss_bce_dice(logits, y, eps=eps).item())
        total_loss += batch_loss * bsz
        total_samples += bsz

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).to(y.dtype)

        if dice_mode == "micro":
            # true micro: accumulate intersections/denominators across ALL pixels and ALL samples
            inter = torch.sum(preds * y, dim=(0, 2, 3)).detach().cpu().to(torch.float64)   # (C,)
            denom = torch.sum(preds + y, dim=(0, 2, 3)).detach().cpu().to(torch.float64)   # (C,)
            inter_sum += inter
            denom_sum += denom
        else:
            # macro: compute per-sample dice then average over samples in the batch
            batch_macro = dice_macro_per_sample_from_logits(
                logits, y, eps=eps, threshold=threshold
            ).detach().cpu().to(torch.float64)  # (C,)
            # weight by number of samples to make the final mean over samples
            dice_sum += batch_macro * bsz

    if total_samples == 0:
        return {"loss": 0.0, "WT": 0.0, "TC": 0.0, "ET": 0.0, "Mean": 0.0}

    avg_loss = total_loss / total_samples

    if dice_mode == "micro":
        dice_c = (2.0 * inter_sum + eps) / (denom_sum + eps)
    else:
        dice_c = dice_sum / total_samples

    return {
        "loss": float(avg_loss),
        "WT": float(dice_c[0].item()),
        "TC": float(dice_c[1].item()),
        "ET": float(dice_c[2].item()),
        "Mean": float(dice_c.mean().item()),
    }


# -----------------------
# Local Training Functions (training dice logging FIXED)
# -----------------------
def train_local_epochs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    mu: float,
    global_params: Optional[List[torch.Tensor]],
    weight_decay: float = 0.0,
    cid: str = "",
    dice_mode: str = "macro",
    dice_threshold: float = 0.5,
) -> Dict[str, List[float]]:
    """
    Unified local training for FedAvg (mu=0) and FedProx (mu>0).


    Training dice logging is computed with the SAME definition as evaluation
    (by default macro-per-sample), so it’s not misleading.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=weight_decay)

    epoch_losses: List[float] = []
    epoch_dices: List[float] = []
    epoch_prox_terms: List[float] = []

    for ep in range(epochs):
        total_loss = 0.0
        total_samples = 0
        total_prox = 0.0

        if dice_mode == "micro":
            inter_sum = torch.zeros(3, dtype=torch.float64)
            denom_sum = torch.zeros(3, dtype=torch.float64)
        elif dice_mode == "macro":
            dice_sum = torch.zeros(3, dtype=torch.float64)
        else:
            raise ValueError(f"Invalid dice_mode='{dice_mode}'")

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            bsz = int(x.size(0))

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            base_loss = loss_bce_dice(logits, y)

            prox_value = 0.0
            if mu > 0.0 and global_params is not None:
                prox = 0.0
                for p, p_global in zip(model.parameters(), global_params):
                    prox = prox + torch.sum((p - p_global) ** 2)
                prox_value = prox
                loss = base_loss + (mu / 2.0) * prox
            else:
                loss = base_loss

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * bsz
            total_samples += bsz
            total_prox += (float(prox_value.item()) if isinstance(prox_value, torch.Tensor) else float(prox_value)) * bsz

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= dice_threshold).to(y.dtype)

                if dice_mode == "micro":
                    inter = torch.sum(preds * y, dim=(0, 2, 3)).detach().cpu().to(torch.float64)
                    denom = torch.sum(preds + y, dim=(0, 2, 3)).detach().cpu().to(torch.float64)
                    inter_sum += inter
                    denom_sum += denom
                else:
                    batch_macro = dice_macro_per_sample_from_logits(
                        logits, y, threshold=dice_threshold
                    ).detach().cpu().to(torch.float64)
                    dice_sum += batch_macro * bsz

        avg_loss = total_loss / max(total_samples, 1)
        avg_prox = total_prox / max(total_samples, 1)

        if total_samples == 0:
            avg_dice_mean = 0.0
        else:
            if dice_mode == "micro":
                dice_c = (2.0 * inter_sum + 1e-6) / (denom_sum + 1e-6)
            else:
                dice_c = dice_sum / total_samples
            avg_dice_mean = float(dice_c.mean().item())

        epoch_losses.append(avg_loss)
        epoch_dices.append(avg_dice_mean)
        epoch_prox_terms.append(avg_prox)

        if mu > 0.0:
            print(f"  [Client {cid}] Epoch {ep+1}/{epochs}: loss={avg_loss:.4f}, dice={avg_dice_mean:.4f}, prox={avg_prox:.4f}")
        else:
            print(f"  [Client {cid}] Epoch {ep+1}/{epochs}: loss={avg_loss:.4f}, dice={avg_dice_mean:.4f}")

    return {"losses": epoch_losses, "dices": epoch_dices, "prox_terms": epoch_prox_terms}


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
# Flower Client
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
        weight_decay: float = 0.0,
        use_groupnorm: bool = True,
        dice_mode: str = "macro",
        dice_threshold: float = 0.5,
    ):
        self.cid = cid
        self.client_root = client_root
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mu = mu
        self.weight_decay = weight_decay
        self.use_groupnorm = use_groupnorm
        self.dice_mode = dice_mode
        self.dice_threshold = dice_threshold

        train_ds = BratsNPZSliceDataset(client_root / "train")
        x0, _ = train_ds[0]
        in_ch = int(x0.shape[0])

        self.model = UNet2D(in_ch=in_ch, out_ch=3, base=32, use_groupnorm=use_groupnorm).to(device)

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device.type == "cuda")
        )
        self.val_loader = DataLoader(
            BratsNPZSliceDataset(client_root / "val"), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda")
        )
        self.test_loader = DataLoader(
            BratsNPZSliceDataset(client_root / "test"), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda")
        )

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        global_params = None
        if self.mu > 0.0:
            global_params = [p.detach().clone() for p in self.model.parameters()]

        train_metrics = train_local_epochs(
            model=self.model,
            loader=self.train_loader,
            device=self.device,
            lr=self.lr,
            epochs=self.local_epochs,
            mu=self.mu,
            global_params=global_params,
            weight_decay=self.weight_decay,
            cid=self.cid,
            dice_mode=self.dice_mode,
            dice_threshold=self.dice_threshold,
        )

        va = evaluate_model(
            self.model, self.val_loader, self.device,
            dice_mode=self.dice_mode, threshold=self.dice_threshold
        )

        return (
            get_parameters(self.model),
            len(self.train_loader.dataset),
            {
                "cid": self.cid,
                "mu": float(self.mu),
                "final_train_loss": float(train_metrics["losses"][-1]) if train_metrics["losses"] else 0.0,
                "final_train_dice": float(train_metrics["dices"][-1]) if train_metrics["dices"] else 0.0,
                "val_meanDice": float(va["Mean"]),
                "val_WT": float(va["WT"]),
                "val_TC": float(va["TC"]),
                "val_ET": float(va["ET"]),
            },
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        te = evaluate_model(
            self.model, self.test_loader, self.device,
            dice_mode=self.dice_mode, threshold=self.dice_threshold
        )
        return (
            float(te["loss"]),
            len(self.test_loader.dataset),
            {
                "test_meanDice": float(te["Mean"]),
                "test_WT": float(te["WT"]),
                "test_TC": float(te["TC"]),
                "test_ET": float(te["ET"]),
            },
        )


# -------------------------
# Server Strategy
# -------------------------
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics


@dataclass
class RunCfg:
    strategy: str
    mu: float
    num_clients: int
    rounds: int
    local_epochs: int
    lr: float
    batch_size: int
    seed: int
    partition_dir: str
    out_dir: str
    dice_mode: str
    dice_threshold: float


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition_dir", required=True, help=".../client_data (contains client_0, client_1)")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--out_dir", default="./results/unet_flower_2clients")
    ap.add_argument("--save_model", action="store_true", default=True)
    ap.add_argument("--num_clients", type=int, default=2)
    ap.add_argument("--fraction_fit", type=float, default=1.0,
                    help="Fraction of clients to sample per round (0.0-1.0).")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--use_batchnorm", action="store_true", default=False)
    ap.add_argument("--dice_mode", choices=["macro", "micro"], default="micro",
                    help="Dice definition: macro (per-sample mean) or micro (global).")
    ap.add_argument("--dice_threshold", type=float, default=0.5,
                    help="Threshold for hard Dice computation.")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    partition_dir = Path(args.partition_dir)

    mu = float(args.mu) if args.strategy == "fedprox" else 0.0

    run_name = f"{args.strategy}_mu{mu}_R{args.rounds}_E{args.local_epochs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunCfg(
        strategy=args.strategy,
        mu=mu,
        num_clients=args.num_clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        partition_dir=str(partition_dir),
        out_dir=str(out_dir),
        dice_mode=args.dice_mode,
        dice_threshold=args.dice_threshold,
    )

    print("Client partition sanity check:")
    for cid in range(args.num_clients):
        client_dir = partition_dir / f"client_{cid}"
        train_count = len(list((client_dir / "train").rglob("*.npz")))
        val_count = len(list((client_dir / "val").rglob("*.npz")))
        test_count = len(list((client_dir / "test").rglob("*.npz")))
        print(f"  client_{cid}: train={train_count} val={val_count} test={test_count}")

    pooled_files: List[Path] = []
    for cid in range(args.num_clients):
        pooled_files.extend(sorted((partition_dir / f"client_{cid}" / "test").rglob("*.npz")))

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

    client_test_loaders: Dict[int, DataLoader] = {}
    for cid in range(args.num_clients):
        client_test_loaders[cid] = DataLoader(
            BratsNPZSliceDataset(partition_dir / f"client_{cid}" / "test"),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    use_groupnorm = not args.use_batchnorm
    print("Using GroupNorm" if use_groupnorm else "WARNING: Using BatchNorm in FL")

    def evaluate_fn(server_round: int, parameters, config):
        ds0 = BratsNPZSliceDataset(partition_dir / "client_0" / "train")
        x0, _ = ds0[0]
        in_ch = int(x0.shape[0])
        model = UNet2D(in_ch=in_ch, out_ch=3, base=32, use_groupnorm=use_groupnorm).to(device)
        set_parameters(model, parameters)

        client_metrics = {}
        for cid in range(args.num_clients):
            client_metrics[cid] = evaluate_model(
                model, client_test_loaders[cid], device,
                dice_mode=args.dice_mode, threshold=args.dice_threshold
            )

        pooled_metrics = evaluate_model(
            model, global_test_loader, device,
            dice_mode=args.dice_mode, threshold=args.dice_threshold
        )

        client_strs = " | ".join([f"Client{cid} Mean={client_metrics[cid]['Mean']:.4f}"
                                  for cid in range(args.num_clients)])
        print(f"[Round {server_round}] {client_strs} | Pooled Mean={pooled_metrics['Mean']:.4f}")

        metrics_dict = {}
        for cid in range(args.num_clients):
            metrics_dict[f"client{cid}_meanDice"] = float(client_metrics[cid]["Mean"])
            metrics_dict[f"client{cid}_WT"] = float(client_metrics[cid]["WT"])
            metrics_dict[f"client{cid}_TC"] = float(client_metrics[cid]["TC"])
            metrics_dict[f"client{cid}_ET"] = float(client_metrics[cid]["ET"])

        metrics_dict["global_meanDice"] = float(pooled_metrics["Mean"])
        metrics_dict["global_WT"] = float(pooled_metrics["WT"])
        metrics_dict["global_TC"] = float(pooled_metrics["TC"])
        metrics_dict["global_ET"] = float(pooled_metrics["ET"])

        return float(pooled_metrics["loss"]), metrics_dict

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
            weight_decay=args.weight_decay,
            use_groupnorm=use_groupnorm,
            dice_mode=args.dice_mode,
            dice_threshold=args.dice_threshold,
        )

    # FIX: use ceil for min_fit_clients
    min_fit = max(1, int(math.ceil(args.num_clients * args.fraction_fit)))

    # FIX: min_available_clients consistent with partial participation
    min_available = min_fit

    strategy = SaveModelStrategy(
        fraction_fit=args.fraction_fit,
        min_fit_clients=min_fit,
        min_available_clients=min_available,
        evaluate_fn=evaluate_fn,
    )

    t0 = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1 if device.type == "cuda" else 0},
    )
    total = time.time() - t0

    rounds: List[int] = []
    metrics_store: Dict[str, List[float]] = {}
    for cid in range(args.num_clients):
        metrics_store[f"client{cid}_meanDice"] = []
        metrics_store[f"client{cid}_WT"] = []
        metrics_store[f"client{cid}_TC"] = []
        metrics_store[f"client{cid}_ET"] = []
    metrics_store["global_meanDice"] = []
    metrics_store["global_WT"] = []
    metrics_store["global_TC"] = []
    metrics_store["global_ET"] = []

    if history.metrics_centralized:
        for key, store in history.metrics_centralized.items():
            if key in metrics_store:
                if not rounds:
                    rounds = [int(r) for r, _ in store]
                metrics_store[key] = [float(v) for _, v in store]

    final_metrics: Dict[str, Optional[float]] = {}
    for cid in range(args.num_clients):
        key = f"client{cid}_meanDice"
        final_metrics[f"client{cid}_meanDice"] = metrics_store[key][-1] if metrics_store[key] else None
        final_metrics[f"client{cid}_best_meanDice"] = max(metrics_store[key]) if metrics_store[key] else None

    final_metrics["global_meanDice"] = metrics_store["global_meanDice"][-1] if metrics_store["global_meanDice"] else None
    final_metrics["global_best_meanDice"] = max(metrics_store["global_meanDice"]) if metrics_store["global_meanDice"] else None

    result = {
        "config": asdict(cfg),
        "timing": {"total_seconds": total, "seconds_per_round": total / max(args.rounds, 1)},
        "per_round": {"rounds": rounds, **metrics_store},
        "final": final_metrics,
    }

    print("\n" + "=" * 60)
    print(f"FEDERATED TRAINING COMPLETE - {args.num_clients}-Client Results")
    print("=" * 60)
    for cid in range(args.num_clients):
        key = f"client{cid}_meanDice"
        if metrics_store[key]:
            print(f"Client {cid} (global model): Final Mean Dice = {result['final'][key]:.4f}, "
                  f"Best = {result['final'][f'client{cid}_best_meanDice']:.4f}")
    if metrics_store["global_meanDice"]:
        print(f"Pooled (global model):   Final Mean Dice = {result['final']['global_meanDice']:.4f}, "
              f"Best = {result['final']['global_best_meanDice']:.4f}")
    print("=" * 60)

    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out_dir / 'results.json'}")

    if args.save_model and strategy.final_parameters is not None:
        ds0 = BratsNPZSliceDataset(partition_dir / "client_0" / "train")
        x0, _ = ds0[0]
        in_ch = int(x0.shape[0])
        final_model = UNet2D(in_ch=in_ch, out_ch=3, base=32, use_groupnorm=use_groupnorm).to(device)

        final_weights = fl.common.parameters_to_ndarrays(strategy.final_parameters)
        set_parameters(final_model, final_weights)

        model_path = out_dir / "global_model.pt"
        torch.save({
            "model_state_dict": final_model.state_dict(),
            "config": asdict(cfg),
            "in_ch": in_ch,
            "final_metrics": result["final"],
        }, model_path)
        print(f"Saved final global model: {model_path}")

    log_path = out_dir / "training_log.txt"
    with open(log_path, "w") as f:
        f.write("Federated Learning Training Log\n")
        f.write(f"{'='*60}\n")
        f.write(f"Strategy: {cfg.strategy}\n")
        f.write(f"Mu (FedProx): {cfg.mu}\n")
        f.write(f"Num Clients: {args.num_clients}\n")
        f.write(f"Rounds: {cfg.rounds}\n")
        f.write(f"Local Epochs: {cfg.local_epochs}\n")
        f.write(f"Learning Rate: {cfg.lr}\n")
        f.write(f"Batch Size: {cfg.batch_size}\n")
        f.write(f"Seed: {cfg.seed}\n")
        f.write(f"Dice mode: {cfg.dice_mode} (threshold={cfg.dice_threshold})\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total Time: {total:.2f}s ({total/60:.2f} min)\n")
        f.write(f"Time per Round: {total/max(args.rounds,1):.2f}s\n")
        f.write(f"{'='*60}\n")
        f.write("Final Results:\n")
        for cid in range(args.num_clients):
            key = f"client{cid}_meanDice"
            if result["final"].get(key) is not None:
                f.write(f"  Client {cid} Mean Dice: {result['final'][key]:.4f}\n")
        if result["final"].get("global_meanDice") is not None:
            f.write(f"  Global Mean Dice:   {result['final']['global_meanDice']:.4f}\n")
        f.write(f"{'='*60}\n")
    print(f"Saved training log: {log_path}")


if __name__ == "__main__":
    main()
