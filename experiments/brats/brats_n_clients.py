#!/usr/bin/env python3
"""
Federated Learning on BraTS 2D with N clients: FedAvg and FedProx.

Pure-PyTorch simulation (no Flower). Mirrors the MNIST experiment structure
(experiments/mnist/mnist_n_clients.py) but uses a 2D U-Net on pre-partitioned
BraTS .npz slices.

Usage:
    python brats_n_clients.py --strategy fedavg --rounds 30 \
        --partition_dir data/partitions/brats2d_4client_dirichlet_a0p1/client_data

    python brats_n_clients.py --strategy fedprox --mu 1.0 --rounds 30 \
        --partition_dir data/partitions/brats2d_4client_dirichlet_a0p1/client_data

    python brats_n_clients.py --strategy both --rounds 30 \
        --partition_dir data/partitions/brats2d_4client_dirichlet_a0p1/client_data
"""

import argparse
import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =========================================================================
# Dataset: loads .npz slices from client_root/{train,val,test}/CASE/*.npz
# (copied from federated_unet/unet/unet_flower_any_Nclients.py)
# =========================================================================

def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(str(path), allow_pickle=False)

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
    if img.ndim == 2:
        img = img[None, ...]
    elif img.ndim == 3:
        if img.shape[-1] <= 8 and img.shape[0] != img.shape[-1]:
            img = np.transpose(img, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected image ndim={img.ndim}")
    return img.astype(np.float32)


def _mask_to_wt_tc_et(mask: np.ndarray) -> np.ndarray:
    """BraTS labels {0,1,2,4} -> WT={1,2,4}, TC={1,4}, ET={4}. Output (3,H,W)."""
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
    return np.stack([wt, tc, et], axis=0)


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


# =========================================================================
# 2D U-Net with GroupNorm (recommended for FL)
# (copied from federated_unet/unet/unet_flower_any_Nclients.py)
# =========================================================================

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
    def __init__(self, in_ch: int, out_ch: int = 3, base: int = 32,
                 use_groupnorm: bool = True):
        super().__init__()
        self.use_groupnorm = use_groupnorm

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


# =========================================================================
# Loss functions (copied from Flower script)
# =========================================================================

def _safe_div(numer: torch.Tensor, denom: torch.Tensor,
              default: float = 0.0) -> torch.Tensor:
    return torch.where(denom > 0, numer / denom,
                       torch.full_like(denom, float(default)))


def dice_coef_soft_repo_style(
    logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = torch.sum(probs * targets, dim=dims)
    psum = torch.sum(probs, dim=dims)
    tsum = torch.sum(targets, dim=dims)
    numer = 2.0 * inter + smooth
    denom = psum + tsum + smooth
    return _safe_div(numer, denom, default=0.0)


def dice_coef_loss_repo_style(
    logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0,
) -> torch.Tensor:
    dice_c = dice_coef_soft_repo_style(logits, targets, smooth=smooth)
    return 1.0 - dice_c.mean()


def combined_loss_repo_style(
    logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dloss = dice_coef_loss_repo_style(logits, targets, smooth=smooth)
    return bce + dloss


# =========================================================================
# Evaluation (copied from Flower script)
# =========================================================================

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    inter_sum = torch.zeros(3, dtype=torch.float64)
    psum_sum = torch.zeros(3, dtype=torch.float64)
    tsum_sum = torch.zeros(3, dtype=torch.float64)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        bsz = int(x.size(0))

        logits = model(x)

        loss = combined_loss_repo_style(logits, y)
        total_loss += float(loss.item()) * bsz
        total_samples += bsz

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).to(y.dtype)

        dims = (0, 2, 3)
        inter_sum += torch.sum(preds * y, dim=dims).detach().cpu().to(torch.float64)
        psum_sum += torch.sum(preds, dim=dims).detach().cpu().to(torch.float64)
        tsum_sum += torch.sum(y, dim=dims).detach().cpu().to(torch.float64)

    if total_samples == 0:
        return {
            "loss": 0.0,
            "WT": 0.0, "TC": 0.0, "ET": 0.0,
            "MeanAll": 0.0, "MeanPresent": 0.0,
            "gt_pixels_WT": 0.0, "gt_pixels_TC": 0.0, "gt_pixels_ET": 0.0,
        }

    avg_loss = total_loss / total_samples

    numer = 2.0 * inter_sum
    denom = psum_sum + tsum_sum

    dice_all = torch.zeros(3, dtype=torch.float64)
    for c in range(3):
        if tsum_sum[c] == 0:
            dice_all[c] = 1.0 if psum_sum[c] == 0 else 0.0
        else:
            dice_all[c] = (numer[c] / denom[c]) if denom[c] > 0 else 0.0

    mean_all = float(dice_all.mean().item())

    dice_present = torch.full((3,), float("nan"), dtype=torch.float64)
    has_gt = tsum_sum > 0
    for c in range(3):
        if has_gt[c]:
            dice_present[c] = (numer[c] / denom[c]) if denom[c] > 0 else 0.0

    mean_present = float(dice_present[has_gt].mean().item()) if has_gt.any() else 0.0

    return {
        "loss": float(avg_loss),
        "WT": float(dice_all[0].item()),
        "TC": float(dice_all[1].item()),
        "ET": float(dice_all[2].item()),
        "MeanAll": mean_all,
        "MeanPresent": mean_present,
        "gt_pixels_WT": float(tsum_sum[0].item()),
        "gt_pixels_TC": float(tsum_sum[1].item()),
        "gt_pixels_ET": float(tsum_sum[2].item()),
    }


# =========================================================================
# Local training (adapted from MNIST — uses combined_loss_repo_style)
# =========================================================================

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
    Uses combined_loss_repo_style (BCE + Dice) instead of cross_entropy.
    Returns average loss (single float, like MNIST).
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0,
                                weight_decay=weight_decay)

    total_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = combined_loss_repo_style(logits, y)

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


# =========================================================================
# FedAvg / FedProx aggregation (identical to MNIST)
# =========================================================================

def federated_averaging(
    global_model: nn.Module,
    client_models: List[nn.Module],
    client_sizes: List[int],
) -> None:
    """Weighted averaging: w_global = sum(n_k / N) * w_k."""
    total_samples = sum(client_sizes)
    global_dict = global_model.state_dict()

    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
        for model, n_k in zip(client_models, client_sizes):
            weight = n_k / total_samples
            global_dict[key] += weight * model.state_dict()[key].float()

    global_model.load_state_dict(global_dict)


# =========================================================================
# Convergence helper (from MNIST)
# =========================================================================

def _convergence_round(
    values: List[float], threshold: float, window: int = 3,
) -> int:
    """Return first round where metric stays >= threshold for `window` rounds. -1 if never."""
    if len(values) < window:
        return -1
    for i in range(len(values) - window + 1):
        if all(v >= threshold for v in values[i : i + window]):
            return i + 1  # 1-indexed round
    return -1


# =========================================================================
# Main FL simulation loop (mirrors MNIST structure)
# =========================================================================

def run_federated(args: argparse.Namespace) -> Dict:
    device = torch.device(
        "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    )

    partition_dir = Path(args.partition_dir)

    # Discover clients
    client_dirs = sorted([
        d for d in partition_dir.iterdir()
        if d.is_dir() and d.name.startswith("client_")
    ])
    num_clients = len(client_dirs)
    if num_clients == 0:
        raise FileNotFoundError(
            f"No client_* directories found in {partition_dir}"
        )

    # For FedAvg, force mu=0
    mu_raw = float(args.mu) if args.strategy == "fedprox" else 0.0

    # Build train loaders + test loaders
    client_train_loaders: List[DataLoader] = []
    client_test_loaders: List[DataLoader] = []
    client_sizes: List[int] = []
    pin = device.type == "cuda"

    for cdir in client_dirs:
        train_ds = BratsNPZSliceDataset(cdir / "train")
        test_ds = BratsNPZSliceDataset(cdir / "test")
        client_train_loaders.append(
            DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=pin)
        )
        client_test_loaders.append(
            DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=pin)
        )
        client_sizes.append(len(train_ds))

    # Pooled global test loader
    pooled_files: List[Path] = []
    for cdir in client_dirs:
        pooled_files.extend(sorted((cdir / "test").rglob("*.npz")))

    class _PooledTestDS(Dataset):
        def __len__(self):
            return len(pooled_files)

        def __getitem__(self, i):
            img, mask = _load_npz(pooled_files[i])
            return torch.from_numpy(_to_chw(img)), \
                   torch.from_numpy(_mask_to_wt_tc_et(mask))

    global_test_loader = DataLoader(
        _PooledTestDS(), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )

    # Auto-detect in_ch from first sample
    x0, _ = client_train_loaders[0].dataset[0]
    in_ch = int(x0.shape[0])

    # Init global model
    torch.manual_seed(args.seed)
    global_model = UNet2D(in_ch=in_ch, out_ch=3, base=args.base,
                          use_groupnorm=True).to(device)
    num_params = sum(p.numel() for p in global_model.parameters())

    # Normalize mu by parameter count so the proximal term has comparable
    # strength to small models (MCLR 7850 params) used in the FedProx paper.
    # ||w - w^t||^2 scales linearly with d (number of params), so we divide
    # mu by (d / d_ref) to keep the per-parameter penalty constant.
    FEDPROX_REF_PARAMS = 7850  # MCLR on MNIST (FedProx paper baseline)
    if mu_raw > 0.0:
        mu = mu_raw * (FEDPROX_REF_PARAMS / num_params)
    else:
        mu = 0.0

    # Config banner
    print(f"\n{'='*60}")
    print(f"Federated Learning -- {args.strategy.upper()}")
    print(f"{'='*60}")
    print(f"Model:          UNet2D (base={args.base}, {num_params:,} params)")
    print(f"Input channels: {in_ch}")
    print(f"Clients:        {num_clients}")
    print(f"Rounds:         {args.rounds}")
    print(f"Local epochs:   {args.local_epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.lr}")
    print(f"Weight decay:   {args.weight_decay}")
    print(f"Fraction fit:   {args.fraction_fit}")
    print(f"Drop percent:   {args.drop_percent}")
    if args.strategy == "fedprox":
        print(f"Proximal mu:    {args.mu} (raw) -> {mu:.6f} (normalized for {num_params:,} params)")
    print(f"Device:         {device}")
    print(f"Partition dir:  {partition_dir}")
    print(f"{'='*60}")

    # Print client sizes
    print("\nClient data distribution:")
    for k, cdir in enumerate(client_dirs):
        train_n = len(list((cdir / "train").rglob("*.npz")))
        test_n = len(list((cdir / "test").rglob("*.npz")))
        print(f"  {cdir.name}: train={train_n}  test={test_n}")
    print()

    # Per-round tracking
    round_metrics: Dict[str, list] = {
        "rounds": [],
        "global_loss": [],
        "global_MeanPresent": [],
        "global_WT": [],
        "global_TC": [],
        "global_ET": [],
    }
    for k in range(num_clients):
        round_metrics[f"client{k}_MeanPresent"] = []
        round_metrics[f"client{k}_WT"] = []
        round_metrics[f"client{k}_TC"] = []
        round_metrics[f"client{k}_ET"] = []

    start_time = time.time()

    for rnd in range(1, args.rounds + 1):
        rnd_start = time.time()

        # Select fraction of clients
        num_selected = max(1, int(num_clients * args.fraction_fit))
        rng = np.random.default_rng(args.seed + rnd)
        selected = rng.choice(num_clients, size=num_selected, replace=False)

        # Straggler simulation (identical logic to MNIST)
        num_stragglers = int(len(selected) * args.drop_percent)
        if num_stragglers > 0:
            straggler_set = set(
                rng.choice(selected, size=num_stragglers, replace=False)
            )
        else:
            straggler_set = set()

        # Local training
        client_models: List[nn.Module] = []
        selected_sizes: List[int] = []
        client_losses: List[float] = []

        for k in selected:
            is_straggler = k in straggler_set

            local_model = copy.deepcopy(global_model)

            global_params = None
            if mu > 0.0:
                global_params = [p.detach().clone() for p in local_model.parameters()]

            if is_straggler and args.local_epochs > 1:
                local_ep = int(rng.integers(1, args.local_epochs))
            else:
                local_ep = args.local_epochs

            avg_loss = train_local(
                model=local_model,
                loader=client_train_loaders[k],
                device=device,
                lr=args.lr,
                epochs=local_ep,
                mu=mu,
                global_params=global_params,
                weight_decay=args.weight_decay,
            )

            tag = f" [straggler, {local_ep} ep]" if is_straggler else ""
            print(f"    Client {k}: loss={avg_loss:.4f}, epochs={local_ep}{tag}")

            client_models.append(local_model)
            selected_sizes.append(client_sizes[k])
            client_losses.append(avg_loss)

        # Aggregate
        federated_averaging(global_model, client_models, selected_sizes)

        # Evaluate global model on pooled test
        pooled_m = evaluate_model(global_model, global_test_loader, device)

        # Evaluate per-client test
        per_client_m: List[Dict[str, float]] = []
        for k in range(num_clients):
            cm = evaluate_model(global_model, client_test_loaders[k], device)
            per_client_m.append(cm)

        # Record metrics
        round_metrics["rounds"].append(rnd)
        round_metrics["global_loss"].append(pooled_m["loss"])
        round_metrics["global_MeanPresent"].append(pooled_m["MeanPresent"])
        round_metrics["global_WT"].append(pooled_m["WT"])
        round_metrics["global_TC"].append(pooled_m["TC"])
        round_metrics["global_ET"].append(pooled_m["ET"])

        for k in range(num_clients):
            round_metrics[f"client{k}_MeanPresent"].append(per_client_m[k]["MeanPresent"])
            round_metrics[f"client{k}_WT"].append(per_client_m[k]["WT"])
            round_metrics[f"client{k}_TC"].append(per_client_m[k]["TC"])
            round_metrics[f"client{k}_ET"].append(per_client_m[k]["ET"])

        rnd_elapsed = time.time() - rnd_start
        client_strs = " | ".join([
            f"C{k} Mean={per_client_m[k]['MeanPresent']:.4f}"
            for k in range(num_clients)
        ])
        print(
            f"  Round {rnd:3d}/{args.rounds}: "
            f"Pooled Loss={pooled_m['loss']:.4f}, "
            f"Pooled Mean={pooled_m['MeanPresent']:.4f} "
            f"(WT={pooled_m['WT']:.4f} TC={pooled_m['TC']:.4f} ET={pooled_m['ET']:.4f}) | "
            f"{client_strs} "
            f"[{rnd_elapsed:.1f}s]"
        )

    total_time = time.time() - start_time

    # Convergence metrics (on MeanPresent)
    mp_vals = round_metrics["global_MeanPresent"]
    convergence_50 = _convergence_round(mp_vals, 0.50)
    convergence_60 = _convergence_round(mp_vals, 0.60)
    convergence_70 = _convergence_round(mp_vals, 0.70)

    final_mp = mp_vals[-1] if mp_vals else 0.0
    best_mp = max(mp_vals) if mp_vals else 0.0
    final_loss = round_metrics["global_loss"][-1] if mp_vals else float("inf")

    results = {
        "experiment": {
            "strategy": args.strategy,
            "mu_raw": args.mu if args.strategy == "fedprox" else None,
            "mu_effective": mu if args.strategy == "fedprox" else None,
            "model": f"UNet2D_base{args.base}",
            "in_channels": in_ch,
            "num_params": num_params,
            "num_clients": num_clients,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "fraction_fit": args.fraction_fit,
            "drop_percent": args.drop_percent,
            "partition_dir": str(partition_dir),
            "seed": args.seed,
            "device": str(device),
            "timestamp": datetime.now().isoformat(),
        },
        "per_round_metrics": round_metrics,
        "final_metrics": {
            "global_MeanPresent": final_mp,
            "global_best_MeanPresent": best_mp,
            "global_final_loss": final_loss,
            "global_best_loss": min(round_metrics["global_loss"]) if mp_vals else float("inf"),
            "global_final_WT": round_metrics["global_WT"][-1] if mp_vals else 0.0,
            "global_final_TC": round_metrics["global_TC"][-1] if mp_vals else 0.0,
            "global_final_ET": round_metrics["global_ET"][-1] if mp_vals else 0.0,
            "total_time_seconds": total_time,
        },
        "convergence": {
            "round_to_50_MeanPresent": convergence_50,
            "round_to_60_MeanPresent": convergence_60,
            "round_to_70_MeanPresent": convergence_70,
        },
        "client_data_distribution": {
            client_dirs[k].name: {
                "train_samples": client_sizes[k],
                "test_samples": len(client_test_loaders[k].dataset),
            }
            for k in range(num_clients)
        },
    }

    # Per-client final metrics
    for k in range(num_clients):
        key_mp = f"client{k}_MeanPresent"
        if round_metrics[key_mp]:
            results["final_metrics"][f"client{k}_MeanPresent"] = round_metrics[key_mp][-1]
            results["final_metrics"][f"client{k}_best_MeanPresent"] = max(round_metrics[key_mp])

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Strategy:            {args.strategy.upper()}"
          + (f" (mu_raw={args.mu}, mu_eff={mu:.6f})" if args.strategy == "fedprox" else ""))
    print(f"Global MeanPresent:  {final_mp:.4f} (best={best_mp:.4f})")
    print(f"Global WT:           {round_metrics['global_WT'][-1]:.4f}")
    print(f"Global TC:           {round_metrics['global_TC'][-1]:.4f}")
    print(f"Global ET:           {round_metrics['global_ET'][-1]:.4f}")
    for k in range(num_clients):
        key_mp = f"client{k}_MeanPresent"
        print(f"Client {k} MeanPresent: {round_metrics[key_mp][-1]:.4f} "
              f"(best={max(round_metrics[key_mp]):.4f})")
    print(f"Convergence 50%:     round {convergence_50}")
    print(f"Convergence 60%:     round {convergence_60}")
    print(f"Convergence 70%:     round {convergence_70}")
    print(f"Total Time:          {total_time:.2f}s ({total_time/60:.1f}min)")
    print(f"{'='*60}")

    return results


# =========================================================================
# Entry point
# =========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Federated BraTS 2D with N clients (FedAvg / FedProx)"
    )
    ap.add_argument("--partition_dir", required=True,
                    help="Path to client_data dir (contains client_0, client_1, ...)")
    ap.add_argument("--strategy", choices=["fedavg", "fedprox", "both"],
                    default="fedavg")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--local_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--mu", type=float, default=1.0,
                    help="FedProx proximal term coefficient (ignored for FedAvg)")
    ap.add_argument("--weight_decay", type=float, default=0.001)
    ap.add_argument("--drop_percent", type=float, default=0.5,
                    help="Fraction of selected clients that are stragglers")
    ap.add_argument("--fraction_fit", type=float, default=0.75,
                    help="Fraction of clients selected per round")
    ap.add_argument("--base", type=int, default=32,
                    help="UNet2D base filter count")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--output_dir", type=str, default="./results/brats",
                    help="Directory to save results JSON")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.strategy == "both":
        # Run FedAvg then FedProx sequentially
        for strat in ["fedavg", "fedprox"]:
            # Reset seeds between runs
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            args_copy = argparse.Namespace(**vars(args))
            args_copy.strategy = strat

            results = run_federated(args_copy)

            mu_tag = f"_mu{args.mu}" if strat == "fedprox" else ""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"{strat}_brats_r{args.rounds}"
                f"{mu_tag}_{timestamp}_results.json"
            )
            output_path = os.path.join(args.output_dir, filename)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_path}")
    else:
        results = run_federated(args)

        mu_tag = f"_mu{args.mu}" if args.strategy == "fedprox" else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{args.strategy}_brats_r{args.rounds}"
            f"{mu_tag}_{timestamp}_results.json"
        )
        output_path = os.path.join(args.output_dir, filename)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
