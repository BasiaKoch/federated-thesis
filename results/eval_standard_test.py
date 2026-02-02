#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# COPIED from your training script (keep identical)
# -------------------------
def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(str(path), allow_pickle=False)
    img_keys = ["image", "img", "x", "X"]
    msk_keys = ["mask", "y", "Y", "seg", "label", "labels"]

    img = None
    msk = None
    for k in img_keys:
        if k in d:
            img = d[k]; break
    for k in msk_keys:
        if k in d:
            msk = d[k]; break

    if img is None or msk is None:
        keys = list(d.keys())
        arrays = [(k, d[k]) for k in keys]
        arrays.sort(key=lambda kv: (kv[1].ndim, kv[1].dtype.kind != "i"))
        for k, arr in arrays:
            if arr.ndim in (2, 3) and arr.dtype.kind in ("i", "u"):
                msk = arr; break
        for k, arr in arrays:
            if arr.dtype.kind == "f" and arr.ndim in (2, 3):
                img = arr; break

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

def _safe_div(numer: torch.Tensor, denom: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    return torch.where(denom > 0, numer / denom, torch.full_like(denom, float(default)))

import torch.nn.functional as F

def dice_coef_soft_repo_style(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = torch.sum(probs * targets, dim=dims)
    psum = torch.sum(probs, dim=dims)
    tsum = torch.sum(targets, dim=dims)
    numer = 2.0 * inter + smooth
    denom = psum + tsum + smooth
    return _safe_div(numer, denom, default=0.0)

def dice_coef_loss_repo_style(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    dice_c = dice_coef_soft_repo_style(logits, targets, smooth=smooth)
    return 1.0 - dice_c.mean()

def combined_loss_repo_style(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dloss = dice_coef_loss_repo_style(logits, targets, smooth=smooth)
    return bce + dloss

@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5):
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
        return {"loss": 0.0, "WT": 0.0, "TC": 0.0, "ET": 0.0, "MeanAll": 0.0, "MeanPresent": 0.0}

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

# -------------------------
# Your UNet2D (copied exactly from your script)
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

        return self.head(d1)

# -------------------------
# Dataset (same as yours, but takes root directly)
# -------------------------
class NPZSliceDataset(Dataset):
    def __init__(self, split_dir: Path):
        self.files = sorted([p for p in split_dir.rglob("*.npz") if p.is_file()])
        if not self.files:
            raise FileNotFoundError(f"No .npz found under {split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img, mask = _load_npz(self.files[idx])
        x = _to_chw(img)
        y = _mask_to_wt_tc_et(mask)
        return torch.from_numpy(x), torch.from_numpy(y)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True,
                    help="Standardized eval root containing client_0/test and client_1/test")
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to global_model.pt saved by your FL script")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint does not contain 'model_state_dict' key. Are you sure this is your saved global_model.pt?")
    in_ch = int(ckpt.get("in_ch", 4))
    cfg = ckpt.get("config", {})
    use_groupnorm = True
    # If you ever ran with --use_batchnorm, this will reflect it (because config stores strategy/mu etc, not norm),
    # so we default to GroupNorm unless you explicitly know otherwise.
    # You can add your own config flag here if needed.

    model = UNet2D(in_ch=in_ch, out_ch=3, base=32, use_groupnorm=use_groupnorm).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Evaluate per-client
    results = {}
    for cid in [0, 1]:
        test_dir = args.data_root / f"client_{cid}" / "test"
        ds = NPZSliceDataset(test_dir)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
        r = evaluate_model(model, dl, device, threshold=args.threshold)
        results[f"client_{cid}"] = r
        print(f"client_{cid}: n={len(ds)} MeanPresent={r['MeanPresent']:.4f} MeanAll={r['MeanAll']:.4f} "
              f"WT={r['WT']:.4f} TC={r['TC']:.4f} ET={r['ET']:.4f}")

    # Evaluate pooled (union of both clients)
    pooled_files = []
    for cid in [0, 1]:
        pooled_files.extend(sorted((args.data_root / f"client_{cid}" / "test").rglob("*.npz")))

    class Pooled(Dataset):
        def __len__(self): return len(pooled_files)
        def __getitem__(self, i):
            img, mask = _load_npz(pooled_files[i])
            return torch.from_numpy(_to_chw(img)), torch.from_numpy(_mask_to_wt_tc_et(mask))

    pooled_ds = Pooled()
    pooled_dl = DataLoader(pooled_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    rp = evaluate_model(model, pooled_dl, device, threshold=args.threshold)
    print(f"POOLED:   n={len(pooled_ds)} MeanPresent={rp['MeanPresent']:.4f} MeanAll={rp['MeanAll']:.4f} "
          f"WT={rp['WT']:.4f} TC={rp['TC']:.4f} ET={rp['ET']:.4f}")

if __name__ == "__main__":
    main()
