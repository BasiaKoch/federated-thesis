#!/usr/bin/env python3
import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -------------------------
# Dataset helpers (SAME LOGIC as your federated script)
# -------------------------
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
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
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
# Model (SAME UNet2D)
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

        return self.head(d1)


# -------------------------
# Loss + metrics (SAME)
# -------------------------
def dice_per_channel_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = torch.sum(probs * targets, dim=dims)
    denom = torch.sum(probs + targets, dim=dims)
    return (2.0 * inter + eps) / (denom + eps)


def loss_bce_dice(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_per_channel_from_logits(logits, targets)
    return bce + (1.0 - dice.mean())


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
    d = torch.stack(dices, dim=0).mean(dim=0)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "WT": float(d[0].item()),
        "TC": float(d[1].item()),
        "ET": float(d[2].item()),
        "Mean": float(d.mean().item()),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, opt) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_bce_dice(logits, y)
        loss.backward()
        opt.step()
        running_loss += float(loss.item())
        with torch.no_grad():
            running_dice += float(dice_per_channel_from_logits(logits, y).mean().item())
        n += 1
    return {"loss": running_loss / max(n, 1), "dice": running_dice / max(n, 1)}


@dataclass
class RunCfg:
    client_id: int
    client_root: str
    epochs: int
    lr: float
    batch_size: int
    seed: int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition_dir", required=True, help=".../brats2d_one_slice_per_patient_clients")
    ap.add_argument("--client_id", type=int, choices=[0, 1], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--out_dir", default="./results/local_unet")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    client_root = Path(args.partition_dir) / f"client_{args.client_id}"
    train_dir = client_root / "train"
    val_dir = client_root / "val"
    test_dir = client_root / "test"
    for p in [train_dir, val_dir, test_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Missing split dir: {p}")

    train_ds = BratsNPZSliceDataset(train_dir)
    x0, _ = train_ds[0]
    in_ch = int(x0.shape[0])

    model = UNet2D(in_ch=in_ch, out_ch=3, base=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(BratsNPZSliceDataset(val_dir), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(BratsNPZSliceDataset(test_dir), batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    run_name = f"client{args.client_id}_E{args.epochs}_lr{args.lr}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunCfg(
        client_id=args.client_id,
        client_root=str(client_root),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print("==============================================")
    print(f"LOCAL TRAINING (no federation) client_{args.client_id}")
    print(f"client_root: {client_root}")
    print(f"train={len(train_ds)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    print(f"in_ch={in_ch} device={device}")
    print("==============================================")

    best_val = -1.0
    best_epoch = -1
    best_path = out_dir / "best_model.pt"

    t0 = time.time()
    history = {"train": [], "val": []}

    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, device, opt)
        va = evaluate_model(model, val_loader, device)

        history["train"].append({"epoch": ep, **tr})
        history["val"].append({"epoch": ep, **va})

        print(f"Epoch {ep:02d}/{args.epochs} | train loss={tr['loss']:.4f} dice={tr['dice']:.4f} "
              f"| val meanDice={va['Mean']:.4f} (WT={va['WT']:.4f} TC={va['TC']:.4f} ET={va['ET']:.4f})")

        if va["Mean"] > best_val:
            best_val = va["Mean"]
            best_epoch = ep
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_ch": in_ch,
                    "cfg": asdict(cfg),
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                },
                best_path,
            )

    total = time.time() - t0

    # Evaluate LAST model on test
    test_last = evaluate_model(model, test_loader, device)

    # Evaluate BEST-VAL model on test
    ckpt = torch.load(best_path, map_location=device)
    best_model = UNet2D(in_ch=in_ch, out_ch=3, base=32).to(device)
    best_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    test_best = evaluate_model(best_model, test_loader, device)

    result = {
        "config": asdict(cfg),
        "timing": {"total_seconds": total, "seconds_per_epoch": total / max(args.epochs, 1)},
        "best": {"best_epoch": best_epoch, "best_val_meanDice": float(best_val)},
        "test_last": test_last,
        "test_best": test_best,
        "history": history,
    }

    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    print("==============================================")
    print(f"Saved: {out_dir / 'results.json'}")
    print(f"Saved: {best_path}")
    print(f"TEST (best-val ckpt): Mean={test_best['Mean']:.4f} WT={test_best['WT']:.4f} TC={test_best['TC']:.4f} ET={test_best['ET']:.4f}")
    print(f"TEST (last epoch):    Mean={test_last['Mean']:.4f} WT={test_last['WT']:.4f} TC={test_last['TC']:.4f} ET={test_last['ET']:.4f}")
    print("==============================================")


if __name__ == "__main__":
    main()
