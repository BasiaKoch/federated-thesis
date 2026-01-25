import os
import math
import time
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# Config (edit these)
# =========================

DATA_ROOT = Path(os.environ.get(
    "BRATS_DATA_DIR",
    "/home/bk489/federated/federated-thesis/data/brats2020_top10_slices_split_npz"
))
# Output
RUN_DIR = Path("./runs_unet_brats2d")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = 42

# Training
EPOCHS = 30
BATCH_SIZE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0

# Classes
# After remap: 0=bg, 1=NCR/NET, 2=ED, 3=ET
NUM_CLASSES = 4


# =========================
# Utils
# =========================

def seed_everything(seed: int) -> None:
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
    Returns region masks (H,W) bool:
      WT (whole tumor) = 1 or 2 or 3
      TC (tumor core)  = 1 or 3
      ET (enhancing)   = 3
    """
    wt = (y > 0)
    tc = (y == 1) | (y == 3)
    et = (y == 3)
    return {"WT": wt, "TC": tc, "ET": et}


def dice_binary(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """
    pred/target: (H,W) bool or {0,1} tensors
    """
    pred = pred.bool()
    target = target.bool()
    inter = (pred & target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    return float((2.0 * inter + eps) / (denom + eps))


def dice_per_class_from_logits(
    logits: torch.Tensor, y_true: torch.Tensor
) -> Dict[str, float]:
    """
    logits: (B,C,H,W)
    y_true: (B,H,W) int64
    Computes Dice for BraTS regions WT/TC/ET using argmax prediction.
    """
    with torch.no_grad():
        y_pred = torch.argmax(logits, dim=1)  # (B,H,W)

        dices = {"WT": 0.0, "TC": 0.0, "ET": 0.0}
        b = y_true.shape[0]
        for i in range(b):
            regs_t = compute_regions_from_labels(y_true[i])
            regs_p = compute_regions_from_labels(y_pred[i])
            for k in dices.keys():
                dices[k] += dice_binary(regs_p[k], regs_t[k])
        for k in dices.keys():
            dices[k] /= max(b, 1)
        dices["Mean"] = (dices["WT"] + dices["TC"] + dices["ET"]) / 3.0
        return dices


# =========================
# Dataset
# =========================

class BratsNPZDataset(Dataset):
    def __init__(self, split_dir: Path, augment: bool = False):
        self.split_dir = split_dir
        self.augment = augment
        self.files = sorted(split_dir.rglob("*.npz"))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found under: {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _augment(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (C,H,W), y: (H,W)
        Simple spatial augments (same transform for x and y).
        """
        # random horizontal flip
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
        # random vertical flip
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[0])
        # random 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            x = torch.rot90(x, k=k, dims=[1, 2])
            y = torch.rot90(y, k=k, dims=[0, 1])
        return x, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.files[idx]
        d = np.load(p, allow_pickle=False)

        x = d["x"].astype(np.float32)  # (4,H,W)
        y = d["y"].astype(np.uint8)    # (H,W) labels {0,1,2,4}

        y = remap_brats_labels(y).astype(np.int64)

        x_t = torch.from_numpy(x)  # float32
        y_t = torch.from_numpy(y)  # int64

        if self.augment:
            x_t, y_t = self._augment(x_t, y_t)

        return x_t, y_t


# =========================
# Model (2D U-Net)
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
# Loss (CE + Soft Dice)
# =========================

def soft_dice_loss(logits: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Multi-class soft dice over classes 1..C-1 (exclude background).
    logits: (B,C,H,W)
    y_true: (B,H,W) int64
    """
    probs = torch.softmax(logits, dim=1)  # (B,C,H,W)
    b, c, h, w = probs.shape

    y_onehot = F.one_hot(y_true, num_classes=c).permute(0, 3, 1, 2).float()  # (B,C,H,W)

    # exclude background channel 0
    probs_fg = probs[:, 1:, :, :]
    y_fg = y_onehot[:, 1:, :, :]

    dims = (0, 2, 3)
    inter = torch.sum(probs_fg * y_fg, dims)
    denom = torch.sum(probs_fg + y_fg, dims)
    dice = (2.0 * inter + eps) / (denom + eps)

    return 1.0 - dice.mean()


class CELossWithIgnore(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, y_true):
        return self.ce(logits, y_true)


# =========================
# Train / Eval
# =========================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Tuple[float, Dict[str, float]]:
    """
    If optimizer is provided -> train mode, else eval mode.
    Returns: avg_loss, avg_dices (WT/TC/ET/Mean)
    """
    is_train = optimizer is not None
    model.train(is_train)

    ce_loss_fn = CELossWithIgnore().to(device)

    total_loss = 0.0
    total_d = {"WT": 0.0, "TC": 0.0, "ET": 0.0, "Mean": 0.0}
    n_batches = 0

    for x, y in loader:
        x = x.to(device)              # (B,4,H,W)
        y = y.to(device).long()       # (B,H,W)

        logits = model(x)

        loss_ce = ce_loss_fn(logits, y)
        loss_dice = soft_dice_loss(logits, y)
        loss = loss_ce + loss_dice

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        d = dice_per_class_from_logits(logits, y)

        total_loss += float(loss.item())
        for k in total_d.keys():
            total_d[k] += float(d[k])
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    for k in total_d.keys():
        total_d[k] /= max(n_batches, 1)
    return avg_loss, total_d


def main():
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = BratsNPZDataset(DATA_ROOT / "train", augment=True)
    val_ds   = BratsNPZDataset(DATA_ROOT / "val", augment=False)
    test_ds  = BratsNPZDataset(DATA_ROOT / "test", augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = UNet2D(in_channels=4, num_classes=NUM_CLASSES, base=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_mean = -1.0
    best_path = RUN_DIR / "best_unet2d.pt"

    print(f"Train slices: {len(train_ds)} | Val slices: {len(val_ds)} | Test slices: {len(test_ds)}")
    print("Saving best model to:", best_path)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_d = run_epoch(model, train_loader, device, optimizer=optimizer)
        va_loss, va_d = run_epoch(model, val_loader, device, optimizer=None)

        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train loss {tr_loss:.4f} meanDice {tr_d['Mean']:.4f} (WT {tr_d['WT']:.4f} TC {tr_d['TC']:.4f} ET {tr_d['ET']:.4f}) | "
            f"val loss {va_loss:.4f} meanDice {va_d['Mean']:.4f} (WT {va_d['WT']:.4f} TC {va_d['TC']:.4f} ET {va_d['ET']:.4f}) | "
            f"{dt:.1f}s"
        )

        # Save best on val mean dice
        if va_d["Mean"] > best_val_mean:
            best_val_mean = va_d["Mean"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_mean": best_val_mean}, best_path)
            print(f"  -> saved new best (val meanDice={best_val_mean:.4f})")

    # Test evaluation with best model
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_d = run_epoch(model, test_loader, device, optimizer=None)

    print("\n=== Best checkpoint ===")
    print("Epoch:", ckpt["epoch"], "| val meanDice:", float(ckpt["val_mean"]))
    print("=== Test results ===")
    print(f"test loss {te_loss:.4f} | meanDice {te_d['Mean']:.4f} (WT {te_d['WT']:.4f} TC {te_d['TC']:.4f} ET {te_d['ET']:.4f})")


if __name__ == "__main__":
    main()
