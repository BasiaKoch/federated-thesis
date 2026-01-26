#this code trains individual clients using unet 
#it trains locally on each client's data only for 50:50 and 70:30 distributions
#will be use for the comparison with federated learning
#!/usr/bin/env python3
import argparse
import time
import random
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# Utils (same as your code)
# =========================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def remap_brats_labels(y: np.ndarray) -> np.ndarray:
    y2 = y.copy()
    y2[y2 == 4] = 3
    return y2


def compute_regions_from_labels(y: torch.Tensor):
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
    y_pred = torch.argmax(logits, dim=1)
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
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[0])
        k = random.randint(0, 3)
        if k > 0:
            x = torch.rot90(x, k=k, dims=[1, 2])
            y = torch.rot90(y, k=k, dims=[0, 1])
        return x, y

    def __getitem__(self, idx: int):
        d = np.load(self.files[idx], allow_pickle=False)
        x = d["x"].astype(np.float32)
        y = d["y"].astype(np.uint8)
        y = remap_brats_labels(y).astype(np.int64)
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)
        if self.augment:
            x_t, y_t = self._augment(x_t, y_t)
        return x_t, y_t


# =========================
# Model (your UNet2D)
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
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, num_classes, 1)

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
# Loss (CE + soft dice)
# =========================
def soft_dice_loss(logits: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
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


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_d = {"WT": 0.0, "TC": 0.0, "ET": 0.0, "Mean": 0.0}
    n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long()
        logits = model(x)
        loss = combined_loss(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        d = dice_per_class_from_logits(logits, y)
        total_loss += float(loss.item())
        for k in total_d:
            total_d[k] += float(d[k])
        n += 1

    total_loss /= max(n, 1)
    for k in total_d:
        total_d[k] /= max(n, 1)
    return total_loss, total_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client_root", type=Path, required=True,
                    help="Path to client_i folder containing train/val/test")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_cuda", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.out_dir / "best_unet2d.pt"

    train_ds = BratsNPZDataset(args.client_root / "train", augment=True)
    val_ds   = BratsNPZDataset(args.client_root / "val", augment=False)
    test_ds  = BratsNPZDataset(args.client_root / "test", augment=False)

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    model = UNet2D(in_channels=4, num_classes=4, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Device: {device}")
    print(f"Client root: {args.client_root}")
    print(f"Train slices: {len(train_ds)} | Val slices: {len(val_ds)} | Test slices: {len(test_ds)}")
    print(f"Saving best to: {best_path}")

    best_val = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_d = run_epoch(model, train_loader, device, optimizer=opt)
        va_loss, va_d = run_epoch(model, val_loader, device, optimizer=None)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} meanDice {tr_d['Mean']:.4f} (WT {tr_d['WT']:.4f} TC {tr_d['TC']:.4f} ET {tr_d['ET']:.4f}) | "
            f"val loss {va_loss:.4f} meanDice {va_d['Mean']:.4f} (WT {va_d['WT']:.4f} TC {va_d['TC']:.4f} ET {va_d['ET']:.4f}) | "
            f"{dt:.1f}s"
        )

        if va_d["Mean"] > best_val:
            best_val = va_d["Mean"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_mean": best_val}, best_path)
            print(f"  -> saved new best (val meanDice={best_val:.4f})")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_d = run_epoch(model, test_loader, device, optimizer=None)

    print("\n=== Best checkpoint ===")
    print("Epoch:", ckpt["epoch"], "| val meanDice:", float(ckpt["val_mean"]))
    print("=== Test results ===")
    print(f"test loss {te_loss:.4f} | meanDice {te_d['Mean']:.4f} (WT {te_d['WT']:.4f} TC {te_d['TC']:.4f} ET {te_d['ET']:.4f})")

if __name__ == "__main__":
    main()
