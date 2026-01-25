#!/usr/bin/env python3
from __future__ import annotations

import re
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Basics / reproducibility
# -------------------------
MODALITIES = ["flair", "t1", "t1ce", "t2"]
SLICE_RE = re.compile(r"_slice(\d{4})\.png$", re.IGNORECASE)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def find_case_dirs(split_root: Path) -> List[Path]:
    return sorted([p for p in split_root.iterdir() if p.is_dir() and p.name.startswith("BraTS20_Training_")])

def folder_for_suffix(case_dir: Path, suffix: str) -> Optional[Path]:
    """Find a subfolder inside case_dir that ends with _{suffix} (case-insensitive)."""
    sfx = f"_{suffix}".lower()
    for p in case_dir.iterdir():
        if p.is_dir() and p.name.lower().endswith(sfx):
            return p
    return None

def load_png_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)

def resize_img(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), resample=Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)

def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape[0] == size and mask.shape[1] == size:
        return mask
    pil = Image.fromarray(mask)
    pil = pil.resize((size, size), resample=Image.NEAREST)
    return np.array(pil, dtype=np.uint8)

def remap_brats_mask(mask_u8: np.ndarray) -> np.ndarray:
    """
    BraTS labels are typically {0,1,2,4}. We remap 4 -> 3 so classes become {0,1,2,3}.
    """
    m = mask_u8.astype(np.uint8).copy()
    m[m == 4] = 3
    return m


# -------------------------
# Indexing your slice structure
# -------------------------
@dataclass(frozen=True)
class SampleRef:
    case_id: str
    slice_idx: int
    mod_paths: Dict[str, Path]   # flair/t1/t1ce/t2 -> png slice path
    seg_path: Path               # seg png slice path

def build_samples(
    split_root: Path,
    image_size: int,
    min_fg: int = 1,
    keep_empty_prob: float = 0.1,
    seed: int = 42,
) -> List[SampleRef]:
    """
    Build slice samples by looking at seg slices and matching modalities by slice index.

    This can *downsample* empty-mask slices via keep_empty_prob.
    For val/test, keep_empty_prob=1.0 keeps all slices, but metrics below will IGNORE
    tumor-absent targets so they don't inflate Dice.
    """
    rng = random.Random(seed)
    samples: List[SampleRef] = []

    for case_dir in find_case_dirs(split_root):
        case_id = case_dir.name
        seg_dir = folder_for_suffix(case_dir, "seg")
        if seg_dir is None:
            continue

        mod_dirs = {m: folder_for_suffix(case_dir, m) for m in MODALITIES}
        if any(mod_dirs[m] is None for m in MODALITIES):
            continue

        seg_files = sorted(seg_dir.glob("*.png"))
        for seg_path in seg_files:
            m = SLICE_RE.search(seg_path.name)
            if not m:
                continue
            slice_idx = int(m.group(1))

            mod_paths: Dict[str, Path] = {}
            ok = True
            for mod in MODALITIES:
                d = mod_dirs[mod]
                assert d is not None
                hits = list(d.glob(f"*slice{slice_idx:04d}.png"))
                if not hits:
                    ok = False
                    break
                mod_paths[mod] = hits[0]
            if not ok:
                continue

            seg = load_png_gray(seg_path)
            seg = resize_mask(seg, image_size)
            seg = remap_brats_mask(seg)
            fg = int((seg > 0).sum())

            # downsample empty-ish slices if desired
            if fg < min_fg and rng.random() > keep_empty_prob:
                continue

            samples.append(SampleRef(case_id, slice_idx, mod_paths, seg_path))

    return samples


# -------------------------
# Dataset + light augmentation
# -------------------------
class Brats2DDataset(Dataset):
    def __init__(self, samples: List[SampleRef], image_size: int, augment: bool):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal
            y = torch.flip(y, dims=[1])
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[1])  # vertical
            y = torch.flip(y, dims=[0])

        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            x = torch.rot90(x, k, dims=[1, 2])
            y = torch.rot90(y, k, dims=[0, 1])

        return x, y

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        chans = []
        for mod in MODALITIES:
            img = load_png_gray(s.mod_paths[mod])
            img = resize_img(img, self.image_size)
            chans.append(img)

        x = np.stack(chans, axis=0).astype(np.float32) / 255.0  # (4,H,W)

        y = load_png_gray(s.seg_path)
        y = resize_mask(y, self.image_size)
        y = remap_brats_mask(y).astype(np.int64)  # (H,W) in {0..3}

        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)

        if self.augment:
            if torch.rand(1).item() < 0.3:
                g = 0.9 + 0.2 * torch.rand(1).item()
                xt = torch.clamp(xt, 0, 1) ** g

            xt, yt = self._augment(xt, yt)

        return xt, yt


# -------------------------
# U-Net (clean baseline)
# -------------------------
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

    def forward(self, x):  # type: ignore[override]
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 4, base: int = 32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):  # type: ignore[override]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


# -------------------------
# Loss + metrics (IGNORE NO-TUMOR CASES)
# -------------------------
def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 4, eps: float = 1e-6) -> torch.Tensor:
    """
    Multiclass soft dice, excluding background (still trains on all samples).
    """
    probs = torch.softmax(logits, dim=1)
    target_1h = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    probs_fg = probs[:, 1:, :, :]
    targ_fg = target_1h[:, 1:, :, :]

    dims = (0, 2, 3)
    inter = torch.sum(probs_fg * targ_fg, dims)
    denom = torch.sum(probs_fg + targ_fg, dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

@torch.no_grad()
def dice_binary_ignore_empty_target(pred_bin: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6) -> Optional[float]:
    """
    Dice for a binary region, but IGNORE samples where target has no positives.
    Returns None if target is empty (so it doesn't inflate averages).
    """
    if target_bin.sum().item() == 0:
        return None
    inter = (pred_bin & target_bin).sum().item()
    denom = pred_bin.sum().item() + target_bin.sum().item()
    return (2 * inter + eps) / (denom + eps)

@torch.no_grad()
def brats_region_dice_per_sample(pred_hw: torch.Tensor, target_hw: torch.Tensor) -> Dict[str, Optional[float]]:
    """
    pred_hw/target_hw: (H,W) labels {0,1,2,3} where 3 is ET (original 4 remapped).
    Regions:
      WT = {1,2,3}, TC = {1,3}, ET = {3}
    Returns Optional dice per region (None if target region absent).
    """
    wt_p = pred_hw > 0
    wt_t = target_hw > 0

    tc_p = (pred_hw == 1) | (pred_hw == 3)
    tc_t = (target_hw == 1) | (target_hw == 3)

    et_p = pred_hw == 3
    et_t = target_hw == 3

    return {
        "WT": dice_binary_ignore_empty_target(wt_p, wt_t),
        "TC": dice_binary_ignore_empty_target(tc_p, tc_t),
        "ET": dice_binary_ignore_empty_target(et_p, et_t),
    }

@torch.no_grad()
def mean_fg_dice_ignore_empty_target(pred_hw: torch.Tensor, target_hw: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Mean Dice over classes 1..3, but ignore classes that are absent in TARGET
    (per-sample). This prevents empty-target classes from producing fake 1.0.
    """
    dices: List[float] = []
    for c in (1, 2, 3):
        t = (target_hw == c)
        if t.sum().item() == 0:
            continue  # ignore absent target class
        p = (pred_hw == c)
        inter = (p & t).sum().item()
        denom = p.sum().item() + t.sum().item()
        dices.append((2 * inter + eps) / (denom + eps))
    return float(np.mean(dices)) if dices else 0.0


# -------------------------
# Train / eval loops
# -------------------------
def train_one_epoch(model, loader, opt, scaler, device, use_amp: bool) -> float:
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = ce(logits, y) + soft_dice_loss(logits, y, num_classes=4)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        total += loss.item() * x.size(0)
        n += x.size(0)

    return total / max(1, n)

@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    n = 0

    fg_dices: List[float] = []
    wt: List[float] = []
    tc: List[float] = []
    et: List[float] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = ce(logits, y) + soft_dice_loss(logits, y, num_classes=4)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        pred = torch.argmax(logits, dim=1).cpu()
        targ = y.cpu()

        # IMPORTANT: compute per-sample dice and IGNORE empty-target cases
        for i in range(pred.size(0)):
            fg_dices.append(mean_fg_dice_ignore_empty_target(pred[i], targ[i]))

            r = brats_region_dice_per_sample(pred[i], targ[i])
            if r["WT"] is not None: wt.append(r["WT"])
            if r["TC"] is not None: tc.append(r["TC"])
            if r["ET"] is not None: et.append(r["ET"])

    return {
        "loss": total_loss / max(1, n),
        "mean_fg_dice": float(np.mean(fg_dices)) if fg_dices else 0.0,
        "WT": float(np.mean(wt)) if wt else 0.0,
        "TC": float(np.mean(tc)) if tc else 0.0,
        "ET": float(np.mean(et)) if et else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_root", type=str, default="/Users/basiakoch/Downloads/BraTS2020_2D_png_split")
    ap.add_argument("--image_size", type=int, default=240)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--min_fg_train", type=int, default=1)
    ap.add_argument("--keep_empty_prob", type=float, default=0.1)
    ap.add_argument("--ckpt", type=str, default="unet_brats2d_best.pt")
    ap.add_argument("--smoke", action="store_true", help="Run a tiny 1-epoch sanity test on a few batches")
    args = ap.parse_args()

    set_seed(args.seed)
    split_root = Path(args.split_root).expanduser()

    train_root = split_root / "train"
    val_root   = split_root / "val"
    test_root  = split_root / "test"

    for p in [train_root, val_root, test_root]:
        if not p.exists():
            raise SystemExit(f"Missing split folder: {p}")

    # Build samples
    train_samples = build_samples(train_root, args.image_size, min_fg=args.min_fg_train, keep_empty_prob=args.keep_empty_prob, seed=args.seed)
    val_samples   = build_samples(val_root,   args.image_size, min_fg=1, keep_empty_prob=1.0, seed=args.seed)
    test_samples  = build_samples(test_root,  args.image_size, min_fg=1, keep_empty_prob=1.0, seed=args.seed)

    if args.smoke:
        train_samples = train_samples[:16]
        val_samples = val_samples[:16]
        test_samples = test_samples[:16]
        args.epochs = 1
        args.batch_size = 2
        args.num_workers = 0
        print("SMOKE MODE: using 16 samples per split, 1 epoch, batch_size=2, num_workers=0")

    if not train_samples:
        raise SystemExit("No training samples found.")
    if not val_samples or not test_samples:
        raise SystemExit("Val/test samples empty.")

    print(f"Samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    print("NOTE: Metrics IGNORE empty-target regions (no-tumor cases won't inflate Dice).")

    train_ds = Brats2DDataset(train_samples, args.image_size, augment=True)
    val_ds   = Brats2DDataset(val_samples,   args.image_size, augment=False)
    test_ds  = Brats2DDataset(test_samples,  args.image_size, augment=False)

    # pin_memory only helps on CUDA
    pin = (str(args.device).lower().startswith("cuda"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    device = torch.device(args.device)
    model = UNet(in_channels=4, num_classes=4, base=args.base).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))
    use_amp = bool(args.amp and device.type == "cuda")

    best_score = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, scaler, device, use_amp=use_amp)
        val_metrics = evaluate(model, val_loader, device)

        brats_mean = (val_metrics["WT"] + val_metrics["TC"] + val_metrics["ET"]) / 3.0

        print(
            f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_fgDice={val_metrics['mean_fg_dice']:.4f} | "
            f"val_WT={val_metrics['WT']:.4f} val_TC={val_metrics['TC']:.4f} val_ET={val_metrics['ET']:.4f} | "
            f"val_BraTSmean={brats_mean:.4f}"
        )

        if brats_mean > best_score:
            best_score = brats_mean
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics, "args": vars(args)}, args.ckpt)

    print(f"\nBest checkpoint: epoch={best_epoch} val_BraTSmean={best_score:.4f} saved to {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device)
    test_brats_mean = (test_metrics["WT"] + test_metrics["TC"] + test_metrics["ET"]) / 3.0

    print("\nTEST (best checkpoint)")
    print(f"test_loss={test_metrics['loss']:.4f}")
    print(f"test_mean_fg_dice={test_metrics['mean_fg_dice']:.4f}")
    print(f"test_WT={test_metrics['WT']:.4f} test_TC={test_metrics['TC']:.4f} test_ET={test_metrics['ET']:.4f}")
    print(f"test_BraTSmean={test_brats_mean:.4f}")

if __name__ == "__main__":
    main()
