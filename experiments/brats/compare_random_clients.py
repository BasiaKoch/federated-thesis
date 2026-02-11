#!/usr/bin/env python3
"""
Side-by-side visualization of a random slice from each client.

Example:
  python experiments/brats/compare_random_clients.py \
      --partition_dir data/partitions/brats2d_2client_extreme \
      --split train \
      --output compare_random.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(str(path), allow_pickle=False)
    img = None
    msk = None
    for k in ["image", "img", "x", "X"]:
        if k in d:
            img = d[k]
            break
    for k in ["mask", "y", "Y", "seg", "label", "labels"]:
        if k in d:
            msk = d[k]
            break
    if img is None or msk is None:
        raise KeyError(f"Missing image/mask in {path.name}. Keys={list(d.keys())}")
    return img, msk


def to_chw(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = img[None, ...]
    elif img.ndim == 3:
        if img.shape[-1] <= 8 and img.shape[0] > 8:
            img = np.transpose(img, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected image ndim={img.ndim}")
    return img.astype(np.float32)


def pick_random_file(files: List[Path], rng: np.random.Generator) -> Path:
    if not files:
        raise FileNotFoundError("No .npz files found.")
    return files[int(rng.integers(0, len(files)))]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare random slices across clients (side-by-side)."
    )
    ap.add_argument("--partition_dir", type=Path, required=True,
                    help="Partition root (contains client_data or client_* dirs)")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=Path("compare_random.png"))
    args = ap.parse_args()

    root = args.partition_dir
    client_root = root / "client_data" if (root / "client_data").is_dir() else root
    if not client_root.is_dir():
        sys.exit(f"Partition dir not found: {client_root}")

    client_dirs = sorted([d for d in client_root.iterdir()
                          if d.is_dir() and d.name.startswith("client_")])
    if not client_dirs:
        sys.exit(f"No client_* dirs under {client_root}")

    rng = np.random.default_rng(args.seed)

    samples = []
    for cdir in client_dirs:
        split_dir = cdir / args.split
        files = sorted(split_dir.rglob("*.npz")) if split_dir.is_dir() else []
        if not files:
            sys.exit(f"No .npz files found for {cdir.name}/{args.split}")
        chosen = pick_random_file(files, rng)
        img, msk = load_npz(chosen)
        img = to_chw(img)
        samples.append((cdir.name, chosen.name, img, msk))

    # Determine number of channels to display
    max_ch = max(s[2].shape[0] for s in samples)
    ch_names = ["FLAIR", "T1", "T1ce", "T2"]
    cols = max_ch + 1  # +1 for mask

    fig, axes = plt.subplots(len(samples), cols, figsize=(3.2 * cols, 3.2 * len(samples)))
    if len(samples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (client_name, fname, img, msk) in enumerate(samples):
        for c in range(max_ch):
            ax = axes[row, c]
            if c < img.shape[0]:
                ax.imshow(img[c], cmap="gray")
                title = ch_names[c] if c < len(ch_names) else f"Ch {c}"
                ax.set_title(f"{client_name} {title}")
            else:
                ax.axis("off")
            ax.axis("off")
        axm = axes[row, max_ch]
        axm.imshow(msk, cmap="nipy_spectral", vmin=0, vmax=4)
        axm.set_title(f"{client_name} Mask")
        axm.axis("off")

    fig.suptitle(f"Random {args.split} slice per client (seed={args.seed})", fontsize=12)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
