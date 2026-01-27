import argparse
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

MOD_NAMES = ["FLAIR", "T1", "T1ce", "T2"]

def find_npz(root: Path, client: str, split: str):
    base = root / client / split
    files = sorted(base.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {base}")
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (contains client_0/client_1)")
    ap.add_argument("--client", default="client_0", choices=["client_0", "client_1"])
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.root).expanduser()

    files = find_npz(root, args.client, args.split)
    if args.seed is not None:
        random.seed(args.seed)

    f = random.choice(files)
    d = np.load(f)

    if "image" not in d.files or "mask" not in d.files:
        raise KeyError(f"{f} missing keys. Found: {d.files}")

    x = d["image"]  # expected (4,H,W)
    y = d["mask"]   # expected (H,W)

    # normalize shapes if needed
    if x.ndim == 3 and x.shape[0] == 4:
        pass  # (4,H,W)
    elif x.ndim == 3 and x.shape[-1] == 4:
        x = np.transpose(x, (2, 0, 1))  # (H,W,4) -> (4,H,W)
    else:
        raise ValueError(f"Unexpected image shape: {x.shape}")

    if y.ndim != 2:
        # handle (1,H,W) or (H,W,1)
        if y.ndim == 3 and y.shape[0] == 1:
            y = y[0]
        elif y.ndim == 3 and y.shape[-1] == 1:
            y = y[..., 0]
        else:
            raise ValueError(f"Unexpected mask shape: {y.shape}")

    print("File:", f)
    print("image shape:", x.shape, "dtype:", x.dtype)
    print("mask  shape:", y.shape, "dtype:", y.dtype)
    print("mask unique values:", np.unique(y))

    # Plot 4 modalities + mask + overlay
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    for i in range(4):
        axes[i].imshow(x[i], cmap="gray")
        axes[i].set_title(MOD_NAMES[i])
        axes[i].axis("off")

    axes[4].imshow(y, cmap="viridis")
    axes[4].set_title("Mask (labels)")
    axes[4].axis("off")

    # Overlay mask on FLAIR
    axes[5].imshow(x[0], cmap="gray")
    axes[5].imshow(y > 0, alpha=0.35)  # tumor vs background
    axes[5].set_title("Overlay (mask>0 on FLAIR)")
    axes[5].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
