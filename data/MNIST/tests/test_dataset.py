from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/Users/basiakoch/Downloads/brats2020_top10_slices_split_npz")
N = 5
SEED = 42

files = sorted(ROOT.rglob("*.npz"))
print("Found .npz files:", len(files))
if not files:
    raise SystemExit(f"No .npz files found under {ROOT}")

random.seed(SEED)
sample = random.sample(files, k=min(N, len(files)))

mod_names = ["t1", "t1ce", "t2", "flair"]

def pick_best_modality(x: np.ndarray) -> int:
    """Pick channel with highest nonzero std (often gives good contrast)."""
    best_c, best_score = 0, -1.0
    for c in range(x.shape[0]):
        xc = x[c]
        nz = xc != 0
        score = float(np.std(xc[nz])) if np.any(nz) else 0.0
        if score > best_score:
            best_score, best_c = score, c
    return best_c

for i, f in enumerate(sample, 1):
    d = np.load(f, allow_pickle=False)
    x = d["x"]  # (4,H,W)
    y = d["y"]  # (H,W)

    uniq = np.unique(y)
    has_tumor = bool(np.any(y > 0))

    print("\n==============================")
    print(f"[{i}] {f}")
    print("x:", x.shape, x.dtype, "| y:", y.shape, y.dtype)
    print("mask unique labels:", uniq, "| tumor present:", has_tumor)
    if "case_id" in d:
        print("case_id:", d["case_id"])
    if "slice_idx" in d:
        print("slice_idx:", int(d["slice_idx"]))

    c_best = pick_best_modality(x)

    # Side-by-side: image (best modality) and mask
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(x[c_best], cmap="gray")
    axes[0].set_title(f"Image ({mod_names[c_best]})")
    axes[0].axis("off")

    axes[1].imshow(y)  # label image
    axes[1].set_title(f"Mask (labels {uniq.tolist()})")
    axes[1].axis("off")

    plt.suptitle(f"{f.name}", y=0.98)
    plt.tight_layout()
    plt.show()

    # Optional: overlay panel (useful to confirm alignment)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(x[c_best], cmap="gray")
    ax.imshow((y > 0).astype(np.uint8), alpha=0.35)
    ax.set_title(f"Overlay on {mod_names[c_best]}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

print("\nDone.")
