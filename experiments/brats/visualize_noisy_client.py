#!/usr/bin/env python3
"""
Visualize corruption: side-by-side clean vs noisy client slices,
and verify that the noisy client's TEST set is uncorrupted.

Usage:
    python experiments/brats/visualize_noisy_client.py \
        --partition_dir data/partitions/brats2d_8client_noisy_heavy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_npz(path: Path):
    d = np.load(str(path), allow_pickle=False)
    return d["image"], d["mask"]  # (4,H,W), (H,W)


def pick_sample(directory: Path, idx: int = 0) -> Path:
    """Pick the idx-th .npz file from a directory (sorted)."""
    files = sorted(directory.rglob("*.npz"))
    if not files:
        sys.exit(f"No .npz files in {directory}")
    return files[idx % len(files)]


def intensity_stats(img: np.ndarray) -> dict:
    """Basic stats on the flair channel (channel 0), non-zero voxels."""
    ch = img[0]
    nz = ch[ch != 0]
    if nz.size == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    return {
        "mean": float(nz.mean()),
        "std": float(nz.std()),
        "min": float(nz.min()),
        "max": float(nz.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition_dir", type=Path,
                    default=Path("data/partitions/brats2d_8client_noisy_heavy"))
    ap.add_argument("--clean_client", type=int, default=0,
                    help="A clean client id to compare against")
    ap.add_argument("--sample_idx", type=int, default=0,
                    help="Which slice index to pick from each client")
    ap.add_argument("--output", type=str, default="noisy_vs_clean.png")
    args = ap.parse_args()

    # Load config
    config_path = args.partition_dir / "client_map.json"
    if not config_path.exists():
        sys.exit(f"client_map.json not found in {args.partition_dir}")
    config = json.loads(config_path.read_text())
    noisy_id = config["experiment_config"]["noisy_client_id"]
    corrupt_params = config["experiment_config"]["corruption_params"]

    client_data = args.partition_dir / "client_data"
    clean_id = args.clean_client
    if clean_id == noisy_id:
        # Pick a different clean client
        for c in range(config["experiment_config"]["num_clients"]):
            if c != noisy_id:
                clean_id = c
                break

    print(f"Noisy client: {noisy_id}")
    print(f"Clean client: {clean_id}")
    print(f"Corruption: {config['experiment_config']['corrupt_mode']} "
          f"({config['experiment_config']['corrupt_strength']})")
    print(f"  params: {corrupt_params}")

    # ── Part 1: Side-by-side TRAIN comparison ──
    clean_train = client_data / f"client_{clean_id}" / "train"
    noisy_train = client_data / f"client_{noisy_id}" / "train"

    clean_path = pick_sample(clean_train, args.sample_idx)
    noisy_path = pick_sample(noisy_train, args.sample_idx)

    clean_img, clean_mask = load_npz(clean_path)
    noisy_img, noisy_mask = load_npz(noisy_path)

    print(f"\nClean train sample: {clean_path.name}")
    print(f"  stats (flair): {intensity_stats(clean_img)}")
    print(f"Noisy train sample: {noisy_path.name}")
    print(f"  stats (flair): {intensity_stats(noisy_img)}")

    # Plot: 2 rows (clean, noisy) x 5 cols (flair, t1, t1ce, t2, mask)
    mod_names = ["FLAIR", "T1", "T1ce", "T2", "Mask"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(
        f"Clean client {clean_id} (top) vs Noisy client {noisy_id} (bottom)\n"
        f"Corruption: {config['experiment_config']['corrupt_mode']} "
        f"({config['experiment_config']['corrupt_strength']})",
        fontsize=14,
    )

    for col in range(4):
        # Clean
        axes[0, col].imshow(clean_img[col], cmap="gray")
        axes[0, col].set_title(f"{mod_names[col]} (clean)")
        axes[0, col].axis("off")
        # Noisy
        axes[1, col].imshow(noisy_img[col], cmap="gray")
        axes[1, col].set_title(f"{mod_names[col]} (noisy)")
        axes[1, col].axis("off")

    # Masks
    axes[0, 4].imshow(clean_mask, cmap="nipy_spectral", vmin=0, vmax=4)
    axes[0, 4].set_title("Mask (clean)")
    axes[0, 4].axis("off")
    axes[1, 4].imshow(noisy_mask, cmap="nipy_spectral", vmin=0, vmax=4)
    axes[1, 4].set_title("Mask (noisy — should be unmodified)")
    axes[1, 4].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization → {args.output}")

    # ── Part 2: Verify noisy client TEST is uncorrupted ──
    print(f"\n{'='*60}")
    print("VERIFICATION: Is the noisy client's TEST set uncorrupted?")
    print(f"{'='*60}")

    # Strategy: re-apply the corruption pipeline to a noisy test slice
    # and compare with what's on disk. If the on-disk version does NOT
    # match the corrupted version, the test set was saved clean.
    noisy_test_dir = client_data / f"client_{noisy_id}" / "test"
    noisy_test_files = sorted(noisy_test_dir.rglob("*.npz"))

    if not noisy_test_files:
        print("  WARNING: No test files for noisy client!")
    else:
        # Compare intensity distributions: noisy train vs noisy test
        # If test is clean, its std should be close to 1.0 (z-scored)
        # while train will have inflated std due to corruption
        train_stds = []
        for f in sorted(noisy_train.rglob("*.npz"))[:10]:
            img, _ = load_npz(f)
            ch = img[0]
            nz = ch[ch != 0]
            if nz.size > 0:
                train_stds.append(float(nz.std()))

        test_stds = []
        for f in noisy_test_files[:10]:
            img, _ = load_npz(f)
            ch = img[0]
            nz = ch[ch != 0]
            if nz.size > 0:
                test_stds.append(float(nz.std()))

        # Also compare with a clean client's test
        clean_test_dir = client_data / f"client_{clean_id}" / "test"
        clean_test_stds = []
        for f in sorted(clean_test_dir.rglob("*.npz"))[:10]:
            img, _ = load_npz(f)
            ch = img[0]
            nz = ch[ch != 0]
            if nz.size > 0:
                clean_test_stds.append(float(nz.std()))

        mean_train_std = np.mean(train_stds) if train_stds else 0
        mean_test_std = np.mean(test_stds) if test_stds else 0
        mean_clean_test_std = np.mean(clean_test_stds) if clean_test_stds else 0

        print(f"\n  Noisy client TRAIN — mean flair std: {mean_train_std:.4f}")
        print(f"  Noisy client TEST  — mean flair std: {mean_test_std:.4f}")
        print(f"  Clean client TEST  — mean flair std: {mean_clean_test_std:.4f}")

        # If test is clean, its std should be similar to clean client's test
        # (both ~1.0 from z-scoring). Train std will be inflated by corruption.
        if mean_train_std > 0 and mean_test_std > 0:
            train_test_ratio = mean_train_std / mean_test_std
            clean_noisy_test_diff = abs(mean_test_std - mean_clean_test_std)

            print(f"\n  Train/Test std ratio: {train_test_ratio:.2f} "
                  f"(>>1 means train is corrupted, test is clean)")
            print(f"  |Noisy test - Clean test| std: {clean_noisy_test_diff:.4f} "
                  f"({'similar → test is CLEAN' if clean_noisy_test_diff < 0.3 else 'different → CHECK!'})")

            if train_test_ratio > 1.3 and clean_noisy_test_diff < 0.3:
                print("\n  PASS: Noisy client test set appears UNCORRUPTED.")
            elif train_test_ratio < 1.1:
                print("\n  WARNING: Train and test have similar std — "
                      "corruption may not be applied, or test may also be corrupted.")
            else:
                print("\n  INCONCLUSIVE: Check the visualization manually.")

    # ── Part 3: Histogram comparison ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle("FLAIR intensity distributions (non-zero voxels)", fontsize=13)

    def collect_flair(directory, n=10):
        vals = []
        for f in sorted(directory.rglob("*.npz"))[:n]:
            img, _ = load_npz(f)
            ch = img[0]
            vals.extend(ch[ch != 0].tolist())
        return np.array(vals)

    clean_vals = collect_flair(clean_train)
    noisy_train_vals = collect_flair(noisy_train)
    noisy_test_vals = collect_flair(noisy_test_dir)

    for ax, vals, title in zip(
        axes2,
        [clean_vals, noisy_train_vals, noisy_test_vals],
        [f"Clean client {clean_id} TRAIN",
         f"Noisy client {noisy_id} TRAIN",
         f"Noisy client {noisy_id} TEST"],
    ):
        if vals.size > 0:
            ax.hist(vals, bins=80, density=True, alpha=0.7, color="steelblue")
            ax.set_title(f"{title}\nstd={vals.std():.3f}")
        else:
            ax.set_title(f"{title}\n(no data)")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Density")

    plt.tight_layout()
    hist_path = args.output.replace(".png", "_histograms.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    print(f"Saved histograms → {hist_path}")

    # plt.show()  # uncomment for interactive viewing


if __name__ == "__main__":
    main()
