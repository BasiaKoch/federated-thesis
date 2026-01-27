#!/usr/bin/env python3
"""
Create ET-skewed 70/30 partition for federated learning.

Produces a label-heterogeneous split where:
  - Client 0 (70% of data): biased toward LOW-ET cases
  - Client 1 (30% of data): biased toward HIGH-ET cases
  - BOTH clients see all three classes (WT, TC, ET)

Algorithm:
  1. Scan all training cases and compute ET ratio per case
     (ET voxels / total tumor voxels)
  2. Sort cases by ET ratio and split into quartiles Q1..Q4
  3. Assign cases:
       Client 0: all Q1 + all Q2 + 20% of Q3 + 10% of Q4
       Client 1: 80% of Q3 + 90% of Q4
  4. Within each client, split into train/val/test (80/10/10)
  5. Create symlinks to original NPZ files

Output structure:
    <output_dir>/
    ├── client_map.json          # full metadata + ET stats
    ├── client_0_cases.txt
    ├── client_1_cases.txt
    └── client_data/
        ├── client_0/
        │   ├── train/CASE/*.npz
        │   ├── val/CASE/*.npz
        │   └── test/CASE/*.npz
        └── client_1/
            ├── train/CASE/*.npz
            ├── val/CASE/*.npz
            └── test/CASE/*.npz

Usage on HPC:
    python data/partitions/make_et_skewed_partition.py

    # Or with custom paths:
    python data/partitions/make_et_skewed_partition.py \
        --source_dir /path/to/brats2d_one_slice_per_patient_clients \
        --output_dir /path/to/partitions/brats2d_7030_et_skewed \
        --seed 42
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ── ET ratio computation ─────────────────────────────────────────────────────

def load_mask(npz_path: Path) -> np.ndarray:
    """Load the mask array from an NPZ file."""
    d = np.load(str(npz_path), allow_pickle=False)
    for k in ["mask", "y", "Y", "seg", "label", "labels"]:
        if k in d:
            return d[k]
    # fallback: pick integer array
    for k in d:
        if d[k].dtype.kind in ("i", "u"):
            return d[k]
    raise KeyError(f"No mask found in {npz_path}, keys={list(d.keys())}")


def compute_et_ratio(mask: np.ndarray) -> float:
    """
    Compute ET ratio for a BraTS mask.
    BraTS labels: 0=background, 1=NCR/NET, 2=ED, 4=ET
    WT = {1, 2, 4}, ET = {4}
    Returns: ET_voxels / (WT_voxels + 1)
    """
    m = mask.astype(np.int32).ravel()
    wt = np.count_nonzero(m > 0)
    et = np.count_nonzero(m == 4)
    return et / (wt + 1)


def scan_cases(source_dir: Path) -> Dict[str, float]:
    """
    Scan all cases under source_dir and compute ET ratio per case.

    Expects structure:
        source_dir/
        ├── client_0/{train,val,test}/CASE/*.npz
        └── client_1/{train,val,test}/CASE/*.npz

    Returns: {case_id: et_ratio}
    """
    et_ratios = {}

    # Collect all unique case directories across all clients and splits
    case_dirs = {}
    for client_dir in sorted(source_dir.iterdir()):
        if not client_dir.is_dir() or not client_dir.name.startswith("client_"):
            continue
        for split in ["train", "val", "test"]:
            split_dir = client_dir / split
            if not split_dir.exists():
                continue
            for case_dir in sorted(split_dir.iterdir()):
                if case_dir.is_dir() and case_dir.name.startswith("BraTS"):
                    case_dirs[case_dir.name] = case_dir

    print(f"Found {len(case_dirs)} unique cases across all clients/splits")

    for case_id, case_dir in sorted(case_dirs.items()):
        npz_files = sorted(case_dir.glob("*.npz"))
        if not npz_files:
            print(f"  WARNING: no .npz in {case_dir}, skipping")
            continue

        # For one-slice-per-patient, there's typically 1 NPZ per case
        # If multiple, average the ET ratio
        total_et = 0
        total_wt = 0
        for npz_path in npz_files:
            # Resolve symlinks to actual file
            real_path = npz_path.resolve() if npz_path.is_symlink() else npz_path
            mask = load_mask(real_path)
            m = mask.astype(np.int32).ravel()
            total_wt += np.count_nonzero(m > 0)
            total_et += np.count_nonzero(m == 4)

        et_ratios[case_id] = total_et / (total_wt + 1)

    return et_ratios


def find_original_case(case_id: str, source_dir: Path) -> Path:
    """Find the original case directory (resolving symlinks)."""
    for client_dir in sorted(source_dir.iterdir()):
        if not client_dir.is_dir() or not client_dir.name.startswith("client_"):
            continue
        for split in ["train", "val", "test"]:
            case_path = client_dir / split / case_id
            if case_path.exists():
                return case_path
    raise FileNotFoundError(f"Case {case_id} not found in {source_dir}")


# ── Partitioning ──────────────────────────────────────────────────────────────

def partition_by_et(
    et_ratios: Dict[str, float],
    seed: int,
    q3_to_c0: float = 0.20,
    q4_to_c0: float = 0.10,
) -> Tuple[List[str], List[str]]:
    """
    Partition cases into two clients based on ET ratio quartiles.

    Client 0 (majority, low-ET bias):
        all Q1 + all Q2 + q3_to_c0 of Q3 + q4_to_c0 of Q4
    Client 1 (minority, high-ET bias):
        (1 - q3_to_c0) of Q3 + (1 - q4_to_c0) of Q4

    Both clients see all classes. The overlap from shared Q3/Q4
    prevents complete distribution mismatch.
    """
    rng = random.Random(seed)

    # Sort cases by ET ratio
    sorted_cases = sorted(et_ratios.keys(), key=lambda c: et_ratios[c])
    n = len(sorted_cases)

    # Split into quartiles
    q_size = n // 4
    q1 = sorted_cases[:q_size]
    q2 = sorted_cases[q_size:2 * q_size]
    q3 = sorted_cases[2 * q_size:3 * q_size]
    q4 = sorted_cases[3 * q_size:]

    # Shuffle within each quartile for randomness
    rng.shuffle(q3)
    rng.shuffle(q4)

    # Split Q3 and Q4 between clients
    n_q3_c0 = max(1, int(round(len(q3) * q3_to_c0)))
    n_q4_c0 = max(1, int(round(len(q4) * q4_to_c0)))

    q3_c0 = q3[:n_q3_c0]
    q3_c1 = q3[n_q3_c0:]
    q4_c0 = q4[:n_q4_c0]
    q4_c1 = q4[n_q4_c0:]

    client_0 = sorted(q1 + q2 + q3_c0 + q4_c0)
    client_1 = sorted(q3_c1 + q4_c1)

    return client_0, client_1


def split_train_val_test(
    cases: List[str],
    seed: int,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> Tuple[List[str], List[str], List[str]]:
    """Split cases into train/val/test."""
    rng = random.Random(seed)
    cases = sorted(cases)
    rng.shuffle(cases)

    n = len(cases)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))

    test = sorted(cases[:n_test])
    val = sorted(cases[n_test:n_test + n_val])
    train = sorted(cases[n_test + n_val:])

    return train, val, test


def create_symlinks(src_case_dir: Path, dst_dir: Path, case_id: str) -> int:
    """Create symlinks for all NPZ files of a case. Returns slice count."""
    dst_case = dst_dir / case_id
    dst_case.mkdir(parents=True, exist_ok=True)

    count = 0
    for npz in sorted(src_case_dir.glob("*.npz")):
        real = npz.resolve()
        dst = dst_case / npz.name
        if not dst.exists() and not dst.is_symlink():
            dst.symlink_to(real)
        count += 1
    return count


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Create ET-skewed 70/30 partition for federated learning"
    )
    ap.add_argument(
        "--source_dir", type=Path,
        default=Path("/home/bk489/federated/federated-thesis/data/partitions/brats2d_one_slice_per_patient_clients"),
        help="Source directory with existing client data (contains client_0/, client_1/ with train/val/test)",
    )
    ap.add_argument(
        "--output_dir", type=Path,
        default=Path("/home/bk489/federated/federated-thesis/data/partitions/brats2d_7030_et_skewed"),
        help="Output directory for new partition",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument(
        "--q3_to_c0", type=float, default=0.20,
        help="Fraction of Q3 (mid-high ET) cases assigned to Client 0 (default: 0.20)",
    )
    ap.add_argument(
        "--q4_to_c0", type=float, default=0.10,
        help="Fraction of Q4 (highest ET) cases assigned to Client 0 (default: 0.10)",
    )
    args = ap.parse_args()

    print("=" * 70)
    print("ET-SKEWED PARTITION FOR FEDERATED LEARNING")
    print("=" * 70)
    print(f"Source:  {args.source_dir}")
    print(f"Output:  {args.output_dir}")
    print(f"Seed:    {args.seed}")
    print(f"Q3->C0:  {args.q3_to_c0:.0%}   Q4->C0: {args.q4_to_c0:.0%}")
    print()

    # Step 1: Scan all cases and compute ET ratios
    print("Step 1: Computing ET ratio per case...")
    et_ratios = scan_cases(args.source_dir)
    if not et_ratios:
        raise RuntimeError("No cases found. Check --source_dir path.")

    # Print ET distribution stats
    ratios = list(et_ratios.values())
    print(f"\nET ratio statistics across {len(ratios)} cases:")
    print(f"  min={min(ratios):.4f}  Q1={np.percentile(ratios,25):.4f}  "
          f"median={np.median(ratios):.4f}  Q3={np.percentile(ratios,75):.4f}  "
          f"max={max(ratios):.4f}")
    n_zero = sum(1 for r in ratios if r < 1e-6)
    print(f"  Cases with zero/near-zero ET: {n_zero}/{len(ratios)}")

    # Step 2: Partition by ET quartiles
    print("\nStep 2: Partitioning by ET ratio quartiles...")
    client_0_cases, client_1_cases = partition_by_et(
        et_ratios, args.seed, args.q3_to_c0, args.q4_to_c0
    )

    print(f"\n  Client 0: {len(client_0_cases)} cases ({len(client_0_cases)/len(et_ratios)*100:.1f}%)")
    c0_ratios = [et_ratios[c] for c in client_0_cases]
    print(f"    ET ratio: mean={np.mean(c0_ratios):.4f}  median={np.median(c0_ratios):.4f}  "
          f"max={max(c0_ratios):.4f}")

    print(f"  Client 1: {len(client_1_cases)} cases ({len(client_1_cases)/len(et_ratios)*100:.1f}%)")
    c1_ratios = [et_ratios[c] for c in client_1_cases]
    print(f"    ET ratio: mean={np.mean(c1_ratios):.4f}  median={np.median(c1_ratios):.4f}  "
          f"max={max(c1_ratios):.4f}")

    print(f"\n  ET ratio gap (C1 mean - C0 mean): {np.mean(c1_ratios) - np.mean(c0_ratios):.4f}")

    # Step 3: Train/val/test splits per client
    print("\nStep 3: Creating train/val/test splits...")
    splits = {}
    for cid, cases in [("client_0", client_0_cases), ("client_1", client_1_cases)]:
        train, val, test = split_train_val_test(
            cases, seed=args.seed + int(cid[-1]), val_frac=args.val_frac, test_frac=args.test_frac
        )
        splits[cid] = {"train": train, "val": val, "test": test}
        print(f"  {cid}: train={len(train)} val={len(val)} test={len(test)}")

    # Step 4: Create output directory and symlinks
    print("\nStep 4: Creating symlinks...")
    output = args.output_dir
    if output.exists():
        import shutil
        print(f"  Removing existing output dir: {output}")
        shutil.rmtree(output)

    total_slices = 0
    for cid in ["client_0", "client_1"]:
        for split_name, case_list in splits[cid].items():
            dst = output / "client_data" / cid / split_name
            for case_id in case_list:
                src_case = find_original_case(case_id, args.source_dir)
                n = create_symlinks(src_case, dst, case_id)
                total_slices += n

    print(f"  Total slices linked: {total_slices}")

    # Step 5: Write metadata
    print("\nStep 5: Writing metadata...")

    # Case lists
    (output / "client_0_cases.txt").write_text("\n".join(client_0_cases) + "\n")
    (output / "client_1_cases.txt").write_text("\n".join(client_1_cases) + "\n")

    # Detailed client map
    client_map = {
        "client_0": client_0_cases,
        "client_1": client_1_cases,
        "splits": {
            cid: {s: cases for s, cases in split_data.items()}
            for cid, split_data in splits.items()
        },
        "et_ratios": {c: float(r) for c, r in sorted(et_ratios.items())},
        "metadata": {
            "partition_type": "et_skewed",
            "description": "Client 0 biased toward low-ET cases, Client 1 biased toward high-ET cases",
            "seed": args.seed,
            "total_cases": len(et_ratios),
            "q3_to_c0": args.q3_to_c0,
            "q4_to_c0": args.q4_to_c0,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "client_0_stats": {
                "n_cases": len(client_0_cases),
                "et_ratio_mean": float(np.mean(c0_ratios)),
                "et_ratio_median": float(np.median(c0_ratios)),
            },
            "client_1_stats": {
                "n_cases": len(client_1_cases),
                "et_ratio_mean": float(np.mean(c1_ratios)),
                "et_ratio_median": float(np.median(c1_ratios)),
            },
        },
    }
    (output / "client_map.json").write_text(json.dumps(client_map, indent=2))

    # Step 6: Print verification summary
    print(f"\n{'='*70}")
    print("PARTITION SUMMARY")
    print(f"{'='*70}")
    print(f"Output: {output}")
    print()
    print("Client 0 (low-ET bias):")
    print(f"  Cases: {len(client_0_cases)} ({len(client_0_cases)/len(et_ratios)*100:.1f}%)")
    print(f"  Train/Val/Test: {len(splits['client_0']['train'])}/{len(splits['client_0']['val'])}/{len(splits['client_0']['test'])}")
    print(f"  ET ratio: mean={np.mean(c0_ratios):.4f}  median={np.median(c0_ratios):.4f}")
    print()
    print("Client 1 (high-ET bias):")
    print(f"  Cases: {len(client_1_cases)} ({len(client_1_cases)/len(et_ratios)*100:.1f}%)")
    print(f"  Train/Val/Test: {len(splits['client_1']['train'])}/{len(splits['client_1']['val'])}/{len(splits['client_1']['test'])}")
    print(f"  ET ratio: mean={np.mean(c1_ratios):.4f}  median={np.median(c1_ratios):.4f}")
    print()
    print("Expected federated behavior:")
    print("  - FedAvg: oscillation in ET dice due to conflicting ET distributions")
    print("  - FedProx (mu~0.001-0.01): should stabilize ET convergence")
    print("  - Both clients see all 3 classes -> both produce useful gradients")
    print(f"{'='*70}")

    # Point to training
    print(f"\nTo train:")
    print(f"  python federated_unet/unet/unet_flower_train_70_30.py \\")
    print(f"    --partition_dir {output / 'client_data'} \\")
    print(f"    --strategy fedavg --mu 0.0 --rounds 30 --local_epochs 3")
    print()
    print(f"  python federated_unet/unet/unet_flower_train_70_30.py \\")
    print(f"    --partition_dir {output / 'client_data'} \\")
    print(f"    --strategy fedprox --mu 0.001 --rounds 30 --local_epochs 3")


if __name__ == "__main__":
    main()
