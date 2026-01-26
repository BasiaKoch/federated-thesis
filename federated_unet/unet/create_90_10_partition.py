#!/usr/bin/env python3
"""
Create 90/10 client partition for federated learning.

Creates the complete directory structure:
    federated_clients_2_90_10/
    ├── client_0_cases.txt
    ├── client_1_cases.txt
    ├── client_map.json
    └── client_data/
        ├── client_0/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── client_1/
            ├── train/
            ├── val/
            └── test/

Usage on cluster:
    python create_90_10_partition.py

Or with custom paths:
    python create_90_10_partition.py \
        --brats_root /path/to/brats2020_top10_slices_split_npz \
        --output_dir /path/to/partitions/federated_clients_2_90_10
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def load_train_cases(brats_root: Path) -> List[str]:
    """Load training case IDs from BraTS data."""
    # Try split.json first
    split_json = brats_root / "split.json"
    if split_json.exists():
        data = json.loads(split_json.read_text())
        cases = data.get("train", [])
        if cases:
            print(f"Loaded {len(cases)} training cases from split.json")
            return sorted(cases)

    # Fallback: list directories in train/
    train_dir = brats_root / "train"
    if train_dir.exists():
        cases = [d.name for d in train_dir.iterdir()
                 if d.is_dir() and d.name.startswith("BraTS")]
        print(f"Loaded {len(cases)} training cases from train/ directory")
        return sorted(cases)

    raise FileNotFoundError(f"Could not find training cases in {brats_root}")


def partition_cases(
    cases: List[str],
    split_ratios: List[float],
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Partition cases into two clients based on split ratios."""
    rng = random.Random(seed)
    cases = sorted(cases)
    rng.shuffle(cases)

    n = len(cases)
    n_client0 = int(round(n * split_ratios[0]))

    client_0_cases = sorted(cases[:n_client0])
    client_1_cases = sorted(cases[n_client0:])

    return client_0_cases, client_1_cases


def split_client_cases(
    cases: List[str],
    seed: int,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> Tuple[List[str], List[str], List[str]]:
    """Split a client's cases into train/val/test."""
    rng = random.Random(seed)
    cases = sorted(cases)
    rng.shuffle(cases)

    n = len(cases)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))

    test_cases = sorted(cases[:n_test])
    val_cases = sorted(cases[n_test:n_test + n_val])
    train_cases = sorted(cases[n_test + n_val:])

    return train_cases, val_cases, test_cases


def create_symlinks(
    src_train_dir: Path,
    dst_dir: Path,
    cases: List[str],
) -> int:
    """Create symlinks for cases. Returns number of slices linked."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    total_slices = 0

    for case_id in cases:
        src_case = src_train_dir / case_id
        if not src_case.exists():
            raise FileNotFoundError(f"Missing case: {src_case}")

        dst_case = dst_dir / case_id
        dst_case.mkdir(parents=True, exist_ok=True)

        npz_files = list(src_case.glob("*.npz"))
        if not npz_files:
            raise RuntimeError(f"No .npz files in {src_case}")

        for f in npz_files:
            dst_file = dst_case / f.name
            if not dst_file.exists() and not dst_file.is_symlink():
                dst_file.symlink_to(f.resolve())
            total_slices += 1

    return total_slices


def write_case_list(path: Path, cases: List[str]) -> None:
    """Write case list to file."""
    path.write_text("\n".join(cases) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Create 90/10 federated partition with train/val/test splits"
    )
    ap.add_argument(
        "--brats_root",
        type=Path,
        default=Path("/home/bk489/federated/federated-thesis/data/brats2020_top10_slices_split_npz"),
        help="Path to BraTS data root",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/bk489/federated/federated-thesis/data/partitions/federated_clients_2_90_10"),
        help="Output directory for partition",
    )
    ap.add_argument(
        "--split",
        type=float,
        nargs=2,
        default=[0.9, 0.1],
        help="Split ratios for client_0 and client_1",
    )
    ap.add_argument(
        "--val_frac",
        type=float,
        default=0.10,
        help="Fraction of each client's data for validation",
    )
    ap.add_argument(
        "--test_frac",
        type=float,
        default=0.10,
        help="Fraction of each client's data for test",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing partition and rebuild",
    )
    args = ap.parse_args()

    brats_train = args.brats_root / "train"
    if not brats_train.exists():
        raise FileNotFoundError(f"BraTS train dir not found: {brats_train}")

    # Load all training cases
    cases = load_train_cases(args.brats_root)
    print(f"Total training cases: {len(cases)}")

    # Create 90/10 partition
    client_0_cases, client_1_cases = partition_cases(cases, args.split, args.seed)

    print(f"\n{'='*60}")
    print(f"PARTITION: {args.split[0]*100:.0f}/{args.split[1]*100:.0f}")
    print(f"{'='*60}")
    print(f"Client 0: {len(client_0_cases)} cases ({len(client_0_cases)/len(cases)*100:.1f}%)")
    print(f"Client 1: {len(client_1_cases)} cases ({len(client_1_cases)/len(cases)*100:.1f}%)")

    # Delete existing if --rebuild
    if args.rebuild and args.output_dir.exists():
        print(f"\nRemoving existing: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client_data_dir = args.output_dir / "client_data"

    # Write case lists
    write_case_list(args.output_dir / "client_0_cases.txt", client_0_cases)
    write_case_list(args.output_dir / "client_1_cases.txt", client_1_cases)

    # Write client_map.json
    client_map = {
        "client_0": client_0_cases,
        "client_1": client_1_cases,
        "metadata": {
            "split_ratios": args.split,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "total_cases": len(cases),
        }
    }
    (args.output_dir / "client_map.json").write_text(json.dumps(client_map, indent=2))

    # Process each client
    for cid, client_cases in enumerate([client_0_cases, client_1_cases]):
        client_root = client_data_dir / f"client_{cid}"

        # Split into train/val/test
        train_cases, val_cases, test_cases = split_client_cases(
            client_cases,
            seed=args.seed + cid,  # Different seed per client
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )

        print(f"\nClient {cid}:")
        print(f"  Total cases: {len(client_cases)}")
        print(f"  Train: {len(train_cases)} cases")
        print(f"  Val:   {len(val_cases)} cases")
        print(f"  Test:  {len(test_cases)} cases")

        # Create symlinks for each split
        for split_name, split_cases in [
            ("train", train_cases),
            ("val", val_cases),
            ("test", test_cases),
        ]:
            split_dir = client_root / split_name
            n_slices = create_symlinks(brats_train, split_dir, split_cases)
            print(f"    {split_name}: {n_slices} slices")

        # Write split case lists for reference
        write_case_list(client_root / "train_cases.txt", train_cases)
        write_case_list(client_root / "val_cases.txt", val_cases)
        write_case_list(client_root / "test_cases.txt", test_cases)

    print(f"\n{'='*60}")
    print("PARTITION CREATED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Output: {args.output_dir}")
    print(f"\nStructure:")
    print(f"  {args.output_dir}/")
    print(f"  ├── client_0_cases.txt")
    print(f"  ├── client_1_cases.txt")
    print(f"  ├── client_map.json")
    print(f"  └── client_data/")
    print(f"      ├── client_0/")
    print(f"      │   ├── train/  (symlinks)")
    print(f"      │   ├── val/    (symlinks)")
    print(f"      │   └── test/   (symlinks)")
    print(f"      └── client_1/")
    print(f"          ├── train/  (symlinks)")
    print(f"          ├── val/    (symlinks)")
    print(f"          └── test/   (symlinks)")


if __name__ == "__main__":
    main()
