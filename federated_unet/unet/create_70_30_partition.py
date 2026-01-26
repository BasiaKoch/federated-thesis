#!/usr/bin/env python3
"""
Create 70/30 client partition for federated learning.

This script creates the initial case assignment for 2 clients:
- Client 0: 70% of training cases
- Client 1: 30% of training cases

Following the reference repo practice:
https://github.com/nedeljkovicmajaa/Federated-Learning-And-Class-Imbalances

Usage:
    python create_70_30_partition.py \
        --brats_root /path/to/brats2020_top10_slices_split_npz \
        --output_dir /path/to/partitions/federated_clients_2_70_30 \
        --split 0.7 0.3 \
        --seed 42
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple


def load_train_cases(brats_root: Path) -> List[str]:
    """Load training case IDs from BraTS data or split.json."""
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
        cases = [d.name for d in train_dir.iterdir() if d.is_dir() and d.name.startswith("BraTS")]
        print(f"Loaded {len(cases)} training cases from train/ directory")
        return sorted(cases)

    raise FileNotFoundError(f"Could not find training cases in {brats_root}")


def partition_cases(
    cases: List[str],
    split_ratios: List[float],
    seed: int,
) -> Tuple[List[str], List[str]]:
    """
    Partition cases into two clients based on split ratios.

    Args:
        cases: List of case IDs
        split_ratios: [client_0_ratio, client_1_ratio] e.g., [0.7, 0.3]
        seed: Random seed for reproducibility

    Returns:
        (client_0_cases, client_1_cases)
    """
    assert len(split_ratios) == 2, "Expected 2 split ratios for 2 clients"
    assert abs(sum(split_ratios) - 1.0) < 0.01, "Split ratios must sum to 1.0"

    rng = random.Random(seed)
    cases = sorted(cases)  # Ensure deterministic order
    rng.shuffle(cases)

    n = len(cases)
    n_client0 = int(round(n * split_ratios[0]))

    client_0_cases = sorted(cases[:n_client0])
    client_1_cases = sorted(cases[n_client0:])

    return client_0_cases, client_1_cases


def main():
    ap = argparse.ArgumentParser(description="Create 70/30 client partition for federated learning")
    ap.add_argument(
        "--brats_root",
        type=Path,
        required=True,
        help="Path to BraTS data root (containing split.json or train/)",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for partition files",
    )
    ap.add_argument(
        "--split",
        type=float,
        nargs=2,
        default=[0.7, 0.3],
        help="Split ratios for client_0 and client_1 (default: 0.7 0.3)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = ap.parse_args()

    # Load training cases
    cases = load_train_cases(args.brats_root)
    print(f"Total training cases: {len(cases)}")

    # Create partition
    client_0_cases, client_1_cases = partition_cases(cases, args.split, args.seed)

    print(f"\nPartition (seed={args.seed}):")
    print(f"  Client 0: {len(client_0_cases)} cases ({len(client_0_cases)/len(cases)*100:.1f}%)")
    print(f"  Client 1: {len(client_1_cases)} cases ({len(client_1_cases)/len(cases)*100:.1f}%)")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write case lists (for rebuild_client_data.py)
    c0_file = args.output_dir / "client_0_cases.txt"
    c1_file = args.output_dir / "client_1_cases.txt"

    c0_file.write_text("\n".join(client_0_cases) + "\n")
    c1_file.write_text("\n".join(client_1_cases) + "\n")

    print(f"\nWritten: {c0_file}")
    print(f"Written: {c1_file}")

    # Also write client_map.json (alternative format)
    client_map = {
        "client_0": client_0_cases,
        "client_1": client_1_cases,
        "metadata": {
            "split_ratios": args.split,
            "seed": args.seed,
            "total_cases": len(cases),
        }
    }
    map_file = args.output_dir / "client_map.json"
    map_file.write_text(json.dumps(client_map, indent=2))
    print(f"Written: {map_file}")

    # Print summary for verification
    print(f"\n{'='*60}")
    print("PARTITION SUMMARY")
    print(f"{'='*60}")
    print(f"Client 0 (70%): {len(client_0_cases)} cases")
    print(f"  First 5: {client_0_cases[:5]}")
    print(f"  Last 5:  {client_0_cases[-5:]}")
    print(f"\nClient 1 (30%): {len(client_1_cases)} cases")
    print(f"  First 5: {client_1_cases[:5]}")
    print(f"  Last 5:  {client_1_cases[-5:]}")
    print(f"{'='*60}")

    print(f"\nNext step: Run rebuild_client_data.py to create symlinks:")
    print(f"  python rebuild_client_data.py \\")
    print(f"    --partition_dir {args.output_dir} \\")
    print(f"    --brats_root {args.brats_root} \\")
    print(f"    --val_frac 0.10 --test_frac 0.10 --seed 42 --rebuild")


if __name__ == "__main__":
    main()
