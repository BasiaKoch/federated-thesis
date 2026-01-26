#!/usr/bin/env python3
"""
Rebuild client_data directory with proper symlinks for 2-client federated partitions.

All symlinks point directly to the original BraTS dataset (not through train/).
Creates train/val/test splits by MOVING case folders (which contain symlinks),
so symlinks remain valid.

Usage:
    python rebuild_client_data.py \
        --partition_dir /home/bk489/federated/federated-thesis/data/partitions/federated_clients_2_70_30 \
        --brats_root /home/bk489/federated/federated-thesis/data/brats2020_top10_slices_split_npz \
        --val_frac 0.10 --test_frac 0.10 --seed 42 --rebuild
"""
import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def read_cases_from_txt(txt: Path) -> List[str]:
    """Read case IDs from a text file (one per line)."""
    return [line.strip() for line in txt.read_text().splitlines() if line.strip()]


def read_cases_from_json(json_path: Path, client_key: str) -> List[str]:
    """Read case IDs from client_map.json."""
    data = json.loads(json_path.read_text())
    return data.get(client_key, [])


def ensure_symlink(src: Path, dst: Path) -> None:
    """Create symlink dst -> src (skip if exists)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src)


def split_cases(
    cases: List[str], seed: int, val_frac: float, test_frac: float
) -> Tuple[List[str], List[str], List[str]]:
    """Split case IDs into train/val/test."""
    rng = random.Random(seed)
    cases = sorted(cases)
    rng.shuffle(cases)
    n = len(cases)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = sorted(cases[:n_test])
    val = sorted(cases[n_test : n_test + n_val])
    train = sorted(cases[n_test + n_val :])
    return train, val, test


def build_train_symlinks(
    client_train_dir: Path, brats_train_dir: Path, cases: List[str]
) -> None:
    """Create symlinks in client_train_dir pointing to original BraTS data."""
    for case_id in cases:
        src_case = brats_train_dir / case_id
        if not src_case.exists():
            raise FileNotFoundError(f"Missing case in BraTS train: {src_case}")
        dst_case = client_train_dir / case_id
        dst_case.mkdir(parents=True, exist_ok=True)
        npzs = list(src_case.glob("*.npz"))
        if not npzs:
            raise RuntimeError(f"No .npz files in {src_case}")
        for f in npzs:
            ensure_symlink(f, dst_case / f.name)


def move_cases(
    client_root: Path, case_ids: List[str], src_split: str, dst_split: str
) -> None:
    """Move entire case folders from src_split to dst_split."""
    src = client_root / src_split
    dst = client_root / dst_split
    dst.mkdir(parents=True, exist_ok=True)
    for case_id in case_ids:
        src_case = src / case_id
        if not src_case.exists():
            raise FileNotFoundError(f"Expected case dir missing: {src_case}")
        shutil.move(str(src_case), str(dst / case_id))


def write_list(p: Path, xs: List[str]) -> None:
    """Write list of strings to file (one per line)."""
    p.write_text("\n".join(xs) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Rebuild client_data with proper symlinks"
    )
    ap.add_argument(
        "--partition_dir",
        type=Path,
        required=True,
        help="Path to partition dir (e.g., .../data/partitions/federated_clients_2_70_30)",
    )
    ap.add_argument(
        "--brats_root",
        type=Path,
        required=True,
        help="Path to BraTS data root (must contain train/ with case folders)",
    )
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing client_data and rebuild from scratch",
    )
    args = ap.parse_args()

    part = args.partition_dir
    client_data = part / "client_data"
    brats_train = args.brats_root / "train"

    if not brats_train.exists():
        raise FileNotFoundError(f"BraTS train dir not found: {brats_train}")

    # Try to find case lists - check multiple locations
    client_cases = {}

    # Option 1: client_X_cases.txt files
    c0_txt = part / "client_0_cases.txt"
    c1_txt = part / "client_1_cases.txt"

    # Option 2: client_map.json
    client_map_json = part / "client_map.json"

    if c0_txt.exists() and c1_txt.exists():
        client_cases[0] = read_cases_from_txt(c0_txt)
        client_cases[1] = read_cases_from_txt(c1_txt)
        print(f"Read case lists from {c0_txt.name} and {c1_txt.name}")
    elif client_map_json.exists():
        client_cases[0] = read_cases_from_json(client_map_json, "client_0")
        client_cases[1] = read_cases_from_json(client_map_json, "client_1")
        print(f"Read case lists from {client_map_json.name}")
    else:
        raise FileNotFoundError(
            f"Could not find case lists. Expected either:\n"
            f"  - {c0_txt} and {c1_txt}, or\n"
            f"  - {client_map_json}"
        )

    if not client_cases[0] or not client_cases[1]:
        raise RuntimeError("Empty case lists for one or both clients")

    print(f"Client 0: {len(client_cases[0])} cases")
    print(f"Client 1: {len(client_cases[1])} cases")

    # Delete and recreate client_data if --rebuild
    if args.rebuild and client_data.exists():
        print(f"Removing existing client_data: {client_data}")
        shutil.rmtree(client_data)

    client_data.mkdir(parents=True, exist_ok=True)

    for cid in [0, 1]:
        client_root = client_data / f"client_{cid}"
        train_dir = client_root / "train"
        val_dir = client_root / "val"
        test_dir = client_root / "test"

        # Clean existing split dirs if present
        for d in [train_dir, val_dir, test_dir]:
            if d.exists():
                shutil.rmtree(d)

        train_dir.mkdir(parents=True, exist_ok=True)

        # 1) Create symlinks for ALL cases in train/ first
        all_cases = client_cases[cid]
        print(f"\nClient {cid}: Creating symlinks for {len(all_cases)} cases...")
        build_train_symlinks(train_dir, brats_train, all_cases)

        # 2) Split cases and MOVE whole case folders to val/test
        tr_cases, va_cases, te_cases = split_cases(
            all_cases,
            seed=args.seed + cid,  # slightly different seed per client
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )

        print(f"  Split: train={len(tr_cases)} val={len(va_cases)} test={len(te_cases)}")

        # Move val cases from train -> val
        move_cases(client_root, va_cases, "train", "val")
        # Move test cases from train -> test
        move_cases(client_root, te_cases, "train", "test")

        # 3) Write split case lists (useful for debugging)
        write_list(client_root / "train_cases.txt", tr_cases)
        write_list(client_root / "val_cases.txt", va_cases)
        write_list(client_root / "test_cases.txt", te_cases)

        print(f"  Done: {client_root}")

    print(f"\nRebuilt client_data at: {client_data}")
    print("\nVerify with:")
    print(f"  ls -la {client_data}/client_0/val/*/")
    print(f"  stat {client_data}/client_0/val/*/*slice*.npz | head")


if __name__ == "__main__":
    main()
