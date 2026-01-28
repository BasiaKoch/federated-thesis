#!/usr/bin/env python3
"""
Create a 4-client partition based on ET ratio quartiles.
Each client gets cases from one quartile, creating maximum heterogeneity.

This version COPIES files (not symlinks) for better HPC compatibility.

Usage:
    python make_4client_quartile_partition.py
"""

import json
import shutil
import numpy as np
from pathlib import Path

# Configuration
SEED = 42
VAL_FRAC = 0.1
TEST_FRAC = 0.1
NUM_CLIENTS = 4

np.random.seed(SEED)

# Paths - use absolute paths for HPC compatibility
SCRIPT_DIR = Path(__file__).parent.resolve()
ET_SKEWED_DIR = SCRIPT_DIR / "brats2d_7030_et_skewed"
OUTPUT_DIR = SCRIPT_DIR / "brats2d_4client_quartile"
SOURCE_CLIENT_DATA = ET_SKEWED_DIR / "client_data"


def load_et_ratios():
    """Load ET ratios from existing client map."""
    client_map_path = ET_SKEWED_DIR / "client_map.json"
    print(f"Loading ET ratios from: {client_map_path}")
    if not client_map_path.exists():
        raise FileNotFoundError(f"client_map.json not found at {client_map_path}")
    with open(client_map_path) as f:
        data = json.load(f)
    return data['et_ratios']


def create_quartile_partition(et_ratios):
    """Partition cases into 4 clients based on ET ratio quartiles."""
    sorted_cases = sorted(et_ratios.keys(), key=lambda c: et_ratios[c])
    n = len(sorted_cases)

    q1_idx = n // 4
    q2_idx = n // 2
    q3_idx = 3 * n // 4

    clients = {
        'client_0': sorted_cases[:q1_idx],           # Q1: lowest ET
        'client_1': sorted_cases[q1_idx:q2_idx],     # Q2: low-medium ET
        'client_2': sorted_cases[q2_idx:q3_idx],     # Q3: medium-high ET
        'client_3': sorted_cases[q3_idx:]            # Q4: highest ET
    }
    return clients


def split_train_val_test(cases):
    """Split cases into train/val/test."""
    n = len(cases)
    n_test = max(1, int(n * TEST_FRAC))
    n_val = max(1, int(n * VAL_FRAC))
    n_train = n - n_test - n_val

    shuffled = np.random.permutation(cases).tolist()

    return {
        'train': shuffled[:n_train],
        'val': shuffled[n_train:n_train + n_val],
        'test': shuffled[n_train + n_val:]
    }


def find_source_data(case_id):
    """Find the source data directory for a case in the 2-client partition."""
    # Check both client directories in the source (2-client partition)
    for client_dir in ['client_0', 'client_1']:
        for split in ['train', 'val', 'test']:
            path = SOURCE_CLIENT_DATA / client_dir / split / case_id
            if path.exists() and path.is_dir():
                # Verify it has .npz files
                npz_files = list(path.glob("*.npz"))
                if npz_files:
                    return path
    return None


def copy_case_data(source_dir, dest_dir):
    """Copy all .npz files from source case directory to destination."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    npz_files = list(source_dir.glob("*.npz"))
    for npz_file in npz_files:
        shutil.copy2(npz_file, dest_dir / npz_file.name)
    return len(npz_files)


def create_data_structure(clients, splits, et_ratios):
    """Create the directory structure and COPY data files."""

    output_data_dir = OUTPUT_DIR / "client_data"

    # Clean existing output
    if output_data_dir.exists():
        print(f"Removing existing output directory: {output_data_dir}")
        shutil.rmtree(output_data_dir)

    total_files_copied = 0
    missing_cases = []

    for client_id in sorted(clients.keys()):
        client_splits = splits[client_id]

        for split_name, split_cases in client_splits.items():
            split_dir = output_data_dir / client_id / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            split_files = 0
            for case_id in split_cases:
                source = find_source_data(case_id)
                if source:
                    dest = split_dir / case_id
                    n_files = copy_case_data(source, dest)
                    split_files += n_files
                    total_files_copied += n_files
                else:
                    missing_cases.append(case_id)
                    print(f"  WARNING: Could not find source for {case_id}")

            print(f"  {client_id}/{split_name}: {len(split_cases)} cases, {split_files} files")

    if missing_cases:
        print(f"\nWARNING: {len(missing_cases)} cases not found in source!")
        print(f"  Missing: {missing_cases[:5]}{'...' if len(missing_cases) > 5 else ''}")

    return total_files_copied


def verify_data_structure(num_clients):
    """Verify the created data structure has .npz files."""
    output_data_dir = OUTPUT_DIR / "client_data"

    print("\nVerifying data structure:")
    all_ok = True

    for cid in range(num_clients):
        for split in ['train', 'val', 'test']:
            split_dir = output_data_dir / f"client_{cid}" / split
            if not split_dir.exists():
                print(f"  ERROR: {split_dir} does not exist!")
                all_ok = False
                continue

            # Count .npz files (recursively)
            npz_files = list(split_dir.rglob("*.npz"))
            if not npz_files:
                print(f"  ERROR: No .npz files in {split_dir}")
                all_ok = False
            else:
                print(f"  OK: client_{cid}/{split} has {len(npz_files)} .npz files")

    return all_ok


def main():
    print("=" * 70)
    print("Creating 4-Client Quartile Partition")
    print("=" * 70)
    print(f"Source: {SOURCE_CLIENT_DATA}")
    print(f"Output: {OUTPUT_DIR}")

    # Verify source exists
    if not SOURCE_CLIENT_DATA.exists():
        print(f"\nERROR: Source directory does not exist: {SOURCE_CLIENT_DATA}")
        print("Make sure the 2-client ET-skewed partition exists first.")
        return

    # Load ET ratios
    et_ratios = load_et_ratios()
    print(f"Loaded {len(et_ratios)} cases with ET ratios")

    # Create quartile partition
    clients = create_quartile_partition(et_ratios)

    # Create train/val/test splits for each client
    splits = {}
    for client_id, cases in clients.items():
        splits[client_id] = split_train_val_test(cases)

    # Print statistics
    print("\n" + "-" * 70)
    print("Partition Statistics:")
    print("-" * 70)

    client_stats = {}
    for client_id in sorted(clients.keys()):
        cases = clients[client_id]
        ratios = [et_ratios[c] for c in cases]
        client_splits = splits[client_id]

        stats = {
            'n_cases': len(cases),
            'n_train': len(client_splits['train']),
            'n_val': len(client_splits['val']),
            'n_test': len(client_splits['test']),
            'et_ratio_mean': float(np.mean(ratios)),
            'et_ratio_std': float(np.std(ratios)),
            'et_ratio_min': float(np.min(ratios)),
            'et_ratio_max': float(np.max(ratios))
        }
        client_stats[client_id] = stats

        print(f"\n{client_id}:")
        print(f"  Cases: {stats['n_cases']} (train={stats['n_train']}, val={stats['n_val']}, test={stats['n_test']})")
        print(f"  ET ratio: mean={stats['et_ratio_mean']:.3f}, std={stats['et_ratio_std']:.3f}")
        print(f"  ET range: [{stats['et_ratio_min']:.3f}, {stats['et_ratio_max']:.3f}]")

    # Calculate heterogeneity metrics
    c0_mean = client_stats['client_0']['et_ratio_mean']
    c3_mean = client_stats['client_3']['et_ratio_mean']

    print("\n" + "-" * 70)
    print("Heterogeneity Metrics:")
    print("-" * 70)
    if c0_mean > 0:
        print(f"ET ratio gap (C3/C0): {c3_mean / c0_mean:.1f}x")
    else:
        print(f"ET ratio gap: C0 mean is 0, C3 mean is {c3_mean:.3f}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy data files
    print("\n" + "-" * 70)
    print("Copying data files (this may take a moment)...")
    print("-" * 70)
    total_files = create_data_structure(clients, splits, et_ratios)
    print(f"\nTotal files copied: {total_files}")

    # Verify the structure
    print("\n" + "-" * 70)
    verify_ok = verify_data_structure(NUM_CLIENTS)

    if not verify_ok:
        print("\nERROR: Data verification failed!")
        return

    # Save client map
    client_map = {
        **{client_id: cases for client_id, cases in clients.items()},
        'splits': splits,
        'et_ratios': et_ratios,
        'metadata': {
            'partition_type': '4client_quartile',
            'description': 'Each client gets one quartile of ET ratio distribution',
            'seed': SEED,
            'total_cases': len(et_ratios),
            'num_clients': NUM_CLIENTS,
            'val_frac': VAL_FRAC,
            'test_frac': TEST_FRAC,
            **{f'{cid}_stats': stats for cid, stats in client_stats.items()}
        }
    }

    with open(OUTPUT_DIR / "client_map.json", 'w') as f:
        json.dump(client_map, f, indent=2)

    print(f"\nSaved client_map.json to {OUTPUT_DIR}")

    print("\n" + "=" * 70)
    print("DONE! Partition created successfully.")
    print("=" * 70)
    print(f"""
To use this partition, run:

  STRATEGY=fedavg sbatch federated_unet/unet/run_flower_70_30.sbatch

Or for FedProx:

  STRATEGY=fedprox MU=0.01 sbatch federated_unet/unet/run_flower_70_30.sbatch
""")


if __name__ == "__main__":
    main()
