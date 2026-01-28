"""
Create a 4-client partition based on ET ratio quartiles.
Each client gets cases from one quartile, creating maximum heterogeneity.

Usage:
    python make_4client_quartile_partition.py
"""
#:

import json
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuration
SEED = 42
VAL_FRAC = 0.1
TEST_FRAC = 0.1
NUM_CLIENTS = 4

np.random.seed(SEED)

# Paths
SCRIPT_DIR = Path(__file__).parent
ET_SKEWED_DIR = SCRIPT_DIR / "brats2d_7030_et_skewed"
OUTPUT_DIR = SCRIPT_DIR / "brats2d_4client_quartile"
SOURCE_CLIENT_DATA = ET_SKEWED_DIR / "client_data"

def load_et_ratios():
    """Load ET ratios from existing client map."""
    with open(ET_SKEWED_DIR / "client_map.json") as f:
        data = json.load(f)
    return data['et_ratios']

def create_quartile_partition(et_ratios):
    """Partition cases into 4 clients based on ET ratio quartiles."""

    # Sort cases by ET ratio
    sorted_cases = sorted(et_ratios.keys(), key=lambda c: et_ratios[c])
    n = len(sorted_cases)

    # Calculate quartile boundaries
    q1_idx = n // 4
    q2_idx = n // 2
    q3_idx = 3 * n // 4

    # Assign to clients
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
    """Find the source data directory for a case."""
    # Check both client directories in the source
    for client_dir in ['client_0', 'client_1']:
        for split in ['train', 'val', 'test']:
            path = SOURCE_CLIENT_DATA / client_dir / split / case_id
            if path.exists():
                return path
    return None

def create_symlinks(clients, splits, et_ratios):
    """Create the directory structure with symlinks to actual data."""

    output_data_dir = OUTPUT_DIR / "client_data"

    # Clean existing output
    if output_data_dir.exists():
        shutil.rmtree(output_data_dir)

    for client_id, cases in clients.items():
        client_splits = splits[client_id]

        for split_name, split_cases in client_splits.items():
            split_dir = output_data_dir / client_id / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for case_id in split_cases:
                source = find_source_data(case_id)
                if source:
                    dest = split_dir / case_id
                    # Create symlink
                    dest.symlink_to(source.resolve())
                else:
                    print(f"Warning: Could not find source for {case_id}")

def main():
    print("=" * 70)
    print("Creating 4-Client Quartile Partition")
    print("=" * 70)

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
    print(f"ET ratio gap (C3/C0): {c3_mean / c0_mean:.1f}x")
    print(f"This is {'HIGH' if c3_mean / c0_mean > 10 else 'VERY HIGH' if c3_mean / c0_mean > 20 else 'MODERATE'} heterogeneity")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create symlinks to data
    print("\n" + "-" * 70)
    print("Creating data symlinks...")
    create_symlinks(clients, splits, et_ratios)

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
    print(f"Data symlinks created in {OUTPUT_DIR / 'client_data'}")

    print("\n" + "=" * 70)
    print("DONE! To use this partition:")
    print("=" * 70)
    print(f"""
1. Update your training script to use 4 clients
2. Set partition_dir to: {OUTPUT_DIR / 'client_data'}
3. Recommended hyperparameters for 4 clients:
   - local_epochs: 3-5 (less data per client)
   - rounds: 30-50
   - lr: 0.01 (same as before)
   - batch_size: 8-10
""")

if __name__ == "__main__":
    main()
