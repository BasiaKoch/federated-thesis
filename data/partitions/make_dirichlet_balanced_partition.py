#!/usr/bin/env python3
"""
Create BALANCED Dirichlet-based Non-IID Partition for Federated Learning.

This version ensures EQUAL client sizes while maintaining label heterogeneity.
This is important because FedAvg weights updates by sample count, which can
mask the effects of heterogeneity when client sizes differ significantly.

Key difference from standard Dirichlet:
- Standard: Dirichlet naturally creates unequal client sizes
- Balanced: Each client gets N/K cases, but the COMPOSITION varies by Dirichlet

Algorithm:
    1. Compute ET ratio for each case
    2. Bin cases into K bins based on ET ratio quantiles
    3. For each client, sample from Dirichlet(α) to get bin proportions
    4. Allocate n_cases/n_clients cases to each client according to proportions
    5. This creates heterogeneity in label distribution while keeping sizes equal

Usage:
    python make_dirichlet_balanced_partition.py --alpha 0.5 --num_clients 4
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy.stats import entropy, ks_2samp


SEED = 42
VAL_FRAC = 0.1
TEST_FRAC = 0.1
NUM_ET_BINS = 4


def load_mask(npz_path: Path) -> np.ndarray:
    """Load the mask array from an NPZ file."""
    d = np.load(str(npz_path), allow_pickle=False)
    for k in ["mask", "y", "Y", "seg", "label", "labels"]:
        if k in d:
            return d[k]
    for k in d:
        if d[k].dtype.kind in ("i", "u"):
            return d[k]
    raise KeyError(f"No mask found in {npz_path}, keys={list(d.keys())}")


def scan_all_cases(source_dir: Path) -> Dict[str, Tuple[float, Path]]:
    """Scan all unique cases from source directory."""
    case_info = {}
    for client_dir in sorted(source_dir.iterdir()):
        if not client_dir.is_dir() or not client_dir.name.startswith("client_"):
            continue
        for split in ["train", "val", "test"]:
            split_dir = client_dir / split
            if not split_dir.exists():
                continue
            for case_dir in sorted(split_dir.iterdir()):
                if not case_dir.is_dir() or not case_dir.name.startswith("BraTS"):
                    continue
                if case_dir.name in case_info:
                    continue

                npz_files = list(case_dir.glob("*.npz"))
                if not npz_files:
                    continue

                total_et = 0
                total_wt = 0
                for npz_path in npz_files:
                    real_path = npz_path.resolve() if npz_path.is_symlink() else npz_path
                    mask = load_mask(real_path)
                    m = mask.astype(np.int32).ravel()
                    total_wt += np.count_nonzero(m > 0)
                    total_et += np.count_nonzero(m == 4)

                et_ratio = total_et / (total_wt + 1)
                case_info[case_dir.name] = (et_ratio, case_dir)

    return case_info


def dirichlet_balanced_partition(
    case_info: Dict[str, Tuple[float, Path]],
    num_clients: int,
    alpha: float,
    num_bins: int,
    seed: int
) -> Dict[str, List[str]]:
    """
    Create balanced partition with Dirichlet-based label heterogeneity.

    Each client gets exactly N/K cases, but the label distribution varies.
    """
    rng = np.random.RandomState(seed)

    # Sort cases by ET ratio and create bins
    sorted_cases = sorted(case_info.keys(), key=lambda c: case_info[c][0])
    n_total = len(sorted_cases)

    # Create bins based on ET ratio quantiles
    bin_edges = np.linspace(0, n_total, num_bins + 1, dtype=int)
    bins = []
    for i in range(num_bins):
        bin_cases = sorted_cases[bin_edges[i]:bin_edges[i+1]]
        bins.append(list(bin_cases))

    print(f"\n  ET Ratio Bins:")
    for i, bin_cases in enumerate(bins):
        if bin_cases:
            ratios = [case_info[c][0] for c in bin_cases]
            print(f"    Bin {i}: {len(bin_cases)} cases, ET range [{min(ratios):.3f}, {max(ratios):.3f}]")

    # Calculate target cases per client (balanced)
    cases_per_client = n_total // num_clients
    remainder = n_total % num_clients

    print(f"\n  Target: {cases_per_client} cases per client (remainder: {remainder})")

    # For each client, sample Dirichlet proportions and allocate from bins
    client_cases = {f"client_{i}": [] for i in range(num_clients)}

    # Track which cases have been assigned
    available_in_bin = [set(b) for b in bins]

    print(f"\n  Dirichlet allocation (α={alpha}):")

    for client_idx in range(num_clients):
        # Sample proportions from Dirichlet for this client
        proportions = rng.dirichlet([alpha] * num_bins)

        # How many cases this client gets (distribute remainder)
        n_client = cases_per_client + (1 if client_idx < remainder else 0)

        # Calculate target from each bin
        targets = (proportions * n_client).astype(int)

        # Adjust for rounding
        diff = n_client - targets.sum()
        if diff > 0:
            # Add to bins with most available
            for _ in range(diff):
                available_counts = [len(available_in_bin[b]) for b in range(num_bins)]
                best_bin = np.argmax(available_counts)
                targets[best_bin] += 1
        elif diff < 0:
            # Remove from bins with fewest targets
            for _ in range(-diff):
                nonzero_bins = [b for b in range(num_bins) if targets[b] > 0]
                if nonzero_bins:
                    best_bin = nonzero_bins[np.argmin([targets[b] for b in nonzero_bins])]
                    targets[best_bin] -= 1

        # Allocate cases from each bin
        allocated = []
        for bin_idx in range(num_bins):
            n_from_bin = min(targets[bin_idx], len(available_in_bin[bin_idx]))

            if n_from_bin > 0:
                # Randomly sample from available cases in this bin
                available_list = list(available_in_bin[bin_idx])
                rng.shuffle(available_list)
                selected = available_list[:n_from_bin]
                allocated.extend(selected)

                # Remove from available
                for c in selected:
                    available_in_bin[bin_idx].remove(c)

        # If we didn't get enough (due to bin depletion), take from any available bin
        while len(allocated) < n_client:
            for bin_idx in range(num_bins):
                if available_in_bin[bin_idx] and len(allocated) < n_client:
                    c = available_in_bin[bin_idx].pop()
                    allocated.append(c)

        client_cases[f"client_{client_idx}"] = allocated

        # Print allocation
        actual_per_bin = [sum(1 for c in allocated if c in bins[b]) for b in range(num_bins)]
        print(f"    Client {client_idx}: proportions {proportions.round(2)} → bins {actual_per_bin} → total {len(allocated)}")

    return client_cases


def split_train_val_test(
    cases: List[str],
    seed: int,
    val_frac: float = 0.10,
    test_frac: float = 0.10
) -> Dict[str, List[str]]:
    """Split cases into train/val/test sets."""
    rng = np.random.RandomState(seed)
    cases = list(cases)
    rng.shuffle(cases)

    n = len(cases)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_train = n - n_test - n_val

    if n_train < 1:
        n_train = 1
        n_val = max(1, (n - 1) // 2)
        n_test = n - n_train - n_val

    return {
        "train": sorted(cases[:n_train]),
        "val": sorted(cases[n_train:n_train + n_val]),
        "test": sorted(cases[n_train + n_val:])
    }


def compute_jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute JSD between two distributions."""
    eps = 1e-10
    p = np.array(p) + eps
    q = np.array(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2
    return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))


def compute_heterogeneity_metrics(
    client_cases: Dict[str, List[str]],
    case_info: Dict[str, Tuple[float, Path]],
    num_bins: int = 10
) -> Dict[str, float]:
    """Compute heterogeneity metrics."""
    clients = list(client_cases.keys())
    n_clients = len(clients)

    client_et_ratios = {}
    for cid in clients:
        client_et_ratios[cid] = [case_info[c][0] for c in client_cases[cid]]

    all_ratios = [case_info[c][0] for c in case_info.keys()]
    bin_edges = np.linspace(min(all_ratios), max(all_ratios) + 1e-6, num_bins + 1)

    client_hists = {}
    for cid in clients:
        hist, _ = np.histogram(client_et_ratios[cid], bins=bin_edges)
        client_hists[cid] = hist

    jsd_values = []
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            jsd = compute_jensen_shannon_divergence(
                client_hists[clients[i]], client_hists[clients[j]]
            )
            jsd_values.append(jsd)

    ks_values = []
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            ks_stat, _ = ks_2samp(
                client_et_ratios[clients[i]], client_et_ratios[clients[j]]
            )
            ks_values.append(ks_stat)

    client_means = [np.mean(client_et_ratios[cid]) for cid in clients if client_et_ratios[cid]]
    et_gap = max(client_means) / (min(client_means) + 1e-10)

    return {
        "mean_jsd": float(np.mean(jsd_values)) if jsd_values else 0.0,
        "max_jsd": float(np.max(jsd_values)) if jsd_values else 0.0,
        "mean_ks": float(np.mean(ks_values)) if ks_values else 0.0,
        "max_ks": float(np.max(ks_values)) if ks_values else 0.0,
        "et_gap": float(et_gap)
    }


def copy_case_data(source_dir: Path, dest_dir: Path) -> int:
    """Copy all .npz files from source to destination."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    npz_files = list(source_dir.glob("*.npz"))
    for npz_file in npz_files:
        real_path = npz_file.resolve() if npz_file.is_symlink() else npz_file
        shutil.copy2(real_path, dest_dir / npz_file.name)
    return len(npz_files)


def verify_no_overlap(splits: Dict[str, Dict[str, List[str]]]) -> bool:
    """Verify no case appears multiple times."""
    all_cases = []
    for _, client_splits in splits.items():
        for _, cases in client_splits.items():
            for case in cases:
                if case in all_cases:
                    print(f"ERROR: Case {case} appears multiple times!")
                    return False
                all_cases.append(case)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create BALANCED Dirichlet-based Non-IID Partition"
    )
    parser.add_argument(
        "--source_dir", type=Path,
        default=Path(__file__).parent / "brats2d_one_slice_per_patient_clients"
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--num_clients", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num_bins", type=int, default=NUM_ET_BINS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val_frac", type=float, default=VAL_FRAC)
    parser.add_argument("--test_frac", type=float, default=TEST_FRAC)

    args = parser.parse_args()

    if args.output_dir is None:
        alpha_str = f"{args.alpha:.1f}".replace(".", "p")
        args.output_dir = Path(__file__).parent / f"brats2d_{args.num_clients}client_dirichlet_balanced_a{alpha_str}"

    print("=" * 70)
    print("BALANCED DIRICHLET NON-IID PARTITION")
    print("=" * 70)
    print(f"Source:      {args.source_dir}")
    print(f"Output:      {args.output_dir}")
    print(f"Clients:     {args.num_clients}")
    print(f"Dirichlet α: {args.alpha}")
    print(f"\nKey feature: ALL CLIENTS HAVE EQUAL SIZE (label heterogeneity only)")

    if args.alpha <= 0.1:
        mu_rec = "0.1, 0.4, or 0.7"
        het_level = "EXTREME"
    elif args.alpha <= 0.5:
        mu_rec = "0.05, 0.1, or 0.3"
        het_level = "MODERATE"
    else:
        mu_rec = "0.01 or 0.05"
        het_level = "MILD"

    print(f"Heterogeneity level: {het_level}")
    print(f"Recommended FedProx μ: {mu_rec}")

    # Step 1: Scan cases
    print("\n" + "-" * 70)
    print("Step 1: Scanning source data...")
    print("-" * 70)

    case_info = scan_all_cases(args.source_dir)
    print(f"\nFound {len(case_info)} unique cases")

    # Step 2: Balanced Dirichlet partitioning
    print("\n" + "-" * 70)
    print("Step 2: Creating BALANCED Dirichlet partition...")
    print("-" * 70)

    client_cases = dirichlet_balanced_partition(
        case_info, args.num_clients, args.alpha, args.num_bins, args.seed
    )

    # Step 3: Train/val/test splits
    print("\n" + "-" * 70)
    print("Step 3: Creating train/val/test splits...")
    print("-" * 70)

    splits = {}
    for client_id, cases in sorted(client_cases.items()):
        client_seed = args.seed + int(client_id.split("_")[1])
        splits[client_id] = split_train_val_test(cases, client_seed, args.val_frac, args.test_frac)
        s = splits[client_id]
        print(f"  {client_id}: train={len(s['train'])} val={len(s['val'])} test={len(s['test'])} (total={len(cases)})")

    print("\n  Verifying no overlap...")
    if verify_no_overlap(splits):
        print("  ✓ No overlap detected")
    else:
        return

    # Step 4: Heterogeneity metrics
    print("\n" + "-" * 70)
    print("Step 4: Computing heterogeneity metrics...")
    print("-" * 70)

    het_metrics = compute_heterogeneity_metrics(client_cases, case_info)

    print(f"\n  Jensen-Shannon Divergence: Mean={het_metrics['mean_jsd']:.4f}, Max={het_metrics['max_jsd']:.4f}")
    print(f"  Kolmogorov-Smirnov Stat:   Mean={het_metrics['mean_ks']:.4f}, Max={het_metrics['max_ks']:.4f}")
    print(f"  ET Ratio Gap (max/min):    {het_metrics['et_gap']:.2f}x")

    # Step 5: Copy data
    print("\n" + "-" * 70)
    print("Step 5: Copying data files...")
    print("-" * 70)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    for client_id in sorted(splits.keys()):
        for split_name, case_list in splits[client_id].items():
            dest_dir = args.output_dir / "client_data" / client_id / split_name
            for case_id in case_list:
                _, source_path = case_info[case_id]
                n = copy_case_data(source_path, dest_dir / case_id)
                total_files += n

    print(f"  Total files copied: {total_files}")

    # Step 6: Save metadata
    print("\n" + "-" * 70)
    print("Step 6: Saving metadata...")
    print("-" * 70)

    client_stats = {}
    for client_id, cases in sorted(client_cases.items()):
        et_ratios = [case_info[c][0] for c in cases]
        client_stats[client_id] = {
            "n_cases": len(cases),
            "n_train": len(splits[client_id]["train"]),
            "n_val": len(splits[client_id]["val"]),
            "n_test": len(splits[client_id]["test"]),
            "et_ratio_mean": float(np.mean(et_ratios)),
            "et_ratio_std": float(np.std(et_ratios)),
            "et_ratio_min": float(np.min(et_ratios)),
            "et_ratio_max": float(np.max(et_ratios))
        }

    client_map = {
        **{client_id: cases for client_id, cases in client_cases.items()},
        "splits": splits,
        "et_ratios": {c: float(info[0]) for c, info in case_info.items()},
        "metadata": {
            "partition_type": "dirichlet_balanced",
            "description": f"Balanced Dirichlet partition with α={args.alpha} (equal client sizes)",
            "dirichlet_alpha": args.alpha,
            "num_clients": args.num_clients,
            "num_bins": args.num_bins,
            "seed": args.seed,
            "total_cases": len(case_info),
            "heterogeneity_metrics": het_metrics,
            "recommended_mu": mu_rec,
            **{f"{cid}_stats": stats for cid, stats in client_stats.items()}
        }
    }

    with open(args.output_dir / "client_map.json", "w") as f:
        json.dump(client_map, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("PARTITION SUMMARY")
    print("=" * 70)
    print(f"\nBalanced partition with α={args.alpha}")
    print(f"All clients have approximately equal size (label heterogeneity only)\n")

    for client_id, stats in sorted(client_stats.items()):
        print(f"{client_id}: {stats['n_cases']} cases, ET mean={stats['et_ratio_mean']:.3f} ± {stats['et_ratio_std']:.3f}")

    print(f"\nHeterogeneity: JSD={het_metrics['mean_jsd']:.3f}, KS={het_metrics['mean_ks']:.3f}, Gap={het_metrics['et_gap']:.1f}x")

    print(f"""
Training commands:

# FedAvg:
STRATEGY=fedavg NUM_CLIENTS={args.num_clients} \\
PARTITION_DIR="{args.output_dir / 'client_data'}" \\
sbatch federated_unet/unet/run_flower_70_30.sbatch

# FedProx:
STRATEGY=fedprox MU=0.1 NUM_CLIENTS={args.num_clients} \\
PARTITION_DIR="{args.output_dir / 'client_data'}" \\
sbatch federated_unet/unet/run_flower_70_30.sbatch
""")

    print("=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
