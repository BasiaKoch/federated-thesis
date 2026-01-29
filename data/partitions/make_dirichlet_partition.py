#!/usr/bin/env python3
"""
Create Dirichlet-based Non-IID Partition for Federated Learning on BraTS Data.

This implements the standard Dirichlet distribution method for creating
label-heterogeneous partitions, as used in peer-reviewed federated learning literature.

References:
- Hsu et al. (2019) "Measuring the Effects of Non-Identical Data Distribution for FL"
- Li et al. (2022) "Federated Learning on Non-IID Data Silos: An Experimental Study"
- Wang et al. (2023) taxonomy paper: Dirichlet is used in 27% of non-IID FL papers

The Dirichlet parameter α controls heterogeneity:
    α = 0.1  → Extreme non-IID (each client dominated by 1-2 ET bins)
    α = 0.5  → Moderate non-IID (recommended for FedProx demonstration)
    α = 1.0  → Mild non-IID
    α → ∞   → Approaches IID

Algorithm:
    1. Compute ET ratio for each case
    2. Bin cases into K bins based on ET ratio quantiles
    3. For each bin, sample from Dirichlet(α) to get client proportions
    4. Allocate cases to clients based on these proportions
    5. Split each client's data into train/val/test (no overlap)
    6. Compute heterogeneity metrics (JSD, KS statistic) for thesis reporting

Usage:
    # Moderate heterogeneity (recommended start)
    python make_dirichlet_partition.py --alpha 0.5 --num_clients 4

    # Extreme heterogeneity (needs higher μ for FedProx)
    python make_dirichlet_partition.py --alpha 0.1 --num_clients 4

    # Mild heterogeneity
    python make_dirichlet_partition.py --alpha 1.0 --num_clients 4
"""
#check
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import entropy, ks_2samp


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

SEED = 42
VAL_FRAC = 0.1
TEST_FRAC = 0.1
NUM_ET_BINS = 4  # Number of bins for ET ratio (quartiles)


# ═══════════════════════════════════════════════════════════════════════════════
# ET Ratio Computation
# ═══════════════════════════════════════════════════════════════════════════════

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


def scan_all_cases(source_dir: Path) -> Dict[str, Tuple[float, Path]]:
    """
    Scan all unique cases from source directory.

    Returns: {case_id: (et_ratio, source_path)}
    """
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
                    continue  # Skip duplicates

                npz_files = list(case_dir.glob("*.npz"))
                if not npz_files:
                    print(f"  WARNING: no .npz in {case_dir}, skipping")
                    continue

                # Compute ET ratio
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


# ═══════════════════════════════════════════════════════════════════════════════
# Dirichlet Partitioning (Literature-Standard Method)
# ═══════════════════════════════════════════════════════════════════════════════

def dirichlet_partition(
    case_info: Dict[str, Tuple[float, Path]],
    num_clients: int,
    alpha: float,
    num_bins: int,
    seed: int
) -> Dict[str, List[str]]:
    """
    Partition cases using Dirichlet distribution over ET ratio bins.

    This is the standard method used in FL literature for creating non-IID data.

    Algorithm:
    1. Bin cases by ET ratio into `num_bins` quantile-based bins
    2. For each bin, sample proportions from Dirichlet(α, α, ..., α)
    3. Allocate cases in that bin to clients according to proportions

    Lower α → More heterogeneous (clients specialize in certain bins)
    Higher α → More homogeneous (clients get balanced samples from all bins)

    Args:
        case_info: {case_id: (et_ratio, source_path)}
        num_clients: Number of clients to partition to
        alpha: Dirichlet concentration parameter
        num_bins: Number of ET ratio bins
        seed: Random seed for reproducibility

    Returns:
        {client_id: [case_ids]}
    """
    rng = np.random.RandomState(seed)

    # Sort cases by ET ratio
    sorted_cases = sorted(case_info.keys(), key=lambda c: case_info[c][0])
    n_cases = len(sorted_cases)

    # Create bins based on ET ratio quantiles
    bin_edges = np.linspace(0, n_cases, num_bins + 1, dtype=int)
    bins = []
    for i in range(num_bins):
        bin_cases = sorted_cases[bin_edges[i]:bin_edges[i+1]]
        bins.append(bin_cases)

    print(f"\n  ET Ratio Bins (based on {num_bins} quantiles):")
    for i, bin_cases in enumerate(bins):
        if bin_cases:
            ratios = [case_info[c][0] for c in bin_cases]
            print(f"    Bin {i}: {len(bin_cases)} cases, ET range [{min(ratios):.3f}, {max(ratios):.3f}]")

    # Initialize client case lists
    client_cases = {f"client_{i}": [] for i in range(num_clients)}

    # For each bin, sample Dirichlet proportions and allocate cases
    print(f"\n  Dirichlet allocation (α={alpha}):")
    for bin_idx, bin_cases in enumerate(bins):
        if not bin_cases:
            continue

        # Sample proportions from Dirichlet(α, α, ..., α)
        proportions = rng.dirichlet([alpha] * num_clients)

        # Shuffle cases within bin
        bin_cases_shuffled = list(bin_cases)
        rng.shuffle(bin_cases_shuffled)

        # Allocate cases to clients based on proportions
        # Use cumulative proportions to determine split points
        cumulative = np.cumsum(proportions)
        cumulative[-1] = 1.0  # Ensure last client gets remaining cases

        n_bin = len(bin_cases_shuffled)
        split_indices = (cumulative * n_bin).astype(int)

        prev_idx = 0
        for client_idx in range(num_clients):
            end_idx = split_indices[client_idx]
            allocated = bin_cases_shuffled[prev_idx:end_idx]
            client_cases[f"client_{client_idx}"].extend(allocated)
            prev_idx = end_idx

        # Print allocation for this bin
        allocations = [(split_indices[0] if i == 0 else split_indices[i] - split_indices[i-1])
                       for i in range(num_clients)]
        print(f"    Bin {bin_idx}: proportions {proportions.round(2)} → allocations {allocations}")

    return client_cases


# ═══════════════════════════════════════════════════════════════════════════════
# Train/Val/Test Splitting
# ═══════════════════════════════════════════════════════════════════════════════

def split_train_val_test(
    cases: List[str],
    seed: int,
    val_frac: float = 0.10,
    test_frac: float = 0.10
) -> Dict[str, List[str]]:
    """
    Split cases into train/val/test sets.
    Ensures no overlap between splits.
    """
    rng = np.random.RandomState(seed)
    cases = list(cases)
    rng.shuffle(cases)

    n = len(cases)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_train = n - n_test - n_val

    # Ensure at least 1 case in train
    if n_train < 1:
        n_train = 1
        n_val = max(1, (n - 1) // 2)
        n_test = n - n_train - n_val

    return {
        "train": sorted(cases[:n_train]),
        "val": sorted(cases[n_train:n_train + n_val]),
        "test": sorted(cases[n_train + n_val:])
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Heterogeneity Metrics (For Thesis Reporting)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    JSD ∈ [0, 1] where 0 = identical, 1 = completely different
    """
    # Add small epsilon to avoid log(0)
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
    """
    Compute heterogeneity metrics for thesis reporting.

    Returns:
        - mean_jsd: Mean pairwise Jensen-Shannon divergence
        - max_jsd: Maximum pairwise JSD
        - mean_ks: Mean pairwise Kolmogorov-Smirnov statistic
        - et_gap: Ratio of max to min mean ET ratio across clients
    """
    clients = list(client_cases.keys())
    n_clients = len(clients)

    # Get ET ratios per client
    client_et_ratios = {}
    for cid in clients:
        client_et_ratios[cid] = [case_info[c][0] for c in client_cases[cid]]

    # Compute ET ratio histograms for each client (for JSD)
    all_ratios = [case_info[c][0] for c in case_info.keys()]
    bin_edges = np.linspace(min(all_ratios), max(all_ratios) + 1e-6, num_bins + 1)

    client_hists = {}
    for cid in clients:
        hist, _ = np.histogram(client_et_ratios[cid], bins=bin_edges)
        client_hists[cid] = hist

    # Pairwise JSD
    jsd_values = []
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            jsd = compute_jensen_shannon_divergence(
                client_hists[clients[i]],
                client_hists[clients[j]]
            )
            jsd_values.append(jsd)

    # Pairwise KS statistic
    ks_values = []
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            ks_stat, _ = ks_2samp(
                client_et_ratios[clients[i]],
                client_et_ratios[clients[j]]
            )
            ks_values.append(ks_stat)

    # ET gap
    client_means = [np.mean(client_et_ratios[cid]) for cid in clients if client_et_ratios[cid]]
    et_gap = max(client_means) / (min(client_means) + 1e-10)

    return {
        "mean_jsd": float(np.mean(jsd_values)) if jsd_values else 0.0,
        "max_jsd": float(np.max(jsd_values)) if jsd_values else 0.0,
        "mean_ks": float(np.mean(ks_values)) if ks_values else 0.0,
        "max_ks": float(np.max(ks_values)) if ks_values else 0.0,
        "et_gap": float(et_gap)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Data Copying
# ═══════════════════════════════════════════════════════════════════════════════

def copy_case_data(source_dir: Path, dest_dir: Path) -> int:
    """
    Copy all .npz files from source case directory to destination.
    Returns number of files copied.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    npz_files = list(source_dir.glob("*.npz"))
    for npz_file in npz_files:
        # Resolve symlinks to actual file
        real_path = npz_file.resolve() if npz_file.is_symlink() else npz_file
        shutil.copy2(real_path, dest_dir / npz_file.name)
    return len(npz_files)


# ═══════════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_no_overlap(splits: Dict[str, Dict[str, List[str]]]) -> bool:
    """
    Verify that no case appears in multiple clients or splits.
    Returns True if no overlap found.
    """
    all_cases = []
    for _, client_splits in splits.items():
        for _, cases in client_splits.items():
            for case in cases:
                if case in all_cases:
                    print(f"ERROR: Case {case} appears multiple times!")
                    return False
                all_cases.append(case)
    return True


def verify_data_structure(output_dir: Path, num_clients: int) -> bool:
    """Verify the created data structure has .npz files."""
    output_data_dir = output_dir / "client_data"
    all_ok = True

    for cid in range(num_clients):
        for split in ['train', 'val', 'test']:
            split_dir = output_data_dir / f"client_{cid}" / split
            if not split_dir.exists():
                print(f"  ERROR: {split_dir} does not exist!")
                all_ok = False
                continue

            npz_files = list(split_dir.rglob("*.npz"))
            if not npz_files:
                print(f"  ERROR: No .npz files in {split_dir}")
                all_ok = False
            else:
                print(f"  OK: client_{cid}/{split} has {len(npz_files)} .npz files")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Create Dirichlet-based Non-IID Partition for Federated Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Moderate heterogeneity (good starting point for FedProx demonstration)
    python make_dirichlet_partition.py --alpha 0.5 --num_clients 4

    # Extreme heterogeneity (use with μ ≥ 0.4 for FedProx)
    python make_dirichlet_partition.py --alpha 0.1 --num_clients 4 --mu_recommendation 0.4

    # Mild heterogeneity
    python make_dirichlet_partition.py --alpha 1.0 --num_clients 4

Recommended μ values based on literature:
    α = 0.1 (extreme) → μ ∈ {0.4, 0.7, 1.0}
    α = 0.5 (moderate) → μ ∈ {0.1, 0.4}
    α = 1.0 (mild) → μ ∈ {0.01, 0.1}
        """
    )

    parser.add_argument(
        "--source_dir", type=Path,
        default=Path(__file__).parent / "brats2d_one_slice_per_patient_clients",
        help="Source directory with BraTS data"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help="Output directory (default: auto-generated based on parameters)"
    )
    parser.add_argument("--num_clients", type=int, default=4, help="Number of clients")
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Dirichlet concentration parameter (lower = more heterogeneous)"
    )
    parser.add_argument("--num_bins", type=int, default=NUM_ET_BINS, help="Number of ET ratio bins")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--val_frac", type=float, default=VAL_FRAC, help="Validation fraction")
    parser.add_argument("--test_frac", type=float, default=TEST_FRAC, help="Test fraction")

    args = parser.parse_args()

    # Auto-generate output directory name if not specified
    if args.output_dir is None:
        alpha_str = f"{args.alpha:.1f}".replace(".", "p")
        args.output_dir = Path(__file__).parent / f"brats2d_{args.num_clients}client_dirichlet_a{alpha_str}"

    print("=" * 70)
    print("DIRICHLET-BASED NON-IID PARTITION FOR FEDERATED LEARNING")
    print("=" * 70)
    print(f"Source:      {args.source_dir}")
    print(f"Output:      {args.output_dir}")
    print(f"Clients:     {args.num_clients}")
    print(f"Dirichlet α: {args.alpha}")
    print(f"ET bins:     {args.num_bins}")
    print(f"Seed:        {args.seed}")

    # Determine recommended μ based on α
    if args.alpha <= 0.1:
        mu_rec = "0.4, 0.7, or 1.0"
        het_level = "EXTREME"
    elif args.alpha <= 0.5:
        mu_rec = "0.1 or 0.4"
        het_level = "MODERATE"
    else:
        mu_rec = "0.01 or 0.1"
        het_level = "MILD"

    print(f"\nHeterogeneity level: {het_level}")
    print(f"Recommended FedProx μ: {mu_rec}")

    # ─── Step 1: Scan all cases ───────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Step 1: Scanning source data and computing ET ratios...")
    print("-" * 70)

    if not args.source_dir.exists():
        print(f"ERROR: Source directory does not exist: {args.source_dir}")
        return

    case_info = scan_all_cases(args.source_dir)
    print(f"\nFound {len(case_info)} unique cases")

    # Print ET distribution stats
    ratios = [info[0] for info in case_info.values()]
    print(f"\nET ratio statistics:")
    print(f"  min={min(ratios):.4f}  Q1={np.percentile(ratios,25):.4f}  "
          f"median={np.median(ratios):.4f}  Q3={np.percentile(ratios,75):.4f}  "
          f"max={max(ratios):.4f}")

    # ─── Step 2: Dirichlet partitioning ───────────────────────────────────────
    print("\n" + "-" * 70)
    print("Step 2: Applying Dirichlet partitioning...")
    print("-" * 70)

    client_cases = dirichlet_partition(
        case_info, args.num_clients, args.alpha, args.num_bins, args.seed
    )

    # ─── Step 3: Train/val/test splits ────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Step 3: Creating train/val/test splits per client...")
    print("-" * 70)

    splits = {}
    for client_id, cases in sorted(client_cases.items()):
        # Use different seed per client to avoid correlation
        client_seed = args.seed + int(client_id.split("_")[1])
        splits[client_id] = split_train_val_test(
            cases, client_seed, args.val_frac, args.test_frac
        )
        s = splits[client_id]
        print(f"  {client_id}: train={len(s['train'])} val={len(s['val'])} test={len(s['test'])} (total={len(cases)})")

    # Verify no overlap
    print("\n  Verifying no case overlap...")
    if verify_no_overlap(splits):
        print("  ✓ No overlap detected")
    else:
        print("  ✗ OVERLAP DETECTED - aborting!")
        return

    # ─── Step 4: Compute heterogeneity metrics ────────────────────────────────
    print("\n" + "-" * 70)
    print("Step 4: Computing heterogeneity metrics (for thesis reporting)...")
    print("-" * 70)

    het_metrics = compute_heterogeneity_metrics(client_cases, case_info)

    print(f"\n  Jensen-Shannon Divergence (JSD):")
    print(f"    Mean pairwise JSD: {het_metrics['mean_jsd']:.4f}")
    print(f"    Max pairwise JSD:  {het_metrics['max_jsd']:.4f}")
    print(f"  Kolmogorov-Smirnov Statistic:")
    print(f"    Mean pairwise KS:  {het_metrics['mean_ks']:.4f}")
    print(f"    Max pairwise KS:   {het_metrics['max_ks']:.4f}")
    print(f"  ET Ratio Gap (max/min mean): {het_metrics['et_gap']:.2f}x")

    # Literature thresholds for interpretation
    print(f"\n  Interpretation (based on literature):")
    if het_metrics['mean_jsd'] > 0.4:
        print(f"    JSD > 0.4: SUBSTANTIAL heterogeneity (use μ ≥ 0.4)")
    elif het_metrics['mean_jsd'] > 0.2:
        print(f"    JSD ∈ [0.2, 0.4]: MODERATE heterogeneity (use μ ∈ [0.1, 0.4])")
    else:
        print(f"    JSD < 0.2: MILD heterogeneity (use μ ∈ [0.01, 0.1])")

    # ─── Step 5: Copy data files ──────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Step 5: Copying data files (for HPC compatibility)...")
    print("-" * 70)

    # Clean existing output
    if args.output_dir.exists():
        print(f"  Removing existing output: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_data_dir = args.output_dir / "client_data"

    total_files = 0
    for client_id in sorted(splits.keys()):
        for split_name, case_list in splits[client_id].items():
            dest_split_dir = output_data_dir / client_id / split_name
            split_files = 0
            for case_id in case_list:
                _, source_path = case_info[case_id]
                dest_case_dir = dest_split_dir / case_id
                n_files = copy_case_data(source_path, dest_case_dir)
                split_files += n_files
                total_files += n_files
            print(f"  {client_id}/{split_name}: {len(case_list)} cases, {split_files} files")

    print(f"\n  Total files copied: {total_files}")

    # ─── Step 6: Verify data structure ────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Step 6: Verifying data structure...")
    print("-" * 70)

    if not verify_data_structure(args.output_dir, args.num_clients):
        print("\nERROR: Verification failed!")
        return

    # ─── Step 7: Save metadata ────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("Step 7: Saving metadata...")
    print("-" * 70)

    # Compute per-client statistics
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
            "partition_type": "dirichlet",
            "description": f"Dirichlet-based non-IID partition with α={args.alpha}",
            "dirichlet_alpha": args.alpha,
            "num_clients": args.num_clients,
            "num_bins": args.num_bins,
            "seed": args.seed,
            "total_cases": len(case_info),
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "heterogeneity_metrics": het_metrics,
            "recommended_mu": mu_rec,
            **{f"{cid}_stats": stats for cid, stats in client_stats.items()}
        }
    }

    with open(args.output_dir / "client_map.json", "w") as f:
        json.dump(client_map, f, indent=2)

    print(f"  Saved: {args.output_dir / 'client_map.json'}")

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PARTITION SUMMARY")
    print("=" * 70)

    print(f"\nPartition: Dirichlet (α={args.alpha})")
    print(f"Heterogeneity level: {het_level}")
    print(f"Output: {args.output_dir}")

    print(f"\nPer-Client Statistics:")
    for client_id, stats in sorted(client_stats.items()):
        print(f"\n  {client_id}:")
        print(f"    Cases: {stats['n_cases']} (train={stats['n_train']}, val={stats['n_val']}, test={stats['n_test']})")
        print(f"    ET ratio: mean={stats['et_ratio_mean']:.4f} ± {stats['et_ratio_std']:.4f}")
        print(f"    ET range: [{stats['et_ratio_min']:.4f}, {stats['et_ratio_max']:.4f}]")

    print(f"\nHeterogeneity Metrics (for thesis):")
    print(f"  Mean JSD: {het_metrics['mean_jsd']:.4f}")
    print(f"  Max JSD:  {het_metrics['max_jsd']:.4f}")
    print(f"  Mean KS:  {het_metrics['mean_ks']:.4f}")
    print(f"  ET Gap:   {het_metrics['et_gap']:.2f}x")

    print("\n" + "=" * 70)
    print("TRAINING COMMANDS")
    print("=" * 70)

    partition_path = args.output_dir / "client_data"

    print(f"""
# FedAvg baseline:
STRATEGY=fedavg NUM_CLIENTS={args.num_clients} \\
PARTITION_DIR="{partition_path}" \\
sbatch federated_unet/unet/run_flower_70_30.sbatch

# FedProx with recommended μ:
STRATEGY=fedprox MU={mu_rec.split(',')[0].strip()} NUM_CLIENTS={args.num_clients} \\
PARTITION_DIR="{partition_path}" \\
sbatch federated_unet/unet/run_flower_70_30.sbatch
""")

    print("=" * 70)
    print("DONE! Partition created successfully.")   
    print("=" * 70)


if __name__ == "__main__":
    main()

