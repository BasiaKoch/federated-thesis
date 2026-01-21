"""
Comprehensive Analysis Script for MNIST Federated Learning Results

This script provides the analysis required for the dissertation:
1. Convergence plots (accuracy/loss vs rounds) for FedAvg vs FedProx
2. Per-client performance analysis
3. Comparison with local-only baseline
4. Statistical summaries

Usage:
    python analyze_results.py --workspace ./workspace/mnist_fedprox
    python analyze_results.py --compare ./workspace/mnist_fedavg ./workspace/mnist_fedprox
    python analyze_results.py --full-analysis ./workspace/mnist_fedavg ./workspace/mnist_fedprox --baseline ./results/local_baseline
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_metrics_from_logs(workspace: str) -> Dict:
    """
    Parse structured METRICS lines from client logs.

    Returns:
        Dict with per-round, per-client metrics
    """
    workspace_path = Path(workspace)
    metrics = {
        'per_round': defaultdict(lambda: defaultdict(list)),
        'validation': defaultdict(dict),
        'config': {}
    }

    # Find all client log files
    for site_dir in sorted(workspace_path.glob("site-*")):
        log_file = site_dir / "log.txt"
        if not log_file.exists():
            continue

        with open(log_file, 'r') as f:
            for line in f:
                # Parse METRICS lines (per-round training metrics)
                if "METRICS|" in line:
                    match = re.search(
                        r"METRICS\|round=(\d+)\|client=(\d+)\|"
                        r"train_loss=([\d.]+)\|train_acc=([\d.]+)\|"
                        r"local_test_loss=([\d.]+)\|local_test_acc=([\d.]+)\|"
                        r"global_test_loss=([\d.]+)\|global_test_acc=([\d.]+)\|"
                        r"samples=(\d+)\|digits=\[(\d+),\s*(\d+)\]",
                        line
                    )
                    if match:
                        round_num = int(match.group(1))
                        client_id = int(match.group(2))
                        metrics['per_round'][round_num][client_id] = {
                            'train_loss': float(match.group(3)),
                            'train_acc': float(match.group(4)),
                            'local_test_loss': float(match.group(5)),
                            'local_test_acc': float(match.group(6)),
                            'global_test_loss': float(match.group(7)),
                            'global_test_acc': float(match.group(8)),
                            'samples': int(match.group(9)),
                            'digits': [int(match.group(10)), int(match.group(11))]
                        }

                # Parse VALIDATION lines (cross-site validation)
                if "VALIDATION|" in line:
                    match = re.search(
                        r"VALIDATION\|client=(\d+)\|"
                        r"local_loss=([\d.]+)\|local_acc=([\d.]+)\|"
                        r"global_loss=([\d.]+)\|global_acc=([\d.]+)\|"
                        r"samples=(\d+)\|digits=\[(\d+),\s*(\d+)\]",
                        line
                    )
                    if match:
                        client_id = int(match.group(1))
                        metrics['validation'][client_id] = {
                            'local_loss': float(match.group(2)),
                            'local_acc': float(match.group(3)),
                            'global_loss': float(match.group(4)),
                            'global_acc': float(match.group(5)),
                            'samples': int(match.group(6)),
                            'digits': [int(match.group(7)), int(match.group(8))]
                        }

    return metrics


def compute_round_statistics(metrics: Dict) -> Dict:
    """
    Compute per-round aggregate statistics from parsed metrics.

    Returns:
        Dict with aggregated statistics per round
    """
    round_stats = {}

    for round_num in sorted(metrics['per_round'].keys()):
        round_data = metrics['per_round'][round_num]
        if not round_data:
            continue

        # Aggregate metrics across clients
        train_losses = [c['train_loss'] for c in round_data.values()]
        train_accs = [c['train_acc'] for c in round_data.values()]
        global_losses = [c['global_test_loss'] for c in round_data.values()]
        global_accs = [c['global_test_acc'] for c in round_data.values()]
        samples = [c['samples'] for c in round_data.values()]

        # Weighted average (by sample count)
        total_samples = sum(samples)
        weighted_global_acc = sum(
            c['global_test_acc'] * c['samples'] for c in round_data.values()
        ) / total_samples if total_samples > 0 else 0

        round_stats[round_num] = {
            'num_clients': len(round_data),
            'train_loss_mean': np.mean(train_losses),
            'train_loss_std': np.std(train_losses),
            'train_acc_mean': np.mean(train_accs),
            'train_acc_std': np.std(train_accs),
            'global_test_loss_mean': np.mean(global_losses),
            'global_test_loss_std': np.std(global_losses),
            'global_test_acc_mean': np.mean(global_accs),
            'global_test_acc_std': np.std(global_accs),
            'global_test_acc_weighted': weighted_global_acc,
            'global_test_acc_min': np.min(global_accs),
            'global_test_acc_max': np.max(global_accs),
        }

    return round_stats


def load_cross_val_results(workspace: str) -> Optional[Dict]:
    """Load cross-site validation results from workspace."""
    cross_val_path = Path(workspace) / "server" / "simulate_job" / "cross_site_val" / "cross_val_results.json"

    if not cross_val_path.exists():
        print(f"Warning: Cross-validation results not found at {cross_val_path}")
        return None

    with open(cross_val_path, "r") as f:
        data = json.load(f)
        return data if data else None


def load_local_baseline(baseline_dir: str) -> Optional[Dict]:
    """Load local-only baseline results."""
    baseline_path = Path(baseline_dir) / "local_baseline_results.json"

    if not baseline_path.exists():
        print(f"Warning: Local baseline results not found at {baseline_path}")
        return None

    with open(baseline_path, "r") as f:
        return json.load(f)


def plot_convergence_curves(
    round_stats: Dict,
    title: str = "Training Convergence",
    output_path: Optional[str] = None
):
    """Plot convergence curves (loss and accuracy vs rounds)."""
    if not round_stats:
        print("No round statistics to plot")
        return

    rounds = sorted(round_stats.keys())
    global_acc = [round_stats[r]['global_test_acc_mean'] for r in rounds]
    global_acc_std = [round_stats[r]['global_test_acc_std'] for r in rounds]
    global_loss = [round_stats[r]['global_test_loss_mean'] for r in rounds]
    global_loss_std = [round_stats[r]['global_test_loss_std'] for r in rounds]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curve with std band
    axes[0].plot(rounds, global_acc, marker='o', markersize=3, label='Mean')
    axes[0].fill_between(
        rounds,
        np.array(global_acc) - np.array(global_acc_std),
        np.array(global_acc) + np.array(global_acc_std),
        alpha=0.3
    )
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Global Test Accuracy")
    axes[0].set_title("Global Test Accuracy vs Rounds")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Loss curve with std band
    axes[1].plot(rounds, global_loss, marker='o', markersize=3, color='orange', label='Mean')
    axes[1].fill_between(
        rounds,
        np.array(global_loss) - np.array(global_loss_std),
        np.array(global_loss) + np.array(global_loss_std),
        alpha=0.3, color='orange'
    )
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Global Test Loss")
    axes[1].set_title("Global Test Loss vs Rounds")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    plt.show()


def plot_fedavg_vs_fedprox(
    fedavg_stats: Dict,
    fedprox_stats: Dict,
    output_path: Optional[str] = None
):
    """Plot FedAvg vs FedProx comparison (required by dissertation)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get common rounds
    fedavg_rounds = sorted(fedavg_stats.keys())
    fedprox_rounds = sorted(fedprox_stats.keys())

    # Accuracy comparison
    fedavg_acc = [fedavg_stats[r]['global_test_acc_mean'] for r in fedavg_rounds]
    fedprox_acc = [fedprox_stats[r]['global_test_acc_mean'] for r in fedprox_rounds]

    axes[0].plot(fedavg_rounds, fedavg_acc, marker='o', markersize=3, label='FedAvg', color='blue')
    axes[0].plot(fedprox_rounds, fedprox_acc, marker='s', markersize=3, label='FedProx', color='green')
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Global Test Accuracy")
    axes[0].set_title("Accuracy: FedAvg vs FedProx")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss comparison
    fedavg_loss = [fedavg_stats[r]['global_test_loss_mean'] for r in fedavg_rounds]
    fedprox_loss = [fedprox_stats[r]['global_test_loss_mean'] for r in fedprox_rounds]

    axes[1].plot(fedavg_rounds, fedavg_loss, marker='o', markersize=3, label='FedAvg', color='blue')
    axes[1].plot(fedprox_rounds, fedprox_loss, marker='s', markersize=3, label='FedProx', color='green')
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Global Test Loss")
    axes[1].set_title("Loss: FedAvg vs FedProx")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("FedAvg vs FedProx Convergence Comparison")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to {output_path}")

    plt.show()


def plot_per_client_performance(
    validation_metrics: Dict,
    title: str = "Per-Client Performance",
    output_path: Optional[str] = None
):
    """Plot per-client validation performance (global model on each client's test set)."""
    if not validation_metrics:
        print("No validation metrics to plot")
        return

    # Sort by client ID
    clients = sorted(validation_metrics.keys())
    local_accs = [validation_metrics[c]['local_acc'] for c in clients]
    global_accs = [validation_metrics[c]['global_acc'] for c in clients]
    samples = [validation_metrics[c]['samples'] for c in clients]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Bar plot: local vs global accuracy per client
    x = np.arange(len(clients))
    width = 0.35

    axes[0].bar(x - width/2, local_accs, width, label='Local Test (2 digits)', color='steelblue')
    axes[0].bar(x + width/2, global_accs, width, label='Global Test (10 digits)', color='coral')
    axes[0].set_xlabel("Client ID")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Per-Client Accuracy (Global Model)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{c}" for c in clients], fontsize=8)
    axes[0].legend()
    axes[0].axhline(y=np.mean(global_accs), color='red', linestyle='--',
                    label=f'Global Mean: {np.mean(global_accs):.3f}')
    axes[0].grid(True, alpha=0.3, axis='y')

    # Scatter plot: samples vs accuracy (shows correlation with data quantity)
    scatter = axes[1].scatter(samples, global_accs, c=local_accs, cmap='viridis',
                               s=80, edgecolors='black', alpha=0.7)
    axes[1].set_xlabel("Training Samples")
    axes[1].set_ylabel("Global Test Accuracy")
    axes[1].set_title("Accuracy vs Training Data Size")
    plt.colorbar(scatter, ax=axes[1], label='Local Test Accuracy')
    axes[1].grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(samples, global_accs, 1)
    p = np.poly1d(z)
    axes[1].plot(sorted(samples), p(sorted(samples)), "r--", alpha=0.8, label='Trend')
    axes[1].legend()

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved client performance plot to {output_path}")

    plt.show()


def plot_three_way_comparison(
    local_baseline: Dict,
    fedavg_validation: Dict,
    fedprox_validation: Dict,
    output_path: Optional[str] = None
):
    """
    Plot three-way comparison: Local-only vs FedAvg vs FedProx.
    This is the key comparison required by the dissertation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Get client IDs (assuming they match)
    local_results = local_baseline.get('per_client_results', [])
    clients = list(range(len(local_results)))

    # Extract accuracies
    local_global_acc = [r['final_global_test_acc'] for r in local_results]
    local_local_acc = [r['final_local_test_acc'] for r in local_results]

    fedavg_global_acc = [fedavg_validation.get(c, {}).get('global_acc', 0) for c in clients]
    fedprox_global_acc = [fedprox_validation.get(c, {}).get('global_acc', 0) for c in clients]

    # Bar plot comparison
    x = np.arange(len(clients))
    width = 0.25

    axes[0].bar(x - width, local_global_acc, width, label='Local-Only', color='gray', alpha=0.7)
    axes[0].bar(x, fedavg_global_acc, width, label='FedAvg', color='blue', alpha=0.7)
    axes[0].bar(x + width, fedprox_global_acc, width, label='FedProx', color='green', alpha=0.7)
    axes[0].set_xlabel("Client ID")
    axes[0].set_ylabel("Global Test Accuracy")
    axes[0].set_title("Global Test Accuracy: Local vs Federated")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Summary statistics
    summary_data = {
        'Local-Only': local_global_acc,
        'FedAvg': fedavg_global_acc,
        'FedProx': fedprox_global_acc
    }

    means = [np.mean(v) for v in summary_data.values()]
    stds = [np.std(v) for v in summary_data.values()]
    mins = [np.min(v) for v in summary_data.values()]

    x_summary = np.arange(3)
    bars = axes[1].bar(x_summary, means, yerr=stds, capsize=5,
                       color=['gray', 'blue', 'green'], alpha=0.7)
    axes[1].scatter(x_summary, mins, marker='v', color='red', s=100, zorder=5, label='Worst Client')
    axes[1].set_xlabel("Method")
    axes[1].set_ylabel("Global Test Accuracy")
    axes[1].set_title("Summary: Mean Accuracy (error bars = std)")
    axes[1].set_xticks(x_summary)
    axes[1].set_xticklabels(summary_data.keys())
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                     f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle("Three-Way Comparison: Local-Only vs FedAvg vs FedProx")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved three-way comparison to {output_path}")

    plt.show()


def compute_convergence_metrics(round_stats: Dict, threshold: float = 0.8) -> Dict:
    """
    Compute convergence metrics.

    Returns:
        - Rounds to reach threshold accuracy
        - Best accuracy achieved
        - Final accuracy
    """
    rounds = sorted(round_stats.keys())
    accs = [round_stats[r]['global_test_acc_mean'] for r in rounds]

    # Rounds to threshold
    rounds_to_threshold = None
    for r, acc in zip(rounds, accs):
        if acc >= threshold:
            rounds_to_threshold = r
            break

    return {
        'rounds_to_threshold': rounds_to_threshold,
        'threshold': threshold,
        'best_accuracy': max(accs) if accs else 0,
        'best_round': rounds[np.argmax(accs)] if accs else None,
        'final_accuracy': accs[-1] if accs else 0,
        'final_round': rounds[-1] if rounds else None,
        'total_rounds': len(rounds)
    }


def print_summary(workspace: str, round_stats: Dict, validation_metrics: Dict):
    """Print comprehensive summary of experiment results."""
    print("\n" + "=" * 70)
    print(f"EXPERIMENT SUMMARY: {Path(workspace).name}")
    print("=" * 70)

    if round_stats:
        final_round = max(round_stats.keys())
        final_stats = round_stats[final_round]

        print(f"\nTraining completed: {final_round + 1} rounds")
        print(f"\nFinal Global Test Metrics (Round {final_round}):")
        print(f"  Accuracy: {final_stats['global_test_acc_mean']:.4f} (+/- {final_stats['global_test_acc_std']:.4f})")
        print(f"  Loss:     {final_stats['global_test_loss_mean']:.4f} (+/- {final_stats['global_test_loss_std']:.4f})")
        print(f"  Min/Max:  {final_stats['global_test_acc_min']:.4f} / {final_stats['global_test_acc_max']:.4f}")

        # Convergence metrics
        conv = compute_convergence_metrics(round_stats)
        print(f"\nConvergence Metrics:")
        print(f"  Best accuracy: {conv['best_accuracy']:.4f} (round {conv['best_round']})")
        if conv['rounds_to_threshold']:
            print(f"  Rounds to {conv['threshold']*100:.0f}%: {conv['rounds_to_threshold']}")
        else:
            print(f"  Did not reach {conv['threshold']*100:.0f}% threshold")

    if validation_metrics:
        global_accs = [v['global_acc'] for v in validation_metrics.values()]
        local_accs = [v['local_acc'] for v in validation_metrics.values()]

        print(f"\nCross-Site Validation ({len(validation_metrics)} clients):")
        print(f"  Global Test - Mean: {np.mean(global_accs):.4f}, Std: {np.std(global_accs):.4f}")
        print(f"  Local Test  - Mean: {np.mean(local_accs):.4f}, Std: {np.std(local_accs):.4f}")

        # Worst performing clients
        worst_idx = np.argmin(global_accs)
        worst_client = list(validation_metrics.keys())[worst_idx]
        worst_data = validation_metrics[worst_client]
        print(f"\n  Worst client: {worst_client} (digits {worst_data['digits']}, "
              f"{worst_data['samples']} samples) -> {worst_data['global_acc']:.4f}")


def analyze_partition(partition_file: str):
    """Analyze and document data partition (required for dissertation)."""
    with open(partition_file, 'r') as f:
        partition = json.load(f)

    print("\n" + "=" * 70)
    print("DATA PARTITION DOCUMENTATION")
    print("=" * 70)

    print(f"\nHeterogeneity Parameters:")
    print(f"  Number of clients: {partition['num_clients']}")
    print(f"  Power-law alpha: {partition['alpha']}")
    print(f"  Random seed: {partition['seed']}")
    print(f"  Digits per client: 2 (label skew)")

    sample_counts = partition['sample_counts']
    print(f"\nQuantity Skew (sample distribution):")
    print(f"  Min samples: {min(sample_counts)}")
    print(f"  Max samples: {max(sample_counts)}")
    print(f"  Mean samples: {np.mean(sample_counts):.1f}")
    print(f"  Std samples: {np.std(sample_counts):.1f}")
    print(f"  Total samples: {sum(sample_counts)}")

    print(f"\nPer-Client Details:")
    print("-" * 50)
    print(f"{'Client':<8} {'Digits':<12} {'Train Samples':<15} {'Test Samples':<12}")
    print("-" * 50)

    for i in range(partition['num_clients']):
        digits = partition['client_digits'][str(i)]
        train_samples = len(partition['train_partition'][str(i)])
        test_samples = len(partition['test_partition'][str(i)])
        print(f"{i:<8} {str(digits):<12} {train_samples:<15} {test_samples:<12}")

    # Label distribution
    print(f"\n\nLabel Histogram per Client:")
    print("-" * 50)
    for i in range(min(5, partition['num_clients'])):  # Show first 5 clients
        digits = partition['client_digits'][str(i)]
        print(f"  Client {i}: Only digits {digits}")
    print(f"  ... (showing first 5 of {partition['num_clients']} clients)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NVFlare federated learning results for dissertation"
    )
    parser.add_argument(
        "--workspace", type=str,
        help="Path to single experiment workspace"
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("FEDAVG", "FEDPROX"),
        help="Compare two workspaces (FedAvg vs FedProx)"
    )
    parser.add_argument(
        "--baseline", type=str,
        help="Path to local baseline results directory"
    )
    parser.add_argument(
        "--partition", type=str,
        help="Path to partition file for documentation"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--full-analysis", nargs=2, metavar=("FEDAVG", "FEDPROX"),
        help="Run full three-way analysis with baseline"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze partition if provided
    if args.partition:
        analyze_partition(args.partition)

    # Single workspace analysis
    if args.workspace:
        print(f"\nAnalyzing workspace: {args.workspace}")
        metrics = parse_metrics_from_logs(args.workspace)
        round_stats = compute_round_statistics(metrics)

        print_summary(args.workspace, round_stats, metrics['validation'])

        if round_stats:
            plot_convergence_curves(
                round_stats,
                title=f"Convergence: {Path(args.workspace).name}",
                output_path=str(output_dir / f"{Path(args.workspace).name}_convergence.png")
            )

        if metrics['validation']:
            plot_per_client_performance(
                metrics['validation'],
                title=f"Per-Client Performance: {Path(args.workspace).name}",
                output_path=str(output_dir / f"{Path(args.workspace).name}_clients.png")
            )

    # Compare FedAvg vs FedProx
    if args.compare:
        fedavg_ws, fedprox_ws = args.compare
        print(f"\nComparing: {fedavg_ws} vs {fedprox_ws}")

        fedavg_metrics = parse_metrics_from_logs(fedavg_ws)
        fedprox_metrics = parse_metrics_from_logs(fedprox_ws)

        fedavg_stats = compute_round_statistics(fedavg_metrics)
        fedprox_stats = compute_round_statistics(fedprox_metrics)

        if fedavg_stats and fedprox_stats:
            plot_fedavg_vs_fedprox(
                fedavg_stats, fedprox_stats,
                output_path=str(output_dir / "fedavg_vs_fedprox.png")
            )

            # Print comparison summary
            print("\n" + "=" * 70)
            print("FEDAVG vs FEDPROX COMPARISON")
            print("=" * 70)

            fedavg_conv = compute_convergence_metrics(fedavg_stats)
            fedprox_conv = compute_convergence_metrics(fedprox_stats)

            print(f"\n{'Metric':<30} {'FedAvg':<15} {'FedProx':<15}")
            print("-" * 60)
            print(f"{'Final Accuracy':<30} {fedavg_conv['final_accuracy']:.4f}{'':<10} {fedprox_conv['final_accuracy']:.4f}")
            print(f"{'Best Accuracy':<30} {fedavg_conv['best_accuracy']:.4f}{'':<10} {fedprox_conv['best_accuracy']:.4f}")
            print(f"{'Best Round':<30} {fedavg_conv['best_round']:<15} {fedprox_conv['best_round']:<15}")

    # Full three-way analysis
    if args.full_analysis and args.baseline:
        fedavg_ws, fedprox_ws = args.full_analysis
        print(f"\nFull analysis: Local vs FedAvg vs FedProx")

        local_baseline = load_local_baseline(args.baseline)
        fedavg_metrics = parse_metrics_from_logs(fedavg_ws)
        fedprox_metrics = parse_metrics_from_logs(fedprox_ws)

        if local_baseline and fedavg_metrics['validation'] and fedprox_metrics['validation']:
            plot_three_way_comparison(
                local_baseline,
                fedavg_metrics['validation'],
                fedprox_metrics['validation'],
                output_path=str(output_dir / "three_way_comparison.png")
            )


if __name__ == "__main__":
    main()
