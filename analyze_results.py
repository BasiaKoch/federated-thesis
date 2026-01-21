"""
Analysis Script for NVFlare MNIST Federated Learning Results

This script analyzes and visualizes the results from federated learning experiments,
comparing FedAvg vs FedProx performance.

Usage:
    python analyze_results.py --workspace ./workspace/mnist_fedprox
    python analyze_results.py --compare ./workspace/mnist_fedavg ./workspace/mnist_fedprox
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_cross_val_results(workspace: str) -> Optional[Dict]:
    """Load cross-site validation results from workspace."""
    cross_val_path = Path(workspace) / "server" / "simulate_job" / "cross_site_val" / "cross_val_results.json"

    if not cross_val_path.exists():
        print(f"Warning: Cross-validation results not found at {cross_val_path}")
        return None

    with open(cross_val_path, "r") as f:
        return json.load(f)


def load_server_logs(workspace: str) -> List[Dict]:
    """Parse server logs to extract training metrics per round."""
    log_path = Path(workspace) / "server" / "log.txt"

    if not log_path.exists():
        print(f"Warning: Log file not found at {log_path}")
        return []

    metrics = []
    # This is a simplified parser - actual NVFlare logs may need different parsing
    with open(log_path, "r") as f:
        for line in f:
            if "Round" in line and ("Loss" in line or "Acc" in line):
                # Extract metrics from log line
                # Format varies by logging setup
                pass

    return metrics


def plot_training_curves(
    results: Dict[str, List[float]],
    title: str = "Training Curves",
    output_path: Optional[str] = None
):
    """Plot training loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    if "loss" in results:
        axes[0].plot(results["loss"], marker="o", markersize=3)
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    if "accuracy" in results:
        axes[1].plot(results["accuracy"], marker="o", markersize=3, color="green")
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training Accuracy")
        axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    plt.show()


def plot_comparison(
    fedavg_results: Dict,
    fedprox_results: Dict,
    output_path: Optional[str] = None
):
    """Plot comparison between FedAvg and FedProx."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss comparison
    if "loss" in fedavg_results and "loss" in fedprox_results:
        axes[0].plot(fedavg_results["loss"], label="FedAvg", marker="o", markersize=3)
        axes[0].plot(fedprox_results["loss"], label="FedProx", marker="s", markersize=3)
        axes[0].set_xlabel("Round")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Accuracy comparison
    if "accuracy" in fedavg_results and "accuracy" in fedprox_results:
        axes[1].plot(fedavg_results["accuracy"], label="FedAvg", marker="o", markersize=3)
        axes[1].plot(fedprox_results["accuracy"], label="FedProx", marker="s", markersize=3)
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training Accuracy Comparison")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle("FedAvg vs FedProx Comparison")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot to {output_path}")

    plt.show()


def plot_client_performance(
    cross_val_results: Dict,
    title: str = "Per-Client Validation Accuracy",
    output_path: Optional[str] = None
):
    """Plot per-client validation performance."""
    if not cross_val_results:
        print("No cross-validation results to plot")
        return

    # Extract client accuracies
    client_accs = {}
    for client_name, results in cross_val_results.items():
        if isinstance(results, dict):
            # Results might be nested differently based on NVFlare version
            for model_name, metrics in results.items():
                if isinstance(metrics, dict) and "val_accuracy" in metrics:
                    client_accs[client_name] = metrics["val_accuracy"]

    if not client_accs:
        print("Could not extract client accuracies from results")
        return

    # Sort by client number
    sorted_clients = sorted(
        client_accs.items(),
        key=lambda x: int(x[0].split("-")[-1]) if "-" in x[0] else 0
    )
    names, accs = zip(*sorted_clients)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(names)), accs, color="steelblue", edgecolor="darkblue")

    # Color bars by performance
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        if acc >= 0.9:
            bar.set_color("green")
        elif acc >= 0.7:
            bar.set_color("steelblue")
        else:
            bar.set_color("orange")

    ax.set_xlabel("Client")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(title)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.axhline(y=np.mean(accs), color="red", linestyle="--", label=f"Mean: {np.mean(accs):.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved client performance plot to {output_path}")

    plt.show()


def analyze_partition_statistics(partition_file: str):
    """Analyze and visualize data partition statistics."""
    with open(partition_file, "r") as f:
        partition = json.load(f)

    sample_counts = partition["sample_counts"]
    client_digits = partition["client_digits"]

    print("\n" + "=" * 60)
    print("DATA PARTITION ANALYSIS")
    print("=" * 60)
    print(f"Number of clients: {partition['num_clients']}")
    print(f"Power-law alpha: {partition['alpha']}")
    print()
    print("Sample Distribution:")
    print(f"  Min: {min(sample_counts)}")
    print(f"  Max: {max(sample_counts)}")
    print(f"  Mean: {np.mean(sample_counts):.1f}")
    print(f"  Std: {np.std(sample_counts):.1f}")
    print(f"  Median: {np.median(sample_counts):.1f}")
    print()

    # Plot sample distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of sample counts
    axes[0].hist(sample_counts, bins=20, edgecolor="black", color="steelblue")
    axes[0].set_xlabel("Number of Samples")
    axes[0].set_ylabel("Number of Clients")
    axes[0].set_title("Distribution of Samples per Client")
    axes[0].axvline(np.mean(sample_counts), color="red", linestyle="--", label=f"Mean: {np.mean(sample_counts):.0f}")
    axes[0].legend()

    # Bar plot showing samples per client
    sorted_indices = np.argsort(sample_counts)[::-1]
    sorted_counts = [sample_counts[i] for i in sorted_indices]

    colors = []
    for i, idx in enumerate(sorted_indices):
        digit_pair = client_digits[str(idx)]
        # Color by digit pair
        hue = (digit_pair[0] / 10)
        colors.append(plt.cm.tab10(hue))

    axes[1].bar(range(len(sorted_counts)), sorted_counts, color=colors, edgecolor="gray")
    axes[1].set_xlabel("Client (sorted by sample count)")
    axes[1].set_ylabel("Number of Samples")
    axes[1].set_title("Samples per Client (Power-Law Distribution)")

    plt.suptitle("Non-IID Data Partition Statistics")
    plt.tight_layout()
    plt.show()


def print_summary(workspace: str):
    """Print summary of experiment results."""
    print("\n" + "=" * 60)
    print(f"EXPERIMENT SUMMARY: {Path(workspace).name}")
    print("=" * 60)

    # Check for model files
    model_path = Path(workspace) / "server" / "simulate_job" / "app_server"
    if model_path.exists():
        models = list(model_path.glob("*.pt"))
        print(f"Saved models: {len(models)}")

    # Load and display cross-val results
    cross_val = load_cross_val_results(workspace)
    if cross_val:
        print("\nCross-Site Validation Results Available")
        # Count clients
        num_clients = len([k for k in cross_val.keys() if k.startswith("site")])
        print(f"Number of clients evaluated: {num_clients}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NVFlare federated learning results"
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
        "--partition", type=str,
        help="Path to partition file for analysis"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Directory to save output plots"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.partition:
        analyze_partition_statistics(args.partition)

    if args.workspace:
        print_summary(args.workspace)
        cross_val = load_cross_val_results(args.workspace)
        if cross_val:
            plot_client_performance(
                cross_val,
                title=f"Per-Client Validation - {Path(args.workspace).name}",
                output_path=str(output_dir / "client_performance.png")
            )

    if args.compare:
        fedavg_ws, fedprox_ws = args.compare
        print(f"\nComparing: {fedavg_ws} vs {fedprox_ws}")

        # Load results from both
        fedavg_cross = load_cross_val_results(fedavg_ws)
        fedprox_cross = load_cross_val_results(fedprox_ws)

        if fedavg_cross and fedprox_cross:
            print("\nCross-validation results found for both experiments")
            # Additional comparison plots could be added here


if __name__ == "__main__":
    main()
