#!/usr/bin/env python3
"""
Analyze and compare Flower MNIST 2-Digit experiment results.

This script loads results from FedAvg and FedProx experiments and generates:
1. Convergence plots (accuracy and loss over rounds)
2. Stability comparison
3. Summary statistics table

Usage:
    python analyze_flower_results.py --results_dir ./results/flower_mnist_2digits
    python analyze_flower_results.py --fedavg_file fedavg_results.json --fedprox_file fedprox_results.json
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_results(file_path: str) -> Dict:
    """Load a single results JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def find_latest_results(results_dir: str, strategy: str) -> Optional[str]:
    """Find the latest results file for a given strategy."""
    pattern = os.path.join(results_dir, f"{strategy}_*_results.json")
    files = glob.glob(pattern)
    if not files:
        return None
    # Sort by modification time, get latest
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def plot_convergence(
    fedavg_results: Optional[Dict],
    fedprox_results: Optional[Dict],
    output_dir: str,
    show: bool = False,
) -> None:
    """Plot accuracy and loss convergence curves for both strategies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Colors and styles
    colors = {"fedavg": "#1f77b4", "fedprox": "#ff7f0e"}

    # Plot accuracy
    ax1 = axes[0]
    if fedavg_results:
        rounds = fedavg_results["per_round_metrics"]["rounds"]
        acc = fedavg_results["per_round_metrics"]["global_test_accuracy"]
        ax1.plot(rounds, acc, label="FedAvg", color=colors["fedavg"], linewidth=2, marker="o", markersize=3)

    if fedprox_results:
        rounds = fedprox_results["per_round_metrics"]["rounds"]
        acc = fedprox_results["per_round_metrics"]["global_test_accuracy"]
        mu = fedprox_results["experiment"]["mu"]
        ax1.plot(rounds, acc, label=f"FedProx (mu={mu})", color=colors["fedprox"], linewidth=2, marker="s", markersize=3)

    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("Global Test Accuracy", fontsize=12)
    ax1.set_title("Convergence: Accuracy over Rounds", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Add convergence threshold lines
    for thresh in [0.90, 0.95]:
        ax1.axhline(y=thresh, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax1.text(0.5, thresh + 0.01, f"{int(thresh*100)}%", fontsize=9, color="gray")

    # Plot loss
    ax2 = axes[1]
    if fedavg_results:
        rounds = fedavg_results["per_round_metrics"]["rounds"]
        loss = fedavg_results["per_round_metrics"]["global_test_loss"]
        ax2.plot(rounds, loss, label="FedAvg", color=colors["fedavg"], linewidth=2, marker="o", markersize=3)

    if fedprox_results:
        rounds = fedprox_results["per_round_metrics"]["rounds"]
        loss = fedprox_results["per_round_metrics"]["global_test_loss"]
        mu = fedprox_results["experiment"]["mu"]
        ax2.plot(rounds, loss, label=f"FedProx (mu={mu})", color=colors["fedprox"], linewidth=2, marker="s", markersize=3)

    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("Global Test Loss", fontsize=12)
    ax2.set_title("Convergence: Loss over Rounds", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, "convergence_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved convergence plot to: {output_path}")

    if show:
        plt.show()
    plt.close()


def plot_stability(
    fedavg_results: Optional[Dict],
    fedprox_results: Optional[Dict],
    output_dir: str,
    show: bool = False,
) -> None:
    """Plot stability metrics comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy changes between rounds
    ax1 = axes[0]
    if fedavg_results:
        acc = np.array(fedavg_results["per_round_metrics"]["global_test_accuracy"])
        diffs = np.abs(np.diff(acc))
        rounds = range(1, len(diffs) + 1)
        ax1.bar([r - 0.2 for r in rounds], diffs, width=0.4, label="FedAvg", color="#1f77b4", alpha=0.7)

    if fedprox_results:
        acc = np.array(fedprox_results["per_round_metrics"]["global_test_accuracy"])
        diffs = np.abs(np.diff(acc))
        rounds = range(1, len(diffs) + 1)
        mu = fedprox_results["experiment"]["mu"]
        ax1.bar([r + 0.2 for r in rounds], diffs, width=0.4, label=f"FedProx (mu={mu})", color="#ff7f0e", alpha=0.7)

    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("Absolute Accuracy Change", fontsize=12)
    ax1.set_title("Stability: Per-Round Accuracy Oscillation", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")

    # Summary stability metrics
    ax2 = axes[1]
    metrics = ["Variance\n(x1000)", "Max\nOscillation", "Smoothness"]
    x = np.arange(len(metrics))
    width = 0.35

    fedavg_vals = []
    fedprox_vals = []

    if fedavg_results:
        stab = fedavg_results["stability"]["accuracy"]
        fedavg_vals = [stab["variance"] * 1000, stab["max_oscillation"], stab["smoothness"]]

    if fedprox_results:
        stab = fedprox_results["stability"]["accuracy"]
        fedprox_vals = [stab["variance"] * 1000, stab["max_oscillation"], stab["smoothness"]]

    if fedavg_vals:
        ax2.bar(x - width/2, fedavg_vals, width, label="FedAvg", color="#1f77b4")
    if fedprox_vals:
        mu = fedprox_results["experiment"]["mu"] if fedprox_results else 0
        ax2.bar(x + width/2, fedprox_vals, width, label=f"FedProx (mu={mu})", color="#ff7f0e")

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel("Value", fontsize=12)
    ax2.set_title("Stability Metrics Comparison", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    output_path = os.path.join(output_dir, "stability_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved stability plot to: {output_path}")

    if show:
        plt.show()
    plt.close()


def print_summary_table(
    fedavg_results: Optional[Dict],
    fedprox_results: Optional[Dict],
) -> None:
    """Print a formatted summary comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY: FedAvg vs FedProx")
    print("=" * 80)

    # Header
    print(f"\n{'Metric':<35} {'FedAvg':>15} {'FedProx':>15} {'Diff':>12}")
    print("-" * 80)

    def get_val(results: Optional[Dict], *keys, default="-"):
        if not results:
            return default
        val = results
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def fmt(val, decimals=4):
        if val == "-" or val is None:
            return "-"
        if isinstance(val, (int, float)):
            return f"{val:.{decimals}f}"
        return str(val)

    def calc_diff(v1, v2, higher_better=True):
        if v1 == "-" or v2 == "-" or v1 is None or v2 is None:
            return "-"
        diff = v2 - v1
        if higher_better:
            return f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
        else:
            return f"{diff:.4f}" if diff > 0 else f"+{abs(diff):.4f}"

    # Accuracy and Loss
    fa_acc = get_val(fedavg_results, "final_metrics", "final_accuracy")
    fp_acc = get_val(fedprox_results, "final_metrics", "final_accuracy")
    print(f"{'Final Accuracy':<35} {fmt(fa_acc):>15} {fmt(fp_acc):>15} {calc_diff(fa_acc, fp_acc):>12}")

    fa_best = get_val(fedavg_results, "final_metrics", "best_accuracy")
    fp_best = get_val(fedprox_results, "final_metrics", "best_accuracy")
    print(f"{'Best Accuracy':<35} {fmt(fa_best):>15} {fmt(fp_best):>15} {calc_diff(fa_best, fp_best):>12}")

    fa_loss = get_val(fedavg_results, "final_metrics", "final_loss")
    fp_loss = get_val(fedprox_results, "final_metrics", "final_loss")
    print(f"{'Final Loss':<35} {fmt(fa_loss):>15} {fmt(fp_loss):>15} {calc_diff(fa_loss, fp_loss, False):>12}")

    # Convergence
    print("-" * 80)
    print("Convergence (rounds to reach threshold, -1 = never):")

    fa_c90 = get_val(fedavg_results, "convergence", "round_to_90_acc")
    fp_c90 = get_val(fedprox_results, "convergence", "round_to_90_acc")
    print(f"{'  Rounds to 90% Accuracy':<35} {fmt(fa_c90, 0):>15} {fmt(fp_c90, 0):>15}")

    fa_c95 = get_val(fedavg_results, "convergence", "round_to_95_acc")
    fp_c95 = get_val(fedprox_results, "convergence", "round_to_95_acc")
    print(f"{'  Rounds to 95% Accuracy':<35} {fmt(fa_c95, 0):>15} {fmt(fp_c95, 0):>15}")

    fa_c98 = get_val(fedavg_results, "convergence", "round_to_98_acc")
    fp_c98 = get_val(fedprox_results, "convergence", "round_to_98_acc")
    print(f"{'  Rounds to 98% Accuracy':<35} {fmt(fa_c98, 0):>15} {fmt(fp_c98, 0):>15}")

    # Stability
    print("-" * 80)
    print("Stability (lower variance/oscillation = more stable):")

    fa_var = get_val(fedavg_results, "stability", "accuracy", "variance")
    fp_var = get_val(fedprox_results, "stability", "accuracy", "variance")
    print(f"{'  Accuracy Variance':<35} {fmt(fa_var, 6):>15} {fmt(fp_var, 6):>15}")

    fa_osc = get_val(fedavg_results, "stability", "accuracy", "max_oscillation")
    fp_osc = get_val(fedprox_results, "stability", "accuracy", "max_oscillation")
    print(f"{'  Max Oscillation':<35} {fmt(fa_osc):>15} {fmt(fp_osc):>15}")

    fa_smooth = get_val(fedavg_results, "stability", "accuracy", "smoothness")
    fp_smooth = get_val(fedprox_results, "stability", "accuracy", "smoothness")
    print(f"{'  Smoothness (higher = smoother)':<35} {fmt(fa_smooth):>15} {fmt(fp_smooth):>15}")

    # Timing
    print("-" * 80)
    fa_time = get_val(fedavg_results, "final_metrics", "total_time_seconds")
    fp_time = get_val(fedprox_results, "final_metrics", "total_time_seconds")
    print(f"{'Total Time (seconds)':<35} {fmt(fa_time, 1):>15} {fmt(fp_time, 1):>15}")

    # Experiment config
    print("-" * 80)
    print("Experiment Configuration:")
    if fedavg_results:
        exp = fedavg_results["experiment"]
        print(f"  FedAvg: rounds={exp['rounds']}, lr={exp['learning_rate']}, local_epochs={exp['local_epochs']}")
    if fedprox_results:
        exp = fedprox_results["experiment"]
        print(f"  FedProx: rounds={exp['rounds']}, lr={exp['learning_rate']}, local_epochs={exp['local_epochs']}, mu={exp['mu']}")

    print("=" * 80)


def save_summary_csv(
    fedavg_results: Optional[Dict],
    fedprox_results: Optional[Dict],
    output_dir: str,
) -> None:
    """Save summary as CSV for further analysis."""
    import csv

    output_path = os.path.join(output_dir, "comparison_summary.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "FedAvg", "FedProx"])

        def get_val(results, *keys):
            if not results:
                return ""
            val = results
            for k in keys:
                if isinstance(val, dict) and k in val:
                    val = val[k]
                else:
                    return ""
            return val

        writer.writerow(["Final Accuracy", get_val(fedavg_results, "final_metrics", "final_accuracy"),
                        get_val(fedprox_results, "final_metrics", "final_accuracy")])
        writer.writerow(["Best Accuracy", get_val(fedavg_results, "final_metrics", "best_accuracy"),
                        get_val(fedprox_results, "final_metrics", "best_accuracy")])
        writer.writerow(["Final Loss", get_val(fedavg_results, "final_metrics", "final_loss"),
                        get_val(fedprox_results, "final_metrics", "final_loss")])
        writer.writerow(["Rounds to 90%", get_val(fedavg_results, "convergence", "round_to_90_acc"),
                        get_val(fedprox_results, "convergence", "round_to_90_acc")])
        writer.writerow(["Rounds to 95%", get_val(fedavg_results, "convergence", "round_to_95_acc"),
                        get_val(fedprox_results, "convergence", "round_to_95_acc")])
        writer.writerow(["Variance", get_val(fedavg_results, "stability", "accuracy", "variance"),
                        get_val(fedprox_results, "stability", "accuracy", "variance")])
        writer.writerow(["Smoothness", get_val(fedavg_results, "stability", "accuracy", "smoothness"),
                        get_val(fedprox_results, "stability", "accuracy", "smoothness")])
        writer.writerow(["Total Time (s)", get_val(fedavg_results, "final_metrics", "total_time_seconds"),
                        get_val(fedprox_results, "final_metrics", "total_time_seconds")])

        if fedprox_results:
            writer.writerow(["FedProx mu", "", get_val(fedprox_results, "experiment", "mu")])

    print(f"Saved summary CSV to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Flower MNIST experiment results")
    parser.add_argument("--results_dir", type=str, default="./results/flower_mnist_2digits",
                       help="Directory containing results JSON files")
    parser.add_argument("--fedavg_file", type=str, default=None,
                       help="Specific FedAvg results file (overrides auto-detection)")
    parser.add_argument("--fedprox_file", type=str, default=None,
                       help="Specific FedProx results file (overrides auto-detection)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots (defaults to results_dir)")
    parser.add_argument("--show", action="store_true",
                       help="Display plots interactively")
    args = parser.parse_args()

    # Set output directory
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Find or load results
    fedavg_results = None
    fedprox_results = None

    if args.fedavg_file:
        fedavg_results = load_results(args.fedavg_file)
        print(f"Loaded FedAvg results from: {args.fedavg_file}")
    else:
        fedavg_file = find_latest_results(args.results_dir, "fedavg")
        if fedavg_file:
            fedavg_results = load_results(fedavg_file)
            print(f"Loaded FedAvg results from: {fedavg_file}")
        else:
            print("No FedAvg results found.")

    if args.fedprox_file:
        fedprox_results = load_results(args.fedprox_file)
        print(f"Loaded FedProx results from: {args.fedprox_file}")
    else:
        fedprox_file = find_latest_results(args.results_dir, "fedprox")
        if fedprox_file:
            fedprox_results = load_results(fedprox_file)
            print(f"Loaded FedProx results from: {fedprox_file}")
        else:
            print("No FedProx results found.")

    if not fedavg_results and not fedprox_results:
        print("\nNo results to analyze. Run experiments first:")
        print("  ./run_flower_local.sh fedavg 30")
        print("  ./run_flower_local.sh fedprox 30 0.01")
        return

    # Generate analysis
    print("\n" + "=" * 60)
    print("Generating Analysis...")
    print("=" * 60)

    plot_convergence(fedavg_results, fedprox_results, output_dir, show=args.show)
    plot_stability(fedavg_results, fedprox_results, output_dir, show=args.show)
    print_summary_table(fedavg_results, fedprox_results)
    save_summary_csv(fedavg_results, fedprox_results, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
