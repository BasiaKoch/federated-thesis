#!/usr/bin/env python3
"""
Universal analysis script for FedAvg vs FedProx comparison.

Parses results.json files and generates:
- Summary tables (per-client and global metrics)
- Convergence plots
- Stability analysis
- Markdown report

Usage:
    python analyze_fed_results.py --results_a path/to/fedavg/results.json \
                                   --results_b path/to/fedprox/results.json \
                                   --out_dir results/analysis/

    # Or compare multiple runs:
    python analyze_fed_results.py --results_a run1.json run2.json \
                                   --results_b run3.json run4.json \
                                   --out_dir results/analysis/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_results(path: Path) -> Dict:
    """Load results.json file."""
    with open(path, "r") as f:
        return json.load(f)


def extract_metrics(data: Dict) -> Dict:
    """Extract key metrics from results data."""
    config = data.get("config", {})
    per_round = data.get("per_round", {})
    final = data.get("final", {})

    # Determine number of clients
    num_clients = config.get("num_clients", 2)

    # Extract rounds
    rounds = per_round.get("rounds", [])

    # Extract per-client metrics
    client_metrics = {}
    for cid in range(num_clients):
        key = f"client{cid}_MeanPresent"
        if key in per_round:
            client_metrics[cid] = {
                "MeanPresent": per_round[key],
                "WT": per_round.get(f"client{cid}_WT", []),
                "TC": per_round.get(f"client{cid}_TC", []),
                "ET": per_round.get(f"client{cid}_ET", []),
            }

    # Extract global metrics
    global_metrics = {
        "MeanPresent": per_round.get("global_MeanPresent", []),
        "WT": per_round.get("global_WT", []),
        "TC": per_round.get("global_TC", []),
        "ET": per_round.get("global_ET", []),
    }

    return {
        "config": config,
        "rounds": rounds,
        "client_metrics": client_metrics,
        "global_metrics": global_metrics,
        "final": final,
        "num_clients": num_clients,
    }


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistics for a metric series."""
    if not values:
        return {"final": 0.0, "best": 0.0, "mean": 0.0, "std": 0.0, "best_round": -1}

    arr = np.array(values)
    best_idx = int(np.argmax(arr))

    return {
        "final": float(arr[-1]),
        "best": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "best_round": best_idx,
        "final_round": len(arr) - 1,
    }


def compute_stability(values: List[float], window: int = 5) -> float:
    """Compute stability as inverse of variance in last N rounds."""
    if len(values) < window:
        return 0.0
    last_values = values[-window:]
    std = np.std(last_values)
    return float(1.0 / (std + 1e-6))


def generate_comparison_table(metrics_a: Dict, metrics_b: Dict,
                               label_a: str, label_b: str) -> str:
    """Generate markdown comparison table."""
    lines = []
    lines.append(f"## Comparison: {label_a} vs {label_b}\n")

    # Config comparison
    cfg_a = metrics_a["config"]
    cfg_b = metrics_b["config"]
    lines.append("### Configuration\n")
    lines.append(f"| Parameter | {label_a} | {label_b} |")
    lines.append("|-----------|---------|---------|")
    lines.append(f"| Strategy | {cfg_a.get('strategy', 'N/A')} | {cfg_b.get('strategy', 'N/A')} |")
    lines.append(f"| μ (mu) | {cfg_a.get('mu', 0.0)} | {cfg_b.get('mu', 0.0)} |")
    lines.append(f"| Rounds | {cfg_a.get('rounds', 'N/A')} | {cfg_b.get('rounds', 'N/A')} |")
    lines.append(f"| Local Epochs | {cfg_a.get('local_epochs', 'N/A')} | {cfg_b.get('local_epochs', 'N/A')} |")
    lines.append(f"| Clients | {cfg_a.get('num_clients', 'N/A')} | {cfg_b.get('num_clients', 'N/A')} |")
    lines.append("")

    # Final metrics comparison
    lines.append("### Final Metrics\n")
    lines.append(f"| Metric | {label_a} | {label_b} | Δ | Winner |")
    lines.append("|--------|---------|---------|---|--------|")

    num_clients = metrics_a["num_clients"]

    # Per-client comparison
    for cid in range(num_clients):
        if cid in metrics_a["client_metrics"] and cid in metrics_b["client_metrics"]:
            vals_a = metrics_a["client_metrics"][cid]["MeanPresent"]
            vals_b = metrics_b["client_metrics"][cid]["MeanPresent"]

            if vals_a and vals_b:
                final_a = vals_a[-1]
                final_b = vals_b[-1]
                delta = final_b - final_a
                delta_pct = (delta / final_a * 100) if final_a > 0 else 0
                winner = label_b if delta > 0.005 else (label_a if delta < -0.005 else "Tie")

                a_str = f"**{final_a:.4f}**" if winner == label_a else f"{final_a:.4f}"
                b_str = f"**{final_b:.4f}**" if winner == label_b else f"{final_b:.4f}"

                lines.append(f"| Client{cid} Final | {a_str} | {b_str} | {delta_pct:+.1f}% | {winner} |")

    # Global comparison
    global_a = metrics_a["global_metrics"]["MeanPresent"]
    global_b = metrics_b["global_metrics"]["MeanPresent"]

    if global_a and global_b:
        final_a = global_a[-1]
        final_b = global_b[-1]
        delta = final_b - final_a
        delta_pct = (delta / final_a * 100) if final_a > 0 else 0
        winner = label_b if delta > 0.005 else (label_a if delta < -0.005 else "Tie")

        a_str = f"**{final_a:.4f}**" if winner == label_a else f"{final_a:.4f}"
        b_str = f"**{final_b:.4f}**" if winner == label_b else f"{final_b:.4f}"

        lines.append(f"| **Global Final** | {a_str} | {b_str} | {delta_pct:+.1f}% | {winner} |")

        # Best metrics
        best_a = max(global_a)
        best_b = max(global_b)
        delta = best_b - best_a
        delta_pct = (delta / best_a * 100) if best_a > 0 else 0
        winner = label_b if delta > 0.005 else (label_a if delta < -0.005 else "Tie")

        lines.append(f"| Global Best | {best_a:.4f} | {best_b:.4f} | {delta_pct:+.1f}% | {winner} |")

    lines.append("")

    # Per-class breakdown
    lines.append("### Per-Class Breakdown (Final)\n")
    lines.append(f"| Class | {label_a} | {label_b} | Δ |")
    lines.append("|-------|---------|---------|---|")

    for cls in ["WT", "TC", "ET"]:
        cls_a = metrics_a["global_metrics"].get(cls, [])
        cls_b = metrics_b["global_metrics"].get(cls, [])

        if cls_a and cls_b:
            final_a = cls_a[-1]
            final_b = cls_b[-1]
            delta = final_b - final_a
            delta_pct = (delta / final_a * 100) if final_a > 0 else 0
            lines.append(f"| {cls} | {final_a:.4f} | {final_b:.4f} | {delta_pct:+.1f}% |")

    lines.append("")

    return "\n".join(lines)


def generate_stability_analysis(metrics_a: Dict, metrics_b: Dict,
                                 label_a: str, label_b: str) -> str:
    """Analyze convergence stability."""
    lines = []
    lines.append("### Stability Analysis (Last 5 Rounds)\n")
    lines.append(f"| Metric | {label_a} Std | {label_b} Std | More Stable |")
    lines.append("|--------|-------------|-------------|-------------|")

    num_clients = metrics_a["num_clients"]

    for cid in range(num_clients):
        if cid in metrics_a["client_metrics"] and cid in metrics_b["client_metrics"]:
            vals_a = metrics_a["client_metrics"][cid]["MeanPresent"]
            vals_b = metrics_b["client_metrics"][cid]["MeanPresent"]

            if len(vals_a) >= 5 and len(vals_b) >= 5:
                std_a = np.std(vals_a[-5:])
                std_b = np.std(vals_b[-5:])
                more_stable = label_b if std_b < std_a else label_a
                lines.append(f"| Client{cid} | {std_a:.4f} | {std_b:.4f} | {more_stable} |")

    global_a = metrics_a["global_metrics"]["MeanPresent"]
    global_b = metrics_b["global_metrics"]["MeanPresent"]

    if len(global_a) >= 5 and len(global_b) >= 5:
        std_a = np.std(global_a[-5:])
        std_b = np.std(global_b[-5:])
        more_stable = label_b if std_b < std_a else label_a
        lines.append(f"| **Global** | {std_a:.4f} | {std_b:.4f} | {more_stable} |")

    lines.append("")
    return "\n".join(lines)


def generate_convergence_summary(metrics_a: Dict, metrics_b: Dict,
                                  label_a: str, label_b: str) -> str:
    """Generate text summary of convergence behavior."""
    lines = []
    lines.append("### Convergence Summary\n")

    num_clients = metrics_a["num_clients"]

    for cid in range(num_clients):
        if cid in metrics_a["client_metrics"] and cid in metrics_b["client_metrics"]:
            vals_a = metrics_a["client_metrics"][cid]["MeanPresent"]
            vals_b = metrics_b["client_metrics"][cid]["MeanPresent"]

            if vals_a and vals_b:
                # Find when each reached 90% of their best
                best_a, best_b = max(vals_a), max(vals_b)
                thresh_a, thresh_b = 0.9 * best_a, 0.9 * best_b

                round_a = next((i for i, v in enumerate(vals_a) if v >= thresh_a), len(vals_a))
                round_b = next((i for i, v in enumerate(vals_b) if v >= thresh_b), len(vals_b))

                lines.append(f"**Client{cid}:**")
                lines.append(f"- {label_a}: Reached 90% of best ({best_a:.3f}) at round {round_a}")
                lines.append(f"- {label_b}: Reached 90% of best ({best_b:.3f}) at round {round_b}")
                lines.append("")

    return "\n".join(lines)


def plot_convergence(metrics_a: Dict, metrics_b: Dict,
                     label_a: str, label_b: str,
                     out_path: Path) -> None:
    """Generate convergence plots."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plots")
        return

    num_clients = metrics_a["num_clients"]
    rounds_a = metrics_a["rounds"]
    rounds_b = metrics_b["rounds"]

    # Create figure with subplots
    fig, axes = plt.subplots(1, num_clients + 1, figsize=(5 * (num_clients + 1), 4))
    if num_clients + 1 == 1:
        axes = [axes]

    # Plot per-client
    for cid in range(num_clients):
        ax = axes[cid]

        if cid in metrics_a["client_metrics"]:
            vals_a = metrics_a["client_metrics"][cid]["MeanPresent"]
            ax.plot(rounds_a[:len(vals_a)], vals_a, marker="o", markersize=3,
                   label=label_a, alpha=0.8)

        if cid in metrics_b["client_metrics"]:
            vals_b = metrics_b["client_metrics"][cid]["MeanPresent"]
            ax.plot(rounds_b[:len(vals_b)], vals_b, marker="s", markersize=3,
                   label=label_b, alpha=0.8)

        ax.set_title(f"Client {cid}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Dice (MeanPresent)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1)

    # Plot global
    ax = axes[-1]
    global_a = metrics_a["global_metrics"]["MeanPresent"]
    global_b = metrics_b["global_metrics"]["MeanPresent"]

    ax.plot(rounds_a[:len(global_a)], global_a, marker="o", markersize=3,
           label=label_a, alpha=0.8, linewidth=2)
    ax.plot(rounds_b[:len(global_b)], global_b, marker="s", markersize=3,
           label=label_b, alpha=0.8, linewidth=2)

    ax.set_title("Global (Pooled)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Dice (MeanPresent)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved convergence plot: {out_path}")


def plot_per_class(metrics_a: Dict, metrics_b: Dict,
                   label_a: str, label_b: str,
                   out_path: Path) -> None:
    """Generate per-class comparison plot."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    rounds_a = metrics_a["rounds"]
    rounds_b = metrics_b["rounds"]

    for idx, cls in enumerate(["WT", "TC", "ET"]):
        ax = axes[idx]

        cls_a = metrics_a["global_metrics"].get(cls, [])
        cls_b = metrics_b["global_metrics"].get(cls, [])

        if cls_a:
            ax.plot(rounds_a[:len(cls_a)], cls_a, marker="o", markersize=3,
                   label=label_a, alpha=0.8)
        if cls_b:
            ax.plot(rounds_b[:len(cls_b)], cls_b, marker="s", markersize=3,
                   label=label_b, alpha=0.8)

        ax.set_title(f"Global {cls} Dice")
        ax.set_xlabel("Round")
        ax.set_ylabel("Dice")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved per-class plot: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Analyze and compare federated learning results")
    ap.add_argument("--results_a", type=Path, required=True,
                    help="Path to first results.json (e.g., FedAvg)")
    ap.add_argument("--results_b", type=Path, required=True,
                    help="Path to second results.json (e.g., FedProx)")
    ap.add_argument("--label_a", type=str, default=None,
                    help="Label for first result (default: auto-detect from config)")
    ap.add_argument("--label_b", type=str, default=None,
                    help="Label for second result (default: auto-detect from config)")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Output directory for plots and report (default: current dir)")
    ap.add_argument("--no_plots", action="store_true",
                    help="Skip generating plots")
    args = ap.parse_args()

    # Load results
    print(f"Loading {args.results_a}...")
    data_a = load_results(args.results_a)
    metrics_a = extract_metrics(data_a)

    print(f"Loading {args.results_b}...")
    data_b = load_results(args.results_b)
    metrics_b = extract_metrics(data_b)

    # Auto-detect labels
    if args.label_a is None:
        cfg = metrics_a["config"]
        mu = cfg.get("mu", 0.0)
        strategy = cfg.get("strategy", "unknown")
        args.label_a = f"{strategy.capitalize()}" + (f" (μ={mu})" if mu > 0 else "")

    if args.label_b is None:
        cfg = metrics_b["config"]
        mu = cfg.get("mu", 0.0)
        strategy = cfg.get("strategy", "unknown")
        args.label_b = f"{strategy.capitalize()}" + (f" (μ={mu})" if mu > 0 else "")

    # Setup output directory
    if args.out_dir is None:
        args.out_dir = Path(".")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Generate report
    report_lines = []
    report_lines.append("# Federated Learning Results Analysis\n")
    report_lines.append(f"**Comparison:** {args.label_a} vs {args.label_b}\n")
    report_lines.append(f"**Results A:** `{args.results_a}`\n")
    report_lines.append(f"**Results B:** `{args.results_b}`\n")
    report_lines.append("")

    # Add comparison table
    report_lines.append(generate_comparison_table(metrics_a, metrics_b,
                                                   args.label_a, args.label_b))

    # Add stability analysis
    report_lines.append(generate_stability_analysis(metrics_a, metrics_b,
                                                     args.label_a, args.label_b))

    # Add convergence summary
    report_lines.append(generate_convergence_summary(metrics_a, metrics_b,
                                                      args.label_a, args.label_b))

    # Print to console
    report_text = "\n".join(report_lines)
    print("\n" + "=" * 70)
    print(report_text)
    print("=" * 70)

    # Save report
    report_path = args.out_dir / "analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nSaved report: {report_path}")

    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        plot_convergence(metrics_a, metrics_b, args.label_a, args.label_b,
                        args.out_dir / "convergence.png")
        plot_per_class(metrics_a, metrics_b, args.label_a, args.label_b,
                      args.out_dir / "per_class.png")

    # Print quick summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)

    global_a = metrics_a["global_metrics"]["MeanPresent"]
    global_b = metrics_b["global_metrics"]["MeanPresent"]

    if global_a and global_b:
        final_a, final_b = global_a[-1], global_b[-1]
        best_a, best_b = max(global_a), max(global_b)

        print(f"\nGlobal Dice (Final): {args.label_a}={final_a:.4f} vs {args.label_b}={final_b:.4f}")
        print(f"Global Dice (Best):  {args.label_a}={best_a:.4f} vs {args.label_b}={best_b:.4f}")

        delta = final_b - final_a
        if abs(delta) < 0.005:
            print(f"\n→ Results are essentially tied (Δ={delta:+.4f})")
        elif delta > 0:
            print(f"\n→ {args.label_b} is BETTER by {delta:.4f} ({delta/final_a*100:+.1f}%)")
        else:
            print(f"\n→ {args.label_a} is BETTER by {-delta:.4f} ({-delta/final_b*100:+.1f}%)")


if __name__ == "__main__":
    main()
