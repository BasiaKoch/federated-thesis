#!/usr/bin/env python3
"""
Comprehensive analysis of FedProx vs FedAvg on ET-skewed federated learning data.

This script generates all visualizations and tables for the thesis results section.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, Any, Tuple

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color scheme
COLORS = {
    'fedprox': '#2E86AB',      # Blue
    'fedavg': '#E94F37',       # Red
    'client0': '#7B2D8E',      # Purple
    'client1': '#F39C12',      # Orange
    'global': '#27AE60',       # Green
    'wt': '#3498DB',           # Light blue
    'tc': '#9B59B6',           # Purple
    'et': '#E74C3C',           # Red
}


def load_results(base_path: Path) -> Tuple[Dict, Dict, Dict]:
    """Load FedProx, FedAvg results and client map."""

    # Find the JSON files
    fedprox_path = base_path / "federated/brats2d_7030_c0ET10/fedprox_mu0.01_R30_E5_20260128_002555/results.json"
    fedavg_path = base_path / "federated/brats2d_7030_c0ET10/fedavg_mu0.0_R30_E5_20260128_103516/results.json"
    client_map_path = base_path.parent / "data/partitions/brats2d_7030_et_skewed/client_map.json"

    with open(fedprox_path) as f:
        fedprox = json.load(f)
    with open(fedavg_path) as f:
        fedavg = json.load(f)
    with open(client_map_path) as f:
        client_map = json.load(f)

    return fedprox, fedavg, client_map


def create_et_distribution_figure(client_map: Dict, output_dir: Path):
    """Figure 1: ET ratio distribution by client."""

    et_ratios = client_map['et_ratios']
    client0_cases = client_map['client_0']
    client1_cases = client_map['client_1']

    c0_ratios = [et_ratios[c] for c in client0_cases if c in et_ratios]
    c1_ratios = [et_ratios[c] for c in client1_cases if c in et_ratios]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Histogram
    ax = axes[0]
    bins = np.linspace(0, 0.8, 25)
    ax.hist(c0_ratios, bins=bins, alpha=0.7, label=f'Client 0 (n={len(c0_ratios)})',
            color=COLORS['client0'], edgecolor='white', linewidth=0.5)
    ax.hist(c1_ratios, bins=bins, alpha=0.7, label=f'Client 1 (n={len(c1_ratios)})',
            color=COLORS['client1'], edgecolor='white', linewidth=0.5)
    ax.axvline(np.mean(c0_ratios), color=COLORS['client0'], linestyle='--', linewidth=2,
               label=f'C0 mean: {np.mean(c0_ratios):.3f}')
    ax.axvline(np.mean(c1_ratios), color=COLORS['client1'], linestyle='--', linewidth=2,
               label=f'C1 mean: {np.mean(c1_ratios):.3f}')
    ax.set_xlabel('ET Ratio (ET voxels / Total tumor voxels)')
    ax.set_ylabel('Number of Cases')
    ax.set_title('(a) ET Ratio Distribution by Client')
    ax.legend(loc='upper right', fontsize=9)

    # (b) Box plot
    ax = axes[1]
    bp = ax.boxplot([c0_ratios, c1_ratios], labels=['Client 0\n(Low-ET)', 'Client 1\n(High-ET)'],
                    patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(COLORS['client0'])
    bp['boxes'][1].set_facecolor(COLORS['client1'])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_ylabel('ET Ratio')
    ax.set_title('(b) ET Ratio Box Plot')

    # Add statistics
    stats_text = (f"Client 0: μ={np.mean(c0_ratios):.3f}, σ={np.std(c0_ratios):.3f}\n"
                  f"Client 1: μ={np.mean(c1_ratios):.3f}, σ={np.std(c1_ratios):.3f}\n"
                  f"Ratio: {np.mean(c1_ratios)/np.mean(c0_ratios):.1f}×")
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # (c) Cumulative distribution
    ax = axes[2]
    c0_sorted = np.sort(c0_ratios)
    c1_sorted = np.sort(c1_ratios)
    ax.plot(c0_sorted, np.arange(1, len(c0_sorted)+1)/len(c0_sorted),
            color=COLORS['client0'], linewidth=2, label='Client 0')
    ax.plot(c1_sorted, np.arange(1, len(c1_sorted)+1)/len(c1_sorted),
            color=COLORS['client1'], linewidth=2, label='Client 1')
    ax.set_xlabel('ET Ratio')
    ax.set_ylabel('Cumulative Proportion')
    ax.set_title('(c) Cumulative Distribution Function')
    ax.legend()
    ax.set_xlim(0, 0.8)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_et_distribution.png')
    plt.savefig(output_dir / 'fig1_et_distribution.pdf')
    plt.close()

    print("Created: fig1_et_distribution.png/pdf")


def create_global_convergence_figure(fedprox: Dict, fedavg: Dict, output_dir: Path):
    """Figure 2: Global convergence curves."""

    rounds = fedprox['per_round']['rounds']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ('global_meanDice', 'Mean Dice', '(a) Global Mean Dice'),
        ('global_WT', 'WT Dice', '(b) Whole Tumor (WT)'),
        ('global_TC', 'TC Dice', '(c) Tumor Core (TC)'),
        ('global_ET', 'ET Dice', '(d) Enhancing Tumor (ET)')
    ]

    for ax, (metric, ylabel, title) in zip(axes.flat, metrics):
        fp_data = fedprox['per_round'][metric]
        fa_data = fedavg['per_round'][metric]

        ax.plot(rounds, fp_data, color=COLORS['fedprox'], linewidth=2,
                label='FedProx (μ=0.01)', marker='o', markersize=3, markevery=3)
        ax.plot(rounds, fa_data, color=COLORS['fedavg'], linewidth=2,
                label='FedAvg', marker='s', markersize=3, markevery=3)

        ax.set_xlabel('Round')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.set_xlim(0, 30)
        ax.set_ylim(0, max(max(fp_data), max(fa_data)) * 1.1)

        # Add final values annotation
        ax.annotate(f'{fp_data[-1]:.3f}', xy=(30, fp_data[-1]),
                    xytext=(27, fp_data[-1]+0.03), fontsize=9, color=COLORS['fedprox'])
        ax.annotate(f'{fa_data[-1]:.3f}', xy=(30, fa_data[-1]),
                    xytext=(27, fa_data[-1]-0.05), fontsize=9, color=COLORS['fedavg'])

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_global_convergence.png')
    plt.savefig(output_dir / 'fig2_global_convergence.pdf')
    plt.close()

    print("Created: fig2_global_convergence.png/pdf")


def create_perclient_performance_figure(fedprox: Dict, fedavg: Dict, output_dir: Path):
    """Figure 3: Per-client performance comparison."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Mean Dice by client
    ax = axes[0]
    x = np.arange(2)
    width = 0.35

    fp_means = [fedprox['final']['client0_meanDice'], fedprox['final']['client1_meanDice']]
    fa_means = [fedavg['final']['client0_meanDice'], fedavg['final']['client1_meanDice']]

    bars1 = ax.bar(x - width/2, fp_means, width, label='FedProx (μ=0.01)',
                   color=COLORS['fedprox'], alpha=0.8)
    bars2 = ax.bar(x + width/2, fa_means, width, label='FedAvg',
                   color=COLORS['fedavg'], alpha=0.8)

    ax.set_ylabel('Mean Dice Score')
    ax.set_title('(a) Final Mean Dice by Client')
    ax.set_xticks(x)
    ax.set_xticklabels(['Client 0\n(Low-ET, 57%)', 'Client 1\n(High-ET, 43%)'])
    ax.legend(loc='upper left')
    ax.set_ylim(0, 0.85)

    # Add value labels
    for bar, val in zip(bars1, fp_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, fa_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Add gap annotations
    fp_gap = fp_means[1] - fp_means[0]
    fa_gap = fa_means[1] - fa_means[0]
    ax.annotate(f'Gap: {fp_gap:.3f}', xy=(0.5, 0.78), fontsize=10,
                ha='center', color=COLORS['fedprox'])
    ax.annotate(f'Gap: {fa_gap:.3f}', xy=(0.5, 0.73), fontsize=10,
                ha='center', color=COLORS['fedavg'])

    # (b) Per-class performance
    ax = axes[1]
    classes = ['WT', 'TC', 'ET']
    x = np.arange(len(classes))
    width = 0.2

    # Get final round values
    fp_c0 = [fedprox['per_round']['client0_WT'][-1],
             fedprox['per_round']['client0_TC'][-1],
             fedprox['per_round']['client0_ET'][-1]]
    fp_c1 = [fedprox['per_round']['client1_WT'][-1],
             fedprox['per_round']['client1_TC'][-1],
             fedprox['per_round']['client1_ET'][-1]]
    fa_c0 = [fedavg['per_round']['client0_WT'][-1],
             fedavg['per_round']['client0_TC'][-1],
             fedavg['per_round']['client0_ET'][-1]]
    fa_c1 = [fedavg['per_round']['client1_WT'][-1],
             fedavg['per_round']['client1_TC'][-1],
             fedavg['per_round']['client1_ET'][-1]]

    ax.bar(x - 1.5*width, fp_c0, width, label='FedProx C0', color=COLORS['fedprox'], alpha=0.6)
    ax.bar(x - 0.5*width, fp_c1, width, label='FedProx C1', color=COLORS['fedprox'], alpha=1.0)
    ax.bar(x + 0.5*width, fa_c0, width, label='FedAvg C0', color=COLORS['fedavg'], alpha=0.6)
    ax.bar(x + 1.5*width, fa_c1, width, label='FedAvg C1', color=COLORS['fedavg'], alpha=1.0)

    ax.set_ylabel('Dice Score')
    ax.set_title('(b) Per-Class Dice by Client')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    ax.set_ylim(0, 0.9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_perclient_performance.png')
    plt.savefig(output_dir / 'fig3_perclient_performance.pdf')
    plt.close()

    print("Created: fig3_perclient_performance.png/pdf")


def create_et_convergence_figure(fedprox: Dict, fedavg: Dict, output_dir: Path):
    """Figure 4: ET convergence by client (MOST IMPORTANT FIGURE)."""

    rounds = fedprox['per_round']['rounds']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) ET Dice by client
    ax = axes[0]

    # FedProx
    ax.plot(rounds, fedprox['per_round']['client0_ET'],
            color=COLORS['client0'], linewidth=2, linestyle='-',
            label='FedProx C0 (Low-ET)', marker='o', markersize=4, markevery=3)
    ax.plot(rounds, fedprox['per_round']['client1_ET'],
            color=COLORS['client1'], linewidth=2, linestyle='-',
            label='FedProx C1 (High-ET)', marker='o', markersize=4, markevery=3)

    # FedAvg
    ax.plot(rounds, fedavg['per_round']['client0_ET'],
            color=COLORS['client0'], linewidth=2, linestyle='--',
            label='FedAvg C0 (Low-ET)', marker='s', markersize=4, markevery=3)
    ax.plot(rounds, fedavg['per_round']['client1_ET'],
            color=COLORS['client1'], linewidth=2, linestyle='--',
            label='FedAvg C1 (High-ET)', marker='s', markersize=4, markevery=3)

    ax.set_xlabel('Round')
    ax.set_ylabel('ET Dice Score')
    ax.set_title('(a) ET Dice Convergence by Client')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 0.7)

    # Add shaded region showing the gap
    ax.fill_between(rounds,
                    fedprox['per_round']['client0_ET'],
                    fedprox['per_round']['client1_ET'],
                    alpha=0.1, color='gray', label='_nolegend_')

    # (b) ET gap over time
    ax = axes[1]

    fp_gap = np.array(fedprox['per_round']['client1_ET']) - np.array(fedprox['per_round']['client0_ET'])
    fa_gap = np.array(fedavg['per_round']['client1_ET']) - np.array(fedavg['per_round']['client0_ET'])

    ax.plot(rounds, fp_gap, color=COLORS['fedprox'], linewidth=2,
            label='FedProx Gap (C1-C0)', marker='o', markersize=4, markevery=3)
    ax.plot(rounds, fa_gap, color=COLORS['fedavg'], linewidth=2,
            label='FedAvg Gap (C1-C0)', marker='s', markersize=4, markevery=3)

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Round')
    ax.set_ylabel('ET Dice Gap (Client 1 - Client 0)')
    ax.set_title('(b) Client Performance Gap on ET')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 30)

    # Annotate final gaps
    ax.annotate(f'Final: {fp_gap[-1]:.3f}', xy=(30, fp_gap[-1]),
                xytext=(25, fp_gap[-1]+0.02), fontsize=10, color=COLORS['fedprox'])
    ax.annotate(f'Final: {fa_gap[-1]:.3f}', xy=(30, fa_gap[-1]),
                xytext=(25, fa_gap[-1]-0.03), fontsize=10, color=COLORS['fedavg'])

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_et_convergence.png')
    plt.savefig(output_dir / 'fig4_et_convergence.pdf')
    plt.close()

    print("Created: fig4_et_convergence.png/pdf")


def create_stability_figure(fedprox: Dict, fedavg: Dict, output_dir: Path):
    """Figure 5: Training stability analysis."""

    rounds = fedprox['per_round']['rounds'][1:]  # Skip round 0 for diff

    fp_global = np.array(fedprox['per_round']['global_meanDice'])
    fa_global = np.array(fedavg['per_round']['global_meanDice'])

    fp_diff = np.diff(fp_global)
    fa_diff = np.diff(fa_global)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) Round-over-round changes
    ax = axes[0]
    ax.bar(np.array(rounds) - 0.2, fp_diff, 0.4, label='FedProx',
           color=COLORS['fedprox'], alpha=0.7)
    ax.bar(np.array(rounds) + 0.2, fa_diff, 0.4, label='FedAvg',
           color=COLORS['fedavg'], alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Round')
    ax.set_ylabel('Δ Mean Dice')
    ax.set_title('(a) Round-over-Round Change')
    ax.legend()
    ax.set_xlim(0, 31)

    # (b) Cumulative performance vs best
    ax = axes[1]
    fp_best = np.maximum.accumulate(fp_global)
    fa_best = np.maximum.accumulate(fa_global)

    ax.plot(fedprox['per_round']['rounds'], fp_global, color=COLORS['fedprox'],
            linewidth=2, label='FedProx Current')
    ax.plot(fedprox['per_round']['rounds'], fp_best, color=COLORS['fedprox'],
            linewidth=1, linestyle='--', label='FedProx Best-so-far')
    ax.plot(fedprox['per_round']['rounds'], fa_global, color=COLORS['fedavg'],
            linewidth=2, label='FedAvg Current')
    ax.plot(fedprox['per_round']['rounds'], fa_best, color=COLORS['fedavg'],
            linewidth=1, linestyle='--', label='FedAvg Best-so-far')

    ax.set_xlabel('Round')
    ax.set_ylabel('Mean Dice')
    ax.set_title('(b) Current vs Best-so-far Performance')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 30)

    # (c) Performance drop from best
    ax = axes[2]
    fp_drop = fp_best - fp_global
    fa_drop = fa_best - fa_global

    ax.fill_between(fedprox['per_round']['rounds'], fp_drop, alpha=0.5,
                    color=COLORS['fedprox'], label='FedProx')
    ax.fill_between(fedprox['per_round']['rounds'], fa_drop, alpha=0.5,
                    color=COLORS['fedavg'], label='FedAvg')
    ax.plot(fedprox['per_round']['rounds'], fp_drop, color=COLORS['fedprox'], linewidth=2)
    ax.plot(fedprox['per_round']['rounds'], fa_drop, color=COLORS['fedavg'], linewidth=2)

    ax.set_xlabel('Round')
    ax.set_ylabel('Drop from Best (Best - Current)')
    ax.set_title('(c) Performance Drop from Best')
    ax.legend()
    ax.set_xlim(0, 30)

    # Annotate final drops
    ax.annotate(f'Final: {fp_drop[-1]:.3f}', xy=(30, fp_drop[-1]),
                xytext=(25, 0.02), fontsize=10, color=COLORS['fedprox'])
    ax.annotate(f'Final: {fa_drop[-1]:.3f}', xy=(30, fa_drop[-1]),
                xytext=(25, fa_drop[-1]+0.01), fontsize=10, color=COLORS['fedavg'])

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_stability.png')
    plt.savefig(output_dir / 'fig5_stability.pdf')
    plt.close()

    print("Created: fig5_stability.png/pdf")


def create_heatmap_figure(fedprox: Dict, fedavg: Dict, output_dir: Path):
    """Figure 6: Heatmap of per-client, per-class performance."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Prepare data
    classes = ['WT', 'TC', 'ET', 'Mean']
    clients = ['Client 0\n(Low-ET)', 'Client 1\n(High-ET)', 'Global']

    fp_data = np.array([
        [fedprox['per_round']['client0_WT'][-1], fedprox['per_round']['client0_TC'][-1],
         fedprox['per_round']['client0_ET'][-1], fedprox['final']['client0_meanDice']],
        [fedprox['per_round']['client1_WT'][-1], fedprox['per_round']['client1_TC'][-1],
         fedprox['per_round']['client1_ET'][-1], fedprox['final']['client1_meanDice']],
        [fedprox['per_round']['global_WT'][-1], fedprox['per_round']['global_TC'][-1],
         fedprox['per_round']['global_ET'][-1], fedprox['final']['global_meanDice']]
    ])

    fa_data = np.array([
        [fedavg['per_round']['client0_WT'][-1], fedavg['per_round']['client0_TC'][-1],
         fedavg['per_round']['client0_ET'][-1], fedavg['final']['client0_meanDice']],
        [fedavg['per_round']['client1_WT'][-1], fedavg['per_round']['client1_TC'][-1],
         fedavg['per_round']['client1_ET'][-1], fedavg['final']['client1_meanDice']],
        [fedavg['per_round']['global_WT'][-1], fedavg['per_round']['global_TC'][-1],
         fedavg['per_round']['global_ET'][-1], fedavg['final']['global_meanDice']]
    ])

    for ax, data, title in [(axes[0], fp_data, 'FedProx (μ=0.01)'),
                             (axes[1], fa_data, 'FedAvg')]:
        im = ax.imshow(data, cmap='RdYlGn', vmin=0.4, vmax=0.8, aspect='auto')

        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(clients)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(clients)
        ax.set_title(title)

        # Add text annotations
        for i in range(len(clients)):
            for j in range(len(classes)):
                text = ax.text(j, i, f'{data[i, j]:.3f}',
                              ha='center', va='center', color='black', fontsize=11)

    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04,
                 label='Dice Score')

    plt.suptitle('Per-Client, Per-Class Dice Scores', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_heatmap.png')
    plt.savefig(output_dir / 'fig6_heatmap.pdf')
    plt.close()

    print("Created: fig6_heatmap.png/pdf")


def create_summary_comparison_figure(fedprox: Dict, fedavg: Dict, output_dir: Path):
    """Figure 7: Summary comparison bar chart."""

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Global\nMean', 'Global\nWT', 'Global\nTC', 'Global\nET',
               'Client 0\nMean', 'Client 1\nMean']

    fp_values = [
        fedprox['final']['global_meanDice'],
        fedprox['per_round']['global_WT'][-1],
        fedprox['per_round']['global_TC'][-1],
        fedprox['per_round']['global_ET'][-1],
        fedprox['final']['client0_meanDice'],
        fedprox['final']['client1_meanDice']
    ]

    fa_values = [
        fedavg['final']['global_meanDice'],
        fedavg['per_round']['global_WT'][-1],
        fedavg['per_round']['global_TC'][-1],
        fedavg['per_round']['global_ET'][-1],
        fedavg['final']['client0_meanDice'],
        fedavg['final']['client1_meanDice']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, fp_values, width, label='FedProx (μ=0.01)',
                   color=COLORS['fedprox'], alpha=0.8)
    bars2 = ax.bar(x + width/2, fa_values, width, label='FedAvg',
                   color=COLORS['fedavg'], alpha=0.8)

    ax.set_ylabel('Dice Score')
    ax.set_title('FedProx vs FedAvg: Final Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 0.85)

    # Add value labels and improvement percentages
    for i, (fp, fa) in enumerate(zip(fp_values, fa_values)):
        improvement = (fp - fa) / fa * 100
        color = 'green' if improvement > 0 else 'red'
        ax.annotate(f'+{improvement:.1f}%' if improvement > 0 else f'{improvement:.1f}%',
                   xy=(i, max(fp, fa) + 0.02), ha='center', fontsize=9, color=color)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_summary_comparison.png')
    plt.savefig(output_dir / 'fig7_summary_comparison.pdf')
    plt.close()

    print("Created: fig7_summary_comparison.png/pdf")


def generate_tables(fedprox: Dict, fedavg: Dict, client_map: Dict, output_dir: Path):
    """Generate all LaTeX and CSV tables."""

    # Table 1: Data Distribution
    c0_ratios = [client_map['et_ratios'][c] for c in client_map['client_0'] if c in client_map['et_ratios']]
    c1_ratios = [client_map['et_ratios'][c] for c in client_map['client_1'] if c in client_map['et_ratios']]

    table1_data = {
        'Property': ['Total cases', 'Percentage', 'Train/Val/Test',
                     'ET ratio (mean)', 'ET ratio (median)', 'ET ratio (std)'],
        'Client 0 (Low-ET)': [
            len(client_map['client_0']),
            f"{len(client_map['client_0'])/(len(client_map['client_0'])+len(client_map['client_1']))*100:.1f}%",
            '135/17/17',
            f"{np.mean(c0_ratios):.3f}",
            f"{np.median(c0_ratios):.3f}",
            f"{np.std(c0_ratios):.3f}"
        ],
        'Client 1 (High-ET)': [
            len(client_map['client_1']),
            f"{len(client_map['client_1'])/(len(client_map['client_0'])+len(client_map['client_1']))*100:.1f}%",
            '100/13/13',
            f"{np.mean(c1_ratios):.3f}",
            f"{np.median(c1_ratios):.3f}",
            f"{np.std(c1_ratios):.3f}"
        ]
    }
    df1 = pd.DataFrame(table1_data)
    df1.to_csv(output_dir / 'table1_data_distribution.csv', index=False)
    df1.to_latex(output_dir / 'table1_data_distribution.tex', index=False,
                 caption='Client Data Distribution', label='tab:data_dist')

    # Table 2: Global Performance
    table2_data = {
        'Metric': ['Mean Dice', 'WT Dice', 'TC Dice', 'ET Dice'],
        'FedProx (μ=0.01)': [
            f"{fedprox['final']['global_meanDice']:.3f}",
            f"{fedprox['per_round']['global_WT'][-1]:.3f}",
            f"{fedprox['per_round']['global_TC'][-1]:.3f}",
            f"{fedprox['per_round']['global_ET'][-1]:.3f}"
        ],
        'FedAvg': [
            f"{fedavg['final']['global_meanDice']:.3f}",
            f"{fedavg['per_round']['global_WT'][-1]:.3f}",
            f"{fedavg['per_round']['global_TC'][-1]:.3f}",
            f"{fedavg['per_round']['global_ET'][-1]:.3f}"
        ],
        'Δ (pp)': [
            f"+{(fedprox['final']['global_meanDice'] - fedavg['final']['global_meanDice'])*100:.1f}",
            f"+{(fedprox['per_round']['global_WT'][-1] - fedavg['per_round']['global_WT'][-1])*100:.1f}",
            f"+{(fedprox['per_round']['global_TC'][-1] - fedavg['per_round']['global_TC'][-1])*100:.1f}",
            f"+{(fedprox['per_round']['global_ET'][-1] - fedavg['per_round']['global_ET'][-1])*100:.1f}"
        ]
    }
    df2 = pd.DataFrame(table2_data)
    df2.to_csv(output_dir / 'table2_global_performance.csv', index=False)
    df2.to_latex(output_dir / 'table2_global_performance.tex', index=False,
                 caption='Global Segmentation Performance', label='tab:global_perf')

    # Table 3: Per-Client Performance
    table3_data = {
        'Client': ['Client 0', 'Client 0', 'Client 1', 'Client 1'],
        'Strategy': ['FedProx', 'FedAvg', 'FedProx', 'FedAvg'],
        'WT': [
            f"{fedprox['per_round']['client0_WT'][-1]:.3f}",
            f"{fedavg['per_round']['client0_WT'][-1]:.3f}",
            f"{fedprox['per_round']['client1_WT'][-1]:.3f}",
            f"{fedavg['per_round']['client1_WT'][-1]:.3f}"
        ],
        'TC': [
            f"{fedprox['per_round']['client0_TC'][-1]:.3f}",
            f"{fedavg['per_round']['client0_TC'][-1]:.3f}",
            f"{fedprox['per_round']['client1_TC'][-1]:.3f}",
            f"{fedavg['per_round']['client1_TC'][-1]:.3f}"
        ],
        'ET': [
            f"{fedprox['per_round']['client0_ET'][-1]:.3f}",
            f"{fedavg['per_round']['client0_ET'][-1]:.3f}",
            f"{fedprox['per_round']['client1_ET'][-1]:.3f}",
            f"{fedavg['per_round']['client1_ET'][-1]:.3f}"
        ],
        'Mean': [
            f"{fedprox['final']['client0_meanDice']:.3f}",
            f"{fedavg['final']['client0_meanDice']:.3f}",
            f"{fedprox['final']['client1_meanDice']:.3f}",
            f"{fedavg['final']['client1_meanDice']:.3f}"
        ]
    }
    df3 = pd.DataFrame(table3_data)
    df3.to_csv(output_dir / 'table3_perclient_performance.csv', index=False)
    df3.to_latex(output_dir / 'table3_perclient_performance.tex', index=False,
                 caption='Per-Client, Per-Class Dice Scores', label='tab:perclient_perf')

    # Table 4: Stability Metrics
    fp_global = np.array(fedprox['per_round']['global_meanDice'])
    fa_global = np.array(fedavg['per_round']['global_meanDice'])

    table4_data = {
        'Metric': ['Final Mean Dice', 'Best Mean Dice', 'Final = Best?',
                   'Drop from Best (pp)', 'Positive rounds', 'Max single-round drop'],
        'FedProx (μ=0.01)': [
            f"{fedprox['final']['global_meanDice']:.3f}",
            f"{fedprox['final']['global_best_meanDice']:.3f}",
            'Yes' if abs(fedprox['final']['global_meanDice'] - fedprox['final']['global_best_meanDice']) < 0.001 else 'No',
            f"{(fedprox['final']['global_best_meanDice'] - fedprox['final']['global_meanDice'])*100:.1f}",
            f"{np.sum(np.diff(fp_global) > 0)}/30",
            f"{np.min(np.diff(fp_global))*100:.1f}"
        ],
        'FedAvg': [
            f"{fedavg['final']['global_meanDice']:.3f}",
            f"{fedavg['final']['global_best_meanDice']:.3f}",
            'Yes' if abs(fedavg['final']['global_meanDice'] - fedavg['final']['global_best_meanDice']) < 0.001 else 'No',
            f"{(fedavg['final']['global_best_meanDice'] - fedavg['final']['global_meanDice'])*100:.1f}",
            f"{np.sum(np.diff(fa_global) > 0)}/30",
            f"{np.min(np.diff(fa_global))*100:.1f}"
        ]
    }
    df4 = pd.DataFrame(table4_data)
    df4.to_csv(output_dir / 'table4_stability.csv', index=False)
    df4.to_latex(output_dir / 'table4_stability.tex', index=False,
                 caption='Training Stability Metrics', label='tab:stability')

    # Table 5: ET Progression over rounds
    rounds_to_show = [0, 5, 10, 15, 20, 25, 30]
    table5_data = {'Round': rounds_to_show}
    for name, data in [('FedProx Global', fedprox['per_round']['global_ET']),
                       ('FedAvg Global', fedavg['per_round']['global_ET']),
                       ('FedProx C0', fedprox['per_round']['client0_ET']),
                       ('FedAvg C0', fedavg['per_round']['client0_ET']),
                       ('FedProx C1', fedprox['per_round']['client1_ET']),
                       ('FedAvg C1', fedavg['per_round']['client1_ET'])]:
        table5_data[name] = [f"{data[r]:.3f}" for r in rounds_to_show]

    df5 = pd.DataFrame(table5_data)
    df5.to_csv(output_dir / 'table5_et_progression.csv', index=False)
    df5.to_latex(output_dir / 'table5_et_progression.tex', index=False,
                 caption='ET Dice Score Progression Over Training', label='tab:et_prog')

    print("Created: table1-5 (CSV and LaTeX)")


def generate_full_results_csv(fedprox: Dict, fedavg: Dict, output_dir: Path):
    """Generate comprehensive per-round results CSV."""

    rounds = fedprox['per_round']['rounds']

    data = {
        'round': rounds,
        'fedprox_global_mean': fedprox['per_round']['global_meanDice'],
        'fedprox_global_wt': fedprox['per_round']['global_WT'],
        'fedprox_global_tc': fedprox['per_round']['global_TC'],
        'fedprox_global_et': fedprox['per_round']['global_ET'],
        'fedprox_c0_mean': fedprox['per_round']['client0_meanDice'],
        'fedprox_c0_wt': fedprox['per_round']['client0_WT'],
        'fedprox_c0_tc': fedprox['per_round']['client0_TC'],
        'fedprox_c0_et': fedprox['per_round']['client0_ET'],
        'fedprox_c1_mean': fedprox['per_round']['client1_meanDice'],
        'fedprox_c1_wt': fedprox['per_round']['client1_WT'],
        'fedprox_c1_tc': fedprox['per_round']['client1_TC'],
        'fedprox_c1_et': fedprox['per_round']['client1_ET'],
        'fedavg_global_mean': fedavg['per_round']['global_meanDice'],
        'fedavg_global_wt': fedavg['per_round']['global_WT'],
        'fedavg_global_tc': fedavg['per_round']['global_TC'],
        'fedavg_global_et': fedavg['per_round']['global_ET'],
        'fedavg_c0_mean': fedavg['per_round']['client0_meanDice'],
        'fedavg_c0_wt': fedavg['per_round']['client0_WT'],
        'fedavg_c0_tc': fedavg['per_round']['client0_TC'],
        'fedavg_c0_et': fedavg['per_round']['client0_ET'],
        'fedavg_c1_mean': fedavg['per_round']['client1_meanDice'],
        'fedavg_c1_wt': fedavg['per_round']['client1_WT'],
        'fedavg_c1_tc': fedavg['per_round']['client1_TC'],
        'fedavg_c1_et': fedavg['per_round']['client1_ET'],
    }

    df = pd.DataFrame(data)
    df.to_csv(output_dir / 'full_results_per_round.csv', index=False)
    print("Created: full_results_per_round.csv")


def main():
    """Main function to generate all visualizations and tables."""

    # Setup paths
    script_dir = Path(__file__).parent
    results_base = script_dir.parent
    output_dir = script_dir

    print("=" * 60)
    print("FedProx vs FedAvg Analysis on ET-Skewed Data")
    print("=" * 60)

    # Load data
    print("\nLoading results...")
    fedprox, fedavg, client_map = load_results(results_base)

    print(f"FedProx config: {fedprox['config']}")
    print(f"FedAvg config: {fedavg['config']}")

    # Generate figures
    print("\nGenerating figures...")
    create_et_distribution_figure(client_map, output_dir)
    create_global_convergence_figure(fedprox, fedavg, output_dir)
    create_perclient_performance_figure(fedprox, fedavg, output_dir)
    create_et_convergence_figure(fedprox, fedavg, output_dir)
    create_stability_figure(fedprox, fedavg, output_dir)
    create_heatmap_figure(fedprox, fedavg, output_dir)
    create_summary_comparison_figure(fedprox, fedavg, output_dir)

    # Generate tables
    print("\nGenerating tables...")
    generate_tables(fedprox, fedavg, client_map, output_dir)
    generate_full_results_csv(fedprox, fedavg, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 60)
    print(f"\nGlobal Mean Dice:")
    print(f"  FedProx: {fedprox['final']['global_meanDice']:.3f}")
    print(f"  FedAvg:  {fedavg['final']['global_meanDice']:.3f}")
    print(f"  Δ: +{(fedprox['final']['global_meanDice'] - fedavg['final']['global_meanDice'])*100:.1f} pp")

    print(f"\nGlobal ET Dice:")
    print(f"  FedProx: {fedprox['per_round']['global_ET'][-1]:.3f}")
    print(f"  FedAvg:  {fedavg['per_round']['global_ET'][-1]:.3f}")
    print(f"  Δ: +{(fedprox['per_round']['global_ET'][-1] - fedavg['per_round']['global_ET'][-1])*100:.1f} pp")

    print(f"\nClient Gap (Mean Dice):")
    fp_gap = fedprox['final']['client1_meanDice'] - fedprox['final']['client0_meanDice']
    fa_gap = fedavg['final']['client1_meanDice'] - fedavg['final']['client0_meanDice']
    print(f"  FedProx: {fp_gap:.3f}")
    print(f"  FedAvg:  {fa_gap:.3f}")

    print(f"\nStability (Final vs Best):")
    print(f"  FedProx: Final={fedprox['final']['global_meanDice']:.3f}, Best={fedprox['final']['global_best_meanDice']:.3f}")
    print(f"  FedAvg:  Final={fedavg['final']['global_meanDice']:.3f}, Best={fedavg['final']['global_best_meanDice']:.3f}")

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
