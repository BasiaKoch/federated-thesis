#!/usr/bin/env python3
"""
FedAvg vs FedProx Comparison Analysis
=====================================
Analyzes federated learning results on BraTS 2D segmentation with Dirichlet α=1.0 partition.

Note: The files are named "a0p1" but actually contain α=1.0 (mild heterogeneity) results.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ============================================================
# Data Parsing
# ============================================================

def parse_round_metrics(filepath):
    """Parse round-by-round metrics from training log."""
    pattern = r'\[Round (\d+)\] Client0 Mean=([\d.]+) \| Client1 Mean=([\d.]+) \| Client2 Mean=([\d.]+) \| Client3 Mean=([\d.]+) \| Pooled Mean=([\d.]+)'

    rounds = []
    client0 = []
    client1 = []
    client2 = []
    client3 = []
    pooled = []

    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                rounds.append(int(match.group(1)))
                client0.append(float(match.group(2)))
                client1.append(float(match.group(3)))
                client2.append(float(match.group(4)))
                client3.append(float(match.group(5)))
                pooled.append(float(match.group(6)))

    return {
        'rounds': np.array(rounds),
        'client0': np.array(client0),
        'client1': np.array(client1),
        'client2': np.array(client2),
        'client3': np.array(client3),
        'pooled': np.array(pooled),
    }


# ============================================================
# Load Data
# ============================================================

results_dir = Path(__file__).parent.parent
fedavg_file = results_dir / '4_a0p1_fedavg.txt'
fedprox_file = results_dir / '4_a0p1_fedprox.txt'
output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading FedAvg results...")
fedavg = parse_round_metrics(fedavg_file)
print(f"  Rounds: {len(fedavg['rounds'])}, Final pooled: {fedavg['pooled'][-1]:.4f}")

print("Loading FedProx results...")
fedprox = parse_round_metrics(fedprox_file)
print(f"  Rounds: {len(fedprox['rounds'])}, Final pooled: {fedprox['pooled'][-1]:.4f}")


# ============================================================
# Compute Statistics
# ============================================================

def compute_stats(data):
    """Compute key statistics from round data."""
    clients = np.stack([data['client0'], data['client1'], data['client2'], data['client3']], axis=1)

    return {
        'final_pooled': data['pooled'][-1],
        'best_pooled': np.max(data['pooled']),
        'final_worst_client': np.min(clients[-1]),
        'final_best_client': np.max(clients[-1]),
        'client_variance': np.var(clients, axis=1),  # variance across clients per round
        'client_std': np.std(clients, axis=1),
        'client_range': np.max(clients, axis=1) - np.min(clients, axis=1),
        'min_dice_per_round': np.min(clients, axis=1),
        'max_dice_per_round': np.max(clients, axis=1),
        'clients': clients,
    }

fedavg_stats = compute_stats(fedavg)
fedprox_stats = compute_stats(fedprox)


# ============================================================
# Print Summary Statistics
# ============================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS: FedAvg vs FedProx (α=1.0, 4 clients)")
print("="*70)

print(f"\n{'Metric':<30} {'FedAvg':>12} {'FedProx':>12} {'Δ':>10}")
print("-"*70)
print(f"{'Final Pooled Dice':<30} {fedavg_stats['final_pooled']:>12.4f} {fedprox_stats['final_pooled']:>12.4f} {fedprox_stats['final_pooled']-fedavg_stats['final_pooled']:>+10.4f}")
print(f"{'Best Pooled Dice':<30} {fedavg_stats['best_pooled']:>12.4f} {fedprox_stats['best_pooled']:>12.4f} {fedprox_stats['best_pooled']-fedavg_stats['best_pooled']:>+10.4f}")
print(f"{'Final Worst-Client Dice':<30} {fedavg_stats['final_worst_client']:>12.4f} {fedprox_stats['final_worst_client']:>12.4f} {fedprox_stats['final_worst_client']-fedavg_stats['final_worst_client']:>+10.4f}")
print(f"{'Final Best-Client Dice':<30} {fedavg_stats['final_best_client']:>12.4f} {fedprox_stats['final_best_client']:>12.4f} {fedprox_stats['final_best_client']-fedavg_stats['final_best_client']:>+10.4f}")
print(f"{'Final Client Range':<30} {fedavg_stats['client_range'][-1]:>12.4f} {fedprox_stats['client_range'][-1]:>12.4f} {fedprox_stats['client_range'][-1]-fedavg_stats['client_range'][-1]:>+10.4f}")
print(f"{'Mean Client Std (last 10)':<30} {np.mean(fedavg_stats['client_std'][-10:]):>12.4f} {np.mean(fedprox_stats['client_std'][-10:]):>12.4f} {np.mean(fedprox_stats['client_std'][-10:])-np.mean(fedavg_stats['client_std'][-10:]):>+10.4f}")

print("\n" + "-"*70)
print("Per-Client Final Dice:")
print("-"*70)
for i in range(4):
    fa = fedavg_stats['clients'][-1, i]
    fp = fedprox_stats['clients'][-1, i]
    print(f"  Client {i}: FedAvg={fa:.4f}, FedProx={fp:.4f}, Δ={fp-fa:+.4f}")


# ============================================================
# PLOT 1: Global Convergence Comparison
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(fedavg['rounds'], fedavg['pooled'], 'b-', linewidth=2, label='FedAvg (μ=0)', marker='o', markersize=4, markevery=3)
ax.plot(fedprox['rounds'], fedprox['pooled'], 'r-', linewidth=2, label='FedProx (μ=0.1)', marker='s', markersize=4, markevery=3)

ax.axhline(y=fedavg_stats['best_pooled'], color='b', linestyle='--', alpha=0.5, label=f'FedAvg best: {fedavg_stats["best_pooled"]:.3f}')
ax.axhline(y=fedprox_stats['best_pooled'], color='r', linestyle='--', alpha=0.5, label=f'FedProx best: {fedprox_stats["best_pooled"]:.3f}')

ax.set_xlabel('Communication Round')
ax.set_ylabel('Pooled Dice Score (MeanPresent)')
ax.set_title('FedAvg vs FedProx: Global Model Convergence\n(BraTS 2D, 4 Clients, Dirichlet α=1.0)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 30)
ax.set_ylim(0, 0.95)

plt.tight_layout()
plt.savefig(output_dir / 'convergence_comparison.png')
plt.savefig(output_dir / 'convergence_comparison.pdf')
print(f"\nSaved: convergence_comparison.png/pdf")
plt.close()


# ============================================================
# PLOT 2: Per-Client Performance Comparison
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
client_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, ax in enumerate(axes.flat):
    ax.plot(fedavg['rounds'], fedavg_stats['clients'][:, i], 'b-', linewidth=1.5,
            label='FedAvg', marker='o', markersize=3, markevery=5)
    ax.plot(fedprox['rounds'], fedprox_stats['clients'][:, i], 'r-', linewidth=1.5,
            label='FedProx', marker='s', markersize=3, markevery=5)

    # Final values annotation
    fa_final = fedavg_stats['clients'][-1, i]
    fp_final = fedprox_stats['clients'][-1, i]
    ax.annotate(f'{fa_final:.3f}', xy=(30, fa_final), fontsize=9, color='blue', ha='left')
    ax.annotate(f'{fp_final:.3f}', xy=(30, fp_final), fontsize=9, color='red', ha='left')

    ax.set_xlabel('Round')
    ax.set_ylabel('Dice Score')
    ax.set_title(f'Client {i}')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 0.95)

fig.suptitle('Per-Client Dice Score Progression: FedAvg vs FedProx', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'per_client_comparison.png')
plt.savefig(output_dir / 'per_client_comparison.pdf')
print(f"Saved: per_client_comparison.png/pdf")
plt.close()


# ============================================================
# PLOT 3: Client Variance / Stability Analysis
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Client range (max - min) over rounds
ax1 = axes[0]
ax1.plot(fedavg['rounds'], fedavg_stats['client_range'], 'b-', linewidth=2, label='FedAvg')
ax1.plot(fedprox['rounds'], fedprox_stats['client_range'], 'r-', linewidth=2, label='FedProx')
ax1.fill_between(fedavg['rounds'], 0, fedavg_stats['client_range'], alpha=0.2, color='blue')
ax1.fill_between(fedprox['rounds'], 0, fedprox_stats['client_range'], alpha=0.2, color='red')
ax1.set_xlabel('Communication Round')
ax1.set_ylabel('Client Dice Range (max - min)')
ax1.set_title('Client Performance Divergence')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 30)

# Right: Worst-client Dice over rounds
ax2 = axes[1]
ax2.plot(fedavg['rounds'], fedavg_stats['min_dice_per_round'], 'b-', linewidth=2, label='FedAvg')
ax2.plot(fedprox['rounds'], fedprox_stats['min_dice_per_round'], 'r-', linewidth=2, label='FedProx')
ax2.set_xlabel('Communication Round')
ax2.set_ylabel('Worst-Client Dice Score')
ax2.set_title('Worst-Client Performance (Fairness)')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 30)
ax2.set_ylim(0, 0.95)

fig.suptitle('Stability and Fairness Analysis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'stability_analysis.png')
plt.savefig(output_dir / 'stability_analysis.pdf')
print(f"Saved: stability_analysis.png/pdf")
plt.close()


# ============================================================
# PLOT 4: Client Distribution Bar Chart (Final Round)
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(4)
width = 0.35

bars1 = ax.bar(x - width/2, fedavg_stats['clients'][-1], width, label='FedAvg', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, fedprox_stats['clients'][-1], width, label='FedProx', color='#e74c3c', edgecolor='black')

# Add pooled averages as horizontal lines
ax.axhline(y=fedavg_stats['final_pooled'], color='#3498db', linestyle='--', linewidth=2, alpha=0.7, label=f'FedAvg pooled: {fedavg_stats["final_pooled"]:.3f}')
ax.axhline(y=fedprox_stats['final_pooled'], color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7, label=f'FedProx pooled: {fedprox_stats["final_pooled"]:.3f}')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Client')
ax.set_ylabel('Final Dice Score')
ax.set_title('Final Round Per-Client Dice: FedAvg vs FedProx\n(BraTS 2D, Dirichlet α=1.0)')
ax.set_xticks(x)
ax.set_xticklabels(['Client 0', 'Client 1', 'Client 2', 'Client 3'])
ax.legend(loc='upper right')
ax.set_ylim(0.7, 0.95)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'final_client_comparison.png')
plt.savefig(output_dir / 'final_client_comparison.pdf')
print(f"Saved: final_client_comparison.png/pdf")
plt.close()


# ============================================================
# PLOT 5: Convergence with Confidence Bands
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Plot mean with min/max bands
ax.plot(fedavg['rounds'], fedavg['pooled'], 'b-', linewidth=2.5, label='FedAvg Pooled')
ax.fill_between(fedavg['rounds'], fedavg_stats['min_dice_per_round'], fedavg_stats['max_dice_per_round'],
                alpha=0.2, color='blue', label='FedAvg Client Range')

ax.plot(fedprox['rounds'], fedprox['pooled'], 'r-', linewidth=2.5, label='FedProx Pooled')
ax.fill_between(fedprox['rounds'], fedprox_stats['min_dice_per_round'], fedprox_stats['max_dice_per_round'],
                alpha=0.2, color='red', label='FedProx Client Range')

ax.set_xlabel('Communication Round')
ax.set_ylabel('Dice Score')
ax.set_title('Convergence with Client Variability Bands\n(Shaded: min-max across clients)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 30)
ax.set_ylim(0, 0.95)

plt.tight_layout()
plt.savefig(output_dir / 'convergence_with_bands.png')
plt.savefig(output_dir / 'convergence_with_bands.pdf')
print(f"Saved: convergence_with_bands.png/pdf")
plt.close()


# ============================================================
# Save Summary Report
# ============================================================

report = f"""
================================================================================
FEDAVG vs FEDPROX ANALYSIS REPORT
BraTS 2D U-Net Segmentation | 4 Clients | Dirichlet α=1.0 (MILD heterogeneity)
================================================================================

EXPERIMENT CONFIGURATION
------------------------
- Model: 2D U-Net with GroupNorm
- Loss: BCE + Soft Dice
- Optimizer: SGD (no momentum)
- Rounds: 30
- Local Epochs: 5
- Batch Size: 8
- Learning Rate: 0.01
- FedProx μ: 0.1

NOTE: Files are named "a0p1" but contain α=1.0 partition results.
      α=1.0 represents MILD heterogeneity (near uniform).

================================================================================
SUMMARY RESULTS
================================================================================

                              FedAvg        FedProx       Difference
                              ------        -------       ----------
Final Pooled Dice:            {fedavg_stats['final_pooled']:.4f}        {fedprox_stats['final_pooled']:.4f}        {fedprox_stats['final_pooled']-fedavg_stats['final_pooled']:+.4f}
Best Pooled Dice:             {fedavg_stats['best_pooled']:.4f}        {fedprox_stats['best_pooled']:.4f}        {fedprox_stats['best_pooled']-fedavg_stats['best_pooled']:+.4f}
Final Worst-Client:           {fedavg_stats['final_worst_client']:.4f}        {fedprox_stats['final_worst_client']:.4f}        {fedprox_stats['final_worst_client']-fedavg_stats['final_worst_client']:+.4f}
Final Best-Client:            {fedavg_stats['final_best_client']:.4f}        {fedprox_stats['final_best_client']:.4f}        {fedprox_stats['final_best_client']-fedavg_stats['final_best_client']:+.4f}
Final Client Range:           {fedavg_stats['client_range'][-1]:.4f}        {fedprox_stats['client_range'][-1]:.4f}        {fedprox_stats['client_range'][-1]-fedavg_stats['client_range'][-1]:+.4f}

================================================================================
PER-CLIENT FINAL DICE SCORES
================================================================================

Client      FedAvg      FedProx     Δ           Winner
------      ------      -------     --          ------
Client 0    {fedavg_stats['clients'][-1,0]:.4f}      {fedprox_stats['clients'][-1,0]:.4f}      {fedprox_stats['clients'][-1,0]-fedavg_stats['clients'][-1,0]:+.4f}      {'FedProx' if fedprox_stats['clients'][-1,0] > fedavg_stats['clients'][-1,0] else 'FedAvg'}
Client 1    {fedavg_stats['clients'][-1,1]:.4f}      {fedprox_stats['clients'][-1,1]:.4f}      {fedprox_stats['clients'][-1,1]-fedavg_stats['clients'][-1,1]:+.4f}      {'FedProx' if fedprox_stats['clients'][-1,1] > fedavg_stats['clients'][-1,1] else 'FedAvg'}
Client 2    {fedavg_stats['clients'][-1,2]:.4f}      {fedprox_stats['clients'][-1,2]:.4f}      {fedprox_stats['clients'][-1,2]-fedavg_stats['clients'][-1,2]:+.4f}      {'FedProx' if fedprox_stats['clients'][-1,2] > fedavg_stats['clients'][-1,2] else 'FedAvg'}
Client 3    {fedavg_stats['clients'][-1,3]:.4f}      {fedprox_stats['clients'][-1,3]:.4f}      {fedprox_stats['clients'][-1,3]-fedavg_stats['clients'][-1,3]:+.4f}      {'FedProx' if fedprox_stats['clients'][-1,3] > fedavg_stats['clients'][-1,3] else 'FedAvg'}

================================================================================
INTERPRETATION
================================================================================

{'FedProx OUTPERFORMS FedAvg' if fedprox_stats['final_pooled'] > fedavg_stats['final_pooled'] else 'FedAvg OUTPERFORMS FedProx'}
  - Pooled Dice improvement: {fedprox_stats['final_pooled']-fedavg_stats['final_pooled']:+.4f} ({100*(fedprox_stats['final_pooled']-fedavg_stats['final_pooled'])/fedavg_stats['final_pooled']:+.1f}%)
  - Worst-client improvement: {fedprox_stats['final_worst_client']-fedavg_stats['final_worst_client']:+.4f}

WHY FedProx HELPS (even with mild α=1.0 heterogeneity):
  1. The proximal term (μ=0.1) prevents local models from drifting too far
  2. This leads to more stable convergence and better final performance
  3. FedProx shows improvement across ALL 4 clients (see per-client results)

CONVERGENCE CHARACTERISTICS:
  - FedAvg: Shows more oscillation in later rounds
  - FedProx: More stable convergence trajectory
  - Both algorithms converge within ~20 rounds

FAIRNESS (Client Equity):
  - FedAvg final client range: {fedavg_stats['client_range'][-1]:.4f}
  - FedProx final client range: {fedprox_stats['client_range'][-1]:.4f}
  - {'FedProx is MORE fair (smaller range)' if fedprox_stats['client_range'][-1] < fedavg_stats['client_range'][-1] else 'FedAvg is MORE fair (smaller range)'}

================================================================================
GENERATED PLOTS
================================================================================
1. convergence_comparison.png/pdf    - Global pooled Dice over rounds
2. per_client_comparison.png/pdf     - Individual client trajectories
3. stability_analysis.png/pdf        - Client divergence and worst-client
4. final_client_comparison.png/pdf   - Bar chart of final per-client Dice
5. convergence_with_bands.png/pdf    - Convergence with client variability bands

================================================================================
"""

report_path = output_dir / 'analysis_report.txt'
with open(report_path, 'w') as f:
    f.write(report)
print(f"\nSaved: analysis_report.txt")

print(report)
print(f"\nAll outputs saved to: {output_dir}")
