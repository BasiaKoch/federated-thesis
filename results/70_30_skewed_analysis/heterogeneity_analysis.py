#!/usr/bin/env python3
"""
Analysis of heterogeneity options for federated learning experiments.
Explores different partition strategies to increase heterogeneity.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Load client map
client_map_path = Path(__file__).parent.parent.parent / "data/partitions/brats2d_7030_et_skewed/client_map.json"
with open(client_map_path) as f:
    client_map = json.load(f)

et_ratios = client_map['et_ratios']
all_cases = list(et_ratios.keys())
all_ratios = np.array(list(et_ratios.values()))

print("=" * 70)
print("HETEROGENEITY ANALYSIS FOR FEDERATED LEARNING EXPERIMENTS")
print("=" * 70)

print(f"\n{'='*70}")
print("1. CURRENT DATASET STATISTICS")
print("=" * 70)
print(f"Total cases: {len(all_cases)}")
print(f"ET ratio range: [{all_ratios.min():.3f}, {all_ratios.max():.3f}]")
print(f"ET ratio mean: {all_ratios.mean():.3f}")
print(f"ET ratio std: {all_ratios.std():.3f}")
print(f"Cases with ET=0: {np.sum(all_ratios < 0.001)}")
print(f"Cases with ET>0.5: {np.sum(all_ratios > 0.5)}")

# Quartile analysis
q1, q2, q3, q4 = np.percentile(all_ratios, [25, 50, 75, 100])
print(f"\nQuartiles: Q1={q1:.3f}, Q2(median)={q2:.3f}, Q3={q3:.3f}, Q4(max)={q4:.3f}")

# Count cases per quartile
n_total = len(all_ratios)
n_q1 = np.sum(all_ratios <= np.percentile(all_ratios, 25))
n_q2 = np.sum((all_ratios > np.percentile(all_ratios, 25)) & (all_ratios <= np.percentile(all_ratios, 50)))
n_q3 = np.sum((all_ratios > np.percentile(all_ratios, 50)) & (all_ratios <= np.percentile(all_ratios, 75)))
n_q4 = np.sum(all_ratios > np.percentile(all_ratios, 75))
print(f"Cases per quartile: Q1={n_q1}, Q2={n_q2}, Q3={n_q3}, Q4={n_q4}")


print(f"\n{'='*70}")
print("2. CURRENT 2-CLIENT HETEROGENEITY")
print("=" * 70)
c0_stats = client_map['metadata']['client_0_stats']
c1_stats = client_map['metadata']['client_1_stats']
print(f"Client 0: {c0_stats['n_cases']} cases, ET mean={c0_stats['et_ratio_mean']:.3f}")
print(f"Client 1: {c1_stats['n_cases']} cases, ET mean={c1_stats['et_ratio_mean']:.3f}")
print(f"ET ratio gap: {c1_stats['et_ratio_mean'] / c0_stats['et_ratio_mean']:.1f}x")
print(f"Current heterogeneity level: MODERATE (3x difference)")


print(f"\n{'='*70}")
print("3. OPTION A: MORE EXTREME 2-CLIENT SKEW")
print("=" * 70)
print("Strategy: Give Client 0 only Q1+Q2, Client 1 only Q3+Q4 (no overlap)")

sorted_cases = sorted(et_ratios.keys(), key=lambda c: et_ratios[c])
n = len(sorted_cases)
extreme_c0 = sorted_cases[:n//2]  # Bottom 50%
extreme_c1 = sorted_cases[n//2:]  # Top 50%

c0_extreme_mean = np.mean([et_ratios[c] for c in extreme_c0])
c1_extreme_mean = np.mean([et_ratios[c] for c in extreme_c1])

print(f"Client 0: {len(extreme_c0)} cases, ET mean={c0_extreme_mean:.3f}")
print(f"Client 1: {len(extreme_c1)} cases, ET mean={c1_extreme_mean:.3f}")
print(f"ET ratio gap: {c1_extreme_mean / c0_extreme_mean:.1f}x")
print(f"Heterogeneity level: HIGH (5x difference, no overlap)")
print("\nPros: Maximum label skew between clients")
print("Cons: Complete distribution shift - may be too extreme")


print(f"\n{'='*70}")
print("4. OPTION B: 3-CLIENT PARTITION (Low/Medium/High ET)")
print("=" * 70)
print("Strategy: Divide by terciles - each client sees different ET distribution")

tercile_1 = np.percentile(all_ratios, 33.3)
tercile_2 = np.percentile(all_ratios, 66.6)

c0_3way = [c for c in all_cases if et_ratios[c] <= tercile_1]
c1_3way = [c for c in all_cases if tercile_1 < et_ratios[c] <= tercile_2]
c2_3way = [c for c in all_cases if et_ratios[c] > tercile_2]

print(f"Client 0 (Low-ET):    {len(c0_3way)} cases, ET mean={np.mean([et_ratios[c] for c in c0_3way]):.3f}")
print(f"Client 1 (Medium-ET): {len(c1_3way)} cases, ET mean={np.mean([et_ratios[c] for c in c1_3way]):.3f}")
print(f"Client 2 (High-ET):   {len(c2_3way)} cases, ET mean={np.mean([et_ratios[c] for c in c2_3way]):.3f}")
print(f"\nMin cases per client (train, 80%): ~{int(min(len(c0_3way), len(c1_3way), len(c2_3way)) * 0.8)}")
print("\nPros: More realistic multi-site scenario")
print("Cons: Fewer cases per client - may underfit")


print(f"\n{'='*70}")
print("5. OPTION C: 4-CLIENT PARTITION (Quartiles)")
print("=" * 70)
print("Strategy: Each quartile becomes a separate client")

q_bounds = [0] + list(np.percentile(all_ratios, [25, 50, 75])) + [1.0]
clients_4way = []
for i in range(4):
    if i == 0:
        cases = [c for c in all_cases if et_ratios[c] <= q_bounds[i+1]]
    elif i == 3:
        cases = [c for c in all_cases if et_ratios[c] > q_bounds[i]]
    else:
        cases = [c for c in all_cases if q_bounds[i] < et_ratios[c] <= q_bounds[i+1]]
    clients_4way.append(cases)
    mean_et = np.mean([et_ratios[c] for c in cases]) if cases else 0
    print(f"Client {i} (Q{i+1}): {len(cases)} cases, ET mean={mean_et:.3f}")

print(f"\nMin cases per client (train, 80%): ~{int(min(len(c) for c in clients_4way) * 0.8)}")
print("\nPros: Maximum gradient diversity, each client specialized")
print("Cons: Only ~59 train cases per client - risk of overfitting")


print(f"\n{'='*70}")
print("6. OPTION D: 5-CLIENT PARTITION (Quintiles)")
print("=" * 70)
print("Strategy: Each quintile becomes a separate client")

q_bounds = [0] + list(np.percentile(all_ratios, [20, 40, 60, 80])) + [1.0]
clients_5way = []
for i in range(5):
    if i == 0:
        cases = [c for c in all_cases if et_ratios[c] <= q_bounds[i+1]]
    elif i == 4:
        cases = [c for c in all_cases if et_ratios[c] > q_bounds[i]]
    else:
        cases = [c for c in all_cases if q_bounds[i] < et_ratios[c] <= q_bounds[i+1]]
    clients_5way.append(cases)
    mean_et = np.mean([et_ratios[c] for c in cases]) if cases else 0
    print(f"Client {i} (Q{i+1}): {len(cases)} cases, ET mean={mean_et:.3f}")

print(f"\nMin cases per client (train, 80%): ~{int(min(len(c) for c in clients_5way) * 0.8)}")
print("\nPros: High heterogeneity, more clients = more aggregation benefit")
print("Cons: Only ~47 train cases per client - significant overfitting risk")


print(f"\n{'='*70}")
print("7. OPTION E: QUANTITY SKEW (Imbalanced Data Amounts)")
print("=" * 70)
print("Strategy: Keep 2 clients but with extreme quantity imbalance")

# 90/10 split
n_c0_90 = int(0.9 * n)
n_c1_10 = n - n_c0_90

print(f"90/10 split:")
print(f"  Client 0: {n_c0_90} cases (90%)")
print(f"  Client 1: {n_c1_10} cases (10%)")
print(f"  Client 1 train: ~{int(n_c1_10 * 0.8)} cases")
print("\nCombine with ET skew: Client 0 gets 90% (mostly low-ET), Client 1 gets 10% (all high-ET)")
print("This creates BOTH quantity skew AND label skew")


print(f"\n{'='*70}")
print("8. OPTION F: DIRICHLET PARTITION (Standard FL Benchmark)")
print("=" * 70)
print("Strategy: Use Dirichlet distribution to create non-IID splits")
print("\nDirichlet(α) where:")
print("  α = 0.1: Extreme heterogeneity (each client sees ~1-2 dominant classes)")
print("  α = 0.5: High heterogeneity")
print("  α = 1.0: Moderate heterogeneity")
print("  α = 10.0: Nearly IID")
print("\nFor BraTS with 3 classes (WT, TC, ET), Dirichlet would skew class proportions")


print(f"\n{'='*70}")
print("9. RECOMMENDATIONS")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    RECOMMENDED EXPERIMENTS                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  EXPERIMENT 1: Extreme 2-Client (Baseline comparison)               │
│  ─────────────────────────────────────────────────────              │
│  • Client 0: Q1+Q2 only (bottom 50% ET)                            │
│  • Client 1: Q3+Q4 only (top 50% ET)                               │
│  • 5x ET ratio gap (vs current 3x)                                 │
│  • Expected: FedProx advantage increases                           │
│                                                                     │
│  EXPERIMENT 2: 4-Client Quartile Split (More clients)              │
│  ─────────────────────────────────────────────────────              │
│  • 4 clients, each with ~74 cases                                  │
│  • Maximum gradient diversity                                       │
│  • Expected: FedProx shows clearer advantage                       │
│  • Caveat: May need to reduce local epochs due to fewer samples    │
│                                                                     │
│  EXPERIMENT 3: 3-Client with Quantity Skew                         │
│  ─────────────────────────────────────────────────────              │
│  • Client 0: 60% data (low-ET) - large hospital                    │
│  • Client 1: 30% data (medium-ET)                                  │
│  • Client 2: 10% data (high-ET) - specialized center               │
│  • Combines label + quantity heterogeneity                         │
│  • Most realistic clinical scenario                                │
│                                                                     │
│  EXPERIMENT 4: 2-Client Extreme + More Local Epochs                │
│  ─────────────────────────────────────────────────────              │
│  • Keep 2 clients but increase local_epochs: 5 → 10                │
│  • More local training = more client drift                         │
│  • FedProx's proximal term becomes more important                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

print(f"\n{'='*70}")
print("10. LITERATURE-BASED CLIENT NUMBER RECOMMENDATIONS")
print("=" * 70)

print("""
Research shows FedProx advantages are clearest when:

1. NUMBER OF CLIENTS: 4-10+ clients
   • 2 clients: FedAvg can often "average out" differences
   • 4+ clients: More diverse gradients, harder to find common optimum
   • Your dataset (295 cases) supports up to 4-5 clients well

2. LOCAL EPOCHS: 5-20 epochs
   • More local epochs = more client drift = bigger FedProx advantage
   • Current: 5 epochs (moderate)
   • Suggestion: Try 10-15 epochs

3. HETEROGENEITY LEVEL: High non-IID
   • Your current 3x ET gap is "moderate"
   • 5x+ gap or completely disjoint distributions show clearer differences

4. PARTICIPATION RATE: Partial participation
   • If only subset of clients participate each round, FedProx helps more
   • Not applicable to your 2-client setup

MINIMUM VIABLE EXPERIMENT for clear FedProx advantage:
  • 4 clients (quartile split)
  • 10 local epochs
  • 30+ communication rounds
  • ~60 train cases per client (feasible with your 295 cases)
""")


# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Current distribution
ax = axes[0, 0]
ax.hist(all_ratios, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
ax.axvline(np.median(all_ratios), color='red', linestyle='--', label=f'Median: {np.median(all_ratios):.3f}')
ax.set_xlabel('ET Ratio')
ax.set_ylabel('Count')
ax.set_title('Full Dataset ET Distribution')
ax.legend()

# Current 2-client
ax = axes[0, 1]
c0_current = [et_ratios[c] for c in client_map['client_0']]
c1_current = [et_ratios[c] for c in client_map['client_1']]
ax.hist(c0_current, bins=20, alpha=0.6, label=f'C0 (n={len(c0_current)})', color='purple')
ax.hist(c1_current, bins=20, alpha=0.6, label=f'C1 (n={len(c1_current)})', color='orange')
ax.set_xlabel('ET Ratio')
ax.set_ylabel('Count')
ax.set_title('Current 2-Client Split (3x gap)')
ax.legend()

# Extreme 2-client
ax = axes[0, 2]
c0_extreme = [et_ratios[c] for c in extreme_c0]
c1_extreme = [et_ratios[c] for c in extreme_c1]
ax.hist(c0_extreme, bins=20, alpha=0.6, label=f'C0 (n={len(c0_extreme)})', color='purple')
ax.hist(c1_extreme, bins=20, alpha=0.6, label=f'C1 (n={len(c1_extreme)})', color='orange')
ax.set_xlabel('ET Ratio')
ax.set_ylabel('Count')
ax.set_title('Option A: Extreme 2-Client (5x gap)')
ax.legend()

# 3-client
ax = axes[1, 0]
colors = ['purple', 'green', 'orange']
for i, (cases, label) in enumerate([(c0_3way, 'C0 Low'), (c1_3way, 'C1 Med'), (c2_3way, 'C2 High')]):
    ax.hist([et_ratios[c] for c in cases], bins=15, alpha=0.6, label=f'{label} (n={len(cases)})', color=colors[i])
ax.set_xlabel('ET Ratio')
ax.set_ylabel('Count')
ax.set_title('Option B: 3-Client Tercile Split')
ax.legend()

# 4-client
ax = axes[1, 1]
colors = ['purple', 'blue', 'green', 'orange']
for i, cases in enumerate(clients_4way):
    ax.hist([et_ratios[c] for c in cases], bins=12, alpha=0.6, label=f'C{i} Q{i+1} (n={len(cases)})', color=colors[i])
ax.set_xlabel('ET Ratio')
ax.set_ylabel('Count')
ax.set_title('Option C: 4-Client Quartile Split')
ax.legend()

# Summary comparison
ax = axes[1, 2]
scenarios = ['Current\n2-client', 'Extreme\n2-client', '3-client', '4-client', '5-client']
et_gaps = [
    c1_stats['et_ratio_mean'] / c0_stats['et_ratio_mean'],
    c1_extreme_mean / c0_extreme_mean,
    np.mean([et_ratios[c] for c in c2_3way]) / np.mean([et_ratios[c] for c in c0_3way]),
    np.mean([et_ratios[c] for c in clients_4way[3]]) / np.mean([et_ratios[c] for c in clients_4way[0]]),
    np.mean([et_ratios[c] for c in clients_5way[4]]) / np.mean([et_ratios[c] for c in clients_5way[0]])
]
min_cases = [
    min(len(client_map['client_0']), len(client_map['client_1'])),
    min(len(extreme_c0), len(extreme_c1)),
    min(len(c0_3way), len(c1_3way), len(c2_3way)),
    min(len(c) for c in clients_4way),
    min(len(c) for c in clients_5way)
]

ax2 = ax.twinx()
bars1 = ax.bar(np.arange(len(scenarios)) - 0.2, et_gaps, 0.4, color='steelblue', alpha=0.7, label='ET Gap (x)')
bars2 = ax2.bar(np.arange(len(scenarios)) + 0.2, min_cases, 0.4, color='coral', alpha=0.7, label='Min Cases')
ax.set_xticks(np.arange(len(scenarios)))
ax.set_xticklabels(scenarios)
ax.set_ylabel('Max ET Ratio Gap (x)', color='steelblue')
ax2.set_ylabel('Min Cases per Client', color='coral')
ax.set_title('Heterogeneity vs Data Availability Tradeoff')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
output_dir = Path(__file__).parent
plt.savefig(output_dir / 'heterogeneity_options.png', dpi=150)
plt.savefig(output_dir / 'heterogeneity_options.pdf')
print(f"\nSaved: heterogeneity_options.png/pdf")

plt.close()
