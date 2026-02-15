#!/usr/bin/env python3
"""Plot 2-client mu sweep: FedAvg vs FedProx mu=0.2, 0.3, 0.5."""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def parse_log(path):
    """Extract per-round metrics from a BraTS log file."""
    rounds, pooled_mean, pooled_loss, wt, tc, et = [], [], [], [], [], []
    client_means = {i: [] for i in range(2)}
    worst_client = []

    pattern = (
        r"Round\s+(\d+)/\d+:\s+Pooled Loss=([\d.]+),\s+Pooled Mean=([\d.]+)\s+"
        r"\(WT=([\d.]+)\s+TC=([\d.]+)\s+ET=([\d.]+)\)"
        r"(.*?)WorstC Mean=([\d.]+)"
    )
    client_pattern = r"C(\d) Mean=([\d.]+)"

    with open(path) as f:
        for line in f:
            m = re.search(pattern, line)
            if m:
                rounds.append(int(m.group(1)))
                pooled_loss.append(float(m.group(2)))
                pooled_mean.append(float(m.group(3)))
                wt.append(float(m.group(4)))
                tc.append(float(m.group(5)))
                et.append(float(m.group(6)))
                worst_client.append(float(m.group(8)))
                rest = m.group(7)
                for cm in re.finditer(client_pattern, rest):
                    cid = int(cm.group(1))
                    client_means[cid].append(float(cm.group(2)))

    return {
        "rounds": np.array(rounds),
        "pooled_mean": np.array(pooled_mean),
        "pooled_loss": np.array(pooled_loss),
        "wt": np.array(wt),
        "tc": np.array(tc),
        "et": np.array(et),
        "worst_client": np.array(worst_client),
        "client_means": {k: np.array(v) for k, v in client_means.items()},
    }


LOG_DIR = "/Users/basiakoch/Federated/federated-thesis/experiments/brats/logs"
fedavg  = parse_log(f"{LOG_DIR}/brats_fed_N_22536464.out")
prox02  = parse_log(f"{LOG_DIR}/brats_fed_N_22536467.out")
prox03  = parse_log(f"{LOG_DIR}/brats_fed_N_22543504.out")
prox05  = parse_log(f"{LOG_DIR}/brats_fed_N_22541274.out")

runs = [
    (fedavg, "#2196F3", "FedAvg"),
    (prox02, "#F44336", r"FedProx $\mu$=0.2"),
    (prox03, "#4CAF50", r"FedProx $\mu$=0.3"),
    (prox05, "#FF9800", r"FedProx $\mu$=0.5"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "2-Client Mu Sweep  —  Equal Data, Corruption Heterogeneity, E=10, LR=0.01",
    fontsize=13, fontweight="bold",
)

# --- 1. Global Pooled Mean Dice ---
ax = axes[0, 0]
for data, clr, lbl in runs:
    ax.plot(data["rounds"], data["pooled_mean"], color=clr, linewidth=1.5, label=lbl)
    ax.axhline(max(data["pooled_mean"]), color=clr, linestyle="--", alpha=0.3, linewidth=0.7)
# Annotate FedAvg crash
crash_idx = np.argmin(fedavg["pooled_mean"][20:]) + 20
ax.annotate(
    f'R{fedavg["rounds"][crash_idx]} crash\n{fedavg["pooled_mean"][crash_idx]:.3f}',
    xy=(fedavg["rounds"][crash_idx], fedavg["pooled_mean"][crash_idx]),
    xytext=(fedavg["rounds"][crash_idx] - 10, fedavg["pooled_mean"][crash_idx] + 0.07),
    arrowprops=dict(arrowstyle="->", color="#2196F3", lw=1.2),
    fontsize=8, color="#2196F3", fontweight="bold",
)
ax.set_title("Global Pooled Mean Dice")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend(fontsize=8)
ax.set_ylim(0.35, 0.85)
ax.grid(True, alpha=0.3)

# --- 2. Client 0 (clean) ---
ax = axes[0, 1]
for data, clr, lbl in runs:
    ax.plot(data["rounds"], data["client_means"][0], color=clr, linewidth=1.5, label=lbl)
ax.set_title("Client 0 (Clean) — Mean Dice")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend(fontsize=8)
ax.set_ylim(0.3, 0.85)
ax.grid(True, alpha=0.3)

# --- 3. Client 1 (corrupted) ---
ax = axes[0, 2]
for data, clr, lbl in runs:
    ax.plot(data["rounds"], data["client_means"][1], color=clr, linewidth=1.5, label=lbl)
ax.set_title("Client 1 (Corrupted) — Mean Dice")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend(fontsize=8)
ax.set_ylim(0.1, 0.65)
ax.grid(True, alpha=0.3)

# --- 4. Rolling volatility ---
ax = axes[1, 0]
window = 5
for data, clr, lbl in runs:
    rolling_std = np.array([data["pooled_mean"][max(0, i-window):i+1].std()
                            for i in range(len(data["pooled_mean"]))])
    ax.plot(data["rounds"], rolling_std, color=clr, linewidth=1.5, label=lbl)
ax.set_title(f"Rolling Volatility (std, window={window})")
ax.set_xlabel("Round")
ax.set_ylabel("Std of Pooled Mean Dice")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- 5. Mu vs Best/Final Global Dice (bar chart) ---
ax = axes[1, 1]
mu_labels = ["FedAvg\n(0.0)", r"$\mu$=0.2", r"$\mu$=0.3", r"$\mu$=0.5"]
colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
best_vals = [max(d["pooled_mean"]) for d, _, _ in runs]
final_vals = [d["pooled_mean"][-1] for d, _, _ in runs]

x = np.arange(len(mu_labels))
w = 0.3
bars1 = ax.bar(x - w/2, best_vals, w, color=colors, alpha=0.9, label="Best", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + w/2, final_vals, w, color=colors, alpha=0.5, label="Final", edgecolor="black", linewidth=0.5)
ax.set_title(r"Global Dice vs $\mu$")
ax.set_xlabel(r"$\mu$ value")
ax.set_ylabel("Dice Score")
ax.set_xticks(x)
ax.set_xticklabels(mu_labels, fontsize=9)
ax.legend(fontsize=9)
ax.set_ylim(0.65, 0.85)
ax.grid(True, alpha=0.3, axis="y")
# Annotate best values
for i, (bv, fv) in enumerate(zip(best_vals, final_vals)):
    ax.text(x[i] - w/2, bv + 0.003, f"{bv:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.text(x[i] + w/2, fv + 0.003, f"{fv:.3f}", ha="center", va="bottom", fontsize=7)

# --- 6. Mu vs Client 1 Best/Final (fairness) ---
ax = axes[1, 2]
c1_best = [max(d["client_means"][1]) for d, _, _ in runs]
c1_final = [d["client_means"][1][-1] for d, _, _ in runs]

bars1 = ax.bar(x - w/2, c1_best, w, color=colors, alpha=0.9, label="Best", edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + w/2, c1_final, w, color=colors, alpha=0.5, label="Final", edgecolor="black", linewidth=0.5)
ax.set_title(r"Client 1 (Corrupted) Dice vs $\mu$")
ax.set_xlabel(r"$\mu$ value")
ax.set_ylabel("Dice Score")
ax.set_xticks(x)
ax.set_xticklabels(mu_labels, fontsize=9)
ax.legend(fontsize=9)
ax.set_ylim(0.3, 0.65)
ax.grid(True, alpha=0.3, axis="y")
for i, (bv, fv) in enumerate(zip(c1_best, c1_final)):
    ax.text(x[i] - w/2, bv + 0.003, f"{bv:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.text(x[i] + w/2, fv + 0.003, f"{fv:.3f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
out_path = "/Users/basiakoch/Federated/federated-thesis/experiments/brats/2client_mu_sweep.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved: {out_path}")
print(f"Saved: {out_path.replace('.png', '.pdf')}")
plt.close()

# --- Summary table ---
print("\n=== 2-Client Mu Sweep Summary ===")
header = f"{'Metric':<25} {'FedAvg':>10} {'mu=0.2':>10} {'mu=0.3':>10} {'mu=0.5':>10}"
print(header)
print("-" * len(header))
all_data = [fedavg, prox02, prox03, prox05]
print(f"{'Best Pooled Mean':<25} " + " ".join(f"{max(d['pooled_mean']):>10.4f}" for d in all_data))
print(f"{'Final Pooled Mean':<25} " + " ".join(f"{d['pooled_mean'][-1]:>10.4f}" for d in all_data))
print(f"{'Best WT':<25} " + " ".join(f"{max(d['wt']):>10.4f}" for d in all_data))
print(f"{'Best TC':<25} " + " ".join(f"{max(d['tc']):>10.4f}" for d in all_data))
print(f"{'Best ET':<25} " + " ".join(f"{max(d['et']):>10.4f}" for d in all_data))
print(f"{'Client 0 best':<25} " + " ".join(f"{max(d['client_means'][0]):>10.4f}" for d in all_data))
print(f"{'Client 1 best':<25} " + " ".join(f"{max(d['client_means'][1]):>10.4f}" for d in all_data))
print(f"{'Client 1 final':<25} " + " ".join(f"{d['client_means'][1][-1]:>10.4f}" for d in all_data))
print(f"{'Best-Final gap':<25} " + " ".join(f"{max(d['pooled_mean'])-d['pooled_mean'][-1]:>10.4f}" for d in all_data))
print(f"{'Late std (R30-50)':<25} " + " ".join(f"{d['pooled_mean'][29:].std():>10.4f}" for d in all_data))
