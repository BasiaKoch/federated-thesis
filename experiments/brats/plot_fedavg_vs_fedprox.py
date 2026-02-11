#!/usr/bin/env python3
"""Plot FedAvg vs FedProx (mu=0.4) from BraTS log files."""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def parse_log(path):
    """Extract per-round metrics from a BraTS log file."""
    rounds, pooled_mean, wt, tc, et, worst_client = [], [], [], [], [], []
    client_means = {i: [] for i in range(8)}

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
        "wt": np.array(wt),
        "tc": np.array(tc),
        "et": np.array(et),
        "worst_client": np.array(worst_client),
        "client_means": {k: np.array(v) for k, v in client_means.items()},
    }


# --- Parse both logs ---
fedavg = parse_log("/Users/basiakoch/Federated/federated-thesis/experiments/brats/logs/brats_fed_N_22491760.out")
fedprox = parse_log("/Users/basiakoch/Federated/federated-thesis/experiments/brats/logs/brats_fed_N_22491847.out")

# --- Colours ---
C_AVG = "#2196F3"   # blue
C_PROX = "#F44336"  # red

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("FedAvg vs FedProx ($\\mu$=0.4)  —  BraTS 2D, 8 clients, E=30, LR=0.01", fontsize=14, fontweight="bold")

# --- 1. Global Pooled Mean Dice ---
ax = axes[0, 0]
ax.plot(fedavg["rounds"], fedavg["pooled_mean"], color=C_AVG, linewidth=1.5, label="FedAvg")
ax.plot(fedprox["rounds"], fedprox["pooled_mean"], color=C_PROX, linewidth=1.5, label="FedProx ($\\mu$=0.4)")
ax.axhline(max(fedavg["pooled_mean"]), color=C_AVG, linestyle="--", alpha=0.4, linewidth=0.8)
ax.axhline(max(fedprox["pooled_mean"]), color=C_PROX, linestyle="--", alpha=0.4, linewidth=0.8)
ax.set_title("Global Pooled Mean Dice")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend()
ax.set_ylim(0.6, 0.85)
ax.grid(True, alpha=0.3)

# --- 2. Per-class Dice (WT, TC, ET) ---
ax = axes[0, 1]
ax.plot(fedavg["rounds"], fedavg["wt"], color=C_AVG, linewidth=1.2, label="FedAvg WT")
ax.plot(fedavg["rounds"], fedavg["tc"], color=C_AVG, linewidth=1.2, linestyle="--", label="FedAvg TC")
ax.plot(fedavg["rounds"], fedavg["et"], color=C_AVG, linewidth=1.2, linestyle=":", label="FedAvg ET")
ax.plot(fedprox["rounds"], fedprox["wt"], color=C_PROX, linewidth=1.2, label="FedProx WT")
ax.plot(fedprox["rounds"], fedprox["tc"], color=C_PROX, linewidth=1.2, linestyle="--", label="FedProx TC")
ax.plot(fedprox["rounds"], fedprox["et"], color=C_PROX, linewidth=1.2, linestyle=":", label="FedProx ET")
ax.set_title("Per-Class Dice (WT / TC / ET)")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend(fontsize=7, ncol=2)
ax.set_ylim(0.55, 0.90)
ax.grid(True, alpha=0.3)

# --- 3. Worst-Client Mean Dice ---
ax = axes[0, 2]
ax.plot(fedavg["rounds"], fedavg["worst_client"], color=C_AVG, linewidth=1.5, label="FedAvg")
ax.plot(fedprox["rounds"], fedprox["worst_client"], color=C_PROX, linewidth=1.5, label="FedProx ($\\mu$=0.4)")
ax.fill_between(fedprox["rounds"],
                fedavg["worst_client"], fedprox["worst_client"],
                where=fedprox["worst_client"] > fedavg["worst_client"],
                alpha=0.15, color=C_PROX, label="FedProx advantage")
ax.set_title("Worst-Client Mean Dice")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend()
ax.set_ylim(0.1, 0.65)
ax.grid(True, alpha=0.3)

# --- 4. Client 7 (hardest small client, 65 samples) ---
ax = axes[1, 0]
ax.plot(fedavg["rounds"], fedavg["client_means"][7], color=C_AVG, linewidth=1.5, label="FedAvg")
ax.plot(fedprox["rounds"], fedprox["client_means"][7], color=C_PROX, linewidth=1.5, label="FedProx ($\\mu$=0.4)")
ax.set_title("Client 7 (65 samples — hardest)")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend()
ax.set_ylim(0.0, 0.65)
ax.grid(True, alpha=0.3)

# --- 5. Client 5 (180 samples — worst performer) ---
ax = axes[1, 1]
ax.plot(fedavg["rounds"], fedavg["client_means"][5], color=C_AVG, linewidth=1.5, label="FedAvg")
ax.plot(fedprox["rounds"], fedprox["client_means"][5], color=C_PROX, linewidth=1.5, label="FedProx ($\\mu$=0.4)")
ax.set_title("Client 5 (180 samples — worst performer)")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend()
ax.set_ylim(0.2, 0.60)
ax.grid(True, alpha=0.3)

# --- 6. Per-client best Dice bar chart ---
ax = axes[1, 2]
clients = list(range(8))
sizes = [135, 295, 140, 265, 180, 180, 55, 65]
best_avg = [max(fedavg["client_means"][c]) for c in clients]
best_prox = [max(fedprox["client_means"][c]) for c in clients]

x = np.arange(len(clients))
w = 0.35
bars1 = ax.bar(x - w/2, best_avg, w, color=C_AVG, alpha=0.8, label="FedAvg")
bars2 = ax.bar(x + w/2, best_prox, w, color=C_PROX, alpha=0.8, label="FedProx ($\\mu$=0.4)")
ax.set_title("Best Dice per Client (across all rounds)")
ax.set_xlabel("Client (train samples)")
ax.set_ylabel("Best Dice Score")
ax.set_xticks(x)
ax.set_xticklabels([f"C{c}\n({sizes[c]})" for c in clients], fontsize=8)
ax.legend()
ax.set_ylim(0.4, 0.90)
ax.grid(True, alpha=0.3, axis="y")

# Annotate deltas on bars
for i in range(len(clients)):
    delta = best_prox[i] - best_avg[i]
    color = "green" if delta > 0 else "red"
    sign = "+" if delta > 0 else ""
    y_pos = max(best_avg[i], best_prox[i]) + 0.005
    ax.text(x[i], y_pos, f"{sign}{delta:.3f}", ha="center", va="bottom", fontsize=7, color=color, fontweight="bold")

plt.tight_layout()
out_path = "/Users/basiakoch/Federated/federated-thesis/experiments/brats/fedavg_vs_fedprox_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved: {out_path}")
print(f"Saved: {out_path.replace('.png', '.pdf')}")
plt.close()
