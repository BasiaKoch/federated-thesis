#!/usr/bin/env python3
"""Plot FedAvg vs FedProx (mu=0.1, mu=0.2) from BraTS log files — E=10 local epochs."""

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


# --- Parse all three E=10 logs ---
fedavg    = parse_log("/Users/basiakoch/Federated/federated-thesis/experiments/brats/logs/brats_fed_N_22517911.out")
prox_01   = parse_log("/Users/basiakoch/Federated/federated-thesis/experiments/brats/logs/brats_fed_N_22517908.out")
prox_02   = parse_log("/Users/basiakoch/Federated/federated-thesis/experiments/brats/logs/brats_fed_N_22517930.out")

# --- Colours ---
C_AVG  = "#2196F3"   # blue
C_P01  = "#F44336"   # red
C_P02  = "#FF9800"   # orange

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "FedAvg vs FedProx  —  BraTS 2D, 8 clients, E=10, LR=0.01",
    fontsize=14, fontweight="bold",
)

# --- 1. Global Pooled Mean Dice ---
ax = axes[0, 0]
ax.plot(fedavg["rounds"],  fedavg["pooled_mean"],  color=C_AVG, linewidth=1.5, label="FedAvg")
ax.plot(prox_01["rounds"], prox_01["pooled_mean"], color=C_P01, linewidth=1.5, label=r"FedProx ($\mu$=0.1)")
ax.plot(prox_02["rounds"], prox_02["pooled_mean"], color=C_P02, linewidth=1.5, label=r"FedProx ($\mu$=0.2)")
ax.axhline(max(fedavg["pooled_mean"]),  color=C_AVG, linestyle="--", alpha=0.4, linewidth=0.8)
ax.axhline(max(prox_01["pooled_mean"]), color=C_P01, linestyle="--", alpha=0.4, linewidth=0.8)
ax.axhline(max(prox_02["pooled_mean"]), color=C_P02, linestyle="--", alpha=0.4, linewidth=0.8)
ax.set_title("Global Pooled Mean Dice")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend()
ax.set_ylim(0.6, 0.85)
ax.grid(True, alpha=0.3)

# --- 2. Per-class Dice (WT, TC, ET) ---
ax = axes[0, 1]
for data, clr, lbl in [(fedavg, C_AVG, "FedAvg"), (prox_01, C_P01, r"FedProx $\mu$=0.1"), (prox_02, C_P02, r"FedProx $\mu$=0.2")]:
    ax.plot(data["rounds"], data["wt"], color=clr, linewidth=1.2, label=f"{lbl} WT")
    ax.plot(data["rounds"], data["tc"], color=clr, linewidth=1.2, linestyle="--", label=f"{lbl} TC")
    ax.plot(data["rounds"], data["et"], color=clr, linewidth=1.2, linestyle=":", label=f"{lbl} ET")
ax.set_title("Per-Class Dice (WT / TC / ET)")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend(fontsize=6, ncol=3)
ax.set_ylim(0.55, 0.92)
ax.grid(True, alpha=0.3)

# --- 3. Worst-Client Mean Dice ---
ax = axes[0, 2]
ax.plot(fedavg["rounds"],  fedavg["worst_client"],  color=C_AVG, linewidth=1.5, label="FedAvg")
ax.plot(prox_01["rounds"], prox_01["worst_client"], color=C_P01, linewidth=1.5, label=r"FedProx ($\mu$=0.1)")
ax.plot(prox_02["rounds"], prox_02["worst_client"], color=C_P02, linewidth=1.5, label=r"FedProx ($\mu$=0.2)")
ax.set_title("Worst-Client Mean Dice")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend()
ax.set_ylim(0.1, 0.65)
ax.grid(True, alpha=0.3)

# --- 4. Client 7 (hardest small client, 65 samples) ---
ax = axes[1, 0]
ax.plot(fedavg["rounds"],  fedavg["client_means"][7],  color=C_AVG, linewidth=1.5, label="FedAvg")
ax.plot(prox_01["rounds"], prox_01["client_means"][7], color=C_P01, linewidth=1.5, label=r"FedProx ($\mu$=0.1)")
ax.plot(prox_02["rounds"], prox_02["client_means"][7], color=C_P02, linewidth=1.5, label=r"FedProx ($\mu$=0.2)")
ax.set_title("Client 7 (65 samples — hardest)")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend()
ax.set_ylim(0.0, 0.65)
ax.grid(True, alpha=0.3)

# --- 5. Client 5 (180 samples — worst performer) ---
ax = axes[1, 1]
ax.plot(fedavg["rounds"],  fedavg["client_means"][5],  color=C_AVG, linewidth=1.5, label="FedAvg")
ax.plot(prox_01["rounds"], prox_01["client_means"][5], color=C_P01, linewidth=1.5, label=r"FedProx ($\mu$=0.1)")
ax.plot(prox_02["rounds"], prox_02["client_means"][5], color=C_P02, linewidth=1.5, label=r"FedProx ($\mu$=0.2)")
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
best_avg  = [max(fedavg["client_means"][c])  for c in clients]
best_p01  = [max(prox_01["client_means"][c]) for c in clients]
best_p02  = [max(prox_02["client_means"][c]) for c in clients]

x = np.arange(len(clients))
w = 0.25
ax.bar(x - w,   best_avg, w, color=C_AVG, alpha=0.8, label="FedAvg")
ax.bar(x,       best_p01, w, color=C_P01, alpha=0.8, label=r"FedProx ($\mu$=0.1)")
ax.bar(x + w,   best_p02, w, color=C_P02, alpha=0.8, label=r"FedProx ($\mu$=0.2)")
ax.set_title("Best Dice per Client (across all rounds)")
ax.set_xlabel("Client (train samples)")
ax.set_ylabel("Best Dice Score")
ax.set_xticks(x)
ax.set_xticklabels([f"C{c}\n({sizes[c]})" for c in clients], fontsize=8)
ax.legend(fontsize=8)
ax.set_ylim(0.4, 0.92)
ax.grid(True, alpha=0.3, axis="y")

# Annotate best deltas (mu=0.1 vs FedAvg)
for i in range(len(clients)):
    delta = best_p01[i] - best_avg[i]
    color = "green" if delta > 0 else "red"
    sign = "+" if delta > 0 else ""
    y_pos = max(best_avg[i], best_p01[i], best_p02[i]) + 0.005
    ax.text(x[i], y_pos, f"{sign}{delta:.3f}", ha="center", va="bottom",
            fontsize=6, color=color, fontweight="bold")

plt.tight_layout()
out_path = "/Users/basiakoch/Federated/federated-thesis/experiments/brats/10_local_fedavg_vs_fedprox.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved: {out_path}")
print(f"Saved: {out_path.replace('.png', '.pdf')}")
plt.close()

# --- Print summary table ---
print("\n=== Summary (E=10) ===")
print(f"{'Metric':<25} {'FedAvg':>10} {'FedProx 0.1':>12} {'FedProx 0.2':>12}")
print("-" * 62)
print(f"{'Best Pooled Mean':<25} {max(fedavg['pooled_mean']):>10.4f} {max(prox_01['pooled_mean']):>12.4f} {max(prox_02['pooled_mean']):>12.4f}")
print(f"{'Final Pooled Mean':<25} {fedavg['pooled_mean'][-1]:>10.4f} {prox_01['pooled_mean'][-1]:>12.4f} {prox_02['pooled_mean'][-1]:>12.4f}")
print(f"{'Best WT':<25} {max(fedavg['wt']):>10.4f} {max(prox_01['wt']):>12.4f} {max(prox_02['wt']):>12.4f}")
print(f"{'Best TC':<25} {max(fedavg['tc']):>10.4f} {max(prox_01['tc']):>12.4f} {max(prox_02['tc']):>12.4f}")
print(f"{'Best ET':<25} {max(fedavg['et']):>10.4f} {max(prox_01['et']):>12.4f} {max(prox_02['et']):>12.4f}")
print(f"{'Best Worst-Client':<25} {max(fedavg['worst_client']):>10.4f} {max(prox_01['worst_client']):>12.4f} {max(prox_02['worst_client']):>12.4f}")
print(f"{'Final Worst-Client':<25} {fedavg['worst_client'][-1]:>10.4f} {prox_01['worst_client'][-1]:>12.4f} {prox_02['worst_client'][-1]:>12.4f}")
