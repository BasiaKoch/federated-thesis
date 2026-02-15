#!/usr/bin/env python3
"""Plot FedAvg vs FedProx (mu=0.2) — 2-client BraTS setup (corruption heterogeneity)."""

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


# --- Parse logs ---
fedavg = parse_log("/Users/basiakoch/Federated/federated-thesis/experiments/brats/logs/brats_fed_N_22536464.out")
prox02 = parse_log("/Users/basiakoch/Federated/federated-thesis/experiments/brats/logs/brats_fed_N_22536467.out")

C_AVG = "#2196F3"   # blue
C_PROX = "#F44336"  # red

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    r"FedAvg vs FedProx ($\mu$=0.2)  —  2 Clients, Equal Data, Corruption Heterogeneity, E=10",
    fontsize=13, fontweight="bold",
)

# --- 1. Global Pooled Mean Dice ---
ax = axes[0, 0]
ax.plot(fedavg["rounds"], fedavg["pooled_mean"], color=C_AVG, linewidth=1.5, label="FedAvg")
ax.plot(prox02["rounds"], prox02["pooled_mean"], color=C_PROX, linewidth=1.5, label=r"FedProx ($\mu$=0.2)")
ax.axhline(max(fedavg["pooled_mean"]), color=C_AVG, linestyle="--", alpha=0.4, linewidth=0.8)
ax.axhline(max(prox02["pooled_mean"]), color=C_PROX, linestyle="--", alpha=0.4, linewidth=0.8)
# Mark the FedAvg crash at R47
crash_idx = np.argmin(fedavg["pooled_mean"][20:]) + 20  # find worst after R20
ax.annotate(
    f'R{fedavg["rounds"][crash_idx]} crash\n{fedavg["pooled_mean"][crash_idx]:.3f}',
    xy=(fedavg["rounds"][crash_idx], fedavg["pooled_mean"][crash_idx]),
    xytext=(fedavg["rounds"][crash_idx] - 8, fedavg["pooled_mean"][crash_idx] + 0.06),
    arrowprops=dict(arrowstyle="->", color=C_AVG, lw=1.2),
    fontsize=8, color=C_AVG, fontweight="bold",
)
ax.set_title("Global Pooled Mean Dice")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend()
ax.set_ylim(0.45, 0.85)
ax.grid(True, alpha=0.3)

# --- 2. Per-class Dice (WT, TC, ET) ---
ax = axes[0, 1]
for data, clr, lbl in [(fedavg, C_AVG, "FedAvg"), (prox02, C_PROX, r"FedProx $\mu$=0.2")]:
    ax.plot(data["rounds"], data["wt"], color=clr, linewidth=1.2, label=f"{lbl} WT")
    ax.plot(data["rounds"], data["tc"], color=clr, linewidth=1.2, linestyle="--", label=f"{lbl} TC")
    ax.plot(data["rounds"], data["et"], color=clr, linewidth=1.2, linestyle=":", label=f"{lbl} ET")
ax.set_title("Per-Class Dice (WT / TC / ET)")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend(fontsize=7, ncol=2)
ax.set_ylim(0.2, 0.90)
ax.grid(True, alpha=0.3)

# --- 3. Client 0 (clean) vs Client 1 (corrupted) — FedAvg ---
ax = axes[0, 2]
ax.plot(fedavg["rounds"], fedavg["client_means"][0], color="#1565C0", linewidth=1.5,
        label="FedAvg — C0 (clean)")
ax.plot(fedavg["rounds"], fedavg["client_means"][1], color="#64B5F6", linewidth=1.5,
        linestyle="--", label="FedAvg — C1 (corrupted)")
ax.plot(prox02["rounds"], prox02["client_means"][0], color="#C62828", linewidth=1.5,
        label="FedProx — C0 (clean)")
ax.plot(prox02["rounds"], prox02["client_means"][1], color="#EF9A9A", linewidth=1.5,
        linestyle="--", label="FedProx — C1 (corrupted)")
ax.set_title("Per-Client Mean Dice")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend(fontsize=7)
ax.set_ylim(0.1, 0.85)
ax.grid(True, alpha=0.3)

# --- 4. Client 1 (corrupted) zoom ---
ax = axes[1, 0]
ax.plot(fedavg["rounds"], fedavg["client_means"][1], color=C_AVG, linewidth=1.5, label="FedAvg")
ax.plot(prox02["rounds"], prox02["client_means"][1], color=C_PROX, linewidth=1.5, label=r"FedProx ($\mu$=0.2)")
ax.fill_between(prox02["rounds"],
                fedavg["client_means"][1], prox02["client_means"][1],
                where=prox02["client_means"][1] > fedavg["client_means"][1],
                alpha=0.15, color=C_PROX, label="FedProx advantage")
ax.axhline(max(fedavg["client_means"][1]), color=C_AVG, linestyle="--", alpha=0.4, linewidth=0.8)
ax.axhline(max(prox02["client_means"][1]), color=C_PROX, linestyle="--", alpha=0.4, linewidth=0.8)
ax.set_title("Client 1 (Corrupted) — Mean Dice")
ax.set_xlabel("Round")
ax.set_ylabel("Dice Score")
ax.legend()
ax.set_ylim(0.1, 0.65)
ax.grid(True, alpha=0.3)

# --- 5. Pooled Loss ---
ax = axes[1, 1]
ax.plot(fedavg["rounds"], fedavg["pooled_loss"], color=C_AVG, linewidth=1.5, label="FedAvg")
ax.plot(prox02["rounds"], prox02["pooled_loss"], color=C_PROX, linewidth=1.5, label=r"FedProx ($\mu$=0.2)")
ax.set_title("Global Pooled Loss")
ax.set_xlabel("Round")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# --- 6. Rolling stability (std of last 10 rounds) ---
ax = axes[1, 2]
window = 5
avg_rolling_std = np.array([fedavg["pooled_mean"][max(0,i-window):i+1].std()
                            for i in range(len(fedavg["pooled_mean"]))])
prox_rolling_std = np.array([prox02["pooled_mean"][max(0,i-window):i+1].std()
                             for i in range(len(prox02["pooled_mean"]))])
ax.plot(fedavg["rounds"], avg_rolling_std, color=C_AVG, linewidth=1.5, label="FedAvg")
ax.plot(prox02["rounds"], prox_rolling_std, color=C_PROX, linewidth=1.5, label=r"FedProx ($\mu$=0.2)")
ax.set_title(f"Rolling Volatility (std, window={window})")
ax.set_xlabel("Round")
ax.set_ylabel("Std of Pooled Mean Dice")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "/Users/basiakoch/Federated/federated-thesis/experiments/brats/2client_fedavg_vs_fedprox.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
print(f"Saved: {out_path}")
print(f"Saved: {out_path.replace('.png', '.pdf')}")
plt.close()

# --- Summary ---
print("\n=== 2-Client Summary ===")
print(f"{'Metric':<25} {'FedAvg':>10} {'FedProx 0.2':>12} {'Delta':>8}")
print("-" * 58)
print(f"{'Best Pooled Mean':<25} {max(fedavg['pooled_mean']):>10.4f} {max(prox02['pooled_mean']):>12.4f} {max(prox02['pooled_mean'])-max(fedavg['pooled_mean']):>+8.4f}")
print(f"{'Final Pooled Mean':<25} {fedavg['pooled_mean'][-1]:>10.4f} {prox02['pooled_mean'][-1]:>12.4f} {prox02['pooled_mean'][-1]-fedavg['pooled_mean'][-1]:>+8.4f}")
print(f"{'Best WT':<25} {max(fedavg['wt']):>10.4f} {max(prox02['wt']):>12.4f} {max(prox02['wt'])-max(fedavg['wt']):>+8.4f}")
print(f"{'Best TC':<25} {max(fedavg['tc']):>10.4f} {max(prox02['tc']):>12.4f} {max(prox02['tc'])-max(fedavg['tc']):>+8.4f}")
print(f"{'Best ET':<25} {max(fedavg['et']):>10.4f} {max(prox02['et']):>12.4f} {max(prox02['et'])-max(fedavg['et']):>+8.4f}")
print(f"{'Client 0 best':<25} {max(fedavg['client_means'][0]):>10.4f} {max(prox02['client_means'][0]):>12.4f} {max(prox02['client_means'][0])-max(fedavg['client_means'][0]):>+8.4f}")
print(f"{'Client 1 best':<25} {max(fedavg['client_means'][1]):>10.4f} {max(prox02['client_means'][1]):>12.4f} {max(prox02['client_means'][1])-max(fedavg['client_means'][1]):>+8.4f}")
print(f"{'Best-Final gap':<25} {max(fedavg['pooled_mean'])-fedavg['pooled_mean'][-1]:>10.4f} {max(prox02['pooled_mean'])-prox02['pooled_mean'][-1]:>12.4f}")
print(f"{'Worst single round':<25} {min(fedavg['pooled_mean']):>10.4f} {min(prox02['pooled_mean']):>12.4f}")
print(f"{'Late-stage std (R30-50)':<25} {fedavg['pooled_mean'][29:].std():>10.4f} {prox02['pooled_mean'][29:].std():>12.4f}")
