#!/usr/bin/env python3
"""Compare FedAvg vs FedProx from result JSON files."""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np

# --- Load results ---
fedavg_path = "results/mnist/fedavg_n10_niid_r30_20260209_115632_results.json"
fedprox_path = "results/mnist/fedprox_n10_niid_mu1.0_r30_20260209_121743_results.json"

with open(fedavg_path) as f:
    fedavg = json.load(f)
with open(fedprox_path) as f:
    fedprox = json.load(f)

rounds = fedavg["per_round_metrics"]["rounds"]
fedavg_acc = fedavg["per_round_metrics"]["global_test_accuracy"]
fedprox_acc = fedprox["per_round_metrics"]["global_test_accuracy"]
fedavg_loss = fedavg["per_round_metrics"]["global_test_loss"]
fedprox_loss = fedprox["per_round_metrics"]["global_test_loss"]

mu = fedprox["experiment"]["mu"]
n_clients = fedavg["experiment"]["num_clients"]
partition = fedavg["experiment"]["partition"]
drop_pct = fedavg["experiment"]["drop_percent"]
local_ep = fedavg["experiment"]["local_epochs"]

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
ax1.plot(rounds, fedavg_acc, "o-", color="#d62728", label="FedAvg", markersize=3, linewidth=1.5)
ax1.plot(rounds, fedprox_acc, "s-", color="#1f77b4", label=f"FedProx (μ={mu})", markersize=3, linewidth=1.5)
ax1.set_xlabel("Communication Round")
ax1.set_ylabel("Test Accuracy")
ax1.set_title("Test Accuracy per Round")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.0)

# Loss
ax2.plot(rounds, fedavg_loss, "o-", color="#d62728", label="FedAvg", markersize=3, linewidth=1.5)
ax2.plot(rounds, fedprox_loss, "s-", color="#1f77b4", label=f"FedProx (μ={mu})", markersize=3, linewidth=1.5)
ax2.set_xlabel("Communication Round")
ax2.set_ylabel("Test Loss")
ax2.set_title("Test Loss per Round")
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle(
    f"FedAvg vs FedProx — MNIST MCLR, {n_clients} clients, {partition} partition,\n"
    f"{int(drop_pct*100)}% stragglers, {local_ep} local epochs",
    fontsize=12, fontweight="bold",
)
plt.tight_layout()

out_path = "results/mnist/fedavg_vs_fedprox_n10_niid_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
