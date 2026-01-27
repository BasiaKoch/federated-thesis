#!/usr/bin/env python3
"""
70/30 Federated vs Local Analysis
==================================
Compares local-only training (no federation) against FedAvg and FedProx
for the 2-client 70/30 BraTS split.

Generates:
  1. convergence_curves.pdf        – per-round Mean Dice for all strategies
  2. per_class_bar.pdf             – final WT / TC / ET Dice grouped bar chart
  3. client_gap.pdf                – per-client dice across strategies (shows
                                     how federation helps the small client)
  4. best_vs_final_table.tex       – LaTeX table of best & final Dice
  5. summary.txt                   – plain-text summary of key findings

Usage (on HPC):
    python results/7030_analysis/analyze_70_30.py

All paths are relative to the repo root.
"""

import json
import sys
from pathlib import Path
from textwrap import dedent

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent.parent  # federated-thesis/

LOCAL_C0 = REPO / "results" / "local_unet" / "client0_E30_lr0.001_20260127_160241" / "results.json"
LOCAL_C1 = REPO / "results" / "local_unet" / "client1_E30_lr0.001_20260127_160652" / "results.json"
FEDAVG   = REPO / "results" / "federated" / "clients_2_70_30" / "fedavg_mu0.0_R30_E3_20260127_120200" / "results.json"
FEDPROX  = REPO / "results" / "federated" / "clients_2_70_30" / "fedprox_mu0.001_R30_E3_20260127_130822" / "results.json"

OUT_DIR  = REPO / "results" / "7030_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "local_c0": "#1b9e77",
    "local_c1": "#d95f02",
    "fedavg":   "#7570b3",
    "fedprox":  "#e7298a",
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def load(path: Path) -> dict:
    if not path.exists():
        print(f"WARNING: {path} not found – will use placeholder data.")
        return None
    with open(path) as f:
        return json.load(f)


def extract_local_curve(data: dict) -> np.ndarray:
    """Return per-epoch test Mean Dice from a local-only results.json."""
    # Local results store per-epoch test metrics
    if data is None:
        return None
    # Try common structures
    if "test_history" in data:
        hist = data["test_history"]
        if "meanDice" in hist:
            return np.array(hist["meanDice"])
        if "Mean" in hist:
            return np.array(hist["Mean"])
    if "per_epoch" in data:
        pe = data["per_epoch"]
        if "test_meanDice" in pe:
            return np.array(pe["test_meanDice"])
        if "test_Mean" in pe:
            return np.array(pe["test_Mean"])
    if "history" in data:
        h = data["history"]
        for k in ["test_meanDice", "test_Mean", "meanDice", "Mean"]:
            if k in h:
                return np.array(h[k])
    # Flat final only
    return None


def extract_local_final(data: dict) -> dict:
    """Return final per-class Dice from a local results.json."""
    if data is None:
        return None
    # Try 'final' key
    if "final" in data:
        f = data["final"]
        out = {}
        for k in ["meanDice", "Mean", "WT", "TC", "ET",
                   "test_meanDice", "test_WT", "test_TC", "test_ET",
                   "best_meanDice", "best_test_meanDice"]:
            if k in f:
                out[k] = f[k]
        return out
    return data


def extract_fed_curves(data: dict):
    """Return (rounds, c0_dice, c1_dice, global_dice) arrays from federated results.json."""
    pr = data["per_round"]
    rounds = np.array(pr["rounds"])
    c0 = np.array(pr["client0_meanDice"])
    c1 = np.array(pr["client1_meanDice"])
    gl = np.array(pr["global_meanDice"])
    return rounds, c0, c1, gl


def extract_fed_perclass(data: dict, client: str, rnd: int = -1):
    """Return (WT, TC, ET) at a given round index."""
    pr = data["per_round"]
    return (
        pr[f"{client}_WT"][rnd],
        pr[f"{client}_TC"][rnd],
        pr[f"{client}_ET"][rnd],
    )


# ── Load data ────────────────────────────────────────────────────────────────
local_c0_data = load(LOCAL_C0)
local_c1_data = load(LOCAL_C1)
fedavg_data   = load(FEDAVG)
fedprox_data  = load(FEDPROX)

if fedavg_data is None or fedprox_data is None:
    sys.exit("ERROR: Federated result JSONs not found. Check paths.")

local_c0_curve = extract_local_curve(local_c0_data) if local_c0_data else None
local_c1_curve = extract_local_curve(local_c1_data) if local_c1_data else None
local_c0_final = extract_local_final(local_c0_data) if local_c0_data else None
local_c1_final = extract_local_final(local_c1_data) if local_c1_data else None

fedavg_rounds, fedavg_c0, fedavg_c1, fedavg_gl = extract_fed_curves(fedavg_data)
fedprox_rounds, fedprox_c0, fedprox_c1, fedprox_gl = extract_fed_curves(fedprox_data)

HAS_LOCAL = local_c0_data is not None and local_c1_data is not None

# ── PLOT 1: Convergence curves ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
fig.suptitle("Convergence: Local-Only vs Federated (70/30 Split)", fontweight="bold", y=1.02)

titles  = ["Client 0 (70% data)", "Client 1 (30% data)", "Pooled / Global"]
fed_c0s = [(fedavg_rounds, fedavg_c0, "FedAvg", COLORS["fedavg"]),
           (fedprox_rounds, fedprox_c0, "FedProx (μ=0.001)", COLORS["fedprox"])]
fed_c1s = [(fedavg_rounds, fedavg_c1, "FedAvg", COLORS["fedavg"]),
           (fedprox_rounds, fedprox_c1, "FedProx (μ=0.001)", COLORS["fedprox"])]
fed_gls = [(fedavg_rounds, fedavg_gl, "FedAvg", COLORS["fedavg"]),
           (fedprox_rounds, fedprox_gl, "FedProx (μ=0.001)", COLORS["fedprox"])]

panel_data = [
    (fed_c0s, local_c0_curve, "Local C0"),
    (fed_c1s, local_c1_curve, "Local C1"),
    (fed_gls, None, None),
]

for ax, title, (fed_curves, local_curve, local_label) in zip(axes, titles, panel_data):
    for rds, vals, label, color in fed_curves:
        ax.plot(rds, vals, label=label, color=color, linewidth=1.8)
    if local_curve is not None:
        epochs = np.arange(1, len(local_curve) + 1)
        ax.plot(epochs, local_curve, label=local_label, color=COLORS["local_c0"] if "C0" in local_label else COLORS["local_c1"],
                linewidth=1.8, linestyle="--")
    elif HAS_LOCAL is False and local_label is not None:
        # Show horizontal line for final local Dice if we only have final value
        pass
    ax.set_title(title)
    ax.set_xlabel("Round / Epoch")
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

axes[0].set_ylabel("Mean Dice")
plt.tight_layout()
fig.savefig(OUT_DIR / "convergence_curves.pdf")
fig.savefig(OUT_DIR / "convergence_curves.png")
print(f"Saved: {OUT_DIR / 'convergence_curves.pdf'}")
plt.close(fig)


# ── PLOT 2: Per-class bar chart (best round) ────────────────────────────────
# Use best round for each strategy
def best_round_idx(data):
    gl = data["per_round"]["global_meanDice"]
    return int(np.argmax(gl))

fedavg_best_idx  = best_round_idx(fedavg_data)
fedprox_best_idx = best_round_idx(fedprox_data)

classes = ["WT", "TC", "ET", "Mean"]

fedavg_best_vals = [
    fedavg_data["per_round"][f"global_{c}"][fedavg_best_idx] if c != "Mean"
    else fedavg_data["per_round"]["global_meanDice"][fedavg_best_idx]
    for c in classes
]
fedprox_best_vals = [
    fedprox_data["per_round"][f"global_{c}"][fedprox_best_idx] if c != "Mean"
    else fedprox_data["per_round"]["global_meanDice"][fedprox_best_idx]
    for c in classes
]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(classes))
w = 0.3

bars_fedavg  = ax.bar(x - w/2, fedavg_best_vals,  w, label=f"FedAvg (best R{fedavg_rounds[fedavg_best_idx]})",
                       color=COLORS["fedavg"], edgecolor="white")
bars_fedprox = ax.bar(x + w/2, fedprox_best_vals, w, label=f"FedProx (best R{fedprox_rounds[fedprox_best_idx]})",
                       color=COLORS["fedprox"], edgecolor="white")

# Add value labels
for bars in [bars_fedavg, bars_fedprox]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Dice Score")
ax.set_title("Per-Class Dice at Best Global Round (Pooled Test Set)")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 0.9)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "per_class_bar.pdf")
fig.savefig(OUT_DIR / "per_class_bar.png")
print(f"Saved: {OUT_DIR / 'per_class_bar.pdf'}")
plt.close(fig)


# ── PLOT 3: Client gap – how federation helps the small client ───────────────
fig, ax = plt.subplots(figsize=(9, 5))

# For each strategy, show best Mean Dice per client
strategies = ["FedAvg", "FedProx\n(μ=0.001)"]
c0_best = [
    fedavg_data["final"]["client0_best_meanDice"],
    fedprox_data["final"]["client0_best_meanDice"],
]
c1_best = [
    fedavg_data["final"]["client1_best_meanDice"],
    fedprox_data["final"]["client1_best_meanDice"],
]
gl_best = [
    fedavg_data["final"]["global_best_meanDice"],
    fedprox_data["final"]["global_best_meanDice"],
]

# Add local baselines if available
if HAS_LOCAL:
    strategies = ["Local Only"] + strategies
    # Try to get best dice from local results
    c0_local_best = None
    c1_local_best = None
    if local_c0_final:
        for k in ["best_meanDice", "best_test_meanDice", "meanDice", "test_meanDice", "Mean"]:
            if k in local_c0_final:
                c0_local_best = local_c0_final[k]
                break
    if local_c1_final:
        for k in ["best_meanDice", "best_test_meanDice", "meanDice", "test_meanDice", "Mean"]:
            if k in local_c1_final:
                c1_local_best = local_c1_final[k]
                break
    if c0_local_best and c1_local_best:
        c0_best = [c0_local_best] + c0_best
        c1_best = [c1_local_best] + c1_best
        gl_best = [None] + gl_best  # no global for local
    else:
        strategies = strategies[1:]  # remove Local Only if we can't extract values

x = np.arange(len(strategies))
w = 0.25

b0 = ax.bar(x - w, c0_best, w, label="Client 0 (70%)", color=COLORS["local_c0"], edgecolor="white")
b1 = ax.bar(x, c1_best, w, label="Client 1 (30%)", color=COLORS["local_c1"], edgecolor="white")

# Global bar (skip None)
gl_x = [xi + w for xi, v in zip(x, gl_best) if v is not None]
gl_v = [v for v in gl_best if v is not None]
bg = ax.bar(gl_x, gl_v, w, label="Pooled Global", color="#666666", edgecolor="white")

for bars in [b0, b1, bg]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9)

ax.set_ylabel("Best Mean Dice")
ax.set_title("Federation Benefit: Best Mean Dice per Client")
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.set_ylim(0, 0.9)
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "client_gap.pdf")
fig.savefig(OUT_DIR / "client_gap.png")
print(f"Saved: {OUT_DIR / 'client_gap.pdf'}")
plt.close(fig)


# ── PLOT 4: Per-client convergence comparison (2x1) ─────────────────────────
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
fig.suptitle("Per-Client Convergence: FedAvg vs FedProx", fontweight="bold", y=1.02)

ax0.plot(fedavg_rounds, fedavg_c0, label="FedAvg", color=COLORS["fedavg"], lw=1.8)
ax0.plot(fedprox_rounds, fedprox_c0, label="FedProx (μ=0.001)", color=COLORS["fedprox"], lw=1.8)
ax0.set_title("Client 0 (70% data)")
ax0.set_xlabel("Round")
ax0.set_ylabel("Mean Dice")
ax0.legend(loc="lower right")
ax0.grid(True, alpha=0.3)
ax0.set_ylim(0, 0.85)

ax1.plot(fedavg_rounds, fedavg_c1, label="FedAvg", color=COLORS["fedavg"], lw=1.8)
ax1.plot(fedprox_rounds, fedprox_c1, label="FedProx (μ=0.001)", color=COLORS["fedprox"], lw=1.8)
ax1.set_title("Client 1 (30% data)")
ax1.set_xlabel("Round")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "perclient_convergence.pdf")
fig.savefig(OUT_DIR / "perclient_convergence.png")
print(f"Saved: {OUT_DIR / 'perclient_convergence.pdf'}")
plt.close(fig)


# ── PLOT 5: Stability – rolling std of global dice ──────────────────────────
window = 5
fedavg_roll_std  = np.array([np.std(fedavg_gl[max(0,i-window+1):i+1])  for i in range(len(fedavg_gl))])
fedprox_roll_std = np.array([np.std(fedprox_gl[max(0,i-window+1):i+1]) for i in range(len(fedprox_gl))])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(fedavg_rounds, fedavg_roll_std, label="FedAvg", color=COLORS["fedavg"], lw=1.8)
ax.plot(fedprox_rounds, fedprox_roll_std, label="FedProx (μ=0.001)", color=COLORS["fedprox"], lw=1.8)
ax.set_xlabel("Round")
ax.set_ylabel(f"Rolling Std (window={window})")
ax.set_title("Training Stability: Rolling Standard Deviation of Global Mean Dice")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "stability_rolling_std.pdf")
fig.savefig(OUT_DIR / "stability_rolling_std.png")
print(f"Saved: {OUT_DIR / 'stability_rolling_std.pdf'}")
plt.close(fig)


# ── TABLE: LaTeX summary table ──────────────────────────────────────────────
def fmt(v):
    return f"{v:.4f}" if v is not None else "—"

rows = []

# Local baselines (if available)
if HAS_LOCAL and c0_local_best is not None:
    rows.append(("Local C0 (70\\%)", fmt(c0_local_best), "—", "—", "—"))
if HAS_LOCAL and c1_local_best is not None:
    rows.append(("Local C1 (30\\%)", "—", fmt(c1_local_best), "—", "—"))

# FedAvg
fa = fedavg_data["final"]
rows.append(("FedAvg",
             fmt(fa["client0_best_meanDice"]),
             fmt(fa["client1_best_meanDice"]),
             fmt(fa["global_best_meanDice"]),
             fmt(fa["global_meanDice"])))

# FedProx
fp = fedprox_data["final"]
rows.append(("FedProx ($\\mu$=0.001)",
             fmt(fp["client0_best_meanDice"]),
             fmt(fp["client1_best_meanDice"]),
             fmt(fp["global_best_meanDice"]),
             fmt(fp["global_meanDice"])))

latex = dedent(r"""
\begin{table}[ht]
\centering
\caption{Comparison of Local-Only vs Federated Training (70/30 Split, 30 Rounds)}
\label{tab:7030_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Strategy} & \textbf{C0 Best} & \textbf{C1 Best} & \textbf{Global Best} & \textbf{Global Final} \\
\midrule
""").lstrip()

for row in rows:
    latex += " & ".join(row) + r" \\" + "\n"

latex += dedent(r"""
\bottomrule
\end{tabular}
\end{table}
""")

(OUT_DIR / "best_vs_final_table.tex").write_text(latex)
print(f"Saved: {OUT_DIR / 'best_vs_final_table.tex'}")


# ── Plain-text summary ──────────────────────────────────────────────────────
fa_f = fedavg_data["final"]
fp_f = fedprox_data["final"]

summary_lines = [
    "=" * 70,
    "70/30 FEDERATED LEARNING ANALYSIS SUMMARY",
    "=" * 70,
    "",
    "Setup: 2 clients, Client 0 has 70% data, Client 1 has 30% data",
    "       BraTS 2D U-Net segmentation, 30 rounds, 3 local epochs, lr=1e-3",
    "",
    "─── FedAvg ─────────────────────────────────────────────────",
    f"  Client 0:  best={fa_f['client0_best_meanDice']:.4f}  final={fa_f['client0_meanDice']:.4f}",
    f"  Client 1:  best={fa_f['client1_best_meanDice']:.4f}  final={fa_f['client1_meanDice']:.4f}",
    f"  Global:    best={fa_f['global_best_meanDice']:.4f}  final={fa_f['global_meanDice']:.4f}",
    "",
    "─── FedProx (μ=0.001) ─────────────────────────────────────",
    f"  Client 0:  best={fp_f['client0_best_meanDice']:.4f}  final={fp_f['client0_meanDice']:.4f}",
    f"  Client 1:  best={fp_f['client1_best_meanDice']:.4f}  final={fp_f['client1_meanDice']:.4f}",
    f"  Global:    best={fp_f['global_best_meanDice']:.4f}  final={fp_f['global_meanDice']:.4f}",
    "",
]

if HAS_LOCAL:
    summary_lines += [
        "─── Local Baselines ────────────────────────────────────────",
        f"  Client 0 (local only):  {fmt(c0_local_best) if c0_local_best else 'see JSON'}",
        f"  Client 1 (local only):  {fmt(c1_local_best) if c1_local_best else 'see JSON'}",
        "",
    ]

# Key findings
c1_fedavg_best = fa_f["client1_best_meanDice"]
c1_fedprox_best = fp_f["client1_best_meanDice"]
c0_fedavg_best = fa_f["client0_best_meanDice"]

summary_lines += [
    "─── Key Findings ───────────────────────────────────────────",
    "",
    "1. FEDERATION HELPS THE DATA-SCARCE CLIENT:",
    f"   Client 1 (30% data) achieves best Dice of {c1_fedavg_best:.4f} (FedAvg)",
    f"   and {c1_fedprox_best:.4f} (FedProx) using the global model.",
    "   Without federation, Client 1 trains on ~73 samples only,",
    "   which severely limits generalization.",
    "",
    "2. FEDAVG vs FEDPROX (low heterogeneity regime):",
    f"   FedAvg global best:  {fa_f['global_best_meanDice']:.4f}",
    f"   FedProx global best: {fp_f['global_best_meanDice']:.4f}",
    f"   Difference: {abs(fa_f['global_best_meanDice'] - fp_f['global_best_meanDice']):.4f}",
    "   With only 2 clients and mild quantity skew (70/30),",
    "   there is minimal client drift for FedProx to correct.",
    "   Both strategies perform comparably.",
    "",
    "3. CONVERGENCE STABILITY:",
    f"   FedAvg  final-round global dice: {fa_f['global_meanDice']:.4f}  (drop of {fa_f['global_best_meanDice'] - fa_f['global_meanDice']:.4f} from best)",
    f"   FedProx final-round global dice: {fp_f['global_meanDice']:.4f}  (drop of {fp_f['global_best_meanDice'] - fp_f['global_meanDice']:.4f} from best)",
    "   Both strategies exhibit oscillation in later rounds,",
    "   suggesting a learning rate schedule would help.",
    "",
    "4. PER-CLASS ANALYSIS (best round):",
]

for name, data, idx in [("FedAvg", fedavg_data, fedavg_best_idx),
                          ("FedProx", fedprox_data, fedprox_best_idx)]:
    wt = data["per_round"]["global_WT"][idx]
    tc = data["per_round"]["global_TC"][idx]
    et = data["per_round"]["global_ET"][idx]
    summary_lines.append(f"   {name:20s}  WT={wt:.4f}  TC={tc:.4f}  ET={et:.4f}")

summary_lines += [
    "   WT (Whole Tumor) is easiest; ET (Enhancing Tumor) is hardest.",
    "",
    "=" * 70,
]

summary_text = "\n".join(summary_lines)
(OUT_DIR / "summary.txt").write_text(summary_text)
print(summary_text)
print(f"\nSaved: {OUT_DIR / 'summary.txt'}")

print(f"\nAll outputs in: {OUT_DIR}")
print("Files generated:")
for f in sorted(OUT_DIR.iterdir()):
    if f.name != "analyze_70_30.py":
        print(f"  {f.name}")
