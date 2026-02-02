#!/usr/bin/env python3
"""
Plot client-1 MeanPresent Dice vs epoch/round from SLURM stdout logs.

Parses lines like:
[Round 30] Client0 Mean=0.8278 | Client1 Mean=0.5935 | Pooled Mean=0.7826
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# Extract round + client1 mean from the per-round evaluation line
ROUND_CLIENT1_RE = re.compile(
    r"\[Round\s+(?P<round>\d+)\]\s+.*?\bClient1\s+Mean=(?P<c1>[0-9]*\.[0-9]+)"
)

def parse_log_client1(log_path: Path) -> Dict[str, List[float]]:
    rounds: List[int] = []
    c1: List[float] = []

    text = log_path.read_text(errors="ignore")
    for line in text.splitlines():
        m = ROUND_CLIENT1_RE.search(line)
        if not m:
            continue
        rounds.append(int(m.group("round")))
        c1.append(float(m.group("c1")))

    # keep last occurrence per round (in case of repeats)
    last: Dict[int, float] = {}
    for r, v in zip(rounds, c1):
        last[r] = v
    rs = sorted(last.keys())
    vs = [last[r] for r in rs]
    return {"rounds": rs, "client1": vs}

def best_and_final(rounds: List[int], values: List[float]) -> Tuple[float, int, float, int]:
    if not values:
        return (float("nan"), -1, float("nan"), -1)
    best_i = max(range(len(values)), key=lambda i: values[i])
    return values[best_i], rounds[best_i], values[-1], rounds[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_a", type=Path, required=True, help="First log (e.g., FedAvg)")
    ap.add_argument("--log_b", type=Path, required=True, help="Second log (e.g., FedProx)")
    ap.add_argument("--label_a", type=str, default="FedAvg")
    ap.add_argument("--label_b", type=str, default="FedProx")
    ap.add_argument("--out_png", type=Path, default=None)
    ap.add_argument("--xlabel", type=str, default="Epoch")  # technically rounds, but matches your thesis labeling choice
    ap.add_argument("--title", type=str, default="Client 1 - Test Dice (Global Model)")
    args = ap.parse_args()

    a = parse_log_client1(args.log_a)
    b = parse_log_client1(args.log_b)

    if not a["rounds"]:
        raise SystemExit(f"No '[Round ...] ... Client1 Mean=...' lines found in {args.log_a}")
    if not b["rounds"]:
        raise SystemExit(f"No '[Round ...] ... Client1 Mean=...' lines found in {args.log_b}")

    a_best, a_best_r, a_final, a_final_r = best_and_final(a["rounds"], a["client1"])
    b_best, b_best_r, b_final, b_final_r = best_and_final(b["rounds"], b["client1"])

    print("=== Client 1 Dice Summary ===")
    print(f"{args.label_a}: best={a_best:.4f} @ round {a_best_r} | final={a_final:.4f} @ round {a_final_r}")
    print(f"{args.label_b}: best={b_best:.4f} @ round {b_best_r} | final={b_final:.4f} @ round {b_final_r}")

    plt.figure()
    plt.plot(a["rounds"], a["client1"], marker="o", label=args.label_a)
    plt.plot(b["rounds"], b["client1"], marker="o", label=args.label_b)
    plt.title(args.title)
    plt.xlabel(args.xlabel)
    plt.ylabel("Dice Coefficient (Client 1 MeanPresent)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    # Optional: annotate final values to make the end difference obvious
    ar, ay = a["rounds"][-1], a["client1"][-1]
    br, by = b["rounds"][-1], b["client1"][-1]
    plt.scatter([ar, br], [ay, by])
    plt.annotate(f"{ay:.4f}", (ar, ay), textcoords="offset points", xytext=(6, -10))
    plt.annotate(f"{by:.4f}", (br, by), textcoords="offset points", xytext=(6, 6))

    if args.out_png:
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out_png, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {args.out_png}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
