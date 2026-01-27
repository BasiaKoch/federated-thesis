#!/usr/bin/env python3
import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


# -----------------------
# Defaults (edit if needed)
# -----------------------
DATA_ROOT = Path(os.environ.get(
    "BRATS_DATA_DIR",
    "/home/bk489/federated/federated-thesis/data/brats2020_top10_slices_split_npz"
))
TRAIN_DIR = DATA_ROOT / "train"

DEFAULT_BASE_OUT = Path("/home/bk489/federated/federated-thesis/data/partitions")

SEED = 42

# BraTS20 Training IDs: LGG range used in your old script
LGG_START = 260
LGG_END = 335  # inclusive

# Create per-client folder with symlinks to NPZ files
MAKE_SYMLINK_DATA_DEFAULT = True


# -----------------------
# Helpers from your old script
# -----------------------
def remap_brats_labels(y: np.ndarray) -> np.ndarray:
    """Map label 4 -> 3 (so labels become 0,1,2,3)."""
    y2 = y.copy()
    y2[y2 == 4] = 3
    return y2


def parse_case_num(case_id: str) -> int:
    # "BraTS20_Training_053" -> 53
    return int(case_id.split("_")[-1])


def is_lgg(case_id: str) -> bool:
    n = parse_case_num(case_id)
    return LGG_START <= n <= LGG_END


def case_stats(case_dir: Path) -> Optional[dict]:
    npz_files = sorted(case_dir.glob("*.npz"))
    if not npz_files:
        return None

    total_pixels = 0
    tumor_pixels = 0
    ed_pixels = 0  # label 2
    et_pixels = 0  # label 3 (original 4)
    tc_pixels = 0  # labels 1 or 3

    for f in npz_files:
        d = np.load(f, allow_pickle=False)
        y = remap_brats_labels(d["y"].astype(np.int64))

        total_pixels += y.size
        tumor_pixels += int((y > 0).sum())
        ed_pixels += int((y == 2).sum())
        et_pixels += int((y == 3).sum())
        tc_pixels += int(((y == 1) | (y == 3)).sum())

    tumor = max(tumor_pixels, 1)
    burden = tumor_pixels / max(total_pixels, 1)

    return {
        "case_id": case_dir.name,
        "grade": "LGG" if is_lgg(case_dir.name) else "HGG",
        "n_slices": len(npz_files),
        "burden": float(burden),
        "frac_ed": float(ed_pixels / tumor),
        "frac_et": float(et_pixels / tumor),
        "frac_tc": float(tc_pixels / tumor),
    }


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    dst.symlink_to(src)


def summarize(name: str, cases: List[str], stats: Dict[str, dict]) -> dict:
    if not cases:
        return {"client": name, "n_cases": 0}

    burdens = [stats[c]["burden"] for c in cases]
    et = [stats[c]["frac_et"] for c in cases]
    ed = [stats[c]["frac_ed"] for c in cases]
    tc = [stats[c]["frac_tc"] for c in cases]
    grades = [stats[c]["grade"] for c in cases]

    return {
        "client": name,
        "n_cases": len(cases),
        "n_LGG": int(sum(g == "LGG" for g in grades)),
        "n_HGG": int(sum(g == "HGG" for g in grades)),
        "mean_burden": float(np.mean(burdens)),
        "mean_frac_et": float(np.mean(et)),
        "mean_frac_ed": float(np.mean(ed)),
        "mean_frac_tc": float(np.mean(tc)),
    }


# -----------------------
# Splitting logic
# -----------------------
def split_cases_2clients(
    all_cases: List[str],
    stats: Dict[str, dict],
    ratio_a: float,
    seed: int,
    stratify_by_grade: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Split into client A and client B with sizes approx ratio_a : (1-ratio_a).
    If stratify_by_grade=True, do the split separately for HGG and LGG then combine.
    """
    rng = random.Random(seed)

    def split_list(items: List[str], r: float) -> Tuple[List[str], List[str]]:
        items = items[:]
        rng.shuffle(items)
        n_a = int(round(len(items) * r))
        a = sorted(items[:n_a])
        b = sorted(items[n_a:])
        return a, b

    if not stratify_by_grade:
        return split_list(sorted(all_cases), ratio_a)

    lgg = [c for c in all_cases if stats[c]["grade"] == "LGG"]
    hgg = [c for c in all_cases if stats[c]["grade"] == "HGG"]

    a_lgg, b_lgg = split_list(sorted(lgg), ratio_a)
    a_hgg, b_hgg = split_list(sorted(hgg), ratio_a)

    a = sorted(a_lgg + a_hgg)
    b = sorted(b_lgg + b_hgg)
    return a, b


def write_partition(
    out_dir: Path,
    client_a_cases: List[str],
    client_b_cases: List[str],
    stats: Dict[str, dict],
    make_symlinks: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    client_map = {
        "client_0": client_a_cases,
        "client_1": client_b_cases,
    }

    # Save mapping
    (out_dir / "client_map.json").write_text(json.dumps(client_map, indent=2))
    (out_dir / "client_0_cases.txt").write_text("\n".join(client_a_cases) + "\n")
    (out_dir / "client_1_cases.txt").write_text("\n".join(client_b_cases) + "\n")

    # Summaries
    summ = [
        summarize("client_0", client_a_cases, stats),
        summarize("client_1", client_b_cases, stats),
    ]
    (out_dir / "client_summary.json").write_text(json.dumps(summ, indent=2))

    print(f"\n=== Client summary for {out_dir.name} ===")
    for s in summ:
        print(
            f"{s['client']}: n={s['n_cases']} (HGG={s.get('n_HGG',0)} LGG={s.get('n_LGG',0)}) "
            f"burden={s['mean_burden']:.4f} ET={s['mean_frac_et']:.3f} "
            f"ED={s['mean_frac_ed']:.3f} TC={s['mean_frac_tc']:.3f}"
        )

    # Optional: symlink data for Flower clients
    if make_symlinks:
        base = out_dir / "client_data"
        for cid, cases in enumerate([client_a_cases, client_b_cases]):
            for case_id in cases:
                src_case = TRAIN_DIR / case_id
                dst_case = base / f"client_{cid}" / "train" / case_id
                dst_case.mkdir(parents=True, exist_ok=True)
                for f in src_case.glob("*.npz"):
                    ensure_symlink(f, dst_case / f.name)

        print(f"Symlinked client train data under: {base}")

    print(f"Wrote outputs to: {out_dir}")


def build_stats(train_dir: Path) -> Dict[str, dict]:
    case_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    if not case_dirs:
        raise RuntimeError(f"No case dirs under {train_dir}")

    stats: Dict[str, dict] = {}
    for cd in case_dirs:
        s = case_stats(cd)
        if s is not None:
            stats[cd.name] = s
    return stats


# -----------------------
# Main
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--base_out", type=Path, default=DEFAULT_BASE_OUT)

    ap.add_argument(
        "--scheme",
        choices=["50_50", "70_30", "both"],
        default="both",
        help="Which partition(s) to generate",
    )
    ap.add_argument(
        "--stratify_by_grade",
        action="store_true",
        help="Stratify split separately for HGG/LGG (recommended).",
    )
    ap.add_argument(
        "--no_stratify_by_grade",
        action="store_true",
        help="Disable stratification and split purely at random.",
    )
    ap.add_argument(
        "--no_symlinks",
        action="store_true",
        help="Do not create client_data symlinks (faster, smaller).",
    )

    args = ap.parse_args()

    seed = args.seed
    make_symlinks = not args.no_symlinks
    stratify = args.stratify_by_grade and not args.no_stratify_by_grade

    random.seed(seed)
    np.random.seed(seed)

    stats = build_stats(TRAIN_DIR)
    all_cases = sorted(stats.keys())
    lgg_cases = [c for c in all_cases if stats[c]["grade"] == "LGG"]
    hgg_cases = [c for c in all_cases if stats[c]["grade"] == "HGG"]

    print(f"Train cases total: {len(all_cases)} | HGG: {len(hgg_cases)} | LGG: {len(lgg_cases)}")
    print(f"Stratify by grade: {stratify} | Symlinks: {make_symlinks} | Seed: {seed}")

    def run_one(ratio_a: float, out_name: str) -> None:
        a, b = split_cases_2clients(
            all_cases=all_cases,
            stats=stats,
            ratio_a=ratio_a,
            seed=seed,
            stratify_by_grade=stratify,
        )
        out_dir = args.base_out / out_name
        write_partition(out_dir, a, b, stats, make_symlinks)

    if args.scheme in ("50_50", "both"):
        run_one(0.50, "federated_clients_2_50_50")

    if args.scheme in ("70_30", "both"):
        run_one(0.70, "federated_clients_2_70_30")

    print("\nDone.")


if __name__ == "__main__":
    main()
