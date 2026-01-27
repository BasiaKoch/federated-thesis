import os
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np


DATA_ROOT = Path(os.environ.get(
    "BRATS_DATA_DIR",
    "/home/bk489/federated/federated-thesis/data/brats2020_top10_slices_split_npz"
))
TRAIN_DIR = DATA_ROOT / "train"

OUT_DIR = Path("/home/bk489/federated/federated-thesis/data/partitions/federated_clients_5_lgg_hgg")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

LGG_START = 260
LGG_END = 335  # inclusive

# Create per-client folder with symlinks to NPZ files
MAKE_SYMLINK_DATA = True


def remap_brats_labels(y: np.ndarray) -> np.ndarray:
    y2 = y.copy()
    y2[y2 == 4] = 3
    return y2


def parse_case_num(case_id: str) -> int:
    # "BraTS20_Training_053" -> 53
    return int(case_id.split("_")[-1])


def is_lgg(case_id: str) -> bool:
    n = parse_case_num(case_id)
    return LGG_START <= n <= LGG_END


def case_stats(case_dir: Path) -> dict:
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


def ensure_symlink(src: Path, dst: Path):
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


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    case_dirs = sorted([p for p in TRAIN_DIR.iterdir() if p.is_dir()])
    if not case_dirs:
        raise RuntimeError(f"No case dirs under {TRAIN_DIR}")

    stats = {}
    for cd in case_dirs:
        s = case_stats(cd)
        if s is not None:
            stats[cd.name] = s

    all_cases = sorted(stats.keys())
    lgg_cases = [c for c in all_cases if stats[c]["grade"] == "LGG"]
    hgg_cases = [c for c in all_cases if stats[c]["grade"] == "HGG"]

    print(f"Train cases total: {len(all_cases)} | HGG: {len(hgg_cases)} | LGG: {len(lgg_cases)}")

    remaining_hgg = set(hgg_cases)

    # Targets (you can tweak)
    # We'll keep client 2 = all LGG, and split HGG into 4 clients.
    n_hgg = len(hgg_cases)
    # roughly equal HGG clients
    t = n_hgg // 4
    targets_hgg = [t, t, t, n_hgg - 3*t]  # sum = n_hgg

    def pick_top_hgg(key: str, k: int) -> List[str]:
        cand = sorted(list(remaining_hgg), key=lambda c: stats[c][key], reverse=True)
        chosen = cand[:k]
        for c in chosen:
            remaining_hgg.remove(c)
        return chosen

    def pick_bottom_hgg(key: str, k: int) -> List[str]:
        cand = sorted(list(remaining_hgg), key=lambda c: stats[c][key], reverse=False)
        chosen = cand[:k]
        for c in chosen:
            remaining_hgg.remove(c)
        return chosen

    client_map = {}

    # Client 0: HGG ET-heavy
    client_map["client_0_HGG_ET_heavy"] = pick_top_hgg("frac_et", targets_hgg[0])

    # Client 1: HGG ED-heavy
    client_map["client_1_HGG_ED_heavy"] = pick_top_hgg("frac_ed", targets_hgg[1])

    # Client 2: LGG only (all)
    client_map["client_2_LGG_only"] = sorted(lgg_cases)

    # Client 3: HGG low-burden
    client_map["client_3_HGG_low_burden"] = pick_bottom_hgg("burden", targets_hgg[2])

    # Client 4: remaining HGG (mixed)
    client_map["client_4_HGG_mixed"] = sorted(list(remaining_hgg))
    remaining_hgg.clear()

    # Save mapping
    (OUT_DIR / "client_map.json").write_text(json.dumps(client_map, indent=2))
    for name, cases in client_map.items():
        (OUT_DIR / f"{name}_cases.txt").write_text("\n".join(cases) + "\n")

    # Summaries
    summ = [summarize(name, cases, stats) for name, cases in client_map.items()]
    (OUT_DIR / "client_summary.json").write_text(json.dumps(summ, indent=2))

    print("\n=== Client summary ===")
    for s in summ:
        print(
            f"{s['client']}: n={s['n_cases']} (HGG={s.get('n_HGG',0)} LGG={s.get('n_LGG',0)}) "
            f"burden={s['mean_burden']:.4f} ET={s['mean_frac_et']:.3f} ED={s['mean_frac_ed']:.3f} TC={s['mean_frac_tc']:.3f}"
        )

    # Optional: symlink data for Flower clients
    if MAKE_SYMLINK_DATA:
        base = OUT_DIR / "client_data"
        # Flower expects numeric client ids; keep stable mapping 0..4
        ordered = [
            "client_0_HGG_ET_heavy",
            "client_1_HGG_ED_heavy",
            "client_2_LGG_only",
            "client_3_HGG_low_burden",
            "client_4_HGG_mixed",
        ]
        for cid, name in enumerate(ordered):
            for case_id in client_map[name]:
                src_case = TRAIN_DIR / case_id
                dst_case = base / f"client_{cid}" / "train" / case_id
                dst_case.mkdir(parents=True, exist_ok=True)
                for f in src_case.glob("*.npz"):
                    ensure_symlink(f, dst_case / f.name)

        print(f"\nSymlinked client train data under: {base}")

    print("\nWrote outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
