"""
Split your exported BraTS 2D .npz slices into train/val/test at the *patient (case) level*
and copy them into: brats2020_top10_slices_split_npz/{train,val,test}/<case_id>/*.npz

Why patient-level? So slices from the same 3D subject never leak across splits.

Input (current):
  brats2020_top10_slices_npz/
    BraTS20_Training_XXX/
      slice_*.npz

Output:
  brats2020_top10_slices_split_npz/
    train/<case_id>/*.npz
    val/<case_id>/*.npz
    test/<case_id>/*.npz

Also writes:
  brats2020_top10_slices_split_npz/split.json
  brats2020_top10_slices_split_npz/split.csv
"""

from __future__ import annotations

import json
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


# ----------------------------
# Config (edit if you want)
# ----------------------------

IN_DIR = Path("/Users/basiakoch/Downloads/brats2020_top10_slices_npz")
OUT_DIR = Path("/Users/basiakoch/Downloads/brats2020_top10_slices_split_npz")

SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10

# Choose how to write files:
#   "copy"     -> duplicates data
#   "hardlink" -> saves disk space, but only works on same filesystem
#   "symlink"  -> saves space, but can break if you move folders
MODE = "hardlink"  # "copy" | "hardlink" | "symlink"


# ----------------------------
# Helpers
# ----------------------------

def _list_cases(in_dir: Path) -> List[Path]:
    if not in_dir.exists():
        raise FileNotFoundError(f"IN_DIR does not exist: {in_dir}")
    cases = sorted([p for p in in_dir.iterdir() if p.is_dir()])
    if not cases:
        raise RuntimeError(f"No case folders found under: {in_dir}")
    return cases


def _split_cases(case_ids: List[str], seed: int) -> Dict[str, List[str]]:
    if abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) > 1e-9:
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must sum to 1.0")

    rng = random.Random(seed)
    ids = case_ids[:]
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    # ensure totals match n
    n_test = n - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    assert len(train_ids) + len(val_ids) + len(test_ids) == n
    assert len(test_ids) == n_test

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    _ensure_dir(dst.parent)
    if dst.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        # hardlink saves space; falls back to copy if not supported
        try:
            dst.hardlink_to(src)
        except Exception:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        # relative symlink is nicer for portability
        rel = src.relative_to(src.anchor) if src.is_absolute() else src
        # safer: use absolute symlink (works even if cwd changes)
        try:
            dst.symlink_to(src)
        except Exception:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown MODE: {mode}")


def _copy_case(case_dir: Path, out_split_dir: Path, mode: str) -> int:
    """
    Copy/link all npz files from one case folder into output split folder.
    Returns number of files written.
    """
    files = sorted(case_dir.glob("*.npz"))
    if not files:
        return 0

    out_case_dir = out_split_dir / case_dir.name
    _ensure_dir(out_case_dir)

    written = 0
    for f in files:
        dst = out_case_dir / f.name
        _link_or_copy(f, dst, mode=mode)
        written += 1
    return written


def _write_split_metadata(out_dir: Path, split: Dict[str, List[str]]) -> None:
    # JSON (case-level)
    with open(out_dir / "split.json", "w", encoding="utf-8") as fp:
        json.dump(split, fp, indent=2, sort_keys=True)

    # CSV (one row per case)
    with open(out_dir / "split.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["case_id", "split"])
        for split_name, case_ids in split.items():
            for cid in case_ids:
                w.writerow([cid, split_name])


def main() -> None:
    cases = _list_cases(IN_DIR)
    case_ids = [c.name for c in cases]

    split = _split_cases(case_ids, seed=SEED)

    # Create output dirs
    _ensure_dir(OUT_DIR)
    for s in ["train", "val", "test"]:
        _ensure_dir(OUT_DIR / s)

    # Map name->Path for quick lookup
    case_map = {c.name: c for c in cases}

    counts = {}
    for s in ["train", "val", "test"]:
        total_files = 0
        for cid in split[s]:
            total_files += _copy_case(case_map[cid], OUT_DIR / s, mode=MODE)
        counts[s] = {"cases": len(split[s]), "files": total_files}

    _write_split_metadata(OUT_DIR, split)

    print("Done.")
    print("Input:", IN_DIR.resolve())
    print("Output:", OUT_DIR.resolve())
    print("Mode:", MODE)
    print("Counts:", counts)


if __name__ == "__main__":
    main()
