#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

def is_case_dir(p: Path) -> bool:
    # BraTS20_Training_001 style folders
    return p.is_dir() and p.name.startswith("BraTS20_Training_")

def write_list(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, default="/Users/basiakoch/Downloads/BraTS2020_2D_png")
    ap.add_argument("--out_root", type=str, default="/Users/basiakoch/Downloads/BraTS2020_2D_png_split")
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["copy", "move"], default="copy",
                    help="copy = keep original dataset intact; move = relocate folders (faster, destructive).")
    args = ap.parse_args()

    # sanity
    s = args.train_frac + args.val_frac + args.test_frac
    if abs(s - 1.0) > 1e-6:
        raise SystemExit(f"Fractions must sum to 1.0, got {s}")

    in_root = Path(args.in_root).expanduser()
    out_root = Path(args.out_root).expanduser()

    if not in_root.exists():
        raise SystemExit(f"in_root not found: {in_root}")

    case_dirs = sorted([p for p in in_root.iterdir() if is_case_dir(p)])
    if not case_dirs:
        raise SystemExit(f"No case folders found under {in_root}")

    rng = random.Random(args.seed)
    rng.shuffle(case_dirs)

    n = len(case_dirs)
    n_train = max(1, int(n * args.train_frac))
    n_val = max(1, int(n * args.val_frac))
    # test gets the remainder to ensure full coverage
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_train = max(1, n - n_val - n_test)

    train_cases = case_dirs[:n_train]
    val_cases = case_dirs[n_train:n_train + n_val]
    test_cases = case_dirs[n_train + n_val:]

    print(f"Found {n} cases")
    print(f"Split: train={len(train_cases)} val={len(val_cases)} test={len(test_cases)}")
    print(f"Output: {out_root}")

    # create split folders
    for split in ["train", "val", "test"]:
        (out_root / split).mkdir(parents=True, exist_ok=True)

    op = shutil.move if args.mode == "move" else shutil.copytree

    def do_one(split: str, cases: list[Path]) -> None:
        for c in cases:
            dst = out_root / split / c.name
            if dst.exists():
                raise SystemExit(f"Destination already exists: {dst} (delete out_root or choose a new out_root)")
            if args.mode == "copy":
                shutil.copytree(c, dst)
            else:
                shutil.move(str(c), str(dst))

    do_one("train", train_cases)
    do_one("val", val_cases)
    do_one("test", test_cases)

    # save case ID lists (super useful later)
    write_list(out_root / "train_cases.txt", [p.name for p in train_cases])
    write_list(out_root / "val_cases.txt", [p.name for p in val_cases])
    write_list(out_root / "test_cases.txt", [p.name for p in test_cases])

    print("Done.")
    print(f"Wrote case lists to:\n  {out_root/'train_cases.txt'}\n  {out_root/'val_cases.txt'}\n  {out_root/'test_cases.txt'}")

if __name__ == "__main__":
    main()
