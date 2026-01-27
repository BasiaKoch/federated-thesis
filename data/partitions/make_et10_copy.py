#!/usr/bin/env python3
import argparse
import hashlib
import os
import shutil
from pathlib import Path

import numpy as np


IMG_KEYS = ["image", "img", "x", "X"]
MSK_KEYS = ["mask", "y", "Y", "seg", "label", "labels"]


def find_key(d, keys):
    for k in keys:
        if k in d:
            return k
    return None


def has_et(mask: np.ndarray) -> bool:
    return np.any(mask == 4)


def stable_shuffle(paths, seed: int):
    """
    Deterministic shuffle independent of Python hash randomization.
    Use md5(path) + seed to produce a stable order.
    """
    def score(p: Path):
        h = hashlib.md5((str(p) + f"::{seed}").encode("utf-8")).hexdigest()
        return h
    return sorted(paths, key=score)


def copy_tree_structure(src: Path, dst: Path):
    if dst.exists():
        raise FileExistsError(f"Output path already exists: {dst}")
    dst.mkdir(parents=True, exist_ok=False)

    # Copy everything except .npz will be handled separately (so we can edit masks)
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            sp = Path(root) / f
            dp = dst / rel / f
            if sp.suffix == ".npz":
                continue
            shutil.copy2(sp, dp)


def process_npz(src_path: Path, dst_path: Path, rewrite_et: bool):
    d = np.load(str(src_path), allow_pickle=False)

    # Load all arrays so we can re-save with same keys
    data = {k: d[k] for k in d.files}

    mkey = find_key(data, MSK_KEYS)
    if mkey is None:
        raise KeyError(f"No mask key found in {src_path}. Keys={list(data.keys())}")

    mask = data[mkey]

    # Ensure we operate on 2D mask; if (1,H,W) or (H,W,1), squeeze it
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            # if weird, take first channel
            mask = mask[..., 0]

    if rewrite_et:
        # map ET label 4 -> 1
        mask = mask.copy()
        mask[mask == 4] = 1

    data[mkey] = mask.astype(np.uint8)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_path, **data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source dataset root (contains client_0, client_1)")
    ap.add_argument("--dst", required=True, help="Output dataset root to create")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--keep_et_frac",
        type=float,
        default=0.10,
        help="Fraction of ET+ cases to keep ET unchanged in client_0/train",
    )
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not (src / "client_0").exists() or not (src / "client_1").exists():
        raise FileNotFoundError(f"Expected client_0 and client_1 under {src}")

    if not (0.0 <= args.keep_et_frac <= 1.0):
        raise ValueError("--keep_et_frac must be in [0,1]")

    print(f"Copying non-npz files + folder structure:\n  {src}\n-> {dst}")
    copy_tree_structure(src, dst)

    # Collect all .npz from source
    all_npz = sorted(src.rglob("*.npz"))
    if not all_npz:
        raise FileNotFoundError(f"No .npz found under {src}")

    # Identify client_0/train npz that are ET+
    c0_train = sorted((src / "client_0" / "train").rglob("*.npz"))
    et_pos = []
    et_neg = []

    for p in c0_train:
        d = np.load(str(p), allow_pickle=False)
        mkey = find_key(d, MSK_KEYS)
        if mkey is None:
            raise KeyError(f"No mask key in {p}, keys={list(d.files)}")
        mask = d[mkey]
        # squeeze possible singleton dims
        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask = mask[0]
            elif mask.shape[-1] == 1:
                mask = mask[..., 0]
            else:
                mask = mask[..., 0]
        if has_et(mask):
            et_pos.append(p)
        else:
            et_neg.append(p)

    et_pos = stable_shuffle(et_pos, args.seed)

    keep_n = int(round(len(et_pos) * args.keep_et_frac))
    keep_set = set(et_pos[:keep_n])          # keep ET unchanged
    rewrite_set = set(et_pos[keep_n:])       # rewrite ET (4->1) for these

    print("client_0/train ET+ summary:")
    print(f"  total train files: {len(c0_train)}")
    print(f"  ET+ files:         {len(et_pos)}")
    print(f"  ET- files:         {len(et_neg)}")
    print(f"  keep ET unchanged: {len(keep_set)} (~{args.keep_et_frac*100:.1f}%)")
    print(f"  rewrite ET->1:     {len(rewrite_set)} (~{(1-args.keep_et_frac)*100:.1f}%)")

    # Now copy all .npz, editing only those in rewrite_set
    edited = 0
    for sp in all_npz:
        rel = sp.relative_to(src)
        dp = dst / rel

        rewrite = (sp in rewrite_set)
        process_npz(sp, dp, rewrite_et=rewrite)
        if rewrite:
            edited += 1

    print(f"Done. Edited masks (ET->1) for {edited} files in client_0/train.")
    print(f"Output dataset: {dst}")


if __name__ == "__main__":
    main()
