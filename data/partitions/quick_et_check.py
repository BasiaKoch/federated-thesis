#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import argparse

def has_label4(npz_path: Path) -> bool:
    d = np.load(str(npz_path), allow_pickle=False)
    if "mask" in d:
        m = d["mask"]
    elif "seg" in d:
        m = d["seg"]
    elif "y" in d:
        m = d["y"]
    else:
        raise KeyError(f"No mask key in {npz_path}, keys={list(d.keys())}")
    return np.any(m == 4)

def count_split(root: Path, split: str):
    files = sorted((root / split).rglob("*.npz"))
    total = len(files)
    et_pos = sum(has_label4(p) for p in files)
    return total, et_pos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root containing client_0, client_1")
    ap.add_argument("--client", default="client_1")
    args = ap.parse_args()

    client_root = Path(args.root) / args.client
    for split in ["train", "val", "test"]:
        total, et_pos = count_split(client_root, split)
        pct = (100.0 * et_pos / total) if total else 0.0
        print(f"{args.client} {split}: total={total} ET+={et_pos} ET%={pct:.2f}%")

if __name__ == "__main__":
    main()
