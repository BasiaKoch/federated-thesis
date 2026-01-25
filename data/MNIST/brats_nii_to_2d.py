#!/usr/bin/env python3
"""
BraTS2020: 3D NIfTI (.nii/.nii.gz) -> 2D slice PNGs,
preserving folder structure and preserving original names in output.

Example output:
IN:  .../BraTS20_Training_001_t1.nii.gz
OUT: .../BraTS20_Training_001_t1.nii/
         BraTS20_Training_001_t1.nii_slice0000.png
         BraTS20_Training_001_t1.nii_slice0001.png
         ...

Run:
python brats_nii_to_2d_keepnames.py \
  --in_root "/Users/basiakoch/Downloads/BraTS2020 Dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" \
  --out_root "/Users/basiakoch/Downloads/BraTS2020_2D"
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import nibabel as nib
import imageio.v2 as imageio


# Match modality files anywhere under in_root
# Works for .nii and .nii.gz
BRATS_FILE_RE = re.compile(
    r"^(BraTS20_Training_\d{3})_(flair|t1ce|t1|t2|seg)\.nii(\.gz)?$",
    re.IGNORECASE,
)


def normalize_to_uint8(vol: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Robust normalize a 3D MRI volume to uint8 [0,255]."""
    vol = vol.astype(np.float32)
    finite = np.isfinite(vol)
    if not finite.any():
        return np.zeros_like(vol, dtype=np.uint8)

    lo, hi = np.percentile(vol[finite], [p_low, p_high])
    if hi <= lo:
        lo = float(np.min(vol[finite]))
        hi = float(np.max(vol[finite]) + 1e-6)

    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-8)
    return (vol * 255.0).round().astype(np.uint8)


def nii_display_name(path: Path) -> str:
    """
    Return filename ending in .nii (even if input is .nii.gz).
    Example:
      BraTS20_Training_001_t1.nii.gz -> BraTS20_Training_001_t1.nii
      BraTS20_Training_001_t1.nii    -> BraTS20_Training_001_t1.nii
    """
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-3]  # strip only ".gz"
    return name


def find_brats_nii_files(in_root: Path) -> list[Path]:
    nii_files = []
    for p in in_root.rglob("*.nii*"):
        if BRATS_FILE_RE.match(p.name):
            nii_files.append(p)
    return sorted(nii_files)


def load_volume(path: Path, canonical: bool) -> np.ndarray:
    img = nib.load(str(path))
    if canonical:
        img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {data.shape} for {path}")
    return data


def save_slices_png(
    vol_3d: np.ndarray,
    out_dir: Path,
    base_name_with_nii: str,
    axis: int,
    is_mask: bool,
    p_low: float,
    p_high: float,
) -> int:
    """
    Saves slices as:
      {base_name_with_nii}_slice0000.png, ...
    Returns number of slices saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_mask:
        # Keep label IDs (typically 0,1,2,4). Save as uint8.
        vol_u8 = np.clip(vol_3d, 0, 255).astype(np.uint8)
    else:
        vol_u8 = normalize_to_uint8(vol_3d, p_low=p_low, p_high=p_high)

    # Move slice axis to last -> (H, W, S)
    vol_hw_s = np.moveaxis(vol_u8, axis, -1)
    num_slices = vol_hw_s.shape[-1]

    for i in range(num_slices):
        sl = vol_hw_s[..., i]
        out_path = out_dir / f"{base_name_with_nii}_slice{i:04d}.png"
        imageio.imwrite(out_path, sl)

    return num_slices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="0=sag, 1=cor, 2=axial")
    ap.add_argument("--no_canonical", action="store_true", help="Do not reorient to closest canonical orientation")
    ap.add_argument("--p_low", type=float, default=0.5)
    ap.add_argument("--p_high", type=float, default=99.5)
    args = ap.parse_args()

    in_root = Path(args.in_root).expanduser()
    out_root = Path(args.out_root).expanduser()
    canonical = not args.no_canonical

    if not in_root.exists():
        raise FileNotFoundError(f"in_root does not exist: {in_root}")

    nii_files = find_brats_nii_files(in_root)
    print(f"Found {len(nii_files)} BraTS NIfTI files under: {in_root}")

    if not nii_files:
        # Print a hint: show a few nii files found (even if names didn't match)
        any_nii = sorted(in_root.rglob("*.nii*"))
        print(f"Total .nii/.nii.gz files found (any name): {len(any_nii)}")
        print("First 10 .nii* paths:")
        for p in any_nii[:10]:
            print("  ", p)
        raise SystemExit(
            "\nNo files matched expected pattern like: BraTS20_Training_001_flair.nii(.gz)\n"
            "Check your folder or naming."
        )

    total_slices = 0
    for p in nii_files:
        m = BRATS_FILE_RE.match(p.name)
        assert m is not None
        modality = m.group(2).lower()
        is_mask = (modality == "seg")

        # Preserve structure: output goes to OUT_ROOT / relative_parent / <filename>.nii /
        rel_parent = p.parent.relative_to(in_root)
        base_with_nii = nii_display_name(p)                 # e.g. "..._t1.nii"
        out_dir = out_root / rel_parent / base_with_nii     # folder named "..._t1.nii"

        vol = load_volume(p, canonical=canonical)
        n = save_slices_png(
            vol_3d=vol,
            out_dir=out_dir,
            base_name_with_nii=base_with_nii,
            axis=args.axis,
            is_mask=is_mask,
            p_low=args.p_low,
            p_high=args.p_high,
        )
        total_slices += n
        print(f"[OK] {p.name} -> {out_dir}  ({n} slices)")

    print(f"Done. Wrote {total_slices} slices total to: {out_root}")


if __name__ == "__main__":
    main()
