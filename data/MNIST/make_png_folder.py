#!/usr/bin/env python3
"""
Convert your existing BraTS2020_2D output into a clean PNG-only tree in:
  /Users/basiakoch/Downloads/BraTS2020_2D_png

This assumes your current output looks like:
  /Users/basiakoch/Downloads/BraTS2020_2D/
    BraTS20_Training_143/
      BraTS20_Training_143_t1ce.nii/
        BraTS20_Training_143_t1ce.nii_slice0000.png
        ...

It copies (or converts if needed) slice images into the new folder, preserving structure:
  BraTS20_Training_143/
    BraTS20_Training_143_t1ce/
      BraTS20_Training_143_t1ce_slice0000.png
      ...

If your files are already PNGs, this is basically a fast restructure/copy.
"""

from pathlib import Path
import shutil

SRC_ROOT = Path("/Users/basiakoch/Downloads/BraTS2020_2D")
DST_ROOT = Path("/Users/basiakoch/Downloads/BraTS2020_2D_png")

def strip_nii_suffix(name: str) -> str:
    return name[:-4] if name.lower().endswith(".nii") else name

def main() -> None:
    if not SRC_ROOT.exists():
        raise SystemExit(f"Source folder not found: {SRC_ROOT}")

    # Find slice files that are already png
    png_files = list(SRC_ROOT.rglob("*.png"))
    if not png_files:
        raise SystemExit(
            f"No .png files found under {SRC_ROOT}.\n"
            "If your slices are not PNG, tell me what extensions you see (e.g. .npy, no suffix)."
        )

    for src in png_files:
        rel = src.relative_to(SRC_ROOT)
        parts = list(rel.parts)

        # Expected rel like: CaseID / <something ending .nii> / filename.png
        # We rewrite the middle folder to remove ".nii"
        if len(parts) >= 2:
            parts[1] = strip_nii_suffix(parts[1])

        dst = DST_ROOT.joinpath(*parts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    print(f"Done. Copied {len(png_files)} PNG slices to:\n  {DST_ROOT}")

if __name__ == "__main__":
    main()
