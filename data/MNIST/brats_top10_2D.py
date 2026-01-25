"""
Extract the 10 most informative 2D axial slices per BraTS2020 3D case.

"Most informative" = slices with the largest tumor area in the *3D segmentation mask*.
For each selected slice index k:
  - image modalities are sliced at k
  - mask is sliced at k (same index)
  - saved as a compressed .npz with:
        x: (C, H, W) float32  where C=4 (t1, t1ce, t2, flair)
        y: (H, W)   uint8     (labels kept as {0,1,2,4} by default)

Dataset root you gave (note the spaces) is handled via pathlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# You need nibabel for .nii / .nii.gz
#   pip install nibabel
import nibabel as nib


# ----------------------------
# Configuration
# ----------------------------

BRA_TS_ROOT = Path(
    "/Users/basiakoch/Downloads/BraTS2020 Dataset /BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
)
OUT_DIR = Path("/Users/basiakoch/Downloads/brats2020_top10_slices_npz")  # change if you want



# ----------------------------
# Helpers
# ----------------------------

def _find_case_files(case_dir: Path) -> Dict[str, Path]:
    """
    BraTS case directory typically contains files like:
      BraTS20_Training_XXX_flair.nii.gz
      BraTS20_Training_XXX_t1.nii.gz
      BraTS20_Training_XXX_t1ce.nii.gz
      BraTS20_Training_XXX_t2.nii.gz
      BraTS20_Training_XXX_seg.nii.gz

    Returns a dict with keys: flair, t1, t1ce, t2, seg
    Raises ValueError if anything is missing.
    """
    # Robust glob: look for *_{mod}.nii* inside the folder
    def pick_one(pattern: str) -> Path:
        hits = sorted(case_dir.glob(pattern))
        if len(hits) != 1:
            raise ValueError(f"Expected exactly 1 match for {pattern} in {case_dir}, got {len(hits)}: {hits}")
        return hits[0]

    files = {
        "flair": pick_one("*_flair.nii*"),
        "t1":    pick_one("*_t1.nii*"),
        "t1ce":  pick_one("*_t1ce.nii*"),
        "t2":    pick_one("*_t2.nii*"),
        "seg":   pick_one("*_seg.nii*"),
    }
    return files


def _load_nii(path: Path) -> np.ndarray:
    """Load NIfTI as a numpy array (float32)."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)  # (H, W, D) usually
    return data


def _topk_informative_slices_from_mask(
    mask3d: np.ndarray,
    k: int = 10,
    axis: int = 2,
    require_tumor: bool = True,
) -> List[int]:
    """
    Score each slice by tumor pixel count (mask > 0) along the chosen axis (default axial: axis=2).
    Return the indices of top-k slices sorted from most->less informative.

    If require_tumor=True:
      - only consider slices where tumor_pixels > 0
      - returns fewer than k indices if not enough tumor slices exist
    """
    if mask3d.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask3d.shape}")

    # Move axis of slicing to last dim for easy iteration
    m = np.moveaxis(mask3d, axis, -1)  # (H, W, D_slices)
    # Tumor pixels per slice
    tumor_counts = np.sum(m > 0, axis=(0, 1))  # (D_slices,)

    if require_tumor:
        valid = np.where(tumor_counts > 0)[0]
        if valid.size == 0:
            return []
        # Sort valid slices by tumor count desc
        valid_sorted = valid[np.argsort(tumor_counts[valid])[::-1]]
        return valid_sorted[:k].tolist()
    else:
        all_sorted = np.argsort(tumor_counts)[::-1]
        return all_sorted[:k].tolist()


def _stack_modalities_slice(
    vols: Dict[str, np.ndarray],
    slice_idx: int,
    axis: int = 2,
    modalities: Tuple[str, ...] = ("t1", "t1ce", "t2", "flair"),
) -> np.ndarray:
    """
    vols[mod] is 3D (H,W,D). Extract slice at slice_idx along axis, and stack into (C,H,W).
    """
    slices = []
    for mod in modalities:
        v = vols[mod]
        v_m = np.moveaxis(v, axis, -1)          # (H, W, D)
        sl = v_m[:, :, slice_idx]               # (H, W)
        slices.append(sl)
    x = np.stack(slices, axis=0).astype(np.float32)  # (C,H,W)
    return x


def _extract_mask_slice(mask3d: np.ndarray, slice_idx: int, axis: int = 2) -> np.ndarray:
    m = np.moveaxis(mask3d, axis, -1)      # (H,W,D)
    y = m[:, :, slice_idx].astype(np.uint8)
    return y


def _zscore_per_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Simple per-channel z-score normalization on nonzero voxels (common for MRI).
    x: (C,H,W)
    """
    x_out = x.copy()
    for c in range(x.shape[0]):
        xc = x[c]
        nz = xc != 0
        if np.any(nz):
            mu = float(xc[nz].mean())
            sd = float(xc[nz].std())
            x_out[c] = (xc - mu) / (sd + eps)
        else:
            x_out[c] = xc
    return x_out


def extract_topk_slices_per_case(
    brats_root: Path,
    out_dir: Path,
    k: int = 10,
    axis: int = 2,
    normalize: bool = True,
    require_tumor: bool = True,
) -> None:
    """
    Iterates all case folders under brats_root, extracts top-k informative slices,
    and writes them to out_dir/<case_id>/slice_<idx>.npz
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in brats_root.iterdir() if p.is_dir()])
    if not case_dirs:
        raise RuntimeError(f"No case folders found under: {brats_root}")

    print(f"Found {len(case_dirs)} case folders under {brats_root}")

    for case_dir in case_dirs:
        case_id = case_dir.name
        try:
            files = _find_case_files(case_dir)
        except ValueError as e:
            print(f"[SKIP] {case_id}: {e}")
            continue

        # Load seg first (cheap scoring vs loading all modalities blindly)
        seg3d = _load_nii(files["seg"])

        top_slices = _topk_informative_slices_from_mask(
            seg3d, k=k, axis=axis, require_tumor=require_tumor
        )

        if len(top_slices) == 0:
            print(f"[SKIP] {case_id}: no tumor slices found (require_tumor={require_tumor})")
            continue

        # Load modalities only if we have slices to export
        vols = {
            "t1": _load_nii(files["t1"]),
            "t1ce": _load_nii(files["t1ce"]),
            "t2": _load_nii(files["t2"]),
            "flair": _load_nii(files["flair"]),
        }

        case_out = out_dir / case_id
        case_out.mkdir(parents=True, exist_ok=True)

        for rank, sl_idx in enumerate(top_slices):
            x2d = _stack_modalities_slice(vols, sl_idx, axis=axis)
            y2d = _extract_mask_slice(seg3d, sl_idx, axis=axis)

            if normalize:
                x2d = _zscore_per_channel(x2d)

            # Save
            np.savez_compressed(
                case_out / f"slice_{sl_idx:03d}_rank{rank:02d}.npz",
                x=x2d.astype(np.float32),     # (4,H,W)
                y=y2d.astype(np.uint8),       # (H,W) labels {0,1,2,4}
                slice_idx=np.int32(sl_idx),
                case_id=case_id,
            )

        print(f"[OK] {case_id}: saved {len(top_slices)} slices -> {case_out}")


# ----------------------------
# Optional: build a PyTorch Dataset over the saved 2D slices
# ----------------------------

class Brats2DTopKDataset:
    """
    Simple dataset that reads the exported .npz files.
    Produces (x, y) where:
      x: float32 (C,H,W)
      y: uint8   (H,W)
    """

    def __init__(self, exported_root: Path):
        self.exported_root = exported_root
        self.items = sorted(exported_root.glob("*/*.npz"))
        if not self.items:
            raise RuntimeError(f"No .npz found under {exported_root}. Did you run extraction?")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        p = self.items[idx]
        data = np.load(p, allow_pickle=False)
        x = data["x"]  # (C,H,W)
        y = data["y"]  # (H,W)
        return x, y


if __name__ == "__main__":
    extract_topk_slices_per_case(
        brats_root=BRA_TS_ROOT,
        out_dir=OUT_DIR,
        k=10,
        axis=2,              # axial slicing
        normalize=True,
        require_tumor=True,  # only tumor-containing slices
    )

    # Quick sanity check
    ds = Brats2DTopKDataset(OUT_DIR)
    x0, y0 = ds[0]
    print("Example item shapes:", x0.shape, y0.shape, "unique labels:", np.unique(y0))
