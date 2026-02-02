#!/usr/bin/env python3
"""
Re-extract test set from original BraTS volumes, using the same held-out test patients
already present in partitioned client test folders.

- Input: partition_out/client_{0,1}/test/<PATIENT_ID>/*.npz  (only used to discover patient IDs)
- Raw BraTS root: .../MICCAI_BraTS2020_TrainingData/<PATIENT_ID>/<PATIENT_ID>_t1.nii(.gz), etc.
- Output: out_dir/client_{0,1}/test/<PATIENT_ID>/<PATIENT_ID>_zZZZ.npz

Supports per-client slice policies to match training slice selection:
  --client_modes "max,p25"  -> client_0 gets max slice, client_1 gets p25 slice

Available modes: max, p75, p50, p25, boundary_upper, boundary_lower
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import nibabel as nib

MODALITY_SUFFIXES = {"t1": "t1", "t1ce": "t1ce", "t2": "t2", "flair": "flair", "seg": "seg"}
DEFAULT_MODALITY_ORDER = ["t1", "t1ce", "t2", "flair"]


def load_nii(path_no_gz: Path) -> np.ndarray:
    """Load .nii or .nii.gz as float32."""
    p = path_no_gz
    if not p.exists():
        p = Path(str(path_no_gz) + ".gz")
    if not p.exists():
        raise FileNotFoundError(f"Missing NIfTI: {path_no_gz} or {path_no_gz}.gz")
    return nib.load(str(p)).get_fdata(dtype=np.float32)


def load_case(case_dir: Path, modality_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vol, seg) where vol is (C,H,W,Z) and seg is (H,W,Z)."""
    vols = []
    for m in modality_order:
        p = case_dir / f"{case_dir.name}_{MODALITY_SUFFIXES[m]}.nii"
        vols.append(load_nii(p))
    vol = np.stack(vols, axis=0).astype(np.float32)

    seg_path = case_dir / f"{case_dir.name}_{MODALITY_SUFFIXES['seg']}.nii"
    seg = load_nii(seg_path).astype(np.int16)
    return vol, seg


def zscore_per_channel(vol: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score each channel using non-zero voxels only (BraTS common practice)."""
    out = vol.copy()
    for c in range(out.shape[0]):
        x = out[c]
        nz = x != 0
        if np.any(nz):
            mu = x[nz].mean()
            sd = x[nz].std()
            out[c] = (x - mu) / (sd + eps)
    return out


def tumor_area_per_slice(seg: np.ndarray) -> np.ndarray:
    """Whole-tumor area per axial slice: count of seg>0 over H,W for each z."""
    return (seg > 0).sum(axis=(0, 1)).astype(np.int64)


def choose_slice_index(tumor_area: np.ndarray, mode: str) -> Optional[int]:
    """
    Select slice index based on mode.

    Modes:
        - "max": slice with maximum tumor area
        - "p75", "p50", "p25": slice closest to that percentile of max area
        - "boundary_upper": first slice from top with tumor
        - "boundary_lower": first slice from bottom with tumor
    """
    positive_indices = np.where(tumor_area > 0)[0]

    if len(positive_indices) == 0:
        return None  # No tumor in this case

    if mode == "max":
        return int(np.argmax(tumor_area))

    if mode.startswith("p"):
        percentile = int(mode[1:]) / 100.0
        max_area = tumor_area.max()
        target_area = percentile * max_area

        # Find slice with tumor area closest to target
        positive_areas = tumor_area[positive_indices]
        closest_idx = np.argmin(np.abs(positive_areas - target_area))
        return int(positive_indices[closest_idx])

    if mode == "boundary_upper":
        return int(positive_indices[-1])  # Highest z with tumor

    if mode == "boundary_lower":
        return int(positive_indices[0])   # Lowest z with tumor

    raise ValueError(f"Unknown mode: {mode}")


def choose_max_slice(seg: np.ndarray) -> Optional[int]:
    """Legacy function - uses choose_slice_index with mode='max'."""
    area = tumor_area_per_slice(seg)
    return choose_slice_index(area, "max")


def extract_slice(vol: np.ndarray, seg: np.ndarray, z: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (C,H,W) image and (H,W) mask for slice z."""
    img2d = vol[:, :, :, z].astype(np.float32)
    msk2d = seg[:, :, z].astype(np.int16)
    return img2d, msk2d


def save_npz(out_path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), image=image, mask=mask)


def list_patient_ids_from_partition(partition_out: Path, client_idx: int, split: str = "test") -> Set[str]:
    """
    Discover patient IDs by listing directories:
      partition_out/client_{i}/test/<PATIENT_ID>/
    """
    base = partition_out / f"client_{client_idx}" / split
    if not base.exists():
        raise FileNotFoundError(f"Missing partition split folder: {base}")
    return {p.name for p in base.iterdir() if p.is_dir()}


def main():
    ap = argparse.ArgumentParser(description="Re-extract test set from BraTS NIfTI with per-client slice policies.")
    ap.add_argument("--partition_out", type=Path, required=True,
                    help="Existing partition output folder containing client_0/test and client_1/test.")
    ap.add_argument("--brats_root", type=Path, required=True,
                    help="Raw BraTS root that contains per-patient folders with NIfTI files.")
    ap.add_argument("--out_dir", type=Path, required=True,
                    help="Where to write test .npz outputs.")
    ap.add_argument("--clients", type=str, default="0,1", help="Comma-separated client indices to process (default: 0,1).")
    ap.add_argument("--client_modes", type=str, default=None,
                    help="Comma-separated slice modes per client (e.g., 'max,p25'). "
                         "If not specified, all clients use --slice_mode. "
                         "Available modes: max, p75, p50, p25, boundary_upper, boundary_lower")
    ap.add_argument("--slice_mode", type=str, default="max",
                    help="Default slice mode for all clients if --client_modes not specified (default: max).")
    ap.add_argument("--split", type=str, default="test", choices=["test", "val", "train"],
                    help="Which split to re-extract (default: test).")
    ap.add_argument("--modalities", type=str, default="t1,t1ce,t2,flair", help="Comma-separated modality order.")
    ap.add_argument("--no_zscore", action="store_true", help="Disable z-scoring.")
    ap.add_argument("--skip_no_tumor", action="store_true", help="Skip patients with empty segmentation.")
    args = ap.parse_args()

    modality_order = [m.strip() for m in args.modalities.split(",") if m.strip()]
    client_list = [int(x.strip()) for x in args.clients.split(",") if x.strip()]

    # Parse per-client slice modes
    if args.client_modes:
        client_modes_list = [m.strip() for m in args.client_modes.split(",") if m.strip()]
        if len(client_modes_list) != len(client_list):
            raise ValueError(f"--client_modes has {len(client_modes_list)} entries but --clients has {len(client_list)}. "
                           f"They must match.")
        client_modes = {cid: mode for cid, mode in zip(client_list, client_modes_list)}
    else:
        # All clients use the same default mode
        client_modes = {cid: args.slice_mode for cid in client_list}

    print("Slice modes per client:")
    for cid in client_list:
        print(f"  client_{cid}: {client_modes[cid]}")

    total_written = 0
    total_skipped = 0

    for client_idx in client_list:
        slice_mode = client_modes[client_idx]
        patient_ids = sorted(list(list_patient_ids_from_partition(args.partition_out, client_idx, split=args.split)))
        print(f"\nClient {client_idx} (mode={slice_mode}): found {len(patient_ids)} patients in {args.split} split.")

        for pid in patient_ids:
            case_dir = args.brats_root / pid
            if not case_dir.exists():
                print(f"  [SKIP] Raw case folder not found for {pid}: {case_dir}")
                total_skipped += 1
                continue

            try:
                vol, seg = load_case(case_dir, modality_order)
            except FileNotFoundError as e:
                print(f"  [SKIP] {pid}: {e}")
                total_skipped += 1
                continue

            if not args.no_zscore:
                vol = zscore_per_channel(vol)

            # Use per-client slice mode
            tumor_area = tumor_area_per_slice(seg)
            z = choose_slice_index(tumor_area, slice_mode)
            if z is None:
                msg = f"  [NO TUMOR] {pid}: segmentation empty"
                if args.skip_no_tumor:
                    print(msg + " -> skipping")
                    total_skipped += 1
                    continue
                else:
                    print(msg + " -> defaulting to mid slice")
                    z = seg.shape[-1] // 2

            img2d, msk2d = extract_slice(vol, seg, z)

            out_path = args.out_dir / f"client_{client_idx}" / args.split / pid / f"{pid}_z{z:03d}.npz"
            save_npz(out_path, img2d, msk2d)
            total_written += 1

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Written: {total_written} samples")
    print(f"Skipped: {total_skipped} samples")
    print(f"Output: {args.out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
