#!/usr/bin/env python3
"""
Federated-learning dataset builder for BraTS2020 2D slices (from NIfTI).

Creates a partition with K clients where EACH client receives a different type
and strength of image corruption, simulating heterogeneous hospital scanners.
This multi-client corruption approach creates strong client drift that FedProx
can mitigate better than FedAvg.

Example commands
----------------
# Default 8 clients, per-client gradient corruption, central-band slices:
python make_fl_partition_brats2d_from_nii_noisy_client.py \
    --source_dir "/path/to/MICCAI_BraTS2020_TrainingData" \
    --output_dir  "./brats2d_8client_heterogeneous"

# No corruption (clean baseline):
python make_fl_partition_brats2d_from_nii_noisy_client.py \
    --source_dir "/path/to/MICCAI_BraTS2020_TrainingData" \
    --output_dir  "./brats2d_8client_clean" \
    --corruption_schedule none

Why this helps FedProx
----------------------
Each client's unique corruption shifts its local loss surface differently.
With many local epochs, clients drift in diverse directions from the global
model.  FedProx's proximal term (mu/2)||w - w_global||^2 penalises this
drift, keeping each client's updates closer to the consensus and stabilising
aggregation -- especially visible on worst-client Dice and on round-to-round
variance of the global metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports -- fail early with a clear message
# ---------------------------------------------------------------------------
try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel is required: pip install nibabel")

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    sys.exit("scipy is required: pip install scipy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODALITIES = ("flair", "t1", "t1ce", "t2")
MODALITY_SUFFIXES = {
    "flair": "flair",
    "t1": "t1",
    "t1ce": "t1ce",
    "t2": "t2",
}

# 7 corruption profiles (index 0 is clean, indices 1-7 are corrupted).
# Each profile simulates a different type of scanner degradation.
_CORRUPTION_PROFILES: List[Dict[str, Any]] = [
    # C0: Clean reference hospital
    {"label": "clean", "steps": []},
    # C1: Inhomogeneous B1 coil — bias field only
    {"label": "bias_only", "steps": ["bias"], "bias": 0.35},
    # C2: Old/noisy scanner — Gaussian noise only
    {"label": "noise_only", "steps": ["gauss"], "gauss_sigma": 0.20},
    # C3: Low-resolution protocol — blur + downscale
    {"label": "blur_downscale", "steps": ["blur", "scale"], "blur": 1.2, "scale": 0.70},
    # C4: Aging scanner — bias + noise
    {"label": "bias_noise", "steps": ["bias", "gauss"], "bias": 0.30, "gauss_sigma": 0.15},
    # C5: Poor quality overall — bias + noise + blur
    {"label": "bias_noise_blur", "steps": ["bias", "gauss", "blur"],
     "bias": 0.40, "gauss_sigma": 0.18, "blur": 0.8},
    # C6: Rural clinic — noise + blur + downscale
    {"label": "noise_blur_downscale", "steps": ["gauss", "blur", "scale"],
     "gauss_sigma": 0.22, "blur": 1.0, "scale": 0.75},
    # C7: Mobile scanner — full pipeline
    {"label": "full_pipeline", "steps": ["bias", "gauss", "blur", "scale"],
     "bias": 0.60, "gauss_sigma": 0.25, "blur": 1.3, "scale": 0.65},
]


def generate_corruption_profiles(num_clients: int) -> List[Dict[str, Any]]:
    """
    Generate per-client corruption profiles.

    Client 0 is always clean.  Clients 1..num_clients-1 cycle through
    the 7 corrupted profiles (indices 1-7 of _CORRUPTION_PROFILES).
    """
    profiles: List[Dict[str, Any]] = []
    # Client 0 is always clean
    profiles.append(dict(_CORRUPTION_PROFILES[0]))
    corrupted = _CORRUPTION_PROFILES[1:]  # 7 corrupted profiles
    for i in range(1, num_clients):
        profiles.append(dict(corrupted[(i - 1) % len(corrupted)]))
    return profiles


# =========================================================================
# Step 0 -- CLI
# =========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BraTS2020 NIfTI -> 2D-slice FL partition with per-client corruption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Paths
    p.add_argument("--source_dir", type=Path, required=True,
                   help="MICCAI_BraTS2020_TrainingData directory")
    p.add_argument("--output_dir", type=Path, required=True)

    # Partition
    p.add_argument("--num_clients", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--dirichlet_alpha", type=float, default=0.1,
                   help="Dirichlet concentration for non-IID label distribution "
                        "(lower = more heterogeneous)")
    p.add_argument("--min_patients_per_client", type=int, default=15,
                   help="Minimum patients per client; rebalance from largest if needed")

    # Slice selection
    p.add_argument("--slice_policy", choices=["central_band", "all"],
                   default="central_band")
    p.add_argument("--central_band", nargs=2, type=float, default=[0.35, 0.65],
                   metavar=("LOW", "HIGH"))
    p.add_argument("--keep_empty_slices", action="store_true", default=False)
    p.add_argument("--tumor_sampling", type=float, default=1.0,
                   help="Target proportion of tumor-containing slices (train only; "
                        "1.0 = tumor slices only)")

    # Normalization
    p.add_argument("--normalize", choices=["zscore", "none"],
                   default="zscore",
                   help="Normalization mode (always per-slice)")
    p.add_argument("--clip_percentiles", nargs=2, type=float, default=[0.5, 99.5],
                   metavar=("LOW", "HIGH"))

    # Corruption
    p.add_argument("--corruption_schedule",
                   choices=["per_client_gradient", "none"],
                   default="per_client_gradient",
                   help="Corruption schedule: per_client_gradient assigns a different "
                        "corruption profile to each client; none disables corruption")

    # Slice limits
    p.add_argument("--min_slices_per_patient", type=int, default=2)
    p.add_argument("--max_slices_per_patient", type=int, default=5)

    return p


# =========================================================================
# Step 1 -- NIfTI discovery + loading helpers
# =========================================================================

def _find_nii(directory: Path, suffix: str) -> Optional[Path]:
    """Find a NIfTI file with the given suffix, trying .nii then .nii.gz."""
    stem = f"{directory.name}_{suffix}"
    for ext in (".nii", ".nii.gz"):
        p = directory / (stem + ext)
        if p.exists():
            return p
    return None


def discover_patients(source_dir: Path) -> List[Path]:
    """Return sorted list of patient directories matching BraTS20_Training_*."""
    patients = sorted([
        d for d in source_dir.iterdir()
        if d.is_dir() and d.name.startswith("BraTS20_Training_")
    ])
    if not patients:
        sys.exit(f"No BraTS20_Training_* directories found in {source_dir}")
    return patients


def load_nii(path: Path) -> np.ndarray:
    """Load a NIfTI file as float32 numpy array."""
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def load_case_volumes(
    case_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all 4 modalities + seg for one patient.

    Returns
    -------
    vol : (4, H, W, Z) float32
    seg : (H, W, Z) int16
    """
    arrays: List[np.ndarray] = []
    for mod in MODALITIES:
        p = _find_nii(case_dir, MODALITY_SUFFIXES[mod])
        if p is None:
            raise FileNotFoundError(
                f"Missing {mod} for {case_dir.name}: expected "
                f"{case_dir.name}_{MODALITY_SUFFIXES[mod]}.nii[.gz]"
            )
        arrays.append(load_nii(p))

    seg_path = _find_nii(case_dir, "seg")
    if seg_path is None:
        raise FileNotFoundError(f"Missing seg for {case_dir.name}")
    seg = load_nii(seg_path).astype(np.int16)

    # Validate shapes
    ref_shape = arrays[0].shape
    for i, (mod, arr) in enumerate(zip(MODALITIES, arrays)):
        if arr.shape != ref_shape:
            raise ValueError(
                f"{case_dir.name}: {mod} shape {arr.shape} != flair shape {ref_shape}"
            )
    if seg.shape != ref_shape:
        raise ValueError(
            f"{case_dir.name}: seg shape {seg.shape} != flair shape {ref_shape}"
        )

    vol = np.stack(arrays, axis=0)  # (4, H, W, Z)
    return vol, seg


# =========================================================================
# Step 2 -- Per-patient tumor signature
# =========================================================================

def compute_patient_stats(seg: np.ndarray) -> Dict[str, float]:
    """Compute tumor-subregion fractions from a 3D segmentation volume."""
    total = float(seg.size)
    m = seg.astype(np.int32).ravel()

    wt = int(np.count_nonzero(m > 0))
    et = int(np.count_nonzero(m == 4))
    ed = int(np.count_nonzero(m == 2))
    ncr = int(np.count_nonzero(m == 1))

    return {
        "WT_voxels": wt,
        "ET_voxels": et,
        "ED_voxels": ed,
        "NCR_voxels": ncr,
        "WT_frac": wt / total,
        "ET_frac": et / (wt + 1),
        "ED_frac": ed / (wt + 1),
        "NCR_frac": ncr / (wt + 1),
        "total_voxels": int(total),
    }


def scan_all_patients(
    patient_dirs: List[Path],
) -> Dict[str, Dict[str, Any]]:
    """First pass: compute stats without keeping volumes in memory."""
    stats: Dict[str, Dict[str, Any]] = {}
    for i, pdir in enumerate(patient_dirs):
        pid = pdir.name
        print(f"\r  Scanning patients: {i+1}/{len(patient_dirs)} ({pid})", end="", flush=True)
        # Only load seg for stats
        seg_path = _find_nii(pdir, "seg")
        if seg_path is None:
            warnings.warn(f"Skipping {pid}: no seg file")
            continue
        seg = load_nii(seg_path).astype(np.int16)
        ps = compute_patient_stats(seg)
        ps["num_slices"] = seg.shape[-1]
        ps["dir"] = str(pdir)
        stats[pid] = ps
    print()
    return stats


# =========================================================================
# Step 3 -- Global test set (stratified by WT_frac quantiles)
# =========================================================================

def stratified_test_split(
    patient_ids: List[str],
    patient_stats: Dict[str, Dict[str, Any]],
    test_frac: float,
    rng: np.random.Generator,
    n_quantiles: int = 4,
) -> Tuple[List[str], List[str]]:
    """
    Split patients into (remaining, global_test) stratified by WT_frac quantiles.
    """
    wt_fracs = np.array([patient_stats[pid]["WT_frac"] for pid in patient_ids])
    bin_edges = np.quantile(wt_fracs, np.linspace(0, 1, n_quantiles + 1))
    # Assign each patient to a bin
    bins = np.digitize(wt_fracs, bin_edges[1:-1])  # 0..n_quantiles-1

    global_test: List[str] = []
    remaining: List[str] = []

    for b in range(n_quantiles):
        indices = [i for i, bb in enumerate(bins) if bb == b]
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * test_frac))
        for j, idx in enumerate(indices):
            if j < n_test:
                global_test.append(patient_ids[idx])
            else:
                remaining.append(patient_ids[idx])

    return remaining, global_test


# =========================================================================
# Step 4 -- Client assignment (Dirichlet, all clients equal + rebalancing)
# =========================================================================

def assign_clients(
    patient_ids: List[str],
    patient_stats: Dict[str, Dict[str, Any]],
    num_clients: int,
    rng: np.random.Generator,
    alpha: float = 0.1,
    min_patients_per_client: int = 15,
    n_quantiles: int = 4,
    max_retries: int = 20,
) -> Dict[int, List[str]]:
    """
    Assign patients to clients using Dirichlet over WT_frac quantile bins.

    All clients are treated equally (no special noisy-client reservation).
    After Dirichlet allocation, if any client has fewer than
    min_patients_per_client, patients are moved from the largest client.
    """
    ids = list(patient_ids)
    rng.shuffle(ids)

    wt_fracs = np.array([patient_stats[p]["WT_frac"] for p in ids])
    bin_edges = np.quantile(wt_fracs, np.linspace(0, 1, n_quantiles + 1))
    bins = np.digitize(wt_fracs, bin_edges[1:-1])  # 0..n_quantiles-1

    # Build bin -> patient list
    bin_patients: Dict[int, List[str]] = defaultdict(list)
    for i, pid in enumerate(ids):
        bin_patients[bins[i]].append(pid)

    best_assignment: Optional[Dict[int, List[str]]] = None
    best_min_count = 0

    for attempt in range(max_retries):
        assignment: Dict[int, List[str]] = {c: [] for c in range(num_clients)}

        trial_rng = np.random.default_rng(rng.integers(0, 2**31) + attempt)

        for b in range(n_quantiles):
            pids_in_bin = list(bin_patients.get(b, []))
            if not pids_in_bin:
                continue
            props = trial_rng.dirichlet([alpha] * num_clients)
            # Convert proportions to counts
            counts = np.round(props * len(pids_in_bin)).astype(int)
            # Fix rounding so counts sum to len(pids_in_bin)
            diff = len(pids_in_bin) - counts.sum()
            for d in range(abs(diff)):
                idx = d % num_clients
                counts[idx] += 1 if diff > 0 else -1
            counts = np.maximum(counts, 0)
            # Re-fix sum
            while counts.sum() < len(pids_in_bin):
                counts[trial_rng.integers(num_clients)] += 1
            while counts.sum() > len(pids_in_bin):
                pos = np.where(counts > 0)[0]
                counts[trial_rng.choice(pos)] -= 1

            trial_rng.shuffle(pids_in_bin)
            offset = 0
            for ci in range(num_clients):
                n = int(counts[ci])
                assignment[ci].extend(pids_in_bin[offset:offset + n])
                offset += n

        # Rebalance: steal from largest client until minimum is met
        for _ in range(len(ids)):  # safety bound
            sizes = {c: len(v) for c, v in assignment.items()}
            min_client = min(sizes, key=sizes.get)
            if sizes[min_client] >= min_patients_per_client:
                break
            max_client = max(sizes, key=sizes.get)
            if sizes[max_client] <= sizes[min_client] + 1:
                break  # cannot improve
            # Move one patient from largest to smallest
            patient = assignment[max_client].pop()
            assignment[min_client].append(patient)

        min_count = min(len(v) for v in assignment.values())
        if min_count >= min_patients_per_client:
            return assignment
        if min_count > best_min_count:
            best_min_count = min_count
            best_assignment = assignment

    warnings.warn(
        f"After {max_retries} attempts, min client has {best_min_count} patients "
        f"(target: {min_patients_per_client}). Proceeding with best attempt."
    )
    return best_assignment  # type: ignore[return-value]


# =========================================================================
# Step 5 -- Per-client train/val/test (patient-level)
# =========================================================================

def split_client_patients(
    patients: List[str],
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> Dict[str, List[str]]:
    """Split a client's patients into train/val/test."""
    pts = list(patients)
    rng.shuffle(pts)
    n = len(pts)
    n_test = max(1, round(n * test_frac)) if n > 2 else 0
    n_val = max(1, round(n * val_frac)) if n > 2 else 0
    # Ensure we leave at least 1 for train
    if n_test + n_val >= n:
        n_test = max(1, n // 3)
        n_val = max(1, n // 3)
        if n_test + n_val >= n:
            n_val = 0
            n_test = min(1, n - 1)

    return {
        "test": pts[:n_test],
        "val": pts[n_test:n_test + n_val],
        "train": pts[n_test + n_val:],
    }


# =========================================================================
# Step 6 -- Normalization helpers
# =========================================================================

def clip_volume(vol: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Clip each channel to [p_low, p_high] percentiles (computed on non-zero voxels)."""
    out = vol.copy()
    for c in range(out.shape[0]):
        x = out[c]
        nz = x[x != 0]
        if nz.size > 0:
            lo = np.percentile(nz, p_low)
            hi = np.percentile(nz, p_high)
            out[c] = np.clip(x, lo, hi)
    return out


def zscore_slice(slc: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score each channel of a single 2D slice (C, H, W)."""
    out = slc.copy()
    for c in range(out.shape[0]):
        x = out[c]
        nz = x != 0
        if np.any(nz):
            mu = x[nz].mean()
            sd = x[nz].std()
            out[c] = np.where(nz, (x - mu) / (sd + eps), 0.0)
    return out


# =========================================================================
# Step 6 (cont.) -- Slice selection
# =========================================================================

def select_slice_indices(
    seg_vol: np.ndarray,
    z_dim: int,
    vol: np.ndarray,
    policy: str,
    central_band: Tuple[float, float],
    keep_empty: bool,
    is_train: bool,
    tumor_sampling: float,
    min_slices: int,
    max_slices: int,
    rng: np.random.Generator,
) -> List[int]:
    """Determine which axial slice indices to keep for one patient."""
    # Candidate range
    if policy == "central_band":
        lo = int(math.floor(central_band[0] * z_dim))
        hi = int(math.ceil(central_band[1] * z_dim))
        candidates = list(range(lo, hi))
    else:
        candidates = list(range(z_dim))

    # Drop empty slices (< 1% non-zero across all modalities)
    if not keep_empty:
        kept: List[int] = []
        for z in candidates:
            brain_mask = np.any(vol[:, :, :, z] != 0, axis=0)  # (H, W)
            if brain_mask.mean() >= 0.01:
                kept.append(z)
        candidates = kept

    if not candidates:
        return []

    # Compute tumor area per candidate slice (used for ranking)
    tumor_area = {z: int(np.count_nonzero(seg_vol[:, :, z] > 0)) for z in candidates}
    tumor_slices = [z for z in candidates if tumor_area[z] > 0]
    nontumor_slices = [z for z in candidates if tumor_area[z] == 0]

    if is_train and tumor_slices:
        # Keep tumor slices, then optionally add non-tumor to reach ratio
        selected = list(tumor_slices)
        if tumor_sampling < 1.0 and nontumor_slices:
            n_tumor = len(tumor_slices)
            target_nontumor = int(
                n_tumor * (1.0 - tumor_sampling) / max(tumor_sampling, 1e-9)
            )
            target_nontumor = min(target_nontumor, len(nontumor_slices))
            if target_nontumor > 0:
                rng.shuffle(nontumor_slices)
                selected.extend(nontumor_slices[:target_nontumor])
    else:
        # Val/test: prefer tumor-containing slices
        selected = list(tumor_slices) if tumor_slices else list(candidates)

    # Cap to max_slices -- keep slices with MOST tumor area (most informative)
    if len(selected) > max_slices:
        selected.sort(key=lambda z: tumor_area.get(z, 0), reverse=True)
        selected = selected[:max_slices]

    # Pad to min_slices if needed
    if len(selected) < min_slices:
        extras = [z for z in candidates if z not in set(selected)]
        extras.sort(key=lambda z: tumor_area.get(z, 0), reverse=True)
        selected.extend(extras[:max(0, min_slices - len(selected))])

    return sorted(set(selected))


# =========================================================================
# Step 7 -- Corruption pipeline
# =========================================================================

def _corruption_rng(seed: int, patient_id: str, slice_idx: int) -> np.random.Generator:
    """Deterministic RNG per (seed, patient_id, slice_idx)."""
    h = hashlib.sha256(f"{seed}:{patient_id}:{slice_idx}".encode()).digest()
    sub_seed = int.from_bytes(h[:8], "little") % (2**63)
    return np.random.default_rng(sub_seed)


def apply_bias_field(
    img: np.ndarray, strength: float, rng: np.random.Generator,
) -> np.ndarray:
    """Smooth multiplicative bias field per channel. img shape (C, H, W)."""
    C, H, W = img.shape
    out = img.copy()
    for c in range(C):
        field = rng.standard_normal((H, W)).astype(np.float32)
        field = gaussian_filter(field, sigma=16.0)
        # Normalize to zero mean, unit std
        std = field.std()
        if std > 1e-8:
            field = field / std
        bias = np.exp(strength * field)
        out[c] = out[c] * bias
    return out


def apply_gaussian_noise(
    img: np.ndarray, sigma: float, rng: np.random.Generator,
) -> np.ndarray:
    """Additive Gaussian noise."""
    noise = rng.standard_normal(img.shape).astype(np.float32) * sigma
    return img + noise


def apply_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur per channel."""
    out = np.empty_like(img)
    for c in range(img.shape[0]):
        out[c] = gaussian_filter(img[c], sigma=sigma)
    return out


def apply_downscale(
    img: np.ndarray, scale: float,
) -> np.ndarray:
    """Downsample then upsample to simulate low-resolution acquisition."""
    C, H, W = img.shape
    new_h, new_w = max(4, int(H * scale)), max(4, int(W * scale))
    from scipy.ndimage import zoom
    out = np.empty_like(img)
    for c in range(C):
        down = zoom(img[c], (new_h / H, new_w / W), order=1)
        out[c] = zoom(down, (H / new_h, W / new_w), order=1)
    return out


def corrupt_slice(
    img: np.ndarray,
    profile: Dict[str, Any],
    seed: int,
    patient_id: str,
    slice_idx: int,
) -> np.ndarray:
    """
    Apply corruption pipeline to a single slice image (C, H, W).

    Parameters
    ----------
    profile : dict with keys:
        "steps": list of str, e.g. ["bias", "gauss", "blur", "scale"]
        "bias": float (optional)
        "gauss_sigma": float (optional)
        "blur": float (optional)
        "scale": float (optional)
    """
    steps = profile.get("steps", [])
    if not steps:
        return img

    crng = _corruption_rng(seed, patient_id, slice_idx)
    out = img.copy()

    for step in steps:
        if step == "bias":
            out = apply_bias_field(out, profile["bias"], crng)
        elif step == "gauss":
            out = apply_gaussian_noise(out, profile["gauss_sigma"], crng)
        elif step == "blur":
            out = apply_blur(out, profile["blur"])
        elif step == "scale":
            out = apply_downscale(out, profile["scale"])

    return out


# =========================================================================
# Step 6+7+8 -- Process patients -> write .npz slices
# =========================================================================

def process_and_write_patient(
    case_dir: Path,
    patient_id: str,
    out_dir: Path,
    corruption_profile: Dict[str, Any],
    is_train: bool,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> int:
    """
    Load one patient, select slices, corrupt in raw space, normalize, write .npz.

    Order: load -> clip (raw data) -> select slices -> for each slice:
      extract raw 2D -> corrupt in raw intensity space -> z-score per-slice -> save
    """
    vol, seg = load_case_volumes(case_dir)
    z_dim = vol.shape[-1]  # (4, H, W, Z)

    # Clip percentiles on raw data
    if args.clip_percentiles[0] > 0 or args.clip_percentiles[1] < 100:
        vol = clip_volume(vol, args.clip_percentiles[0], args.clip_percentiles[1])

    # Select slices (uses raw vol for brain mask detection)
    slice_indices = select_slice_indices(
        seg_vol=seg,
        z_dim=z_dim,
        vol=vol,
        policy=args.slice_policy,
        central_band=tuple(args.central_band),
        keep_empty=args.keep_empty_slices,
        is_train=is_train,
        tumor_sampling=args.tumor_sampling,
        min_slices=args.min_slices_per_patient,
        max_slices=args.max_slices_per_patient,
        rng=rng,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for z in slice_indices:
        img_slice = vol[:, :, :, z].astype(np.float32)  # (4, H, W)
        seg_slice = seg[:, :, z].astype(np.int16)        # (H, W)

        # Corrupt in raw intensity space (before normalization)
        if corruption_profile.get("steps"):
            img_slice = corrupt_slice(
                img_slice, corruption_profile,
                args.seed, patient_id, z,
            )

        # Z-score per-slice (always, unless normalize=none)
        if args.normalize == "zscore":
            img_slice = zscore_slice(img_slice)

        fname = f"{patient_id}_z{z:03d}.npz"
        np.savez_compressed(
            str(out_dir / fname),
            image=img_slice,
            mask=seg_slice,
        )
        count += 1

    return count


# =========================================================================
# Step 9 -- Verification
# =========================================================================

def verify_no_leakage(
    client_splits: Dict[int, Dict[str, List[str]]],
    global_test: List[str],
) -> None:
    """Verify no patient appears in multiple clients or splits."""
    global_set = set(global_test)
    all_assigned: Dict[str, str] = {}  # patient -> "client_X/split"

    for cid, splits in client_splits.items():
        for split_name, pids in splits.items():
            for pid in pids:
                loc = f"client_{cid}/{split_name}"
                if pid in global_set:
                    raise ValueError(
                        f"LEAKAGE: {pid} in both global_test and {loc}"
                    )
                if pid in all_assigned:
                    raise ValueError(
                        f"LEAKAGE: {pid} in both {all_assigned[pid]} and {loc}"
                    )
                all_assigned[pid] = loc

    print(f"  Verification passed: {len(all_assigned)} patients + "
          f"{len(global_set)} global_test, no leakage.")


def verify_npz_files(output_dir: Path, sample_n: int = 10) -> None:
    """Spot-check a sample of .npz files for correct keys/shapes/dtypes."""
    npz_files = list(output_dir.rglob("*.npz"))
    if not npz_files:
        raise ValueError(f"No .npz files found under {output_dir}")

    rng = np.random.default_rng(0)
    sample = rng.choice(npz_files, size=min(sample_n, len(npz_files)), replace=False)

    for p in sample:
        d = np.load(str(p), allow_pickle=False)
        if "image" not in d or "mask" not in d:
            raise ValueError(f"{p}: missing keys. Found {list(d.keys())}")
        img = d["image"]
        msk = d["mask"]
        if img.ndim != 3 or img.shape[0] != 4:
            raise ValueError(f"{p}: image shape {img.shape}, expected (4, H, W)")
        if msk.ndim != 2:
            raise ValueError(f"{p}: mask shape {msk.shape}, expected (H, W)")
        if img.shape[1:] != msk.shape:
            raise ValueError(
                f"{p}: shape mismatch image {img.shape} vs mask {msk.shape}"
            )

    print(f"  NPZ spot-check passed ({len(sample)} files sampled).")


# =========================================================================
# Step 10 -- Diagnostics
# =========================================================================

def _jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two discrete distributions."""
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def compute_diagnostics(
    client_splits: Dict[int, Dict[str, List[str]]],
    patient_stats: Dict[str, Dict[str, Any]],
    slice_counts: Dict[str, int],
    global_test: List[str],
    args: argparse.Namespace,
    corruption_profiles: List[Dict[str, Any]],
    n_quantiles: int = 4,
) -> Dict[str, Any]:
    """Build client_map.json content."""

    # Collect all non-global patients and their WT_frac for quantile bins
    all_pids = []
    for splits in client_splits.values():
        for pids in splits.values():
            all_pids.extend(pids)
    all_wt = np.array([patient_stats[p]["WT_frac"] for p in all_pids])
    bin_edges = np.quantile(all_wt, np.linspace(0, 1, n_quantiles + 1))

    # Per-client statistics
    per_client: Dict[str, Any] = {}
    client_histograms: Dict[int, np.ndarray] = {}

    for cid in sorted(client_splits.keys()):
        splits = client_splits[cid]
        c_pids = []
        for pids in splits.values():
            c_pids.extend(pids)

        wt = [patient_stats[p]["WT_frac"] for p in c_pids]
        et = [patient_stats[p]["ET_frac"] for p in c_pids]
        ed = [patient_stats[p]["ED_frac"] for p in c_pids]
        ncr = [patient_stats[p]["NCR_frac"] for p in c_pids]

        # Histogram over WT quantile bins
        bins = np.digitize(wt, bin_edges[1:-1])
        hist = np.bincount(bins, minlength=n_quantiles).astype(np.float64)
        client_histograms[cid] = hist

        # Slice counts
        sc = {
            split: sum(slice_counts.get(f"client_{cid}/{split}/{p}", 0) for p in pids)
            for split, pids in splits.items()
        }

        # Get this client's corruption profile
        profile = corruption_profiles[cid] if cid < len(corruption_profiles) else {"label": "unknown", "steps": []}

        per_client[f"client_{cid}"] = {
            "num_patients": len(c_pids),
            "patients_per_split": {s: len(p) for s, p in splits.items()},
            "slices_per_split": sc,
            "mean_WT_frac": float(np.mean(wt)) if wt else 0.0,
            "mean_ET_frac": float(np.mean(et)) if et else 0.0,
            "mean_ED_frac": float(np.mean(ed)) if ed else 0.0,
            "mean_NCR_frac": float(np.mean(ncr)) if ncr else 0.0,
            "corruption_profile": profile,
        }

    # Pairwise JSD
    cids = sorted(client_histograms.keys())
    jsd_matrix: Dict[str, float] = {}
    for i, ci in enumerate(cids):
        for cj in cids[i + 1:]:
            key = f"client_{ci}_vs_client_{cj}"
            jsd_matrix[key] = _jsd(client_histograms[ci], client_histograms[cj])

    # WT gap and ET gap
    wt_means = [per_client[f"client_{c}"]["mean_WT_frac"] for c in cids]
    et_means = [per_client[f"client_{c}"]["mean_ET_frac"] for c in cids]

    diag = {
        "experiment_config": {
            "num_clients": args.num_clients,
            "seed": args.seed,
            "corruption_schedule": args.corruption_schedule,
            "dirichlet_alpha": args.dirichlet_alpha,
            "min_patients_per_client": args.min_patients_per_client,
            "slice_policy": args.slice_policy,
            "central_band": args.central_band,
            "normalize": args.normalize,
            "clip_percentiles": args.clip_percentiles,
            "tumor_sampling": args.tumor_sampling,
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "min_slices_per_patient": args.min_slices_per_patient,
            "max_slices_per_patient": args.max_slices_per_patient,
        },
        "per_client": per_client,
        "global_test": {
            "num_patients": len(global_test),
            "num_slices": sum(
                slice_counts.get(f"global_test/{p}", 0) for p in global_test
            ),
        },
        "heterogeneity": {
            "pairwise_jsd": jsd_matrix,
            "mean_jsd": float(np.mean(list(jsd_matrix.values()))) if jsd_matrix else 0.0,
            "WT_gap_max_minus_min": float(max(wt_means) - min(wt_means)) if wt_means else 0.0,
            "ET_gap_max_minus_min": float(max(et_means) - min(et_means)) if et_means else 0.0,
        },
    }

    return diag


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    args = build_parser().parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Build corruption profiles --
    if args.corruption_schedule == "per_client_gradient":
        corruption_profiles = generate_corruption_profiles(args.num_clients)
    else:
        # No corruption: all clients clean
        corruption_profiles = [{"label": "clean", "steps": []}] * args.num_clients

    # -- Step 1: Discover patients --
    print("Step 1: Discovering patients ...")
    patient_dirs = discover_patients(args.source_dir)
    print(f"  Found {len(patient_dirs)} patient directories.")

    # -- Step 2: Compute per-patient stats --
    print("Step 2: Computing per-patient tumor signatures ...")
    patient_stats = scan_all_patients(patient_dirs)
    all_patient_ids = sorted(patient_stats.keys())
    print(f"  Stats computed for {len(all_patient_ids)} patients.")

    # -- Step 3: Global test split --
    print("Step 3: Creating global test split ...")
    remaining_ids, global_test_ids = stratified_test_split(
        all_patient_ids, patient_stats, args.test_frac, rng,
    )
    print(f"  Global test: {len(global_test_ids)} patients, "
          f"Remaining: {len(remaining_ids)} patients.")

    # Save global test patient list
    gt_path = output_dir / "global_test_patients.txt"
    gt_path.write_text("\n".join(sorted(global_test_ids)) + "\n")

    # -- Step 4: Client assignment --
    print("Step 4: Assigning patients to clients ...")
    client_assignment = assign_clients(
        remaining_ids, patient_stats, args.num_clients,
        rng,
        alpha=args.dirichlet_alpha,
        min_patients_per_client=args.min_patients_per_client,
    )
    for cid in sorted(client_assignment.keys()):
        profile = corruption_profiles[cid]
        tag = f" [{profile['label']}]"
        print(f"  Client {cid}: {len(client_assignment[cid])} patients{tag}")

    # -- Step 5: Per-client train/val/test splits --
    print("Step 5: Splitting each client into train/val/test ...")
    client_splits: Dict[int, Dict[str, List[str]]] = {}
    for cid in sorted(client_assignment.keys()):
        split_rng = np.random.default_rng(args.seed + cid + 1000)
        client_splits[cid] = split_client_patients(
            client_assignment[cid], args.val_frac, args.test_frac, split_rng,
        )
        parts = {k: len(v) for k, v in client_splits[cid].items()}
        print(f"  Client {cid}: {parts}")

    # Save patient mapping
    client_patients_path = output_dir / "client_patients.json"
    client_patients_path.write_text(json.dumps(
        {str(cid): splits for cid, splits in client_splits.items()},
        indent=2,
    ))

    # -- Verification (patient level) --
    print("Step 5b: Verifying no patient leakage ...")
    verify_no_leakage(client_splits, global_test_ids)

    # -- Step 6+7+8: Process slices and write .npz --
    print("Step 6-8: Processing patients -> slices -> .npz ...")
    # Build a lookup from patient_id to its directory
    pid_to_dir: Dict[str, Path] = {d.name: d for d in patient_dirs}

    slice_counts: Dict[str, int] = {}
    total_slices = 0

    # Process clients
    for cid in sorted(client_splits.keys()):
        profile = corruption_profiles[cid]
        for split_name, pids in client_splits[cid].items():
            is_train = (split_name == "train")
            for pid in pids:
                case_dir = pid_to_dir[pid]
                out_path = output_dir / "client_data" / f"client_{cid}" / split_name
                patient_rng = np.random.default_rng(
                    args.seed + hash(pid) % (2**31)
                )
                n = process_and_write_patient(
                    case_dir, pid, out_path,
                    corruption_profile=profile,
                    is_train=is_train,
                    args=args,
                    rng=patient_rng,
                )
                key = f"client_{cid}/{split_name}/{pid}"
                slice_counts[key] = n
                total_slices += n

        # Summary for this client
        train_n = sum(
            slice_counts.get(f"client_{cid}/train/{p}", 0)
            for p in client_splits[cid]["train"]
        )
        val_n = sum(
            slice_counts.get(f"client_{cid}/val/{p}", 0)
            for p in client_splits[cid]["val"]
        )
        test_n = sum(
            slice_counts.get(f"client_{cid}/test/{p}", 0)
            for p in client_splits[cid]["test"]
        )
        tag = f" [{profile['label']}]"
        print(f"  Client {cid}{tag}: train={train_n}  val={val_n}  test={test_n} slices")

    # Process global test (always clean, no corruption)
    clean_profile: Dict[str, Any] = {"label": "clean", "steps": []}
    print("  Processing global test ...")
    for pid in global_test_ids:
        case_dir = pid_to_dir[pid]
        out_path = output_dir / "global_test"
        patient_rng = np.random.default_rng(args.seed + hash(pid) % (2**31))
        n = process_and_write_patient(
            case_dir, pid, out_path,
            corruption_profile=clean_profile,
            is_train=False,
            args=args,
            rng=patient_rng,
        )
        slice_counts[f"global_test/{pid}"] = n
        total_slices += n

    global_test_slices = sum(
        slice_counts.get(f"global_test/{p}", 0) for p in global_test_ids
    )
    print(f"  Global test: {global_test_slices} slices "
          f"({len(global_test_ids)} patients)")
    print(f"  Total slices written: {total_slices}")

    # -- Step 9: Verify .npz files --
    print("Step 9: Verifying .npz files ...")
    verify_npz_files(output_dir)

    # -- Step 10: Diagnostics --
    print("Step 10: Computing diagnostics ...")
    diagnostics = compute_diagnostics(
        client_splits, patient_stats, slice_counts,
        global_test_ids, args, corruption_profiles,
    )

    client_map_path = output_dir / "client_map.json"
    client_map_path.write_text(json.dumps(diagnostics, indent=2))
    print(f"  Saved {client_map_path}")

    # -- Step 11: Print usage hints --
    print(f"\n{'='*60}")
    print("DONE -- Dataset ready")
    print(f"{'='*60}")
    print(f"  Output dir:      {output_dir}")
    print(f"  Clients:         {args.num_clients}")
    print(f"  Corruption:      {args.corruption_schedule}")
    print(f"  Total slices:    {total_slices}")
    print()
    print("  Per-client corruption profiles:")
    for cid in range(args.num_clients):
        profile = corruption_profiles[cid]
        print(f"    C{cid}: {profile['label']}  steps={profile.get('steps', [])}")
    print()
    print("Example training commands:")
    print(f"  PARTITION_DIR={output_dir / 'client_data'}")
    print(f"  GLOBAL_TEST_DIR={output_dir / 'global_test'}")
    print()
    print("  # FedAvg")
    print(f"  python brats_n_clients.py --partition_dir $PARTITION_DIR "
          f"--strategy fedavg --rounds 30 --local_epochs 50")
    print()
    print("  # FedProx (recommended mu=0.1-1.0 for heterogeneous corruption)")
    print(f"  python brats_n_clients.py --partition_dir $PARTITION_DIR "
          f"--strategy fedprox --mu 0.5 --rounds 30 --local_epochs 50")
    print(f"{'='*60}")

    # Heterogeneity summary
    het = diagnostics["heterogeneity"]
    print(f"\nHeterogeneity diagnostics:")
    print(f"  Mean pairwise JSD:   {het['mean_jsd']:.4f}")
    print(f"  WT gap (max-min):    {het['WT_gap_max_minus_min']:.6f}")
    print(f"  ET gap (max-min):    {het['ET_gap_max_minus_min']:.6f}")
    print()


if __name__ == "__main__":
    main()
