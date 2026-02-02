#!/usr/bin/env python3
"""
BraTS 2020 single-slice partitioning for Federated Learning (2 clients) with STRONG non-IID.

Goal: make Client 0 and Client 1 genuinely different (hard non-IID), to better separate FedAvg vs FedProx.

Client 0 (easy / informative):
  - selects patients from TOP fraction by max Whole Tumor (WT) area
  - slice mode typically "max"

Client 1 (hard / uninformative):
  - selects patients from BOTTOM fraction by boundary WT area (boundary slice has tiny tumor)
  - slice mode typically "boundary_lower" or "boundary_upper"
  - optional filter: boundary_area / max_area <= threshold (very harsh)

Ensures strict patient-level separation AND strict split separation (train/val/test per client).

Outputs:
  out_dir/
    client_0/{train,val,test}/BraTS.../*.npz
    client_1/{train,val,test}/BraTS.../*.npz
    partition_summary.tsv
    partition_config.txt
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import nibabel as nib

MODALITY_SUFFIXES = {"t1": "t1", "t1ce": "t1ce", "t2": "t2", "flair": "flair", "seg": "seg"}
DEFAULT_MODALITY_ORDER = ["t1", "t1ce", "t2", "flair"]


# -------------------------
# IO helpers
# -------------------------
def find_case_dirs(brats_root: Path) -> List[Path]:
    return sorted([p for p in brats_root.iterdir() if p.is_dir() and p.name.startswith("BraTS")])


def load_nii(path: Path) -> np.ndarray:
    nii_path = path
    if not nii_path.exists():
        nii_path = Path(str(path) + ".gz")
    if not nii_path.exists():
        raise FileNotFoundError(f"Missing: {path} or {path}.gz")
    return nib.load(str(nii_path)).get_fdata(dtype=np.float32)


def load_case(case_dir: Path, modality_order: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    vols = []
    for m in modality_order:
        p = case_dir / f"{case_dir.name}_{MODALITY_SUFFIXES[m]}.nii"
        vols.append(load_nii(p))
    vol = np.stack(vols, axis=0)  # (C,H,W,Z)

    seg_path = case_dir / f"{case_dir.name}_{MODALITY_SUFFIXES['seg']}.nii"
    seg = load_nii(seg_path).astype(np.int16)  # (H,W,Z)

    return vol.astype(np.float32), seg


def zscore_per_channel(vol: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    out = vol.copy()
    for c in range(out.shape[0]):
        x = out[c]
        nz = x != 0
        if np.any(nz):
            mu, sd = x[nz].mean(), x[nz].std()
            out[c] = (x - mu) / (sd + eps)
    return out


# -------------------------
# Tumor + slice selection
# -------------------------
def compute_tumor_area_per_slice(seg: np.ndarray) -> np.ndarray:
    """Compute whole tumor (WT) area per axial slice. seg: (H,W,Z)."""
    return (seg > 0).sum(axis=(0, 1)).astype(np.int64)


def choose_slice_index(tumor_area: np.ndarray, mode: str) -> Optional[int]:
    """
    Select slice index based on mode.

    Modes:
        - "max": slice with maximum tumor area
        - "p75", "p50", "p25": slice closest to that percentile of max area
        - "boundary_upper": highest z slice with tumor (last index with tumor)
        - "boundary_lower": lowest z slice with tumor (first index with tumor)
    """
    positive_indices = np.where(tumor_area > 0)[0]
    if len(positive_indices) == 0:
        return None

    if mode == "max":
        return int(np.argmax(tumor_area))

    if mode.startswith("p"):
        percentile = int(mode[1:]) / 100.0
        max_area = float(tumor_area.max())
        target_area = percentile * max_area
        positive_areas = tumor_area[positive_indices]
        closest_idx = int(np.argmin(np.abs(positive_areas - target_area)))
        return int(positive_indices[closest_idx])

    if mode == "boundary_upper":
        return int(positive_indices[-1])  # highest z with tumor
    if mode == "boundary_lower":
        return int(positive_indices[0])   # lowest z with tumor

    raise ValueError(f"Unknown mode: {mode}")


def extract_slice(vol: np.ndarray, seg: np.ndarray, z: int) -> Tuple[np.ndarray, np.ndarray]:
    img2d = vol[:, :, :, z].astype(np.float32)  # (C,H,W)
    msk2d = seg[:, :, z].astype(np.int16)       # (H,W)
    return img2d, msk2d


def save_npz(out_path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), image=image, mask=mask)


# -------------------------
# Verification
# -------------------------
def verify_patient_separation(out_dir: Path, num_clients: int) -> None:
    """Verify no patient appears in multiple clients or multiple splits."""
    from collections import defaultdict
    patient_locations = defaultdict(list)

    for client_idx in range(num_clients):
        for split in ["train", "val", "test"]:
            split_dir = out_dir / f"client_{client_idx}" / split
            if not split_dir.exists():
                continue
            for case_dir in split_dir.iterdir():
                if case_dir.is_dir():
                    patient_locations[case_dir.name].append((client_idx, split))

    errors = []
    for patient, locs in patient_locations.items():
        clients = set(c for c, _ in locs)
        splits = set(s for _, s in locs)
        if len(clients) > 1:
            errors.append(f"CROSS-CLIENT: {patient} in clients {clients}")
        if len(splits) > 1:
            errors.append(f"CROSS-SPLIT: {patient} in splits {splits}")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        raise ValueError("Patient separation violated!")

    print(f"  Verified: {len(patient_locations)} patients correctly separated")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Single-slice BraTS partitioning with STRONG non-IID (2 clients).")

    ap.add_argument("--brats_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--no_zscore", action="store_true")
    ap.add_argument("--modalities", type=str, default="t1,t1ce,t2,flair")

    # Slice modes used for actually writing slices
    ap.add_argument("--client0_mode", type=str, default="max",
                    help="Slice selection for client 0: max, p75, p50, p25, boundary_upper, boundary_lower")
    ap.add_argument("--client1_mode", type=str, default="boundary_lower",
                    help="Slice selection for client 1: max, p75, p50, p25, boundary_upper, boundary_lower")

    # Strong non-IID: assign patients by tumor stats before splitting
    ap.add_argument("--client0_top_frac", type=float, default=0.40,
                    help="Client0 takes TOP fraction by max WT area (e.g. 0.4 = top 40%% largest tumors).")
    ap.add_argument("--client1_bottom_frac", type=float, default=0.40,
                    help="Client1 takes BOTTOM fraction by boundary WT area (e.g. 0.4 = bottom 40%% smallest boundary tumors).")

    ap.add_argument("--client1_max_boundary_frac", type=float, default=0.02,
                    help="Keep only cases where boundary_area/max_area <= this. Smaller = harsher. "
                         "Set to -1 to disable.")

    args = ap.parse_args()

    if not (0 < args.train_frac < 1 and 0 < args.val_frac < 1 and (args.train_frac + args.val_frac) < 1):
        raise ValueError("Require 0<train_frac<1, 0<val_frac<1, and train_frac+val_frac<1.")

    if not (0 < args.client0_top_frac <= 1.0):
        raise ValueError("--client0_top_frac must be in (0,1].")
    if not (0 < args.client1_bottom_frac <= 1.0):
        raise ValueError("--client1_bottom_frac must be in (0,1].")

    boundary_frac_thr = None if args.client1_max_boundary_frac is None or args.client1_max_boundary_frac < 0 else float(args.client1_max_boundary_frac)

    modality_order = [m.strip() for m in args.modalities.split(",") if m.strip()]
    rng = np.random.RandomState(args.seed)

    case_dirs = find_case_dirs(args.brats_root)
    if not case_dirs:
        raise RuntimeError(f"No BraTS cases found in {args.brats_root}")

    print(f"Found {len(case_dirs)} cases")

    # ------------------------------------------------------------
    # STEP 1: Score all patients for (max WT area) and (boundary WT area)
    # ------------------------------------------------------------
    print("\nScoring all patients for severity / boundary difficulty...")
    scored = []  # list of dict-like tuples

    for case_dir in case_dirs:
        pid = case_dir.name
        try:
            _, seg = load_case(case_dir, modality_order)
        except FileNotFoundError as e:
            print(f"  Warning: Skipping {pid}: {e}")
            continue

        tumor_area = compute_tumor_area_per_slice(seg)

        z_max = choose_slice_index(tumor_area, "max")
        z_bnd = choose_slice_index(tumor_area, args.client1_mode)  # boundary mode drives "hardness"

        if z_max is None or z_bnd is None:
            # no tumor -> skip (you could also route to a specific client if desired)
            continue

        max_area = int(tumor_area[z_max])
        bnd_area = int(tumor_area[z_bnd])
        bnd_frac = float(bnd_area / (max_area + 1e-8))

        scored.append((case_dir, max_area, bnd_area, bnd_frac, int(z_max), int(z_bnd)))

    if not scored:
        raise RuntimeError("No valid cases after scoring (did you point to correct BraTS root?).")

    print(f"Scored {len(scored)} cases with tumor present.")

    # ------------------------------------------------------------
    # STEP 2: Choose patient sets for each client (strong non-IID)
    # ------------------------------------------------------------
    scored_by_max = sorted(scored, key=lambda t: t[1])  # max_area ascending
    scored_by_bnd = sorted(scored, key=lambda t: t[2])  # boundary_area ascending

    n_total = len(scored)

    # Client0: top fraction by max_area
    start0 = int((1.0 - args.client0_top_frac) * n_total)
    client0_pool = scored_by_max[start0:]
    rng.shuffle(client0_pool)
    client0_cases = [t[0] for t in client0_pool]

    # Client1: bottom fraction by boundary_area + optional boundary_frac threshold
    end1 = int(args.client1_bottom_frac * n_total)
    client1_pool = scored_by_bnd[:end1]
    if boundary_frac_thr is not None:
        client1_pool = [t for t in client1_pool if t[3] <= boundary_frac_thr]
    rng.shuffle(client1_pool)
    client1_cases = [t[0] for t in client1_pool]

    # Remove overlap, then equalize sizes
    c0_ids = {c.name for c in client0_cases}
    client1_cases = [c for c in client1_cases if c.name not in c0_ids]

    m = min(len(client0_cases), len(client1_cases))
    if m < 10:
        raise RuntimeError(
            f"Too few matched cases after filtering (m={m}). "
            f"Relax --client1_max_boundary_frac or increase *_frac."
        )

    client0_cases = client0_cases[:m]
    client1_cases = client1_cases[:m]

    print("\nClient assignment (STRONG non-IID):")
    print(f"  Client 0: {len(client0_cases)} patients (TOP max WT area), slice mode='{args.client0_mode}'")
    print(f"  Client 1: {len(client1_cases)} patients (BOTTOM boundary WT area"
          f"{'' if boundary_frac_thr is None else f', boundary_frac<={boundary_frac_thr}'}), "
          f"slice mode='{args.client1_mode}'")

    client_cases = [client0_cases, client1_cases]
    client_modes = [args.client0_mode, args.client1_mode]

    # ------------------------------------------------------------
    # STEP 3: Split train/val/test within each client and write NPZ
    # ------------------------------------------------------------
    summary_lines = ["patient_id\tclient\tsplit\tslice_mode\tz_index\twt_area\twt_fraction\n"]
    stats: Dict[int, Dict[str, int]] = {0: {"train": 0, "val": 0, "test": 0},
                                        1: {"train": 0, "val": 0, "test": 0}}
    skipped = {0: 0, 1: 0}
    wt_areas: Dict[int, List[int]] = {0: [], 1: []}
    wt_fracs: Dict[int, List[float]] = {0: [], 1: []}

    for client_idx in range(2):
        cases = list(client_cases[client_idx])
        mode = client_modes[client_idx]

        rng.shuffle(cases)
        n = len(cases)
        n_train = int(n * args.train_frac)
        n_val = int(n * args.val_frac)
        splits = {
            "train": cases[:n_train],
            "val": cases[n_train:n_train + n_val],
            "test": cases[n_train + n_val:],
        }

        print(f"\nWriting client {client_idx}...")
        for split_name, split_cases in splits.items():
            for case_dir in split_cases:
                patient_id = case_dir.name
                try:
                    vol, seg = load_case(case_dir, modality_order)
                except FileNotFoundError as e:
                    print(f"  Warning: Skipping {patient_id}: {e}")
                    skipped[client_idx] += 1
                    continue

                if not args.no_zscore:
                    vol = zscore_per_channel(vol)

                tumor_area = compute_tumor_area_per_slice(seg)
                z = choose_slice_index(tumor_area, mode)
                if z is None:
                    skipped[client_idx] += 1
                    continue

                img2d, msk2d = extract_slice(vol, seg, z)

                # Stats
                area = int(np.count_nonzero(msk2d > 0))
                frac = float((msk2d > 0).mean())
                wt_areas[client_idx].append(area)
                wt_fracs[client_idx].append(frac)

                # Save
                out_path = (args.out_dir / f"client_{client_idx}" / split_name /
                            patient_id / f"{patient_id}_z{z:03d}.npz")
                save_npz(out_path, img2d, msk2d)

                stats[client_idx][split_name] += 1
                summary_lines.append(
                    f"{patient_id}\tclient_{client_idx}\t{split_name}\t{mode}\t{z}\t{area}\t{frac:.6f}\n"
                )

    # ------------------------------------------------------------
    # STEP 4: Print summary + save metadata
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PARTITION SUMMARY")
    print("=" * 70)

    for client_idx in range(2):
        s = stats[client_idx]
        total = s["train"] + s["val"] + s["test"]
        areas = wt_areas[client_idx]
        fracs = wt_fracs[client_idx]

        mean_area = float(np.mean(areas)) if areas else 0.0
        std_area = float(np.std(areas)) if areas else 0.0
        mean_frac = float(np.mean(fracs)) if fracs else 0.0
        std_frac = float(np.std(fracs)) if fracs else 0.0

        print(f"\nClient {client_idx} (mode='{client_modes[client_idx]}'):")
        print(f"  Samples: train={s['train']}, val={s['val']}, test={s['test']}, total={total}")
        print(f"  Skipped: {skipped[client_idx]}")
        print(f"  WT area: mean={mean_area:.1f} ± {std_area:.1f} pixels")
        print(f"  WT frac: mean={mean_frac:.6f} ± {std_frac:.6f}")

    if wt_areas[0] and wt_areas[1]:
        ratio = (np.mean(wt_areas[0]) + 1e-8) / (np.mean(wt_areas[1]) + 1e-8)
        print(f"\nHeterogeneity ratio (Client0/Client1 mean WT area): {ratio:.2f}x")

    print("=" * 70)

    print("\nVerifying patient-level separation...")
    verify_patient_separation(args.out_dir, num_clients=2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "partition_summary.tsv").write_text("".join(summary_lines))

    config_text = f"""Federated Learning Partition Configuration (STRONG non-IID)
==========================================
Seed: {args.seed}
Total cases on disk: {len(case_dirs)}
Scored tumor-present cases: {len(scored)}

Modalities: {",".join(modality_order)}
Z-score per channel: {"NO" if args.no_zscore else "YES"}

Client 0:
  - Patient selection: TOP {args.client0_top_frac:.2f} by max WT area
  - Slice mode: {args.client0_mode}
  - Patients written: {len(client_cases[0])}
  - Train/Val/Test: {stats[0]['train']}/{stats[0]['val']}/{stats[0]['test']}
  - WT area mean: {float(np.mean(wt_areas[0])) if wt_areas[0] else 0.0:.1f}
  - WT frac mean: {float(np.mean(wt_fracs[0])) if wt_fracs[0] else 0.0:.6f}

Client 1:
  - Patient selection: BOTTOM {args.client1_bottom_frac:.2f} by boundary WT area (mode={args.client1_mode})
  - Boundary fraction filter (boundary_area/max_area): {("DISABLED" if boundary_frac_thr is None else boundary_frac_thr)}
  - Slice mode: {args.client1_mode}
  - Patients written: {len(client_cases[1])}
  - Train/Val/Test: {stats[1]['train']}/{stats[1]['val']}/{stats[1]['test']}
  - WT area mean: {float(np.mean(wt_areas[1])) if wt_areas[1] else 0.0:.1f}
  - WT frac mean: {float(np.mean(wt_fracs[1])) if wt_fracs[1] else 0.0:.6f}

Heterogeneity:
  - Strong non-IID via patient selection + slice mode difference
  - Expectation: FedProx should reduce client drift more than FedAvg (especially with higher local epochs)

Notes:
  - If too few cases remain after filtering, relax --client1_max_boundary_frac or increase *_frac.
  - For harsher shift: use boundary_lower + smaller boundary_frac threshold (e.g. 0.01).
"""
    (args.out_dir / "partition_config.txt").write_text(config_text)

    print(f"\nSaved: {args.out_dir / 'partition_summary.tsv'}")
    print(f"Saved: {args.out_dir / 'partition_config.txt'}")
    print("\nDone!")


if __name__ == "__main__":
    main()
