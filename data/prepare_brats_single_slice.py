#!/usr/bin/env python3
"""
BraTS 2020 single-slice partitioning for Federated Learning.
Client 0: Max tumor slice (most informative)
Client 1: Low percentile slice (less informative)

Ensures strict patient-level separation.
"""
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import nibabel as nib

MODALITY_SUFFIXES = {"t1": "t1", "t1ce": "t1ce", "t2": "t2", "flair": "flair", "seg": "seg"}
DEFAULT_MODALITY_ORDER = ["t1", "t1ce", "t2", "flair"]


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
    vol = np.stack(vols, axis=0)

    seg_path = case_dir / f"{case_dir.name}_{MODALITY_SUFFIXES['seg']}.nii"
    seg = load_nii(seg_path).astype(np.int16)

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


def compute_tumor_area_per_slice(seg: np.ndarray) -> np.ndarray:
    """Compute whole tumor area for each axial slice."""
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


def extract_slice(vol: np.ndarray, seg: np.ndarray, z: int) -> Tuple[np.ndarray, np.ndarray]:
    img2d = vol[:, :, :, z].astype(np.float32)  # (C, H, W)
    msk2d = seg[:, :, z].astype(np.int16)       # (H, W)
    return img2d, msk2d


def save_npz(out_path: Path, image: np.ndarray, mask: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), image=image, mask=mask)


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
        clients = set(c for c, s in locs)
        splits = set(s for c, s in locs)
        
        if len(clients) > 1:
            errors.append(f"CROSS-CLIENT: {patient} in clients {clients}")
        if len(splits) > 1:
            errors.append(f"CROSS-SPLIT: {patient} in splits {splits}")
    
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        raise ValueError("Patient separation violated!")
    
    print(f"  Verified: {len(patient_locations)} patients correctly separated")


def main():
    ap = argparse.ArgumentParser(
        description="Single-slice BraTS partitioning with quality-based non-IID"
    )
    ap.add_argument("--brats_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--no_zscore", action="store_true")
    ap.add_argument("--modalities", type=str, default="t1,t1ce,t2,flair")
    
    # Non-IID configuration
    ap.add_argument("--client0_mode", type=str, default="max",
                    help="Slice selection for client 0: max, p75, p50, p25, boundary_upper, boundary_lower")
    ap.add_argument("--client1_mode", type=str, default="p25",
                    help="Slice selection for client 1: max, p75, p50, p25, boundary_upper, boundary_lower")
    
    args = ap.parse_args()
    
    assert args.train_frac + args.val_frac < 1.0
    
    modality_order = [m.strip() for m in args.modalities.split(",") if m.strip()]
    rng = np.random.RandomState(args.seed)
    
    case_dirs = find_case_dirs(args.brats_root)
    if not case_dirs:
        raise RuntimeError(f"No BraTS cases found in {args.brats_root}")
    
    print(f"Found {len(case_dirs)} cases")
    
    # Shuffle and split patients into 2 clients
    case_dirs_shuffled = list(case_dirs)
    rng.shuffle(case_dirs_shuffled)
    
    mid = len(case_dirs_shuffled) // 2
    client_cases = [case_dirs_shuffled[:mid], case_dirs_shuffled[mid:]]
    client_modes = [args.client0_mode, args.client1_mode]
    
    print(f"\nClient 0: {len(client_cases[0])} patients, slice mode = '{args.client0_mode}'")
    print(f"Client 1: {len(client_cases[1])} patients, slice mode = '{args.client1_mode}'")
    
    # Statistics tracking
    summary_lines = ["patient_id\tclient\tsplit\tslice_mode\tz_index\ttumor_area\ttumor_fraction\n"]
    stats: Dict[int, Dict[str, int]] = {0: {"train": 0, "val": 0, "test": 0},
                                         1: {"train": 0, "val": 0, "test": 0}}
    skipped = {0: 0, 1: 0}
    tumor_areas: Dict[int, List[float]] = {0: [], 1: []}
    
    for client_idx in range(2):
        cases = client_cases[client_idx]
        mode = client_modes[client_idx]
        
        # Shuffle cases for train/val/test split
        cases_shuffled = list(cases)
        rng.shuffle(cases_shuffled)
        
        n = len(cases_shuffled)
        n_train = int(n * args.train_frac)
        n_val = int(n * args.val_frac)
        
        splits = {
            "train": cases_shuffled[:n_train],
            "val": cases_shuffled[n_train:n_train + n_val],
            "test": cases_shuffled[n_train + n_val:]
        }
        
        print(f"\nProcessing client {client_idx}...")
        
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
                    print(f"  Warning: {patient_id} has no tumor, skipping")
                    skipped[client_idx] += 1
                    continue
                
                img2d, msk2d = extract_slice(vol, seg, z)
                
                # Save
                out_path = (args.out_dir / f"client_{client_idx}" / split_name /
                           patient_id / f"{patient_id}_z{z:03d}.npz")
                save_npz(out_path, img2d, msk2d)
                
                # Track statistics
                area = int(tumor_area[z])
                frac = float((msk2d > 0).mean())
                tumor_areas[client_idx].append(area)
                stats[client_idx][split_name] += 1
                
                summary_lines.append(
                    f"{patient_id}\tclient_{client_idx}\t{split_name}\t{mode}\t{z}\t{area}\t{frac:.6f}\n"
                )
    
    # Print summary
    print("\n" + "=" * 70)
    print("PARTITION SUMMARY")
    print("=" * 70)
    
    for client_idx in range(2):
        s = stats[client_idx]
        total = s["train"] + s["val"] + s["test"]
        mode = client_modes[client_idx]
        
        areas = tumor_areas[client_idx]
        mean_area = np.mean(areas) if areas else 0
        std_area = np.std(areas) if areas else 0
        
        print(f"\nClient {client_idx} (mode='{mode}'):")
        print(f"  Samples: train={s['train']}, val={s['val']}, test={s['test']}, total={total}")
        print(f"  Skipped: {skipped[client_idx]}")
        print(f"  Tumor area: mean={mean_area:.1f} ± {std_area:.1f} pixels")
    
    # Compare heterogeneity
    if tumor_areas[0] and tumor_areas[1]:
        ratio = np.mean(tumor_areas[0]) / (np.mean(tumor_areas[1]) + 1e-8)
        print(f"\nHeterogeneity ratio (Client0/Client1 mean tumor area): {ratio:.2f}x")
    
    print("=" * 70)
    
    # Verify patient separation
    print("\nVerifying patient-level separation...")
    verify_patient_separation(args.out_dir, num_clients=2)
    
    # Save outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.out_dir / "partition_summary.tsv", "w") as f:
        f.writelines(summary_lines)
    
    config_text = f"""Federated Learning Partition Configuration
==========================================
Date: {np.datetime64('now')}
Seed: {args.seed}
Total cases: {len(case_dirs)}

Client 0:
  - Slice mode: {args.client0_mode}
  - Patients: {len(client_cases[0])}
  - Train/Val/Test: {stats[0]['train']}/{stats[0]['val']}/{stats[0]['test']}
  - Mean tumor area: {np.mean(tumor_areas[0]):.1f} pixels

Client 1:
  - Slice mode: {args.client1_mode}
  - Patients: {len(client_cases[1])}
  - Train/Val/Test: {stats[1]['train']}/{stats[1]['val']}/{stats[1]['test']}
  - Mean tumor area: {np.mean(tumor_areas[1]):.1f} pixels

Heterogeneity:
  - This partition creates QUALITY-based non-IID
  - Client 0 sees most informative slices (max tumor)
  - Client 1 sees less informative slices ({args.client1_mode})
  - Expected: FedProx should handle this better than FedAvg
"""
    
    with open(args.out_dir / "partition_config.txt", "w") as f:
        f.write(config_text)
    
    print(f"\nSaved: {args.out_dir / 'partition_summary.tsv'}")
    print(f"Saved: {args.out_dir / 'partition_config.txt'}")
    print("\nDone!")


if __name__ == "__main__":
    main()