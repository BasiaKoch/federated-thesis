#!/usr/bin/env python3
import math
import random
from pathlib import Path
import argparse
import numpy as np
import nibabel as nib

MOD_ORDER = ["flair", "t1", "t1ce", "t2"]  # channel order in saved array


def load_nii(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata())


def find_patient_folders(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("BraTS20_Training_")])


def patient_files(pdir: Path):
    pid = pdir.name
    files = {m: pdir / f"{pid}_{m}.nii" for m in MOD_ORDER}
    files["seg"] = pdir / f"{pid}_seg.nii"

    # Handle occasional Segm naming variant (as you saw before)
    if not files["seg"].exists():
        alt = pdir / f"{pid}_Segm.nii"
        if alt.exists():
            files["seg"] = alt

    missing = [k for k, v in files.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {pdir}")
    return pid, files


def choose_z_max(seg3d: np.ndarray, seed: int) -> int:
    """
    Choose the axial slice index with maximum tumor area.
    If no tumor exists, fallback to middle slice (deterministic).
    """
    tumor_area = (seg3d > 0).sum(axis=(0, 1))  # per-slice tumor pixels
    if np.max(tumor_area) > 0:
        # if ties, pick the smallest z for determinism
        return int(np.argmax(tumor_area))
    # no-tumor case (rare): pick middle slice
    return int(seg3d.shape[2] // 2)


def save_slice(out_split_dir: Path, pid: str, z: int, x4: np.ndarray, y: np.ndarray):
    """
    Output structure: out_split_dir/pid/slice_{z:03d}.npz
    Keys: image (4,H,W), mask (H,W)
    """
    patient_dir = out_split_dir / pid
    patient_dir.mkdir(parents=True, exist_ok=True)
    fname = patient_dir / f"slice_{z:03d}.npz"
    np.savez_compressed(fname, image=x4.astype(np.float32), mask=y.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True, help="MICCAI_BraTS2020_TrainingData folder")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)

    ap.add_argument("--client0_frac", type=float, default=0.7, help="fraction of TRAIN patients for client_0")
    args = ap.parse_args()

    if not math.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0, rel_tol=1e-6):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    patients = find_patient_folders(input_root)
    if len(patients) == 0:
        raise RuntimeError(f"No patient folders found in {input_root}")

    rng = random.Random(args.seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(round(args.train_frac * n))
    n_val = int(round(args.val_frac * n))
    splits = {
        "train": patients[:n_train],
        "val": patients[n_train:n_train + n_val],
        "test": patients[n_train + n_val:],
    }

    # Split TRAIN patients across clients (70/30 by patient)
    train_pats = splits["train"][:]
    rng.shuffle(train_pats)
    cut = int(round(args.client0_frac * len(train_pats)))
    client_train = {"client_0": train_pats[:cut], "client_1": train_pats[cut:]}

    print(f"Patients total={n} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"Train->clients: client_0={len(client_train['client_0'])} client_1={len(client_train['client_1'])}")

    # Create directory structure
    for client in ["client_0", "client_1"]:
        for split in ["train", "val", "test"]:
            (output_root / client / split).mkdir(parents=True, exist_ok=True)

    def process_patients(plist, out_split_dir: Path):
        for pdir in plist:
            pid, files = patient_files(pdir)

            seg = load_nii(files["seg"])
            if seg.ndim != 3:
                raise ValueError(f"Unexpected seg ndim for {pid}: {seg.shape}")

            # Choose z_max based on tumor area (deterministic per patient)
            patient_seed = (hash((pid, args.seed)) & 0xFFFFFFFF)
            z = choose_z_max(seg, patient_seed)

            # Load modalities and stack at same z
            mods = [load_nii(files[m]) for m in MOD_ORDER]
            for m, arr in zip(MOD_ORDER, mods):
                if arr.shape != seg.shape:
                    raise ValueError(f"Shape mismatch {pid}: {m} {arr.shape} vs seg {seg.shape}")

            x4 = np.stack([arr[:, :, z] for arr in mods], axis=0)  # (4,H,W)
            y = seg[:, :, z]  # (H,W)
            save_slice(out_split_dir, pid, int(z), x4, y)

            # Optional: print minimal info
            area = int((y > 0).sum())
            print(f"  {pid}: z={z} tumor_pixels={area}")

    # Write TRAIN per client
    print("\n== CLIENT_0 TRAIN ==")
    process_patients(client_train["client_0"], output_root / "client_0" / "train")
    print("\n== CLIENT_1 TRAIN ==")
    process_patients(client_train["client_1"], output_root / "client_1" / "train")

    # Shared VAL/TEST copied to both clients (same patients, stable evaluation)
    for split in ["val", "test"]:
        print(f"\n== SHARED {split.upper()} ==")
        for client in ["client_0", "client_1"]:
            process_patients(splits[split], output_root / client / split)

    print("\nDone.")


if __name__ == "__main__":
    main()
