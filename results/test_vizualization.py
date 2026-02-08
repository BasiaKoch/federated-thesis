import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ---- EDIT THESE TWO PATHS ----
P253_DIR = "/Users/basiakoch/Downloads/Brats2020 Dataset /BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_236"
P268_DIR = "/Users/basiakoch/Downloads/Brats2020 Dataset /BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_271"
# ------------------------------

def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)

def brats_volumes(seg: np.ndarray) -> dict:
    # BraTS labels: 0=bg, 1=NET/NCR, 2=ED, 4=ET
    wt = np.isin(seg, [1, 2, 4]).sum()
    tc = np.isin(seg, [1, 4]).sum()
    et = (seg == 4).sum()
    return {
        "WT_vox": int(wt),
        "TC_vox": int(tc),
        "ET_vox": int(et),
        "ET/WT": float(et / (wt + 1e-8)),
        "TC/WT": float(tc / (wt + 1e-8)),
    }

def patient_summary(patient_dir: str) -> dict:
    pid = os.path.basename(patient_dir.rstrip("/"))
    seg_path = os.path.join(patient_dir, f"{pid}_seg.nii")
    t1ce_path = os.path.join(patient_dir, f"{pid}_t1ce.nii")

    seg = load_nifti(seg_path)
    t1ce = load_nifti(t1ce_path)

    vols = brats_volumes(seg)

    # choose a representative axial slice: where WT area is largest
    wt_mask = np.isin(seg, [1, 2, 4])
    wt_per_slice = wt_mask.sum(axis=(0, 1))  # assuming shape (H,W,Z)
    z = int(np.argmax(wt_per_slice))

    return {"id": pid, "seg": seg, "t1ce": t1ce, "z": z, **vols}

def save_preview(pA: dict, pB: dict, out_png="hgg_lgg_compare.png"):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for r, p in enumerate([pA, pB]):
        z = p["z"]
        img = p["t1ce"][:, :, z]
        seg = p["seg"][:, :, z]

        axes[r, 0].imshow(img.T, origin="lower", cmap="gray")
        axes[r, 0].set_title(f"{p['id']} t1ce (z={z})")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(img.T, origin="lower", cmap="gray")
        # overlay tumor mask
        tumor = np.isin(seg, [1, 2, 4]).astype(float)
        axes[r, 1].imshow(tumor.T, origin="lower", alpha=0.35)
        axes[r, 1].set_title(f"{p['id']} WT overlay")
        axes[r, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved preview: {out_png}")

if __name__ == "__main__":
    p253 = patient_summary(P253_DIR)
    p268 = patient_summary(P268_DIR)

    def pretty(p):
        return (
            f"{p['id']}: WT={p['WT_vox']:,}  TC={p['TC_vox']:,}  ET={p['ET_vox']:,}  "
            f"ET/WT={p['ET/WT']:.4f}  TC/WT={p['TC/WT']:.4f}"
        )

    print("Comparison (voxel counts):")
    print(" -", pretty(p253))
    print(" -", pretty(p268))

    save_preview(p253, p268)
