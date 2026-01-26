#this code trains unet on just clients data 
#it is for 50:50 distribution and 70:30 distribution
#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# -----------------------
# Utilities
# -----------------------
def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    dst.symlink_to(src)

def load_client_map(partition_dir: Path) -> Dict[str, List[str]]:
    p = partition_dir / "client_map.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return json.loads(p.read_text())

def split_cases(cases: List[str], seed: int, val_frac: float, test_frac: float) -> Tuple[List[str], List[str], List[str]]:
    assert 0.0 <= val_frac < 1.0 and 0.0 <= test_frac < 1.0 and (val_frac + test_frac) < 1.0
    rng = random.Random(seed)
    cases = sorted(cases)
    rng.shuffle(cases)
    n = len(cases)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = sorted(cases[:n_test])
    val = sorted(cases[n_test:n_test + n_val])
    train = sorted(cases[n_test + n_val:])
    return train, val, test

def build_split_root_from_cases(
    train_dir: Path,
    cases_train: List[str],
    cases_val: List[str],
    cases_test: List[str],
    out_split_root: Path,
    force: bool = False,
) -> None:
    """
    Creates:
      out_split_root/train/<case_id>/*.npz (symlinks)
      out_split_root/val/<case_id>/*.npz
      out_split_root/test/<case_id>/*.npz
    """
    if force and out_split_root.exists():
        shutil.rmtree(out_split_root)
    out_split_root.mkdir(parents=True, exist_ok=True)

    def link_case(case_id: str, split_name: str) -> None:
        src_case = train_dir / case_id
        if not src_case.exists():
            raise FileNotFoundError(f"Case dir not found: {src_case}")
        dst_case = out_split_root / split_name / case_id
        dst_case.mkdir(parents=True, exist_ok=True)
        npz_files = list(src_case.glob("*.npz"))
        if not npz_files:
            raise RuntimeError(f"No .npz files in {src_case}")
        for f in npz_files:
            ensure_symlink(f, dst_case / f.name)

    for cid in cases_train:
        link_case(cid, "train")
    for cid in cases_val:
        link_case(cid, "val")
    for cid in cases_test:
        link_case(cid, "test")

def run_training(
    python_exe: str,
    train_script: Path,
    split_root: Path,
    device: str,
    amp: bool,
    image_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    base: int,
    num_workers: int,
    min_fg_train: int,
    keep_empty_prob: float,
    ckpt_path: Path,
    extra_args: List[str],
) -> None:
    cmd = [
        python_exe, str(train_script),
        "--split_root", str(split_root),
        "--device", device,
        "--image_size", str(image_size),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--base", str(base),
        "--num_workers", str(num_workers),
        "--min_fg_train", str(min_fg_train),
        "--keep_empty_prob", str(keep_empty_prob),
        "--ckpt", str(ckpt_path),
    ]
    if amp:
        cmd.append("--amp")
    cmd.extend(extra_args)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n>>> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# -----------------------
# Main
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    # Paths
    ap.add_argument("--project_dir", type=Path, default=Path.home() / "federated" / "federated-thesis")
    ap.add_argument("--train_script", type=Path, default=None,
                    help="Path to train_unet_brats2d.py (default: <project_dir>/unet/train_unet_brats2d.py)")
    ap.add_argument("--data_root", type=Path,
                    default=Path(os.environ.get("BRATS_DATA_DIR", str(Path.home() / "federated" / "federated-thesis" / "data" / "brats2020_top10_slices_split_npz")))
    ap.add_argument("--partitions_base", type=Path,
                    default=Path.home() / "federated" / "federated-thesis" / "data" / "partitions")
    ap.add_argument("--out_base", type=Path,
                    default=Path.home() / "federated" / "federated-thesis" / "results" / "local_baselines_2clients")

    # Which partitions to run
    ap.add_argument("--run_50_50", action="store_true", help="Run partition federated_clients_2_50_50")
    ap.add_argument("--run_70_30", action="store_true", help="Run partition federated_clients_2_70_30")
    ap.add_argument("--run_both", action="store_true", help="Run both (default if none specified)")

    # Split settings (within each client)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_rebuild_splits", action="store_true")

    # Training hyperparams (match your centralized run for fair comparison)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--image_size", type=int, default=240)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--min_fg_train", type=int, default=50)
    ap.add_argument("--keep_empty_prob", type=float, default=0.05)

    # Misc
    ap.add_argument("--python", type=str, default="python", help="Python executable to use")
    ap.add_argument("--extra_args", nargs="*", default=[], help="Extra args passed to train_unet_brats2d.py")

    args = ap.parse_args()

    if args.train_script is None:
        args.train_script = args.project_dir / "unet" / "train_unet_brats2d.py"

    train_dir = args.data_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Expected train dir at: {train_dir}")

    if not args.train_script.exists():
        raise FileNotFoundError(f"Training script not found: {args.train_script}")

    # Decide which partitions to run
    to_run = []
    if args.run_both or (not args.run_50_50 and not args.run_70_30 and not args.run_both):
        to_run = ["federated_clients_2_50_50", "federated_clients_2_70_30"]
    else:
        if args.run_50_50:
            to_run.append("federated_clients_2_50_50")
        if args.run_70_30:
            to_run.append("federated_clients_2_70_30")

    for part_name in to_run:
        partition_dir = args.partitions_base / part_name
        if not partition_dir.exists():
            raise FileNotFoundError(f"Partition dir not found: {partition_dir}")

        client_map = load_client_map(partition_dir)

        # Accept either {"client_0":[...], "client_1":[...]} or any other keys containing 0/1
        if "client_0" in client_map and "client_1" in client_map:
            cases0 = client_map["client_0"]
            cases1 = client_map["client_1"]
        else:
            # Fallback: try to infer
            keys = sorted(client_map.keys())
            if len(keys) != 2:
                raise RuntimeError(f"Expected exactly 2 clients in {partition_dir}, got keys: {keys}")
            cases0 = client_map[keys[0]]
            cases1 = client_map[keys[1]]

        print(f"\n=== Local baseline: {part_name} ===")
        print(f"Client 0 cases: {len(cases0)} | Client 1 cases: {len(cases1)}")

        for cid, cases in [(0, cases0), (1, cases1)]:
            # Split within this client
            train_cases, val_cases, test_cases = split_cases(
                cases=cases,
                seed=args.seed + cid,  # slight change per client
                val_frac=args.val_frac,
                test_frac=args.test_frac,
            )

            split_root = args.out_base / part_name / f"client_{cid}" / "split_root"
            build_split_root_from_cases(
                train_dir=train_dir,
                cases_train=train_cases,
                cases_val=val_cases,
                cases_test=test_cases,
                out_split_root=split_root,
                force=args.force_rebuild_splits,
            )

            ckpt_path = args.out_base / part_name / f"client_{cid}" / "checkpoints" / f"unet_client{cid}_best.pt"

            print(f"\n--- Training local-only client {cid} on {part_name} ---")
            print(f"Split sizes: train={len(train_cases)} val={len(val_cases)} test={len(test_cases)}")
            print(f"Split root: {split_root}")
            print(f"Checkpoint: {ckpt_path}")

            run_training(
                python_exe=args.python,
                train_script=args.train_script,
                split_root=split_root,
                device=args.device,
                amp=args.amp,
                image_size=args.image_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                base=args.base,
                num_workers=args.num_workers,
                min_fg_train=args.min_fg_train,
                keep_empty_prob=args.keep_empty_prob,
                ckpt_path=ckpt_path,
                extra_args=args.extra_args,
            )

    print("\nAll local-only baseline trainings completed.")

if __name__ == "__main__":
    main()
