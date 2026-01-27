#this code makes all individual clients in 50;50 and 30;70 have their own test and val sets 
#!/usr/bin/env python3

#after running for the identical 50:50 - [OK] client_0: train=118 val=15 test=15 [OK] client_1: train=116 val=15 test=15 saved to 
# /home/bk489/federated/federated-thesis/data/partitions/federated_clients_2_50_50


#for the 70:30 [OK] client_0: train=164 val=21 test=21 [OK] client_1: train=70 val=9 test=9
#saved to : /home/bk489/federated/federated-thesis/data/partitions/federated_clients_2_70_30
import argparse
import random
from pathlib import Path
from typing import List, Tuple

def read_cases(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing: {txt_path}")
    return [ln.strip() for ln in txt_path.read_text().splitlines() if ln.strip()]

def write_cases(txt_path: Path, cases: List[str]) -> None:
    txt_path.write_text("\n".join(cases) + "\n")

def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    dst.symlink_to(src)

def split_cases(cases: List[str], seed: int, val_frac: float, test_frac: float) -> Tuple[List[str], List[str], List[str]]:
    assert 0 <= val_frac < 1 and 0 <= test_frac < 1 and (val_frac + test_frac) < 1
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

def link_case_dir(src_case_dir: Path, dst_case_dir: Path) -> None:
    if not src_case_dir.exists():
        raise FileNotFoundError(f"Missing case dir: {src_case_dir}")
    dst_case_dir.mkdir(parents=True, exist_ok=True)
    npz_files = list(src_case_dir.glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz files in: {src_case_dir}")
    for f in npz_files:
        ensure_symlink(f, dst_case_dir / f.name)

def build_split_for_client(client_dir: Path, seed: int, val_frac: float, test_frac: float, rebuild: bool) -> None:
    """
    Expected layout before:
      client_dir/train/<case_id>/*.npz

    After:
      client_dir/train/<case_id>/*.npz
      client_dir/val/<case_id>/*.npz
      client_dir/test/<case_id>/*.npz

    Also writes:
      client_dir/train_cases.txt, val_cases.txt, test_cases.txt
    """
    train_dir = client_dir / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train dir: {train_dir}")

    all_cases = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    if not all_cases:
        raise RuntimeError(f"No case dirs found under: {train_dir}")

    # If already split and not rebuilding, just print and return
    val_dir = client_dir / "val"
    test_dir = client_dir / "test"
    if (val_dir.exists() or test_dir.exists()) and not rebuild:
        print(f"[SKIP] {client_dir.name}: val/test already exist (use --rebuild to recreate)")
        return

    if rebuild:
        # remove existing val/test folders if present
        if val_dir.exists():
            for p in val_dir.rglob("*"):
                pass
            # safest: delete directory tree
            import shutil
            shutil.rmtree(val_dir)
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)

    train_cases, val_cases, test_cases = split_cases(all_cases, seed=seed, val_frac=val_frac, test_frac=test_frac)

    # Create val/test by symlinking from train, then remove the case folder from train
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for cid in val_cases:
        link_case_dir(train_dir / cid, val_dir / cid)
    for cid in test_cases:
        link_case_dir(train_dir / cid, test_dir / cid)

    # Now remove those case folders from train to prevent leakage
    # (train set should be disjoint from val/test)
    import shutil
    for cid in val_cases + test_cases:
        shutil.rmtree(train_dir / cid)

    # Write case lists
    write_cases(client_dir / "train_cases.txt", train_cases)
    write_cases(client_dir / "val_cases.txt", val_cases)
    write_cases(client_dir / "test_cases.txt", test_cases)

    print(f"[OK] {client_dir.name}: train={len(train_cases)} val={len(val_cases)} test={len(test_cases)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition_dir", type=Path, required=True,
                    help="e.g. .../data/partitions/federated_clients_2_50_50")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rebuild", action="store_true", help="recreate val/test splits")
    args = ap.parse_args()

    part = args.partition_dir
    client_data = part / "client_data"
    if not client_data.exists():
        raise FileNotFoundError(f"Missing: {client_data}")

    # Determine client folders (client_0, client_1, ...)
    clients = sorted([p for p in client_data.iterdir() if p.is_dir() and p.name.startswith("client_")])
    if not clients:
        raise RuntimeError(f"No client_* dirs under {client_data}")

    print(f"Partition: {part}")
    print(f"Clients found: {[c.name for c in clients]}")
    print(f"val_frac={args.val_frac} test_frac={args.test_frac} seed={args.seed}")

    for c in clients:
        # use different seed per client for independence but reproducible
        cid_int = int(c.name.split("_")[-1])
        build_split_for_client(
            client_dir=c,
            seed=args.seed + cid_int,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            rebuild=args.rebuild,
        )

if __name__ == "__main__":
    main()
