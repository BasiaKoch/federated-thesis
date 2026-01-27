import json, random
from pathlib import Path

DATA_TRAIN = Path("/home/bk489/federated/federated-thesis/data/brats2020_top10_slices_split_npz/train")
OUT_DIR = Path("/home/bk489/federated/federated-thesis/data/partitions/federated_clients_10_mixed/client_data")
SEED = 42

# modality indices (adjust if your channel order differs)
T1, T1CE, T2, FLAIR = 0, 1, 2, 3

def is_lgg(case_id: str) -> bool:
    # case_id like "BraTS20_Training_260"
    num = int(case_id.split("_")[-1])
    return 260 <= num <= 335

def list_cases():
    return sorted([p.name for p in DATA_TRAIN.iterdir() if p.is_dir() and p.name.startswith("BraTS20_Training_")])

def case_npz_files(case_id: str):
    return sorted((DATA_TRAIN / case_id).rglob("*.npz"))

def take_cases(case_ids, n_cases):
    return case_ids[:n_cases]

def write_client(client_name: str, case_ids, keep_channels, mode="fixed"):
    """
    mode:
      - fixed: always keep the same channels
      - per_sample_mixed: client has a pool of channel sets, chosen per slice (store in meta)
    """
    cdir = OUT_DIR / client_name / "train"
    cdir.mkdir(parents=True, exist_ok=True)

    # store a manifest of files (so training doesn't rely on folder copying)
    files = []
    for cid in case_ids:
        for f in case_npz_files(cid):
            files.append(str(f))

    meta = {
        "client": client_name,
        "n_cases": len(case_ids),
        "n_slices": len(files),
        "case_ids": case_ids,
        "mode": mode,
        "keep_channels": keep_channels,
    }

    (OUT_DIR / client_name).mkdir(parents=True, exist_ok=True)
    (OUT_DIR / client_name / "manifest_train.json").write_text(json.dumps(files, indent=2))
    (OUT_DIR / client_name / "meta.json").write_text(json.dumps(meta, indent=2))

def main():
    random.seed(SEED)

    cases = list_cases()
    lgg_cases = [c for c in cases if is_lgg(c)]
    hgg_cases = [c for c in cases if not is_lgg(c)]

    random.shuffle(lgg_cases)
    random.shuffle(hgg_cases)

    # Choose case counts to enforce quantity skew
    # Adjust numbers if you want more/less extreme skew
    plan = [
        ("client_0_HGG_full",        "HGG", 4, [T1,T1CE,T2,FLAIR], "fixed"),
        ("client_1_LGG_full_small",  "LGG", 2, [T1,T1CE,T2,FLAIR], "fixed"),
        ("client_2_HGG_T1ce_big",    "HGG", 8, [T1CE],            "fixed"),
        ("client_3_HGG_FLAIR",       "HGG", 4, [FLAIR],           "fixed"),
        ("client_4_LGG_T2",          "LGG", 4, [T2],              "fixed"),
        ("client_5_HGG_T1_T1ce",     "HGG", 4, [T1,T1CE],         "fixed"),
        ("client_6_mixed_T2_FLAIR",  "MIX", 4, [T2,FLAIR],        "fixed"),
        ("client_7_mixed_3mods_noCE","MIX", 4, [T1,T2,FLAIR],     "fixed"),
        ("client_8_HGG_tiny_full",   "HGG", 1, [T1,T1CE,T2,FLAIR], "fixed"),
        ("client_9_mixed_per_sample","MIX", 1, [T1,T1CE,T2,FLAIR], "per_sample_mixed"),
    ]

    # simple allocator
    lgg_ptr, hgg_ptr = 0, 0
    for name, grade, n_cases, keep, mode in plan:
        if grade == "LGG":
            chosen = lgg_cases[lgg_ptr:lgg_ptr+n_cases]
            lgg_ptr += n_cases
        elif grade == "HGG":
            chosen = hgg_cases[hgg_ptr:hgg_ptr+n_cases]
            hgg_ptr += n_cases
        else:
            # MIX: half from each if possible
            half = n_cases // 2
            chosen = lgg_cases[lgg_ptr:lgg_ptr+half] + hgg_cases[hgg_ptr:hgg_ptr+(n_cases-half)]
            lgg_ptr += half
            hgg_ptr += (n_cases-half)
            random.shuffle(chosen)

        write_client(name, chosen, keep, mode=mode)

    print("Wrote clients to:", OUT_DIR)

if __name__ == "__main__":
    main()
