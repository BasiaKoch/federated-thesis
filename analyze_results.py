import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import create_model  # must match training

class MNISTArray(Dataset):
    def __init__(self, x, y):
        x = x.astype(np.float32) / 255.0
        x = (x - 0.1307) / 0.3081
        self.x = x
        self.y = y.astype(np.int64)

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]).unsqueeze(0), torch.tensor(self.y[i], dtype=torch.long)

def evaluate(model, loader, device):
    crit = nn.CrossEntropyLoss()
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            bs = x.size(0)
            loss_sum += loss.item() * bs
            correct += (logits.argmax(1) == y).sum().item()
            total += bs
    return loss_sum / total, correct / total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to server global model checkpoint")
    ap.add_argument("--npz", required=True, help="Path to mnist.npz")
    ap.add_argument("--model_type", default="logistic", choices=["logistic", "cnn"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(args.npz)
    x_test, y_test = data["x_test"], data["y_test"]
    loader = DataLoader(MNISTArray(x_test, y_test), batch_size=256, shuffle=False, num_workers=0)

    model = create_model(args.model_type).to(device)

    # Common patterns:
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # If ckpt is a raw state_dict:
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()) and "state_dict" not in ckpt:
        model.load_state_dict(ckpt, strict=True)
    # If ckpt has a wrapper:
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        raise ValueError(f"Unknown checkpoint format: keys={list(ckpt.keys())[:10]}")

    loss, acc = evaluate(model, loader, device)
    print(f"GLOBAL_TEST loss={loss:.6f} acc={acc:.6f}")

if __name__ == "__main__":
    main()
