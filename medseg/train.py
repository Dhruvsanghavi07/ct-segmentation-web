"""Train the U-Net on preprocessed slices; keep the checkpoint with the best validation 3D Dice.

python -m medseg.train data/processed --out runs/latest --epochs 30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import load_split
from .metrics import dice
from .model import DiceCELoss, UNet, n_params


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class SliceDataset(Dataset):
    """Slices as (1, H, W) float tensors in [0, 1]. Training augmentation is deliberately anatomy-safe:
    no left-right flips (the spleen is always on the patient's left), only small shifts and
    intensity jitter that mimic scanner and positioning differences."""

    def __init__(self, split: dict[str, np.ndarray], augment: bool = False, max_shift: int = 8, seed: int = 0):
        self.x, self.y = split["images"], split["masks"]
        self.augment, self.max_shift = augment, max_shift
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        x = self.x[i].astype(np.float32) / 255.0
        y = self.y[i].astype(np.int64)
        if self.augment:
            dy, dx = self.rng.integers(-self.max_shift, self.max_shift + 1, size=2)
            x, y = np.roll(x, (dy, dx), axis=(0, 1)), np.roll(y, (dy, dx), axis=(0, 1))
            x = np.clip(x * self.rng.uniform(0.9, 1.1) + self.rng.uniform(-0.05, 0.05), 0.0, 1.0)
        return torch.from_numpy(np.ascontiguousarray(x))[None], torch.from_numpy(np.ascontiguousarray(y))


@torch.no_grad()
def predict(model: torch.nn.Module, images: np.ndarray, batch: int = 64) -> np.ndarray:
    """uint8 images (N, H, W) -> predicted foreground masks (N, H, W) bool."""
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i : i + batch].astype(np.float32) / 255.0)[:, None]
        out.append(model(x).argmax(1).numpy().astype(bool))
    return np.concatenate(out) if out else np.zeros((0, *images.shape[1:]), bool)


def volume_dice(model: torch.nn.Module, split: dict[str, np.ndarray]) -> list[float]:
    pred = predict(model, split["images"])
    return [dice(pred[split["case"] == c], split["masks"][split["case"] == c]) for c in np.unique(split["case"])]


def train(
    processed: Path,
    out: Path,
    epochs: int = 30,
    batch: int = 16,
    lr: float = 2e-3,
    base: int = 16,
    seed: int = 0,
    patience: int = 8,
    max_minutes: float = 0,
) -> dict:
    seed_everything(seed)
    torch.set_num_threads(os.cpu_count() or 1)
    out.mkdir(parents=True, exist_ok=True)
    tr, va = load_split(processed, "train"), load_split(processed, "val")
    loader = DataLoader(
        SliceDataset(tr, augment=True, seed=seed),
        batch_size=batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        drop_last=True,
    )
    model = UNet(base=base)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    loss_fn = DiceCELoss()
    print(f"train slices {len(tr['images'])}, val volumes {len(np.unique(va['case']))}, params {n_params(model):,}")

    history, best, since_best, t_start = [], -1.0, 0, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(x)
        vd = volume_dice(model, va)
        row = {
            "epoch": epoch,
            "train_loss": round(total / (len(loader) * batch), 4),
            "val_dice": round(float(np.mean(vd)), 4),
            "lr": round(sched.get_last_lr()[0], 6),
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_dice"] > best:
            best, since_best = row["val_dice"], 0
            torch.save({"state_dict": model.state_dict(), "base": base, "epoch": epoch}, out / "best.pt")
        else:
            since_best += 1
        if since_best >= patience or (max_minutes and time.time() - t_start > max_minutes * 60):
            print(f"stopping after epoch {epoch} (best val Dice {best})")
            break

    summary = {
        "best_val_dice": best,
        "best_epoch": max(history, key=lambda r: r["val_dice"])["epoch"],
        "epochs_run": len(history),
        "train_slices": int(len(tr["images"])),
        "params": n_params(model),
        "minutes": round((time.time() - t_start) / 60, 1),
        "batch": batch,
        "lr": lr,
        "base": base,
        "seed": seed,
    }
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_model(path: Path) -> UNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = UNet(base=ckpt["base"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("processed", type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/latest"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--max-minutes", type=float, default=0, help="stop early to fit a CI time budget (0 = no limit)")
    a = p.parse_args(argv)
    print(
        json.dumps(
            train(a.processed, a.out, a.epochs, a.batch, a.lr, a.base, a.seed, a.patience, a.max_minutes), indent=2
        )
    )


if __name__ == "__main__":
    main()
