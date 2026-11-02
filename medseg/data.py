"""Dataset: Medical Segmentation Decathlon Task09 (Spleen), 41 labeled abdominal CT volumes.

Preprocessing, in order:
1. Split by patient (never by slice), so no slice of a test patient is ever seen in training.
   Adjacent CT slices are nearly identical; a slice-level split leaks and inflates Dice.
2. Window Hounsfield units to the abdominal soft-tissue range [-160, 240] and scale to [0, 1].
3. Resize each axial slice to SIZE x SIZE (bilinear for CT, nearest for masks).
4. Training keeps every slice that contains spleen plus a random NEG_RATIO share of empty slices;
   validation and test keep every slice so metrics are computed on whole 3D volumes.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
HU_MIN, HU_MAX = -160.0, 240.0
SIZE = 128
NEG_RATIO = 0.25


def window(ct: np.ndarray, lo: float = HU_MIN, hi: float = HU_MAX) -> np.ndarray:
    """Clip Hounsfield units to [lo, hi] and scale to [0, 1] float32."""
    return ((np.clip(ct, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def resize_slice(img: np.ndarray, size: int, mask: bool = False) -> np.ndarray:
    pil = Image.fromarray(img.astype(np.uint8) if mask else img.astype(np.float32), mode="L" if mask else "F")
    out = pil.resize((size, size), Image.NEAREST if mask else Image.BILINEAR)
    return np.asarray(out, dtype=np.uint8 if mask else np.float32)


def to_display(axial: np.ndarray) -> np.ndarray:
    """NIfTI arrays are (x, y); rotate to the radiology view (anterior up, patient left on image right)."""
    return np.rot90(axial)


def patient_split(case_ids: list[str], seed: int = 0, val: float = 0.15, test: float = 0.15) -> dict[str, list[str]]:
    ids = sorted(case_ids)
    random.Random(seed).shuffle(ids)
    n_test, n_val = max(1, round(len(ids) * test)), max(1, round(len(ids) * val))
    return {
        "test": sorted(ids[:n_test]),
        "val": sorted(ids[n_test : n_test + n_val]),
        "train": sorted(ids[n_test + n_val :]),
    }


def list_cases(root: Path) -> list[str]:
    # The MSD tarball ships macOS "._" resource-fork files next to every volume; skip them.
    return sorted(p.name.split(".nii")[0] for p in (root / "imagesTr").glob("*.nii.gz") if not p.name.startswith("._"))


@dataclass
class Volume:
    case: str
    image: np.ndarray  # (Z, SIZE, SIZE) float32 in [0, 1]
    mask: np.ndarray  # (Z, SIZE, SIZE) uint8 in {0, 1}
    spacing: tuple[float, float, float]  # (z, y, x) mm after in-plane resizing


def load_volume(root: Path, case: str, size: int = SIZE) -> Volume:
    import nibabel as nib

    img = nib.load(str(root / "imagesTr" / f"{case}.nii.gz"))
    lab = nib.load(str(root / "labelsTr" / f"{case}.nii.gz"))
    ct, gt = np.asarray(img.dataobj, dtype=np.float32), np.asarray(lab.dataobj, dtype=np.uint8)
    if ct.shape != gt.shape:
        raise ValueError(f"{case}: image {ct.shape} and label {gt.shape} differ")
    sx, sy, sz = (float(v) for v in img.header.get_zooms()[:3])
    ct = window(ct)
    xs = np.stack([resize_slice(to_display(ct[:, :, k]), size) for k in range(ct.shape[2])])
    ys = np.stack(
        [resize_slice(to_display((gt[:, :, k] > 0).astype(np.uint8)), size, mask=True) for k in range(gt.shape[2])]
    )
    # np.rot90 turns an (X, Y) slice into (Y, X): rows run along y and columns along x.
    # Resizing to SIZE x SIZE then scales each spacing by original/size.
    spacing = (sz, sy * ct.shape[1] / size, sx * ct.shape[0] / size)
    return Volume(case, xs, ys, spacing)


def preprocess(root: Path, out: Path, size: int = SIZE, neg_ratio: float = NEG_RATIO, seed: int = 0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cases = list_cases(root)
    if not cases:
        raise FileNotFoundError(f"no volumes under {root / 'imagesTr'}")
    split = patient_split(cases, seed=seed)
    rng = np.random.default_rng(seed)
    stats = {"cases": len(cases), "size": size, "hu_window": [HU_MIN, HU_MAX], "split": split}
    for name, ids in split.items():
        imgs, masks, case_idx, spacings = [], [], [], []
        for ci, case in enumerate(ids):
            v = load_volume(root, case, size)
            keep = np.arange(len(v.mask))
            if name == "train":
                pos = v.mask.reshape(len(v.mask), -1).any(1)
                neg = np.flatnonzero(~pos)
                keep = np.concatenate([np.flatnonzero(pos), rng.choice(neg, int(len(neg) * neg_ratio), replace=False)])
                keep.sort()
            imgs.append((v.image[keep] * 255).round().astype(np.uint8))
            masks.append(v.mask[keep])
            case_idx.append(np.full(len(keep), ci, dtype=np.int16))
            spacings.append(v.spacing)
        np.savez_compressed(
            out / f"{name}.npz",
            images=np.concatenate(imgs),
            masks=np.concatenate(masks),
            case=np.concatenate(case_idx),
            case_ids=np.array(ids),
            spacing=np.array(spacings),
        )
        stats[f"{name}_slices"] = int(sum(len(m) for m in masks))
    (out / "split.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_split(processed: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(processed / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Preprocess MSD Task09 Spleen into 2D slices")
    p.add_argument("root", type=Path, help="extracted Task09_Spleen folder")
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--size", type=int, default=SIZE)
    a = p.parse_args()
    print(json.dumps(preprocess(a.root, a.out, a.size), indent=2))
