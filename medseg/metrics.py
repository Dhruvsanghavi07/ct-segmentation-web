"""Segmentation metrics computed on whole 3D volumes, the way they are reported in the literature.

- Dice = 2|A n B| / (|A| + |B|); two empty masks score 1.0 (nothing to find, nothing found).
- HD95: 95th percentile of surface-to-surface distances in millimetres, symmetric. Reported
  as NaN when one mask is empty (the distance is undefined).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(pred, gt).sum() / denom)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(mask.ndim, 1))


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, ...]) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    d_to_gt = ndimage.distance_transform_edt(~sg, sampling=spacing)[sp]
    d_to_pred = ndimage.distance_transform_edt(~sp, sampling=spacing)[sg]
    return float(np.percentile(np.concatenate([d_to_gt, d_to_pred]), 95))


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 3D connected component: a patient has one spleen, so isolated blobs are false positives."""
    labels, n = ndimage.label(mask.astype(bool))
    if n <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def summarize(values: list[float]) -> dict[str, float]:
    v = np.array([x for x in values if not np.isnan(x)], dtype=float)
    if not len(v):
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "min": float("nan"), "n": 0}
    return {
        "mean": round(float(v.mean()), 4),
        "std": round(float(v.std()), 4),
        "median": round(float(np.median(v)), 4),
        "min": round(float(v.min()), 4),
        "n": int(len(v)),
    }
