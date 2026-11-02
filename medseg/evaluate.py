"""Evaluate on held-out test patients: per-volume 3D Dice and HD95, with and without post-processing.
Also writes the demo samples (held-out slices as PNG) used by the browser app.

    python -m medseg.evaluate data/processed runs/latest/best.pt --out runs/latest --samples web/samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import load_split
from .metrics import dice, hd95, largest_component, summarize
from .train import load_model, predict


def evaluate(processed: Path, checkpoint: Path, out: Path, split_name: str = "test") -> dict:
    split = load_split(processed, split_name)
    model = load_model(checkpoint)
    pred_all = predict(model, split["images"])
    rows = []
    for ci, case in enumerate(split["case_ids"]):
        sel = split["case"] == ci
        gt, raw = split["masks"][sel].astype(bool), pred_all[sel]
        post = largest_component(raw)
        spacing = tuple(float(s) for s in split["spacing"][ci])
        rows.append(
            {
                "case": str(case),
                "slices": int(sel.sum()),
                "spleen_voxels": int(gt.sum()),
                "dice_raw": round(dice(raw, gt), 4),
                "dice": round(dice(post, gt), 4),
                "hd95_mm_raw": round(hd95(raw, gt, spacing), 2),
                "hd95_mm": round(hd95(post, gt, spacing), 2),
            }
        )
        print(json.dumps(rows[-1]))
    result = {
        "split": split_name,
        "patients": len(rows),
        "dice": summarize([r["dice"] for r in rows]),
        "dice_raw": summarize([r["dice_raw"] for r in rows]),
        "hd95_mm": summarize([r["hd95_mm"] for r in rows]),
        "hd95_mm_raw": summarize([r["hd95_mm_raw"] for r in rows]),
        "per_case": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{split_name}_results.json").write_text(json.dumps(result, indent=2))
    return result


def write_samples(processed: Path, dest: Path, n_cases: int = 3, per_case: int = 12) -> dict:
    """Save evenly spaced slices through the spleen of each held-out test patient: CT and ground-truth mask PNGs."""
    split = load_split(processed, "test")
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "note": "Held-out test patients from MSD Task09 Spleen (CC-BY-SA 4.0). Never used for training.",
        "size": int(split["images"].shape[-1]),
        "cases": [],
    }
    for ci, case in enumerate(split["case_ids"][:n_cases]):
        idx = np.flatnonzero(split["case"] == ci)
        pos = idx[split["masks"][idx].reshape(len(idx), -1).any(1)]
        if not len(pos):
            continue
        chosen = sorted(set(np.linspace(pos[0], pos[-1], per_case).round().astype(int)))
        items = []
        for k, i in enumerate(chosen):
            stem = f"{case}_{k:02d}"
            Image.fromarray(split["images"][i]).save(dest / f"{stem}_ct.png")
            Image.fromarray(split["masks"][i] * 255).save(dest / f"{stem}_gt.png")
            items.append({"ct": f"{stem}_ct.png", "gt": f"{stem}_gt.png", "slice": int(i - idx[0])})
        manifest["cases"].append({"case": str(case), "slices": items})
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("processed", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, default=Path("runs/latest"))
    p.add_argument("--samples", type=Path, default=None)
    a = p.parse_args(argv)
    for name in ("val", "test"):
        r = evaluate(a.processed, a.checkpoint, a.out, name)
        print(
            f"{name}: Dice {r['dice']['mean']} +/- {r['dice']['std']} (raw {r['dice_raw']['mean']}), "
            f"HD95 {r['hd95_mm']['median']} mm median over {r['patients']} patients"
        )
    if a.samples:
        write_samples(a.processed, a.samples)


if __name__ == "__main__":
    main()
