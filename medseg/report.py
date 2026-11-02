"""Collect one run's results into web/metrics.json (read by the demo) and a Markdown summary.

python -m medseg.report runs/latest --web web --markdown runs/latest/SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect(run: Path) -> dict:
    load = lambda name: json.loads((run / name).read_text())  # noqa: E731
    test, val = load("test_results.json"), load("val_results.json")
    return {
        "test": {k: test[k] for k in ("patients", "dice", "dice_raw", "hd95_mm", "hd95_mm_raw")},
        "val": {k: val[k] for k in ("patients", "dice", "hd95_mm")},
        "per_case": test["per_case"],
        "train": load("train_summary.json"),
        "export": load("export.json"),
    }


def markdown(m: dict) -> str:
    t, tr, ex = m["test"], m["train"], m["export"]
    rows = "\n".join(
        f"| {r['case']} | {r['dice']:.3f} | {r['dice_raw']:.3f} | {r['hd95_mm']:.1f} |" for r in m["per_case"]
    )
    return f"""## Results on {t["patients"]} held-out test patients

| Metric | Value |
|---|---|
| 3D Dice, mean ± std | {t["dice"]["mean"]:.3f} ± {t["dice"]["std"]:.3f} |
| 3D Dice without post-processing | {t["dice_raw"]["mean"]:.3f} |
| HD95, median | {t["hd95_mm"]["median"]:.1f} mm |
| Validation 3D Dice ({m["val"]["patients"]} patients) | {m["val"]["dice"]["mean"]:.3f} |
| Training | {tr["epochs_run"]} epochs, {tr["train_slices"]:,} slices, {tr["minutes"]} min on a CI CPU |
| Parameters | {tr["params"]:,} |
| ONNX model | {ex["onnx_mb"]:.1f} MB, {ex["cpu_ms_per_slice"]:.1f} ms per slice (onnxruntime CPU) |
| ONNX vs PyTorch | max logit difference {ex["max_abs_logit_diff"]:.1e}, class agreement {ex["class_agreement"]:.4%} |

| Test patient | Dice | Dice (raw) | HD95 (mm) |
|---|---|---|---|
{rows}
"""


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--web", type=Path, default=Path("web"))
    p.add_argument("--markdown", type=Path, default=None)
    a = p.parse_args(argv)
    m = collect(a.run)
    (a.web / "metrics.json").write_text(json.dumps(m, indent=2))
    md = markdown(m)
    if a.markdown:
        a.markdown.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
