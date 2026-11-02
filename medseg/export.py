"""Export the trained model to ONNX and prove it matches PyTorch before it ships to the browser.

Checks: max absolute logit difference, and the share of pixels whose predicted class agrees, on real
held-out slices. Also records file size and CPU latency per slice with onnxruntime.

    python -m medseg.export runs/latest/best.pt data/processed --out web/model.onnx
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import load_split
from .train import load_model


def export(model: torch.nn.Module, path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, size, size)
    torch.onnx.export(
        model.eval(),
        dummy,
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def verify(model: torch.nn.Module, path: Path, images: np.ndarray, repeats: int = 50) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = images[:, None].astype(np.float32) / 255.0
    with torch.no_grad():
        ref = model.eval()(torch.from_numpy(x)).numpy()
    got = sess.run(["logits"], {"image": x})[0]
    one = x[:1]
    sess.run(["logits"], {"image": one})
    t0 = time.perf_counter()
    for _ in range(repeats):
        sess.run(["logits"], {"image": one})
    return {
        "max_abs_logit_diff": float(np.abs(ref - got).max()),
        "class_agreement": float((ref.argmax(1) == got.argmax(1)).mean()),
        "cpu_ms_per_slice": round((time.perf_counter() - t0) / repeats * 1000, 2),
        "onnx_mb": round(path.stat().st_size / 1e6, 2),
        "checked_slices": int(len(images)),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("processed", type=Path)
    p.add_argument("--out", type=Path, default=Path("web/model.onnx"))
    p.add_argument("--report", type=Path, default=Path("runs/latest/export.json"))
    a = p.parse_args(argv)
    model = load_model(a.checkpoint)
    test = load_split(a.processed, "test")
    export(model, a.out, int(test["images"].shape[-1]))
    rng = np.random.default_rng(0)
    sample = test["images"][rng.choice(len(test["images"]), min(64, len(test["images"])), replace=False)]
    report = verify(model, a.out, sample)
    print(json.dumps(report, indent=2))
    if report["max_abs_logit_diff"] > 1e-3 or report["class_agreement"] < 0.9999:
        raise SystemExit("ONNX output does not match PyTorch; refusing to ship this model")
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
