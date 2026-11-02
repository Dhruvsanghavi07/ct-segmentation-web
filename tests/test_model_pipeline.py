"""Model, training, evaluation and ONNX export on a tiny synthetic dataset (needs PyTorch; runs in CI)."""

import json
import os

import numpy as np
import pytest

if os.environ.get("REQUIRE_TORCH"):  # CI sets this so these tests can never be skipped silently
    import torch
else:
    torch = pytest.importorskip("torch")

from medseg.data import preprocess  # noqa: E402
from medseg.evaluate import evaluate, write_samples  # noqa: E402
from medseg.model import DiceCELoss, UNet, n_params  # noqa: E402
from medseg.train import load_model, predict, train  # noqa: E402
from tests.test_metrics_data import make_fake_dataset  # noqa: E402


def test_unet_shapes_and_size():
    m = UNet(base=8)
    assert m(torch.zeros(2, 1, 64, 64)).shape == (2, 2, 64, 64)
    assert n_params(UNet(base=16)) < 3_000_000  # small enough for a browser


def test_loss_is_lower_for_a_correct_prediction():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[0, 2:6, 2:6] = 1
    good = torch.stack([(1 - target.float()) * 10, target.float() * 10], 1)
    bad = good.flip(1)
    loss = DiceCELoss()
    assert loss(good, target) < loss(bad, target)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    root = make_fake_dataset(tmp / "Task09_Spleen", n=10)
    proc = tmp / "proc"
    preprocess(root, proc, size=32, neg_ratio=0.5)
    summary = train(proc, tmp / "out", epochs=25, batch=4, lr=5e-3, base=8, patience=25)
    return tmp, proc, summary


def test_training_learns_the_synthetic_organ(trained):
    tmp, proc, summary = trained
    assert summary["best_val_dice"] > 0.8, summary
    hist = json.loads((tmp / "out" / "history.json").read_text())
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]


def test_evaluate_and_samples(trained):
    tmp, proc, _ = trained
    r = evaluate(proc, tmp / "out" / "best.pt", tmp / "out", "test")
    assert r["patients"] >= 1 and r["dice"]["mean"] > 0.8
    assert {"dice", "dice_raw", "hd95_mm", "per_case"} <= r.keys()
    m = write_samples(proc, tmp / "samples", per_case=3)
    first = m["cases"][0]["slices"][0]
    assert (tmp / "samples" / first["ct"]).exists() and (tmp / "samples" / first["gt"]).exists()


def test_onnx_export_matches_pytorch(trained, tmp_path):
    if not os.environ.get("REQUIRE_TORCH"):
        pytest.importorskip("onnxruntime")
    from medseg.export import export, verify

    tmp, proc, _ = trained
    model = load_model(tmp / "out" / "best.pt")
    images = np.load(proc / "test.npz")["images"]
    export(model, tmp_path / "m.onnx", images.shape[-1])
    report = verify(model, tmp_path / "m.onnx", images, repeats=3)
    assert report["max_abs_logit_diff"] < 1e-3 and report["class_agreement"] > 0.9999


def test_predict_handles_empty_input():
    assert predict(UNet(base=8), np.zeros((0, 32, 32), np.uint8)).shape == (0, 32, 32)
