"""Metric and preprocessing tests. No PyTorch needed, so these run anywhere."""

import json

import numpy as np
import pytest

from medseg.data import list_cases, patient_split, preprocess, to_display, window
from medseg.metrics import dice, hd95, largest_component, summarize


def cube(shape, lo, hi):
    m = np.zeros(shape, bool)
    m[tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))] = True
    return m


def test_dice_cases():
    a = cube((10, 10, 10), (2, 2, 2), (6, 6, 6))
    assert dice(a, a) == 1.0
    assert dice(a, np.zeros_like(a)) == 0.0
    assert dice(np.zeros_like(a), np.zeros_like(a)) == 1.0  # nothing to find, nothing found
    half = cube((10, 10, 10), (2, 2, 2), (4, 6, 6))  # half of a
    assert dice(half, a) == pytest.approx(2 * 32 / (32 + 64))


def test_hd95_matches_a_known_shift_and_uses_spacing():
    a = cube((20, 20, 20), (5, 5, 5), (10, 10, 10))
    b = cube((20, 20, 20), (7, 5, 5), (12, 10, 10))  # shifted 2 voxels along z
    assert hd95(a, a, (1, 1, 1)) == 0.0
    assert hd95(a, b, (1, 1, 1)) == pytest.approx(2.0)
    assert hd95(a, b, (3, 1, 1)) == pytest.approx(6.0)  # 2 voxels x 3 mm
    assert np.isnan(hd95(a, np.zeros_like(a), (1, 1, 1)))


def test_largest_component_removes_islands():
    m = cube((20, 20, 20), (2, 2, 2), (10, 10, 10)) | cube((20, 20, 20), (15, 15, 15), (17, 17, 17))
    kept = largest_component(m)
    assert kept.sum() == 8**3 and not kept[16, 16, 16]
    assert not largest_component(np.zeros_like(m)).any()


def test_summarize_ignores_nan():
    s = summarize([0.9, float("nan"), 0.7])
    assert s["n"] == 2 and s["mean"] == pytest.approx(0.8)


def test_window_clips_and_scales():
    out = window(np.array([-1000.0, -160.0, 40.0, 240.0, 3000.0]))
    np.testing.assert_allclose(out, [0, 0, 0.5, 1, 1])
    assert out.dtype == np.float32


def test_patient_split_is_disjoint_complete_and_deterministic():
    ids = [f"spleen_{i}" for i in range(41)]
    s = patient_split(ids, seed=0)
    groups = [set(s[k]) for k in ("train", "val", "test")]
    assert set().union(*groups) == set(ids)
    assert all(not (a & b) for i, a in enumerate(groups) for b in groups[i + 1 :])
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (29, 6, 6)
    assert patient_split(ids, seed=0) == s and patient_split(ids, seed=1) != s


def make_fake_dataset(root, n=8, shape=(32, 32, 10), seed=0):
    """Tiny NIfTI volumes with a bright 'organ' block, plus the macOS '._' junk files the real tarball contains."""
    nib = pytest.importorskip("nibabel")
    rng = np.random.default_rng(seed)
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    affine = np.diag([0.8, 0.8, 5.0, 1.0])
    for i in range(n):
        ct = rng.normal(-500, 50, shape).astype(np.float32)
        lab = np.zeros(shape, np.uint8)
        lab[8:20, 10:22, 3:7] = 1
        ct[lab == 1] = 60
        nib.save(nib.Nifti1Image(ct, affine), str(root / "imagesTr" / f"spleen_{i}.nii.gz"))
        nib.save(nib.Nifti1Image(lab, affine), str(root / "labelsTr" / f"spleen_{i}.nii.gz"))
    (root / "imagesTr" / "._spleen_0.nii.gz").write_bytes(b"junk")
    return root


def test_preprocess_end_to_end(tmp_path):
    root = make_fake_dataset(tmp_path / "Task09_Spleen")
    assert len(list_cases(root)) == 8  # the "._" file is skipped
    stats = preprocess(root, tmp_path / "proc", size=16, neg_ratio=0.5)
    split = json.loads((tmp_path / "proc" / "split.json").read_text())["split"]
    assert not set(split["train"]) & set(split["test"])
    tr = np.load(tmp_path / "proc" / "train.npz")
    te = np.load(tmp_path / "proc" / "test.npz")
    assert tr["images"].shape[1:] == (16, 16) and tr["images"].dtype == np.uint8
    # every slice with organ is kept for training; test keeps whole volumes (10 slices each)
    n_train = len(split["train"])
    assert tr["masks"].reshape(len(tr["masks"]), -1).any(1).sum() == 4 * n_train
    assert len(te["images"]) == 10 * len(split["test"])
    assert stats["test_slices"] == len(te["images"])
    # spacing: 32 px at 0.8 mm resized to 16 px -> 1.6 mm in-plane; slices 5 mm apart
    np.testing.assert_allclose(te["spacing"][0], [5.0, 1.6, 1.6])
    # organ is bright after windowing (60 HU -> 0.55 * 255) and air-like background is black
    assert tr["images"][tr["masks"] == 1].mean() > 120 and tr["images"][tr["masks"] == 0].mean() < 30


def test_display_orientation_follows_radiology_convention():
    # axial slice indexed (x, y) in RAS orientation: x increases toward the patient's right, y toward anterior
    a = np.zeros((4, 3))
    a[0, 2] = 1  # far patient-left, far anterior
    out = to_display(a)
    assert out.shape == (3, 4)
    assert out[0, 3] == 1  # anterior at the top, patient's left on the image right
