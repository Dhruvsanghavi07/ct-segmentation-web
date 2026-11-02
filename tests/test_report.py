import json

from medseg.report import collect, markdown


def test_report_collects_and_renders(tmp_path):
    case = {"case": "spleen_1", "dice": 0.9, "dice_raw": 0.85, "hd95_mm": 4.2}
    stat = {"mean": 0.9, "std": 0.02, "median": 0.9, "min": 0.88, "n": 1}
    res = {"patients": 1, "dice": stat, "dice_raw": stat, "hd95_mm": stat, "hd95_mm_raw": stat, "per_case": [case]}
    files = {
        "test_results.json": res,
        "val_results.json": res,
        "train_summary.json": {"epochs_run": 3, "train_slices": 100, "minutes": 1.5, "params": 1000},
        "export.json": {"onnx_mb": 7.4, "cpu_ms_per_slice": 3.1, "max_abs_logit_diff": 1e-5, "class_agreement": 1.0},
    }
    for name, data in files.items():
        (tmp_path / name).write_text(json.dumps(data))
    m = collect(tmp_path)
    md = markdown(m)
    assert "0.900 ± 0.020" in md and "| spleen_1 | 0.900 |" in md and "7.4 MB" in md
    json.dumps(m)  # serializable for the web page
