import json

import cv2
import numpy as np

from objectcount.cli.evaluate import evaluate_dataset, main

MANIFEST = {
    "schema_version": 1,
    "product_id": "tiny",
    "product_name": "테스트 상수 모델",
    "model_version": "test-1",
    "input_size": [64, 64],
    "classes": ["can", "box"],
}


def make_dataset(tmp_path, expected):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name in expected:
        cv2.imwrite(str(dataset / name),
                    np.zeros((64, 64, 3), dtype=np.uint8))
    (dataset / "expected_counts.json").write_text(
        json.dumps(expected), encoding="utf-8")
    return dataset


def test_evaluate_reports_match(tiny_package, tmp_path):
    # 상수 모델은 이미지당 정확히 1건 감지한다
    dataset = make_dataset(tmp_path, {"a.png": 1, "b.png": 2})
    report = evaluate_dataset(tiny_package, dataset)
    assert report["per_image"]["a.png"] == {"expected": 1, "detected": 1}
    assert report["per_image"]["b.png"] == {"expected": 2, "detected": 1}
    assert report["matched"] == 1 and report["total"] == 2


def test_main_prints_summary_and_exits_zero(tiny_package, tmp_path, capsys):
    dataset = make_dataset(tmp_path, {"a.png": 1})
    exit_code = main(["--package", str(tiny_package),
                      "--dataset", str(dataset)])
    assert exit_code == 0
    assert "1/1" in capsys.readouterr().out


def test_missing_expected_counts_reports_error_exits_zero(tiny_package, tmp_path, capsys):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    cv2.imwrite(str(dataset / "a.png"), np.zeros((64, 64, 3), dtype=np.uint8))
    exit_code = main(["--package", str(tiny_package), "--dataset", str(dataset)])
    assert exit_code == 0
    assert "오류" in capsys.readouterr().err


def test_unlisted_images_are_skipped(tiny_package, tmp_path):
    dataset = make_dataset(tmp_path, {"a.png": 1})
    cv2.imwrite(str(dataset / "z.png"), np.zeros((64, 64, 3), dtype=np.uint8))
    report = evaluate_dataset(tiny_package, dataset)
    assert list(report["per_image"].keys()) == ["a.png"]
    assert report["total"] == 1


def test_corrupt_model_reports_error_exits_zero(tmp_path, capsys):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    (package / "model.onnx").write_bytes(b"\x00")
    dataset = make_dataset(tmp_path, {"a.png": 1})
    exit_code = main(["--package", str(package), "--dataset", str(dataset)])
    assert exit_code == 0
    assert "오류" in capsys.readouterr().err
