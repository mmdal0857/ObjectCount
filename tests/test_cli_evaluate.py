import json

import cv2
import numpy as np

from objectcount.cli.evaluate import evaluate_dataset, main


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
