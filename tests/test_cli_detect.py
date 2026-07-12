import json

import cv2
import numpy as np

from objectcount.cli.detect import main


def test_detect_cli_end_to_end(tiny_package, tmp_path):
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))
    out_dir = tmp_path / "out"

    exit_code = main(["--package", str(tiny_package),
                      "--input", str(image_path),
                      "--out", str(out_dir)])

    assert exit_code == 0
    lines = (out_dir / "detections.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["source"] == "frame.png" and record["frame"] == 0
    assert len(record["detections"]) == 1
    assert record["detections"][0]["class_name"] == "can"
    assert (out_dir / "annotated" / "frame.png_0.jpg").is_file()


def test_detect_cli_roi_filters_all(tiny_package, tmp_path):
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))
    out_dir = tmp_path / "out"

    # 상수 모델의 박스 중심은 (0.5, 0.5) — 왼쪽 위 사분면 ROI로 걸러낸다
    main(["--package", str(tiny_package), "--input", str(image_path),
          "--out", str(out_dir), "--roi", "0,0;0.4,0;0.4,0.4;0,0.4"])

    record = json.loads(
        (out_dir / "detections.jsonl").read_text(encoding="utf-8"))
    assert record["detections"] == []
