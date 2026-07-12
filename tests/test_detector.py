import json

import numpy as np
import pytest

from objectcount.detection.detector import OnnxDetector, select_providers
from objectcount.models.package import ModelPackageError, load_package

MANIFEST = {
    "schema_version": 1,
    "product_id": "tiny",
    "product_name": "테스트 상수 모델",
    "model_version": "test-1",
    "input_size": [64, 64],
    "classes": ["can", "box"],
}


def test_select_providers_prefers_cuda_then_dml():
    available = ["CPUExecutionProvider", "DmlExecutionProvider",
                 "CUDAExecutionProvider"]
    assert select_providers(available) == [
        "CUDAExecutionProvider", "DmlExecutionProvider",
        "CPUExecutionProvider"]


def test_select_providers_falls_back_to_cpu():
    assert select_providers(["FooProvider"]) == ["CPUExecutionProvider"]


def test_detect_returns_expected_constant_box(tiny_package):
    detector = OnnxDetector(load_package(tiny_package))
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    dets = detector.detect(image)
    assert len(dets) == 1
    d = dets[0]
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx(
        (0.375, 0.375, 0.625, 0.625))
    assert d.class_name == "can" and d.score == pytest.approx(0.9, abs=1e-6)


def test_detect_respects_conf_threshold(tiny_package):
    detector = OnnxDetector(load_package(tiny_package), conf_threshold=0.95)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    assert detector.detect(image) == []


def test_corrupt_model_raises_model_package_error(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    (tmp_path / "model.onnx").write_bytes(b"\x00")
    with pytest.raises(ModelPackageError, match="로드 실패"):
        OnnxDetector(load_package(tmp_path))
