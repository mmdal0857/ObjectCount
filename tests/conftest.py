"""테스트 공용 픽스처 — 입력과 무관하게 상수 출력을 내는 초소형 ONNX 감지 모델.

출력 (1, 6, 1): [cx=32, cy=32, w=16, h=16, score_can=0.9, score_box=0.1]
64x64 입력 기준 → 정규화 (0.375, 0.375, 0.625, 0.625)의 can 1개.
"""
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

RAW_OUTPUT = np.array([[[32.0], [32.0], [16.0], [16.0], [0.9], [0.1]]],
                      dtype=np.float32)  # (1, 6, 1)

MANIFEST = {
    "schema_version": 1,
    "product_id": "tiny",
    "product_name": "테스트 상수 모델",
    "model_version": "test-1",
    "input_size": [64, 64],
    "classes": ["can", "box"],
}


def build_constant_onnx(path: Path, raw_output: np.ndarray) -> None:
    """output = const + 0 * ReduceSum(input) — 입력을 소비하되 항상 상수를 낸다."""
    inp = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 64, 64])
    out = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, list(raw_output.shape))
    const = helper.make_tensor("const_val", TensorProto.FLOAT,
                               raw_output.shape,
                               raw_output.flatten().tolist())
    zero = helper.make_tensor("zero_val", TensorProto.FLOAT, [], [0.0])
    nodes = [
        helper.make_node("Constant", [], ["const_out"], value=const),
        helper.make_node("Constant", [], ["zero"], value=zero),
        helper.make_node("ReduceSum", ["images"], ["input_sum"], keepdims=0),
        helper.make_node("Mul", ["input_sum", "zero"], ["zeroed"]),
        helper.make_node("Add", ["const_out", "zeroed"], ["output0"]),
    ]
    graph = helper.make_graph(nodes, "constant-detector", [inp], [out])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture
def tiny_package(tmp_path: Path) -> Path:
    """상수 출력 ONNX가 담긴 완전한 모델 패키지 디렉토리를 만든다."""
    (tmp_path / "manifest.json").write_text(
        json.dumps(MANIFEST), encoding="utf-8")
    build_constant_onnx(tmp_path / "model.onnx", RAW_OUTPUT)
    return tmp_path
