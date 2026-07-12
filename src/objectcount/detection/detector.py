"""ONNX 감지기 — 전처리·추론·후처리 조립. 실행 프로바이더 자동 선택 (스펙 §4-④)."""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

from ..models.package import ModelPackage, ModelPackageError
from .postprocess import decode_yolo, nms
from .preprocess import letterbox
from .types import Detection

_PROVIDER_PRIORITY = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)


def select_providers(available: list[str] | None = None) -> list[str]:
    if available is None:
        available = ort.get_available_providers()
    chosen = [p for p in _PROVIDER_PRIORITY if p in available]
    return chosen or ["CPUExecutionProvider"]


class OnnxDetector:
    def __init__(
        self,
        package: ModelPackage,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        providers: list[str] | None = None,
    ) -> None:
        self.package = package
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        try:
            self._session = ort.InferenceSession(
                str(package.model_path),
                providers=providers if providers is not None else select_providers(),
            )
        except Exception as error:
            raise ModelPackageError(
                f"model.onnx 로드 실패 ({package.model_path}): {error}") from error
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        tensor, meta = letterbox(image_bgr, self.package.input_size)
        raw = self._session.run(None, {self._input_name: tensor})[0]
        detections = decode_yolo(
            raw, meta, self.package.classes,
            conf_threshold=self.conf_threshold,
            active_class_ids=self.package.active_class_ids,
        )
        return nms(detections, self.iou_threshold)
