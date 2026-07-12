"""YOLOv8/11 ONNX 출력 디코드와 NMS. 전부 numpy — torch 의존 없음."""
from __future__ import annotations

import numpy as np

from .preprocess import LetterboxMeta
from .types import Detection


def decode_yolo(
    raw: np.ndarray,
    meta: LetterboxMeta,
    class_names: list[str],
    conf_threshold: float = 0.25,
    active_class_ids: list[int] | None = None,
) -> list[Detection]:
    pred = raw[0]                      # (4+nc, N)
    boxes = pred[:4, :]                # cx, cy, w, h — letterbox 픽셀
    scores = pred[4:, :]               # (nc, N)
    class_ids = scores.argmax(axis=0)
    confs = scores.max(axis=0)

    keep = confs >= conf_threshold
    if active_class_ids is not None:
        keep &= np.isin(class_ids, active_class_ids)

    detections: list[Detection] = []
    for cx, cy, w, h, conf, cid in zip(
        boxes[0, keep], boxes[1, keep], boxes[2, keep], boxes[3, keep],
        confs[keep], class_ids[keep],
    ):
        # letterbox 픽셀 → 원본 픽셀 → [0,1] 정규화
        x1 = (cx - w / 2 - meta.pad_x) / meta.scale / meta.src_width
        y1 = (cy - h / 2 - meta.pad_y) / meta.scale / meta.src_height
        x2 = (cx + w / 2 - meta.pad_x) / meta.scale / meta.src_width
        y2 = (cy + h / 2 - meta.pad_y) / meta.scale / meta.src_height
        detections.append(Detection(
            x1=float(np.clip(x1, 0.0, 1.0)),
            y1=float(np.clip(y1, 0.0, 1.0)),
            x2=float(np.clip(x2, 0.0, 1.0)),
            y2=float(np.clip(y2, 0.0, 1.0)),
            score=float(conf),
            class_id=int(cid),
            class_name=class_names[int(cid)],
        ))
    return detections


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        if all(k.class_id != det.class_id or _iou(k, det) < iou_threshold
               for k in kept):
            kept.append(det)
    return kept
