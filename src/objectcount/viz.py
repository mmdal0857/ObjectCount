"""감지 결과 시각화."""
from __future__ import annotations

import cv2
import numpy as np

from .detection.types import Detection

BOX_COLOR = (0, 200, 0)      # BGR
TEXT_COLOR = (0, 0, 0)


def draw_detections(image_bgr: np.ndarray,
                    detections: list[Detection]) -> np.ndarray:
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det.to_pixels(width, height)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{det.class_name} {det.score:.2f}"
        cv2.rectangle(canvas, (x1, max(0, y1 - 18)),
                      (x1 + 8 * len(label), y1), BOX_COLOR, -1)
        cv2.putText(canvas, label, (x1 + 2, max(10, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    return canvas
