"""letterbox 전처리 — 종횡비 유지 리사이즈 + 회색(114) 패딩."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAD_COLOR = 114


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    pad_x: int
    pad_y: int
    src_width: int
    src_height: int


def letterbox(image_bgr: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, LetterboxMeta]:
    target_w, target_h = size
    src_h, src_w = image_bgr.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas = np.full((target_h, target_w, 3), PAD_COLOR, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
    meta = LetterboxMeta(scale=scale, pad_x=pad_x, pad_y=pad_y,
                         src_width=src_w, src_height=src_h)
    return np.ascontiguousarray(tensor), meta
