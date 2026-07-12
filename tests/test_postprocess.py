import numpy as np
import pytest

from objectcount.detection.postprocess import decode_yolo, nms
from objectcount.detection.preprocess import LetterboxMeta
from objectcount.detection.types import Detection

CLASSES = ["can", "box"]
# 64x64 원본을 64x64로 letterbox → scale 1, 패딩 0
META = LetterboxMeta(scale=1.0, pad_x=0, pad_y=0, src_width=64, src_height=64)


def make_raw(rows: list[list[float]]) -> np.ndarray:
    """rows: [cx, cy, w, h, score_can, score_box] (letterbox 픽셀 좌표) → (1, 6, N)"""
    return np.array(rows, dtype=np.float32).T[np.newaxis, ...]


def test_decode_converts_center_box_to_normalized_xyxy():
    raw = make_raw([[32, 32, 16, 16, 0.9, 0.1]])
    dets = decode_yolo(raw, META, CLASSES)
    assert len(dets) == 1
    d = dets[0]
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx((0.375, 0.375, 0.625, 0.625))
    assert d.score == pytest.approx(0.9)
    assert d.class_id == 0 and d.class_name == "can"


def test_decode_filters_below_confidence():
    raw = make_raw([[32, 32, 16, 16, 0.1, 0.05]])
    assert decode_yolo(raw, META, CLASSES, conf_threshold=0.25) == []


def test_decode_respects_active_class_filter():
    raw = make_raw([
        [32, 32, 16, 16, 0.9, 0.1],   # can
        [10, 10, 8, 8, 0.1, 0.8],     # box
    ])
    dets = decode_yolo(raw, META, CLASSES, active_class_ids=[1])
    assert [d.class_name for d in dets] == ["box"]


def test_decode_unmaps_letterbox_padding():
    # 원본 100x50 → 64x64 letterbox: scale 0.64, pad_y 16
    meta = LetterboxMeta(scale=0.64, pad_x=0, pad_y=16, src_width=100, src_height=50)
    # letterbox 픽셀 (32, 32) = 원본 픽셀 (50, 25) = 정규화 (0.5, 0.5)
    raw = make_raw([[32, 32, 12.8, 6.4, 0.9, 0.0]])
    d = decode_yolo(raw, meta, CLASSES)[0]
    assert d.center == pytest.approx((0.5, 0.5))
    assert (d.x2 - d.x1) == pytest.approx(0.2)   # 12.8/0.64/100
    assert (d.y2 - d.y1) == pytest.approx(0.2)   # 6.4/0.64/50


def test_decode_clips_to_unit_range():
    raw = make_raw([[2, 2, 20, 20, 0.9, 0.0]])  # 박스가 프레임 밖으로 나감
    d = decode_yolo(raw, META, CLASSES)[0]
    assert d.x1 == 0.0 and d.y1 == 0.0


def _det(x1, y1, x2, y2, score, cid=0):
    return Detection(x1, y1, x2, y2, score, cid, CLASSES[cid])


def test_nms_suppresses_overlapping_same_class():
    kept = nms([_det(0.1, 0.1, 0.5, 0.5, 0.9),
                _det(0.12, 0.12, 0.52, 0.52, 0.8)], iou_threshold=0.45)
    assert len(kept) == 1 and kept[0].score == 0.9


def test_nms_keeps_different_classes():
    kept = nms([_det(0.1, 0.1, 0.5, 0.5, 0.9, cid=0),
                _det(0.1, 0.1, 0.5, 0.5, 0.8, cid=1)])
    assert len(kept) == 2


def test_nms_keeps_disjoint_boxes():
    kept = nms([_det(0.0, 0.0, 0.2, 0.2, 0.9),
                _det(0.5, 0.5, 0.9, 0.9, 0.8)])
    assert len(kept) == 2


def test_decode_rejects_class_count_mismatch():
    raw = make_raw([[32, 32, 16, 16, 0.9, 0.1]])  # nc=2
    with pytest.raises(ValueError, match="클래스 수"):
        decode_yolo(raw, META, ["can"])  # manifest엔 1개
