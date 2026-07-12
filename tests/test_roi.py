import pytest

from objectcount.detection.roi import RoiPolygon, parse_roi
from objectcount.detection.types import Detection

SQUARE = RoiPolygon([(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])


def _det(cx, cy, half=0.05):
    return Detection(cx - half, cy - half, cx + half, cy + half,
                     0.9, 0, "can")


def test_contains_inside_and_outside():
    assert SQUARE.contains(0.75, 0.75)
    assert not SQUARE.contains(0.25, 0.25)


def test_filter_keeps_detections_with_center_inside():
    dets = [_det(0.75, 0.75), _det(0.25, 0.25)]
    kept = SQUARE.filter(dets)
    assert kept == [dets[0]]


def test_polygon_requires_three_points():
    with pytest.raises(ValueError):
        RoiPolygon([(0, 0), (1, 1)])


def test_parse_roi_string():
    roi = parse_roi("0.5,0.5;1,0.5;1,1;0.5,1")
    assert roi.contains(0.75, 0.75)
    assert not roi.contains(0.1, 0.1)
