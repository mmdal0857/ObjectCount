from objectcount.detection.types import Detection


def make(x1=0.25, y1=0.25, x2=0.75, y2=0.75, score=0.9):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score,
                     class_id=0, class_name="can")


def test_area_is_normalized_box_area():
    assert make().area == 0.25


def test_center_is_box_midpoint():
    assert make().center == (0.5, 0.5)


def test_to_pixels_scales_and_rounds():
    assert make().to_pixels(width=200, height=100) == (50, 25, 150, 75)


def test_degenerate_box_has_zero_area():
    assert make(x1=0.8, x2=0.7).area == 0.0
