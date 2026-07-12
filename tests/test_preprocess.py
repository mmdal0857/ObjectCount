import numpy as np

from objectcount.detection.preprocess import LetterboxMeta, letterbox


def test_letterbox_output_shape_and_dtype():
    image = np.zeros((50, 100, 3), dtype=np.uint8)  # h=50, w=100
    tensor, meta = letterbox(image, size=(64, 64))
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == np.float32


def test_letterbox_meta_scale_and_padding():
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    _, meta = letterbox(image, size=(64, 64))
    assert meta == LetterboxMeta(scale=0.64, pad_x=0, pad_y=16,
                                 src_width=100, src_height=50)


def test_letterbox_pads_with_gray_114():
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    tensor, _ = letterbox(image, size=(64, 64))
    pad_value = 114.0 / 255.0
    assert np.allclose(tensor[0, :, 0, :], pad_value)   # 위쪽 패딩 행
    assert np.allclose(tensor[0, :, 63, :], pad_value)  # 아래쪽 패딩 행


def test_letterbox_square_input_no_padding():
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    tensor, meta = letterbox(image, size=(64, 64))
    assert meta.scale == 1.0 and meta.pad_x == 0 and meta.pad_y == 0
    assert np.allclose(tensor, 1.0)
