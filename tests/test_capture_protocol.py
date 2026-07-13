import pytest

from objectcount.capture.protocol import pack_frame, unpack_frame


def test_roundtrip():
    header = {"seq": 7, "client_ts": 1.5, "w": 64, "h": 48}
    payload = b"\xff\xd8jpeg-bytes"
    header2, payload2 = unpack_frame(pack_frame(header, payload))
    assert header2 == header
    assert payload2 == payload


def test_korean_header_roundtrip():
    header, payload = unpack_frame(pack_frame({"메모": "한글"}, b""))
    assert header == {"메모": "한글"}
    assert payload == b""


def test_too_short_raises():
    with pytest.raises(ValueError):
        unpack_frame(b"\x00\x01")


def test_header_length_overflow_raises():
    with pytest.raises(ValueError):
        unpack_frame(b"\x00\x00\xff\xff{}")
