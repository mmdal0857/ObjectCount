"""WS 바이너리 프레이밍 — [4바이트 BE 헤더 길이][헤더 JSON][페이로드]. 폰·뷰어 공용."""
from __future__ import annotations

import json
import struct


def pack_frame(header: dict, payload: bytes) -> bytes:
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(header_bytes)) + header_bytes + payload


def unpack_frame(data: bytes) -> tuple[dict, bytes]:
    if len(data) < 4:
        raise ValueError("프레임 메시지가 너무 짧습니다")
    (header_len,) = struct.unpack(">I", data[:4])
    if len(data) < 4 + header_len:
        raise ValueError("헤더 길이가 메시지 크기를 초과합니다")
    header = json.loads(data[4:4 + header_len].decode("utf-8"))
    return header, data[4 + header_len:]
