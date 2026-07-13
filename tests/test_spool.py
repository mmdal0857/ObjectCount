"""캡처 세션 스풀 — 기록·리플레이 왕복."""
import json

import cv2
import numpy as np
import pytest

from objectcount.frames.sources import iter_frames
from objectcount.frames.spool import SpoolWriter, iter_spool


def make_jpeg(value: int = 128) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def test_roundtrip_with_gap(tmp_path):
    session = tmp_path / "20260713-120000-phone"
    writer = SpoolWriter(session)
    writer.write_session_info({"device": "test", "started_at": 100.0})
    writer.append(1, make_jpeg(10), server_ts=100.1, client_ts=99.9, width=64, height=48)
    writer.append(2, make_jpeg(20), server_ts=100.3)
    writer.append(5, make_jpeg(50), server_ts=100.9)  # 3-4 공백
    writer.close()

    records = list(iter_spool(session))
    assert [r.frame_index for r in records] == [1, 2, 5]
    assert [r.timestamp for r in records] == [100.1, 100.3, 100.9]
    assert all(r.source_id == "20260713-120000-phone" for r in records)
    assert records[0].image.shape == (48, 64, 3)

    info = json.loads((session / "session.json").read_text(encoding="utf-8"))
    assert info["device"] == "test"


def test_iter_frames_dispatches_to_spool(tmp_path):
    session = tmp_path / "cap"
    writer = SpoolWriter(session)
    writer.append(1, make_jpeg(), server_ts=42.0)
    writer.close()

    records = list(iter_frames(session))
    assert records[0].timestamp == 42.0
    assert records[0].frame_index == 1


def test_missing_frame_file_skipped(tmp_path):
    session = tmp_path / "cap"
    writer = SpoolWriter(session)
    writer.append(1, make_jpeg(), server_ts=1.0)
    writer.append(2, make_jpeg(), server_ts=2.0)
    writer.close()
    (session / "frames" / "000001.jpg").unlink()

    records = list(iter_spool(session))
    assert [r.frame_index for r in records] == [2]


def test_reopen_appends(tmp_path):
    session = tmp_path / "cap"
    writer = SpoolWriter(session)
    writer.append(1, make_jpeg(), server_ts=1.0)
    writer.close()
    writer2 = SpoolWriter(session)  # 재접속 이어쓰기
    writer2.append(2, make_jpeg(), server_ts=2.0)
    writer2.close()

    assert [r.frame_index for r in iter_spool(session)] == [1, 2]
