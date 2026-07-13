"""캡처 서버 통합 — 가짜 폰 클라이언트로 스풀 기록·ack·공백 통계 검증."""
import json

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from objectcount.capture.protocol import pack_frame
from objectcount.capture.server import create_app
from objectcount.frames.sources import iter_frames


def make_jpeg(value: int = 128) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def hello(ws, session=None):
    ws.send_text(json.dumps({"type": "hello", "device": "TestPhone",
                             "session": session,
                             "capabilities": {"zoom": {"min": 1, "max": 8}},
                             "settings": {"zoom": 1}}))
    return json.loads(ws.receive_text())


def test_ingest_spools_then_acks(tmp_path):
    client = TestClient(create_app(out_dir=tmp_path))
    with client.websocket_connect("/ws/ingest") as ws:
        hello_ack = hello(ws)
        assert hello_ack["type"] == "hello_ack"
        assert hello_ack["fps"] == 5
        session = hello_ack["session"]

        ws.send_bytes(pack_frame(
            {"seq": 1, "client_ts": 0.5, "w": 64, "h": 48}, make_jpeg(10)))
        ack = json.loads(ws.receive_text())
        assert ack == {"type": "ack", "seq": 1}

    session_dir = tmp_path / session
    assert (session_dir / "frames" / "000001.jpg").exists()
    entry = json.loads((session_dir / "manifest.jsonl")
                       .read_text(encoding="utf-8").splitlines()[0])
    assert entry["seq"] == 1
    assert entry["client_ts"] == 0.5
    info = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert info["device"] == "TestPhone"
    assert info["capabilities"]["zoom"]["max"] == 8


def test_gap_counted_in_stats(tmp_path):
    client = TestClient(create_app(out_dir=tmp_path))
    with client.websocket_connect("/ws/ingest") as ws:
        hello(ws)
        for seq in (1, 4):  # 2-3 공백
            ws.send_bytes(pack_frame({"seq": seq, "client_ts": 0, "w": 64, "h": 48},
                                     make_jpeg()))
            ws.receive_text()
    stats = client.get("/stats").json()
    assert stats["frames"] == 2
    assert stats["gaps"] == 2


def test_reconnect_resumes_session(tmp_path):
    client = TestClient(create_app(out_dir=tmp_path))
    with client.websocket_connect("/ws/ingest") as ws:
        session = hello(ws)["session"]
        ws.send_bytes(pack_frame({"seq": 1, "client_ts": 0, "w": 64, "h": 48},
                                 make_jpeg()))
        ws.receive_text()
    with client.websocket_connect("/ws/ingest") as ws:
        resumed = hello(ws, session=session)["session"]
        assert resumed == session
        ws.send_bytes(pack_frame({"seq": 2, "client_ts": 0, "w": 64, "h": 48},
                                 make_jpeg()))
        ws.receive_text()
    lines = (tmp_path / session / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert [json.loads(l)["seq"] for l in lines] == [1, 2]


def test_spooled_session_replayable(tmp_path):
    client = TestClient(create_app(out_dir=tmp_path))
    with client.websocket_connect("/ws/ingest") as ws:
        session = hello(ws)["session"]
        ws.send_bytes(pack_frame({"seq": 1, "client_ts": 0, "w": 64, "h": 48},
                                 make_jpeg()))
        ws.receive_text()
    records = list(iter_frames(tmp_path / session))
    assert len(records) == 1
    assert records[0].timestamp is not None


def test_config_reports_fps(tmp_path):
    client = TestClient(create_app(out_dir=tmp_path, default_fps=10))
    assert client.get("/config").json() == {"fps": 10}
