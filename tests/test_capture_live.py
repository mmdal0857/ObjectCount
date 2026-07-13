"""라이브 감지 워커 — 뷰어 오버레이 브로드캐스트, 감지 실패 시 원본+경고."""
import json

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from objectcount.capture.protocol import pack_frame, unpack_frame
from objectcount.capture.server import create_app
from objectcount.detection.types import Detection


def make_jpeg(value: int = 128) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


class FakeDetector:
    def detect(self, image):
        return [Detection(0.25, 0.25, 0.75, 0.75, 0.9, 0, "can")]


class BrokenDetector:
    def detect(self, image):
        raise RuntimeError("의도된 실패")


def send_one_frame(client, seq=1):
    with client.websocket_connect("/ws/ingest") as ws:
        ws.send_text(json.dumps({"type": "hello", "device": "t",
                                 "session": None, "capabilities": {},
                                 "settings": {}}))
        ws.receive_text()
        ws.send_bytes(pack_frame({"seq": seq, "client_ts": 0, "w": 64, "h": 48},
                                 make_jpeg()))
        ws.receive_text()


def test_viewer_receives_annotated_frame(tmp_path):
    client = TestClient(create_app(out_dir=tmp_path, detector=FakeDetector()))
    with client:  # lifespan(워커) 기동
        with client.websocket_connect("/ws/view") as viewer:
            send_one_frame(client)
            header, payload = unpack_frame(viewer.receive_bytes())
            assert header["type"] == "frame"
            assert header["frames"] == 1
            assert header["warning"] is None
            image = cv2.imdecode(np.frombuffer(payload, np.uint8),
                                 cv2.IMREAD_COLOR)
            assert image is not None
            assert image.shape == (48, 64, 3)


def test_detector_failure_sets_warning_keeps_capture(tmp_path):
    client = TestClient(create_app(out_dir=tmp_path, detector=BrokenDetector()))
    with client:
        with client.websocket_connect("/ws/view") as viewer:
            send_one_frame(client, seq=1)
            header1, _ = unpack_frame(viewer.receive_bytes())
            assert "의도된 실패" in header1["warning"]
            send_one_frame(client, seq=2)  # 캡처·스풀은 계속
            header2, payload2 = unpack_frame(viewer.receive_bytes())
            assert header2["warning"] is not None
            assert cv2.imdecode(np.frombuffer(payload2, np.uint8),
                                cv2.IMREAD_COLOR) is not None
    sessions = [p for p in tmp_path.iterdir() if p.is_dir()]
    total = sum(len((s / "manifest.jsonl").read_text(encoding="utf-8")
                    .splitlines()) for s in sessions)
    assert total == 2


def test_no_detector_streams_raw(tmp_path):
    client = TestClient(create_app(out_dir=tmp_path, detector=None,
                                   startup_warning="감지 없이 캡처만"))
    with client:
        with client.websocket_connect("/ws/view") as viewer:
            send_one_frame(client)
            header, payload = unpack_frame(viewer.receive_bytes())
            assert header["warning"] == "감지 없이 캡처만"
            assert cv2.imdecode(np.frombuffer(payload, np.uint8),
                                cv2.IMREAD_COLOR) is not None
