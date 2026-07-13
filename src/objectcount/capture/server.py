"""캡처 서버 — 폰 프레임 수신. 스풀 기록 후에만 ack (무손실 원칙, 스펙 §3·§7)."""
from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse

from ..frames.spool import SpoolWriter
from .protocol import unpack_frame

STATIC_DIR = Path(__file__).parent / "static"
_FPS_WINDOW_S = 5.0


def _fps(times: deque) -> float:
    now = time.time()
    return sum(1 for t in times if now - t <= _FPS_WINDOW_S) / _FPS_WINDOW_S


def _session_dir_name(device: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z가-힣_-]+", "-", device).strip("-")[:24] or "phone"
    return time.strftime("%Y%m%d-%H%M%S") + "-" + slug


@dataclass
class CaptureState:
    out_dir: Path
    default_fps: int = 5
    session_name: str | None = None
    writer: SpoolWriter | None = None
    session_info: dict = field(default_factory=dict)
    last_seq: int | None = None
    gap_count: int = 0
    total_frames: int = 0
    recv_times: deque = field(default_factory=lambda: deque(maxlen=200))
    detect_times: deque = field(default_factory=lambda: deque(maxlen=200))
    warning: str | None = None
    on_frame: Callable[[bytes], None] | None = None  # Task 5가 mailbox 연결

    def snapshot(self) -> dict:
        return {"recv_fps": _fps(self.recv_times),
                "detect_fps": _fps(self.detect_times),
                "gaps": self.gap_count,
                "frames": self.total_frames,
                "session": self.session_name}


def _handle_hello(state: CaptureState, message: dict) -> str:
    requested = message.get("session")
    if requested and (state.out_dir / requested / "manifest.jsonl").exists():
        name = requested  # 재접속 — 같은 캡처 세션에 이어쓰기 (스펙 §7)
    else:
        name = _session_dir_name(message.get("device") or "phone")
        state.last_seq = None
    if state.writer is not None:
        state.writer.close()
    state.session_name = name
    state.writer = SpoolWriter(state.out_dir / name)
    if name == requested and state.session_info.get("session") == name:
        state.session_info["reconnected_at"] = time.time()
    else:
        state.session_info = {"session": name,
                              "device": message.get("device"),
                              "capabilities": message.get("capabilities", {}),
                              "settings_history": [],
                              "started_at": time.time(),
                              "ended_at": None}
    _record_settings(state, message.get("settings", {}))
    return name


def _record_settings(state: CaptureState, settings: dict) -> None:
    state.session_info["settings_history"].append(
        {"ts": time.time(), "settings": settings})
    state.writer.write_session_info(state.session_info)


def _handle_frame(state: CaptureState, data: bytes) -> int:
    header, jpeg = unpack_frame(data)
    seq = int(header["seq"])
    server_ts = time.time()
    if state.last_seq is not None and seq > state.last_seq + 1:
        state.gap_count += seq - state.last_seq - 1
    state.last_seq = seq
    state.writer.append(seq, jpeg, server_ts=server_ts,
                        client_ts=header.get("client_ts"),
                        width=header.get("w", 0), height=header.get("h", 0))
    state.total_frames += 1
    state.recv_times.append(server_ts)
    if state.on_frame is not None:
        state.on_frame(jpeg)
    return seq


def create_app(out_dir: str | Path, detector=None, default_fps: int = 5,
               startup_warning: str | None = None) -> FastAPI:
    app = FastAPI()
    state = CaptureState(out_dir=Path(out_dir), default_fps=default_fps,
                         warning=startup_warning)
    app.state.capture = state
    app.state.detector = detector

    @app.websocket("/ws/ingest")
    async def ingest(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("text") is not None:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "hello":
                        name = _handle_hello(state, payload)
                        await ws.send_text(json.dumps(
                            {"type": "hello_ack", "session": name,
                             "fps": state.default_fps}, ensure_ascii=False))
                    elif payload.get("type") == "settings":
                        _record_settings(state, payload.get("settings", {}))
                elif message.get("bytes") is not None and state.writer is not None:
                    seq = _handle_frame(state, message["bytes"])
                    await ws.send_text(json.dumps({"type": "ack", "seq": seq}))
        finally:
            if state.writer is not None:
                state.session_info["ended_at"] = time.time()
                state.writer.write_session_info(state.session_info)
                state.writer.close()
                state.writer = None

    @app.get("/stats")
    async def stats() -> JSONResponse:
        return JSONResponse(state.snapshot())

    @app.get("/config")
    async def config() -> JSONResponse:
        return JSONResponse({"fps": state.default_fps})

    @app.get("/phone")
    async def phone_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "phone.html")

    @app.get("/")
    async def viewer_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "viewer.html")

    return app
