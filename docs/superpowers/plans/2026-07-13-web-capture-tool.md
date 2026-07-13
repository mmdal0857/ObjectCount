# 웹 캡처 도구 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** adb 유선 폰 카메라를 웹 페이지로 제어하고, 프레임을 PC로 전송해 감지 오버레이를 라이브로 확인하며, 전 프레임을 스풀 녹화해 리플레이 하네스 자산으로 축적하는 개발용 도구.

**Architecture:** FastAPI 단일 서버가 폰용 캡처 페이지(`/phone`)와 PC용 뷰어 페이지(`/`)를 서빙한다. 폰 브라우저가 `getUserMedia`로 카메라를 제어하고 5fps JPEG를 WS(`/ws/ingest`)로 전송하면, 서버는 **스풀에 기록한 뒤에만 ack**하고, 최신 프레임만 감지 워커(mailbox 크기 1)로 넘겨 뷰어 WS(`/ws/view`)에 오버레이를 브로드캐스트한다. 스풀 세션 폴더는 `iter_frames()`로 그대로 리플레이된다.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, OpenCV, onnxruntime(기존), 바닐라 JS 정적 페이지(빌드 스텝 없음).

**스펙:** `docs/superpowers/specs/2026-07-13-web-capture-tool-design.md`

## Global Constraints

- Detection 좌표는 [0,1] 정규화 xyxy — 픽셀 변환은 `Detection.to_pixels()`로만
- **캡처·스풀이 최우선**: 감지 워커가 죽어도(패키지 로드 실패 포함) 수신·기록은 계속, 뷰어에는 원본 프레임 + 경고 배너
- **기록 후에만 ack** — 무손실 원칙의 도구판
- 서버 수신 시각(`time.time()`, epoch 초)이 단일 시간 기준. `client_ts`는 참고용
- 프레임 입력은 `iter_frames()`의 `Iterator[FrameRecord]` 시그니처로만
- 테스트 실행: `.venv\Scripts\python -m pytest -q` (Windows). 코어 의존성 불변 — 새 의존성은 `[capture]` extra에만
- 에러·로그 메시지는 한국어. 커밋 메시지는 한국어 + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` 트레일러
- WS 바이너리 프레이밍(폰→서버, 서버→뷰어 공용): `[4바이트 BE 헤더 길이][헤더 JSON utf-8][JPEG 페이로드]`
- 캡처 세션 폴더 포맷(스펙 §4): `session.json` + `manifest.jsonl`(프레임당 1줄: `seq, client_ts, server_ts, file, w, h`) + `frames/{seq:06d}.jpg`

---

### Task 1: FrameRecord.timestamp 필드

**Files:**
- Modify: `src/objectcount/frames/sources.py`
- Test: `tests/test_sources.py`

**Interfaces:**
- Produces: `FrameRecord(source_id: str, frame_index: int, image: np.ndarray, timestamp: float | None = None)` — 초 단위, 소스 내 단조 증가. 영상 파일은 `frame_index / fps`(fps 조회 불가 시 `None`), 이미지·폴더는 `None`. 이후 모든 태스크가 이 시그니처를 쓴다.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_sources.py` 끝에 추가:

```python
def test_image_timestamp_is_none(tmp_path):
    write_image(tmp_path / "a.png", 10)
    records = list(iter_frames(tmp_path / "a.png"))
    assert records[0].timestamp is None


def test_video_timestamp_from_fps(tmp_path):
    video_path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video_path),
                             cv2.VideoWriter_fourcc(*"MJPG"),
                             5.0, (32, 32))
    for i in range(5):
        writer.write(np.full((32, 32, 3), i * 40, dtype=np.uint8))
    writer.release()

    records = list(iter_frames(video_path))
    assert records[0].timestamp == pytest.approx(0.0)
    assert records[3].timestamp == pytest.approx(3 / 5.0)
```

- [ ] **Step 2: 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_sources.py -q`
Expected: FAIL — `AttributeError: 'FrameRecord' object has no attribute 'timestamp'` 또는 `timestamp is None` 불일치

- [ ] **Step 3: 구현** — `src/objectcount/frames/sources.py`:

FrameRecord에 필드 추가:

```python
@dataclass(frozen=True, eq=False)
class FrameRecord:
    source_id: str
    frame_index: int
    image: np.ndarray
    timestamp: float | None = None  # 초 단위, 소스 내 단조 증가 (스펙 §5)
```

`_iter_video`를 fps 기반 timestamp로 수정:

```python
def _iter_video(path: Path) -> Iterator[FrameRecord]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            yield FrameRecord(source_id=path.name, frame_index=index,
                              image=image,
                              timestamp=index / fps if fps > 0 else None)
            index += 1
    finally:
        capture.release()
```

- [ ] **Step 4: 전체 테스트 통과 확인**

Run: `.venv\Scripts\python -m pytest -q`
Expected: 전부 PASS (기존 49개 + 신규 2개)

- [ ] **Step 5: 커밋**

```bash
git add src/objectcount/frames/sources.py tests/test_sources.py
git commit -m "feat: FrameRecord timestamp 필드 — 영상은 fps 기반, 추적용 (이월 항목)"
```

---

### Task 2: 스풀 기록·리플레이 (frames/spool.py)

**Files:**
- Create: `src/objectcount/frames/spool.py`
- Modify: `src/objectcount/frames/sources.py` (스풀 디스패치)
- Modify: `docs/superpowers/specs/2026-07-13-web-capture-tool-design.md` (§3 표의 spool.py 위치 수정)
- Test: `tests/test_spool.py`

**설계 노트:** 스펙 §3은 `capture/spool.py`로 두었으나, 캡처 세션 리플레이는 capture extra 없이도(코어만으로) 가능해야 하고 포맷 정의는 한 곳에 있어야 하므로 `frames/spool.py`로 둔다. fastapi 의존성 없음. 스펙 표를 함께 수정한다.

**Interfaces:**
- Consumes: `FrameRecord` (Task 1의 timestamp 포함 시그니처)
- Produces:
  - `SpoolWriter(session_dir: str | Path)` — 생성 시 `frames/` 디렉토리 생성, manifest를 append 모드로 열기(재접속 이어쓰기)
  - `SpoolWriter.write_session_info(info: dict) -> None` — `session.json` 전체 재작성
  - `SpoolWriter.append(seq: int, jpeg_bytes: bytes, *, server_ts: float, client_ts: float | None = None, width: int = 0, height: int = 0) -> Path` — JPEG 기록 + manifest 1줄 flush. **반환된 뒤에만 ack를 보낼 것**
  - `SpoolWriter.close() -> None`
  - `iter_spool(session_dir: str | Path) -> Iterator[FrameRecord]` — manifest 순서대로, `frame_index=seq`, `timestamp=server_ts`, `source_id=세션 폴더명`. 읽기 실패 프레임은 건너뜀
  - `iter_frames(dir)` — 폴더에 `manifest.jsonl`이 있으면 자동으로 `iter_spool` 디스패치

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_spool.py` 생성:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_spool.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'objectcount.frames.spool'`

- [ ] **Step 3: 구현** — `src/objectcount/frames/spool.py` 생성:

```python
"""캡처 세션 스풀 — 기록(SpoolWriter)·리플레이(iter_spool). 포맷 정의는 이 파일 한 곳 (스펙 §4)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import cv2

from .sources import FrameRecord

MANIFEST_NAME = "manifest.jsonl"
SESSION_INFO_NAME = "session.json"
FRAMES_DIR = "frames"


class SpoolWriter:
    """수신 프레임 내구 기록. append가 반환된 뒤에만 ack를 보낼 것 (무손실 원칙)."""

    def __init__(self, session_dir: str | Path) -> None:
        self.session_dir = Path(session_dir)
        (self.session_dir / FRAMES_DIR).mkdir(parents=True, exist_ok=True)
        self._manifest = open(self.session_dir / MANIFEST_NAME, "a",
                              encoding="utf-8")

    def write_session_info(self, info: dict) -> None:
        (self.session_dir / SESSION_INFO_NAME).write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    def append(self, seq: int, jpeg_bytes: bytes, *, server_ts: float,
               client_ts: float | None = None,
               width: int = 0, height: int = 0) -> Path:
        rel = f"{FRAMES_DIR}/{seq:06d}.jpg"
        frame_path = self.session_dir / rel
        frame_path.write_bytes(jpeg_bytes)
        self._manifest.write(json.dumps(
            {"seq": seq, "client_ts": client_ts, "server_ts": server_ts,
             "file": rel, "w": width, "h": height},
            ensure_ascii=False) + "\n")
        self._manifest.flush()
        return frame_path

    def close(self) -> None:
        self._manifest.close()


def iter_spool(session_dir: str | Path) -> Iterator[FrameRecord]:
    session_dir = Path(session_dir)
    with open(session_dir / MANIFEST_NAME, encoding="utf-8") as manifest:
        for line in manifest:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            image = cv2.imread(str(session_dir / entry["file"]))
            if image is None:  # 기록 중단·삭제 — 공백은 건너뛴다 (스펙 §4)
                continue
            yield FrameRecord(source_id=session_dir.name,
                              frame_index=entry["seq"],
                              image=image,
                              timestamp=entry["server_ts"])
```

- [ ] **Step 4: iter_frames 디스패치** — `src/objectcount/frames/sources.py`의 `iter_frames` 폴더 분기 수정:

```python
def iter_frames(path: str | Path) -> Iterator[FrameRecord]:
    path = Path(path)
    if path.is_dir():
        if (path / "manifest.jsonl").exists():
            from .spool import iter_spool  # 순환 import 방지 지연 로드
            yield from iter_spool(path)
        else:
            yield from _iter_folder(path)
    elif path.suffix.lower() in IMAGE_EXTS:
        yield _read_image(path, frame_index=0)
    elif path.suffix.lower() in VIDEO_EXTS:
        yield from _iter_video(path)
    else:
        raise ValueError(f"지원하지 않는 입력: {path}")
```

- [ ] **Step 5: 전체 테스트 통과 확인**

Run: `.venv\Scripts\python -m pytest -q`
Expected: 전부 PASS

- [ ] **Step 6: 스펙 표 수정** — `docs/superpowers/specs/2026-07-13-web-capture-tool-design.md` §3 표에서 다음 행을:

```
| `spool.py` | `SpoolWriter`(JPEG+manifest 기록), `SpoolSource`(캡처 세션 → `FrameRecord` 이터레이터) |
```

아래로 교체 (위치·이름 확정):

```
| `../frames/spool.py` | `SpoolWriter`(JPEG+manifest 기록), `iter_spool`(캡처 세션 → `FrameRecord` 이터레이터) — capture extra 없이 리플레이 가능하도록 코어에 둔다 |
```

- [ ] **Step 7: 커밋**

```bash
git add src/objectcount/frames/spool.py src/objectcount/frames/sources.py tests/test_spool.py docs/superpowers/specs/2026-07-13-web-capture-tool-design.md
git commit -m "feat: 캡처 세션 스풀 — SpoolWriter/iter_spool, iter_frames 자동 디스패치"
```

---

### Task 3: WS 프레이밍 (capture/protocol.py)

**Files:**
- Create: `src/objectcount/capture/__init__.py` (빈 파일)
- Create: `src/objectcount/capture/protocol.py`
- Test: `tests/test_capture_protocol.py`

**Interfaces:**
- Produces:
  - `pack_frame(header: dict, payload: bytes) -> bytes` — `[4바이트 BE 길이][헤더 JSON][페이로드]`
  - `unpack_frame(data: bytes) -> tuple[dict, bytes]` — 역변환, 손상 시 `ValueError`
  - 폰→서버(ingest)와 서버→뷰어(view) 양쪽이 같은 프레이밍을 쓴다. JS 구현은 Task 6·7의 페이지에 내장

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_capture_protocol.py` 생성:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_capture_protocol.py -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현** — `src/objectcount/capture/__init__.py`는 빈 파일로, `src/objectcount/capture/protocol.py`:

```python
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
```

- [ ] **Step 4: 통과 확인**

Run: `.venv\Scripts\python -m pytest tests/test_capture_protocol.py -q`
Expected: 4 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/objectcount/capture/__init__.py src/objectcount/capture/protocol.py tests/test_capture_protocol.py
git commit -m "feat: 캡처 WS 프레이밍 — 길이 접두 헤더 JSON + JPEG 페이로드"
```

---

### Task 4: 캡처 서버 — ingest WS·스풀·ack (capture/server.py)

**Files:**
- Modify: `pyproject.toml` (`[capture]` extra, dev에 httpx)
- Create: `src/objectcount/capture/server.py`
- Test: `tests/test_capture_server.py`

**Interfaces:**
- Consumes: `SpoolWriter`(Task 2), `pack_frame`/`unpack_frame`(Task 3)
- Produces:
  - `create_app(out_dir: str | Path, detector=None, default_fps: int = 5, startup_warning: str | None = None) -> FastAPI`
  - `CaptureState` (dataclass, `app.state.capture`): `out_dir: Path`, `session_name: str | None`, `writer: SpoolWriter | None`, `session_info: dict`, `last_seq: int | None`, `gap_count: int`, `total_frames: int`, `recv_times: deque`, `detect_times: deque`, `warning: str | None`, `on_frame: Callable[[bytes], None] | None`(Task 5가 mailbox 연결), `snapshot() -> dict` — 키: `recv_fps, detect_fps, gaps, frames, session`
  - WS `/ws/ingest` 프로토콜(스펙 §7):
    - 폰→서버 text: `{"type":"hello","device":str,"session":str|null,"capabilities":{},"settings":{}}` / `{"type":"settings","settings":{}}`
    - 폰→서버 binary: `pack_frame({"seq","client_ts","w","h"}, jpeg)`
    - 서버→폰 text: `{"type":"hello_ack","session":str,"fps":int}` / `{"type":"ack","seq":int}` (스풀 기록 후에만)
  - `GET /stats` → `snapshot()` JSON, `GET /phone`·`GET /` → 정적 페이지(파일은 Task 6·7), `GET /config` → `{"fps": default_fps}`

- [ ] **Step 1: 의존성 추가** — `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest>=8", "onnx>=1.16", "httpx>=0.27"]
tools = ["ultralytics>=8.2"]
capture = ["fastapi>=0.110", "uvicorn[standard]>=0.29"]
```

Run: `.venv\Scripts\pip install -e ".[dev,capture]"`
Expected: 성공적으로 설치

- [ ] **Step 2: 실패하는 테스트 작성** — `tests/test_capture_server.py` 생성:

```python
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
```

- [ ] **Step 3: 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_capture_server.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'objectcount.capture.server'`

- [ ] **Step 4: 구현** — `src/objectcount/capture/server.py` 생성:

```python
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
```

- [ ] **Step 5: 통과 확인**

Run: `.venv\Scripts\python -m pytest tests/test_capture_server.py -q`
Expected: 5 PASS

Run: `.venv\Scripts\python -m pytest -q`
Expected: 전부 PASS

- [ ] **Step 6: 커밋**

```bash
git add pyproject.toml src/objectcount/capture/server.py tests/test_capture_server.py
git commit -m "feat: 캡처 서버 ingest — hello/재접속·스풀 기록 후 ack·공백 통계"
```

---

### Task 5: 라이브 감지 워커 + 뷰어 브로드캐스트 (capture/live.py)

**Files:**
- Create: `src/objectcount/capture/live.py`
- Modify: `src/objectcount/capture/server.py` (`/ws/view` + 워커 lifespan 배선)
- Test: `tests/test_capture_live.py`

**Interfaces:**
- Consumes: `CaptureState`(Task 4 — `on_frame`, `detect_times`, `warning`, `snapshot()`), `pack_frame`(Task 3), `draw_detections(image_bgr, detections)`(기존 viz), `Detection`(기존)
- Produces:
  - `LatestFrameMailbox` — `put(jpeg: bytes)`(최신으로 교체, 논블로킹), `async get() -> bytes`
  - `ViewerHub` — `add(ws)`, `remove(ws)`, `async broadcast(message: bytes)`
  - `async live_worker(mailbox, hub, detector, state) -> None` — 무한 루프. detect 예외 시 `state.warning` 설정 후 감지 중단(캡처는 계속), 원본 프레임 송출
  - 뷰어 WS 메시지: `pack_frame(state.snapshot() | {"type": "frame", "warning": state.warning}, 주석JPEG)`
  - `create_app` 내부: mailbox·hub 생성, `state.on_frame = mailbox.put`, lifespan에서 워커 태스크 시작/취소, `/ws/view` 엔드포인트

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_capture_live.py` 생성:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_capture_live.py -q`
Expected: FAIL — 뷰어가 프레임을 받지 못해 timeout 또는 `/ws/view` 404 계열 오류

- [ ] **Step 3: live.py 구현** — `src/objectcount/capture/live.py` 생성:

```python
"""라이브 감지 워커 — 최신 프레임만 감지(표시는 드롭돼도 저장은 전부), 뷰어 브로드캐스트."""
from __future__ import annotations

import asyncio
import time

import cv2
import numpy as np

from ..viz import draw_detections
from .protocol import pack_frame


class LatestFrameMailbox:
    """크기 1 — 새 프레임이 이전 것을 교체한다 (스펙 §3 우선순위 원칙)."""

    def __init__(self) -> None:
        self._item: bytes | None = None
        self._event = asyncio.Event()

    def put(self, item: bytes) -> None:
        self._item = item
        self._event.set()

    async def get(self) -> bytes:
        await self._event.wait()
        self._event.clear()
        item = self._item
        self._item = None
        return item


class ViewerHub:
    def __init__(self) -> None:
        self._clients: set = set()

    def add(self, ws) -> None:
        self._clients.add(ws)

    def remove(self, ws) -> None:
        self._clients.discard(ws)

    async def broadcast(self, message: bytes) -> None:
        for ws in list(self._clients):
            try:
                await ws.send_bytes(message)
            except Exception:
                self._clients.discard(ws)


async def live_worker(mailbox: LatestFrameMailbox, hub: ViewerHub,
                      detector, state) -> None:
    loop = asyncio.get_running_loop()
    while True:
        jpeg = await mailbox.get()
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        annotated = image
        if detector is not None and state.warning is None:
            try:
                detections = await loop.run_in_executor(
                    None, detector.detect, image)
                annotated = draw_detections(image, detections)
                state.detect_times.append(time.time())
            except Exception as error:  # 감지 중단 — 캡처·스풀은 계속 (스펙 §9)
                state.warning = f"감지 워커 중단: {error} — 캡처·스풀은 계속됩니다"
        ok, buf = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        header = state.snapshot() | {"type": "frame", "warning": state.warning}
        await hub.broadcast(pack_frame(header, buf.tobytes()))
```

- [ ] **Step 4: server.py 배선** — `src/objectcount/capture/server.py` 수정.

import 추가:

```python
from contextlib import asynccontextmanager

from .live import LatestFrameMailbox, ViewerHub, live_worker
```

`create_app` 본문을 다음처럼 확장 (기존 엔드포인트는 그대로, 앞뒤에 추가):

```python
def create_app(out_dir: str | Path, detector=None, default_fps: int = 5,
               startup_warning: str | None = None) -> FastAPI:
    state = CaptureState(out_dir=Path(out_dir), default_fps=default_fps,
                         warning=startup_warning)
    mailbox = LatestFrameMailbox()
    hub = ViewerHub()
    state.on_frame = mailbox.put

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker = asyncio.create_task(
            live_worker(mailbox, hub, detector, state))
        yield
        worker.cancel()

    app = FastAPI(lifespan=lifespan)
    app.state.capture = state
    app.state.detector = detector
    # ... (기존 /ws/ingest, /stats, /config, /phone, / 엔드포인트 그대로) ...

    @app.websocket("/ws/view")
    async def view(ws: WebSocket) -> None:
        await ws.accept()
        hub.add(ws)
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
        finally:
            hub.remove(ws)

    return app
```

`import asyncio`도 server.py 상단에 추가.

- [ ] **Step 5: 통과 확인**

Run: `.venv\Scripts\python -m pytest tests/test_capture_live.py tests/test_capture_server.py -q`
Expected: 전부 PASS (기존 server 테스트도 lifespan 추가 후 여전히 통과해야 함 — TestClient는 `with client:` 없이도 요청 시 lifespan을 생략하므로 기존 테스트 무변경)

Run: `.venv\Scripts\python -m pytest -q`
Expected: 전부 PASS

- [ ] **Step 6: 커밋**

```bash
git add src/objectcount/capture/live.py src/objectcount/capture/server.py tests/test_capture_live.py
git commit -m "feat: 라이브 감지 워커 — 최신 프레임 mailbox·뷰어 브로드캐스트·감지 실패 강등"
```

---

### Task 6: 폰 캡처 페이지 (static/phone.html)

**Files:**
- Create: `src/objectcount/capture/static/phone.html`
- Test: `tests/test_capture_pages.py`

**Interfaces:**
- Consumes: WS `/ws/ingest` 프로토콜(Task 4), 프레이밍(Task 3의 JS 대응 구현 내장), `GET /config`
- Produces: 없음 (말단 UI)

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_capture_pages.py` 생성:

```python
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from objectcount.capture.server import create_app


def test_phone_page_served(tmp_path):
    client = TestClient(create_app(out_dir=tmp_path))
    response = client.get("/phone")
    assert response.status_code == 200
    assert "getUserMedia" in response.text
    assert "/ws/ingest" in response.text
```

- [ ] **Step 2: 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_capture_pages.py -q`
Expected: FAIL — phone.html 부재로 500/404

- [ ] **Step 3: 구현** — `src/objectcount/capture/static/phone.html` 생성 (전체 내용):

```html
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ObjectCount 캡처</title>
<style>
  body { font-family: sans-serif; margin: 8px; background: #111; color: #eee; }
  video { width: 100%; max-height: 45vh; background: #000; }
  .row { display: flex; align-items: center; gap: 8px; margin: 8px 0; }
  .row label { width: 88px; flex-shrink: 0; }
  .row input[type=range] { flex: 1; }
  .off { opacity: 0.4; }
  #status { font-size: 14px; }
  button { padding: 10px 16px; font-size: 16px; }
  #banner { background: #a33; padding: 6px; display: none; }
</style>
</head>
<body>
<div id="banner"></div>
<video id="preview" autoplay playsinline muted></video>
<div class="row"><button id="start">캡처 시작</button><span id="status">대기</span></div>
<div class="row">
  <label>해상도</label>
  <select id="res">
    <option value="1280x720" selected>1280×720</option>
    <option value="1920x1080">1920×1080</option>
    <option value="640x480">640×480</option>
  </select>
  <label style="width:auto">fps</label>
  <input id="fps" type="number" min="1" max="30" value="5" style="width:56px">
</div>
<div class="row" id="zoomRow"><label>줌</label><input id="zoom" type="range" disabled><span id="zoomVal"></span></div>
<div class="row" id="focusRow"><label>포커스</label>
  <select id="focusMode" disabled><option value="continuous">자동</option><option value="manual">수동</option></select>
  <input id="focusDist" type="range" disabled><span id="focusVal"></span></div>
<div class="row" id="expRow"><label>노출 잠금</label><input id="expLock" type="checkbox" disabled><span id="expVal"></span></div>
<script>
const $ = id => document.getElementById(id);
let stream = null, track = null, ws = null, timer = null, wakeLock = null;
let seq = 0, ackSeq = 0, running = false, session = null, backoff = 1000;

function setBanner(text) {
  const banner = $('banner');
  banner.textContent = text || '';
  banner.style.display = text ? 'block' : 'none';
}
function refreshStatus() {
  $('status').textContent = running
    ? `전송 ${seq} · 확인 ${ackSeq} · 미확인 ${seq - ackSeq}` : '중지됨';
}

async function openCamera() {
  const [w, h] = $('res').value.split('x').map(Number);
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: 'environment' },
             width: { ideal: w }, height: { ideal: h } },
    audio: false });
  track = stream.getVideoTracks()[0];
  $('preview').srcObject = stream;
  bindControls();
}

function caps() { return track && track.getCapabilities ? track.getCapabilities() : {}; }
function currentSettings() { return track && track.getSettings ? track.getSettings() : {}; }

async function apply(constraints) {
  try {
    await track.applyConstraints({ advanced: [constraints] });
    if (ws && ws.readyState === 1)
      ws.send(JSON.stringify({ type: 'settings', settings: currentSettings() }));
  } catch (e) { setBanner('설정 적용 실패: ' + e.message); }
}

function bindControls() {
  const c = caps();
  if (c.zoom) {
    $('zoomRow').classList.remove('off');
    const zoom = $('zoom');
    zoom.min = c.zoom.min; zoom.max = c.zoom.max; zoom.step = c.zoom.step || 0.1;
    zoom.value = currentSettings().zoom || c.zoom.min;
    zoom.disabled = false;
    $('zoomVal').textContent = zoom.value;
    zoom.oninput = async () => {
      await apply({ zoom: Number(zoom.value) });
      $('zoomVal').textContent = zoom.value;
    };
  } else { $('zoomRow').classList.add('off'); $('zoomVal').textContent = '이 기기 미지원'; }

  if (c.focusMode && c.focusMode.includes('manual') && c.focusDistance) {
    $('focusRow').classList.remove('off');
    const mode = $('focusMode'), dist = $('focusDist');
    mode.disabled = false;
    dist.min = c.focusDistance.min; dist.max = c.focusDistance.max;
    dist.step = c.focusDistance.step || 0.01;
    mode.onchange = async () => {
      if (mode.value === 'manual') {
        dist.disabled = false;
        await apply({ focusMode: 'manual', focusDistance: Number(dist.value || dist.min) });
      } else { dist.disabled = true; await apply({ focusMode: 'continuous' }); }
    };
    dist.oninput = async () => {
      await apply({ focusMode: 'manual', focusDistance: Number(dist.value) });
      $('focusVal').textContent = dist.value;
    };
  } else { $('focusRow').classList.add('off'); $('focusVal').textContent = '이 기기 미지원'; }

  if (c.exposureMode && c.exposureMode.includes('manual')) {
    $('expRow').classList.remove('off');
    $('expLock').disabled = false;
    $('expLock').onchange = async () =>
      await apply({ exposureMode: $('expLock').checked ? 'manual' : 'continuous' });
  } else { $('expRow').classList.add('off'); $('expVal').textContent = '이 기기 미지원'; }
}

function packFrame(header, payloadBuf) {
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(4 + headerBytes.length + payloadBuf.byteLength);
  new DataView(out.buffer).setUint32(0, headerBytes.length);
  out.set(headerBytes, 4);
  out.set(new Uint8Array(payloadBuf), 4 + headerBytes.length);
  return out;
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws/ingest`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    backoff = 1000;
    setBanner(null);
    ws.send(JSON.stringify({ type: 'hello', device: navigator.userAgent,
      session, capabilities: caps(), settings: currentSettings() }));
  };
  ws.onmessage = ev => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'hello_ack') {
      session = msg.session;
      if (!$('fps').dataset.touched) $('fps').value = msg.fps;
      startTimer();
    }
    if (msg.type === 'ack') { ackSeq = msg.seq; refreshStatus(); }
  };
  ws.onclose = () => {
    if (!running) return;
    setBanner('서버 연결 끊김 — 재접속 중');
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 10000);
  };
}

const canvas = document.createElement('canvas');
async function captureFrame() {
  if (!running || document.hidden || !ws || ws.readyState !== 1) return;
  const video = $('preview');
  if (!video.videoWidth) return;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.85));
  if (!blob) return;
  seq += 1;
  ws.send(packFrame({ seq, client_ts: Date.now() / 1000,
                      w: canvas.width, h: canvas.height },
                    await blob.arrayBuffer()));
  refreshStatus();
}

function startTimer() {
  clearInterval(timer);
  timer = setInterval(captureFrame, 1000 / Number($('fps').value));
}

async function keepAwake() {
  try { wakeLock = await navigator.wakeLock.request('screen'); } catch (e) {}
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) setBanner('백그라운드 — 캡처 일시정지');
  else { setBanner(null); keepAwake(); }
});

$('start').onclick = async () => {
  if (running) {
    running = false;
    clearInterval(timer);
    if (ws) ws.close();
    $('start').textContent = '캡처 시작';
    refreshStatus();
    return;
  }
  try { await openCamera(); }
  catch (e) { setBanner('카메라 접근 실패: ' + e.message); return; }
  await keepAwake();
  running = true;
  connect();
  $('start').textContent = '캡처 중지';
};
$('fps').onchange = () => { $('fps').dataset.touched = '1'; if (running) startTimer(); };
$('res').onchange = async () => { if (running) await openCamera(); };
</script>
</body>
</html>
```

- [ ] **Step 4: 통과 확인**

Run: `.venv\Scripts\python -m pytest tests/test_capture_pages.py -q`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/objectcount/capture/static/phone.html tests/test_capture_pages.py
git commit -m "feat: 폰 캡처 페이지 — getUserMedia·줌/포커스/노출·WS 전송·재접속"
```

---

### Task 7: 뷰어 페이지 (static/viewer.html)

**Files:**
- Create: `src/objectcount/capture/static/viewer.html`
- Test: `tests/test_capture_pages.py` (테스트 추가)

**Interfaces:**
- Consumes: WS `/ws/view` 메시지(Task 5 — 헤더 키 `recv_fps, detect_fps, gaps, frames, session, warning, type`)
- Produces: 없음 (말단 UI)

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_capture_pages.py`에 추가:

```python
def test_viewer_page_served(tmp_path):
    client = TestClient(create_app(out_dir=tmp_path))
    response = client.get("/")
    assert response.status_code == 200
    assert "/ws/view" in response.text
```

- [ ] **Step 2: 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_capture_pages.py -q`
Expected: `test_viewer_page_served` FAIL

- [ ] **Step 3: 구현** — `src/objectcount/capture/static/viewer.html` 생성 (전체 내용):

```html
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>ObjectCount 뷰어</title>
<style>
  body { font-family: sans-serif; margin: 0; background: #111; color: #eee; }
  #bar { display: flex; gap: 16px; padding: 8px; background: #222;
         font-size: 14px; flex-wrap: wrap; }
  #banner { background: #a33; padding: 6px; display: none; }
  img { width: 100%; max-height: calc(100vh - 60px);
        object-fit: contain; background: #000; }
</style>
</head>
<body>
<div id="bar">
  <span id="recv">수신 —</span><span id="det">감지 —</span>
  <span id="gaps">공백 —</span><span id="frames">프레임 —</span>
  <span id="session">세션 —</span>
</div>
<div id="banner"></div>
<img id="frame" alt="라이브 프레임">
<script>
function unpackFrame(buf) {
  const view = new DataView(buf);
  const len = view.getUint32(0);
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buf, 4, len)));
  return { header, payload: new Uint8Array(buf, 4 + len) };
}
let url = null;
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws/view`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = ev => {
    const { header, payload } = unpackFrame(ev.data);
    document.getElementById('recv').textContent =
      `수신 ${header.recv_fps.toFixed(1)}fps`;
    document.getElementById('det').textContent =
      `감지 ${header.detect_fps.toFixed(1)}fps`;
    document.getElementById('gaps').textContent = `공백 ${header.gaps}`;
    document.getElementById('frames').textContent = `프레임 ${header.frames}`;
    document.getElementById('session').textContent =
      `세션 ${header.session || '—'}`;
    const banner = document.getElementById('banner');
    banner.style.display = header.warning ? 'block' : 'none';
    banner.textContent = header.warning || '';
    if (url) URL.revokeObjectURL(url);
    url = URL.createObjectURL(new Blob([payload], { type: 'image/jpeg' }));
    document.getElementById('frame').src = url;
  };
  ws.onclose = () => setTimeout(connect, 1000);
}
connect();
</script>
</body>
</html>
```

- [ ] **Step 4: 통과 확인**

Run: `.venv\Scripts\python -m pytest tests/test_capture_pages.py -q`
Expected: 2 PASS

- [ ] **Step 5: 커밋**

```bash
git add src/objectcount/capture/static/viewer.html tests/test_capture_pages.py
git commit -m "feat: PC 뷰어 페이지 — 오버레이 스트림·진단 바·경고 배너"
```

---

### Task 8: CLI (objectcount-capture) + 패키징

**Files:**
- Create: `src/objectcount/cli/capture.py`
- Modify: `pyproject.toml` (entry point)
- Modify: `CLAUDE.md` (명령어 섹션에 캡처 명령 추가)
- Test: `tests/test_cli_capture.py`

**Interfaces:**
- Consumes: `create_app`(Task 4·5 시그니처), `OnnxDetector`·`load_package`(기존)
- Produces:
  - `objectcount-capture --package <pkg> --out <dir> [--port 8377] [--no-adb] [--no-detect] [--fps 5]`
  - `try_adb_reverse(port: int) -> str` — 결과 안내 문자열 반환(항상)
  - `check_disk(out_dir: Path, threshold_gb: float = 2.0) -> str | None` — 여유 부족 시 경고 문자열
  - 종료 코드: 인자 오류 1, 정상 기동 후 종료 0. **모델 패키지 로드 실패는 경고 강등 후 캡처만 계속** (스펙 §9)

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_cli_capture.py` 생성:

```python
"""캡처 CLI — adb 안내·디스크 경고·인자 검증. 서버 기동은 mock."""
import pytest

pytest.importorskip("fastapi")

from objectcount.cli import capture as cli


def test_adb_missing_gives_manual_command(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    message = cli.try_adb_reverse(8377)
    assert "adb reverse tcp:8377 tcp:8377" in message


def test_adb_failure_gives_manual_command(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "adb")

    class Result:
        returncode = 1
        stderr = "no devices found"
        stdout = ""
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: Result())
    message = cli.try_adb_reverse(8377)
    assert "no devices found" in message
    assert "adb reverse tcp:8377" in message


def test_check_disk_warns_below_threshold(tmp_path, monkeypatch):
    class Usage:
        free = 1 * 1024 ** 3
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda p: Usage())
    assert "경고" in cli.check_disk(tmp_path, threshold_gb=2.0)


def test_check_disk_ok(tmp_path, monkeypatch):
    class Usage:
        free = 100 * 1024 ** 3
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda p: Usage())
    assert cli.check_disk(tmp_path, threshold_gb=2.0) is None


def test_package_required_without_no_detect(tmp_path, capsys):
    assert cli.main(["--out", str(tmp_path)]) == 1
    assert "--package" in capsys.readouterr().err


def test_no_detect_runs_server(tmp_path, monkeypatch):
    calls = {}
    import uvicorn
    monkeypatch.setattr(uvicorn, "run",
                        lambda app, **kw: calls.update(kw))
    code = cli.main(["--out", str(tmp_path), "--no-detect", "--no-adb",
                     "--port", "9000"])
    assert code == 0
    assert calls["port"] == 9000


def test_broken_package_degrades_to_capture_only(tmp_path, monkeypatch, capsys):
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    code = cli.main(["--out", str(tmp_path), "--no-adb",
                     "--package", str(tmp_path / "없는패키지")])
    assert code == 0  # 캡처가 우선 — 감지 실패는 경고 강등
    assert "감지 없이 캡처만" in capsys.readouterr().err
```

- [ ] **Step 2: 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_cli_capture.py -q`
Expected: FAIL — `ImportError: cannot import name 'capture'`

- [ ] **Step 3: 구현** — `src/objectcount/cli/capture.py` 생성:

```python
"""캡처 서버 CLI — adb reverse 자동 설정·디스크 체크·서버 기동 (스펙 §10)."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def try_adb_reverse(port: int) -> str:
    manual = f"수동 설정: adb reverse tcp:{port} tcp:{port}"
    adb = shutil.which("adb")
    if adb is None:
        return f"adb를 찾을 수 없습니다. {manual}"
    result = subprocess.run([adb, "reverse", f"tcp:{port}", f"tcp:{port}"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return f"adb reverse 실패: {detail}. {manual}"
    return f"adb reverse tcp:{port} 설정 완료"


def check_disk(out_dir: Path, threshold_gb: float = 2.0) -> str | None:
    free_gb = shutil.disk_usage(out_dir).free / 1024 ** 3
    if free_gb < threshold_gb:
        return (f"경고: 디스크 여유 {free_gb:.1f}GB — "
                f"{threshold_gb}GB 미만입니다")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="objectcount-capture")
    parser.add_argument("--package",
                        help="모델 패키지 디렉토리 (--no-detect면 생략)")
    parser.add_argument("--out", required=True, help="캡처 세션 저장 디렉토리")
    parser.add_argument("--port", type=int, default=8377)
    parser.add_argument("--no-adb", action="store_true",
                        help="adb reverse 자동 설정 생략")
    parser.add_argument("--no-detect", action="store_true",
                        help="감지 없이 캡처·스풀만")
    parser.add_argument("--fps", type=int, default=5,
                        help="폰 페이지 기본 캡처 fps")
    args = parser.parse_args(argv)

    try:
        import uvicorn

        from ..capture.server import create_app
    except ImportError:
        print('오류: capture extra가 필요합니다 — pip install -e ".[capture]"',
              file=sys.stderr)
        return 1

    detector = None
    warning = None
    if args.no_detect:
        warning = "감지 없이 캡처만 (--no-detect)"
    elif not args.package:
        print("오류: --package가 필요합니다 (감지를 끄려면 --no-detect)",
              file=sys.stderr)
        return 1
    else:
        try:
            from ..detection.detector import OnnxDetector
            from ..models.package import load_package
            detector = OnnxDetector(load_package(args.package))
        except Exception as error:  # 캡처가 우선 — 감지 실패는 경고 강등 (스펙 §9)
            warning = f"모델 패키지 로드 실패: {error} — 감지 없이 캡처만 진행"
            print(warning, file=sys.stderr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    disk_warning = check_disk(out_dir)
    if disk_warning:
        print(disk_warning, file=sys.stderr)

    if not args.no_adb:
        print(try_adb_reverse(args.port))

    print(f"폰에서 열기: http://localhost:{args.port}/phone")
    print(f"PC 뷰어:     http://localhost:{args.port}/")
    print(f"캡처 세션 저장: {out_dir}")

    app = create_app(out_dir=out_dir, detector=detector,
                     default_fps=args.fps, startup_warning=warning)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: entry point 추가** — `pyproject.toml`:

```toml
[project.scripts]
objectcount-detect = "objectcount.cli.detect:main"
objectcount-eval = "objectcount.cli.evaluate:main"
objectcount-capture = "objectcount.cli.capture:main"
```

Run: `.venv\Scripts\pip install -e ".[dev,capture]"`
Expected: 성공 (entry point 재등록)

- [ ] **Step 5: 통과 확인**

Run: `.venv\Scripts\python -m pytest -q`
Expected: 전부 PASS

- [ ] **Step 6: CLAUDE.md 명령어 섹션 갱신** — `## 명령어` 목록에 추가 (테스트 개수도 실측값으로 갱신):

```markdown
- 캡처: `.venv\Scripts\objectcount-capture --package models/packages/coco-demo --out out/captures` — 폰 Chrome에서 `http://localhost:8377/phone`, PC 뷰어 `http://localhost:8377/`. 캡처 세션 폴더는 `--input`으로 그대로 리플레이 가능
```

- [ ] **Step 7: 커밋**

```bash
git add src/objectcount/cli/capture.py pyproject.toml tests/test_cli_capture.py CLAUDE.md
git commit -m "feat: objectcount-capture CLI — adb reverse 자동 설정·디스크 체크·서버 기동"
```

---

### Task 9: 실기기 수동 검증 (사람 개입 필요)

**Files:** 없음 (검증 전용 — 발견된 버그는 수정 후 개별 커밋)

이 태스크는 실제 안드로이드 폰이 필요하다. **실행자는 사용자에게 폰 연결을 요청하고 함께 진행할 것.** 스펙 §11 체크리스트:

- [ ] **Step 1: 서버 기동** — `.venv\Scripts\objectcount-capture --package models/packages/coco-demo --out out/captures`
  - adb reverse 설정 완료 메시지와 두 URL이 출력되는지 확인
- [ ] **Step 2: 폰 접속** — USB 연결된 폰의 Chrome에서 `http://localhost:8377/phone` 접속, "캡처 시작" → 카메라 권한 허용
- [ ] **Step 3: 카메라 제어** — 줌 슬라이더 조작이 폰 프리뷰와 PC 뷰어 양쪽에 반영되는지, 미지원 컨트롤이 "이 기기 미지원"으로 표시되는지 확인
- [ ] **Step 4: 라이브 감지** — 일상 물체(컵·병·책 등 COCO 클래스)를 카메라 앞에 통과시켜 PC 뷰어(`http://localhost:8377/`)에서 감지 박스·진단 바(수신 fps ≈ 5) 확인
- [ ] **Step 5: 리플레이** — 세션 종료 후 `.venv\Scripts\objectcount-detect --package models/packages/coco-demo --input out/captures/<세션> --out out/replay-check` 실행, JSONL·주석 이미지 정상 생성 확인
- [ ] **Step 6: 재접속** — 캡처 중 USB를 뽑았다 다시 꽂아 재접속·같은 세션 이어쓰기(manifest에 seq 연속 추가) 확인
- [ ] **Step 7: 발견된 문제 수정** — 각 수정은 실패를 재현하는 테스트와 함께 개별 커밋

---

## Self-Review 결과

- **스펙 커버리지**: §3 아키텍처(Task 3-5), §4 스풀(Task 2), §5 timestamp(Task 1), §6 폰 페이지(Task 6), §7 프로토콜(Task 3·4·6), §8 뷰어(Task 5·7), §9 에러 처리(Task 4·5·8), §10 CLI(Task 8), §11 테스트(각 태스크 + Task 9), §12 의존성(Task 4·8) — 전부 매핑됨
- **§3 spool.py 위치**: `frames/spool.py`로 조정 (Task 2에서 스펙 표 함께 수정, 사유 명시)
- **타입 일관성**: `create_app(out_dir, detector, default_fps, startup_warning)` 시그니처를 Task 4에서 정의하고 5·8이 동일하게 사용. 뷰어 헤더 키(`recv_fps, detect_fps, gaps, frames, session, warning, type`)는 Task 5 `snapshot()`과 Task 7 JS가 일치
