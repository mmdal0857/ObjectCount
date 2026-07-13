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
    """수신 프레임 기록. append가 반환된 뒤에만 ack를 보낼 것.

    보장 수준: JPEG 기록 + manifest flush — 프로세스 크래시 수준.
    전원 장애까지 막는 fsync 내구성은 비목표 (스펙 §3).
    """

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
