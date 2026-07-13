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
