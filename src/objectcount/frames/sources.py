"""프레임 소스 — 입력원 무관 원칙의 경계. 모든 소비자는 FrameRecord 이터레이터만 본다."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}


@dataclass(frozen=True, eq=False)
class FrameRecord:
    source_id: str
    frame_index: int
    image: np.ndarray


def iter_frames(path: str | Path) -> Iterator[FrameRecord]:
    path = Path(path)
    if path.is_dir():
        yield from _iter_folder(path)
    elif path.suffix.lower() in IMAGE_EXTS:
        yield _read_image(path, frame_index=0)
    elif path.suffix.lower() in VIDEO_EXTS:
        yield from _iter_video(path)
    else:
        raise ValueError(f"지원하지 않는 입력: {path}")


def _read_image(path: Path, frame_index: int) -> FrameRecord:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return FrameRecord(source_id=path.name, frame_index=frame_index, image=image)


def _iter_folder(folder: Path) -> Iterator[FrameRecord]:
    images = sorted(p for p in folder.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    for index, path in enumerate(images):
        yield _read_image(path, frame_index=index)


def _iter_video(path: Path) -> Iterator[FrameRecord]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {path}")
    try:
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            yield FrameRecord(source_id=path.name, frame_index=index,
                              image=image)
            index += 1
    finally:
        capture.release()
