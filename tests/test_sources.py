from pathlib import Path

import cv2
import numpy as np
import pytest

from objectcount.frames.sources import FrameRecord, iter_frames


def write_image(path: Path, value: int) -> None:
    ok, buf = cv2.imencode(
        path.suffix, np.full((32, 32, 3), value, dtype=np.uint8))
    assert ok
    path.write_bytes(buf.tobytes())


def test_single_image(tmp_path):
    write_image(tmp_path / "a.png", 10)
    records = list(iter_frames(tmp_path / "a.png"))
    assert len(records) == 1
    assert records[0].frame_index == 0
    assert records[0].source_id == "a.png"
    assert records[0].image.shape == (32, 32, 3)


def test_single_image_under_korean_directory(tmp_path):
    image_dir = tmp_path / "한글기기명"
    image_dir.mkdir()
    image_path = image_dir / "frame.png"
    write_image(image_path, 10)

    records = list(iter_frames(image_path))

    assert len(records) == 1
    assert records[0].image.shape == (32, 32, 3)


def test_folder_sorted_by_name(tmp_path):
    write_image(tmp_path / "b.png", 20)
    write_image(tmp_path / "a.png", 10)
    (tmp_path / "notes.txt").write_text("무시되어야 함")
    records = list(iter_frames(tmp_path))
    assert [r.source_id for r in records] == ["a.png", "b.png"]
    assert [r.frame_index for r in records] == [0, 1]


def test_video_frames(tmp_path):
    video_path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video_path),
                             cv2.VideoWriter_fourcc(*"MJPG"),
                             5.0, (32, 32))
    for i in range(5):
        writer.write(np.full((32, 32, 3), i * 40, dtype=np.uint8))
    writer.release()

    records = list(iter_frames(video_path))
    assert len(records) == 5
    assert [r.frame_index for r in records] == [0, 1, 2, 3, 4]
    assert all(r.source_id == "clip.avi" for r in records)


def test_unsupported_path_raises(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("x")
    with pytest.raises(ValueError):
        list(iter_frames(target))


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
