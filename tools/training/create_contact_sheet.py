"""Create a deterministic bounded contact sheet for dataset label review."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

from objectcount.detection.types import Detection
from objectcount.registration.dataset import IMAGE_SUFFIXES, SPLITS

TILE_WIDTH = 256
TILE_HEIGHT = 280
HEADER_HEIGHT = 24
COLUMNS = 6


def _boxes(label_path: Path) -> list[Detection]:
    detections = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, width, height = line.split()
        if int(class_id) != 0:
            raise ValueError(f"unexpected class in contact sheet: {label_path}")
        cx, cy, width, height = map(float, (cx, cy, width, height))
        detections.append(Detection(
            cx - width / 2,
            cy - height / 2,
            cx + width / 2,
            cy + height / 2,
            1.0,
            0,
            "toilet_paper_roll",
        ))
    return detections


def _samples(dataset: Path) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    positives = []
    backgrounds = []
    for split in SPLITS:
        for image_path in sorted((dataset / "images" / split).iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = dataset / "labels" / split / f"{image_path.stem}.txt"
            target = positives if label_path.read_text(encoding="utf-8").strip() else backgrounds
            target.append((image_path, label_path))
    return positives, backgrounds


def _render_tile(image_path: Path, detections: list[Detection], label: str) -> np.ndarray:
    source = cv2.imread(str(image_path))
    if source is None:
        raise ValueError(f"unreadable contact-sheet image: {image_path}")
    available_height = TILE_HEIGHT - HEADER_HEIGHT
    scale = min(TILE_WIDTH / source.shape[1], available_height / source.shape[0])
    width = max(1, round(source.shape[1] * scale))
    height = max(1, round(source.shape[0] * scale))
    resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
    tile = np.full((TILE_HEIGHT, TILE_WIDTH, 3), 245, dtype=np.uint8)
    x_offset = (TILE_WIDTH - width) // 2
    y_offset = HEADER_HEIGHT + (available_height - height) // 2
    tile[y_offset:y_offset + height, x_offset:x_offset + width] = resized
    for detection in detections:
        x1, y1, x2, y2 = detection.to_pixels(width, height)
        cv2.rectangle(
            tile,
            (x_offset + x1, y_offset + y1),
            (x_offset + x2, y_offset + y2),
            (0, 0, 255),
            2,
        )
    cv2.putText(
        tile, label[:36], (5, 17), cv2.FONT_HERSHEY_SIMPLEX,
        0.42, (30, 30, 30), 1, cv2.LINE_AA)
    return tile


def create_contact_sheet(
    dataset: Path,
    output: Path,
    positive_count: int = 24,
    background_count: int = 12,
    positive_offset: int = 0,
    background_offset: int = 0,
    seed: int = 713,
) -> dict[str, object]:
    dataset = Path(dataset)
    output = Path(output)
    positives, backgrounds = _samples(dataset)
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(backgrounds)
    selected = (
        [("positive", *sample) for sample in
         positives[positive_offset:positive_offset + positive_count]]
        + [("background", *sample) for sample in
           backgrounds[background_offset:background_offset + background_count]]
    )
    if not selected:
        raise ValueError("no contact-sheet samples were selected")

    rows = math.ceil(len(selected) / COLUMNS)
    sheet = np.full(
        (rows * TILE_HEIGHT, COLUMNS * TILE_WIDTH, 3), 225, dtype=np.uint8)
    paths = []
    for index, (kind, image_path, label_path) in enumerate(selected):
        relative = image_path.relative_to(dataset).as_posix()
        tile = _render_tile(image_path, _boxes(label_path), f"{kind}: {relative}")
        row, column = divmod(index, COLUMNS)
        sheet[
            row * TILE_HEIGHT:(row + 1) * TILE_HEIGHT,
            column * TILE_WIDTH:(column + 1) * TILE_WIDTH,
        ] = tile
        paths.append({"kind": kind, "image": relative})

    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError(f"failed to encode contact sheet: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded.tobytes())
    selection = {
        "seed": seed,
        "positive_offset": positive_offset,
        "background_offset": background_offset,
        "positive_count": sum(item["kind"] == "positive" for item in paths),
        "background_count": sum(item["kind"] == "background" for item in paths),
        "samples": paths,
    }
    selection_path = output.with_name(f"{output.stem}-selection.json")
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    return selection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--positive-count", type=int, default=24)
    parser.add_argument("--background-count", type=int, default=12)
    parser.add_argument("--positive-offset", type=int, default=0)
    parser.add_argument("--background-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=713)
    args = parser.parse_args()
    selection = create_contact_sheet(
        args.dataset,
        args.out,
        positive_count=args.positive_count,
        background_count=args.background_count,
        positive_offset=args.positive_offset,
        background_offset=args.background_offset,
        seed=args.seed,
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
