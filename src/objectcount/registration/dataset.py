"""Ultralytics YOLO 데이터셋의 학습 전 무결성 검증."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetReport:
    split_images: dict[str, int] = field(
        default_factory=lambda: {split: 0 for split in SPLITS})
    positive_instances: int = 0
    background_images: int = 0
    exact_duplicates: list[tuple[str, str]] = field(default_factory=list)
    near_duplicates: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _difference_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def _validate_label(label_path: Path, class_count: int,
                    report: DatasetReport) -> None:
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        report.background_images += 1
        return

    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        location = f"{label_path}:{line_number}"
        if len(parts) != 5:
            report.errors.append(f"invalid label columns at {location}")
            continue
        try:
            class_value = float(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            report.errors.append(f"invalid numeric label at {location}")
            continue
        class_id = int(class_value)
        if class_value != class_id or not 0 <= class_id < class_count:
            report.errors.append(f"invalid class id at {location}")
            continue

        x_center, y_center, width, height = coords
        valid_box = (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
            and x_center - width / 2 >= 0
            and x_center + width / 2 <= 1
            and y_center - height / 2 >= 0
            and y_center + height / 2 <= 1
        )
        if not valid_box:
            report.errors.append(f"invalid normalized box at {location}")
            continue
        report.positive_instances += 1


def validate_yolo_dataset(root: Path, class_count: int = 1) -> DatasetReport:
    root = Path(root)
    report = DatasetReport()
    fingerprints: list[tuple[str, Path, str, int]] = []

    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            report.errors.append(f"missing split directories: {split}")
            continue

        images = sorted(path for path in image_dir.iterdir()
                        if path.is_file()
                        and path.suffix.lower() in IMAGE_SUFFIXES)
        report.split_images[split] = len(images)
        image_stems = {path.stem for path in images}

        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                report.errors.append(f"unreadable image: {image_path}")
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                report.errors.append(f"missing label: {label_path}")
            else:
                _validate_label(label_path, class_count, report)

            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            fingerprints.append(
                (split, image_path, digest, _difference_hash(image)))

        for label_path in label_dir.glob("*.txt"):
            if label_path.stem not in image_stems:
                report.errors.append(f"orphan label: {label_path}")

    for index, left in enumerate(fingerprints):
        for right in fingerprints[index + 1:]:
            if left[0] == right[0]:
                continue
            pair = (
                left[1].relative_to(root).as_posix(),
                right[1].relative_to(root).as_posix(),
            )
            if left[2] == right[2]:
                report.exact_duplicates.append(pair)
                report.errors.append(
                    f"duplicate image across splits: {pair[0]} / {pair[1]}")
            elif (left[3] ^ right[3]).bit_count() <= 4:
                report.near_duplicates.append(pair)
                report.warnings.append(
                    f"near duplicate across splits: {pair[0]} / {pair[1]}")

    return report
