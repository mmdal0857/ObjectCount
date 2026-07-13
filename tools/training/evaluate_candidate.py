"""Evaluate a Product Candidate against positives and confusion negatives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from objectcount.detection.detector import OnnxDetector
from objectcount.detection.types import Detection
from objectcount.frames.sources import iter_frames
from objectcount.models.package import ModelPackage, load_package
from objectcount.registration.candidate import (
    ThresholdResult,
    select_operating_confidence,
)

DetectorFactory = Callable[[ModelPackage, float], object]


def _default_detector_factory(
        package: ModelPackage, threshold: float) -> OnnxDetector:
    return OnnxDetector(package, conf_threshold=threshold)


def _xywh_to_xyxy(values: list[float]) -> tuple[float, float, float, float]:
    _, cx, cy, width, height = values
    return (
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
    )


def _read_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"invalid label at {path}:{line_number}")
        values = [float(field) for field in fields]
        if int(values[0]) != 0:
            raise ValueError(f"unexpected class id at {path}:{line_number}")
        boxes.append(_xywh_to_xyxy(values))
    return boxes


def _iou(
    left: tuple[float, float, float, float],
    right: Detection,
) -> float:
    x1 = max(left[0], right.x1)
    y1 = max(left[1], right.y1)
    x2 = min(left[2], right.x2)
    y2 = min(left[3], right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    union = left_area + right.area - intersection
    return intersection / union if union > 0 else 0.0


def _run_folder(detector: object, root: Path) -> list[tuple[str, list[Detection]]]:
    images = root / "images" if (root / "images").is_dir() else root
    return [
        (record.source_id, detector.detect(record.image))
        for record in iter_frames(images)
    ]


def evaluate_thresholds(
    package: Path,
    positives: Path,
    negatives: Path,
    thresholds: list[float],
    detector_factory: DetectorFactory | None = None,
) -> dict[str, object]:
    """Choose the highest-recall threshold with zero confusion false positives."""
    if not thresholds:
        raise ValueError("at least one threshold is required")
    if any(not 0 <= threshold <= 1 for threshold in thresholds):
        raise ValueError("thresholds must be between 0 and 1")

    loaded = load_package(package)
    detector = (detector_factory or _default_detector_factory)(
        loaded, min(thresholds))
    positive_results = _run_folder(detector, Path(positives))
    negative_results = _run_folder(detector, Path(negatives))
    labels_dir = Path(positives) / "labels"

    ground_truth = {
        source_id: _read_boxes(labels_dir / f"{Path(source_id).stem}.txt")
        for source_id, _ in positive_results
    }
    eligible_images = sum(bool(boxes) for boxes in ground_truth.values())
    if eligible_images == 0:
        raise ValueError("positive dataset has no ground-truth instances")

    rows = []
    threshold_results = []
    for threshold in thresholds:
        recalled_images = 0
        for source_id, detections in positive_results:
            selected = [d for d in detections if d.score >= threshold]
            if ground_truth[source_id] and any(
                _iou(box, detection) >= 0.5
                for box in ground_truth[source_id]
                for detection in selected
            ):
                recalled_images += 1
        false_positives = sum(
            detection.score >= threshold
            for _, detections in negative_results
            for detection in detections
        )
        row = ThresholdResult(
            threshold=float(threshold),
            recall=recalled_images / eligible_images,
            false_positives=false_positives,
        )
        rows.append(row)
        threshold_results.append({
            "threshold": row.threshold,
            "recall": row.recall,
            "false_positives": row.false_positives,
        })

    selected = select_operating_confidence(rows)
    return {
        "package": str(package),
        "positive_images": eligible_images,
        "negative_images": len(negative_results),
        "thresholds": threshold_results,
        "selected_threshold": selected.threshold,
        "selected_recall": selected.recall,
        "selected_false_positives": selected.false_positives,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument("--negatives", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[round(value / 100, 2) for value in range(25, 91, 5)],
    )
    args = parser.parse_args()
    result = evaluate_thresholds(
        args.package, args.positives, args.negatives, args.thresholds)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
