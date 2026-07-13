"""Assemble reviewed web, phone, and hard-negative Product Candidate data."""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from objectcount.registration.dataset import (
    IMAGE_SUFFIXES,
    SPLITS,
    DatasetReport,
    validate_yolo_dataset,
)
from tools.training.train_candidate import _replace_directory


def _prepare_layout(root: Path) -> None:
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"missing image directory: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _normalized_positive(label: Path, kind: str) -> str:
    if not label.is_file():
        raise ValueError(f"missing {kind} label: {label}")
    normalized = []
    for line_number, line in enumerate(
        label.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"invalid {kind} label: {label}:{line_number}")
        try:
            float(parts[0])
            [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(
                f"invalid {kind} label: {label}:{line_number}"
            ) from exc
        normalized.append("0 " + " ".join(parts[1:]))
    if not normalized:
        raise ValueError(f"empty {kind} positive label: {label}")
    return "\n".join(normalized) + "\n"


def _copy_sample(
    source_root: Path,
    image: Path,
    label_text: str,
    staging: Path,
    split: str,
    prefix: str,
    source_kind: str,
    provenance: list[dict[str, Any]],
    session: str | None = None,
) -> None:
    output_name = f"{prefix}_{image.name}"
    output_image = Path("images") / split / output_name
    output_label = Path("labels") / split / f"{Path(output_name).stem}.txt"
    if (staging / output_image).exists():
        raise ValueError(f"source name collision: {output_image.as_posix()}")
    shutil.copy2(image, staging / output_image)
    (staging / output_label).write_text(label_text, encoding="utf-8")
    provenance.append(
        {
            "source_kind": source_kind,
            "session": session,
            "source_path": image.relative_to(source_root).as_posix(),
            "split": split,
            "output_path": output_image.as_posix(),
        }
    )


def _load_splits(path: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sessions = payload.get("sessions")
    frozen = payload.get("frozen_test_sessions")
    if not isinstance(sessions, dict) or not isinstance(frozen, list):
        raise ValueError("invalid split config")
    frozen_set = set(frozen)
    if len(frozen_set) != len(frozen) or not frozen_set.issubset(sessions):
        raise ValueError("invalid frozen session declarations")
    for session, assignment in sessions.items():
        split = assignment.get("split") if isinstance(assignment, dict) else None
        if split not in SPLITS:
            raise ValueError(f"unknown session split: {session}")
        if session in frozen_set and split != "test":
            raise ValueError(f"frozen session must use test: {session}")
    return sessions, frozen_set


def _copy_web(
    root: Path, staging: Path, provenance: list[dict[str, Any]]
) -> int:
    images = _images(root / "images")
    for image in images:
        label = root / "labels" / f"{image.stem}.txt"
        _copy_sample(
            root,
            image,
            _normalized_positive(label, "web"),
            staging,
            "train",
            "web",
            "web",
            provenance,
        )
    return len(images)


def _copy_phone(
    root: Path,
    staging: Path,
    sessions: dict[str, dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> int:
    count = 0
    for split in SPLITS:
        image_dir = root / split / "images"
        if not image_dir.exists():
            continue
        for image in _images(image_dir):
            session, separator, _sequence = image.stem.rpartition("_")
            if not separator or session not in sessions:
                raise ValueError(
                    f"phone session absent from split config: {image.name}"
                )
            assigned = sessions[session]["split"]
            if assigned != split:
                raise ValueError(
                    f"phone session split mismatch: {session} ({split} != {assigned})"
                )
            label = root / split / "labels" / f"{image.stem}.txt"
            _copy_sample(
                root,
                image,
                _normalized_positive(label, "phone"),
                staging,
                split,
                "phone",
                "phone",
                provenance,
                session,
            )
            count += 1
    return count


def _copy_negatives(
    root: Path, staging: Path, provenance: list[dict[str, Any]]
) -> int:
    images = _images(root / "images")
    for image in images:
        label = root / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            raise ValueError(f"missing negative label: {label}")
        if label.read_text(encoding="utf-8").strip():
            raise ValueError(f"negative label must be empty: {label}")
        _copy_sample(
            root,
            image,
            "",
            staging,
            "train",
            "negative",
            "negative",
            provenance,
        )
    return len(images)


def _composition_warnings(
    report: DatasetReport, web_count: int, phone_count: int, negative_count: int
) -> None:
    for label, count, minimum in (
        ("web positives", web_count, 50),
        ("phone positives", phone_count, 100),
        ("hard negatives", negative_count, 100),
    ):
        if count < minimum:
            report.warnings.append(
                f"composition shortfall: {label} {count} < {minimum}"
            )


def build_candidate_v2(
    web_root: Path,
    phone_root: Path,
    negative_root: Path,
    splits_path: Path,
    output: Path,
) -> DatasetReport:
    web_root = Path(web_root)
    phone_root = Path(phone_root)
    negative_root = Path(negative_root)
    output = Path(output)
    sessions, _frozen = _load_splits(Path(splits_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        _prepare_layout(staging)
        provenance: list[dict[str, Any]] = []
        web_count = _copy_web(web_root, staging, provenance)
        phone_count = _copy_phone(phone_root, staging, sessions, provenance)
        negative_count = _copy_negatives(negative_root, staging, provenance)
        (staging / "data.yaml").write_text(
            json.dumps(
                {
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": ["toilet_paper_roll"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (staging / "provenance.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in provenance),
            encoding="utf-8",
        )
        report = validate_yolo_dataset(staging)
        _composition_warnings(report, web_count, phone_count, negative_count)
        (staging / "dataset-report.json").write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if report.errors:
            raise ValueError("dataset validation failed: " + "; ".join(report.errors))
        _replace_directory(staging, output)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)
