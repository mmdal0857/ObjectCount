import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.training.build_candidate_v2 import build_candidate_v2


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((32, 32, 3), value, np.uint8))


def _label(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _splits(path: Path, split: str = "train", frozen: bool = False) -> Path:
    payload = {
        "schema_version": 1,
        "sessions": {"session-a": {"split": split, "source": "unused"}},
        "frozen_test_sessions": ["session-a"] if frozen else [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sources(tmp_path: Path):
    web = tmp_path / "web"
    _image(web / "images/roll.jpg", 20)
    _label(web / "labels/roll.txt", "0 0.5 0.5 0.5 0.5\n")

    phone = tmp_path / "phone"
    _image(phone / "train/images/session-a_000001.jpg", 80)
    _label(
        phone / "train/labels/session-a_000001.txt",
        "0 0.5 0.5 0.4 0.4\n",
    )

    negatives = tmp_path / "negatives"
    _image(negatives / "images/cup.jpg", 140)
    _label(negatives / "labels/cup.txt", "")
    return web, phone, negatives


def test_builds_valid_dataset_with_provenance_and_source_prefixes(tmp_path):
    web, phone, negatives = _sources(tmp_path)
    output = tmp_path / "dataset"

    report = build_candidate_v2(
        web, phone, negatives, _splits(tmp_path / "splits.json"), output
    )

    assert report.errors == []
    assert (output / "labels/train/web_roll.txt").read_text() == (
        "0 0.5 0.5 0.5 0.5\n"
    )
    assert (output / "labels/train/phone_session-a_000001.txt").read_text() == (
        "0 0.5 0.5 0.4 0.4\n"
    )
    assert (output / "labels/train/negative_cup.txt").read_text() == ""
    provenance = [
        json.loads(line)
        for line in (output / "provenance.jsonl").read_text().splitlines()
    ]
    assert {row["source_kind"] for row in provenance} == {
        "web",
        "phone",
        "negative",
    }
    assert next(row for row in provenance if row["source_kind"] == "phone")[
        "session"
    ] == "session-a"
    assert json.loads((output / "dataset-report.json").read_text())[
        "positive_instances"
    ] == 2


@pytest.mark.parametrize("kind", ["web", "negative"])
def test_rejects_invalid_positive_or_negative_labels_without_replacing_output(
    tmp_path, kind
):
    web, phone, negatives = _sources(tmp_path)
    if kind == "web":
        _label(web / "labels/roll.txt", "")
    else:
        _label(negatives / "labels/cup.txt", "0 0.5 0.5 0.2 0.2\n")
    output = tmp_path / "dataset"
    output.mkdir()
    (output / "keep.txt").write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match=kind):
        build_candidate_v2(
            web, phone, negatives, _splits(tmp_path / "splits.json"), output
        )

    assert (output / "keep.txt").read_text() == "old"


def test_rejects_phone_session_missing_from_split_config(tmp_path):
    web, phone, negatives = _sources(tmp_path)
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"schema_version": 1, "sessions": {}, "frozen_test_sessions": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absent from split config"):
        build_candidate_v2(web, phone, negatives, splits, tmp_path / "dataset")


def test_rejects_frozen_phone_session_outside_test(tmp_path):
    web, phone, negatives = _sources(tmp_path)

    with pytest.raises(ValueError, match="frozen session"):
        build_candidate_v2(
            web,
            phone,
            negatives,
            _splits(tmp_path / "splits.json", "train", frozen=True),
            tmp_path / "dataset",
        )


def test_rejects_exact_duplicate_images_across_splits(tmp_path):
    web, phone, negatives = _sources(tmp_path)
    phone_image = phone / "train/images/session-a_000001.jpg"
    phone_image.write_bytes((web / "images/roll.jpg").read_bytes())
    phone_test = phone / "test/images/session-a_000001.jpg"
    phone_test.parent.mkdir(parents=True)
    phone_image.replace(phone_test)
    phone_label = phone / "train/labels/session-a_000001.txt"
    test_label = phone / "test/labels/session-a_000001.txt"
    test_label.parent.mkdir(parents=True)
    phone_label.replace(test_label)

    with pytest.raises(ValueError, match="duplicate image across splits"):
        build_candidate_v2(
            web,
            phone,
            negatives,
            _splits(tmp_path / "splits.json", "test"),
            tmp_path / "dataset",
        )


def test_preserves_standard_yolo_web_and_negative_splits(tmp_path):
    web = tmp_path / "web"
    _image(web / "images/val/roll.jpg", 30)
    _label(web / "labels/val/roll.txt", "0 0.5 0.5 0.5 0.5\n")
    phone = tmp_path / "phone"
    negatives = tmp_path / "negatives"
    _image(negatives / "images/test/cup.jpg", 160)
    _label(negatives / "labels/test/cup.txt", "")

    build_candidate_v2(
        web, phone, negatives, _splits(tmp_path / "splits.json"), tmp_path / "dataset"
    )

    assert (tmp_path / "dataset/images/val/web_roll.jpg").is_file()
    assert (tmp_path / "dataset/labels/test/negative_cup.txt").read_text() == ""
