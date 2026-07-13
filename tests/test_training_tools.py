import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.training.prepare_dataset import prepare_dataset


def jpeg_bytes(value: int) -> bytes:
    image = np.full((24, 24, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def write_zip(path: Path, files: dict[str, bytes | str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(
                name, content.encode("utf-8") if isinstance(content, str)
                else content)
    return path


def write_config(path: Path) -> Path:
    path.write_text(json.dumps({
        "positive": {
            "class_ids": {"toilet_paper_roll": 1},
            "split_names": {"train": "train", "valid": "val", "test": "test"},
        },
        "negative": {"required_class_ids": [41]},
        "seed": 713,
    }), encoding="utf-8")
    return path


def test_prepare_dataset_remaps_positives_and_imports_cup_backgrounds(tmp_path):
    positive_zip = write_zip(tmp_path / "positive.zip", {
        "train/images/roll.jpg": jpeg_bytes(20),
        "train/labels/roll.txt": "1 0.5 0.5 0.4 0.4\n0 0.2 0.2 0.1 0.1\n",
        "valid/images/roll_val.jpg": jpeg_bytes(80),
        "valid/labels/roll_val.txt": "1 0.5 0.5 0.5 0.5\n",
        "test/images/roll_test.jpg": jpeg_bytes(140),
        "test/labels/roll_test.txt": "1 0.5 0.5 0.3 0.3\n",
    })
    negative_zip = write_zip(tmp_path / "negative.zip", {
        "coco128/images/train2017/cup.jpg": jpeg_bytes(200),
        "coco128/labels/train2017/cup.txt": "41 0.5 0.5 0.4 0.4\n",
        "coco128/images/train2017/dog.jpg": jpeg_bytes(240),
        "coco128/labels/train2017/dog.txt": "16 0.5 0.5 0.4 0.4\n",
    })
    out = tmp_path / "dataset"

    report = prepare_dataset(
        positive_zip, negative_zip, write_config(tmp_path / "source.json"), out)

    assert report.errors == []
    assert (out / "labels/train/roll.txt").read_text(encoding="utf-8") == (
        "0 0.5 0.5 0.4 0.4\n")
    cup_labels = list((out / "labels").rglob("negative_cup.txt"))
    assert len(cup_labels) == 1
    assert cup_labels[0].read_text(encoding="utf-8") == ""
    assert not list((out / "images").rglob("negative_dog.jpg"))
    assert json.loads((out / "data.yaml").read_text(encoding="utf-8"))[
        "names"] == ["toilet_paper_roll"]
    assert (out / "dataset-report.json").is_file()


def test_prepare_dataset_rejects_archive_path_traversal(tmp_path):
    positive_zip = write_zip(tmp_path / "positive.zip", {
        "../outside.jpg": jpeg_bytes(20),
    })
    negative_zip = write_zip(tmp_path / "negative.zip", {})

    with pytest.raises(ValueError, match="unsafe archive path"):
        prepare_dataset(
            positive_zip, negative_zip,
            write_config(tmp_path / "source.json"), tmp_path / "dataset")

    assert not (tmp_path / "outside.jpg").exists()
