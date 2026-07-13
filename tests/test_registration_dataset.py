from pathlib import Path

import cv2
import numpy as np

from objectcount.registration.dataset import validate_yolo_dataset


SPLITS = ("train", "val", "test")


def make_dataset(root: Path) -> Path:
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    return root


def write_sample(root: Path, split: str, stem: str, image: np.ndarray,
                 label: str | None) -> None:
    assert cv2.imwrite(str(root / "images" / split / f"{stem}.jpg"), image)
    if label is not None:
        (root / "labels" / split / f"{stem}.txt").write_text(
            label, encoding="utf-8")


def test_reports_valid_positive_and_background_samples(tmp_path):
    root = make_dataset(tmp_path)
    write_sample(root, "train", "roll", np.zeros((32, 32, 3), np.uint8),
                 "0 0.5 0.5 0.4 0.4\n")
    write_sample(root, "val", "cup", np.full((32, 32, 3), 127, np.uint8),
                 "")

    report = validate_yolo_dataset(root)

    assert report.errors == []
    assert report.split_images == {"train": 1, "val": 1, "test": 0}
    assert report.positive_instances == 1
    assert report.background_images == 1


def test_missing_label_is_an_error(tmp_path):
    root = make_dataset(tmp_path)
    write_sample(root, "train", "roll", np.zeros((32, 32, 3), np.uint8),
                 None)

    report = validate_yolo_dataset(root)

    assert any("missing label" in error for error in report.errors)


def test_invalid_class_id_is_an_error(tmp_path):
    root = make_dataset(tmp_path)
    write_sample(root, "train", "roll", np.zeros((32, 32, 3), np.uint8),
                 "1 0.5 0.5 0.4 0.4\n")

    report = validate_yolo_dataset(root, class_count=1)

    assert any("class id" in error for error in report.errors)


def test_zero_area_box_is_an_error(tmp_path):
    root = make_dataset(tmp_path)
    write_sample(root, "train", "roll", np.zeros((32, 32, 3), np.uint8),
                 "0 0.5 0.5 0.0 0.4\n")

    report = validate_yolo_dataset(root)

    assert any("box" in error for error in report.errors)


def test_out_of_range_box_is_an_error(tmp_path):
    root = make_dataset(tmp_path)
    write_sample(root, "train", "roll", np.zeros((32, 32, 3), np.uint8),
                 "0 1.2 0.5 0.4 0.4\n")

    report = validate_yolo_dataset(root)

    assert any("box" in error for error in report.errors)


def test_duplicate_image_across_splits_is_error(tmp_path):
    root = make_dataset(tmp_path)
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    write_sample(root, "train", "a", image,
                 "0 0.5 0.5 0.4 0.4\n")
    write_sample(root, "val", "b", image,
                 "0 0.5 0.5 0.4 0.4\n")

    report = validate_yolo_dataset(root)

    assert report.exact_duplicates
    assert any("duplicate" in error for error in report.errors)


def test_near_duplicate_across_splits_is_warning(tmp_path):
    root = make_dataset(tmp_path)
    image_a = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
    image_a = cv2.cvtColor(image_a, cv2.COLOR_GRAY2BGR)
    image_b = image_a.copy()
    image_b[0, 0] = 255
    write_sample(root, "train", "a", image_a,
                 "0 0.5 0.5 0.4 0.4\n")
    write_sample(root, "test", "b", image_b,
                 "0 0.5 0.5 0.4 0.4\n")

    report = validate_yolo_dataset(root)

    assert report.near_duplicates
    assert any("near duplicate" in warning for warning in report.warnings)
