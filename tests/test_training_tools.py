import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools.training.prepare_dataset import prepare_dataset
from objectcount.detection.types import Detection
from tools.training.evaluate_candidate import evaluate_thresholds
from tools.training.train_candidate import train_candidate


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


class FakeTrainer:
    instances = []

    def __init__(self, model):
        self.model = Path(model)
        self.__class__.instances.append(self)

    def train(self, **kwargs):
        save_dir = Path(kwargs["project"]) / kwargs["name"]
        best = save_dir / "weights" / "best.pt"
        best.parent.mkdir(parents=True)
        best.write_bytes(b"best")
        return SimpleNamespace(
            save_dir=save_dir,
            results_dict={"metrics/mAP50(B)": 0.91},
        )

    def export(self, **kwargs):
        assert self.model.name == "best.pt"
        exported = self.model.with_suffix(".onnx")
        exported.write_bytes(b"onnx")
        return str(exported)


def write_train_config(path: Path) -> Path:
    path.write_text(json.dumps({
        "model": "yolo11n.pt",
        "imgsz": 640,
        "epochs": 1,
        "project": str(path.parent / "runs"),
        "name": "candidate-v1",
    }), encoding="utf-8")
    return path


def test_train_candidate_exports_best_and_writes_candidate_manifest(tmp_path):
    FakeTrainer.instances.clear()
    output = tmp_path / "package"

    package = train_candidate(
        write_train_config(tmp_path / "train.yaml"),
        tmp_path / "data.yaml",
        output,
        trainer_factory=FakeTrainer,
    )

    assert package == output
    assert [instance.model.name for instance in FakeTrainer.instances] == [
        "yolo11n.pt", "best.pt"]
    manifest = json.loads((output / "manifest.json").read_text("utf-8"))
    assert manifest["classes"] == ["toilet_paper_roll"]
    assert manifest["candidate_status"] == "candidate"
    assert manifest["original_incident"] == "unverified"
    assert (output / "model.onnx").read_bytes() == b"onnx"


def test_train_candidate_keeps_existing_package_when_export_fails(tmp_path):
    class FailingExportTrainer(FakeTrainer):
        def export(self, **kwargs):
            raise RuntimeError("export failed")

    output = tmp_path / "package"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("old", encoding="utf-8")

    with pytest.raises(RuntimeError, match="export failed"):
        train_candidate(
            write_train_config(tmp_path / "train.yaml"),
            tmp_path / "data.yaml",
            output,
            trainer_factory=FailingExportTrainer,
        )

    assert marker.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".package.tmp-*"))


def test_train_candidate_restores_existing_package_when_install_fails(
        tmp_path, monkeypatch):
    output = tmp_path / "package"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("old", encoding="utf-8")
    real_replace = os.replace

    def fail_staging_install(source, destination):
        source = Path(source)
        if source.name.startswith(".package.tmp-"):
            raise OSError("install failed")
        return real_replace(source, destination)

    monkeypatch.setattr(
        "tools.training.train_candidate.os.replace", fail_staging_install)

    with pytest.raises(OSError, match="install failed"):
        train_candidate(
            write_train_config(tmp_path / "train.yaml"),
            tmp_path / "data.yaml",
            output,
            trainer_factory=FakeTrainer,
        )

    assert marker.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".package.backup-*"))


def test_evaluate_thresholds_selects_zero_false_positive_threshold(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "product_id": "toilet-paper-roll",
        "product_name": "Toilet Paper Roll",
        "model_version": "candidate-v1",
        "input_size": [64, 64],
        "classes": ["toilet_paper_roll"],
    }), encoding="utf-8")
    (package / "model.onnx").write_bytes(b"unused")
    positives = tmp_path / "positives"
    negatives = tmp_path / "negatives"
    for root, value in ((positives, 20), (negatives, 220)):
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        cv2.imwrite(str(root / "images/sample.jpg"),
                    np.full((32, 32, 3), value, dtype=np.uint8))
    (positives / "labels/sample.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (negatives / "labels/sample.txt").write_text("", encoding="utf-8")

    class FakeDetector:
        def detect(self, image):
            score = 0.8 if image.mean() < 100 else 0.4
            return [Detection(0.25, 0.25, 0.75, 0.75, score, 0,
                              "toilet_paper_roll")]

    result = evaluate_thresholds(
        package,
        positives,
        negatives,
        [0.25, 0.5, 0.9],
        detector_factory=lambda loaded, threshold: FakeDetector(),
    )

    assert result["selected_threshold"] == 0.5
    assert result["thresholds"][1] == {
        "threshold": 0.5,
        "recall": 1.0,
        "false_positives": 0,
    }
