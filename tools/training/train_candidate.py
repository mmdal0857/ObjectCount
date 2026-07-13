"""Train a one-class Product Candidate and package its best ONNX checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from objectcount.models.package import load_package

TrainerFactory = Callable[[str | Path], Any]


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        import yaml

        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"training config must be a mapping: {path}")
    return loaded


def _default_trainer_factory(model: str | Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(model))


def _replace_directory(staging: Path, destination: Path) -> None:
    backup = destination.parent / (
        f".{destination.name}.backup-{uuid4().hex}")
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def train_candidate(
    config: Path,
    data: Path,
    out: Path,
    trainer_factory: TrainerFactory | None = None,
) -> Path:
    """Train, export, validate, and atomically install a candidate package."""
    config = Path(config)
    data = Path(data)
    out = Path(out)
    settings = _load_config(config)
    factory = trainer_factory or _default_trainer_factory

    required = {"model", "imgsz", "project", "name"}
    missing = sorted(required - settings.keys())
    if missing:
        raise ValueError(f"training config fields missing: {', '.join(missing)}")

    training_args = dict(settings)
    model_name = training_args.pop("model")
    model_path = Path(str(model_name))
    if not model_path.is_absolute() and model_path.parent == Path("."):
        model_path = Path(training_args["project"]).parent / "pretrained" / model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_name = model_path
    training_args["data"] = str(data)
    trainer = factory(model_name)
    result = trainer.train(**training_args)
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"best checkpoint was not produced: {best}")

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{out.name}.tmp-", dir=out.parent))
    try:
        exporter = factory(best)
        exported = Path(exporter.export(
            format="onnx", imgsz=int(settings["imgsz"]), opset=18))
        if not exported.is_file():
            raise FileNotFoundError(f"ONNX export was not produced: {exported}")
        shutil.copy2(exported, staging / "model.onnx")

        manifest = {
            "schema_version": 1,
            "product_id": "toilet-paper-roll",
            "product_name": "Toilet Paper Roll",
            "model_version": str(settings["name"]),
            "input_size": [int(settings["imgsz"]), int(settings["imgsz"])],
            "classes": ["toilet_paper_roll"],
            "active_class_ids": [0],
            "area_prior": None,
            "candidate_status": "candidate",
            "default_confidence": None,
            "original_incident": "unverified",
            "metrics": dict(getattr(result, "results_dict", {}) or {}),
            "created_at": date.today().isoformat(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        load_package(staging)
        _replace_directory(staging, out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package = train_candidate(args.config, args.data, args.out)
    print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
