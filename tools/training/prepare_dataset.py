"""승인된 YOLO ZIP을 단일 클래스 등록 데이터셋으로 정규화한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from objectcount.registration.dataset import SPLITS, DatasetReport, validate_yolo_dataset


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename.replace("\\", "/"))
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe archive path: {info.filename}")
        archive.extractall(destination)


def _find_directory(root: Path, suffix: tuple[str, ...]) -> Path:
    matches = [path for path in root.rglob(suffix[-1])
               if path.is_dir() and path.parts[-len(suffix):] == suffix]
    if len(matches) != 1:
        raise ValueError(f"expected one {'/'.join(suffix)} directory, found {len(matches)}")
    return matches[0]


def _negative_split(name: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()[:8], 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "val"
    return "test"


def _prepare_layout(root: Path) -> None:
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)


def _copy_positives(extracted: Path, output: Path, config: dict) -> None:
    positive = config["positive"]
    accepted_ids = set(positive["class_ids"].values())
    if len(accepted_ids) != 1:
        raise ValueError("exactly one positive source class is required")
    accepted_id = next(iter(accepted_ids))

    for source_split, target_split in positive["split_names"].items():
        image_dir = _find_directory(extracted, (source_split, "images"))
        label_dir = _find_directory(extracted, (source_split, "labels"))
        for image_path in sorted(path for path in image_dir.iterdir()
                                 if path.is_file()):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise ValueError(f"missing source label: {label_path}")
            normalized: list[str] = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"invalid source label: {label_path}")
                if int(parts[0]) == accepted_id:
                    normalized.append("0 " + " ".join(parts[1:]))
            shutil.copy2(image_path, output / "images" / target_split / image_path.name)
            text = "\n".join(normalized)
            if text:
                text += "\n"
            (output / "labels" / target_split / f"{image_path.stem}.txt").write_text(
                text, encoding="utf-8")


def _copy_negatives(extracted: Path, output: Path, config: dict) -> None:
    required_ids = set(config["negative"]["required_class_ids"])
    image_dir = _find_directory(extracted, ("images", "train2017"))
    label_dir = _find_directory(extracted, ("labels", "train2017"))
    for label_path in sorted(label_dir.glob("*.txt")):
        class_ids = {int(line.split()[0]) for line in
                     label_path.read_text(encoding="utf-8").splitlines()
                     if line.strip()}
        if not class_ids.intersection(required_ids):
            continue
        candidates = list(image_dir.glob(f"{label_path.stem}.*"))
        if len(candidates) != 1:
            raise ValueError(f"expected one negative image for {label_path}")
        image_path = candidates[0]
        split = _negative_split(image_path.name, config["seed"])
        target_stem = f"negative_{image_path.stem}"
        shutil.copy2(
            image_path,
            output / "images" / split / f"{target_stem}{image_path.suffix.lower()}")
        (output / "labels" / split / f"{target_stem}.txt").write_text(
            "", encoding="utf-8")


def prepare_dataset(source_zip: Path, negative_zip: Path, config_path: Path,
                    output: Path) -> DatasetReport:
    source_zip = Path(source_zip)
    negative_zip = Path(negative_zip)
    output = Path(output)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        positive_root = temporary_path / "positive"
        negative_root = temporary_path / "negative"
        staging = temporary_path / "dataset"
        positive_root.mkdir()
        negative_root.mkdir()
        _prepare_layout(staging)
        _safe_extract(source_zip, positive_root)
        _safe_extract(negative_zip, negative_root)
        _copy_positives(positive_root, staging, config)
        _copy_negatives(negative_root, staging, config)
        (staging / "data.yaml").write_text(json.dumps({
            "path": str(output.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": ["toilet_paper_roll"],
        }, indent=2), encoding="utf-8")

        report = validate_yolo_dataset(staging)
        (staging / "dataset-report.json").write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8")
        if report.errors:
            raise ValueError("dataset validation failed: " + "; ".join(report.errors))

        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(staging, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--negative-zip", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_dataset(
        args.source_zip, args.negative_zip, args.config, args.out)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
