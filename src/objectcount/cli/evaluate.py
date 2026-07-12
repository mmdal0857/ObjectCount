"""감지 계수 평가 — 리플레이 하네스(스펙 §8)의 감지 단계 시드.

데이터셋 = 이미지들 + expected_counts.json({"파일명": 기대개수}).
이미지별 감지 개수를 기대값과 대조해 보고한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..detection.detector import OnnxDetector
from ..frames.sources import iter_frames
from ..models.package import load_package


def evaluate_dataset(package_path: Path, dataset_dir: Path,
                     conf: float = 0.25) -> dict:
    dataset_dir = Path(dataset_dir)
    expected = json.loads(
        (dataset_dir / "expected_counts.json").read_text(encoding="utf-8"))
    detector = OnnxDetector(load_package(package_path), conf_threshold=conf)

    per_image: dict[str, dict[str, int]] = {}
    for record in iter_frames(dataset_dir):
        if record.source_id not in expected:
            continue
        detections = detector.detect(record.image)
        per_image[record.source_id] = {
            "expected": expected[record.source_id],
            "detected": len(detections),
        }

    matched = sum(1 for r in per_image.values()
                  if r["expected"] == r["detected"])
    return {"per_image": per_image, "matched": matched,
            "total": len(per_image)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="objectcount-eval")
    parser.add_argument("--package", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args(argv)

    report = evaluate_dataset(Path(args.package), Path(args.dataset),
                              conf=args.conf)
    for name, row in sorted(report["per_image"].items()):
        delta = row["detected"] - row["expected"]
        print(f"{name}: 기대 {row['expected']} / 감지 {row['detected']}"
              f" ({'일치' if delta == 0 else f'{delta:+d}'})")
    print(f"일치 {report['matched']}/{report['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
