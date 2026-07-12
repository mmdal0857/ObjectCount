"""감지 CLI — 이미지/폴더/영상에 모델 패키지를 돌려 JSONL과 주석 이미지를 남긴다."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2

from ..detection.detector import OnnxDetector
from ..detection.roi import parse_roi
from ..frames.sources import iter_frames
from ..models.package import load_package
from ..viz import draw_detections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="objectcount-detect")
    parser.add_argument("--package", required=True, help="모델 패키지 디렉토리")
    parser.add_argument("--input", required=True,
                        help="이미지 파일·이미지 폴더·영상 파일")
    parser.add_argument("--out", required=True, help="출력 디렉토리")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--roi", help='정규화 폴리곤 "x1,y1;x2,y2;..."')
    args = parser.parse_args(argv)

    package = load_package(args.package)
    detector = OnnxDetector(package, conf_threshold=args.conf,
                            iou_threshold=args.iou)
    roi = parse_roi(args.roi) if args.roi else None

    out_dir = Path(args.out)
    annotated_dir = out_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_detections = 0
    with open(out_dir / "detections.jsonl", "w", encoding="utf-8") as sink:
        for record in iter_frames(args.input):
            detections = detector.detect(record.image)
            if roi is not None:
                detections = roi.filter(detections)
            sink.write(json.dumps({
                "source": record.source_id,
                "frame": record.frame_index,
                "detections": [asdict(d) for d in detections],
            }, ensure_ascii=False) + "\n")
            annotated = draw_detections(record.image, detections)
            cv2.imwrite(str(annotated_dir /
                            f"{record.source_id}_{record.frame_index}.jpg"),
                        annotated)
            total_frames += 1
            total_detections += len(detections)

    print(f"{total_frames}프레임에서 {total_detections}건 감지 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
