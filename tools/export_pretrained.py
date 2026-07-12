"""사전학습 COCO 모델을 모델 패키지로 변환하는 수동 도구.

사용: python tools/export_pretrained.py [--model yolo11n.pt] [--out models/packages/coco-demo]
ultralytics가 필요하다: pip install -e ".[tools]"
네트워크에서 가중치(~6MB)를 내려받으므로 pytest 대상이 아니다.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from ultralytics import YOLO

INPUT_SIZE = 640


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--out", default="models/packages/coco-demo")
    args = parser.parse_args()

    model = YOLO(args.model)
    onnx_path = Path(model.export(format="onnx", imgsz=INPUT_SIZE, opset=17))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, out_dir / "model.onnx")

    class_names = [model.names[i] for i in sorted(model.names)]
    manifest = {
        "schema_version": 1,
        "product_id": "coco-demo",
        "product_name": "COCO 사전학습 데모 (80클래스)",
        "model_version": f"{Path(args.model).stem}-{date.today().isoformat()}",
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "classes": class_names,
        "active_class_ids": None,
        "area_prior": None,
        "metrics": None,
        "created_at": date.today().isoformat(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"모델 패키지 생성 완료: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
