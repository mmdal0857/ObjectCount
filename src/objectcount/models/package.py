"""모델 패키지 — ONNX 가중치 + manifest.json 배포 단위 (스펙 §4-④)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SCHEMA_VERSION = 1
REQUIRED_FIELDS = ("schema_version", "product_id", "product_name",
                   "model_version", "input_size", "classes")


class ModelPackageError(Exception):
    pass


@dataclass(frozen=True)
class ModelPackage:
    root: Path
    product_id: str
    product_name: str
    model_version: str
    input_size: tuple[int, int]
    classes: list[str]
    active_class_ids: list[int] | None
    area_prior: float | None

    @property
    def model_path(self) -> Path:
        return self.root / "model.onnx"


def load_package(path: str | Path) -> ModelPackage:
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ModelPackageError(f"manifest.json이 없습니다: {root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = [f for f in REQUIRED_FIELDS if f not in manifest]
    if missing:
        raise ModelPackageError(f"manifest 필수 필드 누락: {', '.join(missing)}")
    if manifest["schema_version"] != SUPPORTED_SCHEMA_VERSION:
        raise ModelPackageError(
            f"지원하지 않는 schema_version: {manifest['schema_version']}")

    package = ModelPackage(
        root=root,
        product_id=manifest["product_id"],
        product_name=manifest["product_name"],
        model_version=manifest["model_version"],
        input_size=tuple(manifest["input_size"]),
        classes=list(manifest["classes"]),
        active_class_ids=manifest.get("active_class_ids"),
        area_prior=manifest.get("area_prior"),
    )
    if not package.model_path.is_file():
        raise ModelPackageError(f"model.onnx가 없습니다: {package.model_path}")
    return package
