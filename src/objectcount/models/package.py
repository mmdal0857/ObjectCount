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

    # Validate input_size: must be list/tuple of exactly 2 positive ints
    input_size = manifest["input_size"]
    if not isinstance(input_size, (list, tuple)) or len(input_size) != 2:
        raise ModelPackageError(
            f"input_size는 [w, h] 두 정수여야 합니다: {input_size!r}")
    for v in input_size:
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise ModelPackageError(
                f"input_size는 [w, h] 두 정수여야 합니다: {input_size!r}")

    # Validate classes: must be non-empty list of strings
    classes = manifest["classes"]
    if not isinstance(classes, list) or len(classes) == 0:
        raise ModelPackageError(
            f"classes는 비어있지 않은 문자열 목록이어야 합니다: {classes!r}")
    if not all(isinstance(c, str) for c in classes):
        raise ModelPackageError(
            f"classes는 비어있지 않은 문자열 목록이어야 합니다: {classes!r}")

    # Validate optional active_class_ids: non-empty int list within classes range
    active_class_ids = manifest.get("active_class_ids")
    if active_class_ids is not None:
        valid = (isinstance(active_class_ids, list) and active_class_ids
                 and all(isinstance(i, int) and not isinstance(i, bool)
                         and 0 <= i < len(classes) for i in active_class_ids))
        if not valid:
            raise ModelPackageError(
                f"active_class_ids는 0..{len(classes) - 1} 범위의 "
                f"비어있지 않은 정수 목록이어야 합니다: {active_class_ids!r}")

    # Validate optional area_prior: positive number
    area_prior = manifest.get("area_prior")
    if area_prior is not None:
        if (not isinstance(area_prior, (int, float))
                or isinstance(area_prior, bool) or area_prior <= 0):
            raise ModelPackageError(
                f"area_prior는 양수여야 합니다: {area_prior!r}")

    package = ModelPackage(
        root=root,
        product_id=manifest["product_id"],
        product_name=manifest["product_name"],
        model_version=manifest["model_version"],
        input_size=tuple(manifest["input_size"]),
        classes=list(manifest["classes"]),
        active_class_ids=active_class_ids,
        area_prior=area_prior,
    )
    if not package.model_path.is_file():
        raise ModelPackageError(f"model.onnx가 없습니다: {package.model_path}")
    return package
