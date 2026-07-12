import json

import pytest

from objectcount.models.package import ModelPackageError, load_package

VALID = {
    "schema_version": 1,
    "product_id": "demo",
    "product_name": "데모 품종",
    "model_version": "v1",
    "input_size": [64, 64],
    "classes": ["can", "box"],
}


def write_package(tmp_path, manifest, with_model=True):
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    if with_model:
        (tmp_path / "model.onnx").write_bytes(b"\x00")
    return tmp_path


def test_load_valid_package(tmp_path):
    pkg = load_package(write_package(tmp_path, VALID))
    assert pkg.product_id == "demo"
    assert pkg.input_size == (64, 64)
    assert pkg.classes == ["can", "box"]
    assert pkg.active_class_ids is None
    assert pkg.area_prior is None
    assert pkg.model_path == tmp_path / "model.onnx"


def test_optional_fields_pass_through(tmp_path):
    manifest = dict(VALID, active_class_ids=[0], area_prior=0.02)
    pkg = load_package(write_package(tmp_path, manifest))
    assert pkg.active_class_ids == [0]
    assert pkg.area_prior == 0.02


def test_missing_required_field_raises(tmp_path):
    manifest = {k: v for k, v in VALID.items() if k != "classes"}
    with pytest.raises(ModelPackageError, match="classes"):
        load_package(write_package(tmp_path, manifest))


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(ModelPackageError, match="model.onnx"):
        load_package(write_package(tmp_path, VALID, with_model=False))


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(ModelPackageError, match="manifest.json"):
        load_package(tmp_path)


def test_unsupported_schema_version_raises(tmp_path):
    manifest = dict(VALID, schema_version=99)
    with pytest.raises(ModelPackageError, match="schema_version"):
        load_package(write_package(tmp_path, manifest))
