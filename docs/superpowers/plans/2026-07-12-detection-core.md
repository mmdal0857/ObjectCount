# 감지 코어(Phase 1a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 입력원(이미지 파일·폴더·영상, 향후 폰 스트림)과 무관하게 동작하는 오브젝트 감지 코어 — 이미지를 넣으면 감지 결과가 나오는 데까지.

**Architecture:** 순수 함수 계층(전처리 letterbox → ONNX 추론 → 디코드+NMS 후처리)을 `OnnxDetector`가 조립하고, 모델은 매니페스트가 딸린 "모델 패키지" 디렉토리로 교체한다(스펙 §4-④). 프레임 소스는 제너레이터 프로토콜로 추상화해 나중에 폰 스트림이 같은 자리에 꽂힌다. 추적·카운팅·캡처 앱·대시보드는 이 계획의 범위 밖.

**Tech Stack:** Python 3.11+, numpy, OpenCV, onnxruntime(실행 프로바이더 자동 선택), pytest. 테스트용 가짜 ONNX 생성에 `onnx`, 실모델 내보내기 도구에만 `ultralytics`.

## Global Constraints

- Python `>=3.11`, src 레이아웃 (`src/objectcount/`)
- 모델 가중치 포맷은 ONNX 고정 (스펙 §4-④)
- 추론 실행 프로바이더 우선순위: `CUDAExecutionProvider` → `DmlExecutionProvider` → `CPUExecutionProvider` (스펙 §4-④). 개발 PC는 AMD — 기본 `onnxruntime`(CPU)로 개발하고, GPU가 필요하면 `pip uninstall onnxruntime && pip install onnxruntime-directml`로 교체하면 코드는 그대로 DirectML을 집는다
- Detection 좌표는 원본 프레임 기준 `[0,1]` 정규화 xyxy — 해상도 독립(입력원 무관 원칙)
- 감지 신뢰도 기본값 `conf=0.25`, NMS IoU 기본값 `0.45`
- 커밋 메시지 끝에 트레일러 추가: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- 테스트 실행은 항상 `python -m pytest` (venv 활성 상태 전제)

---

### Task 1: 프로젝트 스캐폴드

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/objectcount/__init__.py`
- Create: `src/objectcount/detection/__init__.py`
- Create: `src/objectcount/models/__init__.py`
- Create: `src/objectcount/frames/__init__.py`
- Create: `src/objectcount/cli/__init__.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: 없음 (첫 태스크)
- Produces: 패키지 `objectcount` (버전 문자열 `objectcount.__version__`), 이후 모든 태스크의 빌드·테스트 환경

- [ ] **Step 1: pyproject.toml 작성**

```toml
[project]
name = "objectcount"
version = "0.1.0"
description = "다중 카메라 컨베이어 오브젝트 카운팅 — 감지 코어"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "opencv-python>=4.9",
    "onnxruntime>=1.17",
]

[project.optional-dependencies]
dev = ["pytest>=8", "onnx>=1.16"]
tools = ["ultralytics>=8.2"]

[project.scripts]
objectcount-detect = "objectcount.cli.detect:main"
objectcount-eval = "objectcount.cli.evaluate:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: .gitignore 작성**

```gitignore
.venv/
__pycache__/
*.pyc
*.egg-info/
build/
dist/
models/packages/
*.onnx
*.pt
out/
```

- [ ] **Step 3: 패키지 뼈대 작성**

`src/objectcount/__init__.py`:
```python
__version__ = "0.1.0"
```

`src/objectcount/detection/__init__.py`, `src/objectcount/models/__init__.py`, `src/objectcount/frames/__init__.py`, `src/objectcount/cli/__init__.py`: 전부 빈 파일.

- [ ] **Step 4: 스모크 테스트 작성**

`tests/test_smoke.py`:
```python
import objectcount


def test_package_importable():
    assert objectcount.__version__ == "0.1.0"
```

- [ ] **Step 5: venv 생성·설치·테스트 통과 확인**

Run (프로젝트 루트 `f:\Project\ObjectCount`):
```powershell
py -3.11 -m venv .venv          # 3.11 미설치 시 py -3 사용
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q
```
Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore src tests
git commit -m "feat: 프로젝트 스캐폴드 — src 레이아웃, pytest"
```

---

### Task 2: Detection 타입

**Files:**
- Create: `src/objectcount/detection/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: 없음
- Produces: `Detection(x1, y1, x2, y2, score, class_id, class_name)` frozen dataclass — 좌표는 [0,1] 정규화 xyxy. 속성 `area: float`, `center: tuple[float, float]`, 메서드 `to_pixels(width: int, height: int) -> tuple[int, int, int, int]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_types.py`:
```python
from objectcount.detection.types import Detection


def make(x1=0.25, y1=0.25, x2=0.75, y2=0.75, score=0.9):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=score,
                     class_id=0, class_name="can")


def test_area_is_normalized_box_area():
    assert make().area == 0.25


def test_center_is_box_midpoint():
    assert make().center == (0.5, 0.5)


def test_to_pixels_scales_and_rounds():
    assert make().to_pixels(width=200, height=100) == (50, 25, 150, 75)


def test_degenerate_box_has_zero_area():
    assert make(x1=0.8, x2=0.7).area == 0.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_types.py -q`
Expected: FAIL — `ModuleNotFoundError` 또는 `ImportError: Detection`

- [ ] **Step 3: 구현**

`src/objectcount/detection/types.py`:
```python
"""감지 결과 타입. 좌표는 원본 프레임 기준 [0,1] 정규화 xyxy — 해상도 독립."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            round(self.x1 * width),
            round(self.y1 * height),
            round(self.x2 * width),
            round(self.y2 * height),
        )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv\Scripts\python -m pytest tests/test_types.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/objectcount/detection/types.py tests/test_types.py
git commit -m "feat: Detection 타입 — 정규화 xyxy 좌표"
```

---

### Task 3: letterbox 전처리

**Files:**
- Create: `src/objectcount/detection/preprocess.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Consumes: 없음
- Produces: `letterbox(image_bgr: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, LetterboxMeta]` — BGR HxWx3 uint8을 (1,3,H,W) float32 [0,1] RGB 텐서로. `LetterboxMeta(scale: float, pad_x: int, pad_y: int, src_width: int, src_height: int)` frozen dataclass

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_preprocess.py`:
```python
import numpy as np

from objectcount.detection.preprocess import LetterboxMeta, letterbox


def test_letterbox_output_shape_and_dtype():
    image = np.zeros((50, 100, 3), dtype=np.uint8)  # h=50, w=100
    tensor, meta = letterbox(image, size=(64, 64))
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == np.float32


def test_letterbox_meta_scale_and_padding():
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    _, meta = letterbox(image, size=(64, 64))
    assert meta == LetterboxMeta(scale=0.64, pad_x=0, pad_y=16,
                                 src_width=100, src_height=50)


def test_letterbox_pads_with_gray_114():
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    tensor, _ = letterbox(image, size=(64, 64))
    pad_value = 114.0 / 255.0
    assert np.allclose(tensor[0, :, 0, :], pad_value)   # 위쪽 패딩 행
    assert np.allclose(tensor[0, :, 63, :], pad_value)  # 아래쪽 패딩 행


def test_letterbox_square_input_no_padding():
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    tensor, meta = letterbox(image, size=(64, 64))
    assert meta.scale == 1.0 and meta.pad_x == 0 and meta.pad_y == 0
    assert np.allclose(tensor, 1.0)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_preprocess.py -q`
Expected: FAIL — `ImportError`

- [ ] **Step 3: 구현**

`src/objectcount/detection/preprocess.py`:
```python
"""letterbox 전처리 — 종횡비 유지 리사이즈 + 회색(114) 패딩."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAD_COLOR = 114


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    pad_x: int
    pad_y: int
    src_width: int
    src_height: int


def letterbox(image_bgr: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, LetterboxMeta]:
    target_w, target_h = size
    src_h, src_w = image_bgr.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas = np.full((target_h, target_w, 3), PAD_COLOR, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]
    meta = LetterboxMeta(scale=scale, pad_x=pad_x, pad_y=pad_y,
                         src_width=src_w, src_height=src_h)
    return np.ascontiguousarray(tensor), meta
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv\Scripts\python -m pytest tests/test_preprocess.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/objectcount/detection/preprocess.py tests/test_preprocess.py
git commit -m "feat: letterbox 전처리"
```

---

### Task 4: YOLO 출력 디코드 + NMS 후처리

**Files:**
- Create: `src/objectcount/detection/postprocess.py`
- Test: `tests/test_postprocess.py`

**Interfaces:**
- Consumes: `Detection` (Task 2), `LetterboxMeta` (Task 3)
- Produces:
  - `decode_yolo(raw: np.ndarray, meta: LetterboxMeta, class_names: list[str], conf_threshold: float = 0.25, active_class_ids: list[int] | None = None) -> list[Detection]` — ultralytics YOLOv8/11 ONNX 출력 `(1, 4+nc, N)`을 정규화 Detection으로
  - `nms(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]` — 클래스별 greedy NMS, 점수 내림차순 유지

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_postprocess.py`:
```python
import numpy as np
import pytest

from objectcount.detection.postprocess import decode_yolo, nms
from objectcount.detection.preprocess import LetterboxMeta
from objectcount.detection.types import Detection

CLASSES = ["can", "box"]
# 64x64 원본을 64x64로 letterbox → scale 1, 패딩 0
META = LetterboxMeta(scale=1.0, pad_x=0, pad_y=0, src_width=64, src_height=64)


def make_raw(rows: list[list[float]]) -> np.ndarray:
    """rows: [cx, cy, w, h, score_can, score_box] (letterbox 픽셀 좌표) → (1, 6, N)"""
    return np.array(rows, dtype=np.float32).T[np.newaxis, ...]


def test_decode_converts_center_box_to_normalized_xyxy():
    raw = make_raw([[32, 32, 16, 16, 0.9, 0.1]])
    dets = decode_yolo(raw, META, CLASSES)
    assert len(dets) == 1
    d = dets[0]
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx((0.375, 0.375, 0.625, 0.625))
    assert d.score == pytest.approx(0.9)
    assert d.class_id == 0 and d.class_name == "can"


def test_decode_filters_below_confidence():
    raw = make_raw([[32, 32, 16, 16, 0.1, 0.05]])
    assert decode_yolo(raw, META, CLASSES, conf_threshold=0.25) == []


def test_decode_respects_active_class_filter():
    raw = make_raw([
        [32, 32, 16, 16, 0.9, 0.1],   # can
        [10, 10, 8, 8, 0.1, 0.8],     # box
    ])
    dets = decode_yolo(raw, META, CLASSES, active_class_ids=[1])
    assert [d.class_name for d in dets] == ["box"]


def test_decode_unmaps_letterbox_padding():
    # 원본 100x50 → 64x64 letterbox: scale 0.64, pad_y 16
    meta = LetterboxMeta(scale=0.64, pad_x=0, pad_y=16, src_width=100, src_height=50)
    # letterbox 픽셀 (32, 32) = 원본 픽셀 (50, 25) = 정규화 (0.5, 0.5)
    raw = make_raw([[32, 32, 12.8, 6.4, 0.9, 0.0]])
    d = decode_yolo(raw, meta, CLASSES)[0]
    assert d.center == pytest.approx((0.5, 0.5))
    assert (d.x2 - d.x1) == pytest.approx(0.2)   # 12.8/0.64/100
    assert (d.y2 - d.y1) == pytest.approx(0.2)   # 6.4/0.64/50


def test_decode_clips_to_unit_range():
    raw = make_raw([[2, 2, 20, 20, 0.9, 0.0]])  # 박스가 프레임 밖으로 나감
    d = decode_yolo(raw, META, CLASSES)[0]
    assert d.x1 == 0.0 and d.y1 == 0.0


def _det(x1, y1, x2, y2, score, cid=0):
    return Detection(x1, y1, x2, y2, score, cid, CLASSES[cid])


def test_nms_suppresses_overlapping_same_class():
    kept = nms([_det(0.1, 0.1, 0.5, 0.5, 0.9),
                _det(0.12, 0.12, 0.52, 0.52, 0.8)], iou_threshold=0.45)
    assert len(kept) == 1 and kept[0].score == 0.9


def test_nms_keeps_different_classes():
    kept = nms([_det(0.1, 0.1, 0.5, 0.5, 0.9, cid=0),
                _det(0.1, 0.1, 0.5, 0.5, 0.8, cid=1)])
    assert len(kept) == 2


def test_nms_keeps_disjoint_boxes():
    kept = nms([_det(0.0, 0.0, 0.2, 0.2, 0.9),
                _det(0.5, 0.5, 0.9, 0.9, 0.8)])
    assert len(kept) == 2
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_postprocess.py -q`
Expected: FAIL — `ImportError`

- [ ] **Step 3: 구현**

`src/objectcount/detection/postprocess.py`:
```python
"""YOLOv8/11 ONNX 출력 디코드와 NMS. 전부 numpy — torch 의존 없음."""
from __future__ import annotations

import numpy as np

from .preprocess import LetterboxMeta
from .types import Detection


def decode_yolo(
    raw: np.ndarray,
    meta: LetterboxMeta,
    class_names: list[str],
    conf_threshold: float = 0.25,
    active_class_ids: list[int] | None = None,
) -> list[Detection]:
    pred = raw[0]                      # (4+nc, N)
    boxes = pred[:4, :]                # cx, cy, w, h — letterbox 픽셀
    scores = pred[4:, :]               # (nc, N)
    class_ids = scores.argmax(axis=0)
    confs = scores.max(axis=0)

    keep = confs >= conf_threshold
    if active_class_ids is not None:
        keep &= np.isin(class_ids, active_class_ids)

    detections: list[Detection] = []
    for cx, cy, w, h, conf, cid in zip(
        boxes[0, keep], boxes[1, keep], boxes[2, keep], boxes[3, keep],
        confs[keep], class_ids[keep],
    ):
        # letterbox 픽셀 → 원본 픽셀 → [0,1] 정규화
        x1 = (cx - w / 2 - meta.pad_x) / meta.scale / meta.src_width
        y1 = (cy - h / 2 - meta.pad_y) / meta.scale / meta.src_height
        x2 = (cx + w / 2 - meta.pad_x) / meta.scale / meta.src_width
        y2 = (cy + h / 2 - meta.pad_y) / meta.scale / meta.src_height
        detections.append(Detection(
            x1=float(np.clip(x1, 0.0, 1.0)),
            y1=float(np.clip(y1, 0.0, 1.0)),
            x2=float(np.clip(x2, 0.0, 1.0)),
            y2=float(np.clip(y2, 0.0, 1.0)),
            score=float(conf),
            class_id=int(cid),
            class_name=class_names[int(cid)],
        ))
    return detections


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        if all(k.class_id != det.class_id or _iou(k, det) < iou_threshold
               for k in kept):
            kept.append(det)
    return kept
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv\Scripts\python -m pytest tests/test_postprocess.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/objectcount/detection/postprocess.py tests/test_postprocess.py
git commit -m "feat: YOLO 디코드 + NMS 후처리 (numpy)"
```

---

### Task 5: 모델 패키지 로더

**Files:**
- Create: `src/objectcount/models/package.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `ModelPackage` frozen dataclass — 필드 `root: Path`, `product_id: str`, `product_name: str`, `model_version: str`, `input_size: tuple[int, int]`, `classes: list[str]`, `active_class_ids: list[int] | None`, `area_prior: float | None`; 속성 `model_path: Path` (= `root / "model.onnx"`)
  - `load_package(path: str | Path) -> ModelPackage` — `manifest.json` 검증 후 로드, 문제 시 `ModelPackageError` (메시지에 원인 명시)
  - `ModelPackageError(Exception)`

모델 패키지 디렉토리 구조 (스펙 §4-④):
```
<package>/
  manifest.json
  model.onnx
```
`manifest.json` 필수 필드: `schema_version`(=1), `product_id`, `product_name`, `model_version`, `input_size`([w, h]), `classes`(list[str]). 선택 필드: `active_class_ids`, `area_prior`, `metrics`, `created_at`.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_package.py`:
```python
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
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_package.py -q`
Expected: FAIL — `ImportError`

- [ ] **Step 3: 구현**

`src/objectcount/models/package.py`:
```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv\Scripts\python -m pytest tests/test_package.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/objectcount/models/package.py tests/test_package.py
git commit -m "feat: 모델 패키지 로더 — manifest 검증"
```

---

### Task 6: OnnxDetector

**Files:**
- Create: `src/objectcount/detection/detector.py`
- Create: `tests/conftest.py`
- Test: `tests/test_detector.py`

**Interfaces:**
- Consumes: `letterbox`/`LetterboxMeta` (Task 3), `decode_yolo`/`nms` (Task 4), `ModelPackage`/`load_package` (Task 5), `Detection` (Task 2)
- Produces:
  - `select_providers(available: list[str] | None = None) -> list[str]` — 우선순위 CUDA → DirectML → CPU 교집합, 전무하면 `["CPUExecutionProvider"]`
  - `OnnxDetector(package: ModelPackage, conf_threshold: float = 0.25, iou_threshold: float = 0.45, providers: list[str] | None = None)` — 메서드 `detect(image_bgr: np.ndarray) -> list[Detection]`
  - conftest 픽스처 `tiny_package` — 상수 출력 ONNX가 담긴 완전한 모델 패키지(다른 태스크의 통합 테스트가 사용)

- [ ] **Step 1: 가짜 ONNX 빌더 + 픽스처 작성 (conftest)**

`tests/conftest.py`:
```python
"""테스트 공용 픽스처 — 입력과 무관하게 상수 출력을 내는 초소형 ONNX 감지 모델.

출력 (1, 6, 1): [cx=32, cy=32, w=16, h=16, score_can=0.9, score_box=0.1]
64x64 입력 기준 → 정규화 (0.375, 0.375, 0.625, 0.625)의 can 1개.
"""
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

RAW_OUTPUT = np.array([[[32.0], [32.0], [16.0], [16.0], [0.9], [0.1]]],
                      dtype=np.float32)  # (1, 6, 1)

MANIFEST = {
    "schema_version": 1,
    "product_id": "tiny",
    "product_name": "테스트 상수 모델",
    "model_version": "test-1",
    "input_size": [64, 64],
    "classes": ["can", "box"],
}


def build_constant_onnx(path: Path, raw_output: np.ndarray) -> None:
    """output = const + 0 * ReduceSum(input) — 입력을 소비하되 항상 상수를 낸다."""
    inp = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 64, 64])
    out = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, list(raw_output.shape))
    const = helper.make_tensor("const_val", TensorProto.FLOAT,
                               raw_output.shape,
                               raw_output.flatten().tolist())
    zero = helper.make_tensor("zero_val", TensorProto.FLOAT, [], [0.0])
    nodes = [
        helper.make_node("Constant", [], ["const_out"], value=const),
        helper.make_node("Constant", [], ["zero"], value=zero),
        helper.make_node("ReduceSum", ["images"], ["input_sum"], keepdims=0),
        helper.make_node("Mul", ["input_sum", "zero"], ["zeroed"]),
        helper.make_node("Add", ["const_out", "zeroed"], ["output0"]),
    ]
    graph = helper.make_graph(nodes, "constant-detector", [inp], [out])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture
def tiny_package(tmp_path: Path) -> Path:
    """상수 출력 ONNX가 담긴 완전한 모델 패키지 디렉토리를 만든다."""
    (tmp_path / "manifest.json").write_text(
        json.dumps(MANIFEST), encoding="utf-8")
    build_constant_onnx(tmp_path / "model.onnx", RAW_OUTPUT)
    return tmp_path
```

- [ ] **Step 2: 실패하는 테스트 작성**

`tests/test_detector.py`:
```python
import numpy as np
import pytest

from objectcount.detection.detector import OnnxDetector, select_providers
from objectcount.models.package import load_package


def test_select_providers_prefers_cuda_then_dml():
    available = ["CPUExecutionProvider", "DmlExecutionProvider",
                 "CUDAExecutionProvider"]
    assert select_providers(available) == [
        "CUDAExecutionProvider", "DmlExecutionProvider",
        "CPUExecutionProvider"]


def test_select_providers_falls_back_to_cpu():
    assert select_providers(["FooProvider"]) == ["CPUExecutionProvider"]


def test_detect_returns_expected_constant_box(tiny_package):
    detector = OnnxDetector(load_package(tiny_package))
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    dets = detector.detect(image)
    assert len(dets) == 1
    d = dets[0]
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx(
        (0.375, 0.375, 0.625, 0.625))
    assert d.class_name == "can" and d.score == pytest.approx(0.9, abs=1e-6)


def test_detect_respects_conf_threshold(tiny_package):
    detector = OnnxDetector(load_package(tiny_package), conf_threshold=0.95)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    assert detector.detect(image) == []
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_detector.py -q`
Expected: FAIL — `ImportError`

- [ ] **Step 4: 구현**

`src/objectcount/detection/detector.py`:
```python
"""ONNX 감지기 — 전처리·추론·후처리 조립. 실행 프로바이더 자동 선택 (스펙 §4-④)."""
from __future__ import annotations

import numpy as np
import onnxruntime as ort

from ..models.package import ModelPackage
from .postprocess import decode_yolo, nms
from .preprocess import letterbox
from .types import Detection

_PROVIDER_PRIORITY = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)


def select_providers(available: list[str] | None = None) -> list[str]:
    if available is None:
        available = ort.get_available_providers()
    chosen = [p for p in _PROVIDER_PRIORITY if p in available]
    return chosen or ["CPUExecutionProvider"]


class OnnxDetector:
    def __init__(
        self,
        package: ModelPackage,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        providers: list[str] | None = None,
    ) -> None:
        self.package = package
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._session = ort.InferenceSession(
            str(package.model_path),
            providers=providers if providers is not None else select_providers(),
        )
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        tensor, meta = letterbox(image_bgr, self.package.input_size)
        raw = self._session.run(None, {self._input_name: tensor})[0]
        detections = decode_yolo(
            raw, meta, self.package.classes,
            conf_threshold=self.conf_threshold,
            active_class_ids=self.package.active_class_ids,
        )
        return nms(detections, self.iou_threshold)
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `.venv\Scripts\python -m pytest tests/test_detector.py -q`
Expected: `4 passed`

- [ ] **Step 6: 전체 테스트 회귀 확인 후 Commit**

Run: `.venv\Scripts\python -m pytest -q`
Expected: 전체 PASS

```bash
git add src/objectcount/detection/detector.py tests/conftest.py tests/test_detector.py
git commit -m "feat: OnnxDetector — EP 자동 선택, 상수 ONNX 픽스처"
```

---

### Task 7: ROI 폴리곤 필터

**Files:**
- Create: `src/objectcount/detection/roi.py`
- Test: `tests/test_roi.py`

**Interfaces:**
- Consumes: `Detection` (Task 2)
- Produces:
  - `RoiPolygon(points: list[tuple[float, float]])` — 정규화 좌표 폴리곤 (꼭짓점 3개 미만이면 `ValueError`)
  - 메서드 `contains(x: float, y: float) -> bool`, `filter(detections: list[Detection]) -> list[Detection]` (중심점 기준)
  - `parse_roi(text: str) -> RoiPolygon` — `"x1,y1;x2,y2;..."` 문자열 파싱 (CLI용)

참고: 스펙 §3-①의 ROI 크롭(추론 입력 축소)은 성능 최적화라 이 단계에서는 하지 않는다 — 감지 후 중심점 필터만. 크롭은 카운팅 단계에서 프로파일링 후 필요하면 추가.

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_roi.py`:
```python
import pytest

from objectcount.detection.roi import RoiPolygon, parse_roi
from objectcount.detection.types import Detection

SQUARE = RoiPolygon([(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])


def _det(cx, cy, half=0.05):
    return Detection(cx - half, cy - half, cx + half, cy + half,
                     0.9, 0, "can")


def test_contains_inside_and_outside():
    assert SQUARE.contains(0.75, 0.75)
    assert not SQUARE.contains(0.25, 0.25)


def test_filter_keeps_detections_with_center_inside():
    dets = [_det(0.75, 0.75), _det(0.25, 0.25)]
    kept = SQUARE.filter(dets)
    assert kept == [dets[0]]


def test_polygon_requires_three_points():
    with pytest.raises(ValueError):
        RoiPolygon([(0, 0), (1, 1)])


def test_parse_roi_string():
    roi = parse_roi("0.5,0.5;1,0.5;1,1;0.5,1")
    assert roi.contains(0.75, 0.75)
    assert not roi.contains(0.1, 0.1)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_roi.py -q`
Expected: FAIL — `ImportError`

- [ ] **Step 3: 구현**

`src/objectcount/detection/roi.py`:
```python
"""ROI 폴리곤 — 벨트 영역 밖 오탐 차단 (스펙 §3-①). 중심점 기준 필터."""
from __future__ import annotations

from .types import Detection


class RoiPolygon:
    def __init__(self, points: list[tuple[float, float]]) -> None:
        if len(points) < 3:
            raise ValueError("ROI 폴리곤은 꼭짓점이 3개 이상이어야 합니다")
        self.points = [(float(x), float(y)) for x, y in points]

    def contains(self, x: float, y: float) -> bool:
        """ray casting — 점이 폴리곤 내부에 있는지."""
        inside = False
        n = len(self.points)
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_cross:
                    inside = not inside
        return inside

    def filter(self, detections: list[Detection]) -> list[Detection]:
        return [d for d in detections if self.contains(*d.center)]


def parse_roi(text: str) -> RoiPolygon:
    """CLI 인자 "x1,y1;x2,y2;..." 파싱."""
    points = []
    for pair in text.split(";"):
        x, y = pair.split(",")
        points.append((float(x), float(y)))
    return RoiPolygon(points)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv\Scripts\python -m pytest tests/test_roi.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/objectcount/detection/roi.py tests/test_roi.py
git commit -m "feat: ROI 폴리곤 필터"
```

---

### Task 8: 프레임 소스

**Files:**
- Create: `src/objectcount/frames/sources.py`
- Test: `tests/test_sources.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `FrameRecord(source_id: str, frame_index: int, image: np.ndarray)` frozen dataclass (`eq=False`)
  - `iter_frames(path: str | Path) -> Iterator[FrameRecord]` — 단일 이미지 파일, 이미지 폴더(이름순), 영상 파일 자동 판별. 지원하지 않는 경로는 `ValueError`
  - 향후 폰 스트림 소스는 같은 `Iterator[FrameRecord]` 시그니처로 추가된다 — 소비자(CLI·카운팅)는 소스 종류를 모른다

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_sources.py`:
```python
from pathlib import Path

import cv2
import numpy as np
import pytest

from objectcount.frames.sources import FrameRecord, iter_frames


def write_image(path: Path, value: int) -> None:
    cv2.imwrite(str(path), np.full((32, 32, 3), value, dtype=np.uint8))


def test_single_image(tmp_path):
    write_image(tmp_path / "a.png", 10)
    records = list(iter_frames(tmp_path / "a.png"))
    assert len(records) == 1
    assert records[0].frame_index == 0
    assert records[0].source_id == "a.png"
    assert records[0].image.shape == (32, 32, 3)


def test_folder_sorted_by_name(tmp_path):
    write_image(tmp_path / "b.png", 20)
    write_image(tmp_path / "a.png", 10)
    (tmp_path / "notes.txt").write_text("무시되어야 함")
    records = list(iter_frames(tmp_path))
    assert [r.source_id for r in records] == ["a.png", "b.png"]
    assert [r.frame_index for r in records] == [0, 1]


def test_video_frames(tmp_path):
    video_path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video_path),
                             cv2.VideoWriter_fourcc(*"MJPG"),
                             5.0, (32, 32))
    for i in range(5):
        writer.write(np.full((32, 32, 3), i * 40, dtype=np.uint8))
    writer.release()

    records = list(iter_frames(video_path))
    assert len(records) == 5
    assert [r.frame_index for r in records] == [0, 1, 2, 3, 4]
    assert all(r.source_id == "clip.avi" for r in records)


def test_unsupported_path_raises(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("x")
    with pytest.raises(ValueError):
        list(iter_frames(target))
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_sources.py -q`
Expected: FAIL — `ImportError`

- [ ] **Step 3: 구현**

`src/objectcount/frames/sources.py`:
```python
"""프레임 소스 — 입력원 무관 원칙의 경계. 모든 소비자는 FrameRecord 이터레이터만 본다."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}


@dataclass(frozen=True, eq=False)
class FrameRecord:
    source_id: str
    frame_index: int
    image: np.ndarray


def iter_frames(path: str | Path) -> Iterator[FrameRecord]:
    path = Path(path)
    if path.is_dir():
        yield from _iter_folder(path)
    elif path.suffix.lower() in IMAGE_EXTS:
        yield _read_image(path, frame_index=0)
    elif path.suffix.lower() in VIDEO_EXTS:
        yield from _iter_video(path)
    else:
        raise ValueError(f"지원하지 않는 입력: {path}")


def _read_image(path: Path, frame_index: int) -> FrameRecord:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return FrameRecord(source_id=path.name, frame_index=frame_index, image=image)


def _iter_folder(folder: Path) -> Iterator[FrameRecord]:
    images = sorted(p for p in folder.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS)
    for index, path in enumerate(images):
        yield _read_image(path, frame_index=index)


def _iter_video(path: Path) -> Iterator[FrameRecord]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {path}")
    try:
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            yield FrameRecord(source_id=path.name, frame_index=index,
                              image=image)
            index += 1
    finally:
        capture.release()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv\Scripts\python -m pytest tests/test_sources.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/objectcount/frames/sources.py tests/test_sources.py
git commit -m "feat: 프레임 소스 — 이미지/폴더/영상 이터레이터"
```

---

### Task 9: 시각화 + detect CLI

**Files:**
- Create: `src/objectcount/viz.py`
- Create: `src/objectcount/cli/detect.py`
- Test: `tests/test_cli_detect.py`

**Interfaces:**
- Consumes: `OnnxDetector`/`select_providers` (Task 6), `load_package` (Task 5), `iter_frames` (Task 8), `RoiPolygon`/`parse_roi` (Task 7), `Detection` (Task 2), conftest `tiny_package` (Task 6)
- Produces:
  - `draw_detections(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray` — 복사본에 박스+라벨
  - `main(argv: list[str] | None = None) -> int` — `objectcount-detect --package P --input I --out O [--conf 0.25] [--roi "x,y;..."]`
  - 출력 계약: `O/detections.jsonl` (프레임당 1줄: `{"source", "frame", "detections": [{"x1","y1","x2","y2","score","class_id","class_name"}]}`) + `O/annotated/<source>_<frame>.jpg`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_cli_detect.py`:
```python
import json

import cv2
import numpy as np

from objectcount.cli.detect import main


def test_detect_cli_end_to_end(tiny_package, tmp_path):
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))
    out_dir = tmp_path / "out"

    exit_code = main(["--package", str(tiny_package),
                      "--input", str(image_path),
                      "--out", str(out_dir)])

    assert exit_code == 0
    lines = (out_dir / "detections.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["source"] == "frame.png" and record["frame"] == 0
    assert len(record["detections"]) == 1
    assert record["detections"][0]["class_name"] == "can"
    assert (out_dir / "annotated" / "frame.png_0.jpg").is_file()


def test_detect_cli_roi_filters_all(tiny_package, tmp_path):
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))
    out_dir = tmp_path / "out"

    # 상수 모델의 박스 중심은 (0.5, 0.5) — 왼쪽 위 사분면 ROI로 걸러낸다
    main(["--package", str(tiny_package), "--input", str(image_path),
          "--out", str(out_dir), "--roi", "0,0;0.4,0;0.4,0.4;0,0.4"])

    record = json.loads(
        (out_dir / "detections.jsonl").read_text(encoding="utf-8"))
    assert record["detections"] == []
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_cli_detect.py -q`
Expected: FAIL — `ImportError`

- [ ] **Step 3: 시각화 구현**

`src/objectcount/viz.py`:
```python
"""감지 결과 시각화."""
from __future__ import annotations

import cv2
import numpy as np

from .detection.types import Detection

BOX_COLOR = (0, 200, 0)      # BGR
TEXT_COLOR = (0, 0, 0)


def draw_detections(image_bgr: np.ndarray,
                    detections: list[Detection]) -> np.ndarray:
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det.to_pixels(width, height)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{det.class_name} {det.score:.2f}"
        cv2.rectangle(canvas, (x1, max(0, y1 - 18)),
                      (x1 + 8 * len(label), y1), BOX_COLOR, -1)
        cv2.putText(canvas, label, (x1 + 2, max(10, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    return canvas
```

- [ ] **Step 4: CLI 구현**

`src/objectcount/cli/detect.py`:
```python
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
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `.venv\Scripts\python -m pytest tests/test_cli_detect.py -q`
Expected: `2 passed`

- [ ] **Step 6: 전체 테스트 회귀 확인 후 Commit**

Run: `.venv\Scripts\python -m pytest -q`
Expected: 전체 PASS

```bash
git add src/objectcount/viz.py src/objectcount/cli/detect.py tests/test_cli_detect.py
git commit -m "feat: detect CLI + 시각화 — JSONL·주석 이미지 출력"
```

---

### Task 10: eval CLI (계수 하네스 시드)

**Files:**
- Create: `src/objectcount/cli/evaluate.py`
- Test: `tests/test_cli_evaluate.py`

**Interfaces:**
- Consumes: `OnnxDetector` (Task 6), `load_package` (Task 5), `iter_frames` (Task 8), conftest `tiny_package` (Task 6)
- Produces:
  - `evaluate_dataset(package_path: Path, dataset_dir: Path, conf: float = 0.25) -> dict` — 반환 `{"per_image": {name: {"expected": int, "detected": int}}, "matched": int, "total": int}`
  - `main(argv: list[str] | None = None) -> int` — `objectcount-eval --package P --dataset D [--conf 0.25]`, 항상 exit 0 (하네스는 보고 도구다)
  - 데이터셋 계약: 디렉토리에 이미지들 + `expected_counts.json` (`{"이미지파일명": 기대개수}`) — 스펙 §8 리플레이 하네스의 감지 단계 시드. 추적·카운팅이 생기면 영상+정답 카운트로 확장된다

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_cli_evaluate.py`:
```python
import json

import cv2
import numpy as np

from objectcount.cli.evaluate import evaluate_dataset, main


def make_dataset(tmp_path, expected):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name in expected:
        cv2.imwrite(str(dataset / name),
                    np.zeros((64, 64, 3), dtype=np.uint8))
    (dataset / "expected_counts.json").write_text(
        json.dumps(expected), encoding="utf-8")
    return dataset


def test_evaluate_reports_match(tiny_package, tmp_path):
    # 상수 모델은 이미지당 정확히 1건 감지한다
    dataset = make_dataset(tmp_path, {"a.png": 1, "b.png": 2})
    report = evaluate_dataset(tiny_package, dataset)
    assert report["per_image"]["a.png"] == {"expected": 1, "detected": 1}
    assert report["per_image"]["b.png"] == {"expected": 2, "detected": 1}
    assert report["matched"] == 1 and report["total"] == 2


def test_main_prints_summary_and_exits_zero(tiny_package, tmp_path, capsys):
    dataset = make_dataset(tmp_path, {"a.png": 1})
    exit_code = main(["--package", str(tiny_package),
                      "--dataset", str(dataset)])
    assert exit_code == 0
    assert "1/1" in capsys.readouterr().out
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `.venv\Scripts\python -m pytest tests/test_cli_evaluate.py -q`
Expected: FAIL — `ImportError`

- [ ] **Step 3: 구현**

`src/objectcount/cli/evaluate.py`:
```python
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `.venv\Scripts\python -m pytest tests/test_cli_evaluate.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/objectcount/cli/evaluate.py tests/test_cli_evaluate.py
git commit -m "feat: eval CLI — 이미지별 기대/감지 개수 대조 (하네스 시드)"
```

---

### Task 11: 사전학습 COCO 데모 패키지 도구 (수동 검증)

**Files:**
- Create: `tools/export_pretrained.py`

**Interfaces:**
- Consumes: 모델 패키지 계약 (Task 5의 manifest 스키마)
- Produces: `models/packages/coco-demo/` — 사전학습 YOLO11n(COCO 80클래스)의 ONNX 모델 패키지. 커스텀 품종 학습 전에 일상 물체(병·컵·과일 등)로 전체 파이프라인을 실물 검증하는 용도. `.gitignore`에 의해 커밋되지 않음(재생성 가능)

- [ ] **Step 1: 도구 스크립트 작성**

`tools/export_pretrained.py`:
```python
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
```

- [ ] **Step 2: tools 의존성 설치 후 패키지 생성**

Run:
```powershell
.venv\Scripts\python -m pip install -e ".[tools]"
.venv\Scripts\python tools\export_pretrained.py
```
Expected: `모델 패키지 생성 완료: models\packages\coco-demo` (가중치 다운로드 포함 수 분)

- [ ] **Step 3: 실사진으로 수동 검증**

아무 사진(병·컵·노트북 등 일상 물체가 찍힌 것)을 `f:\Project\ObjectCount\sample.jpg`로 두고:
```powershell
.venv\Scripts\objectcount-detect --package models/packages/coco-demo --input sample.jpg --out out/demo
```
Expected: `1프레임에서 N건 감지 → out\demo` (N ≥ 1), `out/demo/annotated/` 이미지에 박스가 그려져 있고 라벨이 실제 물체와 일치.

- [ ] **Step 4: 전체 테스트 회귀 확인 후 Commit**

Run: `.venv\Scripts\python -m pytest -q`
Expected: 전체 PASS (도구는 pytest 무관)

```bash
git add tools/export_pretrained.py
git commit -m "feat: 사전학습 COCO 데모 패키지 내보내기 도구"
```

---

## Self-Review 결과

- **스펙 커버리지**: Phase 1a 범위(감지 모듈 ✅ Task 2-6, ONNX 모델 교체 구조 ✅ Task 5-6, 모델 패키지 포맷 ✅ Task 5·11, 시각화 CLI ✅ Task 9, 테스트/하네스 기초 ✅ Task 10, 입력원 무관 ✅ Task 8). 추적·카운팅·캡처 앱·대시보드·등록 파이프라인은 의도적 범위 밖(Phase 1b).
- **EP 우선순위**: 스펙 §4-④ 그대로 CUDA → DirectML → CPU (Task 6).
- **타입 일관성**: `Detection` 필드명(x1/y1/x2/y2/score/class_id/class_name)이 Task 2 정의 → Task 4 생성 → Task 9 `asdict` 직렬화까지 동일. `LetterboxMeta`(scale/pad_x/pad_y/src_width/src_height) Task 3 정의 → Task 4 소비 동일. `iter_frames`→`FrameRecord`(source_id/frame_index/image) Task 8 정의 → Task 9·10 소비 동일. conftest `tiny_package`는 Task 6에서 정의되고 Task 9·10이 재사용.
