# Toilet Paper Roll Product Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible one-class `toilet_paper_roll` training pipeline, train it on the RX 9070 XT through WSL2 ROCm, export an ONNX product-candidate package, and verify cup-confusion behavior.

**Architecture:** Keep downloaded data, weights, and experiment output outside Git while committing source provenance, validation code, training configuration, and reports. Pure-Python registration utilities validate dataset and package metadata; tool scripts lazily import Ultralytics. Orca dispatches two bounded Gemma review-only tasks, while the main agent owns all mutations and verifies every worker result.

**Tech Stack:** Python 3.11+ runtime, Python/WSL training environment, PyTorch ROCm 7.2, Ultralytics YOLO11n, OpenCV, ONNX, onnxruntime, pytest, Orca orchestration, OpenCode with LM Studio Gemma 4 12B.

## Global Constraints

- Canonical terms are `Product`, `Product Candidate`, `Product Instance`, `Detected Object`, `Track`, and `Count Event` as defined in `CONTEXT.md`.
- Train one class named exactly `toilet_paper_roll`; label every visible unpackaged roll as a separate instance.
- Evaluate at most two licensed public dataset sources; never fall back to unverified search-result scraping.
- Keep all images, `.pt`, `.onnx`, and generated experiment directories out of Git.
- Use the dedicated WSL environment `~/objectcount-train-venv`; never modify `~/lora-venv` or the runtime `.venv`.
- Preserve the ROCm builds of `torch` and `torchvision` when installing Ultralytics.
- Candidate creation and Product Promotion are separate; this plan creates a Product Candidate only.
- The original toilet-paper-as-cup incident remains `unverified` until the exact original image is tested.
- Select confidence by zero false positives on the curated confusion set first, then maximum Product Instance recall.
- OpenCode workers are review-only and complete only through valid Orca `worker_done` provenance.

---

## File Structure

- `src/objectcount/registration/dataset.py`: dataset integrity, duplicate, split, and box validation.
- `src/objectcount/registration/candidate.py`: candidate metadata and confidence-selection rules.
- `tools/training/prepare_dataset.py`: normalize an approved YOLO export into deterministic experiment directories.
- `tools/training/train_candidate.py`: train, validate, export ONNX, and create the candidate package.
- `tools/training/evaluate_candidate.py`: evaluate positives and background negatives through ONNX.
- `tools/training/setup_amd_wsl.sh`: isolated ROCm environment and GPU smoke gates.
- `training/products/toilet-paper-roll/source.json`: approved source provenance and class mapping.
- `training/products/toilet-paper-roll/train.yaml`: checked-in training parameters.
- `tests/test_registration_dataset.py`: dataset validation tests.
- `tests/test_registration_candidate.py`: threshold and manifest tests.
- `tests/test_training_tools.py`: orchestration-free tool contract tests with a fake trainer.
- `reports/toilet-paper-roll/`: checked-in Markdown/JSON summaries only, never images or weights.

### Task 1: Product Candidate Metadata

**Files:**
- Create: `src/objectcount/registration/__init__.py`
- Create: `src/objectcount/registration/candidate.py`
- Modify: `src/objectcount/models/package.py`
- Modify: `tests/test_package.py`
- Test: `tests/test_registration_candidate.py`

**Interfaces:**
- Produces: `select_operating_confidence(rows: list[ThresholdResult]) -> ThresholdResult`
- Produces: optional package fields `candidate_status: str | None` and `default_confidence: float | None`

- [ ] **Step 1: Write failing threshold-selection tests**

```python
from objectcount.registration.candidate import ThresholdResult, select_operating_confidence


def test_selects_highest_recall_among_zero_false_positive_thresholds():
    rows = [
        ThresholdResult(0.25, recall=0.98, false_positives=2),
        ThresholdResult(0.40, recall=0.94, false_positives=0),
        ThresholdResult(0.55, recall=0.87, false_positives=0),
    ]
    assert select_operating_confidence(rows).threshold == 0.40


def test_rejects_when_no_threshold_has_zero_false_positives():
    rows = [ThresholdResult(0.50, recall=0.90, false_positives=1)]
    with pytest.raises(ValueError, match="zero false positives"):
        select_operating_confidence(rows)
```

- [ ] **Step 2: Run the tests and confirm RED**

Run: `.venv\Scripts\python -m pytest tests\test_registration_candidate.py -q`
Expected: FAIL because `objectcount.registration.candidate` does not exist.

- [ ] **Step 3: Implement the immutable result and conservative selector**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    recall: float
    false_positives: int


def select_operating_confidence(rows: list[ThresholdResult]) -> ThresholdResult:
    eligible = [row for row in rows if row.false_positives == 0]
    if not eligible:
        raise ValueError("no threshold has zero false positives")
    return max(eligible, key=lambda row: (row.recall, row.threshold))
```

- [ ] **Step 4: Extend package loading without changing schema version**

Add optional dataclass fields and validate `default_confidence` is a real number in `[0, 1]`. Add tests proving absent fields remain `None`, valid candidate metadata passes through, and booleans/out-of-range values fail.

```python
candidate_status: str | None
default_confidence: float | None
```

- [ ] **Step 5: Run focused and full tests**

Run: `.venv\Scripts\python -m pytest tests\test_package.py tests\test_registration_candidate.py -q`
Expected: PASS.

Run: `.venv\Scripts\python -m pytest -q`
Expected: all existing and new tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/objectcount/registration tests/test_registration_candidate.py src/objectcount/models/package.py tests/test_package.py
git commit -m "feat: add product candidate metadata rules"
```

### Task 2: Dataset Integrity Validator

**Files:**
- Create: `src/objectcount/registration/dataset.py`
- Test: `tests/test_registration_dataset.py`

**Interfaces:**
- Produces: `validate_yolo_dataset(root: Path, class_count: int = 1) -> DatasetReport`
- `DatasetReport` contains split counts, positive instances, background images, exact duplicates, and errors.

- [ ] **Step 1: Write failing tests with tiny generated JPEGs and labels**

Cover valid normalized boxes, invalid class IDs, zero-area boxes, missing label files treated as errors, empty labels treated as background negatives, and identical image hashes across splits.

```python
def test_duplicate_image_across_splits_is_error(tmp_path):
    root = make_dataset(tmp_path)
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    write_sample(root, "train", "a", image, "0 0.5 0.5 0.4 0.4\n")
    write_sample(root, "val", "b", image, "0 0.5 0.5 0.4 0.4\n")
    report = validate_yolo_dataset(root)
    assert any("duplicate" in error for error in report.errors)
```

- [ ] **Step 2: Run and confirm RED**

Run: `.venv\Scripts\python -m pytest tests\test_registration_dataset.py -q`
Expected: FAIL because the validator is missing.

- [ ] **Step 3: Implement structured parsing and SHA-256 duplicate detection**

Use `csv`-style whitespace tokenization, `cv2.imread`, `hashlib.sha256`, and explicit numeric/range checks. Do not repair labels silently. Return all errors so one run exposes the whole dataset state.

- [ ] **Step 4: Add perceptual near-duplicate warnings**

Compute a 64-bit difference hash from an 9x8 grayscale resize. Hamming distance `<= 4` across different splits is a warning requiring human review, while exact SHA-256 overlap is an error.

- [ ] **Step 5: Run tests and commit**

Run: `.venv\Scripts\python -m pytest tests\test_registration_dataset.py -q`
Expected: PASS.

```powershell
git add src/objectcount/registration/dataset.py tests/test_registration_dataset.py
git commit -m "feat: validate registration datasets"
```

### Task 3: Reproducible Source Manifest and Dataset Preparation

**Files:**
- Create: `training/products/toilet-paper-roll/source.json`
- Create: `training/products/toilet-paper-roll/train.yaml`
- Create: `tools/training/prepare_dataset.py`
- Modify: `.gitignore`
- Test: `tests/test_training_tools.py`

**Interfaces:**
- Consumes: an approved Roboflow YOLO export ZIP supplied as `--source-zip` and an approved COCO128 archive supplied as `--negative-zip`.
- Produces: `out/training/toilet-paper-roll/dataset/` and `dataset-report.json`.

- [ ] **Step 1: Record the primary source**

Use the 126-image CC BY 4.0 Roboflow project `testroboflow-mzzw9/toilet-paper-detection-pdgv2` as positive source 1. Record landing URL, license, author, retrieval date, expected classes (`basket`, `toilet-paper`), accepted mapping (`toilet-paper` -> `toilet_paper_roll`), and rejected class (`basket`). Record attribution text verbatim in the manifest. A source ZIP is accepted only after its downloaded metadata agrees with this record.

Use Ultralytics COCO128 as source 2 and the final allowed public source. Record its official URL and COCO image-license metadata. Select only images containing COCO class `cup` for the initial background-negative pool, remove COCO labels in the normalized one-class dataset, and manually reject any image that visibly contains a toilet paper roll. Because the two-source ceiling is then exhausted, failure of the positive source blocks this experiment instead of triggering a third download.

- [ ] **Step 2: Write a failing ZIP-normalization test**

Create temporary positive and negative YOLO ZIPs. Assert the script rewrites accepted positive labels to class `0`, drops only rejected-class boxes, imports selected cup images with empty output labels, preserves background images, and writes `data.yaml` with one class.

- [ ] **Step 3: Implement safe archive normalization**

Reject absolute paths and `..` traversal. Extract into a temporary sibling directory, normalize only after validation, then atomically replace the destination. Never mutate the downloaded ZIP.

- [ ] **Step 4: Add artifact ignores**

```gitignore
.venv-train-amd/
out/training/
models/experiments/
runs/
```

- [ ] **Step 5: Add fixed initial training parameters**

```yaml
model: yolo11n.pt
imgsz: 640
epochs: 60
patience: 15
batch: 8
workers: 4
seed: 713
deterministic: true
device: 0
project: out/training/toilet-paper-roll/runs
name: candidate-v1
```

- [ ] **Step 6: Verify and commit**

Run: `.venv\Scripts\python -m pytest tests\test_training_tools.py -q`
Expected: PASS.

```powershell
git add .gitignore training/products/toilet-paper-roll tools/training/prepare_dataset.py tests/test_training_tools.py
git commit -m "feat: prepare toilet paper registration dataset"
```

### Task 4: Isolated AMD WSL Training Environment

**Files:**
- Create: `tools/training/setup_amd_wsl.sh`
- Modify: `training/products/toilet-paper-roll/source.json` only to record environment evidence after verification

**Interfaces:**
- Produces: `~/objectcount-train-venv/bin/python` with working ROCm PyTorch and Ultralytics.

- [ ] **Step 1: Check WSL liveness without terminating existing workloads**

Run: `wsl -d Ubuntu-22.04 -- bash -lc "uname -a; rocminfo | grep -m1 gfx1201"`
Expected: Ubuntu responds and reports `gfx1201`. If WSL remains unresponsive, inspect active WSL processes and ask before `wsl --shutdown`; do not force termination automatically.

- [ ] **Step 2: Implement idempotent isolated setup**

The script creates `~/objectcount-train-venv`, installs the same ROCm 7.2 torch/torchvision wheel family proven in `F:/Project/out/tools/pixel_lora`, records `pip freeze`, then installs Ultralytics with dependencies constrained so torch packages are not replaced. Abort if torch versions differ after installation.

- [ ] **Step 3: Run tensor forward/backward gate**

Run 20 matrix-multiply/autograd/update steps on `cuda`, print `torch.version.hip`, device name, and peak memory, and exit nonzero unless all checks pass.

- [ ] **Step 4: Run one-epoch Ultralytics smoke gate**

Run YOLO11n on `coco8.yaml`, `epochs=1`, `imgsz=320`, `batch=2`, `device=0`. Expected: training completes on the AMD device and writes `out/training/smoke`.

- [ ] **Step 5: Record evidence and commit the script**

```powershell
git add tools/training/setup_amd_wsl.sh training/products/toilet-paper-roll/source.json
git commit -m "build: add isolated AMD training environment"
```

### Task 5: Acquire, Normalize, and Review the Dataset

**Files:**
- Generated: `out/training/toilet-paper-roll/source.zip`
- Generated: `out/training/toilet-paper-roll/dataset/`
- Generated: `out/training/toilet-paper-roll/dataset-report.json`
- Create: `reports/toilet-paper-roll/dataset-review.md`

**Interfaces:**
- Consumes: Task 3 preparation tool and approved source export.
- Produces: reviewed training dataset accepted or blocked before training.

- [ ] **Step 1: Download source candidate 1 through its official export path**

Use the signed/public Roboflow export URL obtained from the dataset page and the official COCO128 archive URL; do not commit credentials or signed URLs. Hash both ZIPs and record both SHA-256 values in the report.

- [ ] **Step 2: Normalize and run automated validation**

Run: `.venv\Scripts\python tools\training\prepare_dataset.py --source-zip out\training\toilet-paper-roll\source.zip --negative-zip out\training\toilet-paper-roll\coco128.zip --config training\products\toilet-paper-roll\source.json --out out\training\toilet-paper-roll\dataset`
Expected: exit 0 and no dataset errors. If source 1 fails license/quality, repeat once with a documented source candidate 2; otherwise stop blocked.

- [ ] **Step 3: Create a bounded contact sheet for label review**

Render 24 seeded samples with boxes plus 12 background negatives. Do not send the full dataset into an LLM context.

- [ ] **Step 4: Dispatch OpenCode worker A through Orca**

Create a review-only task containing only `source.json`, `dataset-report.json`, the contact sheet paths, and a fixed checklist: single-roll semantics, individual boxes for touching rolls, clipped-instance policy, basket removal, and obvious bad boxes. Use `task-create`, a fresh Gemma terminal with 32768 context, `dispatch --inject`, and wait for valid `worker_done`.

- [ ] **Step 5: Main agent verifies and resolves findings**

Inspect each reported sample directly. Correct source labels only through a deterministic corrections file consumed by `prepare_dataset.py`, rerun validation, and document accepted/rejected findings in `dataset-review.md`.

- [ ] **Step 6: Commit reports and deterministic corrections only**

```powershell
git add reports/toilet-paper-roll/dataset-review.md training/products/toilet-paper-roll
git commit -m "docs: record toilet paper dataset review"
```

### Task 6: Train, Export, Package, and Evaluate

**Files:**
- Create: `tools/training/train_candidate.py`
- Create: `tools/training/evaluate_candidate.py`
- Modify: `tests/test_training_tools.py`
- Generated: `models/experiments/toilet-paper-roll/model.onnx`
- Generated: `models/experiments/toilet-paper-roll/manifest.json`

**Interfaces:**
- `train_candidate(config: Path, data: Path, out: Path, trainer_factory=YOLO) -> Path`
- `evaluate_thresholds(package: Path, positives: Path, negatives: Path, thresholds: list[float]) -> dict`

- [ ] **Step 1: Write failing tests with a fake trainer**

Assert training selects `best.pt`, exports ONNX, writes schema-v1 candidate metadata, refuses to replace an existing package after export failure, and records `original_incident: unverified` when no original image is provided.

- [ ] **Step 2: Implement lazy Ultralytics training and atomic packaging**

Import `ultralytics.YOLO` inside `main`/factory only. Build into `models/experiments/.toilet-paper-roll.tmp`, validate the ONNX package with `load_package`, then rename atomically to the final experiment path.

- [ ] **Step 3: Implement ONNX threshold evaluation**

Run the existing `OnnxDetector` over held-out positive and confusion-negative directories at thresholds `[0.25, 0.30, ..., 0.90]`. Count a positive image as recalled when at least one ground-truth instance has IoU `>= 0.5`; count every negative detection as a false positive. Feed rows to `select_operating_confidence`.

- [ ] **Step 4: Verify tool contracts**

Run: `.venv\Scripts\python -m pytest tests\test_training_tools.py tests\test_registration_candidate.py -q`
Expected: PASS without importing real Ultralytics.

- [ ] **Step 5: Commit**

```powershell
git add tools/training/train_candidate.py tools/training/evaluate_candidate.py tests/test_training_tools.py
git commit -m "feat: train and evaluate product candidates"
```

### Task 7: Execute AMD Training and Candidate Gate

**Files:**
- Generated: `out/training/toilet-paper-roll/runs/`
- Generated: `models/experiments/toilet-paper-roll/`
- Create: `reports/toilet-paper-roll/candidate-v1.json`
- Create: `reports/toilet-paper-roll/candidate-v1.md`

- [ ] **Step 1: Run full AMD training in WSL**

Invoke the repository mounted under `/mnt/c/...` with `~/objectcount-train-venv/bin/python`, fixed Task 3 config, and `device=0`. Capture command, git commit, package versions, GPU name, duration, and best epoch.

- [ ] **Step 2: Export and verify ONNX**

Run `load_package`, instantiate `OnnxDetector`, and process at least one held-out positive. Confirm the only class name is `toilet_paper_roll`.

- [ ] **Step 3: Run candidate gates**

Report precision, recall, mAP50, positive-image recall, confusion-set false positives, selected threshold, and Ultralytics/ONNX mapping agreement. Gate passes only at mAP50 `>= 0.90`, positive-image recall `>= 0.90`, and zero confusion false positives at the selected threshold.

- [ ] **Step 4: Run project regression tests**

Run: `.venv\Scripts\python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Commit reports whether pass or fail**

```powershell
git add reports/toilet-paper-roll/candidate-v1.json reports/toilet-paper-roll/candidate-v1.md
git commit -m "docs: report toilet paper candidate results"
```

### Task 8: Independent Result Review and Final Verification

**Files:**
- Modify: `reports/toilet-paper-roll/candidate-v1.md` only if factual corrections are required

- [ ] **Step 1: Dispatch OpenCode worker B through Orca**

Provide only the metric JSON, Markdown report, package manifest, threshold table, and up to 12 positive/negative annotated result images. Ask for review-only checks of gate arithmetic, class mapping, false-positive claims, candidate-vs-Product wording, and original incident state.

- [ ] **Step 2: Wait for valid lifecycle completion**

Use rolling `orca orchestration check --wait --types worker_done,escalation,decision_gate`. A model-generated prose summary without a valid task/dispatch payload is not completion.

- [ ] **Step 3: Main agent independently verifies every claim**

Rerun the evaluator, inspect confusion outputs, compare `git status`, and ensure generated images/weights remain ignored. Record worker task and dispatch IDs in the report.

- [ ] **Step 4: Final verification commands**

```powershell
.venv\Scripts\python -m pytest -q
git diff --check
git status --short
orca orchestration task-list --json
```

Expected: tests pass; no whitespace errors; only intentional user-owned untracked files remain; both review tasks show completed or the report explicitly records their operational failure.

- [ ] **Step 5: Commit factual review corrections if any**

```powershell
git add reports/toilet-paper-roll/candidate-v1.md
git commit -m "docs: finalize toilet paper candidate review"
```
