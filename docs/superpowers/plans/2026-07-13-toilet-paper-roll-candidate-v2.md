# Toilet Paper Roll Candidate v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a licensed web-data and target-phone-data pipeline, train `candidate-v2`, and decide it against disjoint phone sessions and hard negatives.

**Architecture:** Keep acquisition, annotation review, dataset assembly, training, and evaluation as separate resumable stages. Checked-in code and source configuration reproduce generated artifacts under `out/training/toilet-paper-roll-v2/`; downloaded images, labels, weights, and ONNX files stay outside Git. Reuse the existing dataset validator, AMD/ROCm trainer, candidate package format, and ONNX evaluator.

**Tech Stack:** Python 3.11+, standard-library `urllib`, OpenCV, NumPy, Ultralytics YOLO11n, PyTorch ROCm 7.2 under WSL2, ONNX, onnxruntime, pytest.

## Global Constraints

- The only class is exactly `toilet_paper_roll`, class ID `0`.
- Every distinguishable unpackaged roll is one Product Instance and receives one tight box.
- Wikimedia Commons files require page URL, original URL, author, license, license URL, retrieval time, dimensions, and SHA-256 provenance.
- Arbitrary search-result images and third-party thumbnails are not accepted.
- Whole phone capture sessions belong to one split; temporally adjacent frames never cross splits.
- The live session `20260713-163302-Mozilla-5-0-Linux-Androi` and the prior phone A/B recording are frozen evaluation data.
- Auto-generated boxes are proposals only; every accepted image and box requires visual review.
- Downloaded images, annotations, weights, and generated packages remain outside Git.
- Training uses the existing isolated WSL environment `~/objectcount-train-venv`; runtime `.venv` is not mutated with training dependencies.
- Candidate installation is atomic and does not overwrite candidate-v1 after a failed train or export.
- Promotion requires target-domain mAP50 >= 0.90, Product Instance recall >= 0.90, zero curated hard-negative false positives, and correct detections on at least 90% of clearly visible frozen-phone frames.

## File Structure

- `training/products/toilet-paper-roll/commons.json`: immutable Commons category, limits, accepted licenses, and acquisition policy.
- `tools/training/acquire_commons.py`: MediaWiki pagination, metadata validation, bounded download, hashing, and acquisition manifest.
- `tools/training/sample_capture.py`: session-aware temporal sampling without crossing split boundaries.
- `tools/training/build_candidate_v2.py`: merge reviewed web, phone, and negative YOLO sources into one validated dataset.
- `training/products/toilet-paper-roll/v2-splits.json`: source/session-to-split assignment and frozen-session declarations.
- `training/products/toilet-paper-roll/train-v2.yaml`: fixed candidate-v2 training parameters.
- `tests/test_commons_acquisition.py`: mocked API, license, retry, size, and manifest tests.
- `tests/test_capture_sampling.py`: deterministic sampling and frozen-session exclusion tests.
- `tests/test_candidate_v2_dataset.py`: merge, class, empty-label, duplicate, and split tests.
- `reports/toilet-paper-roll/candidate-v2.md`: generated-result interpretation and promotion decision.

---

### Task 1: Licensed Wikimedia Commons Acquisition

**Files:**
- Create: `training/products/toilet-paper-roll/commons.json`
- Create: `tools/training/acquire_commons.py`
- Test: `tests/test_commons_acquisition.py`

**Interfaces:**
- Produces: `CommonsPolicy.from_json(path: Path) -> CommonsPolicy`
- Produces: `iter_category_files(policy: CommonsPolicy, opener: UrlOpener) -> Iterator[dict]`
- Produces: `acquire_commons(policy_path: Path, output: Path, opener: UrlOpener = urlopen) -> dict`
- Writes: `<output>/manifest.json` and `<output>/images/<sha256-prefix>.<extension>`

- [ ] **Step 1: Write failing policy and pagination tests**

Create fixtures that return two MediaWiki API pages. Assert that `gcmcontinue` is passed into the second request, only raster originals are yielded, and configuration loads the exact category and limits.

```python
def test_iter_category_files_follows_continue(tmp_path):
    opener = FakeOpener([
        api_response({"continue": {"gcmcontinue": "next"}, "query": {
            "pages": [{"pageid": 1, "title": "File:One.jpg",
                       "imageinfo": [valid_imageinfo("https://upload/one.jpg")]}]}}),
        api_response({"query": {"pages": [
            {"pageid": 2, "title": "File:Two.png",
             "imageinfo": [valid_imageinfo("https://upload/two.png")]}
        ]}}),
    ])
    files = list(iter_category_files(make_policy(tmp_path), opener))
    assert [item["page_id"] for item in files] == [1, 2]
    assert "gcmcontinue=next" in opener.urls[1]
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `.venv\Scripts\python -m pytest tests\test_commons_acquisition.py -q`

Expected: FAIL because `tools.training.acquire_commons` does not exist.

- [ ] **Step 3: Implement policy parsing and MediaWiki pagination**

Use a frozen dataclass and URL-encoded API parameters. Request `generator=categorymembers`, namespace `6`, `prop=imageinfo`, and `iiprop=url|size|mime|extmetadata`.

```python
@dataclass(frozen=True)
class CommonsPolicy:
    api_url: str
    category: str
    max_files: int
    max_bytes: int
    min_width: int
    min_height: int
    accepted_mime_types: tuple[str, ...]
    accepted_licenses: tuple[str, ...]
    user_agent: str
```

Set `commons.json` to category `Category:Toilet paper rolls`, maximum 100 accepted files, 20 MiB per file, minimum 480x360, JPEG/PNG/WebP raster MIME types, and accepted Commons Creative Commons/Public Domain license identifiers.

- [ ] **Step 4: Write failing provenance and rejection tests**

Cover missing author, missing license URL, rejected MIME, undersized images, oversized `Content-Length`, hash-based duplicate content, interrupted download, and one retry after a transient HTTP 503. Assert every file receives one manifest row with status `accepted`, `rejected`, or `failed` and a concrete reason.

```python
assert manifest["items"][0]["status"] == "rejected"
assert manifest["items"][0]["reason"] == "missing-license-url"
assert manifest["items"][1]["sha256"] == hashlib.sha256(jpeg).hexdigest()
```

- [ ] **Step 5: Implement bounded atomic downloads and manifest writing**

Download to a temporary sibling file, stream in 64 KiB chunks, abort after `max_bytes`, verify with `cv2.imdecode`, hash bytes, then atomically rename. Store the original title and metadata without rendering HTML. Write `manifest.json` through a temporary file and `os.replace` after each completed item so acquisition can resume safely.

- [ ] **Step 6: Verify Task 1**

Run: `.venv\Scripts\python -m pytest tests\test_commons_acquisition.py -q`

Expected: all acquisition tests PASS without network access.

Run: `.venv\Scripts\python -m pytest -q`

Expected: full suite PASS.

- [ ] **Step 7: Commit Task 1**

```powershell
git add training/products/toilet-paper-roll/commons.json tools/training/acquire_commons.py tests/test_commons_acquisition.py
git commit -m "feat: acquire licensed Commons training images"
```

---

### Task 2: Acquire And Review Web Source Candidates

**Files:**
- Modify: `training/products/toilet-paper-roll/corrections.json`
- Create: `reports/toilet-paper-roll/commons-review.md`
- Reuse: `tools/training/create_contact_sheet.py`

**Interfaces:**
- Consumes: `out/training/toilet-paper-roll-v2/commons/manifest.json`
- Produces: source-relative rejection entries under `commons_exclude_source_images`
- Produces: accepted web images awaiting reviewed YOLO labels under `out/training/toilet-paper-roll-v2/web-reviewed/`

- [ ] **Step 1: Run the real Commons acquisition**

```powershell
.venv\Scripts\python tools\training\acquire_commons.py `
  --config training\products\toilet-paper-roll\commons.json `
  --out out\training\toilet-paper-roll-v2\commons
```

Expected: at most 100 accepted image files plus a manifest in which every category member considered has an explicit status.

- [ ] **Step 2: Validate downloaded bytes and provenance totals**

Run a read-only summary command that fails if any accepted row lacks `sha256`, author, license, license URL, page URL, width, or height.

```powershell
.venv\Scripts\python tools\training\acquire_commons.py `
  --verify out\training\toilet-paper-roll-v2\commons
```

Expected: exit 0 and printed accepted/rejected/failed totals.

- [ ] **Step 3: Generate an unannotated source contact sheet**

```powershell
.venv\Scripts\python tools\training\create_contact_sheet.py `
  --images out\training\toilet-paper-roll-v2\commons\images `
  --out out\training\toilet-paper-roll-v2\commons-contact-sheet
```

Expected: deterministic pages with source IDs and no image omitted.

- [ ] **Step 4: Review every source candidate**

Reject drawings, renders, empty cores, wrapped products without distinguishable rolls, ambiguous paper products, unreadably small objects, and duplicate viewpoints. Record every rejection by stable source image ID in `corrections.json`; do not delete downloaded originals.

- [ ] **Step 5: Produce or import reviewed YOLO labels**

For each accepted image, create a same-stem `.txt` file in `web-reviewed/labels`. Each line must contain exactly `0 center_x center_y width height` with values in `[0,1]`. Empty labels are not valid for web-positive images. Box proposals may be used to accelerate review, but each box and each visible roll must be checked before acceptance.

- [ ] **Step 6: Generate annotated contact sheets and document the audit**

Run the contact-sheet tool against reviewed images and labels. Record accepted images, Product Instance count, rejection reasons, missing-box corrections, group-box corrections, and reviewer date in `commons-review.md`.

- [ ] **Step 7: Commit only reproducibility metadata**

```powershell
git add training/products/toilet-paper-roll/corrections.json reports/toilet-paper-roll/commons-review.md
git commit -m "docs: record Commons dataset review"
```

---

### Task 3: Session-Aware Phone Frame Sampling

**Files:**
- Create: `training/products/toilet-paper-roll/v2-splits.json`
- Create: `tools/training/sample_capture.py`
- Test: `tests/test_capture_sampling.py`

**Interfaces:**
- Produces: `sample_session(session_dir: Path, output: Path, split: str, interval: int, frozen: set[str]) -> SampleReport`
- Writes: `<output>/<split>/images/<session>_<sequence>.jpg`
- Writes: `<output>/sampling-report.json`

- [ ] **Step 1: Write failing deterministic sampling tests**

Build temporary sessions with `manifest.jsonl` and numbered frames. Assert interval sampling is stable, destination names retain session identity, missing frames are reported, and frozen sessions raise `ValueError` when assigned to train or validation.

```python
with pytest.raises(ValueError, match="frozen evaluation session"):
    sample_session(session, output, "train", interval=10,
                   frozen={session.name})
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `.venv\Scripts\python -m pytest tests\test_capture_sampling.py -q`

Expected: FAIL because `sample_capture.py` does not exist.

- [ ] **Step 3: Implement manifest-driven session sampling**

Parse the spool manifest as JSON Lines, select by source sequence rather than directory enumeration order, copy with `shutil.copy2`, and record source session, sequence, timestamps, split, and output relative path. Reject an unknown split and any attempt to place a declared frozen session outside `test`.

- [ ] **Step 4: Add explicit session assignments**

`v2-splits.json` must list each source session exactly once and declare both existing evidence sessions under `frozen_test_sessions`. Fresh captures are added only after their intended split is chosen.

- [ ] **Step 5: Verify Task 3**

Run: `.venv\Scripts\python -m pytest tests\test_capture_sampling.py tests\test_registration_dataset.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```powershell
git add training/products/toilet-paper-roll/v2-splits.json tools/training/sample_capture.py tests/test_capture_sampling.py
git commit -m "feat: sample phone captures by session"
```

---

### Task 4: Candidate v2 Dataset Assembly

**Files:**
- Create: `tools/training/build_candidate_v2.py`
- Test: `tests/test_candidate_v2_dataset.py`
- Modify: `training/products/toilet-paper-roll/corrections.json`

**Interfaces:**
- Produces: `build_candidate_v2(web_root: Path, phone_root: Path, negative_root: Path, splits_path: Path, output: Path) -> DatasetReport`
- Writes: `<output>/images/{train,val,test}`, `<output>/labels/{train,val,test}`, `data.yaml`, `dataset-report.json`, and `provenance.jsonl`

- [ ] **Step 1: Write failing merge tests**

Create tiny web, phone, and negative sources. Assert web and phone positives retain class `0`, negative images receive empty labels, source names cannot collide, and provenance records source kind/session/original path for every output image.

```python
report = build_candidate_v2(web, phone, negatives, splits, output)
assert report.errors == []
assert (output / "labels/train/negative_cup.txt").read_text() == ""
assert all(json.loads(line)["source_kind"] in {"web", "phone", "negative"}
           for line in (output / "provenance.jsonl").read_text().splitlines())
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `.venv\Scripts\python -m pytest tests\test_candidate_v2_dataset.py -q`

Expected: FAIL because `build_candidate_v2` is missing.

- [ ] **Step 3: Implement staging and atomic replacement**

Copy into a temporary sibling dataset, normalize all accepted labels to class `0`, call `validate_yolo_dataset`, write reports, and replace the destination only when there are no errors. Reuse `_replace_directory` semantics from `train_candidate.py`; never mutate source images or labels.

- [ ] **Step 4: Enforce composition and frozen-session rules**

Fail when web positives have an empty label, a phone source session is absent from `v2-splits.json`, a frozen session enters train/validation, a hard negative contains a non-empty label, or exact duplicates cross splits. Print composition shortfalls as warnings rather than manufacturing duplicate samples.

- [ ] **Step 5: Verify Task 4**

Run: `.venv\Scripts\python -m pytest tests\test_candidate_v2_dataset.py tests\test_training_tools.py -q`

Expected: PASS.

Run: `.venv\Scripts\python -m pytest -q`

Expected: full suite PASS.

- [ ] **Step 6: Commit Task 4**

```powershell
git add tools/training/build_candidate_v2.py tests/test_candidate_v2_dataset.py training/products/toilet-paper-roll/corrections.json
git commit -m "feat: assemble candidate v2 dataset"
```

---

### Task 5: AMD Training And Candidate v2 Packaging

**Files:**
- Create: `training/products/toilet-paper-roll/train-v2.yaml`
- Modify: `tests/test_training_tools.py`
- Reuse: `tools/training/setup_amd_wsl.sh`
- Reuse: `tools/training/train_candidate.py`

**Interfaces:**
- Consumes: `out/training/toilet-paper-roll-v2/dataset/data.yaml`
- Produces: `models/experiments/toilet-paper-roll-v2/{model.onnx,manifest.json}`

- [ ] **Step 1: Write a failing model-version configuration test**

Extend the fake trainer test to load `train-v2.yaml`, assert `name == "candidate-v2"`, `imgsz == 640`, deterministic seed `713`, and assert the generated package manifest reports `model_version == "candidate-v2"` without overwriting a sibling candidate-v1 directory.

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `.venv\Scripts\python -m pytest tests\test_training_tools.py -q`

Expected: FAIL because `train-v2.yaml` is absent.

- [ ] **Step 3: Add the fixed candidate-v2 training configuration**

```yaml
model: yolo11n.pt
imgsz: 640
epochs: 100
patience: 25
batch: 8
workers: 4
seed: 713
deterministic: true
device: 0
amp: false
degrees: 20
perspective: 0.0005
hsv_v: 0.35
scale: 0.5
project: out/training/toilet-paper-roll-v2/runs
name: candidate-v2
```

- [ ] **Step 4: Verify configuration tests**

Run: `.venv\Scripts\python -m pytest tests\test_training_tools.py -q`

Expected: PASS.

- [ ] **Step 5: Run AMD/ROCm gates before training**

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/f/Project/ObjectCount && source ~/objectcount-train-venv/bin/activate && python -c 'import torch; assert torch.cuda.is_available(); assert torch.version.hip; print(torch.cuda.get_device_name(0), torch.version.hip)'"
```

Expected: Radeon RX 9070 XT and a non-empty HIP version.

- [ ] **Step 6: Train and export candidate-v2**

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "cd /mnt/f/Project/ObjectCount && source ~/objectcount-train-venv/bin/activate && python tools/training/train_candidate.py --config training/products/toilet-paper-roll/train-v2.yaml --data out/training/toilet-paper-roll-v2/dataset/data.yaml --out models/experiments/toilet-paper-roll-v2"
```

Expected: best checkpoint exported to ONNX and package loading succeeds. On failure, candidate-v1 remains unchanged.

- [ ] **Step 7: Verify ONNX loading on Windows**

```powershell
.venv\Scripts\python -c "from objectcount.models.package import load_package; from objectcount.detection.detector import OnnxDetector; OnnxDetector(load_package('models/experiments/toilet-paper-roll-v2')); print('onnx-ok')"
```

Expected: `onnx-ok`.

- [ ] **Step 8: Commit training configuration and tests**

```powershell
git add training/products/toilet-paper-roll/train-v2.yaml tests/test_training_tools.py
git commit -m "chore: configure toilet paper candidate v2 training"
```

---

### Task 6: Threshold Selection And Frozen Phone Evaluation

**Files:**
- Modify: `tools/training/evaluate_candidate.py`
- Modify: `tests/test_training_tools.py`
- Create: `reports/toilet-paper-roll/candidate-v2.md`
- Create: `reports/toilet-paper-roll/candidate-v2.json`

**Interfaces:**
- Consumes: candidate-v2 package, dataset test split, hard negatives, and frozen phone sessions
- Produces: per-threshold positive recall, negative false positives, frozen-phone frame hit rate, and box-location evidence

- [ ] **Step 1: Write failing frozen-session evaluation tests**

Use a fake detector and labeled frames. Assert a detection counts as a hit only when class is `toilet_paper_roll`, score meets threshold, and IoU with a reviewed roll box is at least 0.50. A high-score box on a neighboring object must remain a false positive.

```python
assert evaluate_frame([wrong_location], truth, threshold=0.03).true_positives == 0
assert evaluate_frame([wrong_location], truth, threshold=0.03).false_positives == 1
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `.venv\Scripts\python -m pytest tests\test_training_tools.py -q`

Expected: FAIL because frozen-session IoU evaluation is absent.

- [ ] **Step 3: Implement location-aware frame evaluation**

Add structured per-frame rows containing frame path, threshold, truth count, detection count, true positives, false positives, false negatives, and matched IoUs. Aggregate by source kind and capture session. Keep the existing conservative threshold rule: zero target-negative false positives first, then maximum target recall.

- [ ] **Step 4: Verify evaluation tests**

Run: `.venv\Scripts\python -m pytest tests\test_training_tools.py tests\test_registration_candidate.py -q`

Expected: PASS.

- [ ] **Step 5: Run candidate-v2 evaluation**

```powershell
.venv\Scripts\python tools\training\evaluate_candidate.py `
  --package models\experiments\toilet-paper-roll-v2 `
  --dataset out\training\toilet-paper-roll-v2\dataset `
  --frozen-splits training\products\toilet-paper-roll\v2-splits.json `
  --out reports\toilet-paper-roll\candidate-v2.json
```

Expected: threshold table and source/session aggregates are written even when gates fail.

- [ ] **Step 6: Inspect representative true positives, misses, and false positives**

Generate annotated evidence sheets for every frozen-phone miss and false positive plus a deterministic sample of true positives. Confirm boxes overlap physical rolls rather than nearby white objects.

- [ ] **Step 7: Write the candidate-v2 decision report**

Record dataset composition, AMD environment, best epoch, metrics, selected threshold, hard-negative results, frozen phone hit rate, ONNX parity, failed gates, and a clear `PROMOTE` or `FAIL - keep inactive` decision. Do not claim the original cup incident fixed unless its exact original frame was evaluated.

- [ ] **Step 8: Run full verification and commit the report**

Run: `.venv\Scripts\python -m pytest -q`

Expected: full suite PASS.

```powershell
git add tools/training/evaluate_candidate.py tests/test_training_tools.py reports/toilet-paper-roll/candidate-v2.md reports/toilet-paper-roll/candidate-v2.json
git commit -m "docs: evaluate toilet paper candidate v2"
```

---

### Task 7: Real-Phone Acceptance Run

**Files:**
- Create: `reports/toilet-paper-roll/candidate-v2-phone-test.md`
- Generated only: `out/phone-candidate-v2/`

**Interfaces:**
- Consumes: selected confidence from `candidate-v2.json`
- Produces: lossless phone capture, live detections, and an A/B comparison against candidate-v1 and COCO

- [ ] **Step 1: Start the capture app with candidate-v2 and selected confidence**

Use the existing capture server with the selected threshold. Do not set `startup_warning`, because the current live worker treats a non-null warning as a reason to bypass inference.

- [ ] **Step 2: Capture the acceptance sequence**

Record isolated roll, cup then roll, nearby desk objects, partial roll, moving roll, and two touching rolls. Keep the source spool lossless and confirm `gaps == 0`.

- [ ] **Step 3: Run offline A/B inference on the captured frames**

Evaluate candidate-v1, candidate-v2, and COCO using fixed thresholds. Report correct roll overlaps separately from class text so a wrong-location one-class detection cannot be credited.

- [ ] **Step 4: Write and commit the phone acceptance report**

Include device, sessions, frames, receive/detect FPS, gaps, per-scene hits, false positives, misses, and the final promotion decision.

```powershell
git add reports/toilet-paper-roll/candidate-v2-phone-test.md
git commit -m "docs: report candidate v2 phone acceptance test"
```
