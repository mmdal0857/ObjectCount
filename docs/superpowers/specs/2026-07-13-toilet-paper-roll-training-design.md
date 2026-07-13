# Toilet Paper Roll Custom Detection Design

- Date: 2026-07-13
- Status: approved
- Scope: detect one unpackaged toilet paper roll as a single object class

## Goal

Fine-tune a YOLO11n detector for the `toilet_paper_roll` class, export it to
the existing ONNX model package format, and verify that a toilet paper roll is
detected without relying on the COCO `cup` class.

This first iteration validates the registration and training pipeline with
public web data. It does not claim production accuracy for conveyor footage;
that requires images captured with the target phone, angle, lighting, and
background.

## Dataset

Use a public object-detection dataset with bounding-box annotations as the
primary source. Before downloading, record the dataset URL, version, license,
class mapping, and retrieval date. Do not scrape arbitrary search-result
images when a licensed annotated source is available.

Evaluate at most two public dataset sources. If the first source fails the
license or label-quality review, evaluate one replacement. If both sources are
unacceptable, stop the experiment as blocked rather than lowering the gate or
falling back to unverified search-image scraping.

Dataset rules:

- Map only unpackaged single rolls to `toilet_paper_roll`.
- The dataset has one class, but every visible unpackaged roll in an image is
  labeled as a separate instance. Multiple rolls in one image are expected.
- Label touching rolls with separate boxes whenever their individual extent is
  distinguishable, and include identifiable rolls clipped by a frame edge.
- Exclude wrapped multipacks and ambiguous paper products from positive labels.
- Keep train, validation, and test images disjoint.
- Include cups, kitchen-paper rolls, wrapped packs, and visually similar white
  cylinders as negative validation examples where licensing permits.
- Include a disjoint subset of those confusion classes as background-negative
  training images with no positive boxes. Keep separate validation negatives
  and a final confusion test set that is not inspected during tuning.
- Inspect a representative label sample before training and record rejected or
  corrected samples.
- Detect exact file duplicates and near-identical source images before split
  assignment so related samples cannot leak across train, validation, and test.
- Keep downloaded images and generated weights out of Git. Commit source
  manifests, configuration, scripts, and reports needed to reproduce them.

Target size for the first experiment is 150-300 labeled images. If no suitable
licensed dataset reaches that range, use the largest acceptable source and
report the shortfall instead of silently mixing unverified images.

## Training Pipeline

Add a reproducible command-line workflow with separate stages:

1. Download and normalize the selected source into an Ultralytics YOLO dataset.
2. Validate image readability, class IDs, normalized boxes, empty labels, and
   split overlap.
3. Fine-tune `yolo11n.pt` for one class using a checked-in training config.
4. Evaluate the held-out split and export the best checkpoint to ONNX.
5. Build `models/packages/toilet-paper-roll/` with `model.onnx` and the existing
   schema-v1 `manifest.json`.
6. Run the existing detection CLI against held-out positive and negative
   samples and write a machine-readable and human-readable report.

Training uses the existing `tools` optional dependency. Runtime inference stays
ONNX-only and does not introduce a PyTorch or GPU-vendor dependency into the
`objectcount` package.

Training uses a dedicated WSL2 environment at `~/objectcount-train-venv` with
ROCm 7.2 for the Radeon RX 9070 XT. The environment follows the GPU gate and
ROCm wheel-preservation lessons from `F:/Project/out/tools/pixel_lora`, but it
does not modify or reuse that project's `~/lora-venv`. The Windows project
runtime environment also remains unchanged.

Before the product dataset is trained, the dedicated environment must verify
`torch.cuda.is_available()`, confirm a HIP runtime through `torch.version.hip`,
complete a forward/backward tensor gate, and complete a one-epoch Ultralytics
smoke training on a tiny dataset. Installing Ultralytics must not replace the
known-working ROCm builds of `torch` or `torchvision`. CPU training is a
fallback only when the isolated AMD environment or Ultralytics compatibility
check fails, and that fallback must be reported.

## Quality Gates

The first experiment reports precision, recall, mAP50, and a confusion-focused
sample review. The model passes the experimental gate when:

- mAP50 is at least 0.90 on the held-out public-data test split;
- at least 90% of held-out roll images contain a correct detection at the
  selected confidence threshold;
- no cup negative sample is detected as `toilet_paper_roll` in the curated
  confusion set; and
- ONNX inference through `objectcount-detect` produces the same class mapping
  as Ultralytics validation.

Select the operating confidence threshold according to the conservative
counting principle, not mAP alone. First retain only thresholds that produce no
false positive on the curated cup, kitchen-paper, wrapped-pack, and white-
cylinder negative set. From those thresholds, choose the one with the highest
product-instance recall. Low-confidence ambiguous detections remain excluded
from counting and are reserved as future suspect-event inputs. Record the
selected threshold and the selection evidence in the candidate report and
model-package metadata.

These gates validate the pipeline only. The design specification's production
target of mAP50 >= 0.95 remains blocked until target-domain images are added.

The original toilet-paper-as-cup incident is tracked separately from the
public-data candidate gate. Candidate creation may continue without the
original image, but the incident remains `unverified` until inference succeeds
on that exact image. A visually similar web image cannot close the incident.

## Orca Orchestration

The main agent owns source selection, scripts, training, ONNX export, package
creation, and final verification. OpenCode workers use LM Studio
`google/gemma-4-12b` with a 32768-token context and receive bounded,
review-only tasks.

The supervised DAG is:

1. Main agent prepares the normalized dataset and validation report.
2. OpenCode worker A reviews a bounded label sample, source manifest, and data
   validation summary. It edits no files and returns `worker_done`.
3. Main agent resolves actionable findings and runs training/export.
4. OpenCode worker B reviews the metrics, confusion samples, and package
   manifest. It edits no files and returns `worker_done`.
5. Main agent verifies the worker reports against commands, metrics, generated
   artifacts, and Git status before publishing the final result.

Every worker assignment must be created with `orca orchestration task-create`
and delivered with `dispatch --inject`. Completion is accepted only from a
valid `worker_done` payload for the active task and dispatch. If the local model
cannot complete the lifecycle command, the coordinator records the operational
failure separately from the technical review result and does not present the
worker as independently successful.

## Error Handling

- Missing or unclear dataset license stops ingestion of that source.
- At most one replacement public dataset may be attempted. Two rejected
  sources block candidate creation and make target-domain image collection the
  explicit unblock condition.
- Invalid or overlapping splits stop training.
- Training or export failure leaves the previous model package untouched.
- Failure of the isolated AMD training environment does not mutate the runtime
  environment. Record the failed compatibility check before using CPU fallback.
- A failed quality gate still produces a report but blocks promotion of the
  package as a successful experiment.
- A missing original misclassification image does not block candidate creation,
  but the final report must mark the original incident as `unverified`.
- Downloaded data and model artifacts use deterministic directories so a
  failed run can be inspected and resumed without mixing results.

## Deliverables

- Dataset source manifest and normalized dataset configuration
- Download, validation, training, export, and package scripts or commands
- `toilet_paper_roll` ONNX model package outside Git
- Metrics and confusion-set report
- Orca task and dispatch IDs for both OpenCode review stages
- Explicit list of remaining target-domain data requirements
