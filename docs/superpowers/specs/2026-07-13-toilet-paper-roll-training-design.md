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

Dataset rules:

- Map only unpackaged single rolls to `toilet_paper_roll`.
- Exclude wrapped multipacks and ambiguous paper products from positive labels.
- Keep train, validation, and test images disjoint.
- Include cups, kitchen-paper rolls, wrapped packs, and visually similar white
  cylinders as negative validation examples where licensing permits.
- Inspect a representative label sample before training and record rejected or
  corrected samples.
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

These gates validate the pipeline only. The design specification's production
target of mAP50 >= 0.95 remains blocked until target-domain images are added.

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
- Invalid or overlapping splits stop training.
- Training or export failure leaves the previous model package untouched.
- A failed quality gate still produces a report but blocks promotion of the
  package as a successful experiment.
- Downloaded data and model artifacts use deterministic directories so a
  failed run can be inspected and resumed without mixing results.

## Deliverables

- Dataset source manifest and normalized dataset configuration
- Download, validation, training, export, and package scripts or commands
- `toilet_paper_roll` ONNX model package outside Git
- Metrics and confusion-set report
- Orca task and dispatch IDs for both OpenCode review stages
- Explicit list of remaining target-domain data requirements
