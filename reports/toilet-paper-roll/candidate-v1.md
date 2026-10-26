# Toilet Paper Roll Candidate v1

Date: 2026-07-13

Decision: **FAIL - keep as Product Candidate; do not activate**

## Training

- Training code commit: `fc12385`
- Environment: Ubuntu 22.04 on WSL2
- GPU: AMD Radeon RX 9070 XT
- PyTorch: `2.10.0+rocm7.2.0.gitb6ee5fde`
- Ultralytics: `8.3.169`
- Input: 640 x 640, batch 8, AMP disabled, seed 713
- Planned epochs: 60
- Completed epochs: 22 (early stopping, patience 15)
- Best epoch: 7
- Successful rerun duration: 1 minute 17 seconds

The first run trained successfully but ONNX export failed because
`onnxscript` was absent. No candidate package was installed by that failed
run. Commit `915a692` pins the full export stack in the AMD setup script and
requests opset 18, matching PyTorch 2.10's actual exporter output.

## Package

- Path: `models/experiments/toilet-paper-roll`
- Status: `candidate`
- Class: `toilet_paper_roll` (class ID 0)
- ONNX opset: 18
- ONNX size: 10,494,657 bytes
- ONNX SHA-256:
  `30F480186B22A38F11132F3589FB1F80E8A0F47C605D2F4FE78874F894B1F6CE`
- Original incident status: `unverified`

`load_package` and ONNX Runtime 1.27.0 CPU loading passed. The package exposes
only `toilet_paper_roll`. A held-out image processed through
`OnnxDetector` at confidence 0.25 produced no detections, consistent with the
low confidence distribution measured below.

## Metrics

Best-checkpoint validation on 11 validation images / 11 instances:

| Metric | Result | Gate |
| --- | ---: | ---: |
| Precision | 0.0030 | report only |
| Recall | 0.9091 | report only |
| mAP50 | 0.6145 | >= 0.90 |
| mAP50-95 | 0.2776 | report only |

Candidate evaluation used six held-out positive images and all 29 reviewed
confusion backgrounds. The full threshold table is in
`reports/toilet-paper-roll/candidate-v1.json`.

| Threshold | Positive-image recall | False positives |
| ---: | ---: | ---: |
| 0.001 | 0.6667 | 19,338 |
| 0.005 | 0.5000 | 3,046 |
| 0.010 | 0.5000 | 2,748 |
| 0.020 | 0.3333 | 9 |
| 0.030 | 0.3333 | 4 |
| 0.040 | 0.0000 | 3 |
| 0.100 | 0.0000 | 0 |

The selector returns threshold 0.90 because every zero-false-positive row has
zero recall and the tie resolves to the highest threshold. This is not a
usable operating point.

## Gate result

| Gate | Required | Result | Decision |
| --- | ---: | ---: | --- |
| mAP50 | >= 0.90 | 0.6145 | FAIL |
| Positive-image recall | >= 0.90 | 0.0000 at selected threshold | FAIL |
| Confusion false positives | 0 | 0 at selected threshold | PASS, but recall is 0 |

Overall decision: **FAIL**. The model must not replace an active Product model
and must not be used for line counting.

## Independent review

OpenCode Gemma reviewed the metric JSON, this report, and the package manifest
through Orca task `task_7249c6933f74`, dispatch `ctx_ba14c4a4533c`. Its valid
`worker_done` reported no findings in gate arithmetic, class mapping,
Candidate wording, zero-false-positive interpretation, or
`original_incident` state. The review was read-only and changed no files.

## Interpretation

The curated public subset is too small and too different from conveyor
footage. Validation recall appears high only at extremely low confidence,
where confusion backgrounds generate thousands of detections. Raising the
threshold removes the false positives and all useful recall.

The next candidate needs real conveyor-like frames with separate labels for
every physical roll, especially touching rolls, partial entries/exits, motion
blur, and production lighting. The six-image positive test and one genuinely
held-out negative are too small for a production claim; using all 29
backgrounds makes this failure decision more conservative but does not create
an independent confusion benchmark.
