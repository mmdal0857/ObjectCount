# Toilet Paper Roll Dataset Review

Date: 2026-07-13

Status: **Accepted for a small pilot candidate, not production training**

## Sources

- Positive: Roboflow Universe project
  `testroboflow-mzzw9/toilet-paper-detection-pdgv2`, version 3
- Positive license: CC BY 4.0, author `TestRoboflow`
- Positive archive: `out/training/toilet-paper-roll/source.zip`
- Positive SHA-256:
  `53A36A2ED0218967183530D7E21B1987E16CB2493220E7D6609A08DCFC52453C`
- Confusion negatives: Ultralytics COCO128 images containing class 41
  (`cup`), imported with empty labels
- Negative SHA-256:
  `61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e`

The source export contains 126 images and two classes: `basket` and
`toilet-paper`. Normalization maps only `toilet-paper` to class 0
`toilet_paper_roll`; rejected classes are ignored even when their source
label uses polygon geometry.

## Manual audit

The target contract is one red box per physical, unpackaged toilet-paper-roll
Product Instance. Packages, a box enclosing multiple rolls, printed or
rendered depictions, other rolled materials, and images with obvious missed
instances are not valid positives.

All 102 initially normalized positive images and all 32 initial backgrounds
were reviewed in deterministic contact sheets. The correction manifest
`training/products/toilet-paper-roll/corrections.json` excludes:

- 2 lower-resolution cross-split copies;
- 46 semantically invalid or incompletely labeled positive images;
- 3 background images containing an unlabeled roll.

Mounted physical rolls remain valid positives. They add pose and occlusion
variation while still satisfying the one-instance contract.

After corrections, a second full visual audit found no remaining package,
group-box, or background contamination. The retained set is still web-photo
heavy and unlike the final conveyor viewpoint, so it is suitable only for a
pilot that tests whether the new class can be learned and exported.

## Normalized result

`out/training/toilet-paper-roll/dataset/dataset-report.json` reports:

```text
train images:       67
validation images:  11
test images:         7
positive images:    56
positive instances: 81
background images:  29
exact duplicates:    0
near duplicates:     0
errors:              0
warnings:            0
```

Corrections are source-relative and deterministic. Re-running
`tools/training/prepare_dataset.py` from the two source archives reproduces
the reviewed dataset.

## Orca review provenance

OpenCode Gemma review task `task_c46b26cfaf35`, dispatch
`ctx_a62cc84ddfac`, was assigned as a bounded review-only task. The worker
exceeded its 32,768-token context after listing unrelated output files, then
repeated file-read loops despite two bounded reminders. It did not send
`worker_done` after interruption, so Orca records the task as failed and no
worker result or edits were accepted.

The earlier AMD training-tool review task `task_8b229eb0ce7b`, dispatch
`ctx_6808a23c3090`, also failed to produce a valid `worker_done`. The main
agent independently performed the dataset audits and automated validation.

## Remaining risk

- Only 56 positive images remain, with seven test images. Metrics will have
  high variance and must not be treated as a production quality claim.
- The source is dominated by bathrooms and product photos, not overhead or
  oblique conveyor footage.
- Touching rolls, motion blur, partial exits, and real line lighting remain
  underrepresented.

The next evidence gate is an AMD/ROCm pilot train, ONNX export, threshold
evaluation, and inspection on a real conveyor-like video or image set.
