# Toilet Paper Roll Candidate v2

Date: 2026-07-13

Decision: **FAIL - keep as Product Candidate; do not activate**

## Training

- Training code commit: `d386ad8`
- Environment: Ubuntu 22.04 on WSL2
- GPU: AMD Radeon RX 9070 XT (ROCm/HIP 7.2)
- Ultralytics: 8.3.169
- Input: 640 x 640, batch 8, AMP disabled, seed 713
- Completed epochs: 100
- Dataset: 95 images (77 train, 11 validation, 7 test), 96 Product Instances
- Dataset composition warning: 0 reviewed phone positives and 29 hard negatives

The candidate adds 10 reviewed Wikimedia Commons images and 15 Product
Instances to the candidate-v1 source set. It still has no labeled phone or
conveyor footage, so the domain gap remains material.

## Package

- Path: `models/experiments/toilet-paper-roll-v2`
- Status: `candidate`
- Class: `toilet_paper_roll` (class ID 0)
- ONNX size: 10,494,660 bytes
- ONNX SHA-256:
  `D7E50B0185CB3FFB0AF46D6072EA689DF28631FA2944C9D39A3543B70B9ED362`
- Original incident status: `unverified`

`load_package` and ONNX Runtime 1.27.0 CPU loading passed. The Windows virtual
environment exposes CPU and Azure execution providers; DirectML is not
installed in this environment.

## Training Metrics

| Metric | Result | Gate |
| --- | ---: | ---: |
| Precision | 0.7691 | report only |
| Recall | 0.9086 | report only |
| mAP50 | 0.8830 | >= 0.90 |
| mAP50-95 | 0.5720 | report only |

The mAP50 gate fails narrowly. The best per-epoch mAP50 observed in
`results.csv` was 0.8943 at epoch 78, which also remains below the gate.

## Held-Out Threshold Gate

The full threshold table is in `reports/toilet-paper-roll/candidate-v2.json`.
It uses six held-out web positives and the one held-out negative image. Images
used for training were not reused as threshold-gate negatives.

| Threshold | Positive-image recall | False positives |
| ---: | ---: | ---: |
| 0.25 | 0.6667 | 0 |
| 0.35 (selected) | 0.6667 | 0 |
| 0.40 | 0.5000 | 0 |
| 0.60 | 0.1667 | 0 |
| 0.80 | 0.0000 | 0 |

The selector chooses 0.35 because it has the highest recall among
zero-false-positive rows and wins the equal-recall tie with the higher
threshold. Six positives and one negative are too small for a production
claim.

## Frozen Phone Evaluation

The frozen session contains 530 frames and shows one physical toilet paper
roll throughout the sampled review. At threshold 0.25, candidate v2 produced
502 boxes:

| Frame outcome | Frames | Rate |
| --- | ---: | ---: |
| No detection | 92 | 17.4% |
| Exactly one detection | 376 | 70.9% |
| Multiple detections | 62 | 11.7% |

The 18-frame review sample had 16 detected frames, two clear false negatives,
and three frames with two boxes. Visual inspection confirmed secondary false
positives on the keyboard/desk area and a white plastic bag.

On the same 530 frames, the COCO baseline emitted 1,316 boxes, including 13
`cup` boxes in 12 frames and 285 `toilet` boxes in 266 frames. Candidate v2
cannot emit either class because its package contains only
`toilet_paper_roll`; it usually labels the roll correctly, but its misses and
duplicate Product Instance detections still violate conservative counting.

## Gate Result

| Gate | Required | Result | Decision |
| --- | ---: | ---: | --- |
| mAP50 | >= 0.90 | 0.8830 | FAIL |
| Held-out positive-image recall | >= 0.90 | 0.6667 at 0.35 | FAIL |
| Held-out confusion false positives | 0 | 0 at 0.35 | PASS, weak sample |
| Frozen phone single detection | stable | 70.9% at 0.25 | FAIL |

Overall decision: **FAIL**. Candidate v2 improves the reported class-confusion
behavior, but it must not replace an active Product model or feed line
counting yet. The next dataset revision needs reviewed phone positives,
keyboard/desk/plastic-bag hard negatives, and conveyor-like multi-roll scenes.

## Review Provenance

Previous Orca/OpenCode review dispatches for this candidate did not produce a
valid terminating `worker_done`. This report therefore does not claim an
independent Gemma approval; its figures were reproduced locally from the
package and frozen inputs.
