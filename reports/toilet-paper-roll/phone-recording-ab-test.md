# Phone Recording A/B Test

Date: 2026-07-13

Decision: **FAIL - the original misclassification is not operationally solved**

## Input

- Capture worktree: `OC-proto`
- Session:
  `out/captures/20260713-103448-Mozilla-5-0-Linux-Androi`
- Source frames: 2,052
- Resolution: 720 x 1280
- Existing model: `coco-demo`, confidence 0.25
- Candidate model: `toilet-paper-roll/candidate-v1`
- Candidate thresholds: 0.03 and 0.10

Both models processed the same decoded JPEG frames through the project's
`OnnxDetector`. The complete local result is under
`out/phone-ab-test/`.

## Full-session result

| Condition | Frames with detections |
| --- | ---: |
| Existing model: `cup` | 281 / 2,052 |
| Candidate: `toilet_paper_roll >= 0.03` | 73 / 2,052 |
| Candidate: `toilet_paper_roll >= 0.10` | 0 / 2,052 |

The Candidate's 73 low-confidence detections include false positives on a red
bottle cap, a plastic bottle, and a wall region. Therefore the low threshold
cannot be treated as a usable fix.

## Toilet-paper scene audit

Frames 1300 through 1600 are a continuous 301-frame scene with one physical
toilet-paper roll visible on the right side of the desk.

| Condition | Result |
| --- | ---: |
| Existing model labels the roll as `cup` | 83 / 301 frames |
| Candidate detections at 0.03 | 27 / 301 frames |
| Candidate correct roll boxes at 0.03 | 26 / 301 frames |
| Candidate false wall box at 0.03 | 1 / 301 frames |
| Candidate detections at 0.10 | 0 / 301 frames |

The existing error is reproduced: every reviewed `cup` box in this scene is
on the toilet-paper roll. The Candidate sometimes changes the label to
`toilet_paper_roll`, but correct frame recall is only 8.6% (26 / 301) at
confidence 0.03. Raising confidence to 0.10 removes all detections.

## Conclusion

The Candidate does not pass the phone recording test. It avoids emitting the
literal `cup` class only because it is a one-class model, while introducing
severe missed detections and low-threshold false positives. This is not an
acceptable fix for tracking or line counting.

The most valuable next training data is already available in this capture:

- frames 1300-1600 as hard positive phone-camera examples;
- bottle and bottle-cap closeups before frame 1300 as hard negatives;
- monitor/wall regions around frame 1422 as an additional hard negative.

These frames must be split by contiguous scene, not randomly by frame, to
prevent nearly identical adjacent frames from leaking across train and test.
