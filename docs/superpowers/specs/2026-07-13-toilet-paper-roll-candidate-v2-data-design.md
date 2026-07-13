# Toilet Paper Roll Candidate v2 Data Design

- Date: 2026-07-13
- Status: approved
- Scope: improve one-class `toilet_paper_roll` detection with licensed web data and target-domain captures
- Supersedes: the two-public-source ceiling in the candidate-v1 design for this v2 experiment only

## Goal

Produce a second Product Candidate that detects each visible, unpackaged toilet-paper-roll Product Instance in phone footage. Web images provide shape and pose diversity; fresh phone captures provide the target camera, background, lighting, scale, and blur distribution. The candidate is successful only when its boxes cover the rolls in held-out phone sessions rather than nearby objects.

The model remains a one-class detector. Absence of the COCO `cup` label is not evidence of correctness because this package cannot emit that class.

## Data Sources

### Licensed web positives

Crawl the Wikimedia Commons `Category:Toilet paper rolls` through the MediaWiki API. For every downloaded file, retain the Commons page URL, original file URL, author, license name, license URL, retrieval timestamp, SHA-256, and source dimensions. Do not download search-result thumbnails or images from arbitrary third-party pages.

Select 50-100 usable raster images. Exclude drawings, renders, empty cardboard tubes, wrapped multipacks without separately visible rolls, severe label ambiguity, and images too small for a reliable box. Images containing several distinguishable rolls are allowed, and every visible roll must receive a separate box.

### Target-domain positives

Collect fresh USB-phone captures that resemble the intended line. Include:

- top, side, and oblique views;
- near, medium, and small object scales;
- rolls entering and leaving the frame;
- touching and partially occluded rolls;
- motion blur, exposure variation, and realistic line backgrounds.

Select 100-300 non-near-duplicate frames from multiple capture sessions. Assign whole sessions to train, validation, or test before frame extraction. Frames from one session must never cross split boundaries.

### Hard negatives

Collect at least 100 target-domain negatives containing cups, paper-towel rolls, empty cores, white cylindrical containers, books, keyboards, cables, and other nearby objects observed during the live test. Negative images have empty YOLO label files after a reviewer verifies that no valid toilet-paper-roll instance is present.

### Frozen evidence

The following previously inspected sessions remain evaluation-only and cannot be used for training or threshold selection:

- `F:/Project/ObjectCount/out/live-candidate/20260713-163302-Mozilla-5-0-Linux-Androi`
- the prior phone recording used by `reports/toilet-paper-roll/phone-recording-ab-test.md`

## Acquisition And Annotation

Add reproducible tooling with separate stages:

1. Query Wikimedia Commons category members and image metadata with pagination.
2. Filter unsupported media and download originals with bounded size, timeouts, retries, and a descriptive user agent.
3. Record each success, rejection, and failure in a JSON source manifest.
4. Reject exact and perceptual duplicates against candidate-v1 data before split assignment.
5. Generate optional box proposals, but never accept them as ground truth without visual review.
6. Store reviewed annotations as normalized YOLO `xywh`, one class ID `0` per physical roll.
7. Build deterministic contact sheets with boxes and source IDs for full-set review.
8. Normalize the approved web, target-domain, and hard-negative sources into a candidate-v2 dataset.

Annotation review checks every image, not a sample. A box must cover one physical roll tightly. Group boxes, missing visible instances, zero-area boxes, and boxes on surrounding objects block training.

## Split And Leakage Rules

- Web images are grouped by source file and perceptual similarity before splitting.
- Burst frames are grouped by capture session and temporal neighborhood.
- The final phone test sessions are unseen during model selection.
- Dataset validation rejects exact cross-split duplicates and warns on perceptual near-duplicates.
- Web images should not dominate target-domain validation or the final decision.

The initial target composition is 50-100 web positives, 100-300 fresh target-domain positives, and at least 100 target-domain negatives. A shortfall is reported explicitly; it is not hidden with duplicated or aggressively augmented samples.

## Training And Evaluation

Train `candidate-v2` from YOLO11n with the existing AMD/ROCm backend and ONNX export path. Use augmentations that reflect the line: moderate rotation and perspective, exposure variation, downscale, and motion blur. Keep augmentation settings recorded and deterministic where supported.

Evaluate in three layers:

1. Source-held-out metrics for regression visibility.
2. Target-domain validation for threshold selection.
3. Frozen phone sessions for the final decision.

Select confidence conservatively: first eliminate thresholds that produce false positives on target hard negatives, then choose the highest Product Instance recall. Report precision, recall, mAP50, false positives by confusion type, missed instances, and box-location errors.

## Candidate v2 Gates

Candidate v2 passes the experiment only when all conditions hold:

- target-domain mAP50 is at least 0.90;
- Product Instance recall is at least 0.90 at the selected confidence;
- no false positive appears in the curated target hard-negative test;
- at least 90% of frozen phone frames containing a clearly visible roll have a correct roll-overlapping detection;
- detections on surrounding objects do not count as roll hits;
- the ONNX package reproduces the selected class and threshold behavior.

Failure still produces a report and preserves the package as an inactive Product Candidate.

## Error Handling

- Missing or unsupported license metadata rejects that web file.
- Download failures remain recorded and do not silently reduce the requested sample count.
- Corrupt images, unsupported formats, and excessive dimensions are rejected before annotation.
- Annotation uncertainty rejects the image rather than guessing a box.
- Training never starts while dataset validation reports an error.
- Candidate installation remains atomic, preserving candidate-v1 on export or package failure.
- The runtime environment and the existing phone capture server are not modified by data acquisition or training dependencies.

## Deliverables

- Wikimedia Commons acquisition script and tests
- source manifest with attribution and hashes
- reviewed YOLO annotations and contact-sheet audit report
- deterministic candidate-v2 dataset report
- AMD training and ONNX package artifacts outside Git
- web, target-domain, hard-negative, and frozen-session evaluation results
- explicit pass/fail recommendation for Product Promotion
