# Wikimedia Commons Source Review

Date: 2026-07-13

## Acquisition

- Category: `Category:Toilet paper rolls`
- Manifest rows: 186
- Accepted downloads: 100 (configured cap)
- Policy rejections: 24
- Unretried transient rows after reaching the cap: 62 (`http-error-429`)
- Verification: `--verify` passed for all 100 accepted files, including path, byte count, and SHA-256 checks.

The first live run exposed Wikimedia rate limiting after three downloads. The acquisition tool now uses a contactable project User-Agent, two-second pacing, bounded `429`/`503` backoff, `Retry-After` handling, a rate-limit circuit breaker, and in-place retry of failed manifest rows. The resumed run reached the 100-file cap without modifying previously accepted or rejected rows.

## Visual Audit

- Contact sheets: 3 deterministic pages containing 36, 36, and 28 images.
- Source candidates inspected: 100
- Retained for annotation: 56
- Excluded: 44
- Retained images staged at: `out/training/toilet-paper-roll-v2/web-reviewed/images/`

Exclusions are stored under `commons_exclude_source_images` in `training/products/toilet-paper-roll/corrections.json`. Reasons include packaged-only products, hidden-roll dispensers, paper sheets without a distinguishable roll, distant store/event scenes, duplicate souvenir viewpoints, non-roll cardboard tubes, and histogram overlays.

Holder-mounted, damaged, partial, and multiple-roll scenes were retained when each physical Product Instance remained distinguishable. Retention does not imply label approval: every retained image still requires tight class-0 instance boxes and an annotated contact-sheet review before it may enter training.

## Current Gate

Source acquisition, source-level review, and annotation review are complete.

- Proposal model: `yolov8s-worldv2.pt`, prompted with `toilet paper roll`
- Proposal threshold: confidence `0.05`, IoU `0.50`
- Proposed: 31 of 56 retained images
- Annotation-approved: 10 images, 15 Product Instances
- Annotation-rejected: 46 images
- Reviewed dataset: `out/training/toilet-paper-roll-v2/web-reviewed/`

All 56 retained candidates were inspected, including the 25 images with no proposal. Proposal boxes were accepted only when every visible Product Instance was covered and each box stayed tight to the physical roll body. Images were rejected when proposals were missing, duplicated, grouped, or included unrolled sheets, holders, toilets, artwork, or unrelated objects. Rejections from this gate are stored separately under `commons_exclude_annotation_images`; downloaded Commons originals remain unchanged.

The approved labels contain exactly five YOLO fields with class `0` and normalized coordinates. No empty-label web positives remain in the reviewed dataset.
