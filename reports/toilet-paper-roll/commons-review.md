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

Source acquisition and source-level review are complete. The 56 retained images are not yet training-ready because reviewed YOLO labels have not been produced. Empty labels are not permitted for these web-positive images.
