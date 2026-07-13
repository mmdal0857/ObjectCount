# Toilet Paper Roll Dataset Acquisition Review

Date: 2026-07-13

Status: **Blocked before normalization**

## Positive source

- Project: `testroboflow-mzzw9/toilet-paper-detection-pdgv2`
- Version: 3
- License: CC BY 4.0
- Author: TestRoboflow
- Classes shown by the official page: `basket`, `toilet-paper`
- Images: 126 total (train 96, validation 21, test 9)
- Preprocessing: auto-orient
- Augmentations: none

The official Universe project and version pages match the checked-in source
manifest. The version page exposes a Download Dataset action, but selecting it
while signed out opens a `Login or create a free account` dialog.

The official Roboflow CLI was tested in an isolated ignored environment with:

```text
roboflow download -f yolov8 -l <ignored-output> \
  testroboflow-mzzw9/toilet-paper-detection-pdgv2/3
```

It refused the request because no API key was configured. No matching export
ZIP was found under the existing `F:\Project` training history or the user
Downloads directory. Page-image scraping was not used because it would bypass
the approved annotated export and its metadata.

Required input to resume:

```text
out/training/toilet-paper-roll/source.zip
```

This must be the version 3 YOLOv8 export downloaded through the official
Roboflow flow. Credentials and signed URLs must not be committed.

## Confusion-negative source

- Archive: Ultralytics COCO128 official release
- Local path: `out/training/toilet-paper-roll/coco128.zip`
- SHA-256: `61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e`
- Acquisition status: complete

Only images labeled with COCO class 41 (`cup`) will be imported as empty-label
background negatives. Visual rejection of images containing an actual toilet
paper roll remains pending until normalization.

## Review state

Automated dataset validation, contact-sheet review, and OpenCode dataset worker
A were not run because the positive archive is unavailable. Dispatching a
dataset review without the annotated export would create a false completion.

An additional review-only OpenCode Gemma task for the AMD setup and training
tool code was created as Orca task `task_8b229eb0ce7b`, dispatch
`ctx_6808a23c3090`. The worker produced prose but failed to send a valid
`worker_done` after two reminders and an interrupted generation loop; Orca
records that task as failed. No worker edits were accepted. The main agent
independently added and tested rollback protection for candidate package
installation.
