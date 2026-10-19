# Subtask 03b: `label_store.save_frame_label` and sample rows with role

## Goal
Add the in-app entry points on top of the writer (03a): list sampled frames with their frozen role, and save one frame's review + boxes so that snapshot, validate, and class supervision consume it exactly like a CVAT import.

## Context
Spec §7 (review rules, `reviewed_at`, `annotator`), §9; TZ §3 (a frame reviewed after a class was added is a negative for that class). `class_supervision._reviewed_instant` refuses naive datetimes; `labels._decision` keeps unreviewed frames out of training.

## Acceptance Criteria
- [ ] `sample_rows_with_role(workspace, project_id, video_id) -> list[SampleRowWithRole]` joining `samples/<video_id>/sample_manifest.json` rows with `frozen_fields_for`; rows outside every block are returned with `role=None` (shown, never saved).
- [ ] `save_frame_label(workspace, project_id, video_id, source_frame_index, *, review_status, boxes: list[BoxDraft], annotator, visibility=None, note=None, now=None) -> LabelRecord`: resolves split from the **frozen** manifest (`SplitNotFrozenError` otherwise); reads the sampled JPEG `samples/<video_id>/images/<video_id>_<idx>.jpg`, verifies sha256 against the manifest `image_sha256` (`image_hash_mismatch` otherwise); `reviewed_at = now or datetime.now(UTC)` for `reviewed` and `excluded`, `None` for `unreviewed`; `annotator=None` when blank; validates through `labels.validate_frame` with `require_project_catalog`; writes one `FrameWrite` via `upsert_frames(..., batch_status="labelled")` with `coco_subset=""`, `image_attributes={}`, `sidecar_boxes=[]`.
- [ ] Refuses a frame not in the sample manifest (`frame_not_sampled`), a manifest row without `source_pts`/`time_base` (`sample_without_pts`), and a box outside `[0,width]×[0,height]` or non-positive (`invalid_box`, message with frame id).
- [ ] Tests `tests/unit/test_label_store.py` (extended): reviewed sets tz-aware `reviewed_at`; unreviewed keeps boxes with `trainable=False`; reviewed with zero boxes → `reviewed_empty`, `is_negative=True`; `class_supervision` on an in-app frame reviewed after `add_class("shovel")` yields `shovel` in `negatives`, reviewed before → `unsupervised`; `snapshot_dataset` includes the reviewed in-app frame in `train/_annotations.coco.json` and the unreviewed one nowhere; `validate_dataset` counts it; `role=None` row refused.

## Implementation Notes
- Files: `src/excavation_lab/datasets/label_store.py`, `tests/unit/test_label_store.py`.
- `AnnotatedFrame.annotator` has `min_length=1`; never pass `""`.
- Keep `iscrowd=0`; `instance_local_id` optional.
- `source_pts`/`time_base` for the `FrameWrite` come from the sample manifest rows written in 02 (`SampleRow.source_pts`, `SampleRow.time_base`). A manifest row lacking them (legacy sample) is refused with `sample_without_pts` and the message tells the operator to re-run sampling; nothing is derived from `frame_index / fps`. Test: a manifest with `source_pts=None` → `sample_without_pts`.

## Dependencies
03a.

## Verification
```bash
uv run pytest tests/unit/test_label_store.py tests/unit/test_snapshot_supervision.py tests/unit/test_class_supervision.py tests/unit/test_dataset_validate.py -q
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
