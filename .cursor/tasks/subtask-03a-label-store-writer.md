# Subtask 03a: `datasets/label_store.py` — per-frame upsert writer shared with CVAT import

## Goal
Extract the label persistence from `coco_import._publish` into `src/excavation_lab/datasets/label_store.py` with upsert-by-frame semantics that touch only the incoming JPEGs, and make `coco_import` delegate to it.

## Context
Spec §7, §10; TZ §3. Today `coco_import._publish` rewrites `annotation_batches/<video_id>__<split>/imported_labels.json` and the whole `images/` directory from the ZIP rows only; frames not in the ZIP would be lost. `load_imported_frames` reads every `imported_labels.json`; `snapshot._reject_duplicates` refuses the same frame in two batches. The in-app labeler (03b) therefore writes into the **same** batch directory and the writer must merge per frame — and on a 600-frame batch a single-frame save must not rewrite 600 JPEGs.

## Acceptance Criteria
- [ ] New module `datasets/label_store.py` with:
  - `batch_directory(workspace, project_id, video_id, split) -> Path` (`annotation_batches/<video_id>__<split>`),
  - `FrameWrite` dataclass: manifest fields (`image_file, source_frame_index, source_pts, time_base, source_time_s, episode_id, split, split_group_id, width, height`), `payload: bytes`, `record: LabelRecord`, `coco_subset: str`, `image_attributes: dict`, `sidecar_boxes: list[dict]`,
  - `upsert_frames(workspace, project_id, video_id, split, writes, *, frozen: FrozenSplit, batch_status: str) -> UpsertResult`,
  - `frozen_fields_for(frozen, source_time_s) -> (SplitName, split_group_id, start_s, end_s) | None` promoted from `cvat_export._frozen_fields`; `cvat_export` imports it from here.
- [ ] `upsert_frames` write protocol: (1) write each **incoming** JPEG to `images/<name>.tmp` then `os.replace` onto `images/<name>`; existing images of other frames are not read, copied, or rewritten; (2) load the existing `imported_labels.json` and `attributes_sidecar.json` (if present), replace entries by `(video_id, source_frame_index)`, keep the rest; (3) write both JSON documents to `*.tmp` and `os.replace` them; (4) upsert the `annotation_batches` row (status `imported` or `labelled`). A per-batch-directory `threading.Lock` (module-level dict keyed by resolved path) serialises concurrent saves within the process.
- [ ] `coco_import._publish` delegates to `upsert_frames`; a CVAT import of `<video>__train` keeps frames previously saved in-app that the ZIP does not contain; a ZIP frame replaces the stored frame per the existing `_merge_stored` rules; the old `images/` wipe is gone.
- [ ] `load_imported_frames`, `ImportedFrame`, `_frame_document`, `_sidecar_frame`, `_load_batch` keep their import paths working (`snapshot.py`, `validate.py`, `catalog._labels_have_name`, `models/train.py::_imported_by_file`); re-export from `label_store` is acceptable.
- [ ] Tests `tests/unit/test_label_store.py`: upsert keeps other frames; a 600-frame batch (tiny generated JPEGs) + upsert of 1 frame → exactly one JPEG mtime changes and the JSON lists 600 frames; CVAT import after an in-app-style upsert keeps the in-app frame; crash simulation (exception between JPEG and JSON write) leaves the previous JSON readable; concurrent upserts from two threads produce a JSON with both frames.
- [ ] `tests/integration/test_coco_roundtrip.py` and `tests/unit/test_cvat_export.py` still green.

## Implementation Notes
- Keep the staging logic of `_publish` only for the JSON pair; JPEGs are per-file atomic.
- Compute module stays free of Starlette/Jinja2 (`tests/unit/test_core_imports.py` covers `datasets`).
- Do not change the `annotation_batches` table shape.

## Dependencies
02.

## Verification
```bash
uv run pytest tests/unit/test_label_store.py tests/unit/test_cvat_export.py tests/integration/test_coco_roundtrip.py -q
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
