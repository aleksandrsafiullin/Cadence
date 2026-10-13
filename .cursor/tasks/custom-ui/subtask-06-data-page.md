# Subtask 6: Data and labeling page

## Goal

Render «Данные и разметка» as a form page that calls the existing dataset services. Freeze and the snapshot gate stay as they are.

## Context

Spec §15 «Данные и разметка». Logic is `src/excavation_lab/ui/data_page.py`: `store_episode`, `propose_recorded`, `freeze_recorded_proposal`, `run_sample`, `run_export`, `run_import`, `run_snapshot`, `snapshot_gate`, `review_count_text`, `episode_table`. The page does not assign splits itself. A second freeze reports `split_frozen` and leaves the manifest bytes alone. Contact sheet shows JPEGs the sampler wrote and does not draw boxes. Tests live in `tests/unit/test_ui_data_freeze.py`.

## Acceptance Criteria

- [ ] `GET /data` with no project or no video shows `data_empty` and `review_count_text(None)`. Counts that are unknown are «не рассчитано», not `0`. The snapshot control is disabled and the reason is `snapshot_gate(...).reason`.
- [ ] Episode form posts to `POST /data/episode` and calls `store_episode`. Propose posts to `POST /data/propose` (`propose_recorded`).
- [ ] Sample has two submits. The plan button calls `run_sample` with the English plan id (`diagnostic`, `content`, or `expanded`), `split=None`, `count=None`, and `spacing_s` (default `load_pilot().sample_spacing_s`). The count button calls `run_sample` with `plan=None`, the chosen split (`train`, `validation`, or `test`), the count, and `spacing_s`. The visible plan option is the catalog label. The stored plan id is English.
- [ ] Export posts to `POST /data/export` and calls `run_export` with `split` and `include_predictions`. Import posts to `POST /data/import` (`run_import`) with the uploaded ZIP streamed under the same 8192 MiB cap as video. `format_import_error` / `validation_issue_line` text is shown. Path and ZIP traversal stay inside the dataset service; this page does not unpack the ZIP itself.
- [ ] After propose, the page shows `guard_lines` and `block_lines`. Freeze posts to `POST /data/freeze` (`freeze_recorded_proposal`). The button label is catalog `freeze_split` («Зафиксировать разбиение»). A successful freeze shows catalog `split_hash` and `frozen.content_hash`. A second freeze does not change `split_manifest.json` bytes and shows `split_frozen`.
- [ ] Contact sheet lists paths from `contact_jpegs` only. No box overlay.
- [ ] `limitation_text` / same-scene sentence still comes from the split service (`SAME_SCENE_MESSAGE`), not a rewritten string.
- [ ] Snapshot posts to `POST /data/snapshot` only when `snapshot_gate.enabled` is true. The handler calls `run_snapshot`. A disabled control is `disabled`, and a forced POST does not call `snapshot_dataset`.
- [ ] Rewrite the `AppTest` functions in `tests/unit/test_ui_data_freeze.py` onto `TestClient` in this subtask. The file does not import `AppTest` when this subtask ends. Pure tests (`snapshot_gate`, review counts, import errors) stay. Do not leave those clicks for subtask 11.

## Implementation Notes

Video choice on this page uses `video_choice_label` from `video_page.py`.

Keep Advanced for fields that are already off the main form. Do not dump every Python field.

Import uses the existing `run_import` after the upload is stored at a path the service already accepts. Do not hold the ZIP in RAM past the cap. Reject `..` in the filename before writing.

No new split algorithm. No CVAT server inside the app. `data_page.py` does not import Starlette.

## Dependencies

Subtask 5. The shell and project picker exist. Video registration is how a fixture gets a clip into a project; data tests can keep calling `add_video` directly.

## Verification

```bash
uv run pytest tests/unit/test_ui_data_freeze.py -q
uv run ruff check src tests
```

Manual: `uv run excavation ui`, open `/data`. Empty project shows the empty sentence and a disabled snapshot control. Do not freeze a real pilot split as part of this check.
