# Subtask 8: Analysis page

## Goal

Render «Анализ» as an encoded MP4, one diagnostic frame, and a vertex table. Start and export go through the existing job lock.

## Context

Spec §15 «Анализ». `src/excavation_lab/ui/analysis_page.py` holds `request_analysis`, `request_export`, `playback_file`, `read_source_frame`, `detection_table`, `predicted_list`, `save_zone_table`, `zone_preview_rgb`, `note_threshold_edit`, `start_enabled`, `analysis_header`, `observed_share`. `playback_file` returns the annotated MP4 under `exports/<run_id>/` or `None`. It is never a frame sequence. Tests in `tests/unit/test_ui_analysis_state.py` lock: no detector weights loaded by the page, predicted boxes out of the detection table, threshold edits do not rewrite parquet, one MP4 plus a separate predicted list, zone vertices in unit coordinates.

This page does not import `charts.py`. Do not add a chart.

## Acceptance Criteria

- [ ] No project or no video: `analysis_empty` / `analysis_no_video`, header from `analysis_header(None)` so source time, model version, and observed share are «не рассчитано». No boxes, no `<video>`, no chart.
- [ ] `POST /analysis/start` calls `request_analysis` with `model_id`, `analysis_fps`, a `ThresholdDraft`, and the source interval. The form includes baseline/custom (`baseline` stores `baseline_coco`; custom stores the id from `list_bundles`), an editable `analysis_fps` whose default is `load_pilot().analysis_fps`, the interval from `interval_defaults`, and `threshold_draft_from_pilot`. `start_enabled` disables the button. A second POST while a job blocks does not enqueue. Missing weights show the notice and do not enqueue. Custom mode without a bundle does not enqueue. A measured start offset of `0` stays `0`; a missing offset does not become `0`.
- [ ] `AnnotateToggles` are posted only with `POST /analysis/export`, which calls `request_export`. They are not arguments to `request_analysis`. Changing them does not rewrite a finished run's parquet. Export uses `zones_for_export`.
- [ ] Result playback is `<video>` pointed at `GET /runs/{project_id}/{run_id}/video`, which serves `playback_file` by Range after the same path checks as the video route. No file → no element, and the caption says the export is absent. The route does not walk source frames.
- [ ] Diagnosis is one still: `GET /analysis/frame` encodes the array from `read_source_frame` as one JPEG. That array is upright RGB. Convert RGB to BGR before `cv2.imencode`, or write the JPEG with an RGB encoder. Do not pass the RGB buffer to `cv2.imencode` unchanged. The table beside it is `detection_table` for that `source_time_s`. `predicted_list` is a separate list. A predicted box is not a row in the detection table.
- [ ] Header lines are `header_lines` (source time, model version, observed share). `observed_share` still skips frames that were not analyzed. A measured share of `0` stays `0`. An unknown share is «не рассчитано».
- [ ] `POST /analysis/threshold` calls `note_threshold_edit`. It does not rewrite detections or the saved run.
- [ ] Zones: an HTML table of vertices, `POST /analysis/zones` calls `save_zone_table`. Preview image is `zone_preview_rgb`, which is RGB. Save it with PIL, or use the same RGB-to-BGR conversion as the diagnostic JPEG. Coordinates stay in unit range. No canvas library and no drawable-canvas package. Out-of-range rows are rejected the way `_out_of_range` already rejects them.
- [ ] Progress on this page reuses `GET /jobs/{job_id}/progress` and `lab.js`. It shows the `line` from that payload. It does not draw a loss curve.
- [ ] Rewrite the `AppTest` functions in `tests/unit/test_ui_analysis_state.py` onto `TestClient` in this subtask, and rewrite `tests/unit/test_ui_locale.py::test_analysis_time_survives_the_locale_switch` onto `TestClient` here. `test_ui_analysis_state.py` does not import `AppTest` when this subtask ends. Keep the assertions about predicted boxes, parquet, the lock, one MP4, and unit-square vertices. `analysis_page.py` does not import Starlette. Do not leave those clicks for subtask 11.

## Implementation Notes

Bundle list is `list_bundles` (directories that already contain `weights.pth`). The page does not load the weights.

Frame route query is `video_id` and `source_time_s` for the selected project. Reject a path parameter. One timestamp, one image.

`_analysis_progress` today polls by fragment. The JS poll from subtask 7 is the replacement. Do not redraw the MP4 on each poll; only the status text updates.

Annotate toggles are initial values on the export form. They are not sent to `request_analysis`.

## Dependencies

Subtask 7. The progress route and `lab.js` exist.

## Verification

```bash
uv run pytest tests/unit/test_ui_analysis_state.py -q
uv run ruff check src tests
```

Manual: `uv run excavation ui`, open `/analysis` on an empty workspace. Confirm the three «не рассчитано» header lines and no video element. Do not launch analyze. Stop the process.
