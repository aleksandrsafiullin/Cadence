# Subtask 5: Video page

## Goal

Replace the Video widget stack with a stage, a metadata column, and real buttons, calling `add_video`, `reprobe_video`, and `request_baseline_run`.

## Context

Spec §15 «Видео», §6.1, §16. The behavior lives in `src/excavation_lab/ui/video_page.py`. Rendering today is `st.columns`, `st.video`, and `st.button`. This subtask moves rendering to `templates/video.html` and the routes in the master plan. It does not change probe, the encoder, or `enqueue_analyze`.

Constants that stay in `video_page.py`:

- `_DIRECT_PLAYBACK_MAX_BYTES = 200 * 1024 * 1024` — still used by `playback_path` for the MOV/MKV preview path. The MP4 route does not use it.
- `_PREVIEW_WINDOW_S = 2.0`, `_PREVIEW_FPS = 5.0`, `_PREVIEW_FRAME_CAP = 8` — unchanged, used only by `_short_preview`.
- `_UPLOAD_SUFFIXES`, `_UPLOAD_STEM_LIMIT`, `_upload_leaf`.
- `_BASELINE_MODEL_ID = "baseline_coco"`.

`upload_rejection` returns a catalog string for an empty body and `None` for any positive size, including past 200 MiB and past 8 GiB. Do not add the cap there. `tests/unit/test_ui_video_page.py::test_upload_cap_and_traversal_do_not_write` locks that.

## Acceptance Criteria

- [ ] `GET /` with no video and a project renders `add_local_video` or `allowed_directory_caption`, the path field (`local_path_label`), and the upload field (`upload_label`). The response has no `<video>` tag and no demo URL.
- [ ] The stage/aside layout is a CSS grid in `lab.css` (`.lab-video`). Metadata sits beside the stage, not under a pile of stretch columns. Buttons are `<button type="submit" class="lab-button">` with the catalog labels «Добавить видео», «Добавить загруженный файл», and «Запустить базовую модель».
- [ ] `POST /video/add` reads the path and optional source URL and calls `add_video`. Empty path shows `path_required`. A path outside the allowed roots shows the existing `LabError` message and writes no row. Source URL is stored and not fetched. The URL field is inside `<details>` «Дополнительно» on the empty page.
- [ ] After one video exists, the add form is inside `<details>` «Добавить другой файл». Adding a second file selects it (`video_choice_label`, which includes `video_id` when two names match).
- [ ] `POST /video/upload` streams the body into the project media leaf, then calls `add_video(..., copy=False)`. Declared or counted size above `8192 * 1024 * 1024` returns 413 and writes nothing. Empty body shows `upload_empty`. Names that `_upload_leaf` rejects (`../clip.mp4`, `notes.txt`) write nothing. An accepted small MP4 is registered and lands in the project `media/` directory. If registration fails after the stream, unlink that leaf the way `store_browser_upload` does. `add_video` with `copy=False` does not remove the file. The route does not call `await request.body()` for a file that might be the whole cap.
- [ ] `GET /` for a registered id calls `reprobe_video` once for that response. Mismatch shows `format_probe_line` and does not overwrite the stored row (that is already what `reprobe_video` does). The aside lists `metadata_lines` and `rotation_line`. Unknown duration, fps, frame size, codec, sha, and device render «не рассчитано», not `0`. A stored offset of `0` stays `0` (`offset_line`).
- [ ] An `.mp4` player `src` is `/media/{project_id}/{video_id}`. That route resolves the registered path through storage and allowed roots, then `FileResponse`. `Range: bytes=0-3` returns 206, `Content-Range`, and four bytes. A file whose `size_bytes` is above `_DIRECT_PLAYBACK_MAX_BYTES` does not create `media/previews/` and does not call `write_preview`. The caption is `playback_file`.
- [ ] For a non-MP4, `player_file` returns whatever `playback_path` returns, and the caption follows that result. A `.mov` or `.mkv` at or under `_DIRECT_PLAYBACK_MAX_BYTES` (200 MiB) stays the source file and the caption is `playback_file`. Above 200 MiB, `playback_path` is the short preview and the caption is `playback_preview`. The media route Range-serves that path. `test_large_source_plays_a_short_encoder_preview` still calls `playback_path` directly and still expects a preview file. Do not change that function's 200 MiB branch.
- [ ] Interval inputs default from `interval_bounds` (half-open `[start_offset_s, start_offset_s + duration_s)`). `POST /video/baseline` calls `request_baseline_run`. A bad interval does not enqueue. Missing official Nano weights show `weights_warning` and do not enqueue. `job_blocks` disables the button, and a second POST while a queued, running, or cancel-requested job exists returns `JobAlreadyRunningError` and does not call `enqueue_analyze` again. The queued id is shown from the flash notice. Runs list uses `job_status_line` (orphaned, partial, real `analyzed_end_s` only). «Обновить статус» reloads `GET /`. The runs caption `runs_no_boxes` stays. The page does not construct a detector.
- [ ] English locale renders the English button labels and no leftover Cyrillic on this page.
- [ ] Rewrite the `AppTest` functions in `tests/unit/test_ui_video_page.py` onto `TestClient` in this subtask (`test_empty_project_names_the_workspace_root`, `test_path_outside_the_root_is_rejected_in_russian`, `test_added_file_shows_probe_metadata_and_plays_from_disk`, `test_second_added_file_becomes_the_player`, `test_bad_interval_does_not_enqueue`, `test_missing_weights_show_the_doctor_notice`, `test_running_job_disables_baseline_and_blocks_a_second_start`, `test_baseline_enqueues_once_and_a_rerun_does_not_repeat`, `test_interval_bounds_use_start_offset_and_preview_seeks_there`, `test_offset_widgets_default_to_the_source_window`, `test_same_filename_selectbox_includes_video_id`, `test_cached_weights_clear_a_stored_download_notice`). The file does not import `AppTest` when this subtask ends. Also rewrite `tests/unit/test_ui_empty_states.py::test_project_without_video_still_asks_for_a_local_path` onto `TestClient` here. Pure tests (`metadata_lines`, `upload_rejection`, `_upload_leaf`, `interval_bounds`, `playback_path`) stay. Do not keep a second copy of those cases for subtask 11.

## Implementation Notes

Flash: `preferences.py` `flash.json` carries `error` and `notice` for the add and baseline results. POST returns 303 to `/`. GET clears the flash. Do not put the message in the query string.

The media route never takes a filesystem path from the query. `video_id` is looked up. A symlink that leaves the allowed roots is 404. Traversal in the URL (`..`) is 404.

`player_file` in `video_page.py` returns `(path, caption_kind)`. An `.mp4` returns the registered file and does not call `write_preview`. Any other suffix returns the path `playback_path` already returns: the source when `playback_path` returns the source (`playback_file`), the preview when `playback_path` returns the preview (`playback_preview`). The route and the template both use that result.

Hardware line is the device string `metadata_lines` already builds. Do not add a new report.

Do not import Starlette from `video_page.py`. The route module imports the page helpers.

Baseline still uses `load_pilot().analysis_fps` inside `request_baseline_run`. Do not add a confidence control that rewrites a saved run.

## Dependencies

Subtask 4.

## Verification

```bash
uv run pytest tests/unit/test_ui_video_page.py -q
uv run ruff check src tests
```

Manual: `uv run excavation ui`, open `http://127.0.0.1:8501/`. With no MP4, confirm the add form and the absence of a video element. If a local file is already on disk, add it by path and confirm the element `src` is the media route and the metadata is the probe text. Do not download a substitute. Do not start a real baseline unless weights are already cached; the missing-weights notice is the honest state. Stop the process.
