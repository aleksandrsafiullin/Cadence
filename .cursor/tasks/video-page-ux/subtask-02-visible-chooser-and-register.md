# Subtask 2: Visible chooser and register

## Goal

Choosing a file on «Видео» registers that file in place and selects it in the player. Add controls stay on the page. Source URL and the browser upload form are gone.

## Context

Spec §6.1 and §15 «Видео». Today `templates/video.html` wraps both add forms in `<details>` once a video exists (`add_another`, «Добавить другой файл»). `#upload-file` only posts when «Добавить загруженный файл» is clicked, so `video.src` stays on the previous id (the live page was still `/media/pilot/v_a834183bfe984920be63a24ef1d7875b`). `post_video_add` also reads `source_url`. This subtask wires subtask 1's chooser to `add_video(..., copy=False)` and removes those controls. Metadata placement and the weights paragraph are subtask 3; leave the aside in place except where a control removed here lived inside it.

`POST /video/add` already sets cookie `lab_video` and flash `video_id`. `_wanted_video` prefers that flash id over the old cookie. Reuse `_redirect(video_id)`.

## Acceptance Criteria

- [ ] `video_routes()` includes `POST /video/choose` and no longer includes `POST /video/upload`. No new upload route is added. `UPLOAD_CAP` in `src/excavation_lab/web/uploads.py` is still `8192 * 1024 * 1024`. `stream_import` is unchanged.
- [ ] `post_video_choose` calls `choose_video_path(prompt=book.choose_file_prompt)`. `None` (cancel) redirects to `/` and writes no flash. `ChooseFileFailed` flashes `choose_failed` and writes no video row. A suffix other than `.mp4` / `.mov` / `.mkv` flashes existing `upload_bad_name`, does not call `add_video`, and writes no row.
- [ ] A good path calls `add_video(workspace, project_id, path, copy=False)` with no `source_url`. Success flashes `video_added`, sets `lab_video` to the new id, and redirects to `/`. The following page's `<video src>` is `/media/{project_id}/{new_video_id}`. The stored path equals the chosen path (no copy under `project media/`). `size_bytes` and `sha256` on the row come from the existing probe. A probe or `LabError` after `copy=False` leaves the source file on disk.
- [ ] `post_video_add` still accepts `local_path` for tests, passes `copy=False`, and ignores `source_url` (a posted URL is not stored and `urlopen` is not called). Blank path still flashes `path_required`. `routes.py` still contains the literal `copy=False`. The module does not call `request.body(`.
- [ ] `templates/video.html` shows one add block on both the empty-video state and the has-video state. It is not inside `<details>` and the summary `add_another` is not rendered. The block is a readonly `<input id="local-path" type="text">` whose value is the selected video's stored absolute path, or empty when there is no video, plus `<form method="post" action="/video/choose">` with submit `choose_file_button`. The input is not a file input, has no `name="file"`, and is not posted as `local_path`.
- [ ] The template contains neither source-URL field (`source-url`, `source-url-more`, `source_url_label`) nor the upload form (`upload-file`, `upload_label`, `action="/video/upload"`, `multipart/form-data`). `path_help`, `upload_help`, and `source_url_help` are not rendered. `video_context` stops passing those keys.
- [ ] Empty project with no video: `add_local_video`, the allowed-directory caption, the chooser, no `<video>`, no demo URL. `add_local_video` in both catalogs now says to choose a local MP4, MOV, or MKV that is already on disk, that no sample is included, and that the app does not download a substitute. Russian stays the UI language; English has no Cyrillic.
- [ ] `choose_file_button` is «Выбрать файл» / `Choose file`, on the `Catalog` dataclass, both catalog literals, and the `RU` alias list.
- [ ] No project still shows `need_project` and does not show the chooser. The video `<select>` for two or more videos stays. Baseline, runs, and the aside stay until subtask 3.
- [ ] Page-workflow tests that required the upload form or `POST /video/upload` are updated in this subtask. `test_upload_cap_and_traversal_do_not_write`, `test_small_upload_is_registered_by_add_video`, and `test_real_camera_name_is_a_media_leaf` stay unit tests of the unused helper. `test_declared_and_counted_caps_write_nothing` no longer posts `/video/upload`; it asserts `UPLOAD_CAP` and `declared_over_cap` directly. `tests/unit/test_media_probe.py` source-URL cases and the CLI `--source-url` flag stay.

## Implementation Notes

Files: `src/excavation_lab/web/routes.py`, `src/excavation_lab/web/templates/video.html`, `src/excavation_lab/ui/copy.py`, `tests/unit/test_ui_video_page.py`, `tests/unit/test_ui_empty_states.py` (`test_project_without_video_still_asks_for_a_local_path`), `tests/unit/test_ui_locale.py` (the assertion that `book.upload_label` is on `GET /` after a project exists).

Monkeypatch `excavation_lab.web.routes.choose_video_path` in page tests. Do not call `osascript`.

Replace `test_streamed_upload_registers_without_buffering_the_body` with a chooser test: stub returns a clip under the workspace root, `add_video` is called with `copy=False`, the row path is that clip, and `_video_src` is the new id. Add the second-file case the same way `test_second_added_file_becomes_the_player` does for `/video/add`: an existing row stays, the stub returns another file, the selected option and `src` are the new id, and `#local-path`'s value is that absolute path.

`test_source_url_is_stored_and_not_fetched` in `test_ui_video_page.py` becomes: the HTML has no source-URL control, and `POST /video/add` with `source_url` set stores `None`. Keep the urllib refusal.

`test_failed_upload_unlinks_and_rejected_names_write_nothing` no longer drives `/video/upload`. Cover instead: cancel adds no row and the HTML has no new alert; a `.txt` stub flashes `upload_bad_name` and adds no row; a stubbed path whose probe fails leaves the source bytes in place.

`test_empty_project_names_the_workspace_root` and `test_english_video_page_keeps_only_the_language_endonym`: expect `choose_file_button`, do not expect `UPLOAD_LABEL`, `UPLOAD_BUTTON`, `<summary>Дополнительно</summary>`, or `add_another`. `ADD_VIDEO_BUTTON` is no longer a control on this page. `LOCAL_PATH_LABEL` stays. Do not render `local_path_placeholder` (it tells the user to type a path).

`store_browser_upload` and `stream_upload` are not imported by `routes.py` after this subtask. Do not delete `stream_import` or lower the cap. Do not call `store_browser_upload` from the route.

The readonly path is the selected row's `path` after `assemble_player`, including when `play_error` is set. Empty string when `has_video` is false.

## Dependencies

Subtask 1.

## Verification

```bash
uv run pytest tests/unit/test_ui_video_page.py tests/unit/test_ui_empty_states.py tests/unit/test_ui_locale.py tests/unit/test_choose_file.py tests/unit/test_media_probe.py -q
uv run ruff check src tests
```

Pytest must not open a file dialog. Do not download a clip. Subtask 3 owns the browser pass.
