# Video page UX

## Overview

The «Видео» page registers a local file by a localhost native chooser and plays that file. Add controls stay on the page. Metadata and the baseline interval sit under the player. Source URL, the browser multipart upload, and the weights-cache paragraph leave this page.

Spec §5 step 1, §6.1, and §15 «Видео». A `source_url` in storage or on the CLI is provenance only. The page does not ask for one and does not fetch one. There is no seventh product change: the request stopped at "7." with no text.

Do not edit `.cursor/tasks/master-plan.md`, `.cursor/tasks/custom-ui/`, `.cursor/tasks/i18n/`, or `.cursor/tasks/lab-design/`.

## Phases

1. **Chooser** — macOS `osascript` choose-file in the page-server boundary. Argument array, no `shell=True`. Pytest injects the path and does not open a dialog.
2. **Register** — the chooser calls existing `add_video(..., copy=False)`. The redirect cookie selects the new `video_id`. Add controls stay visible. Source URL and the multipart upload form leave the page.
3. **Player column** — metadata and the baseline form sit under `<video>`. A browser decode error shows the existing `playback_failed` string. The weights-cache sentence is not rendered on this page.

Events, training, analysis layout, the upload cap, and the worker lock are unchanged.

## Dependencies

- Phase 2 depends on phase 1. The route calls `choose_video_path` and must not grow a second dialog implementation.
- Phase 3 depends on phase 2. It moves the block phase 2 leaves on the page; it does not put add controls back inside `<details>`.
- No dependency on a pilot MP4, a checkpoint, or `train --profile smoke`.

## Execution order

| Order | File | Goal | Depends on |
| --- | --- | --- | --- |
| 1 | `subtask-01-native-file-chooser.md` | Stubbable `osascript` chooser, no GUI in pytest | — |
| 2 | `subtask-02-visible-chooser-and-register.md` | Register the chosen path, drop URL and browser upload | 1 |
| 3 | `subtask-03-metadata-under-player.md` | Metadata under the player, decode error, no weights essay, browser pass | 2 |

## Verification

After subtask 3:

```bash
uv run pytest tests -q
uv run ruff check src tests
```

Browser pass is in subtask 3. It uses a temporary workspace and `EXCAVATION_CHOOSE_PATH`. It does not download a substitute MP4, does not open a real file dialog, and does not point at the repo `workspace/` pilot data.

`train --profile smoke` is not a check. A green rectangle with the old `video.src` is not success.

## Resolved decisions

1. **No browser file input.** `<input type="file">` `.value` is a fakepath. The video page has no file input and no `enctype="multipart/form-data"`. The path field is a readonly text input. Its `value` is the selected row's stored absolute path, or empty when no video is selected.

2. **Dialog lives in `web/`, not in probe math.** `src/excavation_lab/web/choose_file.py` builds `["osascript", "-e", script, "--", prompt]` and runs it with `subprocess.run` on that list. `shell` is not passed. `src/excavation_lab/ui/video_page.py` stays free of `subprocess` and Starlette (`tests/unit/test_ui_video_page.py::test_page_does_not_shell_out_or_build_a_detector`). `src/excavation_lab/media/probe.py` does not import the chooser.

3. **Two stubs, neither reads the browser.** Pytest passes a runner or monkeypatches `choose_video_path`. The browser pass sets `EXCAVATION_CHOOSE_PATH` to an absolute path; when that variable is set, `osascript` is not executed, including when the path is invalid. Unset or blank means the real dialog. Normal `uv run excavation ui` does not set it.

4. **Cancel is silence.** AppleScript error `-128` returns an empty string and exit 0. The route redirects to `/` with no flash. A non-cancel `osascript` failure flashes the new catalog sentence `choose_failed` and does not register a row.

5. **Register in place.** `POST /video/choose` calls `add_video(workspace, project_id, path, copy=False)` and does not pass `source_url`. The stored path is the chosen file. SHA-256 and `size_bytes` stay the probe's job. A probe failure must not delete that file (`add_video` unlinks only when `copy` is true). `routes.py` keeps the literal `copy=False` because `test_page_does_not_shell_out_or_build_a_detector` reads it.

6. **Selection.** The choose route uses the existing `_redirect(video_id)` cookie `lab_video` and flash `video_id`. `_wanted_video` already prefers the flash id over the previous cookie. After a successful add, `<video src>` is `/media/{project_id}/{new_video_id}`.

7. **`POST /video/add` stays, without a typed field and without `source_url`.** Tests that post `local_path` still register a file. The handler ignores `source_url` and passes `copy=False`. The template does not submit `local_path`. CLI `excavation video add --source-url` and `tests/unit/test_media_probe.py` stay as they are.

8. **Upload route leaves the page.** Remove `Route("/video/upload")` and `post_video_upload`. Do not add another upload route. Do not change `UPLOAD_CAP` (`8192 * 1024 * 1024`) in `src/excavation_lab/web/uploads.py`. `stream_import` stays for the data page. `store_browser_upload` stays uncalled by the video routes and template. Cap tests move onto `declared_over_cap` and the constant.

9. **Weights sentence leaves «Видео» only.** Delete it from `video.html`, `video_context`, and `PlayerState`. `request_baseline_run` still does not enqueue when official Nano weights are missing, and it does not return that sentence. `visible_notice` on this page returns only `video_added` and a `baseline_queued` notice. `analysis.html`, `analysis_routes.py`, and `analysis_page.py` keep their own `weights_warning`. Do not add a replacement `lab-alert`.

10. **What stays visible.** Real failures: add errors (`LabError` from register, including path-not-allowed and probe), `list_error`, server `play_error`, `probe_note`, and `choose_failed`. Spec §5 label `baseline_caption`. Interval form, including existing `interval_help`. Empty-state `add_local_video` and `allowed_directory`. Runs list. Page lead. Playback captions `playback_file` and `playback_preview` are not rendered. `path_help`, `upload_help`, and `source_url_help` are not rendered.

11. **Decode error.** When the browser fires `error` on the page's `<video>`, a script in `video.html` hides that element and shows the existing `playback_failed` string. Do not change `--pdm-surface` to hide the empty stage. Do not put that script in `lab.js` (analysis and training load it).

12. **CSS lock.** `tests/unit/test_web_shell.py::test_lab_css_uses_the_shell_tokens` currently requires `grid-template-columns: minmax(0, 1fr) minmax(16rem, 22rem)`, which is `.lab-video`'s aside track. Subtask 3 removes that track and updates this assertion. The app shell `grid-template-columns: 21rem 1fr` and `.lab-split` stay.
