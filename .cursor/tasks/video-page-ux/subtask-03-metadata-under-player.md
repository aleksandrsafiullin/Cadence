# Subtask 3: Metadata under the player

## Goal

Put probe metadata and the baseline interval under the video, show `playback_failed` when the browser cannot decode, and drop the weights-cache paragraph from «Видео».

## Context

Spec §15 «Видео»: show metadata, play, pick an interval, start baseline, list runs. Spec §5 requires the baseline mode label «Базовая модель, не обучена на наших классах» (`baseline_caption`).

The live page puts duration, fps, frame size, and codec in `aside.lab-aside`, next to a stage about 830×623. The stage's `<video>` uses `.lab-stage video { background: var(--pdm-surface) }`, so a file the browser does not paint is a colored rectangle. The aside also renders `weights_warning` («Официальные веса RF-DETR могут быть скачаны один раз в …/workspace/cache/weights…»). That sentence stays on Analysis. It leaves this page. Do not add a new informational `lab-alert` in its place.

Subtask 2 already left the chooser outside `<details>`. Do not put it back.

## Acceptance Criteria

- [ ] With a video selected, document order inside `.lab-video` is: the stage's `play_error` or `<video>`, then `ul.lab-meta` (duration, nominal and average fps, frame size, codec, SHA-256, device from `metadata_lines`), then `rotation_text`, then `offset_text`, then the baseline `<form action="/video/baseline">`. There is no `<aside class="lab-aside">`.
- [ ] Unknown duration, fps, frame size, codec, SHA-256, device, and rotation still render «не рассчитано» / `not calculated`. A stored offset of `0` stays `0`. `probe_note` still renders when re-probe reports a mismatch (the existing «Файл изменился» text in `test_large_mp4_is_served_without_a_preview`).
- [ ] The baseline form keeps `video_id`, `start_s`, `end_s`, `interval_help`, `baseline_caption`, `baseline_button`, and the `job_blocks` disabled state. `POST /video/baseline` still calls `request_baseline_run`. A bad interval does not enqueue. A second start while a job blocks still returns `JobAlreadyRunningError`.
- [ ] `playback_file` and `playback_preview` are not rendered. `player_file` and `playback_path` are unchanged, including the 200 MiB MOV/MKV preview branch and the MP4 path that does not call `write_preview`.
- [ ] When `media_url` is set, the template includes a hidden `lab-alert` whose text is `playback_failed`, and a script in `video.html` only. On the `<video>` `error` event the script hides the video element and unhides that alert. It does not run for analysis or training players (`lab.js` stays untouched). Server-side `play_error` still renders when `player_file` fails, and that response has no `<video>` and no green empty stage standing in for the error.
- [ ] `weights_warning` is not in `video.html`, `video_context`, or `PlayerState`. `GET /` does not contain «Официальные веса» or the English cache sentence. Missing official Nano weights still do not enqueue: `request_baseline_run` returns an empty `BaselineAttempt` (no notice, no error, no `job_id`). `post_video_baseline` does not flash the cache sentence. `visible_notice` returns only `video_added` and notices that start with `baseline_queued`.
- [ ] `src/excavation_lab/web/templates/analysis.html`, `analysis_routes.py`, and `ui/analysis_page.py` are unchanged. Training templates are unchanged. The catalog method `weights_warning` stays for Analysis and `tests/unit/test_ui_locale.py`.
- [ ] `.lab-video` in `src/excavation_lab/web/static/lab.css` is a single column. The rule `grid-template-columns: minmax(0, 1fr) minmax(16rem, 22rem)` is gone. `.lab-aside` is removed if nothing uses it. `.lab-split` and the app shell `grid-template-columns: 21rem 1fr` stay. `tests/unit/test_web_shell.py::test_lab_css_uses_the_shell_tokens` still requires `.lab-video` and the shell track, and no longer requires the 22rem aside track. No new hex colors.
- [ ] No new informational `lab-alert` copy. Empty project still has the chooser and no `<video>` (subtask 2). v0.2 events stay hidden. One GPU/MPS job behavior is unchanged.

## Implementation Notes

`assemble_player` stops computing `weights_warning`. Drop the field from `PlayerState` and the template context key.

`test_missing_weights_show_the_doctor_notice`: the page text does not contain the cache sentence before or after `POST /video/baseline`, and `_requests` stays empty.

`test_cached_weights_clear_a_stored_download_notice` stays valid. `test_added_file_shows_probe_metadata_and_plays_from_disk` and `test_large_mp4_is_served_without_a_preview` stop requiring `PLAYBACK_FILE` in the HTML. They still require `metadata_lines`, the media `src`, and Range 206.

Do not restyle `--pdm-surface`. Hiding the video on `error` is what removes the silent rectangle.

The hidden `playback_failed` node is in the HTML whenever a src is rendered, so a raw-text assertion will see it. Visible text on a successful paint does not show it. The browser pass checks that.

## Dependencies

Subtask 2.

## Verification

```bash
uv run pytest tests -q
uv run ruff check src tests
```

Browser pass, localhost only, on a temporary workspace. Do not use the repo `workspace/` and do not stop a server the user already has on port 8501. Pick a free port.

```bash
tmp=$(mktemp -d)
uv run excavation project create --name pilot --workspace "$tmp"
# Write two synthetic MP4s into "$tmp" with tests/fixtures/synthetic.py
# (write_synthetic_mp4). Do not download a substitute.
uv run excavation video add --project pilot --workspace "$tmp" --path "$tmp/first.mp4"
EXCAVATION_CHOOSE_PATH="$tmp/second.mp4" uv run excavation ui --workspace "$tmp" --port <free>
```

Then, on `http://127.0.0.1:<free>/`:

- Add controls are visible without opening a `<details>` element. `#local-path` is readonly text, not `type=file`.
- Before the click, `video.src` contains the first id. Click «Выбрать файл». After the redirect, `src` is `/media/pilot/{second id}` and `#local-path` is the absolute path of `second.mp4`.
- The metadata list is under the video, not in `aside.lab-aside`. The baseline form is under that list.
- The HTML has no «Исходный URL», no «Небольшой файл через браузер», no `action="/video/upload"`, and no weights-cache sentence.
- Dispatch `error` on the `<video>` (the synthetic file will otherwise decode). The element becomes hidden and the visible alert is `playback_failed`.
- A separate load with no registered video (fresh project in the same temp workspace, or a second project) shows `add_local_video` and the chooser, and has no `<video>`.

Stop that server. Delete only `$tmp` after confirming it is the directory `mktemp` created. Do not delete the repo `workspace/` or the user's existing clips. Unset `EXCAVATION_CHOOSE_PATH` in the shell you keep using.

No dialog should appear during this pass. `train --profile smoke` is not a check.
