# Subtask 12: Browser pass

## Goal

Exercise the running page server: Video first, then one sweep of the other pages, including theme and language.

## Context

Spec §15 and §25. A unit suite does not show whether the stage, the sidebar, and the buttons actually work. Subtask 11 left `uv run excavation ui` as the only UI process. No MP4 in the repo is not a license to download one.

## Acceptance Criteria

- [ ] Start `uv run excavation ui`. The process listens on `127.0.0.1` port 8501. Binding is not `0.0.0.0`.
- [ ] `/` (Видео): sidebar shows the project control and the six links. With no clip, the page shows the add path and the upload control, and there is no `<video>` element. Buttons are visible as buttons («Добавить видео», «Добавить загруженный файл»). The source URL is inside «Дополнительно», not the primary row.
- [ ] If a local MP4 is already on disk, add it by the path field. The stage element’s `src` is `/media/...`. Seeking does not request a full-file download when the browser sends Range (check the media response status in the network panel: 206). Metadata is probe text or «не рассчитано», not zeros. Do not add a file by downloading the YouTube URL.
- [ ] Baseline control is on the page. If weights are absent, the notice is the doctor/cache warning and no job directory appears. If a job is already running, the baseline button is disabled. Do not start a second analyze to prove the lock; the unit test did that. Do not run smoke training.
- [ ] Theme switch: choose light, reload, `data-theme` is `light`. Choose dark, reload, it is `dark`.
- [ ] Language switch: English catalog on Video (buttons in English), then Russian again.
- [ ] Visit `/data`, `/training`, `/analysis`, `/quality`, `/events` from the sidebar. Each shows its title. Events shows `events_unavailable`, the caption, the v0.2 caption, and «не рассчитано» / `not calculated` on the indicators, and no chart. Training and quality show no curve and no zero bars on an empty workspace. The fields `epoch`/`loss` and `класс`/`AP50` are proved by `tests/unit/test_web_chart.py`, not by this empty page. Analysis shows no video element on an empty workspace.
- [ ] A second browser request while a job is running does not create a second job. If no job is running, skip the live check and cite `tests/unit/test_ui_training_state.py` and `tests/unit/test_ui_video_page.py`.
- [ ] Failures found in this pass are fixed in the page that owns them and this pass is repeated for the broken screen. Do not file them as follow-ups and stop.

## Implementation Notes

Use the Cursor browser tools: navigate, snapshot, click, reload. A single screenshot is not the pass.

Empty `workspace/` is a valid Video result. Say so in the note you leave in the subtask completion, and do not invent a clip.

Stop the server when the pass is done. One process only.

## Dependencies

Subtask 11.

## Verification

```bash
uv run pytest tests -q
uv run ruff check src tests
```

Plus the manual pass above. Record which screens were clicked and whether a local MP4 was present. No `pilot-small`. No smoke-train quality claim.
