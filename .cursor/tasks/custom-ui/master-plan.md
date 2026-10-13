# Custom pages instead of Streamlit

## Overview

Replace the Streamlit UI with server-rendered pages we control: Jinja2, our CSS, and a little vanilla JS, on a localhost-only Starlette app. Launch with `uv run excavation ui`. Retire `uv run streamlit run app.py`.

The Video page is the reason for the change. It is a stage, a metadata column, and real buttons. It is not a port of `st.columns`.

Compute stays where it is. `media`, `datasets`, `models`, `tracking`, `analytics`, `jobs`, `storage`, `hardware`, and `schemas` do not import Starlette, Jinja2, or Streamlit. Pages call the same services the CLI calls. This plan does not retune probe, PTS, ByteTrack, or training, and it does not open v0.2 events.

Pure page logic stays in `src/excavation_lab/ui/` (`metadata_lines`, `upload_rejection`, `_upload_leaf`, `request_baseline_run`, `snapshot_gate`, `request_training`, `request_analysis`, `charts.py`, catalogs). Rendering moves to `src/excavation_lab/web/`.

## Stack (fixed)

- Python 3.11, `uv`, existing lock. Add `starlette`, `jinja2`, `uvicorn`. Dev: `httpx` (Starlette `TestClient`).
- Templates: `src/excavation_lab/web/templates/`. Static: `src/excavation_lab/web/static/`.
- Bind `127.0.0.1` only. Default port `8501`. No auth, no telemetry, no OpenAPI, no CDN script.
- `altair` becomes a direct dependency when charts move, because it is only transitive through Streamlit today. No Vega runtime, no Plotly.
- Still forbidden: React, Vue, a SPA bundler, FastAPI, Ultralytics, hosted inference, Docker for the app, a substitute MP4, a second detector.

Streamlit stays installed until the pages have moved. Subtask 11 removes it.

## Layout the new code must use

```text
src/excavation_lab/web/
  app.py            # create_app(workspace: Path) -> Starlette
  server.py         # HOST = "127.0.0.1"; serve(workspace, port=8501)
  routes.py
  uploads.py        # 8192 MiB cap, streamed body
  range_files.py    # FileResponse only after storage resolves the path
  preferences.py    # project id + one-shot flash, same symlink rule as locale.py
  templates/
    base.html video.html data.html training.html analysis.html quality.html events.html
  static/
    lab.css lab.js chart.js
```

`src/excavation_lab/ui/lab_theme.css` stays the token file (`--pdm-*`). `lab.css` uses those variables and does not add a second palette. `streamlit_chrome.css` stays until subtask 11, then it is deleted.

`uv run excavation ui` is a Typer command on the existing `excavation` app. The function body imports `excavation_lab.web.server` lazily so `doctor` and `train` do not load the server. `create_app` takes the workspace path. Tests pass `tmp_path`. The CLI passes `storage.default_workspace_root()`.

## Pages and routes

Russian labels, English paths. Order matches `navigation.iter_pages`.

| Page | GET | POST |
|---|---|---|
| Видео | `/` | `/video/add`, `/video/upload`, `/video/baseline` |
| Данные и разметка | `/data` | `/data/episode`, `/data/propose`, `/data/freeze`, `/data/sample`, `/data/export`, `/data/import`, `/data/snapshot` |
| Обучение | `/training` | `/training/start`, `/training/cancel` |
| Анализ | `/analysis` | `/analysis/start`, `/analysis/export`, `/analysis/zones`, `/analysis/threshold` |
| Качество | `/quality` | `/quality/compare`, `/quality/label` |
| События | `/events` | none |

Shared:

- `POST /project`, `POST /theme`, `POST /locale` write the workspace files and return 303 to the page that submitted.
- `GET /jobs/{job_id}/progress` returns the job row fields plus the raw `progress.json` object, or `null` if that file is absent. No invented epoch, ETA, or `analyzed_end_s`.
- `GET /media/{project_id}/{video_id}` serves bytes with HTTP Range.
- `GET /runs/{project_id}/{run_id}/video` serves the annotated MP4 from `playback_file`.
- `GET /analysis/frame` returns one JPEG from `read_source_frame`.
- `GET /static/...` serves `lab.css`, `lab.js`, `chart.js`, and `lab_theme.css`.

There is no `/api` product and no events propose/export route. A second POST hits the existing job lock (`request_baseline_run`, `request_training`, `request_analysis`).

Notices that used to live in `st.session_state` (`video_add_error`, baseline job id, train attempt) go in one atomic `workspace/ui/flash.json`, read and cleared on the next GET. Same symlink-inside-workspace rule as `write_locale_file`. Do not put the message in the query string.

Selected project id is `workspace/ui/project.txt`, one line. Theme stays `workspace/ui/theme.txt`. Locale stays `workspace/ui/locale.txt`. The page server calls `resolve_theme(workspace, session={})` and `resolve_locale(workspace, session={})` so it does not read `st.session_state`. Missing theme file → `dark`. Missing locale file → `ru`.

## Video layout

CSS grid, not widget columns.

- `.lab-app`: sidebar `21rem` (the width already in `lab_theme.css`) and the main column.
- Sidebar: product title, project `<select>`, workflow links, appearance (theme, language). Kickers stay `PROJECT`, `WORKFLOW`, `APPEARANCE`.
- `.lab-video`: a stage and a metadata aside.
- Stage: `<video controls preload="metadata">` when a playable file exists. No element when the project has no clip.
- Aside: `metadata_lines`, `rotation_line`, `offset_line`. Missing values stay the catalog «не рассчитано».
- Actions are `<button type="submit">`: «Добавить видео», «Добавить загруженный файл», «Запустить базовую модель», «Обновить статус». Source URL stays inside `<details>` labeled «Дополнительно». After a clip exists, the add form sits in `<details>` labeled «Добавить другой файл».
- Runs use `job_status_line` and `status_tone` as `.lab-alert` rows. `status.py` gains `status_markup()` that returns the escaped HTML `render_status` already builds. The template prints that string with `| safe` only for this helper.

Playback:

- `.mp4` of any size is the registered file, served by Range. `player_file` does not call `playback_path` or `write_preview` for an MP4.
- A `.mov` or `.mkv` returns whatever `playback_path` returns. At or under 200 MiB that is the source file and the caption is `playback_file`. Above 200 MiB it is the short preview (`_PREVIEW_WINDOW_S = 2.0`, `_PREVIEW_FPS = 5.0`, `_PREVIEW_FRAME_CAP = 8`) and the caption is `playback_preview`.
- `test_large_source_plays_a_short_encoder_preview` keeps calling `playback_path` and still expects a preview. A route test asserts a large MP4 is 206 from the source and does not create `media/previews/`.

`upload_rejection` stays the empty-body check. It still returns `None` for a large size (`test_upload_cap_and_traversal_do_not_write`). The route enforces `8192 * 1024 * 1024` bytes, the retired `server.maxUploadSize`, and refuses with 413 before writing. An accepted body is streamed to the media leaf, then `add_video(..., copy=False)`. If registration fails, that leaf is unlinked. `add_video` with `copy=False` does not remove the file. `_upload_leaf` still rejects `../`, slashes, and suffixes other than `.mp4`, `.mov`, `.mkv`.

Opening `/` calls `reprobe_video` for the selected id. The progress route does not. Mismatch text is `format_probe_line`. A missing file shows `allowed_directory_caption` and the add form, not a player.

## Other pages (behavior already in the modules)

- Data: plan sample calls `run_sample` with the English plan id, `split=None`, `count=None`, and `spacing_s`. Count sample calls it with `plan=None`, split `train` / `validation` / `test`, the count, and `spacing_s` (default `load_pilot().sample_spacing_s`). Export posts `split` and `include_predictions`. After propose, show `guard_lines` and `block_lines`. A successful freeze shows catalog `split_hash` and `content_hash`. A second freeze still reports `split_frozen` and does not rewrite the manifest. Snapshot follows `snapshot_gate`.
- Training: advanced fields include encoder learning rate, workers, and seed, filled by `_apply_profile`. `run_test` stays false and off the form. Progress JSON includes `orphaned`, and the polled line is `job_status_line`. Loss curve only when `should_draw_chart`. ETA only from `format_eta` (needs three measured epochs). Profiles on the form stay `smoke` and `pilot-small`.
- Analysis: start posts baseline/custom, `list_bundles`, editable `analysis_fps`, the source interval, and `threshold_draft_from_pilot` into `request_analysis`. `AnnotateToggles` go only to `request_export`. `read_source_frame` is upright RGB; the JPEG encoder converts to BGR or uses an RGB writer. `zone_preview_rgb` is saved with PIL or the same conversion. Predicted boxes stay out of `detection_table`.
- Quality: `load_quality_report`, `compare`, `send_to_labeling`. «Отправить в разметку» only when `send_enabled` (train). The gated `ap50_chart` call stays in the page module until subtask 10.
- Events: `events_unavailable`, `events_unavailable_caption`, `events_v02_caption`, and each `event_indicator_labels` entry as «не рассчитано». No chart, no zero, no propose/export. `unavailable_error().message` stays off the page.

Charts: subtasks 7 and 9 leave the gated `loss_chart` / `ap50_chart` calls in the page modules. Subtask 10 moves them, retargets `test_pages_call_the_styler_only_behind_the_existing_gates` at the progress route and the quality page, and updates `test_only_training_and_quality_import_charts` if those imports leave the page modules. `chart.js` is unit-tested on the fields `epoch`/`loss` and `класс`/`AP50`. Subtask 11 retargets `test_vega_container_uses_surface_tokens` when `streamlit_chrome.css` is deleted. No chart on Analysis.

## Phases

1. Docs and rules, so the spec matches the order to leave Streamlit.
2. Dependencies, Streamlit still present.
3. Process, bind, base template, CLI command.
4. Sidebar, theme, locale, project, five routes plus the events gate.
5. Video page end to end.
6. Data, then training, then analysis, then quality.
7. Charts on training and quality.
8. Delete Streamlit, `app.py`, `.streamlit/`, and the chrome CSS.
9. Browser pass on Видео and one sweep of the other pages.

## Dependencies

```text
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12
```

5 is the product priority, but 4 has to exist so the video template has a shell. 6–9 are sequential so each page's tests stay green while `render()` still exists for the pages not yet ported. 10 needs the training and quality templates. 11 needs every page on Starlette. 12 needs Streamlit gone so the exercised process is `excavation ui`.

## Execution order

| # | File | Goal |
|---|---|---|
| 1 | `subtask-01-spec-and-rules.md` | Spec, rules, agents, skills, DECISIONS, README |
| 2 | `subtask-02-package-deps.md` | Add Starlette, Jinja2, uvicorn, httpx; keep Streamlit |
| 3 | `subtask-03-page-server.md` | `create_app`, bind, CLI, base template |
| 4 | `subtask-04-shell-nav-theme.md` | Sidebar, theme, locale, project, events gate |
| 5 | `subtask-05-video-page.md` | Video stage, actions, Range, tests |
| 6 | `subtask-06-data-page.md` | Data and labeling forms, freeze lock |
| 7 | `subtask-07-training-page.md` | Train form, start/cancel, progress poll |
| 8 | `subtask-08-analysis-page.md` | MP4, one frame, zone table |
| 9 | `subtask-09-quality-page.md` | Metrics, compare, train-only send |
| 10 | `subtask-10-charts.md` | Altair spec in the page, small SVG script |
| 11 | `subtask-11-remove-streamlit.md` | Delete Streamlit, `app.py`, `.streamlit/`, `render()` |
| 12 | `subtask-12-browser-pass.md` | Exercise Видео and the other routes |

## Verification

```bash
uv run pytest tests -q
uv run ruff check src tests
```

UI subtasks also open `uv run excavation ui` once that command exists (subtask 3 onward) and name the screen. `train --profile smoke` is not a quality claim. No `pilot-small` run. No substitute MP4.

Page subtasks 4–9 rewrite their own `AppTest` functions onto `TestClient`, including `test_project_without_video_still_asks_for_a_local_path` (subtask 5), `test_analysis_time_survives_the_locale_switch` (subtask 8), and `test_quality_page_without_metrics_json_has_no_zero_chart` plus `test_english_pages_reject_leftover_cyrillic` (subtask 9). Subtask 11 deletes Streamlit and owns the remaining named source-lock tests: `render`, `GET /`, `st.navigation`, `on_change`, `.streamlit/config.toml`, `streamlit_chrome.css`, and `combined_css` on `lab_theme.css` only.

## Intentionally not in this plan

- Rewrites of probe, PTS, ByteTrack, training, or the four-class head.
- A second detector, or v0.2 event calculation. «События» is a gate.
- Editing `.cursor/tasks/master-plan.md`, `.cursor/tasks/subtask-*.md`, `.cursor/tasks/i18n/**`, or `.cursor/tasks/lab-design/**`. Those files still say `streamlit run`. They are history.
- Rewriting `docs/TEST_REPORT.md`. The line `streamlit 1.65.0` is the environment of the 2026-10-10 close-out, not a launch instruction.
- ML sections of the spec that do not name the UI (detector, splits, tracking, KPI definitions, M0–M4 acceptance of the model).
