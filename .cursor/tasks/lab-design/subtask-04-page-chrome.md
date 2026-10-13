# Subtask 4: Page chrome

## Goal

Give each screen one primary action and render job status as a left stripe, without changing what the buttons do.

## Context

Spec §15 defines the six screens. Spec §16 defines the job states. `JOB_STATUS_LABELS` in `src/excavation_lab/ui/copy.py` is the Russian text. `JobStatus` in `src/excavation_lab/schemas.py` stays `queued`, `running`, `completed`, `failed`, `cancel_requested`, `cancelled`.

Empty states already use `st.info`, `st.warning`, `st.error`, and `st.success` with `icon=""`. Keep those widget calls. Subtask 1's alert CSS supplies the stripe. Do not replace them with HTML or the empty-state and quality tests lose `app.info` / `app.error`.

Job lines are plain `st.text` today: `job_status_line` on «Видео» and «Анализ», and `format_status` on «Обучение». Those become stripes. The sentence builders stay.

## Acceptance Criteria

- [ ] At most one `type="primary"` button is visible on a screen. The map below is the whole set. Keys and `on_click` handlers do not change. Disabled states do not change.
- [ ] «Качество» send buttons stay `type="secondary"`. An empty quality page still has zero buttons.
- [ ] «События» gains no button, no chart, and no numeric KPI. `EVENTS_UNAVAILABLE` and the twelve «не рассчитано» metrics stay.
- [ ] A job status is a `lab-alert` row: neutral surface, 3px left border, text `--pdm-text`. Tone follows the table below. The row is not a filled red, green, or yellow card.
- [ ] Unknown status still renders «не рассчитано». It does not render `0`. `status_tone(None)` and any string outside the six excavation statuses return `neutral`. `queued` and `running` stay `accent`.
- [ ] `job_status_line` and `format_status` keep their current Russian sentences, including orphaned and partial suffixes.
- [ ] No emoji is added. Existing `icon=""` arguments stay.

## Implementation Notes

Primary map. Set `type="primary"` only on these. Every other `st.button` on these pages stays `type="secondary"`.

| Page | Primary | When | Other buttons |
| --- | --- | --- | --- |
| `video_page.py` | `add_video` («Добавить видео») | No registered video, so «Запустить базовую модель» is absent | `add_upload` secondary |
| `video_page.py` | `baseline` («Запустить базовую модель») | A video is registered | `add_video`, `add_upload`, `probe_video`, `refresh_runs` secondary, including inside the expander |
| `data_page.py` | freeze key `data_freeze:{video_id}` («Зафиксировать разбиение») | Always the single primary when that button is shown | propose, add episode, both sample buttons, CVAT export, COCO import, snapshot secondary |
| `training_page.py` | `train_start` («Старт») | Always, including while disabled because a job is running | `train_cancel` secondary |
| `analysis_page.py` | `analysis_start` («Старт») | Always, including while disabled | `analysis_export` and `save_zones` secondary |
| `quality_page.py` | none | Repeated row action | every «Отправить в разметку» secondary |
| `events_page.py` | none | Page stays unavailable | no new controls |

A disabled primary keeps `type="primary"`. The chrome CSS fades it with opacity `.4`. Do not move the primary onto Cancel or Refresh while a job runs.

New `src/excavation_lab/ui/status.py` (Streamlit allowed):

- `status_tone(status: str | None) -> str` returns one of `accent`, `success`, `warning`, `danger`, `neutral`.
- `queued` and `running` → `accent`. Running is not success green.
- `completed` → `success`.
- `failed` → `danger`.
- `cancel_requested` → `warning`.
- `cancelled` → `neutral`.
- `None` or any string that is not one of those six → `neutral`. The caller still passes «не рассчитано» or the existing fallback from `JOB_STATUS_LABELS`. The words do not change. Unknown is not an accent stripe.

`render_status(text: str, tone: str)` emits `st.html` with `class="lab-alert"`, `data-tone` set to that tone, and `html.escape` on the text. No inline style and no bare `0` in the wrapper. A small `<span class="lab-status-mark">` is enough for the icon color. No emoji, no image, no external SVG file.

Replace:

- `st.text(job_status_line(job))` in `video_page.py` and `analysis_page.py` with `render_status(job_status_line(job), status_tone(job.status.value))`.
- The training status metric in `training_page.py` (`st.metric` of `TRAIN_STATUS`) with `render_status(format_status(status), status_tone(...))`. Epoch, checkpoint, and ETA stay metrics. Job rows that print `format_status` in `st.text` use the same stripe.

Do not change `jobs.py`, `schemas.py`, or the status strings. Do not add `stopping`, `stopped`, or `idle`.

Leave `st.info` / `st.warning` / `st.error` / `st.success` as they are, including `icon=""`. The stripe for those comes from `streamlit_chrome.css`. If a Streamlit alert still paints a solid tone after subtask 1, fix the selector in the chrome file. Do not add hex.

Dataframes, expanders, and checkboxes need no Python changes if subtask 1's selectors match. If a control is still on the default Streamlit blue, fix the selector. Do not restyle `visualization/annotate.py`.

## Dependencies

Subtasks 1 and 3.

## Verification

```bash
uv run pytest tests/unit/test_ui_video_page.py tests/unit/test_ui_data_freeze.py tests/unit/test_ui_training_state.py tests/unit/test_ui_analysis_state.py tests/unit/test_ui_quality_state.py tests/unit/test_ui_empty_states.py tests/unit/test_events_unavailable.py -q
uv run pytest tests -q
uv run ruff check src tests
uv run streamlit run app.py
```

On each page, confirm the primary map and that empty copy is unchanged. If a workspace already has a job row, confirm the stripe color matches the tone table and the Russian label is one of «в очереди», «выполняется», «завершен», «ошибка», «запрошена отмена», «отменен». Do not start a training job to produce that row. Stop the server.
