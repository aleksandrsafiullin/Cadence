# Subtask 8: Browser pass

## Goal

Exercise the locale switch in the running Streamlit app: Russian by default, English after the switch and after reload, all six pages, then back to Russian.

## Context

Spec §15 pages and §22 localhost. Project rules require the changed screens to be exercised, not only pytest. `train --profile smoke` is not part of this check and must not be started. One GPU/MPS job at a time stays true because this subtask starts no job.

The app binds localhost via `.streamlit/config.toml`. Do not change the bind address, telemetry, or auth.

## Acceptance Criteria

- [ ] `uv run pytest tests -q` and `uv run ruff check src tests` are already green from subtask 7.
- [ ] With no `ui/locale.txt` in the active workspace, the first paint is the current Russian UI: titles `Видео`, `Данные и разметка`, `Обучение`, `Анализ`, `Качество`, `События`, and «не рассчитано» where a measurement is missing.
- [ ] APPEARANCE shows Light | Dark, then the language control. The options read `Русский` and `English`. The caption says the control changes only the interface language, not the data, the model, or the measurements.
- [ ] Choosing `English` reruns into English: the six titles, empty states, and `not calculated`. The weights warning, if it is on screen, is English. Events is still an unavailable section, not a KPI dashboard.
- [ ] Reload keeps English. `ui/locale.txt` in that workspace is `en` and is not a symlink out of the workspace.
- [ ] Open all six pages in English. Buttons, headers, empty states, leads, and table headers are English. Class ids, `Light`, `Dark`, kickers, `Excavation Video Lab`, the switch endonym `Русский`, the chart field `класс`, and `not evaluated` stay as they are. No emoji. No zero standing in for a missing metric. An empty project picker shows the English no-project sentence and does not raise.
- [ ] Switch back to `Русский`. The six pages match the pre-switch Russian copy. `locale.txt` is `ru`.
- [ ] No train, analyze, or export job was started. No pilot MP4 was downloaded.

## Implementation Notes

```bash
uv run streamlit run app.py
```

Use the workspace the app already uses. If a previous session left `workspace/ui/locale.txt`, move it aside before the default-Russian check and restore it afterward if it was the developer’s own preference. Do not commit `workspace/`.

If the browser tools are unavailable, walk the same flow with `AppTest` against a temp workspace and say that the live browser was not opened. Subtask 7 already covers the temp-workspace switch; this subtask is the live screen.

Look at Video, Data and labeling, Training, Analysis, Quality, and Events. On Training, confirm the loss chart is absent until a real log exists. On Quality, confirm IDF1/HOTA say `not evaluated` and `not calculated` (or the Russian pair). On Events, confirm the v0.2 caption and twelve uncomputed indicators.

Russian success text from `format_sample_summary`, `format_import_summary`, and `format_export_summary` is out of scope, same as a service `LabError` from datasets, media, or jobs. Anything else that is still Russian — a button, header, empty state, or table header — is a failed switch. Do not click sample, import, or export during this pass.

## Dependencies

Subtask 7.

## Verification

```bash
uv run pytest tests -q
uv run ruff check src tests
uv run streamlit run app.py
```

Do not run `uv run excavation train --profile smoke`.
