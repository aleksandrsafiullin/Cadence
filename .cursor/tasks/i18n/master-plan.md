# Russian and English UI locales

## Overview

Add a real English locale beside the current Russian Streamlit UI. Russian stays the default. A workspace with no `ui/locale.txt` shows the copy that is on screen today. English is a second catalog with the same keys. Each page `render` and its callbacks use that catalog for every visible string, including buttons that already live in `copy.py`. An English screen that still shows Russian buttons is not done.

This is an extension of spec §15, §19, and §22. The spec says the interface is Russian. This slice does not delete that default and does not edit the spec. v0.2 events stay visible and explicitly unavailable in both languages. Missing measurements stay «не рассчитано» in Russian and `not calculated` in English. Never `0`, never a demo number.

Do not edit `.cursor/tasks/master-plan.md`, `.cursor/tasks/subtask-01` through `subtask-46`, or `.cursor/tasks/lab-design/`. Do not edit `.cursor/rules/` or `EXCAVATION_VIDEO_LAB_SPEC.md`. Do not import `src/pdm`. Do not rename `--pdm-*`.

Media, splits, training, tracking, and the worker lock are not reopened. No smoke training. No second web framework, no gettext, no new package. Catalogs are Python objects in `src/excavation_lab/ui/copy.py`. The resolver lives in `src/excavation_lab/ui/locale.py` and may import Streamlit. `copy.py` must not.

## Conflicts resolved

**Spec and rules say the UI is Russian.** Russian remains the default locale and the module-level names in `copy.py`. English is additive and selectable. `docs/DECISIONS.md` records that. The rules files stay as they are: they describe the default, and this plan is the exception for a second locale.

**`docs/lab_design_code.md` invariants.** These strings are identical in `ru` and `en`: product title `Excavation Video Lab`; theme options `Light` and `Dark`; sidebar kickers `PROJECT`, `WORKFLOW`, `APPEARANCE`. The switcher options are the literals `Русский` and `English` in both locales (endonyms, not translations of each other). Job status **keys** stay `queued`, `running`, `completed`, `failed`, `cancel_requested`, `cancelled`. Class **ids** stay `excavator_body`, `bucket`, `haul_truck`, `truck_bed`. The stored token `not_evaluated` stays `not_evaluated`. The caption `not evaluated` stays those English words in both locales, because that is what the Russian UI already shows. `predicted` stays `predicted` (`tests/unit/test_ui_analysis_state.py` locks it). `split_group_id`, profile ids (`smoke`, `pilot-small`), device ids (`auto`, `cuda`, `mps`, `cpu`), and variant ids (`nano`, `small` displayed as `Nano` and `Small`) stay stable. Display labels for classes, pages, job statuses, and metrics translate. JSON keys, Parquet columns, paths, and `--pdm-*` do not.

**`models/quality.py` imports `excavation_lab.ui.copy`.** That import stays. It must keep calling the Russian catalog (`catalog("ru")` or the existing Russian aliases, which are the same object). It must not import `excavation_lab.ui.locale`. `locale.py` imports Streamlit, and a worker process must not follow `st.session_state` or `ui/locale.txt`. A job log does not flip language mid-run. The quality page may re-render a missing value with the active catalog; it does not write that English phrase back into `metrics.json` or the report dict.

**CLI is Russian and large.** `cli.py` Typer help, plus `hardware.russian_summary`, `hardware.weights_download_notice`, `analytics.unavailable_error`, and `LabError` text across datasets, models, media, and jobs, are a second pile of sentences. Translating only Typer lines would be a half-translated CLI. Subtask 9 is out of scope for this slice. Default CLI locale stays Russian so current CLI tests keep passing. No `--lang` flag. No second string system. A later slice may call `catalog("ru"|"en")` from `copy.py`. It must not import the resolver.

**Service `exc.message` shown on a page.** Strings the page authors, including ones that already live in `copy.py`, render from the active catalog. `LabError` text raised outside `ui/` stays Russian in this slice, for the same reason as the CLI. The same bucket holds three operator formatters the Data page already prints: `format_sample_summary` (`datasets/sampler.py`), `format_import_summary` (`datasets/coco_import.py`), and `format_export_summary` (`datasets/cvat_export.py`). They stay Russian. Two screen sentences are catalog text, because they are the main copy of a screen: the Events caption stops displaying `unavailable_error().message` (the page still calls `unavailable_error()`), and the weights warning on Video and Analysis is one catalog sentence. `hardware.weights_download_notice` stays Russian for `excavation doctor`. The Russian weights sentence must still contain `Официальные веса`.

## Phases

Detection, tracking, and export are already shipped. This plan does not redo them. Events stay the existing unavailable page.

1. **Catalog** — `ru` and `en` in `copy.py`, Russian module aliases unchanged, no Streamlit.
2. **Resolver and switch** — `ui/locale.txt`, session, APPEARANCE control, chrome (titles, leads, project picker) from the active catalog.
3. **Page binding** — each page `render` and its `on_click` / `on_change` callbacks resolve one catalog and use it for every user-visible string, including buttons that already live in `copy.py`. Helpers called with no catalog stay Russian. At `ru`, rendered text still equals the module-level constants the current tests assert. Chart data keys stay put.
4. **Tests** — default `ru` keeps the honest-empty Russian assertions. An English page fails if Cyrillic remains outside `Русский` and the chart field `класс`. `not evaluated` stays. The empty project picker and `analysis_time` survive a switch.
5. **Browser** — Russian by default, switch, reload, all six pages, switch back. No smoke training.

Subtask 9 is recorded and is not a phase.

## Dependencies

- Phase 2 depends on phase 1. The switch reads a catalog that already has both locales.
- Phase 3 depends on phase 2. Pages ask the resolver once per `render` or callback and use that object for every visible string. Helpers that tests call with no catalog argument keep returning Russian.
- Subtask 2 stores the empty project option as `no-project` before page bodies are translated, so the switch cannot crash an empty workspace on `project_picker`.
- Subtask 5 stores `analysis_time` as the source timestamp. `train_dataset` already resets when the stored value is missing from the options; that reset stays.
- Subtasks 3–6 all edit `copy.py`. Run them in order, not in parallel.
- Phase 4 depends on phases 1–3. It locks behavior. It does not add a new string system.
- Phase 5 depends on phase 4. Pytest is green before the browser pass.
- No dependency on a pilot MP4, a checkpoint, or `train --profile smoke`.
- The theme control stays `segmented_control[0]`. `tests/unit/test_theme_persistence.py` indexes that slot. The locale control is the next control in the same group.

## Execution order

| Order | File | Goal | Depends on |
| --- | --- | --- | --- |
| 1 | `subtask-01-locale-catalogs.md` | Streamlit-free `ru`/`en` catalogs; Russian aliases stay | — |
| 2 | `subtask-02-resolver-and-switch.md` | Persist `locale.txt`, switch in APPEARANCE, translate chrome, stable empty project id | 1 |
| 3 | `subtask-03-video-events-copy.md` | Bind Video and Events to one active catalog, including existing `copy.py` strings | 2 |
| 4 | `subtask-04-data-page-copy.md` | Bind Data to one active catalog; three service formatters stay Russian | 3 |
| 5 | `subtask-05-analysis-page-copy.md` | Bind Analysis; `baseline`/`custom` and source-time selection | 4 |
| 6 | `subtask-06-training-quality-copy.md` | Bind Training and Quality; keep the `класс` encoding | 5 |
| 7 | `subtask-07-locale-tests.md` | English pages fail on leftover Cyrillic; locale file and widget-switch tests | 6 |
| 8 | `subtask-08-browser-pass.md` | Exercise default Russian, English, reload, all six pages | 7 |
| — | `subtask-09-cli-locale-out-of-scope.md` | Do not implement. CLI stays Russian | — |

## Language-invariant strings

| Surface | Stays |
| --- | --- |
| Product title | `Excavation Video Lab` |
| Theme options | `Light`, `Dark` |
| Kickers | `PROJECT`, `WORKFLOW`, `APPEARANCE` |
| Locale options | `Русский`, `English` |
| Job status keys | `queued`, `running`, `completed`, `failed`, `cancel_requested`, `cancelled` |
| Class ids | `excavator_body`, `bucket`, `haul_truck`, `truck_bed` |
| Stored tokens | `not_evaluated`, and the caption `not evaluated` |
| State word | `predicted` |
| Chart data keys | `класс`, `AP50`, `epoch`, `loss` |
| CSS | `--pdm-*` unchanged |

Page ids used in code, not shown as the only label: `video`, `data`, `training`, `analysis`, `quality`, `events`.

English titles, locked so later subtasks do not invent a second naming:

| Id | Russian (unchanged) | English |
| --- | --- | --- |
| `video` | Видео | Video |
| `data` | Данные и разметка | Data and labeling |
| `training` | Обучение | Training |
| `analysis` | Анализ | Analysis |
| `quality` | Качество | Quality |
| `events` | События | Events |

English class labels: `excavator body`, `bucket`, `haul truck`, `truck bed`. The missing-measurement phrase is exactly `not calculated`.

## Verification

```bash
uv run pytest tests -q
uv run ruff check src tests
uv run streamlit run app.py
```

Browser, after subtask 8: no preference file means Russian; switch to English; reload and English remains; open all six pages; switch back to Russian. Events stays unavailable. Empty metrics say `not calculated` in English and «не рассчитано» in Russian. Do not run `train --profile smoke`.

## Out of scope

- CLI `--lang`, Typer help, and service-layer Russian (`cli.py`, `hardware.russian_summary`, dataset/model/media `LabError` text, `format_sample_summary`, `format_import_summary`, `format_export_summary`). See subtask 9.
- Rewriting existing Russian sentences. Move them unchanged.
- Deleting or weakening honest-empty assertions in `tests/unit/test_ui_empty_states.py`, `test_events_unavailable.py`, `test_ui_video_page.py`, `test_ui_data_freeze.py`, `test_ui_training_state.py`, `test_ui_analysis_state.py`, `test_ui_quality_state.py`, `test_ui_theme.py`, `test_theme_persistence.py`, `test_quality_queue.py`.
- Renaming the `класс` encoding. A translated Altair axis title ships only if `tests/unit/test_ui_charts.py` stays green; otherwise `charts.py` documents that the axis text stays `класс`.
- v0.2 event logic, KPI math, or a second worker.
