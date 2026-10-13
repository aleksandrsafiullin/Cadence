# Subtask 4: Data page copy

## Goal

Bind the Data page to the active catalog. `render` and every `on_click` / `on_change` resolve one catalog and use it for every user-visible string, including freeze, sample, and CVAT controls that already live in `copy.py`. Default `ru` keeps the current sentences, including `ep-form: 0–10 с` and the guard lines the freeze test compares.

## Context

Spec §15 «Данные и разметка» and §8: the freeze button, guard intervals, and the locked snapshot stay honest. `tests/unit/test_ui_data_freeze.py` compares `data_page.guard_lines(proposal)` to the rendered page and expects `ep-form: 0–10 с`. Those helpers must stay Russian when called with no catalog.

The sample-plan selectbox stores Russian labels (`диагностический`, `содержательный`, `расширенный`) and maps them to `diagnostic`, `content`, `expanded`. A locale switch would leave the old label in session state and Streamlit would reject it. Store the English ids. Display the catalog label.

## Acceptance Criteria

- [ ] `data_page.render` and its `on_click` / `on_change` callbacks each resolve one catalog and use it for every label, button, caption, help string, and table header the page authors. That includes existing keys such as `FREEZE_SPLIT`, `PROPOSE_SPLIT`, `SNAPSHOT_DATASET`, and `DATA_EMPTY`. A widget that still reads a module-level `ui_copy` sentence is not done.
- [ ] These literals live in both catalogs, Russian unchanged: guard prefix `Защитный интервал:`, `Файл:`, plan-count lines (`Диагностический:` / `Содержательный:`), the episode boundary error `У эпизода нет числовой границы.`, and the `с` suffix in episode, guard, and block lines.
- [ ] `guard_lines`, `episode_lines`, `block_lines`, `plan_count_text`, and `episode_table` default to the Russian catalog when called with no catalog. `render` passes the resolved catalog. At `ru`, rendered guards equal `guard_lines(document)` with no extra argument, and rendered buttons equal the module-level constants.
- [ ] The dataframe shows translated headers for `эпизод`, `начало, с`, `конец, с`, and `метка`. `split_group_id` and `split` stay those English tokens in both locales.
- [ ] Sample-plan options stored in session are `diagnostic`, `content`, `expanded`. The visible labels are the catalog strings. `format_func` (or an equivalent that does not store the translated label) is required.
- [ ] Review counts still render `COUNT_*: «не рассчитано».` at locale `ru`, never `0`. English uses `not calculated` for those same missing counts.
- [ ] Snapshot lock reasons stay the catalog sentences (`SNAPSHOT_LOCKED`, `SNAPSHOT_READY`). Validation issues that arrive as `exc.message` from `datasets/` stay the service text.
- [ ] These three formatters stay Russian service text, same as dataset `LabError`s. Do not translate them and do not route them through the catalog: `format_sample_summary` in `src/excavation_lab/datasets/sampler.py`, `format_import_summary` in `src/excavation_lab/datasets/coco_import.py`, and `format_export_summary` in `src/excavation_lab/datasets/cvat_export.py`. The Data page may keep displaying their return values after sample, import, and CVAT export.
- [ ] No emoji.

## Implementation Notes

File: `src/excavation_lab/ui/data_page.py` and `copy.py`.

Prefer stable row keys (`episode_id`, `start_s`, `end_s`, `label`, `split_group_id`, `split`) and `column_config` labels from the catalog, if `episode_lines` is updated to those keys. If that risks the `ep-form: 0–10 с` line, keep the current row shape for the Russian default and only change the visible header. The line format is what the test locks, not the dict key. The `с` in that line comes from the formatter and must remain for `ru`.

`_PLANS` should become `(id,)` with labels from the catalog. First option stays `diagnostic` so a fresh page still proposes the same plan as today.

`assigned_split` returns the catalog not-calculated phrase when the episode is missing or ambiguous, not `0`.

Resolve once per `render` and once per callback. Do not leave `ui_copy.FREEZE_SPLIT` (or the other module-level sentences) on the widget path.

Do not translate `LabError` messages raised in `datasets/splits.py`, `datasets/validate.py`, or the sampler. The same-scene Russian sentence from the split service stays Russian. `test_ui_data_freeze.py` also asserts the English word `transfer` is absent on the Russian page. Do not put `transfer` into the Russian catalog.

`format_sample_summary`, `format_import_summary`, and `format_export_summary` are the operator notes those buttons already show (`ActionNote` around the three call sites in `data_page.py`). They are service text. Subtask 9 records them as out of scope. A Russian success note after those actions is expected. A Russian button or header is not.

## Dependencies

Subtask 3.

## Verification

```bash
uv run pytest tests/unit/test_ui_data_freeze.py tests/unit/test_ui_empty_states.py -q
uv run ruff check src/excavation_lab/ui/data_page.py src/excavation_lab/ui/copy.py
```
