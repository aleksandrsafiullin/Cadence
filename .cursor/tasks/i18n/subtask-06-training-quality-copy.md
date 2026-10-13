# Subtask 6: Training and Quality display copy

## Goal

Bind Training and Quality to the active catalog. `render` and every `on_click` / `on_change` resolve one catalog and use it for every user-visible string, including labels that already live in `copy.py`. Missing quality numbers render in the active language. `quality.py` still builds Russian reports. The Altair field name `класс` stays.

## Context

Spec §15 «Обучение» and «Качество», spec §18 and §21. No loss curve without log rows. No ETA until several steps are measured. IDF1 and HOTA stay `not_evaluated` / `not evaluated`. Pilot targets are captions, not measured results. Send-to-labeling is train-only.

`training_page.py`’s only runtime Russian outside the catalog is the ETA suffix `с` in `format_eta`. Profiles, devices, and `Nano` / `Small` are already stable ids. `tests/unit/test_ui_training_state.py` expects `format_eta(...) == ui_copy.NOT_CALCULATED` and `train_variant` values `Nano` and `Small`.

`quality_page.py` displays `models.quality.metric_text`, which returns the Russian token for nulls, and shows `report["target_caption"]` and error prose that `quality.py` built in Russian. The page must display the active catalog for those known sentences. It must not pass the browser locale into `quality.py`, and it must not write `not calculated` back into the report or `metrics.json`.

`charts.ap50_chart` reads the data key `класс`. `tests/unit/test_ui_charts.py` locks mark type, accent color, padding (`56, 16, 48, 16`), font size 12, and the absence of a chart-level `title`. Do not rename `класс`, `AP50`, `epoch`, or `loss`.

## Acceptance Criteria

- [ ] `training_page.render`, `quality_page.render`, and their `on_click` / `on_change` callbacks each resolve one catalog and use it for every label, button, caption, help string, and metric name the page authors. That includes existing keys such as `TRAIN_START`, `TRAINING_EMPTY`, `SEND_TO_LABELING`, and `QUALITY_EMPTY`. A widget that still reads a module-level `ui_copy` sentence is not done.
- [ ] `format_eta` without a catalog still returns `ui_copy.NOT_CALCULATED` or the current `{number} с` form. `render` passes the resolved catalog. At `ru`, the rendered ETA equals that no-argument result. English uses `s` and `not calculated`. A missing ETA is never `0`.
- [ ] Training widget values stay `smoke`, `pilot-small`, `auto`, `cuda`, `mps`, `cpu`, `Nano`, `Small`. Their surrounding labels (`TRAIN_VARIANT`, `TRAIN_PROFILE`, `TRAIN_DEVICE`, Advanced fields) come from the resolved catalog.
- [ ] `train_dataset` keeps the reset that assigns `options[0]` when the stored value is not in `options`. The empty placeholder is the active catalog’s `TRAIN_DATASET_EMPTY` (`нет снимка` in Russian). Do not remove that reset when the placeholder is translated. Dataset ids stay the stored values when a snapshot exists.
- [ ] Quality metrics that are null, non-finite, or the Russian token `не рассчитано` render as the active catalog’s not-calculated phrase. Numeric formatting stays `format(value, "g")` so `metric_text(0.91)` comparisons at locale `ru` still hold.
- [ ] The quality page shows `TARGET_CAPTION`, `ERRORS_UNAVAILABLE`, and `ERRORS_FROM_MATCHES` from the active catalog, not the Russian strings stored on the report. At `ru` those catalog strings still equal today’s module-level constants.
- [ ] IDF1 and HOTA display `not evaluated` and the not-calculated phrase. They do not show a number from `metrics.json`.
- [ ] Class and metric **labels** use the resolved catalog. Class **ids** stay the keys. The AP50 chart still receives rows keyed by `класс`. Do not rename that encoding, or `AP50`, `epoch`, or `loss`.
- [ ] An Altair axis title is allowed only if `tests/unit/test_ui_charts.py` stays green with the same padding, colors, and no chart-level `title`. If the title moves any of those locks, revert it. The accepted outcome is a docstring in `charts.py`: both locales keep the field name `класс` as the axis text. Do not rename the encoding to get an English axis.
- [ ] Train-only labeling copy (`SEND_TO_LABELING`, `not_train_frame_message`, sent-to-queue) uses the active catalog on the page. `quality.py` still raises the Russian `LabError`. Where the page currently prints `exc.message` for a code that already has a catalog function, display the catalog text for the active locale and keep the code. Do not translate unrelated service errors.
- [ ] No fake metrics, no emoji, no smoke run.

## Implementation Notes

Files: `src/excavation_lab/ui/training_page.py`, `src/excavation_lab/ui/quality_page.py`, `copy.py`. Touch `src/excavation_lab/ui/charts.py` only to add a translated axis title that keeps the `класс` encoding, or, when that title breaks `tests/unit/test_ui_charts.py`, a docstring that says the axis text stays `класс`. Do not change `ap50_chart`’s required arguments. Existing `ap50_chart(rows, theme)` calls must keep the current spec when no title is passed.

Resolve once per `render` and once per callback. Do not leave module-level `ui_copy` sentences on the widget path. At `ru`, button labels still equal those constants.

`metric_text` in `models/quality.py` stays Russian. The page can call it for numbers, then if the result equals the Russian `не рассчитано` (or the value is `None`), substitute the active catalog phrase. Do not substring-replace inside longer service messages.

`test_ui_quality_state.py` builds labels with `ui_copy.CLASS_LABELS` and `ui_copy.METRIC_LABELS`. At default `ru` the page must use those same Russian strings. That happens naturally if the resolver returns the Russian catalog.

Dataframe header `класс` on the quality chart input is the chart key. Do not rename it to `class` or the chart helper and `test_ui_charts.py` fixtures break.

## Dependencies

Subtask 5.

## Verification

```bash
uv run pytest tests/unit/test_ui_training_state.py tests/unit/test_ui_quality_state.py tests/unit/test_ui_charts.py tests/unit/test_quality_queue.py -q
uv run ruff check src/excavation_lab/ui/training_page.py src/excavation_lab/ui/quality_page.py src/excavation_lab/ui/charts.py src/excavation_lab/ui/copy.py src/excavation_lab/models/quality.py
```
