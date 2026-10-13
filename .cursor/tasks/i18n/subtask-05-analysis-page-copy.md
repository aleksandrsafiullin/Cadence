# Subtask 5: Analysis page copy

## Goal

Bind the Analysis page to the active catalog. `render` and every `on_click` / `on_change` resolve one catalog and use it for every user-visible string, including labels that already live in `copy.py`. Store `analysis_mode` as `baseline` / `custom` and `analysis_time` as the source timestamp so a locale switch does not raise or seek a different frame. Default `ru` still produces `1 из 2, все классы`.

## Context

Spec §15 «Анализ»: encoded MP4, one diagnostic frame, missing source time and observed share are «не рассчитано», not zero. `tests/unit/test_ui_analysis_state.py` locks `mixed.text == "1 из 2, все классы"`, `PREDICTED_HEADING == "predicted"`, and `set_value(ui_copy.MODE_CUSTOM)` on `analysis_mode`.

`_ensure_defaults` writes `analysis_mode` as the Russian label `Базовая`. The selectbox options are `MODE_BASELINE` and `MODE_CUSTOM`. After the switch, those strings are not in the English option list and Streamlit raises. Store stable ids. That requires one test edit: `set_value("custom")` instead of `set_value(ui_copy.MODE_CUSTOM)`. Do not delete the honest-empty assertions in that file.

The weights warning on this page is the same catalog sentence as subtask 3. `Официальные веса` must remain in the Russian sentence.

## Acceptance Criteria

- [ ] `analysis_page.render` and its `on_click` / `on_change` callbacks each resolve one catalog and use it for every label, button, caption, help string, and toggle the page authors. That includes existing keys such as `ANALYSIS_START`, `ANALYSIS_EXPORT`, `BASELINE_CAPTION`, and `THRESHOLD_NEW_RUN`. A widget that still reads a module-level `ui_copy` sentence is not done.
- [ ] Zone and form literals move unchanged into both catalogs: coordinate out of range (file not written, and preview not built), three vertices, invalid view id, thresholds not set, no run so MP4 is not encoded, `Список рамок не строится.`, `нет наблюдаемых кадров из {total}`, `{observed} из {total}`.
- [ ] `с` suffixes in the analysis time formatters use the catalog. Russian stays `с`. English uses `s`. Missing time stays the catalog not-calculated phrase. A real `0` second value stays `0` where the code already treats 0 as measured (`video_page` offset). Do not turn a measured 0 into not-calculated.
- [ ] Observed-share text at `ru` is still exactly `1 из 2, все классы` for that fixture, and `нет наблюдаемых кадров из {total}, {label}` when nothing was observed. Labels `ковш` and `все классы` come from the catalog (`OBSERVED_BUCKET`, `OBSERVED_ALL`).
- [ ] Session `analysis_mode` is `baseline` or `custom`. The selectbox shows the catalog labels (`Базовая` / `Своя` in Russian, real English such as `Baseline` / `Custom`). Default is `baseline`.
- [ ] `_selected_time` stores the source timestamp in `analysis_time`, not the formatted label (`12.5 с` / `12.5 s`). Options are the timestamps. `format_func` applies the catalog suffix `с` or `s`. The guard replaces the session value only when that timestamp is absent from the run. A language change must not assign the first label and seek a different frame. The empty-list caption `Список рамок не строится.` still comes from the active catalog.
- [ ] `tests/unit/test_ui_analysis_state.py` sets `custom` on that widget. The rest of the file’s Russian empty-state checks stay.
- [ ] `predicted` is still the heading in both locales.
- [ ] Detections and predicted dataframes keep English column names from the parquet/schema. Do not translate those keys.
- [ ] Page-authored `LabError` messages in this file use the catalog. `exc.message` from tracking/media stays the service text.
- [ ] No emoji. No second job. Threshold edits still do not rewrite a finished run.

## Implementation Notes

File: `src/excavation_lab/ui/analysis_page.py` and `copy.py`.

Resolve once per `render` and once per callback. Do not leave module-level `ui_copy` sentences on the widget path.

`_ensure_defaults` must set `"analysis_mode": "baseline"`, not the Russian label. Comparisons use `== "custom"` / `== "baseline"`, not `ui_copy.MODE_CUSTOM`.

`_selected_time` today does `labels = [_format_time(stamp) ...]` and, when `analysis_time` is missing from `labels`, assigns `labels[0]`. After this subtask the membership test is against the timestamps. `format_func` is display-only. Keep the selected timestamp’s identity across locales.

Keep `format_func` so `analysis_mode` stores the id. `tests/unit/test_ui_training_state.py` is not this widget; do not change `train_variant` (`Nano` / `Small` stay). Subtask 7 covers the timestamp switch.

The observed-share helper takes an optional catalog and defaults to Russian, same pattern as subtask 3. The unit assertion calls it without a catalog.

Zone vertex columns stay `x` and `y` (already invariant). `zone` is the data key; its visible header may use `ZONE_NAME`.

Do not edit `charts.py` here.

## Dependencies

Subtask 4.

## Verification

```bash
uv run pytest tests/unit/test_ui_analysis_state.py tests/unit/test_ui_empty_states.py -q
uv run ruff check src/excavation_lab/ui/analysis_page.py src/excavation_lab/ui/copy.py tests/unit/test_ui_analysis_state.py
```
