# Subtask 10: Charts without Streamlit

## Goal

Draw the training loss line and the quality AP50 bars from the Altair charts that already exist. No Streamlit chart call, no Vega runtime, no Plotly.

## Context

`src/excavation_lab/ui/charts.py` builds `loss_chart` and `ap50_chart` with Altair. It does not import Streamlit. `tests/unit/test_ui_charts.py` locks padding 56/16/48/16, a hidden legend, `--pdm-font` at 12px, token colors, no named theme, no forecast band, and no event series. Loss uses `--pdm-series-observed`. AP50 uses `--pdm-accent` for every bar. The axis field for AP50 stays `класс` in both locales.

Subtasks 7 and 9 leave the gated `loss_chart` and `ap50_chart` calls in the page modules. This subtask moves them. Analysis does not get a chart.

Altair is installed because Streamlit depends on it. It is not in `pyproject.toml` `dependencies`. This subtask adds it for real, before subtask 11 removes Streamlit.

## Acceptance Criteria

- [ ] `uv add altair` updates `pyproject.toml` and `uv.lock`. No `plotly` and no `vl-convert`.
- [ ] Training JSON from `GET /jobs/{job_id}/progress` sets `chart` to `loss_chart(...).to_dict()` only when `should_draw_chart` is true. Otherwise `chart` is `null`. A header-only log and a two-epoch ETA case still have `chart: null`. The dict is the Altair spec, not a second geometry.
- [ ] Quality page embeds `ap50_chart(...).to_dict()` in a `<script type="application/json" id="ap50-chart">` only when the existing gate draws a bar. A missing score omits that bar. An all-missing series omits the spec. The page does not call `st.altair_chart`.
- [ ] `static/chart.js` reads those specs and draws one SVG line or one SVG bar series from `data.values`, using the colors already in the spec. The loss mark reads the fields `epoch` and `loss`. The AP50 mark reads `класс` and `AP50`. It does not fetch a script. It does not implement a general Vega renderer. Unknown mark types draw nothing.
- [ ] Retarget `test_pages_call_the_styler_only_behind_the_existing_gates`. `should_draw_chart` wraps `loss_chart` on `GET /jobs/{job_id}/progress`. The missing-score branch wraps `ap50_chart` on the quality page. Neither site calls `st.altair_chart`. The false branches still do not call the styler.
- [ ] `tests/unit/test_ui_charts.py` still passes the padding, token, and theme checks, and the ast check that `charts.py` and `tokens.py` do not import Streamlit. `test_vega_container_uses_surface_tokens` still reads `streamlit_chrome.css` until subtask 11 deletes that file. Do not retarget it here.
- [ ] If this subtask removes the `charts` import from `training_page.py` or `quality_page.py`, update `test_only_training_and_quality_import_charts` in this same subtask to the new call sites (the progress handler and the quality page). Do not leave an assertion that those two modules import `charts` after the import is gone. `visualization/` still does not import `charts.py`. Analysis and events still do not.
- [ ] `tests/unit/test_web_chart.py` builds `loss_chart` and `ap50_chart` and asserts the spec data fields are `epoch`/`loss` and `класс`/`AP50`. It also asserts `chart.js` reads those four names from `data.values`. A progress fixture that `should_draw_chart` accepts puts those epoch values and the observed color in `chart`. A quality fixture with no scores has no `ap50-chart` spec. This test is the proof. Subtask 12's empty workspace does not draw a curve.
- [ ] Events and analysis responses still contain no chart spec.

## Implementation Notes

Disable Altair's theme only inside the existing styler. Do not set a theme at import.

`chart.js` is plain script loaded from `base.html` or from the two pages. Training poll: when `chart` is non-null, replace the SVG in `#loss-chart`. Do not poll quality; that page is a full render.

Do not add hex colors in `chart.js`. Read the spec.

Keep `tokens.theme_custom_properties` as the color source inside `charts.py`. The browser does not parse `lab_theme.css` a second time.

## Dependencies

Subtasks 7 and 9. The progress route and `#ap50-chart` mount exist.

## Verification

```bash
uv run pytest tests/unit/test_ui_charts.py tests/unit/test_web_chart.py tests/unit/test_ui_training_state.py tests/unit/test_ui_quality_state.py -q
uv run ruff check src tests
```

No smoke training. The chart unit test is the check. Do not wait for the browser pass to prove `epoch`/`loss` or `класс`/`AP50`.
