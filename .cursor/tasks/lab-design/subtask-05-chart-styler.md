# Subtask 5: Chart styler

## Goal

Replace the two existing Streamlit charts with one token-driven styler. Do not add a chart anywhere else.

## Context

`training_page.py` calls `st.line_chart` only when `should_draw_chart` is true. That series is measured training loss. `quality_page.py` calls `st.bar_chart` only when `ap50_chart_frame` returns rows. Missing and all-zero AP50 already return `None`. Spec §15 forbids a loss curve before real log rows and forbids invented metrics (spec §19 item 16).

`docs/lab_design_code.md` wants one styler, `template="none"`, and explicit series roles. Plotly is not a project dependency (`uv.lock` has Altair because Streamlit uses it, and it has no `plotly`). Do not add Plotly to `pyproject.toml`.

## Acceptance Criteria

- [ ] `src/excavation_lab/ui/charts.py` is the only place that builds these figures. `training_page.py` and `quality_page.py` call it. No other page imports it.
- [ ] The loss chart encodes `--pdm-series-observed` for the active theme: solid, 2px. It has no forecast line, no band, and no second color.
- [ ] The AP50 chart encodes `--pdm-accent` for every bar. It does not use `--pdm-series-forecast`, `--pdm-series-3`, or `--pdm-success`. Four classes are categories of one measurement, not six equal channels.
- [ ] Neither chart uses a named Altair or Plotly theme. The Vega-Lite spec sets paper and plot background to `--pdm-chart-bg`, font `--pdm-font` at 12px, text `--pdm-chart-text`, X axis line `--pdm-chart-axis` with no X grid, Y axis line and Y grid `--pdm-chart-grid`.
- [ ] Margins are left 56, right 16, bottom 48, top 16. The title stays in the page section above the chart, not inside the figure. Legend is hidden for these two single-series charts.
- [ ] Hover uses `--pdm-surface`, `--pdm-border`, and `--pdm-text` at 12px.
- [ ] Empty training still shows `TRAINING_EMPTY` and does not call the styler. `ap50_chart_frame` still returns `None` for a missing report and for an all-zero series. Events still have no chart.
- [ ] `charts.py` and `tokens.py` do not import Streamlit. `tests/unit/test_ui_charts.py` parses both files with `ast` and fails if either imports `streamlit`. `tests/unit/test_no_streamlit_import.py` allows every module under `ui/`, so it will not catch this. The page passes `st.altair_chart` the object the styler returns. `visualization/` does not import `charts.py`.

## Implementation Notes

Read colors through `tokens.py` from subtask 2. Pass the active theme name (`dark` or `light`) into the styler. Hex in the chart spec is the resolved token value, not a new literal copied by hand. If the parser misses a token, fail the chart call visibly in the test. Do not fall back to a Vega or Plotly default palette.

Altair has no `template="none"`. Document that in the module docstring in one sentence: this chart sets no named theme, which is how this app applies the design rule against `plotly`, `plotly_dark`, and `ggplot2`. Disable Altair's theme inside the styler function, for the chart being built, and restore the previous theme before returning. Do not disable it at import. Other Altair use in the process keeps its own theme.

Layout from the design doc:

- Axis and legend font 12px, `--pdm-chart-text`.
- X grid off, Y grid on, both axis lines 1px.
- Legend hidden here. If a later chart has a legend, the same module puts it horizontal at the top left, with no frame and no fill, and adds 32px to the top margin. Do not build that chart now.
- The Streamlit chart container is already covered by subtask 1 if the selector matches `[data-testid="stVegaLiteChart"]` or the installed equivalent: surface, 1px border, radius 12. Add that selector to `streamlit_chrome.css` if it is missing. No hex.

Keep `should_draw_chart` and `ap50_chart_frame` as the gates. The styler receives rows that have already passed those gates. A unit test may call the styler with a synthetic loss series and a synthetic AP50 frame. That test data must not be rendered by the page.

`tests/unit/test_ui_charts.py` reads `chart.to_dict()` and compares colors to the token parser, not to hex typed into the test. Assert both themes.

- Loss mark color equals parsed `--pdm-series-observed` (`#A1A1A6` dark, `#6E6E73` light).
- AP50 mark color equals parsed `--pdm-accent` (`#0A84FF` dark, `#0071E3` light). It is not the forecast token.
- The spec has no `plotly`, `plotly_dark`, or `ggplot2` string.
- X grid is false and Y grid is true.
- `padding` is left 56, right 16, bottom 48, top 16.
- `config.legend.disable` is true. These two charts have no legend block.
- `config.font` equals parsed `--pdm-font`. `config.axis.labelFontSize` is 12. `config.axis.labelColor` equals parsed `--pdm-chart-text`.
- Hover colors are in the compiled spec: background parsed `--pdm-surface`, border parsed `--pdm-border`, text parsed `--pdm-text`, font size 12. Put them on a key `to_dict()` keeps (`config` if Altair retains it, otherwise `usermeta.hover` with those four fields). The test fails if the fields are missing. The browser pass does not open a chart, so this test is the layout lock.
- `ast` walk of `charts.py` and `tokens.py` finds no `streamlit` import.
- Building a chart does not import or draw inside `events_page`.

Do not add a heatmap, a forecast series, or a hardware chart. The heat-scale table in the design doc stays documentation.

## Dependencies

Subtask 2. Subtask 1's portable CSS is what the parser reads. Subtask 4 is not required.

## Verification

```bash
uv run pytest tests/unit/test_ui_charts.py tests/unit/test_ui_training_state.py tests/unit/test_ui_quality_state.py tests/unit/test_ui_empty_states.py tests/unit/test_events_unavailable.py -q
uv run pytest tests -q
uv run ruff check src tests
```

Do not run `train --profile smoke` to see the loss curve. The unit test is the chart check. On the empty training and quality pages, confirm the existing «не рассчитано» empty copy and the absence of a chart. Stop the server if you opened it.
