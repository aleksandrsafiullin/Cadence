# Subtask 6: Theme contract tests

## Goal

Lock the design contract in pytest so a later edit cannot rename tokens, translate the page titles, or fill the events page.

## Context

Spec §19 item 16 and spec §15. Subtasks 1–5 land the CSS, the preference file, the sidebar, the stripes, and the chart styler. This subtask adds the assertions the rest of the plan must not weaken. Existing empty-state and events tests stay in force.

## Acceptance Criteria

- [ ] `tests/unit/test_ui_theme.py` asserts the default theme is `dark` when the workspace has no `ui/theme.txt` and the session has no `lab_theme`.
- [ ] The same test asserts a file containing `light` is used on a fresh session, and `stopping`, `stopped`, `idle`, and an empty file resolve to `dark`.
- [ ] The portable CSS contains the design-doc fence as a substring, so both `[data-theme="dark"]` and `[data-theme="light"]` are present. `--pdm-accent` is present. `--lab-accent` and `--excavation-accent` are absent.
- [ ] `streamlit_chrome.css` contains no hex color and no `wow_page` string. The portable file's hex values are only hex values that appear in the design fence.
- [ ] Page title constants are still «Видео», «Данные и разметка», «Обучение», «Анализ», «Качество», «События». `JOB_STATUS_LABELS` keys are still exactly `queued`, `running`, `completed`, `failed`, `cancel_requested`, `cancelled`. `THEME_DISPLAY_ONLY` is exactly `Меняет только оформление.`
- [ ] The events page source still renders `EVENTS_UNAVAILABLE` and does not call the chart styler. The existing `tests/unit/test_events_unavailable.py` passes unchanged in meaning: unavailable copy, «не рассчитано», no chart, no zero KPI.
- [ ] The zero-token helper still rejects a metric string `0` in the metric loop. `_strings` / `_joined` ignore a value that contains `--pdm-bg` and `margin: 0`. Skipping a string does not skip the metric loop.
- [ ] Loss role stays observed (`--pdm-series-observed`). AP50 color token stays `--pdm-accent`, not `--pdm-series-forecast`.
- [ ] No module outside `app.py` and `src/excavation_lab/ui/` imports Streamlit.

## Implementation Notes

Put the new assertions in `tests/unit/test_ui_theme.py`. Call the pure helpers in `tokens.py` and `theme.py` where that avoids a browser. Use `AppTest` only for the default-dark session check if the resolver needs Streamlit. Point that app test at a `tmp_path` workspace.

Extract the first CSS fence from `docs/lab_design_code.md` in the test. Do not copy the hex list into the test by hand. Compare the portable file against that fence.

Do not delete or loosen assertions in `tests/unit/test_ui_empty_states.py`, `tests/unit/test_events_unavailable.py`, or `tests/unit/test_no_streamlit_import.py`. If the `_strings` skip from subtask 2 is missing, add it here. The metric `0` case stays in the metric loop, and `test_ui_theme.py` covers it too. Chart layout and the `ast` import check on `charts.py` and `tokens.py` stay in `tests/unit/test_ui_charts.py`. This file asserts the display-only sentence and the token contract. `test_no_streamlit_import.py` allows every module under `ui/`, so it is not the check for those two modules.

Do not add a visual regression screenshot test. Do not start Streamlit from pytest.

After the tests pass, do the browser pass in the master plan once. Fix selector or copy mistakes in the files from subtasks 1–5. Do not invent a second palette while fixing them.

## Dependencies

Subtasks 1, 2, 3, 4, and 5.

## Verification

```bash
uv run pytest tests -q
uv run ruff check src tests
uv run streamlit run app.py
```

Exercise the theme toggle and all six pages as the master plan describes. Stop the server. Do not run smoke training.
