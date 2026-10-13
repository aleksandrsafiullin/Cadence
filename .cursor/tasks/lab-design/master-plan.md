# Lab design code on the Streamlit UI

## Overview

Apply `docs/lab_design_code.md` to the existing Excavation Video Lab Streamlit app. The design doc is the visual source of truth. Product name, Russian page copy, job statuses, and the v0.2 events gate stay as they are.

This plan does not train a model, change the worker, or add a second web framework. It adds one portable stylesheet, one Streamlit chrome stylesheet, a dark-default theme that survives reload, sidebar groups, status stripes, and a single chart styler for the two charts that already exist.

Do not edit `.cursor/tasks/master-plan.md` or `.cursor/tasks/subtask-01` through `subtask-46`. Those files are the historical archive.

## Phases

Media, splits, training, and tracking are already in the app. This plan does not reopen them. Events stay the existing unavailable page. They are not implemented.

1. **Tokens** — paste the design CSS unchanged, then add Streamlit selectors that use only `var(--pdm-…)`.
2. **Theme** — default `dark`, `data-theme` on `<html>`, Light | Dark control, preference file under the active workspace.
3. **Shell** — sidebar groups `PROJECT`, `WORKFLOW`, `APPEARANCE`; Russian page titles; one-sentence lead.
4. **Page chrome** — one primary button per screen, status as a 3px stripe, existing widgets left in place.
5. **Charts** — one Altair styler for the loss line and the AP50 bars. No new charts.
6. **Tests** — token names, both themes, Russian titles, events gate, no wow-page colors. Full pytest before calling the UI done.

## Dependencies

- Phase 2 depends on phase 1. The injector must have both CSS files.
- Phase 3 depends on phase 2. The appearance control already exists before the sidebar is reordered.
- Phase 4 depends on phases 1 and 3. Button and status markup need the chrome classes and the new sidebar.
- Phase 5 depends on phase 2. The styler reads token values for the active theme.
- Phase 6 depends on phases 1–5. It locks the contract; it does not restyle the app.
- `tests/unit/test_ui_empty_states.py` and `tests/unit/test_events_unavailable.py` skip a string that contains `--pdm-bg` inside `_strings` / `_joined` in the same subtask that first injects CSS (phase 2). Style-only CSS goes to the event container, which `_nodes()` does not scan. The skip stays anyway. The metric loop stays separate, so a metric value `0` still fails.
- No dependency on a pilot MP4, a checkpoint, or `train --profile smoke`.

## Execution order

| Order | File | Goal | Depends on |
| --- | --- | --- | --- |
| 1 | `subtask-01-design-stylesheet.md` | Paste the design CSS and add Streamlit chrome rules | — |
| 2 | `subtask-02-theme-persistence.md` | Default dark, persist the choice, set `data-theme` | 1 |
| 3 | `subtask-03-sidebar-groups.md` | PROJECT / WORKFLOW / APPEARANCE and the page lead | 2 |
| 4 | `subtask-04-page-chrome.md` | One primary action and status stripes | 1, 3 |
| 5 | `subtask-05-chart-styler.md` | Style the existing loss and AP50 charts | 2 |
| 6 | `subtask-06-theme-tests.md` | Lock tokens, copy, events, and both themes | 1–5 |

Do them in this order. Do not start page markup before the token file exists. Do not claim the UI done before subtask 6 and the browser pass below.

## Verification

After subtask 6, and again after any fix the browser pass forces:

```bash
uv run pytest tests -q
uv run ruff check src tests
uv run streamlit run app.py
```

Browser pass, on localhost only:

- First load is dark. `<html data-theme="dark">` after the page script runs.
- Switch to `Light`, reload, and confirm light. Switch back to `Dark`, open every page, and confirm dark.
- Pages: «Видео», «Данные и разметка», «Обучение», «Анализ», «Качество», «События».
- «События» still says the section is unavailable. Indicators still say «не рассчитано». No chart and no zero counts.
- Empty «Видео» still asks for a local file. No demo player and no emoji.
- One primary button on «Видео», «Данные и разметка», «Обучение», and «Анализ». «Качество» row actions stay secondary. «События» has no button.
- Sidebar kickers are `PROJECT`, `WORKFLOW`, `APPEARANCE`. Page names stay Russian sentence case. The switch says `Light` and `Dark`.

Stop the server when finished. `train --profile smoke` is not a check for this work. A screenshot is not a model-quality claim (spec §25).

## Resolved conflicts

1. **Job words.** The design doc lists `stopping`, `stopped`, and `idle`. Spec §16 and the worker use `queued`, `running`, `completed`, `failed`, `cancel_requested`, `cancelled`. Keep the excavation enum and the Russian labels in `JOB_STATUS_LABELS`. Style those six. `queued` and `running` use the accent stripe. `status_tone(None)` and any other string use `neutral`. The words stay «не рассчитано». Do not rename the statuses.

2. **Theme persistence.** A query param fights `st.navigation`, which owns the URL. Use `st.session_state["lab_theme"]` for the current Streamlit session and `workspace/ui/theme.txt` for reload. `workspace/` is already gitignored. `persist_state="session"` dies with the process, so it is not the reload path. The segmented control takes `default="Dark"` or `default="Light"` from the resolved theme. Omitting `default` leaves a fresh session unselected while the file already says `light`. The file is written only from `on_change`, not from the first paint. A missing or invalid file means `dark`. The workspace root is the one `render_frame` already uses, so tests that pass `tmp_path` do not write the repo `workspace/`.

3. **`data-theme` on `<html>`.** Streamlit does not set attributes on the document root. One `st.html(..., unsafe_allow_javascript=True)` per run sets `data-theme` to `dark` or `light` after a whitelist check. The same script sets `aria-current="page"` on one `stPageLink-NavLink` by matching `getAttribute("href")`, not `a.pathname` (subtask 3). Style-only CSS does not take a layout slot. That script does. The chrome rule is `[data-testid="stElementContainer"]:has(script) { position: absolute; height: 0; margin: 0; overflow: hidden; }`. `height: 0` alone leaves the main-column gap, so `position: absolute` takes that one container out of flow. Do not hide every `stHtml`, and do not use `display: none` on an ancestor. The portable CSS also applies dark tokens on `:root`, so a failed script stays dark.

4. **Charts are Altair, not Plotly.** The app draws `st.line_chart` (training loss) and `st.bar_chart` (AP50). `plotly` is not in `uv.lock`. Altair is. Do not add Plotly. One styler in `src/excavation_lab/ui/charts.py` builds Altair charts with no named theme and explicit token colors. That is the `template="none"` rule for this stack: no `plotly`, `plotly_dark`, or `ggplot2`. Loss uses `--pdm-series-observed`. AP50 is one measured series and uses `--pdm-accent`, not the forecast role. `tests/unit/test_ui_charts.py` locks padding 56/16/48/16, a hidden legend, `--pdm-font` at 12px, `--pdm-chart-text`, and hover colors from the parsed tokens. The browser pass does not open a chart. The same test parses `charts.py` and `tokens.py` with `ast` because `test_no_streamlit_import.py` allows every module under `ui/`. Altair's theme is disabled inside the styler function and restored before return, not at import.

5. **Video box colors.** `src/excavation_lab/visualization/annotate.py` paints class RGB onto encoded frames. Those colors are not the UI palette. Leave them. Do not import the stylesheet into the encoder, and do not make every box the accent blue.

6. **Wow page.** `docs/wow_page_design.md` is not in this repo. Do not create it and do not copy colors from memory. The portable CSS must contain the design-doc fence as a substring, so extra hex fails the test.

7. **English switch vs Russian UI.** Page titles, descriptions, buttons, empty states, and errors stay Russian. The theme labels are the literals `Light` and `Dark`. Sidebar kickers are the literals `PROJECT`, `WORKFLOW`, `APPEARANCE`. The collapsed label is `Тема`. The caption under the switch is exactly `Меняет только оформление.`

8. **Current page and disabled navigation.** Streamlit 1.65 sets `aria-current="page"` on built-in sidebar nav (`[data-testid="stSidebarNavLink"]`), not on `st.page_link`. The current `st.page_link` is an emotion prop `isCurrentPage` on `[data-testid="stPageLink-NavLink"]`, and that prop does not land on the DOM. Hiding the built-in menu removes the only node an `aria-current` rule could match, and emotion paints `backgroundColor: darkenedBgMix15` on the current link. Keep `page_link` and `position="hidden"`. The theme script matches `getAttribute("href")`, not `a.pathname`. `href=""` is current only when the last segment of `location.pathname` is empty. Each other link is current only when that last segment equals its href. `aria-current="page"` is set on that one link and removed from the other five. A pathname compare is wrong: the default page's `url_path` is `""`, so its `pathname` is the current URL, and on `/data` «Видео» would win. The chrome rule is `[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"]`. Disabled links use that same two-attribute specificity, stay opacity 1, and use `--pdm-muted`. Disabled buttons stay at opacity `.4`. «События» is not a disabled nav item.

9. **One primary on «Качество».** Send-to-labeling is one button per error row, and the empty page has zero buttons (`test_ui_empty_states.py`). Do not add a screen-level primary and do not mark every row primary. Those buttons stay secondary.

10. **Sidebar width.** The portable class `.lab-sidebar` keeps `21rem` for reuse. The live Streamlit sidebar keeps Streamlit's own width and collapse control. Chrome CSS colors it. It does not force `21rem`.

11. **Second palette in `config.toml`.** Add `[theme] base = "dark"` so the first paint is not white. Keep the existing comment that contains `local path`. `test_ui_empty_states.py` asserts that substring and `"auth" not in parsed`. Do not set `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor`, or `font` there. Those would be a second palette. Token hex lives only in the pasted CSS.

12. **Empty-state zero scan.** Style-only CSS is sent to the event container, and `_nodes()` does not scan that container. A script or other `st.html` can still land in the main column with bare `0` (`margin: 0`). Skip those strings inside `_strings` / `_joined` when the value contains `--pdm-bg`. Keep the separate metric loop. A metric value of `0` must still fail. Do not skip alerts or captions.

## Out of scope

- Editing the historical task archive.
- Importing or copying Python from a Predictive Maintenance package. This repo has no `src/pdm`.
- Renaming `--pdm-*` to `--lab-*` or `--excavation-*`.
- Mixing `docs/wow_page_design.md`.
- Changing `JobStatus`, the lock, or the one-GPU rule.
- Implementing v0.2 events, KPI charts, or demo numbers.
- React, FastAPI, a custom CSS framework, or Plotly.
- Recoloring supervision boxes.
- Smoke training as a quality gate.
