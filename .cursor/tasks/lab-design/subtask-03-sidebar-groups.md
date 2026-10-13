# Subtask 3: Sidebar groups

## Goal

Put the project picker, the six pages, and the theme switch into the design's sidebar groups without changing where those pages go.

## Context

Spec §15 order is «Видео», «Данные и разметка», «Обучение», «Анализ», «Качество», «События». `src/excavation_lab/ui/navigation.py` already builds that list. `render_frame` in `src/excavation_lab/ui/shell.py` draws the product title, the project select, the page header, and the one-line description.

The design wants kickers `PROJECT`, `WORKFLOW`, and `APPEARANCE`. Page names stay the Russian sentence-case strings in `copy.py`. «События» stays visible and opens the unavailable page. It is not a disabled item.

`st.navigation` paints its menu at the top of the sidebar, above anything `render_frame` adds. That cannot produce PROJECT, then WORKFLOW, then APPEARANCE. Hide the built-in menu and draw the links in the shell.

## Acceptance Criteria

- [ ] The sidebar order is: product title `Excavation Video Lab`, kicker `PROJECT` and the existing project select, kicker `WORKFLOW` and the six pages, kicker `APPEARANCE` and the Light | Dark switch from subtask 2.
- [ ] Page link labels are `PAGE_VIDEO`, `PAGE_DATA`, `PAGE_TRAINING`, `PAGE_ANALYSIS`, `PAGE_QUALITY`, and `PAGE_EVENTS`. They are not uppercased.
- [ ] The theme script from subtask 2 sets `aria-current="page"` on one `[data-testid="stPageLink-NavLink"]` by matching `getAttribute("href")`, not `a.pathname`. `href=""` is current only when the last segment of `location.pathname` is empty. Each other link is current only when that last segment equals its href (`data`, `training`, `analysis`, `quality`, `events`). The attribute is removed from the other five. On `/data`, «Видео» is not current. There is still one script element. The chrome rule from subtask 1 then paints `--pdm-accent-soft` and overrides emotion's `darkenedBgMix15` background. Disabled nav, if any link is disabled, stays opacity 1 and uses `--pdm-muted`. «События» is enabled.
- [ ] The built-in navigation menu is hidden. Routing still uses `st.navigation` and the existing `url_path` values (`data`, `training`, `analysis`, `quality`, `events`). «Видео» stays the default page.
- [ ] Each page still shows its `st.header` title and the exact `PAGE_DESCRIPTIONS` sentence. The sentence uses the `lab-lead` class (15px / 22px, `--pdm-text-secondary`, max-width 720px). No slogan is added.
- [ ] `tests/unit/test_ui_empty_states.py` still finds the page title, the description, the product title, and the project select. Empty «Видео» still shows the local-path empty state.

## Implementation Notes

In `navigation.py`, pass `position="hidden"` to `st.navigation`. Keep the page list flat. Do not also pass a section titled `WORKFLOW`. That would duplicate the kicker once the shell draws it.

`iter_pages()` returns the registered `st.Page` objects in §15 order and nothing else. `shell.py` calls `iter_pages()` when it draws the links. It must not call `build_navigation()`. A second `st.navigation` during the page run replaces the router. `shell.py` must not import `navigation.py` at module level. `navigation.py` imports the page modules, and those import `shell.py`. Import `iter_pages` inside the function that draws the links, after the modules have finished loading.

Draw each link with `st.sidebar.page_link` on that `Page`, `icon=""`, label from `copy.py`. Do not use emoji. Do not point `page_link` at an external URL.

`st.page_link` does not expose the current page. Streamlit 1.65 keeps `isCurrentPage` on the emotion component and does not copy it to the DOM. Streamlit sets each `page_link` `href` to `Page.url_path`. The default page's `url_path` is `""`, so that anchor's `pathname` is the current URL, not `/`. On `/data` a pathname compare marks both «Видео» and «Данные и разметка», and «Видео» wins because it is first. The other five hrefs are relative (`data`, `training`, `analysis`, `quality`, `events`), not `/data`.

Extend `emit_theme_script` so the same script, after setting `data-theme`, queries `[data-testid="stPageLink-NavLink"]`. Match `getAttribute("href")`, not `a.pathname`. `href=""` is current only when the last segment of `location.pathname` is empty. Each other link is current only when that last segment equals its href. Set `aria-current="page"` on that one link and `removeAttribute("aria-current")` on the other five. Call `emit_theme_script` once, after the six `st.page_link` calls, so those nodes are already in the delta. Do not also call it at the start of `render_frame`. If the script runs before the links mount, one `requestAnimationFrame` retry inside that same script is enough. Do not add a second script element. Do not interpolate a page path into the script from Python. The whitelist is still only `dark` or `light` for the theme value.

Kickers are `st.html` paragraphs with `class="lab-kicker"` and the exact strings `PROJECT`, `WORKFLOW`, `APPEARANCE`. Put those three strings in `copy.py`. They are group labels, not page titles. They contain no `<script>`, so the `:has(script)` rule does not collapse them.

Keep `st.sidebar.header(ui_copy.PRODUCT_TITLE)`. Do not replace it with `st.html`. Subtask 1's sidebar heading rule is 17px / 24px, weight 600. The 28px rule is scoped to `[data-testid="stMain"]`. `tests/unit/test_ui_empty_states.py` looks for `Excavation Video Lab` in element values.

Page description: stop using `st.caption` for `PAGE_DESCRIPTIONS[title]`. Render `st.html` with `class="lab-lead"` and `html.escape` on the sentence. The sentence itself does not change. Other captions on the pages stay `st.caption`.

`render_theme_switch` moves under the `APPEARANCE` kicker. Do not draw it twice.

Project select behavior stays in `_select_project`: same labels, same disabled empty state (`нет проекта`), same help, same `key="project_picker"`. This subtask does not change storage.

`app.header` in the empty-state test must still match the page title. Keep `st.header(title)` in the main area.

## Dependencies

Subtask 2.

## Verification

```bash
uv run pytest tests/unit/test_ui_empty_states.py tests/unit/test_events_unavailable.py tests/unit/test_ui_video_page.py -q
uv run pytest tests -q
uv run ruff check src tests
uv run streamlit run app.py
```

Click all six sidebar links. On `/data` only «Данные и разметка» has `aria-current="page"`. «Видео» does not. Confirm the same one-link rule on `/`, `/training`, `/analysis`, `/quality`, and `/events`. Confirm the groups, the Russian names, and that «События» opens the unavailable copy. Collapse and expand the Streamlit sidebar once. Stop the server.
