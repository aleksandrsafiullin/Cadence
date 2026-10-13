# Subtask 1: Design stylesheet

## Goal

Add the portable design CSS unchanged, plus a Streamlit chrome file that maps widgets onto those tokens.

## Context

`docs/lab_design_code.md` is the visual source of truth. Spec §15 is the Russian Streamlit UI this CSS will skin. The compute core must not import Streamlit, so both files live under `src/excavation_lab/ui/`.

The portable block defines `--pdm-*` and `.lab-*`. Streamlit does not emit `.lab-*` unless we add markup, so widget rules are a second file. They are injected together in subtask 2. This subtask does not wire them into `app.py`.

## Acceptance Criteria

- [ ] `src/excavation_lab/ui/lab_theme.css` contains the fenced CSS from `docs/lab_design_code.md` as an exact substring, including `:root, [data-theme="dark"]`, `[data-theme="light"]`, and the `@media (max-width: 640px)` rule.
- [ ] Token names are unchanged: `--pdm-accent`, `--pdm-bg`, `--pdm-sidebar`, `--pdm-series-observed`, `--pdm-series-forecast`, and the rest of that fence. No `--lab-*` and no `--excavation-*` color tokens.
- [ ] `src/excavation_lab/ui/streamlit_chrome.css` restyles the Streamlit widgets listed below. Every color in that file is `var(--pdm-…)`. It contains no hex (`#RGB` / `#RRGGBB`) and no `wow_page` string.
- [ ] The current-page rule is `[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"]`. Disabled links use those two test ids, opacity `1`, and `--pdm-muted`. The 28px title is scoped to `[data-testid="stMain"]`. The sidebar heading rule is 17px / 24px, weight 600.
- [ ] The chrome file takes only `[data-testid="stElementContainer"]:has(script)` out of flow with `position: absolute`, `height: 0`, `margin: 0`, and `overflow: hidden`. It does not set `display: none` on that container and it does not hide every `stHtml`.
- [ ] The chrome file does not redefine the token values. It does not import another stylesheet.
- [ ] No Python module imports either file yet. `uv run pytest tests -q` still passes.

## Implementation Notes

Copy the CSS fence in `docs/lab_design_code.md` (the block that starts at `:root, [data-theme="dark"]` and ends at the 640px media query). Do not retype hex from memory. Do not add the heat-scale table. That scale is not a CSS variable and this app has no heatmap.

A short comment above the fence is fine. Do not edit a rule inside the fence. `.lab-sidebar { width: 21rem }` stays in the portable file for reuse. The chrome file must not set `[data-testid="stSidebar"]` to `21rem`. Streamlit owns sidebar width and collapse.

`streamlit_chrome.css` targets the installed Streamlit DOM. Confirm test ids in the running app or the installed package if a selector misses. Do not add a hex fallback when a selector misses.

Style these surfaces with tokens only:

- App and main: `[data-testid="stApp"]`, `[data-testid="stMain"]`, `[data-testid="stMainBlockContainer"]`. Background `--pdm-bg`, text `--pdm-text`, font `--pdm-font`, size 14px / 21px. Main max-width 1600px, padding 32px 48px 64px. At `max-width: 640px`, padding 24px 16px 32px. Do not add a 1900px wide mode. None of the six pages is an overview dashboard.
- Sidebar: `[data-testid="stSidebar"]` background `--pdm-sidebar`, right border 1px `--pdm-border-soft`, text `--pdm-text`.
- Page title: 28px / 34px, weight 600, tracking `-0.02em`, `--pdm-text`, scoped to `[data-testid="stMain"]` so the sidebar product title does not inherit it. Section titles in the main area (`st.subheader`): 17px / 24px, weight 600, tracking `-0.01em`, also under `[data-testid="stMain"]`. The sidebar product title is `st.sidebar.header`. Style `[data-testid="stSidebar"]` headings at 17px / 24px, weight 600, color `--pdm-text`.
- Captions: 12px / 17px, `--pdm-muted`. The page lead is not this rule. Subtask 3 gives the lead `lab-lead`.
- Buttons: secondary uses `--pdm-control`, `--pdm-text`, border `--pdm-border`, radius 8px, min-height 36px, weight 500. Primary uses `--pdm-accent`, `--pdm-accent-ink`, weight 600. Hover uses the matching hover token. Disabled `.stButton` opacity `.4`. Transition only `150ms ease-out` on color, background, and border.
- Current page link. Streamlit 1.65 does not put `aria-current` on `st.page_link`. Subtask 3's script sets it. The chrome rule that overrides emotion's `darkenedBgMix15` background is `[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"]`: background `--pdm-accent-soft`, color `--pdm-accent-text`, weight 600. No full-sidebar capsule. Disabled links use the same two test ids plus the disabled attribute Streamlit actually sets (`aria-disabled="true"` or `[disabled]`): opacity `1`, color `--pdm-muted`. Do not target `[data-testid="stSidebarNavLink"]`. That node is the built-in menu, and subtask 3 hides it.
- Inputs, number inputs, select, text area: background `--pdm-input`, text `--pdm-text`, border `--pdm-border`, radius 8px. Focus: border `--pdm-accent` and `box-shadow: 0 0 0 3px var(--pdm-focus-ring)`, no extra outline. Buttons and links use `outline: 2px solid var(--pdm-accent)` with `outline-offset: 2px` on `:focus-visible`.
- Checkbox and toggle checked state: `--pdm-accent`, not `--pdm-success`.
- File uploader drop zone: background `--pdm-surface-subtle`, dashed border `--pdm-border-strong`, radius 12. Hover border `--pdm-accent`.
- Metrics: `[data-testid="stMetric"]` surface `--pdm-surface`, border `--pdm-border`, radius 12, shadow `var(--pdm-shadow)`. Value 28px / 32px, weight 600, tabular numbers. Label 12px `--pdm-muted`. No status fill on the metric card.
- Alerts: `[data-testid="stAlert"]` background `--pdm-surface-subtle`, text `--pdm-text`, radius 12, no shadow. Left border 3px: info `--pdm-accent`, success `--pdm-success`, warning `--pdm-warning`, error `--pdm-danger`. Do not fill the alert with the tone color.
- Dataframe: surface, 1px border, radius 12. Header text `--pdm-muted` on `--pdm-surface-subtle`. Cells `--pdm-text`. Tabular numbers.
- Expander: background `--pdm-surface-subtle`, radius 12, border `--pdm-border`.
- Tabs, if Streamlit renders them: underline `--pdm-accent` on the selected tab, not a pill. Content gap 24px.
- Progress: track `--pdm-border-soft`, fill `--pdm-accent`, height 8px, radius 999px.
- Segmented control: same geometry as `.lab-segmented`. Selected segment `--pdm-accent-soft` / `--pdm-accent-text`.
- Code: `--pdm-surface-subtle`, `--pdm-text-secondary`, radius 4px inline and 8px for a block.
- `.lab-alert[data-tone="neutral"]` in the chrome file only: left border `--pdm-border-strong`, mark color `--pdm-muted`. Do not put this selector in the portable file.
- Theme script slot. The script from subtask 2 is not style-only, so its `st.html` stays in the main flex column. `height: 0` does not remove that column's gap. Take that one container out of flow: `[data-testid="stElementContainer"]:has(script) { position: absolute; height: 0; margin: 0; overflow: hidden; }`. Do not set `display: none` on an ancestor. The script has to run. Do not hide every `[data-testid="stHtml"]`. Kickers, the page lead, and status stripes are `st.html` and must stay visible.

No gradients, glass, neon, or `@keyframes`. Green, yellow, and red are only the success, warning, and danger tokens on alerts and status. Weights 400, 500, and 600. No italic.

## Dependencies

None.

## Verification

```bash
uv run pytest tests -q
uv run ruff check src tests
```

Confirm by reading the two CSS files: the design fence is intact, and `streamlit_chrome.css` has no `#` hex colors. Do not start the app for this subtask. Injection is subtask 2.
