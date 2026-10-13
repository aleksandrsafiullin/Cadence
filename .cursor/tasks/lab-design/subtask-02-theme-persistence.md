# Subtask 2: Theme persistence

## Goal

Load both stylesheets, default the app to dark, and keep a Light or Dark choice across pages and after reload.

## Context

Spec §15 plus `docs/lab_design_code.md`: dark is the main theme, the switch reads `Light` | `Dark`, and `data-theme` on the root is `dark` or `light`. Spec §22 and `.streamlit/config.toml` stay localhost-only with telemetry off.

Session state dies when the Streamlit process restarts. The URL belongs to `st.navigation`. Persist the choice in a file under the active workspace.

## Acceptance Criteria

- [ ] With no preference file and an empty session, the theme is `dark` and the script sets `data-theme="dark"` on `document.documentElement`.
- [ ] Choosing `Light` writes `light` to `<workspace>/ui/theme.txt` and updates `st.session_state["lab_theme"]`. A new session against that workspace starts light, and the segmented control's selected value is `Light`. `Dark` writes `dark` and the control shows `Dark`. The file is written only from `on_change`.
- [ ] The switch options are the literals `Light` and `Dark`. `default` is `Dark` when the resolved theme is `dark` and `Light` when it is `light`. The control has a real Russian label, `Тема`, with `label_visibility="collapsed"`. The caption under it is the exact sentence `Меняет только оформление.`
- [ ] An invalid file (`stopping`, `stopped`, `idle`, empty, or anything other than `dark` or `light`) is treated as `dark` and does not raise.
- [ ] The write stays inside the workspace root. A symlink that leaves the root is refused. Tests that pass `tmp_path` do not create `workspace/ui/theme.txt` in the repo.
- [ ] `.streamlit/config.toml` gains `[theme] base = "dark"` and still has `server.address = "127.0.0.1"`, `maxUploadSize = 200`, and `gatherUsageStats = false`. The existing comment that contains `local path` stays. The file does not set hex theme colors or `0.0.0.0`. `test_ui_empty_states.py` still sees `"local path"` in the file text and `"auth" not in parsed`.
- [ ] One theme script is emitted per run, from `render_frame` only. `apply_theme` may inject the CSS files more than once. Those injections are style-only and do not take a main-column slot.
- [ ] `tests/unit/test_ui_empty_states.py` and `tests/unit/test_events_unavailable.py` still pass. `_strings` / `_joined` skip a value that contains `--pdm-bg`. The metric loop is unchanged, so a metric value `0` still fails.
- [ ] `src/excavation_lab/ui/theme.py` is the only new module that imports Streamlit. `tokens.py` does not. `tests/unit/test_no_streamlit_import.py` still passes.

## Implementation Notes

New modules:

- `src/excavation_lab/ui/tokens.py` — no Streamlit import. Load `lab_theme.css` and `streamlit_chrome.css` from the package directory (`Path(__file__).with_name(...)`). Expose the combined CSS text and a parser that returns the custom-property map for `dark` or `light`. Split the portable file on `[data-theme="light"]`. Dark properties come from the first block. Light properties override that map. This parser is for subtask 5. This subtask only needs it to exist and to reject an unknown theme name.
- `src/excavation_lab/ui/theme.py` — Streamlit allowed. `apply_theme(workspace: Path) -> Literal["dark", "light"]`, `emit_theme_script(theme: Literal["dark", "light"]) -> None`, and `render_theme_switch(workspace: Path) -> None`.

Preference path: `(Path(workspace) / "ui" / "theme.txt")`. One line, UTF-8, stripped. Allowed bytes are `dark` and `light` only. Read does not create directories and does not write the file. Write creates `ui/` with `mkdir(parents=True, exist_ok=True)`, writes a temp file in that directory, and `os.replace`s it onto `theme.txt`. After resolve, the file must be `is_relative_to` the resolved workspace. If the write fails, keep the session value and show a Russian sidebar error. Do not crash the page.

Add these strings to `src/excavation_lab/ui/copy.py`. Subtask 6 asserts the caption exactly.

- `THEME_LABEL = "Тема"`
- `THEME_LIGHT = "Light"`
- `THEME_DARK = "Dark"`
- `THEME_DISPLAY_ONLY = "Меняет только оформление."`

Also add a Russian sentence for a failed write. That sentence is not the display-only caption.

Resolve order:

1. If `st.session_state["lab_theme"]` is `dark` or `light`, use it.
2. Else read the file. A valid value is stored into session state.
3. Else `dark`.

Injection:

- `apply_theme` calls `st.html(path)` for `lab_theme.css`, then for `streamlit_chrome.css`. The installed Streamlit wraps a CSS `Path` in `<style>` and sends it to the event container. That does not take a layout slot. Do not paste the CSS into `st.markdown`.
- `emit_theme_script(theme)` is a separate function. It emits one `st.html` with `unsafe_allow_javascript=True`. The script sets `document.documentElement.setAttribute("data-theme", theme)` after the whitelist. Build that string only from `dark` or `light`. This is the only script element. Subtask 3 adds the current-page lines to this same script and matches `getAttribute("href")`, not `a.pathname`. Do not add a second `st.html` script.

Call `apply_theme(storage.default_workspace_root())` in `app.py` after `st.set_page_config` and before `build_navigation().run()`. That call injects CSS only. Call `apply_theme(root)` from `render_frame`. Call `emit_theme_script` once from `render_frame`, not from `app.py`. Page tests never enter `app.py`, so the script has to come from the page. Subtask 3 moves that one call to after the page links are drawn. Until the links exist, once per `render_frame` is enough, because this subtask's script only sets `data-theme`. A full app run still emits the script once. The chrome rule from subtask 1 collapses that one container. `display: none` would stop the script.

`render_theme_switch` uses `st.segmented_control` with options `Light` and `Dark`, `key="lab_appearance"`, `label=ui_copy.THEME_LABEL`, and `label_visibility="collapsed"`. Pass `default=ui_copy.THEME_DARK` when the resolved theme is `dark`, and `default=ui_copy.THEME_LIGHT` when it is `light`. Do not omit `default`. Do not set `persist_state`. Do not set `bind="query-params"`. `on_change` maps the label to `dark` or `light`, updates `st.session_state["lab_theme"]`, and writes `theme.txt`. No other path writes that file. Place the switch at the bottom of the sidebar in this subtask. Subtask 3 moves it under the `APPEARANCE` kicker. Under the control, `st.caption(ui_copy.THEME_DISPLAY_ONLY)`.

Do not put `http://` or `https://` in any file under `src/excavation_lab/ui/`. `tests/unit/test_ui_empty_states.py` rejects those strings.

Zero-scan fix, in this subtask, inside `_strings` in both `tests/unit/test_ui_empty_states.py` and `tests/unit/test_events_unavailable.py`. `_joined` must use that helper so it inherits the skip. Skip a value that contains `--pdm-bg`. Style-only CSS is already on the event container, which `_nodes()` does not scan. Keep the skip anyway. Do not change the separate metric loop in `_assert_honest`. A metric value of `0` or `0.0` still fails. `не рассчитано` still passes. Do not skip captions or alerts.

When adding `[theme]` to `.streamlit/config.toml`, leave the existing comment that contains the words `local path`. Do not add a key or a comment that makes `"auth" in parsed` true.

Do not create a project, start a job, or write outside `ui/theme.txt`.

## Dependencies

Subtask 1.

## Verification

```bash
uv run pytest tests/unit/test_ui_empty_states.py tests/unit/test_events_unavailable.py tests/unit/test_no_streamlit_import.py -q
uv run pytest tests -q
uv run ruff check src tests
uv run streamlit run app.py
```

In the browser: confirm the first load is dark and the control shows `Dark`. Toggle `Light`, reload, and confirm the control shows `Light` and the page background is the light token `#F5F5F7`. Confirm there is one collapsed script container above the title and no empty gap from a second script. Toggle `Dark` and open one other page. Confirm `workspace/ui/theme.txt` contains `dark` or `light` and is not committed. Stop the server.
