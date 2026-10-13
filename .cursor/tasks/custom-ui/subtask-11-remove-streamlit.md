# Subtask 11: Remove Streamlit

## Goal

Delete Streamlit, `app.py`, `.streamlit/`, and `render()`. Retarget the tests that still import Streamlit. Page behavior already lives on `TestClient` from subtasks 4–10.

## Context

Subtasks 5, 8, and 9 rewrite these `AppTest` functions before this subtask, so they already pass on `TestClient`:

- `tests/unit/test_ui_empty_states.py::test_project_without_video_still_asks_for_a_local_path` — subtask 5
- `tests/unit/test_ui_locale.py::test_analysis_time_survives_the_locale_switch` — subtask 8
- `tests/unit/test_ui_empty_states.py::test_quality_page_without_metrics_json_has_no_zero_chart` and `tests/unit/test_ui_locale.py::test_english_pages_reject_leftover_cyrillic` — subtask 9

After Streamlit is removed, this subtask owns every remaining failure:

- `tests/unit/test_ui_empty_states.py`: `test_page_modules_expose_render` (drop `render`), `test_app_entrypoint_renders_video` (retarget at `GET /`), `test_streamlit_config_is_localhost_only`
- `tests/unit/test_ui_theme.py`: `test_events_page_renders_unavailable_without_the_chart_styler` (drop the `st.warning` source string, keep the no-charts assertion), `test_stylesheet_skip_does_not_skip_the_metric_zero` (retarget the parser that reads `for metric in app.metric:` at the TestClient honesty check), `test_streamlit_import_stays_in_app_and_ui`
- `tests/unit/test_ui_locale.py`: `test_shell_order_and_shared_resolver`
- `tests/unit/test_theme_persistence.py`: `test_only_the_change_callback_writes_the_preference`, `test_app_injects_css_only_before_navigation`, `test_config_theme_base_is_dark_without_hex_colors`, `test_combined_css_loads_both_stylesheets_by_file_name`
- `tests/unit/test_ui_charts.py`: `test_vega_container_uses_surface_tokens`
- `tests/unit/test_no_streamlit_import.py` and `test_shell_does_not_call_auth_or_telemetry`

`tests/unit/test_no_streamlit_import.py` currently allows every file under `ui/` and `app.py`. After this subtask it allows nothing under `src`.

`tokens.STREAMLIT_CHROME_CSS` and `combined_css()` append `streamlit_chrome.css`. The chrome file targets Streamlit DOM. Delete it once nothing injects it.

## Acceptance Criteria

- [ ] `uv remove streamlit` updates `pyproject.toml` and `uv.lock`. `altair`, `starlette`, `jinja2`, and `uvicorn` remain. `uv run python -c "import streamlit"` fails. `import altair` works.
- [ ] `app.py` and `.streamlit/` are deleted. `src/excavation_lab/ui/streamlit_chrome.css` is deleted.
- [ ] No file under `src/` imports `streamlit`. `test_no_streamlit_import.py` asserts that, including `ui/` and `web/`. The old allow-list is gone. The comment-and-string detection tests in that file stay.
- [ ] `theme.py` and `locale.py` no longer import Streamlit. `resolve_theme` / `resolve_locale` default the session argument to an empty mapping. `write_theme_file`, `write_locale_file`, `read_*_file`, and the symlink checks stay. `render_theme_switch`, `render_locale_switch`, `apply_theme`, `emit_theme_script`, and `st.navigation` are deleted. `navigation.py` either goes or becomes a path/title table with no Streamlit import. Page `render()` functions are deleted. Pure helpers stay.
- [ ] This subtask rewrites the tests named in Context that it owns. `test_page_modules_expose_render` no longer requires `render`. `test_app_entrypoint_renders_video` hits `GET /`. `test_events_page_renders_unavailable_without_the_chart_styler` drops the `st.warning` source string and still asserts no chart. `test_stylesheet_skip_does_not_skip_the_metric_zero` no longer parses `for metric in app.metric:`; the honesty check runs against the `TestClient` page. `test_combined_css_loads_both_stylesheets_by_file_name` expects `lab_theme.css` only. The source locks on `app.py`, `st.navigation`, `on_change`, `.streamlit/config.toml`, and `streamlit_chrome.css` are deleted with those files. `combined_css` loads `lab_theme.css` only.
- [ ] `tokens.combined_css` returns the portable stylesheet only. `test_ui_theme.py` chrome-file assertions are rewritten against `lab.css` (no hex, no `wow_page`). `test_streamlit_import_stays_in_app_and_ui` is inverted: `ui/` and `web/` do not import Streamlit.
- [ ] `test_streamlit_config_is_localhost_only` no longer reads `.streamlit/config.toml`. The bind check is `web.server.HOST == "127.0.0.1"` and the upload cap constant is `8192 * 1024 * 1024`. The comment that a file on disk is added by a local path still exists in `docs/ENVIRONMENT.md` or in `uploads.py`.
- [ ] `test_vega_container_uses_surface_tokens` no longer reads `streamlit_chrome.css`. It asserts the chart container in `lab.css` uses `var(--pdm-surface)`, `var(--pdm-border)`, and `border-radius: 12px`, and that the block has no hex.
- [ ] `test_shell_does_not_call_auth_or_telemetry` scans `web/` and `ui/` and does not require `app.py`.
- [ ] `cli.py` docstring no longer grants Streamlit an import exception.
- [ ] `test_forbidden_stack.py` still rejects FastAPI. A scan of `web/` rejects `fastapi` and `streamlit`.
- [ ] `models/quality.py` still imports `copy.py` and not `locale.py` or `web`.

## Implementation Notes

The tests this subtask edits are the ones named in Context. Deleting Streamlit deletes the source locks on `app.py`, `st.navigation`, `on_change`, `.streamlit/config.toml`, and `streamlit_chrome.css`. `combined_css` stays on `lab_theme.css` only.

`test_shell_does_not_call_auth_or_telemetry` scans `web/` and `ui/` and does not require `app.py`.

If `uv remove streamlit` drops a package `charts.py` needs besides altair, add that package explicitly. Do not add Plotly.

Do not delete `lab_theme.css`.

Worker code in `jobs.py` is untouched.

## Dependencies

Subtask 10. Every page is already on Starlette.

## Verification

```bash
uv run pytest tests -q
uv run ruff check src tests
uv run excavation ui --help
```

Confirm `app.py` is gone and `uv run streamlit run app.py` is not a documented command. Do not treat a green suite as a trained model.
