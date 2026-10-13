# Subtask 4: Sidebar, theme, locale, project, events gate

## Goal

Put the real shell on every page: project picker, five v0.1 links plus «События», light/dark, and Russian/English. «События» is the unavailable gate, not a later page.

## Context

Spec §15 and the current `shell.py` / `navigation.py` / `theme.py` / `locale.py`. Visible order is Видео, Данные и разметка, Обучение, Анализ, Качество, События. Kickers are `PROJECT`, `WORKFLOW`, `APPEARANCE`. Preferences already persist at `workspace/ui/theme.txt` and `workspace/ui/locale.txt` through `write_theme_file` and `write_locale_file`. Default theme is dark. Default locale is `ru`. `events_page.render` shows `events_unavailable` and a grid of `event_indicator_labels` as «не рассчитано». It does not chart.

Page bodies other than events stay short empty states from the catalog until their own subtasks. Video's real layout is subtask 5.

## Acceptance Criteria

- [ ] `base.html` sidebar lists the six links with the active catalog's titles. Paths are `/`, `/data`, `/training`, `/analysis`, `/quality`, `/events`. The current link has `aria-current="page"`.
- [ ] Project control is a `<select>` posting to `POST /project`. Options are `storage.list_projects` names. No projects → one disabled option, catalog `no_project` («нет проекта»), and the handler stores nothing. The chosen id is written to `workspace/ui/project.txt` with the same inside-workspace symlink refusal as `write_locale_file`. An id that is not in `list_projects` is ignored.
- [ ] `POST /theme` accepts `dark` or `light` and calls `write_theme_file`. `POST /locale` accepts `ru` or `en` and calls `write_locale_file`. Anything else does not write. Both redirect 303 back to the submitting page. The next GET sets `data-theme` and `lang` from the files. Pass `session={}` into `resolve_theme` and `resolve_locale`.
- [ ] A missing theme file renders `data-theme="dark"`. A missing locale file renders Russian. `read_theme_file` / `read_locale_file` still do not create the file.
- [ ] `GET /events` shows `events_unavailable`, `events_unavailable_caption`, `events_v02_caption`, and all twelve `event_indicator_labels` values paired with `not_calculated`. `unavailable_error().message` is not on the page. There are no propose or export controls. The HTML has no `<video>`, no `<canvas>`, and no chart node. Version digits in those captions (`0.1`, `v0.2`) are not measurements. English locale file `en` renders the English catalog, including the events page, with no Cyrillic in that response.
- [ ] `GET /data`, `/training`, `/analysis`, and `/quality` return 200, the catalog title, and that page's existing empty-state sentence when the workspace has no project. They do not invent metrics.
- [ ] `status_markup(text, tone)` in `src/excavation_lab/ui/status.py` returns the escaped `.lab-alert` HTML. `render_status` calls it. `tests/unit/test_ui_status.py` still passes.
- [ ] This subtask rewrites shell `AppTest` coverage onto `TestClient`: chrome, kickers, and the language switch in `tests/unit/test_ui_locale.py`; `test_switch_defaults_dark_and_persists_light` and `test_escaped_symlink_shows_sidebar_error` in `tests/unit/test_theme_persistence.py`; `tests/unit/test_events_unavailable.py::test_events_page_shows_uncomputed_indicators`; and the empty-workspace shell cases in `tests/unit/test_ui_empty_states.py`. It does not take `test_project_without_video_still_asks_for_a_local_path` (subtask 5), `test_analysis_time_survives_the_locale_switch` (subtask 8), `test_quality_page_without_metrics_json_has_no_zero_chart` or `test_english_pages_reject_leftover_cyrillic` (subtask 9), or the source-lock tests named in subtask 11 (`test_page_modules_expose_render`, `test_app_entrypoint_renders_video`, `test_shell_order_and_shared_resolver`, `test_only_the_change_callback_writes_the_preference`, `test_app_injects_css_only_before_navigation`, `test_config_theme_base_is_dark_without_hex_colors`, `test_combined_css_loads_both_stylesheets_by_file_name`, `test_streamlit_config_is_localhost_only`). `render()` may stay so later page files still run.
- [ ] `web/` still does not import FastAPI or Streamlit. Theme and locale file helpers may still live in modules that import Streamlit until subtask 11. The new handlers call `write_theme_file` / `write_locale_file`, not `st.session_state`.

## Implementation Notes

CSS in `static/lab.css`: `.lab-app` is `grid-template-columns: 21rem 1fr`. Sidebar background `var(--pdm-sidebar)`. Current link uses `var(--pdm-accent-soft)` and `var(--pdm-accent-text)`. Focus ring uses `var(--pdm-focus-ring)`. No hex. Buttons and inputs are real elements, min height about 2rem, visible focus.

`preferences.py` holds the project file and `flash.json` (one JSON object, `os.replace`, symlink check). Flash can be unused until subtask 5. Do not log a history of notices.

The events template prints the three catalog strings and the indicator grid. It does not call a propose or export function and it does not print `unavailable_error().message`. The CLI test `test_events_cli_refuses_before_storage` stays.

Rewrite `test_events_page_shows_uncomputed_indicators` onto `TestClient` in this subtask. Keep assertions for `events_unavailable`, `events_unavailable_caption`, `events_v02_caption`, twelve `not_calculated` indicators, and no propose/export controls. The events zero-check keeps the existing skip: `0.1` and `v0.2` in those captions are not measurements (`_zero_token("версии 0.1 не")` stays `None`).

Locale switch must not import `excavation_lab.ui.locale` from `models/`. Existing `test_compute_modules_do_not_import_locale` stays.

## Dependencies

Subtask 3.

## Verification

```bash
uv run pytest tests/unit/test_web_shell.py tests/unit/test_theme_persistence.py tests/unit/test_ui_locale.py tests/unit/test_events_unavailable.py tests/unit/test_ui_status.py tests/unit/test_ui_empty_states.py -q
uv run ruff check src tests
```

Manual, after `uv run excavation ui`: load `/`, confirm the six links, switch theme, reload, confirm `data-theme` stuck, switch language to English and back. Open `/events` and confirm the unavailable sentence. Stop the process. Empty workspace is enough. Do not download a video.
