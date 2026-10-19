# Subtask 04a: Extract the page shell; link the labeler from the Data page

## Goal
Make the shared page context reusable by a page that is not in the top navigation (`web/shell.py`), and add the entry points to `/label` on «Данные и разметка» (step 3 button and contact-sheet thumbnails). `/label` itself arrives in 04b; until then it may render a placeholder shell with the catalog empty-state text.

## Context
Spec §15. Today `web/app.py::_context` builds the shell (nav links, project list, theme/locale, catalog strings) and `_render` dispatches on `PAGE_ROUTES`. The labeler must share that shell without becoming a seventh nav entry.

## Acceptance Criteria
- [ ] `src/excavation_lab/web/shell.py::page_shell(request, page_id, flash) -> dict` returns exactly what `_context` returned; `app._render` uses it; `page_id="label"` maps to title `book.label_page_title` and keeps the «Данные и разметка» nav link marked current.
- [ ] `web/label_routes.py` with `GET /label` registered in `create_app` (not in `ui/navigation.PAGE_ROUTES`); for now renders `label.html` with the shell and a placeholder `<main>` containing `book.label_loading` (replaced in 04b).
- [ ] `templates/data.html` step 3: button «Разметить в приложении» → `/label?video=<id>`; each contact-sheet `<li>` wraps the image in a link to `/label?video=<id>&frame=<idx>` (index parsed from the file name by the view-model, not in the template); the CVAT export/import forms remain.
- [ ] `web/data_routes._editor` provides `label_links` (`{index, href}`) and `label_open_href`.
- [ ] Copy keys ru/en: `label_page_title`, `label_open_app`, `label_loading`, `label_back_to_data`; `_check_catalogs` passes.
- [ ] Tests: `tests/unit/test_web_shell.py` asserts `page_shell` parity with the previous context keys and that `/label` does not appear in nav links; `tests/unit/test_ui_data_freeze.py` asserts the step-3 button and thumbnail links on a sampled video.

## Implementation Notes
- Files: `src/excavation_lab/web/{app,shell,label_routes,data_routes}.py`, `templates/{data,label}.html`, `ui/copy.py`, tests.
- Keep cookie handling (`lab_video`) in `_render` as is; `/label` sets the same cookie from its `video` query.
- No labeler logic here.

## Dependencies
03b.

## Verification
```bash
uv run pytest tests/unit/test_web_shell.py tests/unit/test_ui_data_freeze.py -q
uv run excavation ui   # pilot → Данные → step 3 button and thumbnail links open /label placeholder
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
