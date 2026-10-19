# Subtask 19: Five-stage workflow stepper across pages; copy parity

## Goal
A compact stepper in `base.html` shows the operator where they are on the path Видео → Данные и разметка → Обучение → Анализ → Действия, with status and the exact next action per stage, in both languages; every new control has an honest disabled reason.

## Context
Spec §5 (workflow), §15 (explain controls). The user's complaint: "the skeleton exists but the interface does not let the work be done". Each page already has local steps (`data_page.workflow_steps`); the project lacks a cross-page view.

## Acceptance Criteria
- [ ] `ui/workflow.py` (no Starlette): `project_stages(workspace, project_id, book) -> tuple[Stage,...]` with `stage_id, title, path, state (done|current|waiting), caption`:
  - `video`: done when ≥1 probed video;
  - `data`: done when a dataset snapshot exists; caption otherwise = first missing among «разбиение не зафиксировано», «нет отобранных кадров», «нет проверенных кадров в обучении/проверке», «снимок не создан» (reuse `data_page` helpers);
  - `training`: done when `list_bundles` non-empty; caption «нет снимка датасета» / «обучение не запускалось» / «идёт обучение»;
  - `analysis`: done when a completed `custom` run exists; caption names the missing piece;
  - `actions`: done when an action inference exists; captions «нет каталога действий», «нет размеченных интервалов в обучении/проверке», «нет прогона обученного детектора», «модель действий не обучена», «распознавание не запускалось».
  Exactly one `current` stage (first non-done); cheap queries only (SQLite counts, directory checks), no parquet reads.
- [ ] `base.html` renders the stepper under the nav on every page; the current page's stage is highlighted; each stage links to its page; captions are plain text, no numbers invented.
- [ ] `web/shell.py::page_shell` adds `stages`; pages without a project show the stepper with all stages waiting and the «создайте проект» caption.
- [ ] Audit every disabled button added in 04a–18b for a visible reason (title attribute + caption), ru and en; fix gaps.
- [ ] `ui/copy.py`: all new keys present in both catalogs; `en` program strings contain no Cyrillic (`_check_catalogs`), `title`/`box`/action `title` pass through untouched.
- [ ] Tests `tests/unit/test_workflow_stages.py`: state progression on a synthetic project at each milestone; exactly one current; en captions have no Cyrillic; `tests/unit/test_web_shell.py` asserts the stepper markup on `/`, `/data`, `/actions`.

## Implementation Notes
- Files: `src/excavation_lab/ui/workflow.py`, `src/excavation_lab/web/shell.py`, `templates/base.html`, `static/lab.css`, `ui/copy.py`, tests.
- Keep «События» out of the stepper (v0.2 stub).
- Do not duplicate per-page step logic; call the existing helpers.

## Dependencies
18b.

## Verification
```bash
uv run pytest tests/unit/test_workflow_stages.py tests/unit/test_web_shell.py tests/unit/test_ui_locale.py -q
uv run excavation ui   # pilot: stepper on every page, ru and en, captions match reality
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
