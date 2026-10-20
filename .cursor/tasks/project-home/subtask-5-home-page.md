# Subtask 5: Home page, project actions, and the return link

## Goal

`GET /` is the project list. The user can open, create, rename, and delete a project there. Opening or creating writes `project.txt` and lands on `/video` only when that write succeeds. A failed write stays on `/` with `project_write_failed`. The workflow sidebar no longer contains the project select or the project-manager dialog. One link returns to the list.

## Context

Spec §15 is the workflow screens; this page sits in front of them. Subtask 1 split `/` and `/video`. Subtask 2 redirects unentered workflow GETs to `/` without consuming flash, and the home stub already shows that sentence on `id="project-home-error"`. Subtask 3 has the suite green. Subtask 4 provides `ui.project_home` display rows.

The current forms live in `src/excavation_lab/web/templates/base.html` inside `<dialog id="project-manager">`, posted to `/project/create`, `/project/rename`, and `/project/delete`. Failures call `_stash_project_notice`, which sets flash `project_manager=1` so the dialog reopens. Catalog behavior is `project_catalog.js` plus `_posted_catalog` in `app.py`.

`_post_project_create` today writes `project.txt` and then redirects to `next` even when `write_project_file` raised. After this subtask that failure must not land on `/video`. The gate would bounce an unentered `/video` back to `/`, but only if the flash survived. Do not depend on that bounce. Redirect to `/` from the handler.

Product name in the catalog stays Cadence. New sentences are Russian and English.

## Acceptance Criteria

- [ ] `GET /` renders the home template, not the workflow nav and not the five-stage stepper. Theme and locale forms are on the page and post back to `/`. Flash errors, including ones left by a gated workflow POST, still render in `id="project-home-error"`.
- [ ] The list shows only summary fields: display name, project id, catalog id, created time (stored UTC text, column labeled as UTC), video count, episode count, dataset count, model count. `None` from subtask 4 renders `not_calculated` («не рассчитано» in Russian). A real zero renders `0`. The page does not contain mAP, loss, ETA, or epoch wording.
- [ ] An empty project list shows the new empty sentence and the create form. `GET /` does not call `create_project`.
- [ ] Create stays `POST /project/create` with the existing catalog modes. Element ids listed in the master plan (`project-create`, `catalog-existing-fields`, `catalog-new-fields`, `catalog-id`, `new-catalog-id`, `copy-from`, `catalog-class-rows`, `add-class-row`, and the class inputs) are unchanged so `project_catalog.js` still runs. The script tag is on the home template, not on workflow pages.
- [ ] Successful create writes `project.txt` and responds `303 /video` even if `next` is something else. Successful `POST /project` for a listed id does the same and ignores `next`.
- [ ] If `create_project` succeeds and `write_project_file` raises, the response is `303 /` with `project_write_failed`, `project.txt` is absent, and the location is not `/video`. The new project directory stays. The same rule applies to `POST /project` when the write fails: do not redirect to `/video`.
- [ ] Rename is `POST /project/rename`, display name only, then `303 /`. Delete is `POST /project/delete` with checkbox `name="confirm" value="yes"`, then `303 /`. Storage still refuses a live `queued` / `running` / `cancel_requested` lock and a symlink `lab.sqlite`. A refused delete leaves the directory in place. Deleting the entered project clears `project.txt`.
- [ ] Any other failure of those four POSTs flashes the same sentence the handler flashes today (`catalog` choice keys or `LabError.message`, including `delete_confirm_required`) and redirects to `/`. The home page shows it in `id="project-home-error"`. Flash no longer sets `project_manager`, and no response includes `data-project-manager-open`.
- [ ] Workflow pages for an entered project still show the workflow nav and stepper. They do not contain `id="lab-project"`, `id="project-manager"`, or `id="project-manager-open"`. They do contain `id="lab-home"` pointing at `/`, plus the entered display name as text.
- [ ] `GET /` with a project already stored still shows the list and marks that row. It does not redirect to `/video`. The open control is a submit button, not a `<select onchange>`.
- [ ] Copy keys exist on both catalogs and pass the existing parity check. Home is not added to `_PAGE_IDS` or `PAGE_ROUTES`.
- [ ] `tests/unit/test_delete_project.py` and `tests/unit/test_rename_project.py` still pass without a change to the lock or symlink checks.

## Implementation Notes

- Replace the stub in `src/excavation_lab/web/templates/home.html`. Keep consuming flash on `GET /` and keep `id="project-home-error"`. Override the sidebar block: product title, theme, locale, no workflow nav. Put the list and the forms in `<main id="content">`. Reuse `lab-*` classes already in `lab.css` (`lab-form`, `lab-table` or `lab-project-rows`, `lab-btn`, `lab-alert`). Add a rule only if a row cannot be aligned with those classes.
- Build the home context in `web/shell.py` or a small `web` helper that calls `ui.project_home`. Pass `current_path="/"`. Do not call `page_shell` for home; that function builds workflow links and stages.
- `POST /project` with a listed id: `write_project_file`, then `_redirect("/video")` only inside the `try` that succeeded. On exception, flash `project_write_failed` and `_redirect("/")`. Unknown id: flash `error`, `_redirect("/")`. Do not call `_safe_next` for this route. Update `test_web_shell.py` so the label-query return check that currently posts `/project` with `next=/label?video=vid&frame=0` is posted to `/theme` or `/locale` instead. The four unsafe-`next` cases stay on `location == "/"`.
- Create: on `write_project_file` success, `_redirect("/video")`. On `write_project_file` failure, `_redirect("/")` with `project_write_failed`. Do not fall through to a shared redirect that still honors `next`. Rename and delete success redirect to `/` and ignore `next`. Failures use `_stash_notice` (error only), not `_stash_project_notice`. Delete `pending_project_manager` handling if nothing sets it.
- Remove the dialog, the select, and both inline scripts from `base.html`. Keep theme and locale there for workflow pages. Their `next` stays `current_path`.
- Sidebar link copy: `back_to_projects` — «К проектам» / “Projects”. Open button: `open_project` — «Открыть» / “Open”. Current row: `home_current` — «Текущий» / “Current”. Empty list: a full sentence, not the short `no_project` placeholder «нет проекта». Column labels are new catalog fields, including a created column that says the value is UTC.
- Enter control: `<form id="project-open-{{ project_id }}" method="post" action="/project">` with a hidden `project` value and a submit button. No `next` field is required, because the handler ignores it.
- Tests that assert `id="project-manager"` in `test_web_shell.py`, `test_ui_video_page.py`, and `test_ui_data_freeze.py` flip: workflow HTML does not contain those ids; `GET /` does contain `id="project-create"` and, after a failed create, `id="project-home-error"`.
- A create test: one existing catalog, `POST /project/create`, `follow_redirects=False`, location `/video`, `project.txt` equals the new id, then `GET /video` is 200 and contains `id="lab-home"`.
- A write-failure test: force `write_project_file` to raise after `create_project` would succeed, `follow_redirects=False`, location `/`, body or the following `GET /` contains `project_write_failed`, and the location of the POST is not `/video`.
- A delete test can stay at the storage layer (`test_delete_project.py`) for the lock. Add one web test that posts delete without `confirm` and expects `303 /` plus the confirm sentence, and one that deletes an idle project and expects the directory gone and the list not to include it.
- Do not import Starlette from `ui/project_home.py` or `storage.py`.
- Localhost binding is unchanged. Do not start a train job.

## Dependencies

Subtasks 1, 2, 3, and 4. The suite is green before this subtask starts.

## Verification

```bash
uv run pytest tests/unit/test_web_shell.py tests/unit/test_ui_video_page.py tests/unit/test_ui_data_freeze.py tests/unit/test_ui_empty_states.py tests/unit/test_web_server.py tests/unit/test_project_home.py tests/unit/test_entered_project_gate.py tests/unit/test_delete_project.py tests/unit/test_rename_project.py tests/unit/test_ui_locale.py -q
uv run ruff check src tests
```

Then run the epic suite and the browser pass from the master plan:

```bash
uv run pytest tests -q
uv run ruff check src tests
uv run excavation ui
```

In the browser, on localhost: open `/`, create or open a project, confirm `/video` and the workflow sidebar, follow «К проектам» back to `/`, and request `/data` with `project.txt` removed and confirm the browser ends on `/`. No training run.
