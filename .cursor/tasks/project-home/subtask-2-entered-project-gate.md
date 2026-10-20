# Subtask 2: Gate workflow pages on an entered project

## Goal

Workflow page GETs and `GET /label` render only when `workspace/ui/project.txt` names a listed project. Otherwise they redirect to `/` without consuming the flash file, and they do not create a project. A workspace with one project and no `project.txt` is not entered.

This subtask lands the gate in code. It is not done until subtask 3’s `uv run pytest tests -q` is green. The pytest command at the bottom of this file is a smoke check of the new behavior. A green smoke check with a red suite does not finish the gate.

## Context

Spec §5 starts with creating a project, then working on video. `choose_project_id` in `src/excavation_lab/web/preferences.py` returns the sole project when `project.txt` is empty, and it does not write the file. Page handlers in `shell.py`, `routes.py`, `data_routes.py`, `training_routes.py`, `analysis_routes.py`, and `quality_routes.py` all use that helper. `actions_routes.py` and `label_routes.py` call `data_routes._project_id`, which uses it too.

`_render` in `src/excavation_lab/web/app.py` calls `_take_flash` before it builds the page. `get_label` calls `_render`, then copies `request.state.label_http_status` onto that response. `label_context` sets that status, including 404.

`post_video_add` writes `need_project` via `_remember` and returns `routes._redirect()`, which subtask 1 points at `/video`. The same flash-then-redirect pattern exists for choose, baseline, and the data, training, analysis, and quality page redirects. If the gate consumes the flash while issuing the 303, `GET /` has no sentence to show.

Subtask 1 has already moved the video page to `/video`. This subtask does not move the project forms and does not change create’s redirect. The failed-`write_project_file` redirect to `/` is subtask 5. Do not “fix” that failure by sending the browser to `/video` and relying on this bounce.

## Acceptance Criteria

- [ ] `choose_project_id` returns the stored id only when that id is in the given project list. The `len(ids) == 1` branch is gone. The docstring says an empty file, a missing file, and a stored id that is not listed all return `None`. The function still does not write.
- [ ] `GET /video`, `/data`, `/training`, `/analysis`, `/quality`, `/actions`, `/events`, and `/label` respond `303` to `/` when no project is entered. The response body is not the workflow shell. `TestClient` checks use `follow_redirects=False`.
- [ ] The unentered 303 returns before `_take_flash`. It does not consume `flash.json`.
- [ ] When the project is entered, `_render` still calls `_take_flash` and the workflow page still shows that sentence. A POST that flashes `path_required`, a failed choose, or a baseline error and redirects to `/video` still shows that sentence on the video page.
- [ ] `GET /` stays 200 with or without a project. It does not redirect to `/video`. `GET /` consumes the flash when the gate redirected, and home shows it. A `POST /video/add` with no `project.txt` (one project may exist) redirects to `/video`, `GET /video` then redirects to `/` with the flash file still present, and the following `GET /` contains `need_project`.
- [ ] `get_label` receives the 303 from `_render` before `label_context` runs. It returns that response with status 303. It does not assign `label_http_status` onto it.
- [ ] The same GETs return 200 when `write_project_file` has stored a listed id. The video page is still `/video`. An entered `GET /label` for a missing frame may still set `label_http_status` to 404; that path is unchanged.
- [ ] Neither the redirect nor a page GET calls `create_project` or `write_project_file`. The empty-workspace test still asserts `project.txt` is absent after the GET.
- [ ] JSON, media, frame, job, and the other non-page routes listed in the master plan are not redirected to `/`. A `GET /data/frame` or `GET /label/frames` without a project keeps its current error response.
- [ ] `POST` handlers that resolve a project through `choose_project_id` see `None` when the file is unset, including when exactly one project exists. They do not gain a new implicit project.

## Implementation Notes

- Put the check at the start of `_render`, before `_take_flash`, `page_shell`, and the `page_id == "label"` branch that calls `label_context`. Return `RedirectResponse("/", status_code=303)`. Widen `get_label`’s return type to `Response`.
- In `get_label`, if the `_render` response status is 303, return it immediately. Do not read `label_http_status` first in a way that runs `label_context`. The status copy stays only for a rendered label page.
- Entered: `read_project_file` value is one of `storage.list_projects` ids. If `list_projects` raises `LabError`, treat it as not entered and redirect without consuming flash. Do not render the workflow shell with the error.
- The unentered 303 returns before `_take_flash` and does not consume `flash.json`; when the project is entered, `_render` still calls `_take_flash` and the workflow page still shows that sentence; `GET /` consumes the flash when the gate redirected, and home shows it. Render that home error in the stub `home.html` (an alert is enough). Use `id="project-home-error"` so subtask 5 can keep the same hook. Do not show the workflow stepper on that stub.
- Do not gate on `request.url.path` alone in a middleware that would also catch `/media` or `/data/frame`. `_render` is the right choke point because those routes never call it.
- Changing `choose_project_id` updates every `_project_id` helper at once. Do not leave a private fallback in one routes module.
- Tests this subtask should update, because they are the ones its smoke command runs: any `client.get("/video")` (or `/data`, `/actions`) after `create_project` that expects 200 now calls `write_project_file` first. That includes `test_web_shell.py` `test_stepper_markup_on_video_data_and_actions`. `test_ui_empty_states.py` expects `303` and `location` `/` for unentered workflow paths, and still expects no `project.txt`.
- Add `tests/unit/test_entered_project_gate.py` for the flash survival sequence (`POST /video/add` → `GET /video` leaves `flash.json` → `GET /` shows `need_project`) and for `GET /label` returning 303 with no `label_http_status` overwrite when `project.txt` is absent. One project and no file is the case that used to slip through the fallback.
- Do not try to green the files listed in the master plan’s fallback section here. That is subtask 3. After this subtask, `uv run pytest tests -q` is expected to fail until subtask 3.
- `test_missing_project_waits_on_every_stage` calls `project_stages` directly. It stays a unit test of captions. Subtask 1 already set the video path to `/video`.
- Delete and rename rules in `storage.py` are out of scope. Do not touch `_refuse_locked_delete`.

## Dependencies

Subtask 1. `/video` must already be the video route, and `GET /` must already exist. Subtask 3 must follow before the gate is treated as done.

## Verification

Smoke check only. A green result here does not finish the gate.

```bash
uv run pytest tests/unit/test_entered_project_gate.py tests/unit/test_ui_empty_states.py tests/unit/test_web_shell.py tests/unit/test_ui_video_page.py tests/unit/test_ui_locale.py tests/unit/test_web_server.py tests/unit/test_theme_persistence.py tests/unit/test_workflow_stages.py tests/unit/test_ui_data_freeze.py -q
uv run ruff check src/excavation_lab/web/preferences.py src/excavation_lab/web/app.py src/excavation_lab/web/shell.py src/excavation_lab/web/label_routes.py src/excavation_lab/web/routes.py tests/unit/test_entered_project_gate.py
```

Completion check, owned by subtask 3:

```bash
uv run pytest tests -q
```
