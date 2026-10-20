# Project home

## Overview

The application home becomes a real page at `GET /`: a list of projects, with create, rename, and delete on that page. Choosing a project writes `workspace/ui/project.txt` and opens `/video`, the first workflow step.

The seven workflow screens stay what they are (Видео, Данные и разметка, Обучение, Анализ, Качество, Действия, События). The labeler stays at `/label` and stays off the nav. Inside a project the workflow sidebar stays. The sidebar project `<select id="lab-project">` and `<dialog id="project-manager">` go away. One control on that sidebar returns to `/`.

`/` and `/video` do not both render the video page. An unentered workflow URL does not render an empty workflow shell and does not create a project. It redirects to `/`.

This epic does not change detection, training, tracking, or export. It does not weaken delete: a live workspace lock (`queued`, `running`, `cancel_requested`) and a symlink `lab.sqlite` still refuse deletion. No `pilot-small` training.

Do not edit `.cursor/tasks/master-plan.md` or `.cursor/tasks/subtask-*.md`. Those files belong to another epic.

## Decisions

1. **Home is `GET /`, registered in `create_app`, not in `PAGE_ROUTES`.** It is not a workflow nav entry and not a stepper stage. `PAGE_ROUTES` stays seven pages. The video tuple becomes `("video", "/video", "page_video")`.
2. **No entered project → `303` to `/`.** This is the only unentered behavior, used for every workflow page GET and for `GET /label`. A one-line need-project shell is not used. JSON, media, frame, and job routes are not redirected. The unentered 303 returns before `_take_flash` and does not consume `flash.json`. When the project is entered, `_render` still calls `_take_flash` and the workflow page still shows that sentence. `GET /` consumes the flash when the gate redirected, and home shows it. `get_label` returns that 303 before `label_context` runs and does not copy `label_http_status` onto it.
3. **Entered means `project.txt` names a project in `list_projects`.** `choose_project_id` loses the single-project fallback. The file is written only by a successful enter (`POST /project`) or a successful create whose `write_project_file` succeeded. Nothing auto-creates a project, and a lone project is not treated as entered. The gate is not done until the full suite is green (subtask 3). A short pytest file list is not enough.
4. **Enter and successful create land on `/video`.** Those two handlers ignore `next`. Create lands on `/video` only when `write_project_file` succeeds. If the project row was stored but the file write fails, the handler redirects to `/` with `project_write_failed`, not to `/video`. The new project directory stays. Rename and delete succeed back to `/`. Failures of all four project POSTs flash `error` and redirect to `/`, where the home page shows the sentence. There is no dialog and no `project_manager=1` reopen flag.
5. **Cookie `Path=/` stays.** That attribute means “send this cookie for the whole origin”. It is not a claim that the video page lives at `/`. Changing it to `Path=/video` would hide the cookie from `/data` and the other pages.
6. **`_safe_next` still falls back to `/`.** After this epic that fallback is the home page (open-redirect protection), not the video page. Theme and locale still honor a `next` in `RETURN_PATHS`.
7. **Project facts are row data from that project’s `project.sqlite`.** Display name, project id, catalog id, stored `created_at`, and cheap `COUNT(*)` values. A failed or missing read renders the existing «не рассчитано» / “not calculated”. A successful count of zero is a real zero. No mAP, loss, epoch, score, or ETA.

## Phases

### 1. Routes and return paths

Move the video page from `/` to `/video`, including the two GET forms in `video.html` that still submit to `/`. Register a non-video `GET /` so the two paths cannot collide. Update every return path, stepper href, and test that treats `/` as the video page. Leave cookie paths and the open-redirect fallback alone.

### 2. Entered-project gate

Workflow page GETs and `GET /label` require a stored, listed project. Otherwise `303` to `/` and do not write `project.txt`. The unentered 303 returns before `_take_flash` and leaves `flash.json` for `GET /`. An entered workflow page still consumes its own flash. Drop the single-project fallback in code. This subtask is not done while the suite is still red from that fallback.

### 3. Fallback test migration

Every test that expects 200, or a successful project-scoped POST, after one `create_project` calls `write_project_file` first. Finish with `uv run pytest tests -q`. That run is the gate’s completion check.

### 4. Project summaries

Add a Starlette-free reader for the home list: catalog id, created time, and four counts from the project database. Open sqlite only through `project_sqlite`. An escaping symlink is a row of `None`s and is not re-raised. That is stricter than `list_projects`, which re-raises `PathNotAllowedError`.

### 5. Home page and sidebar

Render the list and move create, rename, and delete onto it. Preserve the catalog form ids and `project_catalog.js`. Remove the sidebar select and the dialog. Add one link back to `/`. Point project POST success and failure at the paths in decision 4, including the failed `write_project_file` case.

## Dependencies

```text
subtask 1  routes and return paths
    ↓
subtask 2  entered-project gate (code)
    ↓
subtask 3  fallback test migration  ← full suite; the gate is done here
    ↓
subtask 5  home page and sidebar
    ↑
subtask 4  project summaries (no route dependency; must finish before subtask 5)
```

Subtask 4 can be implemented beside subtasks 1–3. Subtask 5 waits until 1, 2, 3, and 4 are done. Do not start subtask 5 on a red suite.

## Execution order

1. `subtask-1-video-route.md`
2. `subtask-2-entered-project-gate.md`
3. `subtask-3-fallback-tests.md`
4. `subtask-4-project-summaries.md` (after or beside 1–3)
5. `subtask-5-home-page.md`

## What treats `/` as the video page

Subtask 1 updates the rows marked “video page”. Rows marked “stay” keep the string `/` because the meaning becomes “home” or “cookie scope”.

### Code

| Location | Today | Action |
| --- | --- | --- |
| `src/excavation_lab/ui/navigation.py` `PAGE_ROUTES` | `("video", "/", "page_video")` | Video page. Path becomes `/video`. Home is not added to this tuple. |
| `src/excavation_lab/ui/workflow.py` `_STAGE_SPEC` | `("video", "page_video", "/")` | Video page. Path becomes `/video`. Stepper stays five stages. |
| `src/excavation_lab/web/templates/video.html` | two `method="get" action="/"` forms: the video switcher (`name="video"`, about line 44) and the run-refresh button (about line 118) | Video page. Both `action` values become `"/video"`. They are GET forms, not POST redirects. |
| `src/excavation_lab/web/routes.py` `_redirect` | `RedirectResponse("/")` | Video page. Video form posts (`/video/add`, `/video/choose`, `/video/baseline`) return to `/video`. |
| `src/excavation_lab/web/shell.py` `RETURN_PATHS` | `_NAV_PATHS \| {"/label"}` | After the path change, `_NAV_PATHS` includes `/video` and stops including `/`. Add `"/"` explicitly: `RETURN_PATHS = _NAV_PATHS \| {"/", "/label"}`. |
| `src/excavation_lab/web/shell.py` `page_shell` | `current_path = ... else "/"` | Stay. Unknown paths fall back to home. A request to `/video` is in `RETURN_PATHS`, so its hidden `next` becomes `/video` without a template edit. |
| `src/excavation_lab/web/app.py` `_safe_next` | unknown `next` returns `"/"` | Stay. Unsafe `next` opens home, not video. |
| `src/excavation_lab/web/templates/base.html` | six `<input name="next" value="{{ current_path \| default('/') }}">` (project select, theme, locale, create, rename, delete) | The Jinja default `"/"` stays. On the video page, `current_path` becomes `/video`. Subtask 5 removes the project select, create, rename, and delete inputs from this file. Theme and locale stay, still posting `current_path`. |
| `src/excavation_lab/web/app.py` `_post_project` | honors `next`, including `/label?...` | Subtask 5: a listed id writes `project.txt` and redirects to `/video`, ignoring `next`. A failed write redirects to `/`, not `/video`. |
| `src/excavation_lab/web/app.py` create / rename / delete | honor `next`; failures set `project_manager=1` | Subtask 5: create success (row and `project.txt`) → `/video`; create whose `write_project_file` fails → `/` with `project_write_failed`; rename and delete success → `/`; every other failure → `/` with `error` only. |

### Cookies (do not retarget)

`path="/"` on these calls is the cookie scope. Leave it.

| Location | Cookie |
| --- | --- |
| `src/excavation_lab/web/app.py` `_render` | `lab_video` |
| `src/excavation_lab/web/app.py` `_set_cookie` | `lab_analysis_time`, `lab_analysis_mode`, `lab_analysis_run`, `lab_analysis_view`, `lab_quality_run`, `lab_quality_run_a`, `lab_quality_run_b` |
| `src/excavation_lab/web/routes.py` `_redirect` | `lab_video` (only the redirect URL changes) |
| `src/excavation_lab/web/data_routes.py` | `lab_video` (redirect is already `/data`) |
| `src/excavation_lab/web/analysis_routes.py` | analysis cookies (redirect is already `/analysis`) |
| `src/excavation_lab/web/quality_routes.py` | set and `delete_cookie` (redirect is already `/quality`) |
| `src/excavation_lab/web/actions_routes.py` `_actions_redirect` | video cookie (redirect is already `/actions`) |

These redirects are already page-specific and are not `/`: `analysis_routes.py` `/analysis`, `quality_routes.py` `/quality`, `data_routes.py` `/data`, `training_routes.py` `/training`, `actions_routes.py` `/actions`, `label_routes.py` label return target.

### Tests that treat `GET /` or stepper href `/` as video

| File | What to change |
| --- | --- |
| `tests/unit/test_ui_video_page.py` | Every `client.get("/")` that loads the video page becomes `client.get("/video")`. `_location(...) == "/"` after `/video/choose` (and any other video form post using `_redirect`) becomes `"/video"`. `Set-Cookie` still contains `lab_video=`; `Path=/` stays. Add a test that `GET /video?video=<id>` selects that video and `GET /?video=<id>` does not. Assertions that the video page contains `id="project-manager"` stay true until subtask 5, then move to the home page. |
| `tests/unit/test_ui_locale.py` | `_ENGLISH_ROUTES` first pair `("/", "Video", "video")` becomes `("/video", "Video", "video")`. `client.get("/")` that expects `page_video` becomes `client.get("/video")`. A locale post whose story is “return to the video page” uses `next=/video`. |
| `tests/unit/test_ui_empty_states.py` | `_SHELL_PAGES` video row `("/", "video", PAGE_VIDEO)` becomes `("/video", ...)`. Empty-workspace `GET /` expectations that the body is the video shell move to `/video` in subtask 1. Subtask 2 then changes an unentered `/video` from 200 to 303. |
| `tests/unit/test_web_server.py` | `test_index_is_an_empty_shell` expects `page_video` in `<main>` of `GET /`. Point the video assertions at `GET /video`. `GET /` is a 200 that is not the video page (no `action="/video/choose"`, no `<video`). |
| `tests/unit/test_web_shell.py` | `_PATHS` and the stepper `hrefs` tuple replace the leading `"/"` with `"/video"`. `aria-current` on the video stage href becomes `/video`. `client.get("/")` that asserts workflow markup, `id="lab-project"`, or the English video title becomes `client.get("/video")`. |
| `tests/unit/test_workflow_stages.py` | `test_missing_project_waits_on_every_stage` path list starts with `"/video"`. |
| `tests/unit/test_theme_persistence.py` | `client.get("/")` that reads the themed video shell becomes `client.get("/video")`. `next` posted to return to that shell becomes `"/video"`. |

### Tests that must keep landing on `/`

These assert open-redirect protection. After the epic, `/` is home. Do not retarget them to `/video`.

`tests/unit/test_web_shell.py` posts that expect `location == "/"`:

- `/locale` with `next=/label?video=vid&frame=1&evil=1`
- `/theme` with `next=/label?video=../secrets&frame=1`
- `/locale` with `next=/label?video=vid&frame=1a`
- `/theme` with `next=https://example.com`

The same test posts `/project` with `next=/label?video=vid&frame=0` and no project id, and expects that label URL. Subtask 5 stops `POST /project` from honoring `next`. Move that label-return assertion onto `/theme` or `/locale`, which still honor `_safe_next`. The four rejection assertions stay on `/`.

`docs/DECISIONS.md` mentions a historical stepper on `/`. That sentence is a walkthrough record. Do not edit it.

Straggler check at the end of subtask 1. This includes template form actions, not only POST redirects:

```bash
rg -n 'client\.get\("/"\)|RedirectResponse\("/"\)|\("video", "/"\)|"/", "video"|_ENGLISH_ROUTES|action="/"' src tests
```

`action="/"` in `src/excavation_lab/web/templates/` must be zero. Remaining `client.get("/")` hits must be home-page tests or the open-redirect cases above, not video markup. `GET /video?video=` selecting a video is expected.

## Pages that redirect when nothing is entered

Subtask 2 applies the gate at the start of `_render`, before `_take_flash` and before `page_shell` or `label_context`. That covers `PAGE_ROUTES` and `GET /label`, because `get_label` calls `_render`.

`get_label` today copies `request.state.label_http_status` onto the `_render` response. The 303 must already have been returned from `_render` before `label_context` runs, and `get_label` must return that redirect unchanged. It must not assign `label_http_status` onto a 303.

`post_video_add` (and the same pattern in choose, baseline, and the data, training, analysis, and quality redirects) writes a flash such as `need_project` and redirects to a workflow page. After subtask 1 that video redirect is `/video`. The gate then sends `/video` to `/`. If `_take_flash` runs on the gated response, the file is consumed by a response that never renders and `GET /` has nothing to show. The unentered 303 returns before `_take_flash` and does not consume `flash.json`. When the project is entered, `_render` still calls `_take_flash` and the workflow page still shows that sentence. `GET /` consumes the flash when the gate redirected, and home shows it. The home stub in subtask 2 renders that sentence; subtask 5 keeps it on `id="project-home-error"`.

The same `_render` order covers every workflow page those POSTs redirect to. Do not special-case only `/video/add`.

Subtask 2 drops the single-project fallback. Tests that `create_project` and then expect 200, or a successful project-scoped POST, without `write_project_file` go red. Subtask 2’s own new tests are not the completion check. Subtask 3 migrates the suite. The gate is done when `uv run pytest tests -q` is green.

Known files the short subtask 2 command does not cover. Subtask 3 updates these, plus any other failure the full suite reports:

- `tests/unit/test_ui_analysis_state.py`
- `tests/unit/test_ui_training_state.py`
- `tests/unit/test_ui_quality_state.py`
- `tests/unit/test_ui_actions_page.py`
- `tests/unit/test_ui_actions_results.py`
- `tests/unit/test_label_page.py`
- `tests/unit/test_label_suggestions.py`
- `tests/unit/test_ui_theme.py`
- `tests/unit/test_web_chart.py`
- `tests/integration/test_inapp_label_to_bundle.py`
- `tests/unit/test_events_unavailable.py`
- `tests/unit/test_web_static.py`

`test_events_unavailable.py` and `test_ui_theme.py` also GET `/events` on an empty workspace and expect 200. That is the gate, not the fallback: an unentered `/events` is 303. Keep the unavailable-page assertions by entering a project (`create_project` plus `write_project_file`) when the test is about that page’s content. `test_web_static.py` loads `/label` through `test_label_page.py` `_lab`, which only calls `create_project`. `GET /jobs/.../progress` and `GET /static/...` are not page shells; leave them if they still pass without `project.txt`.

Redirect to `/`:

- `GET /video`
- `GET /data`
- `GET /training`
- `GET /analysis`
- `GET /quality`
- `GET /actions`
- `GET /events`
- `GET /label`

Do not redirect: `/video/add`, `/video/choose`, `/video/baseline`, `/media/...`, `/jobs/.../progress`, `/data/frame`, `/data/batch`, the other `/data/*` posts, `/label/frames`, `/label/frame`, `/label/suggest`, `/label/annotator`, `/training/start`, `/training/cancel`, `/analysis/start`, `/analysis/export`, `/analysis/zones`, `/analysis/threshold`, `/analysis/frame`, `/quality/compare`, `/quality/label`, `/actions/class`, `/actions/seed`, `/actions/intervals`, `/actions/ready`, `/actions/job`, `/actions/train`, `/actions/cancel`, `/actions/infer`, `/actions/export`, `/actions/evaluate`, `/project`, `/project/create`, `/project/rename`, `/project/delete`, `/theme`, `/locale`, `/static/*`.

Those handlers keep their current error behavior when no project is selected. After the fallback is removed, “no project” means the stored id is missing or not listed, including when the workspace contains exactly one project and `project.txt` was never written.

## Home list

`storage.list_project_summaries` (no Starlette import) opens each project sqlite through `project_sqlite` (or the same resolve check). `connect()` follows symlinks. `project_sqlite` raises `PathNotAllowedError` when the target leaves the workspace, and `list_projects` re-raises that. Summaries do not re-raise. An escaping symlink becomes one row whose fields are `None` except `project_id` from the validated directory name. That is stricter than `list_projects`. A symlink whose target stays inside the workspace is read normally.

Catch `sqlite3.Error` per statement so one missing table does not blank `catalog_id`, `created_at`, or the other counts.

The reader uses one connection per project:

| Field | Source | When unknown |
| --- | --- | --- |
| Display name | `projects.name` | «не рассчитано» |
| Project id | `projects.project_id`, or the validated directory name when the file cannot be opened | Directories that fail `validate_project_name` are skipped. An escaping symlink still produces a row. |
| Catalog id | `projects.catalog_id` | «не рассчитано» |
| Created | `projects.created_at`, the stored ISO UTC text, labeled as UTC | «не рассчитано» |
| Videos | `COUNT(*)` from `videos` | «не рассчитано» |
| Episodes | `COUNT(*)` from `episodes` | «не рассчитано» |
| Datasets | `COUNT(*)` from `dataset_versions` | «не рассчитано» |
| Models | `COUNT(*)` from `model_versions` | «не рассчитано» |

A count cell is an integer when that one statement succeeds, including `0`. It is unknown when that statement fails, the table is missing, or the value is NULL. A failed count does not blank the other cells. Do not open parquet, do not call `jobs.reopen`, do not read `lab.sqlite`, and do not show training or quality numbers.

`ProjectRecord` stays `project_id`, `name`, `workspace`, `project_dir`, `sqlite_path`. Summaries are a separate typed dict so existing `list_projects` callers do not grow fields.

## POST contract (subtask 5)

Handlers stay in `src/excavation_lab/web/app.py`. Storage functions stay `create_project`, `rename_project`, `delete_project`.

| Request | Success | Failure |
| --- | --- | --- |
| `POST /project` with a listed id | `write_project_file` succeeds, `303 /video` | Unknown id, or `write_project_file` raises: do not treat the project as entered. Flash `error` (`project_write_failed` when the write itself failed). `303 /`. |
| `POST /project/create` | `create_project` and `write_project_file` both succeed, `303 /video` | Catalog / `LabError` before the row exists: flash that sentence, `303 /`. Row exists but `write_project_file` fails: flash `project_write_failed`, `303 /`, leave the directory, do not redirect to `/video`. |
| `POST /project/rename` | Display name only. Directory and `project_id` stay. `303 /`. | `303 /`. |
| `POST /project/delete` | Checkbox `confirm=yes` required. Lock and symlink checks stay in storage. If `project.txt` names the deleted id, `clear_project_file`. `303 /`. | Missing checkbox or `LabError`: `303 /`. |

Create still uses the catalog branch already in `_posted_catalog`: existing catalog, or a new catalog id, optional copy, class rows. A blank catalog choice does not silently use the seed.

## Workflow chrome (subtask 5)

Remove from `base.html`: the `/project` select form, `#project-manager-open`, `<dialog id="project-manager">`, and the inline script that opens the dialog and auto-submits `#lab-project`.

Add one link in the sidebar, `id="lab-home"`, href `/`, copy «К проектам» / “Projects”. Under it, show the entered project’s display name as text, not a control.

Keep the workflow kicker, nav, stepper, theme form, and locale form.

Load `/static/project_catalog.js` from the home template only. The script keeps these ids, moved onto the home page: `project-create`, `project-name`, `catalog_mode`, `catalog-existing-fields`, `catalog-id`, `catalog-new-fields`, `new-catalog-id`, `copy-from`, `catalog-editor`, `catalog-class-rows`, `add-class-row`, `class_name`, `class_title`, `class_box`, `project-rename-{id}`, `rename-name-{id}`, `project-delete-{id}`, `project-delete-confirm-{id}`.

New ids: `lab-home`, `project-open-{id}`, `project-home-error`.

Removed ids: `lab-project`, `project-manager`, `project-manager-open`, `project-manager-title`, `project-manager-error`, `project-manager-list`, `data-project-manager-open`.

## Copy

Russian and English strings live in `src/excavation_lab/ui/copy.py` on both catalogs, including the field-parity check. New strings are home title, lead, empty list, open button, back link, current-project marker, UTC created label, and the column labels. Reuse `not_calculated`, `create_project_*`, `catalog_*`, `rename_project_*`, `delete_project_*`, and `delete_confirm_required`.

Do not add a home entry to `_PAGE_IDS`, `page_titles`, or `PAGE_ROUTES`. Home is not a workflow page.

Identifiers and file names stay English. No React, FastAPI, or Streamlit. The page server stays Starlette, Jinja2, and vanilla JS. Localhost only. `storage.py` and `ui/project_home.py` do not import Starlette.

## Verification

Epic, after subtask 5. Subtask 3 is the first time `uv run pytest tests -q` is required; subtask 5 runs it again after the home page:

```bash
uv run pytest tests -q
uv run ruff check src tests
```

Then `uv run excavation ui` and exercise, on localhost:

- `GET /` lists projects or shows the empty sentence, and does not create a project.
- Create with an existing catalog, then confirm the browser lands on `/video` and `project.txt` contains the new id.
- The video sidebar has the workflow links and «К проектам», and has neither `#lab-project` nor `#project-manager`.
- The home link returns to `/`. Open an existing project and confirm `/video` loads.
- With `project.txt` removed, `GET /data` ends on `/` and does not show the data shell.
- Rename changes the display name only. Delete without the checkbox stays. Delete with the checkbox, and no live lock, removes the project.

No training run. No substitute MP4.

Each subtask has its own `uv run` command. A red test in a later subtask is fixed in that subtask, not by pointing video tests back at `/`.
