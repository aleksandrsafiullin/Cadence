# Subtask 1: Move the video page to `/video`

## Goal

`GET /video` is the video page. `GET /` is registered and is not the video page. Return paths, the stepper, and tests stop treating `/` as video. Cookie paths stay `Path=/`.

## Context

Spec §15 «Видео» and §5 step 1. Today `PAGE_ROUTES` in `src/excavation_lab/ui/navigation.py` maps `video` to `/`, and `create_app` registers those tuples as the only page GETs. The inventory of every `/`-means-video assumption is in `.cursor/tasks/project-home/master-plan.md` under “What treats `/` as the video page”. This subtask applies the rows marked as the video page. It does not build the project list and does not add the entered-project redirect.

## Acceptance Criteria

- [ ] `PAGE_ROUTES` video path is `/video`. The tuple still has seven entries in the same order: video, data, training, analysis, quality, actions, events. Home is not an entry. The module docstring still says seven pages and names `/video` as the video path.
- [ ] `ui/workflow.py` `_STAGE_SPEC` video path is `/video`. The stepper is still five stages (video, data, training, analysis, actions). Quality, events, and `/label` stay off the stepper.
- [ ] `create_app` registers `GET /` with its own handler, not via `PAGE_ROUTES`. `GET /` returns 200 and does not render the video page: no `action="/video/choose"`, no `<video`. `GET /video` renders the current video template.
- [ ] `RETURN_PATHS` includes `/video` (through the nav paths) and still includes `/label`, and also includes `"/"` so theme and locale on the future home page are legal. `page_shell` on a `/video` request sets `current_path` to `/video`.
- [ ] `routes.py` `_redirect` sends video form posts to `/video`. The `lab_video` cookie is still set with `path="/"`.
- [ ] Both GET forms in `src/excavation_lab/web/templates/video.html` use `action="/video"`: the video switcher (`method="get"`, `name="video"`) and the run-refresh button. No template under `src/excavation_lab/web/templates/` has `action="/"`.
- [ ] With two videos in an entered-or-fallback project, `GET /video?video=<id>` selects that id (selected option or `video_id` hidden input or `/media/.../<id>`). `GET /?video=<id>` does not: the home response has no `<video` and no `/media/` URL for that id.
- [ ] `_safe_next` still returns `"/"` for an unsafe `next`. Theme and locale still accept `/video`, `/data`, `/training`, `/analysis`, `/quality`, `/actions`, `/events`, `/label`, and a canonical `/label?video=&frame=` return.
- [ ] No `set_cookie` or `delete_cookie` `path="/"` in `app.py`, `routes.py`, `data_routes.py`, `analysis_routes.py`, `quality_routes.py`, or `actions_routes.py` is changed to `/video`.
- [ ] The test files in the master-plan inventory expect the video page at `/video`. Open-redirect tests in `test_web_shell.py` still expect `location == "/"`.
- [ ] `project.txt` is not written by `GET /` or `GET /video`. No project is created.

## Implementation Notes

- Add the `/` route next to the existing page routes in `src/excavation_lab/web/app.py`. A short `home.html` that extends `base.html` and shows the product title is enough. Subtask 5 replaces the body. Do not put create, rename, or delete on it yet, and do not remove the sidebar select or the dialog in this subtask.
- Change only those two `action="/"` values in `video.html`. Leave `action="/video/choose"` and `action="/video/baseline"` as they are. The switcher is rendered only when there is more than one video, so the selection test inserts two videos and requests the second id. The single-project fallback is still in place in this subtask, so `write_project_file` is not required yet. Subtask 3 adds it to this test.
- Do not special-case an empty project inside `_render`. Unentered `/video` still returns the video shell until subtask 2. That keeps this subtask’s test edits to paths only.
- `base.html` hidden `next` inputs use `current_path`. Once `/video` is in `RETURN_PATHS`, the video page posts `next=/video` without hardcoding it. Leave the Jinja default `'/'` in place.
- `navigation.py` is imported by `shell.py` and `app.py`. Change the path in the one tuple; do not add a second video route.
- Tests to update, and only for the video-page meaning of `/`:
  - `tests/unit/test_ui_video_page.py` — `client.get("/")` that loads video markup, and `_location(...) == "/"` after posts that use `routes._redirect` (including `/video/choose`). Cookie assertions look for `lab_video=`; do not require `Path=/video`.
  - `tests/unit/test_ui_locale.py` — `_ENGLISH_ROUTES` and `client.get("/")` that expect `page_video`. Locale posts that mean “return to video” use `next=/video`.
  - `tests/unit/test_ui_empty_states.py` — `_SHELL_PAGES` video path.
  - `tests/unit/test_web_server.py` — video shell assertions move to `GET /video`. Add a `GET /` assertion that status is 200 and the video chooser is absent.
  - `tests/unit/test_web_shell.py` — `_PATHS`, stepper `hrefs`, `aria-current` href, and `client.get("/")` used as the video page. Keep the four unsafe-`next` assertions on `location == "/"`.
  - `tests/unit/test_workflow_stages.py` — the expected stage path list.
  - `tests/unit/test_theme_persistence.py` — GET and `next` that refer to the video shell.
- `tests/unit/test_ui_actions_page.py` only checks that `PAGE_ROUTES[5]` is `actions`. It stays valid if the order is unchanged.
- Do not edit `docs/DECISIONS.md`.
- After the edits, run the straggler search from the master plan, including `action="/"`. A remaining `client.get("/")` must not assert video markup. `action="/"` in `src/excavation_lab/web/templates/` must be zero.

## Dependencies

None. This is the first subtask of the project-home epic.

## Verification

```bash
uv run pytest tests/unit/test_ui_video_page.py tests/unit/test_ui_locale.py tests/unit/test_ui_empty_states.py tests/unit/test_web_server.py tests/unit/test_web_shell.py tests/unit/test_workflow_stages.py tests/unit/test_theme_persistence.py tests/unit/test_ui_actions_page.py -q
uv run ruff check src/excavation_lab/ui/navigation.py src/excavation_lab/ui/workflow.py src/excavation_lab/web/app.py src/excavation_lab/web/shell.py src/excavation_lab/web/routes.py tests/unit/test_ui_video_page.py tests/unit/test_ui_locale.py tests/unit/test_ui_empty_states.py tests/unit/test_web_server.py tests/unit/test_web_shell.py tests/unit/test_workflow_stages.py tests/unit/test_theme_persistence.py
```
