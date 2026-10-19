# Subtask 04b: Labeling screen service — view-model, JSON routes, template

## Goal
Server side of the in-app box labeler: `ui/label_page.py` (no Starlette), JSON routes in `web/label_routes.py`, and `templates/label.html` that mounts the canvas script (05).

## Context
Spec §7, §9, §15. Routes follow `web/data_routes.py` conventions (project from preferences, locale via `catalog(resolve_locale(...))`, flash via `preferences.write_flash`). Storage through `label_store` (03a/03b) only.

## Acceptance Criteria
- [ ] `ui/label_page.py`:
  - `LabelFrame` (index, `source_time_s`, split, image URL, `review_status`, `label_state`, `box_count`), `frames_for_video(workspace, project_id, video_id) -> list[LabelFrame]` merging `label_store.sample_rows_with_role` with stored records,
  - `frame_payload(workspace, project_id, video_id, index, book) -> dict` with image size, boxes (xyxy pixels, `class_name`, `visibility`, `truncated`, `instance_local_id`, `note`), frame status, `annotator`, `reviewed_at`, catalog classes (`id`, `name`, `title`, `box`, key hint 1..9), `suggestions_allowed: bool` (split == train and `list_bundles` non-empty; suggestions themselves arrive in 08), `prev_index`, `next_index`,
  - `save_frame(workspace, project_id, video_id, index, body: Mapping, book) -> dict | ActionNote` validating JSON (`review_status`, `boxes[]`, optional `note`, `visibility`) and calling `label_store.save_frame_label`,
  - `progress(workspace, project_id, video_id) -> dict[split, {reviewed, unreviewed, excluded, total}]` (`total` from the sample manifest),
  - `annotator_name(workspace) / set_annotator(workspace, name)` in the workspace preferences file next to theme/locale (default `operator`).
- [ ] `web/label_routes.py`: `GET /label` (page; `video`, `frame`), `GET /label/frames`, `GET /label/frame`, `POST /label/frame` (JSON), `POST /label/annotator` (form). Errors `{ok:false, error}` with 400; unknown video/frame → 404; traversal-safe tokens as in `data_routes`.
- [ ] `templates/label.html`: header (video label, split badge, `source_time_s`, index), progress line, annotator field, `<canvas id="box-canvas">`, side panel skeleton (class list with `title` + `box` hint, box attribute controls, frame status buttons, prev/next, back link), `<script type="application/json" id="label-state">`, `<script src="/static/box_labeler.js">`.
- [ ] Copy keys ru/en (`label_progress`, `label_mark_reviewed`, `label_mark_excluded`, `label_mark_unreviewed`, `label_unreviewed_hint`, `label_visibility_clear/partial/unlabelable`, `label_truncated`, `label_delete_box`, `label_prev`, `label_next`, `label_annotator`, `label_saved`, `label_save_failed`, `label_not_frozen`, `label_frame_not_sampled`, `label_fit_width_note`); `_check_catalogs` passes.
- [ ] Tests `tests/unit/test_label_page.py` (TestClient): page 200 on a frozen sampled video; 404 unknown frame; POST saves and GET reflects it; unreviewed POST keeps `trainable=False`; validation frame payload has `suggestions_allowed=False`; annotator persists across requests.

## Implementation Notes
- Frame image via the existing `GET /data/frame?video=&name=`.
- Server clamps coordinates to the image and rejects non-positive boxes with the catalog message.
- The route never opens the catalog YAML directly; use `require_project_catalog`.

## Dependencies
04a.

## Verification
```bash
uv run pytest tests/unit/test_label_page.py tests/unit/test_web_shell.py -q
uv run excavation ui   # pilot → /label shows the first frame, panel, and progress (canvas comes in 05)
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
