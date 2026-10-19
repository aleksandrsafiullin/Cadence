# Subtask 12: Page «Действия» — catalog editor and interval timeline

## Goal
New page `/actions` (nav between «Качество» and «События») where the operator edits the action catalog and marks action intervals under the video player with keyboard shortcuts; stored via the services of subtask 11.

## Context
Spec §15 (explain controls; precise seek via source timestamps, not only the browser player), §20.1, TZ §5 help text. Patterns to reuse: `web/static/episode_cutter.js` (timeline pieces, handles, `fetch` save), `ui/data_page.source_interval_from_player` (player → source seconds), `block_editor_payload`, `timeline_pieces`.

## Acceptance Criteria
- [ ] `ui/navigation.PAGE_ROUTES` gains `("actions", "/actions", "page_actions")` and the module docstring says seven pages; `web/app.py::_render` dispatches `actions.html`; `web/actions_routes.py` with `actions_context` + routes: `POST /actions/class` (add/edit title), `POST /actions/seed`, `POST /actions/intervals` (JSON replace for one video), `GET /actions/intervals?video=`. Both carry a `revision` (`{count, sha256 of the canonical interval list}`) computed by `actions/labels.revision(video_id)`; the POST body must echo the revision the client loaded, otherwise the server refuses with `action_labels_stale` and the page offers «Перезагрузить» — a stale tab never overwrites intervals saved elsewhere.
- [ ] `ui/actions_page.py` view-model: `catalog_rows`, `timeline_payload(video, blocks, intervals, book)` (player seconds for JS, source seconds for save), `workflow steps` for this page (1 catalog, 2 labels, 3 train, 4 results — 3–4 rendered as waiting with reasons until subtask 18), `help text` = `book.action_block_help` (TZ string, both languages).
- [ ] `templates/actions.html` step 1: catalog table (`name`, `title`), add-row form, «Заполнить примером» (seed) when empty, explanation that `name` is an English token and `title` is shown as-is.
- [ ] Step 2: video select (same cookie as data page), `<video>` player, role-block strip (read-only, same colours as data page), action interval track below, action `<select>`, inputs start/end (source seconds, 3 decimals), buttons: «Начало = текущее время», «Конец = текущее время», «Добавить», «Удалить», «Сохранить»; keys `i` (set start), `o` (set end), `Enter` (add), `Delete` (remove selected), `1..9` action pick, `,`/`.` step one frame. Frame-step snapping is **cosmetic** (rounds the displayed edge to the nearest frame step from the probed fps); stored values are the source seconds the server computed, never `frame_index / fps`.
- [ ] When the video's blocks are **not frozen**, a visible notice (`action_blocks_unfrozen`, ru/en) says labels are accepted against the draft blocks and will be re-checked against the frozen split before training; the role column is marked «черновик».
- [ ] A coverage line under the table lists each catalog action with its count in train and in validation and marks in red any action with 0 in either (`action_coverage_missing`); the help text names `swing_loaded` vs `swing_empty` explicitly so the operator labels both swings.
- [ ] Client refuses overlap and an interval not inside one block before posting; server errors shown verbatim; no reload on failure.
- [ ] Table of intervals (start, end, duration, action title, role) under the track; role from `labels.role_of`.
- [ ] Copy keys ru/en (`page_actions`, `action_catalog_heading`, `action_name_column`, `action_title_column`, `action_seed`, `action_seed_titles` (dict passed to `seed_actions`), `action_block_help` (TZ strings verbatim), `action_blocks_unfrozen`, `action_coverage_missing`, `action_add`, `action_delete`, `action_save`, `action_overlap`, `action_crosses_block`, `action_outside_block`, `action_set_start`, `action_set_end`, `action_table_*`, shortcuts help…); `_check_catalogs` passes.
- [ ] Tests `tests/unit/test_ui_actions_page.py`: page renders empty state without project/video; catalog add via route; interval save round-trip; a POST with an outdated `revision` → 409 `action_labels_stale` and the stored intervals are unchanged; role column correct; en page has no Cyrillic in program strings.
- [ ] Browser check on the pilot: seed catalog, mark intervals so that **every seeded action (`digging`, `swing_loaded`, `dumping`, `swing_empty`, `idle`) appears at least once inside the train block and at least once inside the validation block**, within the time ranges of runs A and B from subtask 10; the coverage line shows no red; reload shows them.

## Implementation Notes
- Files: `src/excavation_lab/ui/{navigation,actions_page,copy}.py`, `src/excavation_lab/web/{app,actions_routes}.py`, `templates/actions.html`, `static/action_timeline.js`, `tests/unit/`.
- The player clock is the file clock; convert with `source_interval_from_player` before saving (start offset is 0 on the pilot but must not be assumed).
- Keep `events.html` as the v0.2 stub; do not merge actions into it.

## Dependencies
11; 10 (run ranges to label inside).

## Verification
```bash
uv run pytest tests/unit/test_ui_actions_page.py tests/unit/test_web_shell.py tests/unit/test_ui_locale.py -q
uv run excavation ui   # pilot → Действия: seed → label intervals with keyboard → save → reload
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
