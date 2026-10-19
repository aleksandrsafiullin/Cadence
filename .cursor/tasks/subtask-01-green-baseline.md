# Subtask 01: Green baseline — align video-page tests with the project-create panel

## Goal
Make `uv run pytest tests -q --ignore=tests/manual` and `uv run ruff check src tests` green without removing the project-create `<details>` panel the TZ requires.

## Context
TZ §6 (`docs/TZ_PROJECT_CLASSES.md`): project creation lives on «Видео» as a collapsible panel (`src/excavation_lab/web/templates/video.html` line ~32: `<details class="lab-details" id="project-create-panel"><summary>…`). Three tests in `tests/unit/test_ui_video_page.py` (lines ~315–316, ~1432–1433, ~1448) still assert `"<summary>" not in text`, written when the only `<details>` on that page was «Дополнительно». The intent of those assertions was "no Advanced section on the video page", not "no details element at all".

## Acceptance Criteria
- [ ] `test_empty_project_names_the_workspace_root`, `test_added_file_shows_probe_metadata_and_plays_from_disk`, `test_english_video_page_keeps_only_the_language_endonym` pass.
- [ ] The assertions now check: `<summary>Дополнительно</summary>` / `<summary>Advanced</summary>` absent **and** the project-create summary (`book.create_project_heading`) present exactly once.
- [ ] The English variant asserts the English heading, not a Cyrillic string.
- [ ] No template change unless a second, unintended `<details>` is found on the video page.
- [ ] Full suite green; ruff green.

## Implementation Notes
- Files: `tests/unit/test_ui_video_page.py` only (plus `video.html` if a real markup defect is found).
- Use the catalog (`excavation_lab.ui.copy.catalog("ru").create_project_heading`) in the assertion instead of a literal so copy edits do not break it again.
- Do not touch `project_catalog.js`.

## Dependencies
None.

## Verification
```bash
uv run pytest tests/unit/test_ui_video_page.py -q
uv run pytest tests -q --ignore=tests/manual
uv run ruff check src tests
```
