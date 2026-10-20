# Subtask 3: Enter a project in tests that render one

## Goal

After `choose_project_id` no longer treats a lone project as selected, every test that expects 200, or a successful project-scoped POST, calls `write_project_file` first. `uv run pytest tests -q` is green. That run is what finishes the gate from subtask 2.

## Context

Spec §5. Subtask 2 removes the `len(ids) == 1` branch in `src/excavation_lab/web/preferences.py`. A test that only calls `create_project` and then GETs `/video`, `/data`, `/training`, `/analysis`, `/quality`, `/actions`, `/events`, or `/label` now gets `303` to `/`. A project-scoped POST that used the same helper now sees no project and fails.

The smoke command on subtask 2 does not import these modules. A green smoke check leaves the suite red. This subtask is the migration. It does not change the gate, the video route, or storage delete rules.

## Acceptance Criteria

- [ ] Every test that expects 200 from a workflow page or `/label`, or a successful project-scoped POST, after one `create_project` and with no `project.txt` yet, calls `write_project_file` for that id before the request.
- [ ] The known files below are updated. Any other failure from `uv run pytest tests -q` is fixed the same way, not by restoring the single-project fallback.
- [ ] Tests that assert an unentered page redirects (`303` to `/`, no `project.txt`) stay that way. Do not write the file just to force a 200.
- [ ] `GET /events` content tests that currently use an empty workspace (`test_events_unavailable.py`, `test_ui_theme.py`) no longer expect the events shell at 200 with zero projects. To keep the unavailable-page assertions, create a project and write `project.txt`, then GET `/events`.
- [ ] `test_label_page.py` `_lab` is shared with `test_web_static.py`. If every caller of `_lab` wants an entered project, write the file inside `_lab`. Callers that must stay unentered must not use that helper.
- [ ] `GET /static/...` and `GET /jobs/.../progress` are left alone when they already pass without `project.txt`.
- [ ] `choose_project_id` still has no single-project fallback when this subtask ends.

## Implementation Notes

Known files:

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

Also sweep tests subtask 1 and 2 already touched, including the new `GET /video?video=<id>` selection test, if they `create_project` and then expect 200 without `write_project_file`.

`write_project_file` is `excavation_lab.web.preferences.write_project_file`. It validates the id and creates `workspace/ui/`. Pass the same workspace the test passed to `create_app`.

Do not add a fixture that writes `project.txt` for the whole session. Tests that cover the empty workspace and the 303 must keep a missing file.

A POST that expects a validation error (missing path, missing confirm) can stay without a project when the assertion is that error. A POST that expects the handler to mutate the project (add a video, freeze a split, start a job the test then reads) needs the file.

Search after the edits:

```bash
rg -n "create_project\(" tests -g '*.py'
```

For each hit, the next request that expects a rendered workflow page or a successful project write must have `write_project_file` earlier in that test or in the helper that built the workspace.

## Dependencies

Subtask 2. The fallback must already be gone, or this migration has nothing to prove.

## Verification

```bash
uv run pytest tests -q
uv run ruff check src tests
```

This is the completion check for the entered-project gate. Do not mark subtask 2 done on its shorter file list.
