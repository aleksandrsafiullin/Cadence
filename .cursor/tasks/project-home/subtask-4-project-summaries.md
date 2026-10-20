# Subtask 4: Read project summary rows

## Goal

A Starlette-free function returns one summary per project directory: display name, project id, catalog id, stored created time, and four sqlite counts. Unknown fields are `None`. A successful count of zero stays `0`. An escaping sqlite symlink does not raise.

## Context

Spec §15 and the project rule that missing measurements render «не рассчитано», never an invented 0. `ProjectRecord` in `src/excavation_lab/storage.py` is only `project_id`, `name`, `workspace`, `project_dir`, `sqlite_path`. The `projects` table also has `catalog_id` and `created_at` (`SCHEMA_SQL`, written by `_init_database` via `_utc_now()`). `list_projects` selects name and id only.

`connect()` follows symlinks. `project_sqlite` raises `PathNotAllowedError` when the resolved file leaves the workspace. `list_projects` calls `_project_record`, which calls `project_sqlite`, and re-raises that error (`exc.code == "path_not_allowed"`). Matching `list_projects` and also “do not raise” cannot both be true. This reader is stricter: it does not re-raise.

The home page in subtask 5 displays these rows. This subtask does not render HTML and does not change create, rename, or delete.

## Acceptance Criteria

- [ ] `storage.list_project_summaries(workspace)` returns a list of typed dicts. `ProjectRecord` and `list_projects` keep their current keys and their current raise behavior.
- [ ] Each readable project row includes `project_id`, `name`, `catalog_id`, `created_at`, `video_count`, `episode_count`, `dataset_count`, and `model_count`.
- [ ] The database is opened only through `project_sqlite` (or the same resolve check it performs). There is no direct `sqlite3.connect` on a path that skipped that check.
- [ ] Counts are `COUNT(*)` on `videos`, `episodes`, `dataset_versions`, and `model_versions`, using one connection per project. A project with no videos and a successful query has `video_count == 0`.
- [ ] `catalog_id` and `created_at` are the stored strings. `created_at` is not converted to another timezone.
- [ ] `sqlite3.Error` is caught per statement. A missing `videos` table sets `video_count` to `None` and still returns `catalog_id`, `created_at`, and the other counts when those statements succeed.
- [ ] A NULL or a failed read for one field yields `None` for that field only. No field is filled with `0` because a query failed.
- [ ] Directories that fail `validate_project_name`, hidden names, and directories without `project.sqlite` are skipped, same as the directory walk in `list_projects`.
- [ ] An escaping symlink: `project_sqlite` raises `PathNotAllowedError`, and `list_project_summaries` does not re-raise. The result includes one row with that directory’s `project_id` and `None` for name, catalog id, created time, and all four counts. A test shows `list_projects` still raises on that same workspace.
- [ ] The function does not open parquet, does not call `jobs.reopen`, and does not read the workspace `lab.sqlite`.
- [ ] `src/excavation_lab/storage.py` and a thin `src/excavation_lab/ui/project_home.py` do not import Starlette, Jinja2, or Streamlit. `project_home` maps `None` to `catalog(locale).not_calculated` and leaves integers as decimal strings. It does not invent mAP, loss, or ETA keys.
- [ ] Unit tests use a synthetic project created with `create_project` and `insert_video`. They do not need a pilot MP4.

## Implementation Notes

- Walk project directories the way `list_projects` does, then call `project_sqlite` inside `try`. On `PathNotAllowedError` or `LabError`, append the null row and continue. Do not call `list_projects` and reuse its exception policy.
- A symlink whose target stays inside the workspace is a normal database. `project_sqlite` returns the resolved path. Read it.
- Select `project_id, name, catalog_id, created_at` from `projects`, then run each `COUNT(*)` in its own `try`/`except sqlite3.Error`. One missing table must not blank the fields already read.
- If `catalog_id` or `created_at` is missing, blank, or not a string, store `None`. Do not substitute the seed catalog id.
- An empty `videos` table is `0`. Dropping the `videos` table in a test is the `None` case for that count only. Assert those two cases separately.
- `ui/project_home.py` accepts the workspace and a `Catalog` and returns display rows: the eight fields as strings, plus `project_id` unchanged for form values. Subtask 5 calls this module. Keep SQL out of the template.
- Follow the import ban already used in `tests/unit/test_workflow_stages.py` `test_workflow_module_stays_off_the_page_server`: parse the new module with `ast` and reject `starlette`.
- Do not add summary fields to `ProjectRecord`. Callers such as the sidebar and the CLI pass that dict around.
- Delete lock behavior stays in `_refuse_locked_delete`. This reader must not take the workspace lock. Do not weaken the symlink check inside `project_sqlite`.

## Dependencies

None on subtasks 1–3. Subtask 5 needs this function. Implement it before subtask 5, in parallel with 1–3 if useful.

## Verification

```bash
uv run pytest tests/unit/test_project_home.py tests/unit/test_workflow_stages.py -q
uv run ruff check src/excavation_lab/storage.py src/excavation_lab/ui/project_home.py tests/unit/test_project_home.py
```
