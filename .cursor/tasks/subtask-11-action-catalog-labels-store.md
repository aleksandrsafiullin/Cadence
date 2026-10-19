# Subtask 11: Action catalog and interval labels (compute core + CLI)

## Goal
Create `src/excavation_lab/actions/` with the per-project action catalog and the interval label store in project SQLite, plus `excavation actions catalog …` / `excavation actions label …` CLI commands.

## Context
Spec §20.1 (one state per instant on the excavator timeline), §6.2 (half-open source seconds), TZ §2 (catalog rules) and §5 (help text «В блоке обучения должны быть все действия.»). Role blocks: `datasets/blocks.py` (`RoleBlock`, `assignment_at`), frozen manifest via `datasets/splits.require_frozen`.

## Acceptance Criteria
- [ ] `actions/catalog.py`: `ActionClass(id:int≥0, name:str token ^[a-z][a-z0-9_]{0,63}$, title:str, added_at: datetime UTC)`; `ActionCatalog` 1–16 classes, unique `id`/`name`; `SEED_ACTIONS = ("digging", "swing_loaded", "dumping", "swing_empty", "idle")`; `seed_actions(workspace, project_id, titles: Mapping[str, str])` takes the titles as an argument — the ru/en defaults live in `ui/copy.py` as `action_seed_titles: dict[str, str]` and are passed in by `cli.py` and the web route; `actions/` never imports `ui`; `list_actions`, `add_action` (next free id), `edit_title`; `rename`/`delete` refused (`LabError` code `action_in_use`) when any `action_labels` row uses the name **or** when any action model references it — the model check reads `class_order` from each `action_models.path / action_bundle.json` (not the SQLite row), so a bundle copied in by hand is also respected.
- [ ] `actions/labels.py`: `ActionInterval(label_id, video_id, start_s, end_s, action_name, annotator, created_at)`; `add_interval`, `update_interval`, `delete_interval`, `list_intervals(video_id)`, `replace_intervals(video_id, items)`; rules: half-open, `end_s > start_s`, no overlap within a video (`action_overlap`), action name must exist, the interval must lie inside **one** role block of the current (frozen or draft) blocks — otherwise `action_outside_block` / `action_crosses_block` with the offending boundary in the message; `role_of(interval, blocks)` helper; snapping to the frame step is the UI's job, the store accepts any finite seconds.
- [ ] `storage.py`: tables `action_classes(project_id, id, name, title, added_at, PK(project_id,id), UNIQUE(project_id,name))`, `action_labels(label_id PK, project_id, video_id FK, start_s, end_s, action_name, annotator, created_at, CHECK(end_s>start_s))`, `action_models(action_model_id PK, project_id, detector_model_id, source_run_ids TEXT NOT NULL, path, created_at)` where `source_run_ids` is a canonical JSON array (sorted keys irrelevant; list order preserved, `json.dumps(list, separators=(",", ":"))`); `storage.insert_action_model(workspace, project_id, *, action_model_id, detector_model_id, source_run_ids: Sequence[str], path)` serialises it and `list_action_models(workspace, project_id)` returns `source_run_ids` as a `list[str]`; added to `SCHEMA_SQL` **and** created on open for existing projects (`_ensure_action_tables` next to `_ensure_text_columns`). `delete_project` removes them with the project.
- [ ] CLI (`cli.py`): `excavation actions catalog list|add|seed`, `excavation actions label add|list|delete` with `--json`; non-zero exit on `LabError`.
- [ ] `tests/unit/test_core_imports.py`: `_PACKAGE_DIRS` gains `"actions"` (and `"ui"`, since view-models must not import Starlette/Jinja2 either); the test passes.
- [ ] Tests `tests/unit/test_action_catalog.py`, `tests/unit/test_action_labels.py`: 17th class refused; duplicate name refused; overlap refused; interval crossing a block boundary refused with boundary in message; interval in no block refused; role resolution train/validation/test; tables exist in a project created before this change (open an old-schema sqlite fixture); `insert_action_model`/`list_action_models` round-trip a two-element `source_run_ids` list; `delete` of an action named in a bundle's `class_order` refused even when no SQLite row references it.

## Implementation Notes
- Package `src/excavation_lab/actions/__init__.py` with a docstring: compute core, no Starlette/Jinja2.
- `label_id = uuid4().hex`; `created_at` ISO UTC.
- Reuse `storage.require_half_open_interval` and `media.timebase.require_monotonic`.
- Use the project's current blocks (`ui/data_page.load_split_view` is UI; add `datasets/splits.current_blocks(workspace, project_id, video_id)` returning frozen blocks or the draft proposal blocks) so the label screen works before freeze; training (15) re-validates against the **frozen** manifest only.

## Dependencies
02 (blocks exist on the pilot video). Independent of 03–10.

## Verification
```bash
uv run pytest tests/unit/test_action_catalog.py tests/unit/test_action_labels.py tests/unit/test_storage_project.py tests/unit/test_delete_project.py tests/unit/test_core_imports.py -q
uv run excavation actions catalog seed --project mvp --json
uv run excavation actions label add --project mvp --video v_b91dd7f95a9d46a5ba8a442bbd6314f0 --start 12.0 --end 18.5 --action digging --json
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
