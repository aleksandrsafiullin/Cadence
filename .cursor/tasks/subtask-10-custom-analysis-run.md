# Subtask 10: Custom-bundle analysis run and run lookup for actions

## Goal
Two analysis runs with the trained bundle on the pilot (one ≥120 s inside the train block, one ≥60 s inside the validation block) write `frame_status.parquet`, `detections.parquet`, `track_states.parquet`, and the project can list such runs (`model_origin='custom'`, time range, fps, model id) for the multi-run action trainer (15b).

## Context
Spec §13, §17.1–§17.3. Existing: `tracking/analyze.enqueue_analyze`, `storage.insert_analysis_run/get_analysis_run` (`runs` table with `model_origin`, `analysis_fps`, `start_s`, `end_s`, `model_id`), `tracking/export_tables.py` column lists, Analysis page controls. Actions (13–18) need: which runs are custom, what interval they cover, and where their parquet lives.

## Acceptance Criteria
- [ ] `storage.list_analysis_runs(workspace, project_id, *, video_id=None, origin=None) -> list[dict]` ordered by `created_at`, fields: `run_id, job_id, video_id, model_id, model_origin, analysis_fps, start_s, end_s, created_at`, plus `run_dir` resolved via `jobs.run_directory` and `completed: bool` from `lab.sqlite` job status.
- [ ] `tracking/analyze.run_tables(run_dir) -> RunTables` (paths of the three parquet files, raising `LabError("run_incomplete")` when any is missing).
- [ ] Manual: on the pilot, «Анализ» with the smoke bundle at 5 fps: run A ≥120 s fully inside the train block, run B ≥60 s fully inside the validation block (choose ranges that contain the action intervals you plan to label in 12/20); both complete; the three parquet files exist per run; `runs` rows have `model_origin='custom'` and the same `model_id`/`analysis_fps`; frame_status `analyzed=true` rows exist; «Анализ» shows source timestamps and model version. Record wall time per run (expected on this Mac at 5 fps with Nano: roughly 1–3 min per 120 s on MPS, 3–8 min on CPU — measure, do not assume).
- [ ] Detections for classes outside `DetectionClassName` are not needed for the pilot catalog; record the limitation (`Detection.class_name` enum) in `docs/DECISIONS.md` as a known boundary for custom catalogs.
- [ ] Tests `tests/unit/test_storage_runs.py`: `list_analysis_runs` filters by origin and video; `run_tables` refuses a missing parquet.

## Implementation Notes
- Files: `src/excavation_lab/storage.py`, `src/excavation_lab/tracking/analyze.py` (or `export_tables.py`), `tests/unit/`.
- `end_s` may be null for a full-video run; return it as `None`, not duration.
- Do not add another lock or job kind here.

## Dependencies
09 (a bundle exists).

## Verification
```bash
uv run pytest tests/unit/test_storage_runs.py tests/unit/test_analyze_grid.py -q
uv run excavation ui   # pilot → Анализ → custom bundle → run A (train, ≥120 s) and run B (validation, ≥60 s) → parquet in runs/<job_id>/
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```

## Result
Runs A and B were on a deleted copy of `workspace/` because `mvp` has no custom bundle yet. Subtask 20 should repeat them on `mvp` after a real train.
Smoke profile only, model `smoke-d1b06aa606094e7ca8469bf77d586a75`, `model_origin=custom`, `analysis_fps=5`. Train wall 60.2 s.
Run A `[0, 120)` inside train `[0, 760)`: wall 135.1 s, 600 analyzed frames. Run B `[770, 840)` inside validation `[770, 1020)`: wall 60.2 s, 350 analyzed frames.
Each run has `frame_status.parquet`, `detections.parquet`, and `track_states.parquet`. `workspace/ui/project.txt` is still `pilot`; real `mvp` and `pilot` were not trained or analysed.
