# Subtask 15b: `action_train` job kind, multi-run readiness gate, action bundle, CLI

## Goal
Run the action trainer (15a) in the worker over **one or more** custom analysis runs of the same detector, with test-block frames excluded before windowing, and save an action bundle with provenance.

## Context
Spec §16 (worker, statuses, atomic progress JSON, artifacts before `completed`), §12.5 analogy for the bundle, §8 roles. The pilot produces runs that each cover part of one block (one ≥120 s inside train, one ≥60 s inside validation), so the trainer takes `run_ids: list[str]`. Job plumbing: `jobs.py` (`_KINDS`, `_WORKERS`, `_child_main`, `mark_completed`, `_ensure_export_kind`), `models/train.py` as the reference worker.

## Acceptance Criteria
- [ ] Readiness `actions/train.check_ready(workspace, project_id, run_ids) -> Readiness(ok, reasons: list[str], detector_model_id, analysis_fps)` with precise reasons: a run missing/incomplete; `model_origin != 'custom'` (baseline refused by name); runs with different `model_id` or `analysis_fps` (list them); runs overlapping in time on the same video; no frozen split; catalog empty; fewer than one labelled interval per role (train, validation); the union of run intervals does not cover every train/validation labelled interval (list the uncovered intervals). The UI (18a) prints these verbatim.
- [ ] `enqueue_action_train(workspace, project_id, run_ids, config) -> JobRecord` writes `action_train_request.json` (`run_ids`, config) in the run dir; `run_action_train_job(workspace, job_id)`:
  1. compute the frozen `train` and `validation` block intervals of the run's video; call `build_features(run_tables, run_id=…, run_index=…, keep_ranges=<those intervals>)` per run (13) so test-block rows are never materialised and each kept range starts with `run_start=True`; concatenate in run order;
  2. write `kept_time_ranges.json` to the job run dir (`[{run_id, role, start_s, end_s}]` — the ranges actually passed, intersected with the run's `[start_s, end_s)`) and a log line per dropped range; the frame passed to `build_windows` contains no test-block row;
  3. windows train and validation (14) — windows never cross `run_start`;
  4. `Standardizer.fit` on train windows only; `train_action_model` with progress JSON per epoch (`epoch, epochs, val_macro_f1`) and `should_stop` reading the job status (`cancel_requested` → stop, no bundle, status `cancelled`);
  5. bundle `projects/<id>/action_models/<action_model_id>/{weights.pt, action_bundle.json, feature_spec.json, standardizer.json, metrics_history.csv, validation_metrics.json, kept_time_ranges.json, checksums.json}`; `storage.insert_action_model`; `mark_completed` with those artifact paths.
- [ ] `action_bundle.json`: `action_model_id, detector_model_id, source_run_ids, analysis_fps, window_frames, window_s, action_catalog (id,name,title), class_order, classes_scored, feature_spec_hash, dataset_hash, support_train, support_validation, selection_unstable, config, test_read: false, library_versions (torch), created_at`.
- [ ] `jobs.py`: kinds `action_train`, `action_infer` and workers of the same names; `_ensure_export_kind` generalised to `_ensure_kinds(connection, required=_KINDS)` rebuilding `lab_jobs` when the CHECK lacks any kind while keeping rows; `_child_main` dispatch; `mark_completed` artifact rule for `action_train` (weights + bundle JSON).
- [ ] CLI `excavation actions train --project --run <id> [--run <id> …] [--epochs …] --json` → job id; non-zero on `LabError`.
- [ ] Expected wall time documented in the subtask outcome: feature build + training for ~2 × 5 fps × 180 s ≈ 900 frames is seconds on CPU; MPS not used.
- [ ] Tests `tests/unit/test_action_train_job.py`: two synthetic runs (train-only and validation-only coverage) → job completes with bundle files, `source_run_ids` of length 2, `action_models` row; `kept_time_ranges.json` has no test range and the dataframe handed to `build_windows` (monkeypatch spy) has no `source_time_s` inside the test block and `run_start=True` on the first row of each kept range; `action_models.source_run_ids` round-trips as `[A, B]`; a run with the baseline origin refused; runs with different `model_id` refused; uncovered interval refused with the interval in the message; second start refused while a sleep job holds the lock; cancel mid-training → `cancelled`, no `action_models` row; `_ensure_kinds` upgrades a `lab.sqlite` created with the old CHECK and keeps its rows.

## Implementation Notes
- Files: `src/excavation_lab/actions/train.py`, `src/excavation_lab/jobs.py`, `src/excavation_lab/storage.py` (`insert_action_model`, `list_action_models`), `src/excavation_lab/cli.py`, tests.
- Use `storage.list_analysis_runs` and `tracking.analyze.run_tables` from 10.
- `torch.save(state_dict)` only; `weights_only=True` on load.
- `keep_ranges` in step 1 is the first barrier; `build_windows` refusing `role="test"` unconditionally (14) is the second.
- If this subtask slips past 2 h, split off **15b-pre** first: `_ensure_kinds`, the `JobKind`/`WorkerName` Literals and `_KINDS`/`_WORKERS` entries, `_child_main` dispatch stubs, and their migration test; then 15b proper carries the gate, worker body, bundle, and CLI.

## Dependencies
11, 13, 14, 15a.

## Outcome

Feature build plus training for about 2 × 5 fps × 180 s (≈ 900 frames) is seconds on CPU. The worker forces `device=cpu`; MPS is not used. The synthetic job in `tests/unit/test_action_train_job.py` (two short runs, 2 epochs) finishes in well under 30 s, torch import included.

## Verification
```bash
uv run pytest tests/unit/test_action_train_job.py tests/unit/test_job_single_lock.py -q
uv run excavation actions train --project mvp --run <train_run_id> --run <validation_run_id> --json
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
