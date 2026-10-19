# Subtask 21: Acceptance suite — integration and leakage tests

## Goal
Consolidate automated evidence for both new workflows and the leakage invariants; final green run.

## Context
Spec §19 items 2–4, 13, 15–17; master-plan phase F. Unit tests from 03–18 exist; this subtask adds end-to-end tests over synthetic fixtures and a leakage checklist, marked synthetic (not accuracy evidence).

## Acceptance Criteria
- [ ] `tests/integration/test_inapp_label_to_bundle.py` (from 09) extended: a CVAT import on the same split after in-app saves keeps both; `class_supervision.json` in the snapshot marks the in-app negative correctly; bundle `class_mapping.json` equals catalog order.
- [ ] `tests/integration/test_actions_workflow.py`: synthetic `frame_status/detections/track_states` parquet for **three** runs of one detector at 5 fps (run A [0,60) inside train, run B [60,80) inside validation, run C [80,100) inside test; blocks train [0,60), validation [60,80), test [80,100)) with scripted bucket/bed motion (`tests/fixtures/synthetic.py` gains `write_synthetic_run(dir, ...)`) → three `custom` run rows → action catalog seed → intervals in train/validation/test → `run_action_train_job(run_ids=[A, B])` (CPU, few epochs; C is never passed) → `run_action_infer_job(run_id=B)` and `run_action_infer_job(run_id=C)` with the same thresholds → `actions.parquet` + `action_intervals.csv` per inference → `evaluate_actions(B, validation)` returns metrics with support > 0 → `evaluate_actions(C, test)` without `final` refused → `evaluate_actions(B, test, final=True)` refused with `no_frames_for_split` and no `evaluation_test.json` exists → `evaluate_actions(C, test, final=True)` succeeds once and records the B validation file → a second final refused with `final_exists`.
- [ ] Leakage tests `tests/unit/test_action_leakage.py` and additions to `tests/unit/test_label_suggestions.py`:
  - `kept_time_ranges.json` written by the train job contains no range intersecting the test block;
  - the dataframe handed to `build_windows` (spy via monkeypatch) has no row with `source_time_s` inside the test block, for both roles, and **the first row of every kept range has `run_start=True`** (so no delta or window reaches back across a dropped gap);
  - test-block windows never enter `x_train`/`x_val` (assert on `center_time_s` and all frame times);
  - a window straddling a block boundary is absent; a window straddling the A/B run boundary is absent;
  - `Standardizer` mean/std in `standardizer.json` equal the statistics of the train windows only;
  - `action_bundle.json.test_read is False`;
  - validation/test frames get `not_train_frame` from `suggest`;
  - unreviewed in-app frames are absent from `train/_annotations.coco.json`.
- [ ] `tests/unit/test_core_imports.py::_PACKAGE_DIRS` includes `"actions"` and `"ui"` (added in 11) and the test is green against the final tree.
- [ ] Job tests: `action_train` and `action_infer` obey the single lock (second start refused), cancel yields `cancelled` with no `action_models` row.
- [ ] `uv run pytest tests -q --ignore=tests/manual` and `uv run ruff check src tests` green; `tests/manual/test_smoke_train.py` untouched.
- [ ] Any temporary files created during 02–20 outside the repo are removed; none added inside the repo.

## Implementation Notes
- Fixtures write parquet with `pyarrow`; keep them small (< 1 MB).
- Mark synthetic tests with a module docstring: "synthetic; mechanics only, not accuracy evidence".
- Use `monkeypatch.setenv("EXCAVATION_LAB_FAKE_TRAINER", ...)` for the detector part; the action model trains for real on CPU (seconds).

## Dependencies
20.

## Verification
```bash
uv run pytest tests/integration -q
uv run pytest tests/unit/test_action_leakage.py tests/unit/test_label_suggestions.py -q
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
