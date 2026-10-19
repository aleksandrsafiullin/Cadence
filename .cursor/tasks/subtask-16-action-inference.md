# Subtask 16: Action inference → `actions.parquet` and merged intervals

## Goal
Run an action bundle over **one** analysis run, write per-frame probabilities and merged `[start_s, end_s)` intervals with an honest `unknown`, through the `action_infer` job kind and the CLI.

## Context
Spec §17 (results are Parquet + CSV with source timestamps; empty means unknown, not zero), §20.1 (one state per instant, `unknown` allowed), §16 (`completed` only with artifacts).

## Acceptance Criteria
- [ ] `actions/infer.py`: `load_action_bundle(path)` verifies `checksums.json`, feature-spec hash, class order, and loads `standardizer.json`; `predict_frames(bundle, features_df) -> DataFrame[source_frame_index, source_time_s, p_<name>…, action_name, confidence, frame_mask]` with a centred sliding window over the single run's features (built with `build_features(run_id=…)`, `run_index=0`); frames without a full window or with `frame_mask=0` → `action_name="unknown"`, `confidence=NaN`.
- [ ] `merge_intervals(frame_df, *, min_confidence, min_duration_s) -> list[ActionSpan]`: argmax below `min_confidence` → `unknown`; run-length merge over consecutive frames; spans shorter than `min_duration_s` → `unknown`; adjacent `unknown` spans merged; spans are half-open on frame times, the last span ends at the last frame time + `1/analysis_fps`.
- [ ] Outputs in `runs/<job_id>/`: `actions.parquet` (frame rows), `action_intervals.csv` (`start_s, end_s, duration_s, action_name, mean_confidence, frames`), `action_run.json` (`source_run_id, action_model_id, detector_model_id, min_confidence, min_duration_s, unknown_share`), all with a decimal point and UTF-8.
- [ ] Job: `enqueue_action_infer(workspace, project_id, run_id, action_model_id, *, min_confidence=None, min_duration_s=None)` (exactly one `run_id`); worker `run_action_infer_job`; refuses when the action bundle's `detector_model_id` differs from the run's `model_id` (`action_detector_mismatch`), when `analysis_fps` differs from the bundle (`action_fps_mismatch`), or the run is not custom; `storage` table `action_inferences(job_id PK, project_id, source_run_id, action_model_id, path, created_at)` + `list_action_inferences`.
- [ ] CLI `excavation actions infer --project --run --action-model [--min-confidence] [--min-duration] --json`; `excavation actions export --job <id> --csv <path>` copies `action_intervals.csv` (Excel BOM option as the other exports).
- [ ] Tests `tests/unit/test_action_infer.py`: synthetic bundle + features → `unknown` where mask is 0; low confidence → `unknown`; a 0.6 s burst inside a 10 s span becomes `unknown`, not a label; interval edges half-open and non-overlapping; CSV round-trips with `pandas.read_csv`; detector mismatch refused.

## Implementation Notes
- Files: `src/excavation_lab/actions/infer.py`, `src/excavation_lab/jobs.py` (worker dispatch), `src/excavation_lab/storage.py`, `src/excavation_lab/cli.py`, tests.
- `unknown` is never a trainable class; it exists only here and in evaluation.
- Do not post-process with the mean cycle period or any prior; the spec forbids inventing missed phases.

## Dependencies
15b.

## Verification
```bash
uv run pytest tests/unit/test_action_infer.py -q
uv run excavation actions infer --project mvp --run <run_id> --action-model <id> --json
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
