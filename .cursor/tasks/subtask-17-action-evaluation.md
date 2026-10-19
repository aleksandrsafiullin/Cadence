# Subtask 17: Action evaluation on validation; explicit final test command

## Goal
`actions/evaluate.py` reports frame-level accuracy and per-class precision/recall/F1 with support for an action inference against the interval labels of one role block; validation by default, test only via an explicit `--split test --final`.

## Context
Spec §8 roles, §12 ("финальная оценка — отдельная команда"), §18 (no invented metrics), §19 item 16. `evaluate --final` conventions in `.cursor/rules/web-ui-and-worker.mdc`: it evaluates, never trains or retunes.

## Acceptance Criteria
- [ ] `evaluate_actions(workspace, project_id, infer_job_id, *, split="validation", final=False) -> ActionEvaluation`: joins `actions.parquet` frames with labelled intervals whose role is `split` under the **frozen** blocks; frames with no label are excluded and counted (`unlabelled_frames`); predicted `unknown` on a labelled frame counts as a miss (`unknown_rate` reported separately); metrics: `accuracy`, per class `precision, recall, f1, support_frames, support_intervals`, `macro_f1`, `confusion` (names × names + `unknown` column), `evaluated_frames`, `selection_unstable = evaluated_frames < 300`.
- [ ] Refuses with `no_frames_for_split` when `evaluated_frames == 0` (the inference run does not overlap any labelled interval of `split`), for **both** splits, before any file is written; the message names the split and the inference run's `[start_s, end_s)`.
- [ ] `split="test"` requires `final=True` **and** an existing validation evaluation (`evaluation_validation_*.json` in the same action bundle directory) whose inference job has the same `action_model_id`, `min_confidence`, and `min_duration_s` as the inference being evaluated (read from both `action_run.json` files); otherwise refuses with `final_thresholds_mismatch` listing the differing values; on success writes `evaluation_test.json` once (recording `action_model_id`, `min_confidence`, `min_duration_s`, `validation_evaluation_file`, `infer_job_id`) and refuses a second write (`final_exists`); validation writes `evaluation_validation_<infer_job_id>.json`.
- [ ] Macro-F1 follows 15a: averaged over classes with `support_frames > 0`; `classes_scored` stored in the report.
- [ ] A class with `support_frames == 0` reports `precision/recall/f1 = null` (rendered «не рассчитано»), never 0.
- [ ] `format_action_evaluation(report, book)` text used by CLI and UI; CLI `excavation actions evaluate --project --job <infer_job_id> [--split validation|test --final] --json`.
- [ ] Tests `tests/unit/test_action_evaluate.py`: synthetic frames/intervals → exact precision/recall; unlabelled frames excluded; `unknown` miss accounting; test without `--final` refused; final without a validation evaluation refused (`final_thresholds_mismatch`); final after a validation evaluation with a different `min_confidence` refused; final with matching thresholds succeeds and records them; second final refused; zero-support class is null and excluded from macro-F1; an inference covering only validation time evaluated with `split="test", final=True` → `no_frames_for_split` and no `evaluation_test.json`; same code for `split="validation"` on a test-only inference.

## Implementation Notes
- Files: `src/excavation_lab/actions/evaluate.py`, `src/excavation_lab/cli.py`, tests.
- Pure numpy/pandas; no sklearn.
- Evaluation reads the frozen manifest to resolve roles; intervals saved against draft blocks that later moved are re-resolved here (a label now outside its role is counted under `relabelled_outside_role`, not silently used).

## Dependencies
16.

## Verification
```bash
uv run pytest tests/unit/test_action_evaluate.py -q
uv run excavation actions evaluate --project mvp --job <infer_job_id> --json
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
