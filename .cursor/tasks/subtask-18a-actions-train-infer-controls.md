# Subtask 18a: Page «Действия» step 3 — train and infer controls with exact gate reasons

## Goal
Start action training from one or more custom analysis runs and start inference on one run, from `/actions`, with every disabled state explained verbatim and progress read from a dedicated endpoint.

## Context
Spec §15 (progress by a later request reading atomic progress JSON; no ETA before measurement; no chart before real rows), §16 (single job). Services: 15b (`check_ready`, `enqueue_action_train`), 16 (`enqueue_action_infer`, `list_action_inferences`). Patterns: `web/training_routes.py` (Start/Cancel), `ui/analysis_page.list_finished_runs`, `static/chart.js`.

## Acceptance Criteria
- [ ] Step 3 «Обучение действий»: multi-select (checkbox list) of completed custom runs for the chosen video (`storage.list_analysis_runs(origin="custom")`; label: detector model id, `[start_s, end_s)`, fps, role blocks it touches); readiness box listing every `check_ready` reason verbatim for the current selection (recomputed on change via `GET /actions/ready?runs=a,b`); Advanced: epochs, window seconds, min confidence, min duration; «Старт» disabled with the first reason, or with the catalog message while any job is active; «Отмена» for the running job.
- [ ] `GET /actions/job?id=<job_id>` returns `{status, epoch, epochs, val_macro_f1, history: [{epoch, val_macro_f1, train_loss}], error_code}` from the job's `progress.json` and `metrics_history.csv` only (no training endpoint reuse); the page polls it; the chart draws only when `history` is non-empty; no ETA.
- [ ] List of action models with `support_train`/`support_validation`, `classes_scored`, and the `selection_unstable` caption when set.
- [ ] Step 3b «Распознавание»: `<select>` action model and `<select>` of a **single** run of the same detector (others listed disabled with «другой детектор»); «Запустить»; list of inferences with status.
- [ ] Routes `POST /actions/train`, `POST /actions/cancel`, `POST /actions/infer` in `web/actions_routes.py`; before `enqueue_*` each calls `label_routes.release_suggestion_adapter(app)` (W1 from 08) and the same active-job check as the other start routes.
- [ ] Copy ru/en for all new strings; `_check_catalogs` passes.
- [ ] Tests `tests/unit/test_ui_actions_results.py` (part 1): readiness reasons render verbatim; Start refused while a sleep job is active; `GET /actions/job` returns an empty history for a fresh job and rows after a synthetic `metrics_history.csv`; traversal `id` refused; infer select excludes runs of another detector.
- [ ] Browser check on the pilot: select the train-block and validation-block runs, readiness turns green, train completes, infer completes.

## Implementation Notes
- Files: `src/excavation_lab/ui/actions_page.py`, `src/excavation_lab/web/actions_routes.py`, `templates/actions.html`, `ui/copy.py`, tests.
- The page never imports torch or loads a model.
- Job ids are validated with the same token rule as `jobs._require_job_id`.

## Dependencies
12, 15b, 16.

## Verification
```bash
uv run pytest tests/unit/test_ui_actions_results.py tests/unit/test_ui_actions_page.py -q
uv run excavation ui   # pilot → Действия → select runs → readiness → train → infer
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
