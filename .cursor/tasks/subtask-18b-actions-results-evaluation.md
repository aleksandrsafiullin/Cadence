# Subtask 18b: Page «Действия» step 4 — result timeline, table, evaluation

## Goal
Read an action inference as a timeline + table with `unknown` shown honestly, compare it with the labelled intervals without leaking test labels, and run the validation evaluation (and the one-time final test) from the page.

## Context
Spec §15, §18 (no invented metrics), §20.1 (`unknown` is a valid state), §8 (test only for the final check). Services: 16 outputs (`actions.parquet`, `action_intervals.csv`, `action_run.json`), 17 (`evaluate_actions`).

## Acceptance Criteria
- [ ] Selected inference shows: timeline strip under the player (spans coloured per action, `unknown` grey/hatched), table (start, end, duration, action title, mean confidence), `unknown_share`, CSV download `GET /actions/export?job=`.
- [ ] Labelled-interval comparison strip above it renders **only train and validation intervals** (role from the frozen blocks); test-block intervals are drawn only when `evaluation_test.json` exists for this action model; otherwise the test block shows a caption «интервалы блока итоговой оценки скрыты до финальной проверки».
- [ ] Evaluation block: «Оценить на блоке проверки» → table per class (`precision/recall/f1/support_frames/support_intervals`), `accuracy`, `macro_f1`, `classes_scored`, `evaluated_frames`, `unlabelled_frames`, `unknown_rate`, `selection_unstable` caption; nulls render «не рассчитано» / `not calculated`.
- [ ] Collapsed «Итоговая оценка (test)»: enabled only when a validation evaluation exists for an inference with the same `action_model_id`, `min_confidence`, `min_duration_s` (W7 rule from 17); explains it is written once; shows `final_thresholds_mismatch` / `final_exists` messages verbatim.
- [ ] Without a run / model / inference each section shows its specific empty state and no numbers.
- [ ] Routes `GET /actions/export`, `POST /actions/evaluate` (`split`, `final`).
- [ ] Copy ru/en; `_check_catalogs` passes.
- [ ] Tests `tests/unit/test_ui_actions_results.py` (part 2): inference table from a synthetic `action_intervals.csv`; comparison strip contains train/validation spans and no test span before `evaluation_test.json`, and the test span after; evaluation nulls render «не рассчитано»; final button disabled with mismatch reason; CSV route refuses a traversal `job`.
- [ ] Browser check on the pilot: spans and table visible; validation evaluation printed with support counts.

## Implementation Notes
- Files: `src/excavation_lab/ui/actions_page.py`, `src/excavation_lab/web/actions_routes.py`, `templates/actions.html`, `static/action_timeline.js` (read-only span rendering), `ui/copy.py`, tests.
- Read `action_intervals.csv` with the stdlib `csv` module in the view-model; parquet is not read by the page.

## Dependencies
18a, 17.

## Verification
```bash
uv run pytest tests/unit/test_ui_actions_results.py -q
uv run excavation ui   # pilot → Действия → results → evaluate (validation)
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
