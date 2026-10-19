# Subtask 08: Model suggestions for train frames only

## Goal
On `/label`, a train frame can request boxes from a selected custom bundle; they are drawn as dashed suggestions and become labels only when the operator accepts them. Validation and test frames never receive suggestions.

## Context
Spec §9: "Непроверенные предложения не становятся автоматически ground truth. Validation/test желательно размечать без показа предсказаний". Spec §16: the UI must not hold a second model while a worker runs. Existing: `models/bundle.load_bundle(path) -> DetectorAdapter` (`predict(frame_rgb, timestamp_s)`), `ui/analysis_page.list_bundles`, `jobs.reopen` for active jobs.

## Acceptance Criteria
- [ ] `ui/label_page.suggest(workspace, project_id, video_id, index, model_id, *, detector_cache) -> list[dict] | ActionNote`: refuses (`code=not_train_frame`) for validation/test; refuses (`code=job_running`) while any job is `queued/running/cancel_requested`; refuses (`code=bundle_missing`) when `model_id` is not in `list_bundles`; otherwise decodes the sampled JPEG (RGB), runs `predict`, returns boxes with `class_name, score, x1..y2, suggested: true`, filtered by the pilot display threshold (`load_pilot()` display floor), mapped by **name** into the project catalog; names not in the catalog are dropped with a count in the response.
- [ ] One detector per process and **none while a worker runs**: `web/label_routes.py` keeps at most one adapter on `app.state.suggestion_adapter` (keyed by bundle path) and exposes `release_suggestion_adapter(app)` which calls `adapter.close()` and clears the slot. Every job-start route — `web/training_routes.post_training_start`, `web/analysis_routes` analysis start and export start, and later `web/actions_routes` train/infer (18a) — calls `release_suggestion_adapter(app)` **before** `enqueue_*`. `suggest()` itself runs under the same active-job check the start routes use (`jobs.reopen` / lock row): if a job is `queued/running/cancel_requested` it releases the adapter and returns `job_running`. One `threading.Lock` on `app.state.suggestion_lock` serialises `suggest()` and `release_suggestion_adapter()` so a release cannot interleave with a prediction in flight (Starlette runs sync handlers in a thread pool).
- [ ] `POST /label/suggest` JSON route; the page offers a bundle `<select>` (from `list_bundles`) and a button «Предложить рамки (только train)» disabled with the exact reason when not allowed.
- [ ] JS: suggestions drawn dashed with score; «Принять» per box or «Принять все» converts to normal boxes (`visibility=clear`, `truncated=false`); «Отклонить все» clears; suggestions are never included in `POST /label/frame` unless accepted.
- [ ] Tests `tests/unit/test_label_suggestions.py`: validation frame → `not_train_frame`; fake adapter returns two boxes → response marks `suggested=true`; a save without acceptance stores zero boxes; active sleep job → `job_running` and the fake adapter's `closed` flag is True; after a suggestion, `POST /training/start` (fake trainer) sets `closed=True` on the cached adapter before the job row exists; the same for the analysis start route; a release issued while a (slow fake) `predict` holds the lock waits for it and the response of that `suggest` call is still complete.

## Implementation Notes
- Files: `ui/label_page.py`, `web/label_routes.py`, `web/training_routes.py`, `web/analysis_routes.py`, `web/static/box_labeler.js`, `templates/label.html`, `ui/copy.py`.
- Use the `DetectorAdapter` protocol only (`predict`, `metadata`, `close`); do not import `rfdetr` in the UI layer; tests inject a fake adapter through a `loader` argument and read its `closed` flag.
- Suggestions are not written to disk, not to `predictions.coco.json`, not to the sidecar.
- Baseline COCO weights are not offered here (`list_bundles` lists custom bundles only) — the baseline cannot name catalog classes.

## Dependencies
04b–06, 09 (a bundle to exercise with).

## Verification
```bash
uv run pytest tests/unit/test_label_suggestions.py -q
uv run excavation ui   # pilot, after a smoke bundle exists: train frame → suggest → accept one → save; validation frame → button disabled with reason
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
