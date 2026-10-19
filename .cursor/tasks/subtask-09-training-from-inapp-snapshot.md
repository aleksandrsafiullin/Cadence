# Subtask 09: Training from an in-app snapshot → bundle on Analysis

## Goal
Prove the chain snapshot (in-app labels) → «Обучение» → worker → RF-DETR fine-tune → model bundle → listed on «Анализ», with the fake trainer in tests and `--profile smoke` as a manual wiring check. Fix whatever blocks it.

## Context
Spec §12 (profiles, bundle), §16 (worker, single lock), §19 items 4, 13, 15. Existing: `models/train.enqueue_train` (`_require_reviewed_images`, `train_request.json`), `run_train_job`, `FAKE_TRAINER_ENV`, `models/bundle.publish_train_bundle`, `ui/training_page`, `web/training_routes.py`, `ui/analysis_page.list_bundles`.

## Acceptance Criteria
- [ ] Integration test `tests/integration/test_inapp_label_to_bundle.py`: synthetic MP4 (`tests/fixtures/synthetic.py`) → project with catalog `excavation` → blocks (train/validation/test) → freeze → `sample_frames` → `label_store.save_frame_label` on ≥2 train and ≥1 validation frames (reviewed, one reviewed-empty) → `snapshot_dataset` → `enqueue_train(profile="smoke")` with `EXCAVATION_LAB_FAKE_TRAINER` → `run_train_job` → `model_versions` row with `weights.pth` → `list_bundles` contains it → `class_mapping.json` names equal the catalog order → `resolved_train_config.json` has `run_test=False` → the opened-paths log contains no `test/` path.
- [ ] «Обучение» page on the pilot shows the new dataset version, profile `smoke` default, Start enabled only with no active job; a second Start while running is refused with the catalog message.
- [ ] Progress/loss drawn only from `metrics_history.csv` with data rows (existing `should_draw_chart`); no ETA before measured epochs.
- [ ] Manual (optional, record result): real `smoke` on the pilot snapshot on this machine (device auto); if it fails on RF-DETR/MPS, record the error and the fix or the decision in `docs/DECISIONS.md`. It is a wiring check, not quality.
- [ ] Any blocker found (e.g. catalog/class-head size, dataset readiness message, missing copy) is fixed in this subtask with a test.

## Implementation Notes
- Files touched only as needed: `models/train.py`, `models/bundle.py`, `ui/training_page.py`, `web/training_routes.py`, `templates/training.html`, `ui/copy.py`.
- Do not start `pilot-small` automatically anywhere; profile default stays `smoke`.
- Check free disk space before a real smoke run (`df -h .`, ≥10 GiB).
- Keep `run_test=False`; never pass test paths to the library.

## Dependencies
07.

## Verification
```bash
uv run pytest tests/integration/test_inapp_label_to_bundle.py tests/unit/test_train_supervision.py tests/unit/test_job_single_lock.py tests/unit/test_ui_training_state.py -q
uv run excavation ui   # pilot → Обучение → smoke → status/epoch from logs → bundle id appears on Анализ
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
