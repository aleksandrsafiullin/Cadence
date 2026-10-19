# Subtask 07: Data page counts from the store; snapshot gate on reviewed train + validation

## Goal
Step 3 on «Данные и разметка» reflects in-app labels, and «Снимок датасета» becomes available as soon as every video is frozen **and** the project has ≥1 trainable reviewed frame in `train` and ≥1 in `validation`, with the reason spelled out otherwise.

## Context
Spec §15 («количество reviewed/unreviewed/excluded», «до фиксации split экспорт обучающего snapshot заблокирован»), §10 (import checks: "наличие хотя бы нескольких примеров каждого нужного класса в train и validation" stays a validation **warning**, not a gate). Today `data_page.snapshot_gate` checks only freeze; `workflow_steps._label_ready` requires `unreviewed == 0`, which is unreachable on a 600-frame batch and wrong (unreviewed frames are simply not used).

## Acceptance Criteria
- [ ] `data_page.snapshot_gate` → enabled iff all videos frozen and `label_store` counts give `train.reviewed_trainable ≥ 1` and `validation.reviewed_trainable ≥ 1`. Reason strings (ru/en): not frozen (existing), «Нет проверенных кадров в блоке обучения», «…в блоке проверки», each with the current counts.
- [ ] `workflow_steps`: step `label` is `done` when the gate's review condition holds; `current` when frames exist; the status caption shows `reviewed / unreviewed / excluded` for this video from `label_page.progress` (not «не рассчитано» once a sample exists, `0` is a real zero here because the manifest is known).
- [ ] Review heading on step 3 shows **both** the per-video counts (this video, per split) and the project-wide counts per split from the store, each labelled as such (`label_counts_this_video`, `label_counts_project`), plus the latest `dataset validate` counts separately (the latter stays «не рассчитано» until a snapshot exists). The gate caption states that the gate is project-wide.
- [ ] After a snapshot, `review_warnings` from `validate_dataset` (missing class examples etc.) are shown as warnings, not as a disabled button.
- [ ] `clip.mp4` (frozen, 4 sampled, unlabelled frames — no annotation batch) in the pilot does not block the gate and contributes zero frames to the snapshot.
- [ ] Tests: `tests/unit/test_ui_data_freeze.py`, `tests/unit/test_ui_empty_states.py` updated; new cases: gate closed with reviewed train only; open with 1+1; step 3 `done` with unreviewed frames remaining.

## Implementation Notes
- Files: `src/excavation_lab/ui/data_page.py` (`snapshot_gate`, `workflow_steps`, `_label_ready`, `video_review_counts`), `web/data_routes.py::_editor`, `templates/data.html`, `ui/copy.py`.
- Count "trainable reviewed" as `record.trainable and record.review_status is reviewed` (reviewed-empty counts; excluded does not).
- `models/train.py::_require_reviewed_images` remains the second gate at train time; both must agree.
- Keep `snapshot_other` behaviour (another video unfrozen) intact.

## Dependencies
06.

## Verification
```bash
uv run pytest tests/unit/test_ui_data_freeze.py tests/unit/test_ui_empty_states.py tests/unit/test_label_page.py -q
uv run excavation ui   # pilot: after labelling train+validation frames, snapshot button enables; click → dataset_versions/<id> written, counts shown
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
