# Subtask 20: Pilot walkthrough 1→5 in the browser; decisions log

## Goal
Walk the whole MVP on project `mvp` with the real video using only the UI, fix the friction found, and record the design decisions in `docs/DECISIONS.md`.

## Context
Spec §19 ("v0.1 функционально готова, когда пользователь может провести весь workflow"), §25. "Приложение работает" and "модель точна" are separate acceptance items; this subtask claims only the first.

## Acceptance Criteria
- [ ] Walkthrough script executed and its outcome written at the end of this file (what worked, what was fixed, wall times):
  1. «Видео»: project `mvp` selected; real video plays; metadata shown.
  2. «Данные»: blocks (train/validation/test) frozen; `diagnostic` sample; `/label`: ≥20 train and ≥8 validation frames reviewed (some reviewed-empty if true); snapshot created; counts honest.
  3. «Обучение»: `smoke` run on the snapshot completes on this machine (device auto) **or** the concrete failure is recorded; bundle visible on «Анализ».
  4. «Анализ»: run A ≥120 s fully inside the train block and run B ≥60 s fully inside the validation block, same bundle, 5 fps; both complete; parquet present; wall time per run recorded (measured, not assumed; MPS vs CPU noted).
  5. «Действия»: seed catalog; intervals such that **every seeded action — `digging`, `swing_loaded`, `dumping`, `swing_empty`, `idle` — occurs at least once in train (inside run A) and at least once in validation (inside run B)**, coverage line shows no red; action training on runs A+B completes on CPU (expected seconds for ~900 frames; record it) with `kept_time_ranges.json` free of test time; inference on run B; timeline/table visible; validation evaluation printed with support counts and `classes_scored`.
- [ ] Every friction point found (missing link, unclear reason, wrong default, slow request) is either fixed in this subtask with a test or listed under «Follow-ups» with a reason.
- [ ] `docs/DECISIONS.md` entries: «In-app box labeler alongside CVAT» (label store upsert, same files); «Action classifier in MVP» (why a trained temporal model is added despite §20 "not at the start": human-labelled intervals, separate evaluation, rule-based events stay v0.2); «Sampler cost on 60 fps video» (measured numbers, PTS-only tick pass); «`Detection.class_name` enum boundary for custom catalogs»; «Action training over several runs» (same detector and fps, run boundaries never crossed, test-block frames sliced out before windowing); «Final test evaluation requires matching validation thresholds».
- [ ] No «не рассчитано» replaced by a number anywhere during the walk; no demo values.
- [ ] Free disk space checked before training/analysis; temporary files under `/tmp` removed; retained large files listed (run parquet, bundles) in the outcome section.

## Implementation Notes
- Use `uv run excavation ui`; record wall times with the browser dev tools or `time` around CLI equivalents.
- Do not delete or reset `workspace/`; new runs and models are additive.
- If a real smoke train cannot run (RF-DETR install/device), the fake-trainer integration test (09) remains the automated evidence and the decision log says so.

## Dependencies
19.

## Verification
```bash
uv run excavation ui
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```

## Outcome (2026-10-19)

Walked the real UI on `127.0.0.1:8519`. Port 8517 was not listening. Project `mvp`, video `v_b91dd7f95a9d46a5ba8a442bbd6314f0`. `pilot` was not edited. `workspace/ui/project.txt` restored to `pilot`. No labels, no training, no analysis. Disk free before the walk: 16 GiB. No temporary files and no new bundles or parquet from this walk.

1. «Видео»: project `mvp`; player src `/media/mvp/v_b91dd7f95a9d46a5ba8a442bbd6314f0` (206, `video/mp4`, 616963606 bytes); 1920×1080, 1271.084 s, readyState 4, played to 0.197 s. Metadata under the player. Stepper present. Failure alert hidden.
2. «Данные»: frozen train `[0, 760)`, validation `[770, 1020)`, test `[1030, 1271.084)`, gaps `[760, 770)` and `[1020, 1030)` unassigned. Diagnostic sample 60+20, 0 reviewed. Snapshot button disabled: «Нет проверенных кадров в блоке обучения. Проверенных обучаемых: обучение 0, проверка 0.» Label links stay on this video id.
3. «Обучение»: not started. No snapshot. Alert `dataset_not_ready` with that same sentence. No training jobs. Loss «не рассчитано».
4. «Анализ»: not started. Stepper says a custom bundle is required and the baseline is not substituted. Subtask 10's runs A (`[0, 120)`, 135 s) and B (`[770, 840)`, 60 s) on a deleted copy stay pending until after a real train.
5. «Действия»: catalog empty. «Нет завершённого custom-прогона анализа для этого видео.» «Модели действий ещё нет.» Train and infer disabled. Epoch and unknown share «не рассчитано».

`/label?video=v_b91dd7f95a9d46a5ba8a442bbd6314f0&frame=0` opened frame 0 of that video (train, 0.000 s, image 1920 px). Next frame is 720 of the same id. Suggestions disabled: no custom model. Nothing was saved.

No 500, no link to another video, stepper on every page. No product change.

Follow-ups: operator must label frames before a snapshot, a detector train, or an action train. Repeat analysis A/B only after that train. Training «Старт» is enabled while the dataset select is disabled; `request_training` refuses an empty `dataset_id` and does not enqueue. Left as-is.
