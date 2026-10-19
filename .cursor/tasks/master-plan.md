# Master plan: working end-to-end MVP (video → frames → in-app labels → detector → actions)

Spec: `EXCAVATION_VIDEO_LAB_SPEC.md` (§5, §7–§10, §12, §15–§17, §19, §20). Class-catalog TZ: `docs/TZ_PROJECT_CLASSES.md` (implemented). Rules: `.cursor/rules/project-core.mdc`. Decisions: `docs/DECISIONS.md`. Revision 3 after two plan-review rounds (C1–C2, W1–W9, N1–N4, suggestions applied).

## Overview

The operator must walk five steps through the Starlette UI without CVAT and without the CLI:

1. choose a video (exists: «Видео»);
2. cut role blocks, sample frames, **label boxes in the app** (new), snapshot;
3. fine-tune RF-DETR on that snapshot (exists; verify from an in-app snapshot);
4. label **actions** as time intervals and train a small temporal classifier on detector-run features (new module `src/excavation_lab/actions/`);
5. run action inference on an analysis run and read the result (new page «Действия»).

The working tree already holds the class-catalog work (uncommitted). Nothing is reset, stashed, or committed by this plan. `workspace/` is never deleted; the MVP project (`workspace/projects/mvp`, real 21-min video `v_b91dd7f95a9d46a5ba8a442bbd6314f0`, blocks frozen, 80 sampled frames) is the manual exercise target. `workspace/projects/pilot` stays untouched (one-shot freeze; it holds synthetic `clip.mp4` `v_a834183bfe984920be63a24ef1d7875b` that is frozen and has **4 sampled, unlabelled frames and no annotation batch**) is the manual exercise target.

### Fixed design decisions (so subtasks do not re-decide them)

- **One label store per (video, split)**: `projects/<id>/annotation_batches/<video_id>__<split>/{imported_labels.json, attributes_sidecar.json, images/}` — the directory CVAT import already writes. `datasets/label_store.py` owns the writer with **upsert-by-frame** semantics that touch only incoming JPEGs (per-file tmp + `os.replace`, JSON pair replaced atomically, per-directory `threading.Lock`); `coco_import._publish` and the in-app labeler both call it. `load_imported_frames`, `snapshot_dataset`, `validate_dataset`, `class_supervision` consume both sources identically because there is only one source.
- `reviewed_at` is tz-aware UTC and set when the operator marks a frame reviewed/excluded; `annotator` is a workspace preference (default `operator`). Autosave of boxes without a review keeps `review_status=unreviewed` → not a negative, not trainable (existing `labels._decision`).
- In-app suggestions come only from a **custom bundle**, only for `train` frames, are never persisted until accepted, and never coexist with a worker: every job-start route releases (`close()`) the page-server adapter before `enqueue_*`, and `suggest()` runs under the same active-job check.
- Snapshot gate = every video frozen **and** ≥1 trainable reviewed frame in `train` and ≥1 in `validation` (project-wide); the page shows per-video and project-wide counts and says the gate is project-wide.
- Blocks of different roles keep a ≥10 s unassigned gap (spec §8 guard); the data page states it.
- Actions: per-project catalog in project SQLite (`action_classes`, 1–16, same token rules as `CatalogClass.name`), seed `digging, swing_loaded, dumping, swing_empty, idle` (titles from `ui/copy.action_seed_titles`, passed into `actions/` which never imports `ui`); one state per instant (no overlap). Intervals `[start_s, end_s)` in source seconds; role inherited from the role block containing them; an interval not fully inside one block is **rejected**. Every seeded action must occur at least once in train and once in validation before training is meaningful; the UI shows coverage.
- **Multi-run action training**: `enqueue_action_train(run_ids: list[str])` over custom runs with the same `model_id` and `analysis_fps`; features are built per run and concatenated with a `run_start` flag so deltas and windows never cross runs; `action_bundle.json.source_run_ids`. Inference runs on a single run. Walkthrough: run A ≥120 s inside train, run B ≥60 s inside validation.
- **Test block never read by action training**: `build_features(keep_ranges=<frozen train/validation blocks>)` drops test-block rows before any feature exists and marks the first row of each kept range `run_start=True`; kept ranges are logged to the job run dir; `build_windows` refuses `role="test"` unconditionally (no override flag — evaluation compares per-frame predictions with intervals and never builds windows); the standardizer is fitted on train windows only. Evaluation refuses `no_frames_for_split` when an inference overlaps no labelled frame of the requested split.
- `action_models.source_run_ids` is a canonical JSON array column; catalog rename/delete refusal reads `class_order` from each bundle's `action_bundle.json`.
- Model: tiny PyTorch 1D-conv over a ~3 s window at the run's analysis fps, CPU. Macro-F1 over classes with validation support > 0 (`classes_scored` stored), `null` per-class metrics otherwise, `selection_unstable` when validation windows < 300. New job kinds `action_train` and `action_infer` through `jobs.py` (`_ensure_kinds` migration). Output `actions.parquet` + `action_intervals.csv`; low confidence or short runs → `unknown`.
- Final test evaluation (`--split test --final`) requires an existing validation evaluation with identical `action_model_id`, `min_confidence`, `min_duration_s`; test-block labelled intervals are hidden in the UI until `evaluation_test.json` exists.
- Spec §20 says "no temporal network at the start". The user explicitly asked for a trained action classifier on human-labelled intervals (spec §20.2 calls that a later, separate stage). Subtask 20 records this in `docs/DECISIONS.md`; rule-based v0.2 event candidates stay out of scope.

## Phases

| Phase | Subtasks | Outcome |
|---|---|---|
| A. Green baseline | 01 | pytest/ruff green |
| B. Frames + in-app labeling | 02, 03a, 03b, 04a, 04b, 05, 06, 07, 08 | Real video sampled; boxes labelled in the browser; snapshot opens on honest counts |
| C. Detection training | 09, 10 | In-app snapshot → smoke train → bundle on Analysis; two custom analysis runs have parquet |
| D. Actions | 11, 12, 13, 14, 15a, 15b, 16, 17, 18a, 18b | Action catalog, interval labels, features, dataset, train job, inference, evaluation, results UI |
| E. UI glue | 19, 20 | Five-step stepper across pages, ru/en copy, pilot walkthrough, decisions |
| F. Verification | 21 | Integration + leakage suite |

## Subtasks

| # | File | Goal | Est. |
|---|---|---|---|
| 01 | `subtask-01-green-baseline.md` | Fix 3 `test_ui_video_page.py` assertions vs the intended `<details>` project-create panel | 30 min |
| 02 | `subtask-02-real-video-sampling.md` | Freeze (with ≥10 s guard gaps) + sample the real pilot video once; `SampleRow.source_pts/time_base`; CFR-only PTS tick pass (`vfr_sampling_unsupported`) + seek-decode in `_write_chosen` if slow | 2 h |
| 03a | `subtask-03a-label-store-writer.md` | `datasets/label_store.py` upsert writer (incoming JPEGs only, atomic JSON pair, per-dir lock); `_publish` delegation; `frozen_fields_for` | 1.5–2 h |
| 03b | `subtask-03b-label-store-save-frame.md` | `sample_rows_with_role`, `save_frame_label`, supervision/snapshot tests | 1–1.5 h |
| 04a | `subtask-04a-page-shell-and-links.md` | `web/shell.py` extraction, `/label` placeholder route, Data step-3 button + thumbnail links | 1 h |
| 04b | `subtask-04b-label-page-service.md` | `ui/label_page.py` view-model, JSON routes, `label.html` | 1.5–2 h |
| 05 | `subtask-05-box-labeler-canvas.md` | `static/box_labeler.js`: draw/move/resize/delete boxes, attributes, save; `tests/unit/test_web_static.py` | 2 h |
| 06 | `subtask-06-labeler-navigation-progress.md` | Class keys 1..9, prev/next, review/exclude (confirmation on zero boxes incl. `r`), no shortcuts in inputs, per-split progress | 1–1.5 h |
| 07 | `subtask-07-label-counts-snapshot-gate.md` | Per-video + project-wide counts; snapshot gate on ≥1 reviewed train + validation; copy | 1 h |
| 08 | `subtask-08-train-frame-suggestions.md` | Bundle suggestions for train frames only; adapter released by every job-start route | 1.5 h |
| 09 | `subtask-09-training-from-inapp-snapshot.md` | Snapshot → Training page → fake/smoke trainer → bundle listed on Analysis; integration test | 1.5 h |
| 10 | `subtask-10-custom-analysis-run.md` | Runs A (train, ≥120 s) and B (validation, ≥60 s) on the pilot; `storage.list_analysis_runs`, `run_tables` | 1–1.5 h |
| 11 | `subtask-11-action-catalog-labels-store.md` | `actions/catalog.py`, `actions/labels.py`, SQLite tables + migration, CLI; `_PACKAGE_DIRS` += `actions`, `ui` | 2 h |
| 12 | `subtask-12-action-labeling-ui.md` | Page «Действия» steps 1–2: catalog editor + interval timeline, coverage line, unfrozen notice | 2 h |
| 13 | `subtask-13-action-features.md` | `actions/features.py`: per-frame per-class features with masks, `dt_s`, `run_start`, `concat_runs` | 1.5–2 h |
| 14 | `subtask-14-action-windows-dataset.md` | `actions/dataset.py`: windows, centre-label rule, same-block and same-run containment, test refused, hash | 1.5 h |
| 15a | `subtask-15a-action-model-trainer.md` | `ActionNet`, train-only `Standardizer`, trainer with macro-F1 over scored classes | 1.5 h |
| 15b | `subtask-15b-action-train-job.md` | Multi-run `check_ready`, `keep_ranges` slice + `kept_time_ranges.json`, `_ensure_kinds`, bundle, CLI (optional 15b-pre split for the job-kind plumbing if it slips) | 2 h |
| 16 | `subtask-16-action-inference.md` | `actions/infer.py` on one run, job kind `action_infer`, `actions.parquet`, `action_intervals.csv`, CLI | 1.5 h |
| 17 | `subtask-17-action-evaluation.md` | Validation metrics; `--split test --final` gated on matching validation thresholds | 1 h |
| 18a | `subtask-18a-actions-train-infer-controls.md` | Step 3: multi-run selection, verbatim readiness, `GET /actions/job`, infer controls | 1.5–2 h |
| 18b | `subtask-18b-actions-results-evaluation.md` | Step 4: timeline + table, train/validation-only comparison strip, evaluation, final test | 1.5–2 h |
| 19 | `subtask-19-workflow-stepper-copy.md` | `ui/workflow.py` five-stage stepper in `base.html`, disabled-state audit, ru/en parity | 1.5 h |
| 20 | `subtask-20-pilot-walkthrough-decisions.md` | Browser walkthrough 1→5 on the pilot (runs A+B, full action coverage), `docs/DECISIONS.md` | 1.5–2 h |
| 21 | `subtask-21-acceptance-suite.md` | Three-run (A train, B validation, C test) actions integration test incl. `no_frames_for_split` on B-final, leakage tests (kept ranges, `run_start` per kept range, spy on `build_windows`, standardizer), final pytest/ruff | 1.5–2 h |

## Dependencies

```
01 ─┐
02 ─┼─► 03a ─► 03b ─► 04a ─► 04b ─► 05 ─► 06 ─► 07 ─► 09 ─► 08
    │                                                      │
    │                                                      └─► 10 ─┐
    │                                                              │
    └──► 11 ─► 12 (needs 10 run ranges)          13 (needs 10) ─► 14 ─► 15a ─► 15b ─► 16 ─► 17
                                                                                   │          │
                                                                                   └─► 18a ───┴─► 18b ─► 19 ─► 20 ─► 21
```

- 03a is the hinge of phase B: 03b–08 and 09 read/write only through it.
- 08 needs a bundle to exercise; it follows 09.
- 11 needs blocks on the pilot video (02); 12 labels inside the time ranges of runs A/B (10).
- 13 validates against the two real runs (10); unit tests use synthetic parquet and do not wait.
- 15b depends on 11 (labels), 13, 14, 15a and carries the `jobs.py` kind migration used by 16.
- 18a needs 15b and 16; 18b needs 18a and 17. 19 after 18b so the stepper reflects real states.

## Execution order

01 → 02 → 03a → 03b → 04a → 04b → 05 → 06 → 07 → 09 → 08 → 10 → 11 → 12 → 13 → 14 → 15a → 15b → 16 → 17 → 18a → 18b → 19 → 20 → 21

## Verification (every subtask)

```bash
uv run pytest tests -q --ignore=tests/manual
uv run ruff check src tests
```

- UI subtasks (02, 04a, 04b, 05–08, 12, 18a, 18b, 19, 20): `uv run excavation ui`, project `mvp`, exercise the changed screen in the browser (both `ru` and `en` for new copy).
- Training: `--profile smoke` is a wiring check only; no automatic `pilot-small`; a real smoke run on MPS/CPU in subtask 20 is optional and is recorded as "wiring", not quality.
- Compute-core isolation: `tests/unit/test_core_imports.py` covers `actions` and `ui` from subtask 11 on.
- No invented numbers: missing measurements render «не рассчитано» / `not calculated`.
- Leakage: validation/test frames never receive suggestions; action training slices test-block frames out before windowing and logs kept ranges; windows cross neither block nor run boundaries; standardizer is train-only.

## Scope risks

1. **Sampler cost on the real video** (60 fps, 1271 s, synchronous POST). Subtask 02 measures; primary fix is a PTS-only demux pass plus seek-decode per selected tick; fallback is a `sample` job kind (~1 h).
2. **`Detection.class_name: DetectionClassName`** (schemas.py) still binds analysis to the four pilot names. Features read parquet strings, so actions are unaffected for the `excavation` catalog; a custom catalog would break analysis before actions. Out of scope here; flagged in DECISIONS (subtask 10/20).
3. **Canvas labeler UX** is the largest unknown (05–06). Boxes + attributes only; fit-to-width, no zoom.
4. **Action data volume**: two short runs on one video yield few windows; `selection_unstable` and `classes_scored` are shown; metrics are a wiring/sanity signal, not a quality claim.
5. **Job-kind CHECK migration** in `lab.sqlite` (`_ensure_kinds`) must keep existing rows.
6. **Analysis wall time** for runs A+B at 5 fps with Nano is measured in subtask 10 (expected minutes, MPS faster than CPU); it is the slowest manual step of the walkthrough.
