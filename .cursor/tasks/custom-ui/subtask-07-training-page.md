# Subtask 7: Training page

## Goal

Render «Обучение» so Start and Cancel call `request_training` and `request_training_cancel`. A second request does not start a second job. The loss curve is subtask 10.

## Context

Spec §15 «Обучение» and §16. `src/excavation_lab/ui/training_page.py` holds `TrainForm`, `form_settings`, `request_training`, `request_training_cancel`, `start_enabled`, `should_draw_chart`, `training_log_view`, `format_eta`, `format_epoch`, `format_checkpoint`, `format_status`. The page must not import RF-DETR or construct a detector. `form_settings` keeps `run_test` off (`test_form_settings_keep_run_test_off`).

Progress today is a Streamlit fragment that reads the job row and `metrics.csv` / `metrics_history.csv`. Replace the fragment with `GET /jobs/{job_id}/progress`.

## Acceptance Criteria

- [ ] `GET /training` with no dataset shows the catalog not-ready state. No loss chart node, no ETA, no invented epoch. `format_checkpoint` of a missing path is «не рассчитано».
- [ ] The form posts dataset id, variant (`nano` / `small`), profile (`smoke` or `pilot-small` only on this form), device, and the advanced fields `form_settings` already reads: resolution, epochs, batch, gradient accumulation, learning rate, encoder learning rate, workers, and seed. Changing the profile fills those advanced fields the way `_apply_profile` does. `run_test` stays false and is not a control. Visible profile labels come from the catalog. Stored values are the English ids.
- [ ] `POST /training/start` calls `request_training`. The button is disabled when `start_enabled` is false. A POST during `queued`, `running`, or `cancel_requested` does not call `enqueue_train` again and shows `JobAlreadyRunningError`. A missing dataset does not enqueue. Variant mismatch does not enqueue.
- [ ] `POST /training/cancel` calls `request_training_cancel` for the active job id only. It does not signal an unrelated PID.
- [ ] `GET /jobs/{job_id}/progress` returns JSON: `job_id`, `status`, `orphaned`, `partial`, `analyzed_end_s` (number only when the job row has a real positive value, otherwise `null`), `error_code`, `line` (the string from `job_status_line` for that row), and `progress` (the parsed `progress.json` object, or `null`). A missing job is 404. The handler reads the file `jobs.PROGRESS_FILENAME` under that job's run directory. It does not write progress and does not start a job.
- [ ] `static/lab.js` polls that URL while the status is `queued`, `running`, or `cancel_requested`, replaces the status text with `line`, and stops on `completed`, `failed`, or `cancelled`. The poll URL is relative. No chart is drawn in this subtask: the JSON field `chart` is `null` even when a log exists. Leave the existing gated `loss_chart` call in `training_page.py` until subtask 10. Do not delete it here. Epoch and ETA text on the full page still come from `format_epoch` and `format_eta`. Two measured epochs do not show an ETA (`_MIN_ETA_EPOCHS` is 3). A header-only log does not count as a curve (`should_draw_chart` is false).
- [ ] Rewrite the `AppTest` functions in `tests/unit/test_ui_training_state.py` onto `TestClient` in this subtask. The file does not import `AppTest` when this subtask ends. Pure tests (`should_draw_chart`, `format_eta`, `form_settings`, `start_enabled`) stay. The progress test asserts `orphaned` is in the payload and that `line` matches `job_status_line`. A missing progress file is `progress: null` and not a zero epoch.

## Implementation Notes

`training_page.py` does not import Starlette. `job_status_line` is the function in `video_page.py`. The poll prints that string. It does not assemble a second status sentence in JavaScript.

The progress route lives in `web/routes.py` and uses `reopen` to prove the job id belongs to the workspace. Do not accept a run directory from the client.

`overfit` is not a new form option. The page already limits `_PROFILES` to smoke and pilot-small.

Subtask 10 moves `loss_chart` onto the progress route. This subtask does not remove that call from `training_page.py`, or `test_pages_call_the_styler_only_behind_the_existing_gates` fails early.

## Dependencies

Subtask 6.

## Verification

```bash
uv run pytest tests/unit/test_ui_training_state.py -q
uv run ruff check src tests
```

Manual: `uv run excavation ui`, open `/training`. Confirm there is no curve and no ETA on an empty workspace. Do not click Start in a way that launches a real train. Stop the process.
