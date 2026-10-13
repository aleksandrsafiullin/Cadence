# Subtask 9: Quality page

## Goal

Render «Качество» from real `metrics.json` only. Compare two runs. «Отправить в разметку» stays train-only. The AP50 chart is subtask 10.

## Context

Spec §15 «Качество» and §19 item 16. `src/excavation_lab/ui/quality_page.py` calls `load_quality_report`, `compare`, and `send_to_labeling`. It does not import RF-DETR, construct a detector, or enqueue a job. IDF1 and HOTA stay `not evaluated`. A missing score is not drawn as zero. `send_enabled` is true only for train. `list_metric_runs` returns eval ids that have `runs/<id>/metrics.json` and invents none. `models/quality.py` imports `excavation_lab.ui.copy` and must keep using the Russian catalog. It must not import `locale` or Starlette.

Tests: `tests/unit/test_ui_quality_state.py`.

## Acceptance Criteria

- [ ] No project or no `metrics.json`: catalog `quality_empty`, and every score the page prints is «не рассчитано» (or the English `not calculated`). No bar chart node. No `0` standing in for a missing AP.
- [ ] `GET /quality` lists `list_metric_runs`. The select name is catalog `quality_run`. A stale id falls back to a real run id, matching `test_stale_eval_id_uses_a_real_run`.
- [ ] A real report shows per-class metrics, reviewed-frame and GT counts, visibility strata, and the error list the page already builds. IDF1 and HOTA text stay the unevaluated sentence. Match boxes are not drawn.
- [ ] Compare form posts two run ids to `POST /quality/compare` and calls `compare`. A dataset-hash mismatch shows the mismatch text and does not show a delta (`test_compare_shows_hash_mismatch_without_a_delta`).
- [ ] `POST /quality/label` calls `send_to_labeling` only when `send_enabled(split)` is true. The button «Отправить в разметку» is disabled on a test row. A forced POST for test does not queue that frame. A train row still queues. The button key behavior in `send_button_key` can stay as the form field name.
- [ ] Leave the existing `st.altair_chart(ap50_chart(...))` gate inside `quality_page.py` until subtask 10. The new template does not add a second chart. Subtask 10 removes `st.altair_chart` and retargets `test_pages_call_the_styler_only_behind_the_existing_gates`. This subtask does not remove that call.
- [ ] Rewrite `tests/unit/test_ui_empty_states.py::test_quality_page_without_metrics_json_has_no_zero_chart` and `tests/unit/test_ui_locale.py::test_english_pages_reject_leftover_cyrillic` onto `TestClient` here. They are not left for subtask 11.
- [ ] English locale does not leave Russian metric sentences that the catalog already translates. `models.quality` report files stay Russian even if `locale.txt` is `en` (`test_quality_constant_ignores_an_english_locale_file` stays).
- [ ] Rewrite the `AppTest` functions in `tests/unit/test_ui_quality_state.py` onto `TestClient` in this subtask. The file does not import `AppTest` when this subtask ends. Pure tests (no detector import, missing scores, IDF1, hash mismatch, train-only send) stay. Do not leave those clicks for subtask 11.

## Implementation Notes

`_shown_metric` is the display helper. Use it. Do not format a missing float as `0.00`.

Do not add a train or analyze button on this page.

The labeling error shown to the user is the `LabError` message (`test_labeling_error_is_the_lab_message`), localized through the existing `_localized_error` path.

## Dependencies

Subtask 8.

## Verification

```bash
uv run pytest tests/unit/test_ui_quality_state.py tests/unit/test_ui_locale.py::test_quality_constant_ignores_an_english_locale_file -q
uv run ruff check src tests
```

Manual: `uv run excavation ui`, open `/quality` with no metrics file. Confirm the empty sentence and the absence of a chart. Stop the process.
