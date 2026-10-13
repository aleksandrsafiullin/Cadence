# Subtask 7: Locale tests

## Goal

Lock the locale file, the English screen, and the rule that default `ru` still satisfies the existing Russian UI tests. An English page fails when a button or caption is still Russian. Do not delete honest-empty assertions.

## Context

Spec §19 item 16: a UI without a dataset or checkpoint does not invent metrics. Spec §21: missing measurements are not 0. The Russian tests listed in the master plan already encode that. They must pass with no `locale.txt` and with no `lab_locale` in session.

`tests/unit/test_theme_persistence.py` is the template for the file tests: `tmp_path` workspace, symlink escape, invalid text, read does not create the file. Do not use the repo `workspace/ui/` directory. A test that calls `storage.default_workspace_root()` and then writes `locale.txt` is a failure of this subtask.

## Acceptance Criteria

- [ ] New tests live in `tests/unit/test_ui_locale.py`. They use `tmp_path` (or another directory that is not the repo `workspace/`).
- [ ] Missing file and empty session resolve to `ru` and do not create `ui/locale.txt`.
- [ ] File `en` on a fresh session resolves to `en` and sets `lab_locale`.
- [ ] Files `stopping`, empty, `fr`, and `EN` resolve to `ru`, do not raise, and do not store the invalid token.
- [ ] A symlink from `<workspace>/ui` to a directory outside the workspace does not create `locale.txt` outside, and the switch surfaces the write-failed catalog sentence.
- [ ] The switch `on_change` writes exactly `en` (plus newline) under the temp workspace. A read-only resolve does not write.
- [ ] An AppTest with locale `en` shows the six English titles (`Video`, `Data and labeling`, `Training`, `Analysis`, `Quality`, `Events`), the phrase `not calculated`, and the English events-unavailable sentence. Events still has twelve metrics, no buttons, and no chart. No metric value is `0`. Quality still shows `not evaluated`.
- [ ] That English render of all six pages fails if any visible label, help, caption, header, button, or empty state contains Cyrillic other than the switch endonym `Русский` and the chart field `класс`. `not evaluated` is Latin and must remain on Quality. Do not whitelist other Russian words. The three service formatters from subtask 4 are not on the initial render; do not click sample, import, or export in this scan.
- [ ] That English test also switches from the default Russian session to `English` and asserts the temp `locale.txt` is `en`.
- [ ] Empty workspace: `project_picker` stores `no-project`. Switching to English does not raise. The visible label becomes the English no-project sentence, and the stored value stays `no-project`. At `ru` the visible label is still `ui_copy.NO_PROJECT`.
- [ ] Analysis with at least two source timestamps: select the later timestamp, switch locale, and `analysis_time` is still that timestamp. The switch must not assign the first time. Display text uses `с` before the switch and `s` after.
- [ ] English catalog values that are translations contain no Cyrillic. Invariant strings match the Russian catalog. `catalog("en").not_calculated == "not calculated"`.
- [ ] `test_ui_locale.py` parses `src/excavation_lab/ui/copy.py` and asserts it does not import Streamlit. `tests/unit/test_no_streamlit_import.py` allows every file under `ui/`, so it does not prove this.
- [ ] `test_ui_locale.py` asserts `src/excavation_lab/models/quality.py` does not import `excavation_lab.ui.locale`. After a temp `locale.txt` of `en`, `models.quality.NOT_CALCULATED == "не рассчитано"`.
- [ ] The existing Russian tests still pass without editing their honest-empty assertions. Allowed edits are `set_value("custom")` from subtask 5 and the empty-project option assertion from subtask 2 (`options` may be `["no-project"]` when the visible label is still `ui_copy.NO_PROJECT`).
- [ ] `uv run pytest tests -q` and `uv run ruff check src tests` pass.

## Implementation Notes

Drive the English page the same way `test_events_unavailable.py` and `test_ui_empty_states.py` use `AppTest`. Set the locale by writing `locale.txt` in the temp workspace **before** `run()`, and also by `segmented_control` key `lab_locale_choice` set to `English`. Do not index `[0]`; that slot is the theme control.

Scan the strings AppTest already collects (`value`, `label`, `help`, `title`). Strip the allowed substrings `Русский` and `класс`, then fail if Cyrillic remains. A page that still shows `Добавить видео` because `render` read `ui_copy.ADD_VIDEO_BUTTON` fails here even if the English catalog has `Add video`.

Assert product title `Excavation Video Lab`, kickers, and `Light` | `Dark` are unchanged on the English screen.

The `quality.py` import check is an AST walk of that file, not a reliance on `test_no_streamlit_import.py`. Importing `excavation_lab.ui.locale` from `models/` would pull Streamlit into the worker. The `NOT_CALCULATED` assertion runs in a process that has seen `locale.txt` = `en` and still reads the Russian token from `models.quality`.

Build the `analysis_time` case with two timestamps on a finished run, or the smallest fixture `test_ui_analysis_state.py` already uses. Select the second time, switch language, and compare the stored float (or the timestamp the page returns), not the formatted label.

Do not add a test that runs training, downloads weights, or opens a network socket. Do not click sample, CVAT export, or COCO import. Their Russian summaries are out of scope and would fail the Cyrillic scan for a different reason.

## Dependencies

Subtask 6.

## Verification

```bash
uv run pytest tests -q
uv run ruff check src tests
```
