# Subtask 06: Labeler keyboard, review states, navigation, progress

## Goal
Make the labeler fast enough for 80–600 frames: number keys pick the class, one key marks reviewed and advances, progress is visible per split, and leaving a dirty frame never loses work or fakes a review.

## Context
Spec §7 review rules (reviewed-empty is a negative; unreviewed is not; excluded is kept out of training), §9 batch sizes. The rule the UI must make obvious: **a frame counts only after «Проверено»**; drawing boxes and leaving does not review it.

## Acceptance Criteria
- [ ] Keyboard: `1`..`9` set the class of the selected box (or the class for the next drawn box); `v` cycles visibility; `t` toggles truncated; `r` = save as `reviewed` and go to next; `x` = save as `excluded` and go to next; `u` = save as `unreviewed` (keep boxes, not reviewed); `←`/`→` prev/next; `?` shows the shortcut list. Shortcuts are listed in the side panel in the current language.
- [ ] Navigation with unsaved changes: auto-save with the frame's **current** status (unreviewed stays unreviewed); if the save fails the navigation is cancelled and the error shown.
- [ ] «Проверено» on a frame with zero boxes asks one confirmation (`label_confirm_empty`: "Кадр без объектов станет отрицательным примером") before saving — both for the button and for the `r` key.
- [ ] Keyboard shortcuts (including arrows and digits) do not fire while an `<input>`, `<select>`, or `<textarea>` has focus; `Esc` blurs the field first.
- [ ] Progress line per split: `reviewed / unreviewed / excluded / total` from `label_page.progress`; a filter `<select>` «все / непроверенные / проверенные / исключённые» plus split filter changes the prev/next sequence (client side over `GET /label/frames`).
- [ ] Frame header shows `source_time_s` (3 decimals), source frame index, split label, and `reviewed_at`/`annotator` when present.
- [ ] «Назад к данным» link returns to `/data?video=<id>#step-label`.
- [ ] Tests: `tests/unit/test_label_page.py` extended — `progress` counts (excluded not counted as reviewed; `total` = manifest rows); POST with `review_status=reviewed` and zero boxes yields `label_state=reviewed_empty`, `is_negative=True`; `unreviewed` with boxes yields `trainable=False`.
- [ ] Browser check on the pilot: label 5 train frames and 3 validation frames end-to-end using only the keyboard after the first box.

## Implementation Notes
- Files: `web/static/box_labeler.js`, `templates/label.html`, `ui/label_page.py`, `ui/copy.py`.
- Class colour and key mapping follow catalog order (id ascending); more than 9 classes → keys 1..9 for the first nine, the rest via the select.
- Debounce nothing; saves are explicit or on navigation.
- Keep the «Проверено» button visibly primary; «Исключить» secondary with the catalog caption of what excluded means.

## Dependencies
05.

## Verification
```bash
uv run pytest tests/unit/test_label_page.py -q
uv run excavation ui   # pilot: keyboard-only pass over 8 frames; counters update; reload shows same counts
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
