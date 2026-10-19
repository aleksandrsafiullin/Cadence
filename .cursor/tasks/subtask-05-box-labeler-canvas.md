# Subtask 05: `box_labeler.js` — draw, move, resize, delete boxes and save

## Goal
Vanilla-JS canvas editor on `/label`: the operator draws boxes on the sampled frame, moves/resizes them, sets class and attributes, and saves through `POST /label/frame`.

## Context
Spec §7 (visible-extent boxes, `visibility`, `truncated`, nested boxes allowed), §15 (explain controls, no framework). Follows the style of `web/static/episode_cutter.js` (IIFE, state from a JSON `<script>`, `fetch` + reload or in-place update). No React, no canvas framework.

## Acceptance Criteria
- [ ] `src/excavation_lab/web/static/box_labeler.js` renders the JPEG into `<canvas>` scaled to the container width; internal coordinates are image pixels; CSS scaling never changes stored numbers.
- [ ] Interactions: drag on empty area creates a box (min 2×2 px, clamped to the image); click selects; drag inside moves; 8 handles resize; `Delete`/`Backspace` removes the selected box; `Esc` deselects.
- [ ] Selected box panel: class `<select>` (catalog `title`, hint `box` shown under it), visibility radio `clear`/`partial`/`unlabelable`, `truncated` checkbox, optional `instance_local_id`, optional note. Changing `unlabelable` shows the catalog caption that the frame becomes `excluded` (existing rule in `labels._decision`).
- [ ] Boxes are drawn with class colour (deterministic per catalog id) and label text `title`; selected box highlighted; nested boxes remain selectable (smallest-area hit wins).
- [ ] «Сохранить» posts `{review_status: current, boxes, note}`; success updates the status line with `label_saved`; failure shows the server message in a `role="alert"` paragraph, nothing silently dropped.
- [ ] Works on the pilot frames in the browser (1920×1080 JPEG downscaled to the panel); no console errors.
- [ ] New test file `tests/unit/test_web_static.py`: `/static/box_labeler.js` is served with a JS media type; `label.html` references it; the `#label-state` JSON on a sampled frame contains `width`, `height`, `classes` (with `title`/`box`), `image_url`.

## Implementation Notes
- Keep DOM ids: `box-canvas`, `box-class`, `box-visibility`, `box-truncated`, `box-instance`, `box-note`, `box-delete`, `label-save`, `label-error`, `label-status`.
- Use `pointerdown/move/up` with `setPointerCapture`; compute scale from `canvas.getBoundingClientRect()` each event.
- Redraw on `resize`.
- Do not implement zoom/pan in MVP; mention in the help caption that the frame is shown fit-to-width.
- Accessibility: buttons are `<button>`, class select is a real `<select>`; keyboard shortcuts come in subtask 06.

## Dependencies
04b.

## Verification
```bash
uv run excavation ui   # pilot → /label: draw 3 boxes (excavator_body, bucket, truck_bed), move/resize, delete one, save, reload page → boxes persist
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
