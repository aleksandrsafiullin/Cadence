# Subtask 02: Sampling works on the real pilot video

## Goal
Through «Данные и разметка» on project `mvp` (same real file as `pilot`; `pilot` stays untouched, freeze is one-shot), video `v_b91dd7f95a9d46a5ba8a442bbd6314f0` (1271 s, 59.94 fps): add validation and test blocks, freeze, sample the `diagnostic` plan (60 train + 20 validation), and see JPEGs on the contact sheet in acceptable time.

## Context
Spec §8–§9, TZ §5. Today the real video has four episodes and drafted blocks, all `train`; split not frozen; no samples. `datasets/sampler.py::_collect_ticks` streams **every** frame of each block window through `reader.iterate` (≈76 k decoded frames at 1080p); the POST `/data/sample` is synchronous. This must be measured before the labeler can be used on real frames.

## Acceptance Criteria
- [ ] In the browser: blocks `train` / `validation` / `test` saved on the real video (no overlap, source seconds), with a **≥10 s unassigned gap between blocks of different roles** (spec §8 guard interval); the data page states that rule next to the timeline (`book.block_guard_help`, ru/en) and `block_lines` show the gaps; «Зафиксировать разбиение» succeeds, `split_manifests/<video_id>/split_manifest.json` exists.
- [ ] `SampleRow` gains `source_pts: int | None` and `time_base: str | None` (written by `_encode_frame` from the decoded `ReadFrame`; `None` only for a legacy manifest read back), `SampleResult` schema version bumped; the manifest written by this subtask on the pilot carries them for every row. The pilot sample is generated **once here** and reused by 03b onwards.
- [ ] `diagnostic` plan writes `samples/<video_id>/sample_manifest.json` and `images/<video_id>_<idx>.jpg` for train and validation; shortfall (if any) is reported honestly in the flash notice.
- [ ] The PTS-only tick pass is used only for a constant-frame-rate source (probe `fps_nominal == fps_average` within 1e-3 **and** demuxed packet durations constant within one tick); otherwise it refuses with `vfr_sampling_unsupported` and the sampler falls back to the decode path; a VFR synthetic fixture (irregular PTS written with PyAV) asserts the refusal, the CFR fixture asserts identical tick lists.
- [ ] Contact sheet renders the JPEGs via `/data/frame?video=&name=`.
- [ ] Wall time of one `diagnostic` sampling on this machine is measured and written into the subtask result (and `docs/DECISIONS.md` if > 60 s).
- [ ] If > 60 s, primary optimisation: `_collect_ticks` builds `FrameTick`s from **demuxed packet PTS without decoding** (`container.demux(stream)`, sort by pts inside each window to order B-frames, `source_frame_index` from the same rule `reader._source_frame_index` uses), then `_write_chosen` seeks and decodes only the selected ticks via `reader.iterate(t, t + ε)` and takes the frame whose pts equals the tick; a lazy/corrupt frame with `frame.image is None` is reported as `decode_failed` for that tick (and the shortfall counted), never silently skipped. Spacing, representative strategy, and near-duplicate warnings unchanged; `tests/unit/test_sampler_no_leak.py` still pass; a new test asserts the PTS-only tick list equals the decode-based list on the synthetic fixture.
- [ ] If still > ~90 s after that: stop, document, and keep a `sample` job kind as the fallback follow-up (do not implement in this subtask).
- [ ] Unit tests unchanged or extended; ruff green.

## Implementation Notes
- Files: `src/excavation_lab/datasets/sampler.py` (`_collect_ticks`, `_tick`, `_write_chosen`, `_encode_frame`), `src/excavation_lab/media/reader.py` (add `iterate_pts(start_s, end_s)` yielding `(source_frame_index, source_pts, time_base, source_time_s)` from demux only; `ReadFrame.image` may be `None` for corrupt frames already), `src/excavation_lab/ui/data_page.py::run_sample` and `block_lines` (guard gap text), `ui/copy.py`.
- Keep PTS as truth: ticks are selected by `source_time_s` from PTS, never `frame_index / fps`; `source_pts`/`time_base` in the manifest come from the decoded frame, so later consumers (label store 03b, CVAT manifest) never recompute them.
- `tests/fixtures/synthetic.py` gains `write_synthetic_vfr_mp4(path, ...)` for the VFR refusal test.
- Do not change `select_ticks` semantics; only defer pixel conversion.
- Time it with `time.perf_counter` around `sample_frames` in a throwaway script under `/tmp`, not in the repo.
- Check free disk space before sampling (user rule): `df -h .`.

## Dependencies
01.

## Verification
```bash
uv run excavation ui   # project pilot → Данные → blocks → freeze → sample diagnostic
ls workspace/projects/mvp/samples/v_b91dd7f95a9d46a5ba8a442bbd6314f0/images | wc -l
uv run pytest tests/unit/test_sampler_no_leak.py tests/unit/test_split_freeze.py -q
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
