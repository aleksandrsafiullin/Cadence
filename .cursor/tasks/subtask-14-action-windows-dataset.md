# Subtask 14: Windowed action dataset with block containment (no leakage)

## Goal
`actions/dataset.py` builds training/validation windows from features (13) and interval labels (11) under the frozen role blocks, with the test block never read and no window spanning a block boundary.

## Context
Spec §8 ("Пересекающиеся окна временной модели в будущем также должны оставаться в одном split"), roles of the three parts, §19 item 3–4. TZ §5 help: the training block must contain every action.

## Acceptance Criteria
- [ ] `WindowConfig(window_s: float = 3.0, stride_frames: int = 1, min_label_coverage: float = 1.0)`; `window_frames = max(3, round(window_s * analysis_fps))`, stored in the action bundle later. `min_label_coverage` is the minimum fraction of the window's frames whose time lies inside the labelling interval chosen by the centre rule; a window below it is dropped as `unlabelled` (default 1.0 = the whole window inside one interval).
- [ ] `build_windows(features_df, spec, intervals, frozen_blocks, role, config) -> ActionWindows` with `x: np.ndarray [n, window_frames, n_features]`, `y: np.ndarray [n]` (class index in the action catalog order), `center_time_s`, `run_id`, `support: dict[action_name, int]`, `dropped: dict[reason, int]` (`crosses_block`, `crosses_run`, `outside_role`, `unlabelled`, `frame_gap`).
- [ ] Rules: a window is kept only if **all** its frames lie inside role blocks of `role` and inside the **same** block; a window containing a `run_start=True` row anywhere except its first frame is dropped (`crosses_run`) — windows never span two runs; label = the action interval containing the window centre time (half-open); windows whose centre is unlabelled are dropped (not `unknown`-labelled); windows with a gap in `source_time_s` larger than `2 / analysis_fps` are dropped (`frame_gap`).
- [ ] `role="test"` is **always** refused by `build_windows` (`LabError("test_windows_refused")`); there is no override flag. Evaluation (17) never builds windows from labels — it compares per-frame predictions from `actions.parquet` with intervals.
- [ ] `dataset_hash(features_spec, intervals, blocks_hash, config, catalog)` sha256 of canonical JSON; stored in the bundle for honest comparisons.
- [ ] `class_index(catalog) -> dict[name,int]` in catalog id order; `unknown` is not a trainable class (reserved for inference output).
- [ ] Tests `tests/unit/test_action_dataset.py`: synthetic features over 0–100 s at 5 fps with blocks train [0,60), validation [60,80), test [80,100) and intervals; assert no window has a frame ≥ 60 for train; window centred at 59.9 with frames crossing 60 is dropped with `crosses_block`; two concatenated runs [0,30) and [30,60) → no window contains frames from both (`crosses_run` > 0); `role="test"` raises regardless of arguments; a window with 80 % of frames inside the centre interval is dropped at coverage 1.0 and kept at 0.75; support counts match intervals; stride respected; hash changes when an interval moves.

## Implementation Notes
- Files: `src/excavation_lab/actions/dataset.py`, `tests/unit/test_action_dataset.py`.
- Blocks from the **frozen** manifest only (`datasets/splits.require_frozen`); re-validate every interval with `labels.role_of` and skip intervals whose role differs from `role` (count under `outside_role`).
- numpy only; torch tensors are created in 15.

## Dependencies
11, 13.

## Verification
```bash
uv run pytest tests/unit/test_action_dataset.py -q
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
