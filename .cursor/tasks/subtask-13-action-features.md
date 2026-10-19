# Subtask 13: Action features from a detector run

## Goal
`actions/features.py` turns one custom analysis run into a per-analysis-frame feature table with explicit masks, plus a `FeatureSpec` that names every column, so training and inference use the same ordering.

## Context
Spec §17.1–§17.3 (frame axis is `frame_status.parquet`; `detections.parquet` is detector output only; `track_states.parquet` holds predicted boxes with `is_observed=false`), §20 (missing observation is unknown, not absence). Run lookup from subtask 10.

## Acceptance Criteria
- [ ] `FeatureSpec(StrictModel)`: `class_names: tuple[str,...]` (detector bundle `class_mapping.json` order), `columns: tuple[str,...]`, `relation: bool`, `analysis_fps: float`, `schema_version`. `columns` is generated, never hand-typed: per class `c`: `{c}_present, {c}_cx, {c}_cy, {c}_w, {c}_h, {c}_score, {c}_dcx, {c}_dcy, {c}_dw, {c}_dh, {c}_mask`; when both `bucket` and `truck_bed` are in `class_names`: `bucket_in_bed, bucket_bed_dist, relation_mask`; always `dt_s` (actual Δt to the previous analysed frame, NaN on the first frame of a run) and `frame_mask`.
- [ ] `build_features(run_tables, *, run_id, run_index=0, keep_ranges: Sequence[tuple[float, float]] | None = None, spec=None) -> (pandas.DataFrame, FeatureSpec)`: rows outside every half-open `keep_ranges` interval are dropped **first** (`None` keeps all); one row per remaining `frame_status` row ordered by `source_time_s`; key columns `run_id, run_index, run_start (bool, True on the first row of the run and on the first row of each kept range), source_frame_index, source_time_s` kept outside the feature columns; per class the top-score detection in that frame (`observation_state` in `observed|weak`), box normalised by `frame_width/height`; deltas against the previous **analyzed** frame of the same class **within the same run and the same kept range** (a `run_start` row has all delta columns NaN with masks 0), `NaN` + `mask=0` when the class is absent in either frame; `frame_mask=0` for `analyzed=false` or decode errors (all class masks 0); relation: containment ratio of the bucket box inside the top truck_bed box and centre distance normalised by the bed diagonal; masks 0 when either is absent.
- [ ] `concat_runs(frames: list[DataFrame], specs: list[FeatureSpec]) -> (DataFrame, FeatureSpec)`: refuses differing specs (`feature_spec_mismatch`) or differing `analysis_fps`; concatenates in the given order; `run_start` stays True at each boundary so deltas (already per-run) and windows (14) never cross runs.
- [ ] Predicted boxes from `track_states.parquet` are **not** used as observations (optionally a `{c}_predicted` flag column; default off).
- [ ] `write_features(df, spec, directory)` → `features.parquet` + `feature_spec.json`; `read_features(directory)` validates the spec matches.
- [ ] Detections whose `class_name` is not in `spec.class_names` are ignored with a count returned in a `FeatureReport` (not an exception).
- [ ] Tests `tests/unit/test_action_features.py` on synthetic parquet written with `pyarrow` (no video, no detector): column order equals spec; absent class → NaN with mask 0; non-analyzed frame → `frame_mask=0`; deltas skip gaps and `dt_s` reports the real gap; two runs concatenated → delta columns on the first row of run 2 are NaN with mask 0 and `run_start` is True there; `keep_ranges=[(0,10),(20,30)]` on a 0–40 s run → no row in [10,20) or ≥30, the rows at 0 and 20 have `run_start=True` and NaN deltas; mismatched specs refused; relation values for a bucket fully inside a bed = containment 1.0; spec round-trip; a catalog without `truck_bed` has no relation columns.
- [ ] Smoke on the two real runs from subtask 10 in a `/tmp` script: no exception, row count equals the sum of `frame_status` rows, exactly two `run_start=True` rows.

## Implementation Notes
- Files: `src/excavation_lab/actions/features.py`, `tests/unit/test_action_features.py`.
- Read parquet with `pyarrow.parquet.read_table(...).to_pandas()`; column names from `tracking/export_tables.py` constants.
- Keep NaN as NaN in parquet; the model layer (15) replaces NaN with 0 **after** standardisation and relies on mask columns.
- No torch import here.

## Dependencies
10 (real run for the smoke; unit tests only need pyarrow).

## Verification
```bash
uv run pytest tests/unit/test_action_features.py -q
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
