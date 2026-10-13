# Result schemas

`schema_version` is **1**. JSON sidecars for a run include `schema_version`. A schema change is a new version. Old files are not silently reinterpreted.

## Units

Boxes are pixels `x1,y1,x2,y2` in the source frame. `x1,y1` is the top-left corner and `x2,y2` is the bottom-right corner.

Times are seconds from the source start (`source_time_s`). They are not `frame_index / fps` unless that value was the source timestamp stored for the frame. `source_pts` and `time_base` keep the source clock next to that second.

An empty measurement is JSON `null` or an empty CSV cell. It is not `0`.

## `frame_status.parquet`

One row per processed source frame, including a frame with no objects. Spec §17.1, plus the §6.2 fields `source_pts`, `time_base`, `analysis_index`, `width`, and `height`. `decode_ms` and `encode_ms` stay null until those stages are timed.

```text
run_id, video_id, view_id, source_frame_index, source_time_s,
decode_status, analyzed, model_version, inference_ms,
num_detections, bucket_observation_state, diagnostic_flags,
source_pts, time_base, analysis_index, width, height,
decode_ms, encode_ms
```

## `detections.parquet`

Detector boxes for that frame only. Spec §17.2, plus the same §6.2 identity fields. A row with `observation_state=predicted` is not written. `score`, `parent_track_id_nullable`, and `association_score_nullable` are null when unknown.

## `track_states.parquet`

Spec §17.3. Predicted and missing steps use `is_observed=false`. A predicted box lives only here. A missing step keeps `x1_nullable`, `y1_nullable`, `x2_nullable`, and `y2_nullable` null. `parent_track_id_nullable` is null when the parent is unknown. Bed `track_id` uses the bed's stable local id, not the detector row index.

```text
run_id, track_id, class_name, source_time_s,
state, last_observed_time_s, is_observed,
x1_nullable, y1_nullable, x2_nullable, y2_nullable,
parent_track_id_nullable, identity_uncertain
```

## CSV

`excavation export --project --run --format csv` writes UTF-8 CSV with a decimal point. `--excel` adds a UTF-8 BOM. Notes that start with `=`, `+`, `-`, or `@` are stored unchanged in SQLite and exported with a leading apostrophe.

## Not written in v0.1

- Spec §17.4 `events.csv` is not written.
- Spec §17.5 `truck_visits.csv` is not written.

`excavation export --format events` returns `unavailable` and does not create either file.
