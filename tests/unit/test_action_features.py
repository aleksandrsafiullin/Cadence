"""Action features from synthetic analysis parquet. No video and no detector."""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from excavation_lab.actions import features
from excavation_lab.actions.features import (
    FeatureReport,
    FeatureSpec,
    build_features,
    concat_runs,
)
from excavation_lab.errors import LabError
from excavation_lab.tracking.analyze import RunTables
from excavation_lab.tracking.export_tables import (
    DETECTIONS_COLUMNS,
    DETECTIONS_SCHEMA,
    FRAME_STATUS_COLUMNS,
    FRAME_STATUS_SCHEMA,
    TRACK_STATE_COLUMNS,
    TRACK_STATE_SCHEMA,
)

_SUFFIXES = (
    "present",
    "cx",
    "cy",
    "w",
    "h",
    "score",
    "dcx",
    "dcy",
    "dw",
    "dh",
    "mask",
)


def test_export_columns_are_the_feature_inputs() -> None:
    assert set(features.FRAME_FIELDS) <= set(FRAME_STATUS_COLUMNS)
    assert set(features.DETECTION_FIELDS) <= set(DETECTIONS_COLUMNS)


def test_columns_are_generated_from_class_order() -> None:
    spec = FeatureSpec(class_names=("bucket", "truck_bed"), analysis_fps=5)
    assert spec.relation is True
    assert spec.schema_version == features.FEATURE_SCHEMA_VERSION
    per_class = tuple(f"bucket_{suffix}" for suffix in _SUFFIXES) + tuple(
        f"truck_bed_{suffix}" for suffix in _SUFFIXES
    )
    assert spec.columns == per_class + (
        "bucket_in_bed",
        "bucket_bed_dist",
        "relation_mask",
        "dt_s",
        "frame_mask",
    )
    again = FeatureSpec.model_validate(spec.model_dump(mode="json"))
    assert again == spec


def test_catalog_without_truck_bed_has_no_relation_columns() -> None:
    spec = FeatureSpec(class_names=("excavator_body", "bucket"), analysis_fps=5)
    assert spec.relation is False
    assert "bucket_in_bed" not in spec.columns
    assert "relation_mask" not in spec.columns
    with pytest.raises(ValidationError):
        FeatureSpec(
            class_names=("bucket",),
            columns=("bucket_present",),
            relation=False,
            analysis_fps=5,
        )


def test_absent_class_is_nan_with_mask_zero(tmp_path: Path) -> None:
    spec = _spec("bucket", "excavator_body")
    frame = _build(
        tmp_path,
        spec,
        [_status(0, 0.0)],
        [_detection(0, 0.0, "excavator_body", 0.8, 10, 10, 30, 20)],
    )
    row = frame.iloc[0]
    assert tuple(frame.columns) == features.KEY_COLUMNS + spec.columns
    assert row["frame_mask"] == 1
    assert row["excavator_body_present"] == 1
    assert row["excavator_body_cx"] == pytest.approx(0.2)
    assert row["bucket_present"] == 0
    assert row["bucket_mask"] == 0
    assert math.isnan(row["bucket_cx"])
    assert math.isnan(row["bucket_cy"])
    assert math.isnan(row["bucket_score"])
    assert row["bucket_cx"] != 0


def test_non_analyzed_and_decode_error_zero_the_frame_mask(tmp_path: Path) -> None:
    spec = _spec("bucket")
    frame = _build(
        tmp_path,
        spec,
        [
            _status(0, 0.0, analyzed=False),
            _status(1, 0.2, decode_status="corrupt"),
        ],
        [
            _detection(0, 0.0, "bucket", 0.9, 0, 0, 20, 10),
            _detection(1, 0.2, "bucket", 0.9, 0, 0, 20, 10),
        ],
    )
    assert list(frame["frame_mask"]) == [0, 0]
    assert list(frame["bucket_mask"]) == [0, 0]
    assert list(frame["bucket_present"]) == [0, 0]
    assert math.isnan(frame.loc[0, "bucket_cx"])
    assert math.isnan(frame.loc[1, "bucket_cx"])


def test_deltas_skip_gaps_and_dt_reports_the_real_gap(tmp_path: Path) -> None:
    spec = _spec("bucket")
    frame = _build(
        tmp_path,
        spec,
        [
            _status(0, 0.0),
            _status(1, 0.2),
            _status(2, 0.6, analyzed=False),
            _status(3, 1.0),
        ],
        [
            _box(0, 0.0, 0, 20),
            _box(1, 0.2, 30, 50),
            _box(2, 0.6, 80, 100),
            _box(3, 1.0, 40, 60),
        ],
    )
    assert frame.loc[1, "dt_s"] == pytest.approx(0.2)
    assert frame.loc[1, "bucket_dcx"] == pytest.approx(0.3)
    assert frame.loc[1, "bucket_mask"] == 1
    assert frame.loc[2, "frame_mask"] == 0
    assert math.isnan(frame.loc[2, "dt_s"])
    assert frame.loc[3, "dt_s"] == pytest.approx(0.8)
    assert frame.loc[3, "bucket_dcx"] == pytest.approx(0.1)
    assert frame.loc[3, "bucket_mask"] == 1
    assert bool(frame.loc[0, "run_start"]) is True
    assert math.isnan(frame.loc[0, "dt_s"])
    assert math.isnan(frame.loc[0, "bucket_dcx"])
    assert frame.loc[0, "bucket_mask"] == 0
    assert frame.loc[0, "bucket_present"] == 1


def test_keep_ranges_drop_before_deltas(tmp_path: Path) -> None:
    spec = _spec("bucket")
    times = (0.0, 9.0, 10.0, 15.0, 19.0, 20.0, 29.0, 30.0, 35.0)
    boxes = {
        0.0: (0, 20),
        9.0: (30, 50),
        19.0: (70, 90),
        20.0: (15, 35),
        29.0: (40, 60),
    }
    frame = _build(
        tmp_path,
        spec,
        [_status(index, time_s) for index, time_s in enumerate(times)],
        [_box(index, time_s, *boxes[time_s]) for index, time_s in enumerate(times) if time_s in boxes],
        keep_ranges=((0.0, 10.0), (20.0, 30.0)),
    )
    assert set(frame["source_time_s"]) == {0.0, 9.0, 20.0, 29.0}
    assert list(frame.loc[frame["run_start"], "source_time_s"]) == [0.0, 20.0]
    start = frame.loc[frame["source_time_s"] == 20.0].iloc[0]
    assert bool(start["run_start"]) is True
    assert math.isnan(start["bucket_dcx"])
    assert math.isnan(start["dt_s"])
    assert start["bucket_mask"] == 0
    assert start["bucket_present"] == 1
    assert start["bucket_cx"] == pytest.approx(0.25)
    followed = frame.loc[frame["source_time_s"] == 29.0].iloc[0]
    assert followed["bucket_dcx"] == pytest.approx(0.25)
    assert followed["dt_s"] == pytest.approx(9.0)
    earlier = frame.loc[frame["source_time_s"] == 9.0].iloc[0]
    assert earlier["bucket_dcx"] == pytest.approx(0.3)
    assert bool(earlier["run_start"]) is False


def test_concat_runs_keeps_a_boundary_between_runs(tmp_path: Path) -> None:
    spec = _spec("bucket")
    first = _build(
        tmp_path / "a",
        spec,
        [_status(0, 0.0), _status(1, 1.0)],
        [_box(0, 0.0, 0, 20), _box(1, 1.0, 30, 50)],
        run_id="run-a",
        run_index=0,
    )
    second = _build(
        tmp_path / "b",
        spec,
        [_status(0, 2.0), _status(1, 3.0)],
        [_box(0, 2.0, 30, 50), _box(1, 3.0, 60, 80)],
        run_id="run-b",
        run_index=1,
    )
    stacked, used = concat_runs([first, second], [spec, spec])
    assert used == spec
    boundary = stacked.loc[stacked["run_id"] == "run-b"].iloc[0]
    assert bool(boundary["run_start"]) is True
    assert math.isnan(boundary["bucket_dcx"])
    assert math.isnan(boundary["dt_s"])
    assert boundary["bucket_mask"] == 0
    assert boundary["bucket_present"] == 1
    assert int(stacked["run_start"].sum()) == 2
    assert stacked.loc[stacked["run_id"] == "run-a"].iloc[1]["bucket_dcx"] == pytest.approx(0.3)


def test_concat_merges_ignored_counts(tmp_path: Path) -> None:
    spec = _spec("bucket")
    first = _build(
        tmp_path / "a",
        spec,
        [_status(0, 0.0)],
        [
            _box(0, 0.0, 0, 20),
            _detection(0, 0.0, "cone", 0.5, 0, 0, 10, 10),
        ],
        run_id="run-a",
    )
    second = _build(
        tmp_path / "b",
        spec,
        [_status(0, 1.0)],
        [
            _box(0, 1.0, 0, 20),
            _detection(0, 1.0, "cone", 0.4, 0, 0, 10, 10),
            _detection(0, 1.0, "person", 0.6, 10, 0, 20, 10),
            _detection(0, 1.0, "person", 0.3, 20, 0, 30, 10),
        ],
        run_id="run-b",
        run_index=1,
    )
    stacked, used = concat_runs([first, second], [spec, spec])
    assert used == spec
    report = stacked.attrs["feature_report"]
    assert isinstance(report, FeatureReport)
    assert report.ignored_counts == {"cone": 2, "person": 2}
    assert report.ignored_detections == 4
    assert report.ignored_class_names == ("cone", "person")


def test_concat_refuses_mismatched_specs(tmp_path: Path) -> None:
    spec = _spec("bucket")
    frame = _build(tmp_path, spec, [_status(0, 0.0)], [_box(0, 0.0, 0, 20)])
    other = FeatureSpec(class_names=("excavator_body",), analysis_fps=5)
    slower = FeatureSpec(class_names=spec.class_names, analysis_fps=4)
    with pytest.raises(LabError) as mismatched:
        concat_runs([frame, frame], [spec, other])
    assert mismatched.value.code == "feature_spec_mismatch"
    with pytest.raises(LabError) as fps:
        concat_runs([frame, frame], [spec, slower])
    assert fps.value.code == "feature_spec_mismatch"


def test_concat_forces_the_run_boundary(tmp_path: Path) -> None:
    spec = _spec("bucket")
    frame = _build(
        tmp_path,
        spec,
        [_status(0, 0.0), _status(1, 0.2)],
        [_box(0, 0.0, 0, 20), _box(1, 0.2, 30, 50)],
    )
    forged = frame.copy()
    forged.loc[0, "run_start"] = False
    forged.loc[0, "bucket_dcx"] = 0.5
    forged.loc[0, "bucket_mask"] = 1
    forged.loc[0, "dt_s"] = 0.2
    stacked, _spec_out = concat_runs([forged, forged], [spec, spec])
    for index in (0, len(forged)):
        assert bool(stacked.loc[index, "run_start"]) is True
        assert math.isnan(stacked.loc[index, "bucket_dcx"])
        assert math.isnan(stacked.loc[index, "dt_s"])
        assert stacked.loc[index, "bucket_mask"] == 0


def test_predicted_boxes_are_not_observations(tmp_path: Path) -> None:
    spec = _spec("bucket")
    frame = _build(
        tmp_path,
        spec,
        [_status(0, 0.0, width=100, height=100)],
        [
            _detection(0, 0.0, "bucket", 0.99, 0, 0, 100, 100, observation_state="predicted"),
            _detection(0, 0.0, "bucket", 0.4, 0, 0, 20, 10, observation_state="weak"),
        ],
        tracks=[_track(0.0, x2=100.0, y2=100.0)],
    )
    assert frame.loc[0, "bucket_cx"] == pytest.approx(0.1)
    assert frame.loc[0, "bucket_score"] == pytest.approx(0.4)
    assert frame.loc[0, "bucket_present"] == 1


def test_unknown_class_is_counted_not_raised(tmp_path: Path) -> None:
    spec = _spec("bucket")
    frame = _build(
        tmp_path,
        spec,
        [_status(0, 0.0)],
        [
            _box(0, 0.0, 0, 20),
            _detection(0, 0.0, "cone", 0.7, 0, 0, 10, 10),
            _detection(0, 0.0, "person", 0.7, 0, 0, 10, 10),
        ],
    )
    report = frame.attrs["feature_report"]
    assert report.ignored_detections == 2
    assert report.ignored_class_names == ("cone", "person")
    assert "cone_present" not in frame.columns


def test_bucket_inside_bed_containment_is_one(tmp_path: Path) -> None:
    spec = FeatureSpec(
        class_names=("excavator_body", "bucket", "haul_truck", "truck_bed"),
        analysis_fps=5,
    )
    frame = _build(
        tmp_path,
        spec,
        [_status(0, 0.0, width=100, height=100)],
        [
            _detection(0, 0.0, "bucket", 0.9, 20, 20, 40, 40),
            _detection(0, 0.0, "truck_bed", 0.8, 10, 10, 90, 90),
        ],
    )
    row = frame.iloc[0]
    assert row["bucket_in_bed"] == pytest.approx(1.0)
    assert row["bucket_bed_dist"] == pytest.approx(0.25)
    assert row["relation_mask"] == 1
    assert tuple(frame.columns) == features.KEY_COLUMNS + spec.columns


def test_relation_mask_is_zero_when_either_box_is_absent(tmp_path: Path) -> None:
    spec = FeatureSpec(class_names=("bucket", "truck_bed"), analysis_fps=5)
    frame = _build(
        tmp_path,
        spec,
        [_status(0, 0.0, width=100, height=100)],
        [_detection(0, 0.0, "bucket", 0.9, 20, 20, 40, 40)],
    )
    assert math.isnan(frame.loc[0, "bucket_in_bed"])
    assert math.isnan(frame.loc[0, "bucket_bed_dist"])
    assert frame.loc[0, "relation_mask"] == 0


def test_spec_round_trip_keeps_nan(tmp_path: Path) -> None:
    spec = _spec("bucket", "truck_bed")
    frame = _build(
        tmp_path / "run",
        spec,
        [_status(0, 0.0, width=100, height=100)],
        [_detection(0, 0.0, "bucket", 0.9, 20, 20, 40, 40)],
    )
    directory = tmp_path / "features"
    features.write_features(frame, spec, directory)
    loaded, loaded_spec = features.read_features(directory)
    assert loaded_spec == spec
    assert tuple(loaded.columns) == tuple(frame.columns)
    assert math.isnan(loaded.loc[0, "bucket_dcx"])
    assert math.isnan(loaded.loc[0, "truck_bed_cx"])
    assert loaded.loc[0, "bucket_present"] == 1
    assert math.isnan(loaded.loc[0, "bucket_in_bed"])
    tampered = loaded.rename(columns={"dt_s": "delta_t"})
    pq.write_table(
        pa.Table.from_pandas(tampered, preserve_index=False),
        directory / features.FEATURES_NAME,
    )
    with pytest.raises(LabError) as caught:
        features.read_features(directory)
    assert caught.value.code == "feature_spec_mismatch"


def _spec(*names: str) -> FeatureSpec:
    return FeatureSpec(class_names=names, analysis_fps=5)


def _build(
    directory: Path,
    spec: FeatureSpec,
    statuses: list[dict[str, object]],
    detections: list[dict[str, object]],
    *,
    tracks: list[dict[str, object]] | None = None,
    run_id: str = "run-a",
    run_index: int = 0,
    keep_ranges: tuple[tuple[float, float], ...] | None = None,
):
    directory.mkdir(parents=True, exist_ok=True)
    _write(directory / "frame_status.parquet", statuses, FRAME_STATUS_SCHEMA)
    _write(directory / "detections.parquet", detections, DETECTIONS_SCHEMA)
    _write(directory / "track_states.parquet", tracks or [], TRACK_STATE_SCHEMA)
    tables = RunTables(
        frame_status=directory / "frame_status.parquet",
        detections=directory / "detections.parquet",
        track_states=directory / "track_states.parquet",
    )
    return build_features(
        tables,
        run_id=run_id,
        run_index=run_index,
        keep_ranges=keep_ranges,
        spec=spec,
    )


def _write(path: Path, rows: list[dict[str, object]], schema: pa.Schema) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _status(
    index: int,
    time_s: float,
    *,
    analyzed: bool = True,
    decode_status: str = "ok",
    width: int = 100,
    height: int = 100,
) -> dict[str, object]:
    row: dict[str, object] = {name: None for name in FRAME_STATUS_COLUMNS}
    row.update(
        {
            "run_id": "run-a",
            "video_id": "video",
            "source_frame_index": index,
            "source_time_s": time_s,
            "decode_status": decode_status,
            "analyzed": analyzed,
            "model_version": "model",
            "num_detections": 0,
            "diagnostic_flags": [],
            "source_pts": index,
            "time_base": "1/30",
            "analysis_index": index,
            "width": width,
            "height": height,
        }
    )
    return row


def _detection(
    index: int,
    time_s: float,
    class_name: str,
    score: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    observation_state: str = "observed",
    width: int = 100,
    height: int = 100,
) -> dict[str, object]:
    row: dict[str, object] = {name: None for name in DETECTIONS_COLUMNS}
    row.update(
        {
            "run_id": "run-a",
            "observation_id": f"obs-{index}-{class_name}-{score}",
            "video_id": "video",
            "source_frame_index": index,
            "source_time_s": time_s,
            "class_id": 0,
            "class_name": class_name,
            "score": score,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "frame_width": width,
            "frame_height": height,
            "observation_state": observation_state,
            "model_version": "model",
            "source_pts": index,
            "time_base": "1/30",
            "analysis_index": index,
            "width": width,
            "height": height,
        }
    )
    return row


def _box(index: int, time_s: float, x1: float, x2: float) -> dict[str, object]:
    return _detection(index, time_s, "bucket", 0.8, x1, 0, x2, 10)


def _track(time_s: float, *, x2: float, y2: float) -> dict[str, object]:
    row: dict[str, object] = {name: None for name in TRACK_STATE_COLUMNS}
    row.update(
        {
            "run_id": "run-a",
            "track_id": "track-1",
            "class_name": "bucket",
            "source_time_s": time_s,
            "state": "predicted",
            "is_observed": False,
            "x1_nullable": 0.0,
            "y1_nullable": 0.0,
            "x2_nullable": x2,
            "y2_nullable": y2,
            "identity_uncertain": False,
        }
    )
    return row
