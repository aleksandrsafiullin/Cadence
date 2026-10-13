"""Observation states stay off the detection table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from supervision import Detections

from excavation_lab.media.reader import ReadFrame
from excavation_lab.media.timebase import TimeMode, in_half_open
from excavation_lab.models.adapter import (
    CheckpointOrigin,
    FakeDetector,
    FrameDetections,
    ModelMetadata,
    ScriptedDetection,
)
from excavation_lab.schemas import DecodeStatus, ObservationReason, ObservationState
from excavation_lab.tracking import analyze as analyze_mod
from excavation_lab.tracking.analyze import AnalyzePipeline
from excavation_lab.tracking.bed import BedFrame, BedLink, BedTruckAssociator
from excavation_lab.tracking.bucket import (
    BucketAssociator,
    BucketFrame,
    BucketObservation,
)
from excavation_lab.tracking.export_tables import (
    DETECTIONS_COLUMNS,
    FRAME_STATUS_COLUMNS,
)
from excavation_lab.tracking.session import LostBufferVehicle, TrackedRow
from excavation_lab.tracking.states import (
    MAX_PREDICTED_S,
    ObservationLedger,
    TrackStateRow,
    miss_reason,
    state_for_score,
)
from excavation_lab.tracking.vehicles import VehicleTrackers

_RUN = "run-state"
_VIDEO = "clip"
_FLOOR = 0.10
_DISPLAY = 0.35
_BODY = (100.0, 20.0, 220.0, 100.0)
_BUCKET_0 = (0.0, 0.0, 10.0, 10.0)
_BUCKET_1 = (10.0, 0.0, 20.0, 10.0)
_PREDICTED = (20.0, 0.0, 30.0, 10.0)
_TRUCK = (10.0, 30.0, 90.0, 90.0)
_TIE_BED = (40.0, 40.0, 60.0, 60.0)
_TIE_LEFT = (0.0, 0.0, 100.0, 100.0)
_TIE_RIGHT = (10.0, 0.0, 110.0, 100.0)
_BOXES = ("x1", "y1", "x2", "y2")


def test_score_bands_and_ambiguous_override() -> None:
    assert MAX_PREDICTED_S == pytest.approx(0.4)
    assert state_for_score(0.35, detection_floor=_FLOOR, display_threshold=_DISPLAY) is (
        ObservationState.observed
    )
    assert state_for_score(0.20, detection_floor=_FLOOR, display_threshold=_DISPLAY) is (
        ObservationState.weak
    )
    assert state_for_score(0.10, detection_floor=_FLOOR, display_threshold=_DISPLAY) is (
        ObservationState.weak
    )
    assert (
        state_for_score(0.09, detection_floor=_FLOOR, display_threshold=_DISPLAY) is None
    )
    high = state_for_score(
        0.99,
        detection_floor=_FLOOR,
        display_threshold=_DISPLAY,
        ambiguous=True,
    )
    weak = state_for_score(
        0.20,
        detection_floor=_FLOOR,
        display_threshold=_DISPLAY,
        ambiguous=True,
    )
    assert high is ObservationState.ambiguous
    assert weak is ObservationState.ambiguous
    weak_box = (1.0, 2.0, 8.0, 9.0)
    weak_states = _ledger().observe(
        _frame(("haul_truck",), (0.20,), (weak_box,), 0.0),
        tracked_rows=(
            TrackedRow(
                index=0,
                class_name="haul_truck",
                score=0.20,
                xyxy=weak_box,
                source_time_s=0.0,
                local_track_id=3,
                track_id="run:clip:unset:haul_truck:3",
            ),
        ),
        bucket_frame=None,
        bed_frame=None,
        lost_vehicles=(),
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        source_time_s=0.0,
        reset_count=1,
    )
    assert weak_states.detection_states[0] is ObservationState.weak
    assert weak_states.track_states[0].reason is ObservationReason.low_confidence
    assert weak_states.track_states[0].is_observed is True
    assert weak_states.track_states[0].xyxy == pytest.approx(list(weak_box))
    assert {item.value for item in ObservationReason} == {
        "no_detection",
        "low_confidence",
        "occluded_if_known",
        "out_of_frame_if_known",
        "decode_gap",
        "ambiguous_association",
    }
    assert "event_id" not in TrackStateRow.__dataclass_fields__
    assert set(LostBufferVehicle.__dataclass_fields__) == {"track_id", "class_name"}


def test_miss_reason_is_no_detection_unless_flagged() -> None:
    assert miss_reason() is ObservationReason.no_detection
    assert miss_reason(occluded=False, out_of_frame=False) is ObservationReason.no_detection
    assert miss_reason() is not ObservationReason.occluded_if_known
    assert miss_reason(occluded=True) is ObservationReason.occluded_if_known
    assert miss_reason(out_of_frame=True) is ObservationReason.out_of_frame_if_known

    track_id = "run:clip:unset:haul_truck:0"
    lost = (LostBufferVehicle(track_id, "haul_truck"),)
    plain = _ledger().observe(
        _empty(1.0),
        tracked_rows=(),
        bucket_frame=None,
        bed_frame=None,
        lost_vehicles=lost,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        source_time_s=1.0,
        reset_count=1,
    )
    assert plain.track_states[0].state is ObservationState.missing
    assert plain.track_states[0].xyxy is None
    assert plain.track_states[0].is_observed is False
    assert plain.track_states[0].reason is ObservationReason.no_detection
    assert plain.track_states[0].x1 is None

    hidden = _ledger().observe(
        _empty(1.0),
        tracked_rows=(),
        bucket_frame=None,
        bed_frame=None,
        lost_vehicles=lost,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        source_time_s=1.0,
        reset_count=1,
        occluded_track_ids={track_id},
    )
    assert hidden.track_states[0].reason is ObservationReason.occluded_if_known
    assert hidden.track_states[0].xyxy is None
    assert hidden.track_states[0].state is not ObservationState.predicted

    gone = _ledger().observe(
        _empty(1.0),
        tracked_rows=(),
        bucket_frame=None,
        bed_frame=None,
        lost_vehicles=lost,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        source_time_s=1.0,
        reset_count=1,
        out_of_frame_track_ids={track_id},
    )
    assert gone.track_states[0].reason is ObservationReason.out_of_frame_if_known
    assert gone.track_states[0].state is ObservationState.missing


def test_real_detection_keeps_its_box_when_the_buffer_also_lists_the_track() -> None:
    box = (10.0, 20.0, 40.0, 50.0)
    track_id = "run:clip:unset:haul_truck:0"
    detections = _frame(("haul_truck",), (0.95,), (box,), 1.0)
    tracked = TrackedRow(
        index=0,
        class_name="haul_truck",
        score=0.95,
        xyxy=box,
        source_time_s=1.0,
        local_track_id=0,
        track_id=track_id,
    )
    lost = LostBufferVehicle(track_id, "haul_truck")
    ledger = _ledger()
    seen = ledger.observe(
        detections,
        tracked_rows=(tracked,),
        bucket_frame=None,
        bed_frame=None,
        lost_vehicles=(lost,),
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        source_time_s=1.0,
        reset_count=1,
    )
    assert len(seen.track_states) == 1
    row = seen.track_states[0]
    assert row.state is ObservationState.observed
    assert row.is_observed is True
    assert row.xyxy == pytest.approx(list(box))
    assert row.reason is None
    assert seen.detection_states[0] is ObservationState.observed

    missing = ledger.observe(
        _empty(1.2),
        tracked_rows=(),
        bucket_frame=None,
        bed_frame=None,
        lost_vehicles=(lost,),
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        source_time_s=1.2,
        reset_count=1,
    )
    blank = missing.track_states[0]
    assert blank.state is ObservationState.missing
    assert blank.xyxy is None
    assert blank.is_observed is False
    assert blank.reason is ObservationReason.no_detection
    assert blank.last_observed_time_s == pytest.approx(1.0)
    assert blank.state is not ObservationState.predicted


def test_ambiguous_association_marks_identity_uncertain() -> None:
    bucket = (0.0, 0.0, 10.0, 10.0)
    worse = _Body("exc-1", (-33.0, -45.0, 67.0, 55.0))
    better = _Body("exc-7", (-40.0, -45.0, 60.0, 55.0))
    detections = _frame(("bucket",), (0.9,), (bucket,), 1.0)
    associated = BucketAssociator.from_pilot().update(
        detections,
        bodies=(worse, better),
        source_time_s=1.0,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )
    frame = _ledger().observe(
        detections,
        tracked_rows=(),
        bucket_frame=associated,
        bed_frame=None,
        lost_vehicles=(),
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        source_time_s=1.0,
        reset_count=1,
    )
    assert frame.detection_states[0] is ObservationState.ambiguous
    row = frame.track_states[0]
    assert row.state is ObservationState.ambiguous
    assert row.identity_uncertain is True
    assert row.is_observed is True
    assert row.reason is ObservationReason.ambiguous_association
    assert row.xyxy == pytest.approx(list(bucket))
    assert row.parent_track_id is None
    assert row.class_name == "bucket"

    bed_detections = _frame(("truck_bed",), (0.91,), (_TIE_BED,), 0.0)
    left = _Body("truck-left", _TIE_LEFT, class_name="haul_truck")
    right = _Body("truck-right", _TIE_RIGHT, class_name="haul_truck")
    bed_frame = BedTruckAssociator.from_pilot().update(
        bed_detections,
        trucks=(left, right),
        source_time_s=0.0,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )
    bed_states = _ledger().observe(
        bed_detections,
        tracked_rows=(),
        bucket_frame=None,
        bed_frame=bed_frame,
        lost_vehicles=(),
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        source_time_s=0.0,
        reset_count=1,
    )
    assert bed_states.detection_states[0] is ObservationState.ambiguous
    bed = bed_states.track_states[0]
    assert bed.class_name == "truck_bed"
    assert bed.state is ObservationState.ambiguous
    assert bed.identity_uncertain is True
    assert bed.is_observed is True
    assert bed.parent_track_id is None
    assert bed.xyxy == pytest.approx(list(_TIE_BED))
    assert bed.reason is ObservationReason.ambiguous_association


def test_missing_bed_does_not_copy_a_box_or_use_the_detector_index() -> None:
    box = (1.0, 2.0, 11.0, 12.0)
    present = BedLink(
        index=3,
        local_id=4,
        parent_track_id=None,
        ambiguous=False,
        association_state=ObservationState.missing,
        association_score=None,
        xyxy=box,
        reason=ObservationReason.no_detection,
    )
    blank = BedLink(
        index=0,
        local_id=5,
        parent_track_id=None,
        ambiguous=False,
        association_state=ObservationState.missing,
        association_score=None,
        xyxy=None,
        reason=ObservationReason.no_detection,
    )
    frame = _ledger().observe(
        _frame(("truck_bed",), (0.05,), (box,), 1.0),
        tracked_rows=(),
        bucket_frame=None,
        bed_frame=BedFrame(source_time_s=1.0, links=(present, blank)),
        lost_vehicles=(),
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        source_time_s=1.0,
        reset_count=1,
    )
    copied, empty = frame.track_states
    assert copied.state is ObservationState.missing
    assert copied.xyxy is None
    assert copied.is_observed is False
    assert copied.parent_track_id is None
    assert copied.track_id.endswith(":truck_bed:4")
    assert ":truck_bed:3" not in copied.track_id
    assert empty.xyxy is None
    assert empty.track_id.endswith(":truck_bed:5")


def test_bucket_dash_past_0_4s_is_missing_without_a_box() -> None:
    kept = _dash_frame(0.4)
    assert kept.track_states[0].state is ObservationState.predicted
    assert kept.track_states[0].is_observed is False
    assert kept.track_states[0].xyxy == pytest.approx(list(_PREDICTED))
    assert kept.track_states[0].class_name == "bucket"
    dropped = _dash_frame(0.5)
    assert dropped.track_states[0].state is ObservationState.missing
    assert dropped.track_states[0].xyxy is None
    assert dropped.track_states[0].is_observed is False
    assert dropped.track_states[0].reason is ObservationReason.no_detection
    assert dropped.track_states[0].reason is not ObservationReason.occluded_if_known


def test_detection_rows_keep_weak_and_below_floor_boxes(tmp_path: Path) -> None:
    strong = (0.0, 0.0, 10.0, 10.0)
    weak = (20.0, 0.0, 30.0, 10.0)
    low = (40.0, 0.0, 50.0, 10.0)
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    out = tmp_path / "bands"
    pipeline = AnalyzePipeline(
        _ListReader([_read_frame(0, 0.0, image)]),
        FakeDetector(
            class_names=("haul_truck",),
            origin=CheckpointOrigin.custom,
            scripted=(
                ScriptedDetection(strong, 2, 0.50, "haul_truck"),
                ScriptedDetection(weak, 2, 0.20, "haul_truck"),
                ScriptedDetection(low, 2, 0.09, "haul_truck"),
            ),
        ),
        run_id=_RUN,
        output_dir=out,
        analysis_fps=5,
        model_version="custom-test",
    )
    pipeline.run()
    rows = pq.read_table(out / "detections.parquet").to_pylist()
    assert len(rows) == 3
    by_score = {round(float(row["score"]), 2): row for row in rows}
    assert by_score[0.50]["observation_state"] == "observed"
    assert by_score[0.20]["observation_state"] == "weak"
    assert by_score[0.09]["observation_state"] is None
    assert _xyxy(by_score[0.50]) == pytest.approx(strong)
    assert _xyxy(by_score[0.20]) == pytest.approx(weak)
    assert _xyxy(by_score[0.09]) == pytest.approx(low)
    assert all(row["class_name"] == "haul_truck" for row in rows)
    assert all(row["observation_state"] != "predicted" for row in rows)
    assert set(rows[0]) == set(DETECTIONS_COLUMNS)
    tracks = pq.read_table(out / "track_states.parquet").to_pylist()
    assert all(row["state"] != "predicted" for row in tracks)


def test_mutating_predicted_box_leaves_detection_xyxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = {
        0.0: _frame(("excavator_body", "bucket"), (0.95, 0.9), (_BODY, _BUCKET_0), 0.0),
        0.2: _frame(("excavator_body", "bucket"), (0.95, 0.9), (_BODY, _BUCKET_1), 0.2),
        0.4: _frame(("excavator_body",), (0.95,), (_BODY,), 0.4),
        0.8: _frame(("excavator_body",), (0.95,), (_BODY,), 0.8),
    }
    original = {stamp: np.array(frame.xyxy, copy=True) for stamp, frame in frames.items()}
    held: list[dict[str, object]] = []
    real_rows = analyze_mod._detection_rows

    def capture(*args: object, **kwargs: object) -> list[dict[str, object]]:
        rows = real_rows(*args, **kwargs)
        held.extend(rows)
        return rows

    monkeypatch.setattr(analyze_mod, "_detection_rows", capture)
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    out = tmp_path / "dash"
    pipeline = AnalyzePipeline(
        _ListReader([_read_frame(index, stamp, image) for index, stamp in enumerate(frames)]),
        _TimedDetector(frames),
        run_id=_RUN,
        output_dir=out,
        analysis_fps=5,
        model_version="custom-test",
    )
    pipeline.run()

    predicted = next(
        row
        for row in pipeline.states.track_states
        if row.state is ObservationState.predicted
    )
    assert predicted.class_name == "bucket"
    assert predicted.is_observed is False
    assert predicted.xyxy == pytest.approx(list(_PREDICTED))
    assert predicted.reason is ObservationReason.no_detection
    assert all(
        row.class_name == "bucket"
        for row in pipeline.states.track_states
        if row.state is ObservationState.predicted
    )
    missing = next(
        row
        for row in pipeline.states.track_states
        if row.source_time_s == pytest.approx(0.8) and row.class_name == "bucket"
    )
    assert missing.state is ObservationState.missing
    assert missing.xyxy is None
    assert missing.is_observed is False
    assert missing.reason is ObservationReason.no_detection

    snapshot = [_xyxy(row) for row in held]
    assert list(_PREDICTED) not in snapshot
    assert all(isinstance(row["x1"], float) for row in held)
    assert all(row["observation_state"] != "predicted" for row in held)
    predicted.xyxy[0] = -5000.0
    assert [_xyxy(row) for row in held] == snapshot
    assert all(item[0] != -5000.0 for item in snapshot)
    for stamp, frame in frames.items():
        assert np.array_equal(frame.xyxy, original[stamp])
    stored = pq.read_table(out / "detections.parquet").to_pylist()
    assert all(float(row["x1"]) != -5000.0 for row in stored)
    assert all(_xyxy(row) != pytest.approx(_PREDICTED) for row in stored)
    assert [row["class_name"] for row in _at(stored, 0.4)] == ["excavator_body"]
    assert _xyxy(_at(stored, 0.4)[0]) == pytest.approx(_BODY)
    stored_tracks = pq.read_table(out / "track_states.parquet").to_pylist()
    stored_dash = next(row for row in stored_tracks if row["state"] == "predicted")
    assert stored_dash["is_observed"] is False
    assert stored_dash["class_name"] == "bucket"
    stored_box = [
        float(stored_dash[name])
        for name in ("x1_nullable", "y1_nullable", "x2_nullable", "y2_nullable")
    ]
    assert stored_box == pytest.approx(list(_PREDICTED))
    assert stored_dash["x1_nullable"] != -5000.0
    stored_missing = next(
        row
        for row in stored_tracks
        if row["class_name"] == "bucket" and row["source_time_s"] == pytest.approx(0.8)
    )
    assert stored_missing["state"] == "missing"
    assert stored_missing["is_observed"] is False
    assert stored_missing["x1_nullable"] is None


def test_lost_vehicle_inside_the_buffer_is_missing_without_a_box(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _watch_analytics(monkeypatch)
    frames = {
        0.0: _frame(("haul_truck", "excavator_body"), (0.95, 0.95), (_TRUCK, _BODY), 0.0),
        0.2: _frame(("haul_truck", "excavator_body"), (0.95, 0.95), (_TRUCK, _BODY), 0.2),
        0.4: _frame(("haul_truck", "excavator_body"), (0.95, 0.95), (_TRUCK, _BODY), 0.4),
        0.6: _frame((), (), (), 0.6),
    }
    image = np.zeros((140, 260, 3), dtype=np.uint8)
    out = tmp_path / "lost"
    pipeline = AnalyzePipeline(
        _ListReader([_read_frame(index, stamp, image) for index, stamp in enumerate(frames)]),
        _TimedDetector(frames),
        run_id=_RUN,
        output_dir=out,
        analysis_fps=5,
        model_version="custom-test",
    )
    pipeline.run()
    detections = pq.read_table(out / "detections.parquet").to_pylist()
    assert _at(detections, 0.6) == []
    status = _at(pq.read_table(out / "frame_status.parquet").to_pylist(), 0.6)[0]
    assert status["num_detections"] == 0
    assert status["analyzed"] is True
    held = {row["class_name"]: row for row in _at(detections, 0.4)}
    assert _xyxy(held["haul_truck"]) == pytest.approx(_TRUCK)
    assert _xyxy(held["excavator_body"]) == pytest.approx(_BODY)
    assert held["haul_truck"]["class_name"] == "haul_truck"
    missed = pipeline.session.lost_buffer_vehicles()
    assert {item.class_name for item in missed} == {"haul_truck", "excavator_body"}
    assert all(not hasattr(item, "xyxy") for item in missed)
    for class_name, box in (("haul_truck", _TRUCK), ("excavator_body", _BODY)):
        seen = _track_at(pipeline, 0.4, class_name)
        blank = _track_at(pipeline, 0.6, class_name)
        assert seen.state is ObservationState.observed
        assert seen.is_observed is True
        assert seen.xyxy == pytest.approx(list(box))
        assert seen.track_id == held[class_name]["track_id_nullable"]
        assert blank.track_id == seen.track_id
        assert blank.state is ObservationState.missing
        assert blank.state is not ObservationState.predicted
        assert blank.is_observed is False
        assert blank.xyxy is None
        assert (blank.x1, blank.y1, blank.x2, blank.y2) == (None, None, None, None)
        assert blank.reason is ObservationReason.no_detection
        assert blank.reason is not ObservationReason.occluded_if_known
        assert blank.last_observed_time_s == pytest.approx(0.4)
        assert blank.identity_uncertain is False
    assert calls == []
    stored_tracks = pq.read_table(out / "track_states.parquet").to_pylist()
    for class_name in ("haul_truck", "excavator_body"):
        stored_blank = next(
            row
            for row in stored_tracks
            if row["class_name"] == class_name and row["source_time_s"] == pytest.approx(0.6)
        )
        assert stored_blank["state"] == "missing"
        assert stored_blank["is_observed"] is False
        assert stored_blank["x1_nullable"] is None
        assert stored_blank["parent_track_id_nullable"] is None
        assert stored_blank["parent_track_id_nullable"] != 0


def test_ambiguous_bed_through_analyze_sets_identity_uncertain(tmp_path: Path) -> None:
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    detector = FakeDetector(
        class_names=("excavator_body", "bucket", "haul_truck", "truck_bed"),
        origin=CheckpointOrigin.custom,
        scripted=(
            ScriptedDetection(_TIE_BED, 3, 0.91, "truck_bed"),
            ScriptedDetection(_TIE_RIGHT, 2, 0.95, "haul_truck"),
            ScriptedDetection(_TIE_LEFT, 2, 0.94, "haul_truck"),
        ),
    )
    out = tmp_path / "tie"
    pipeline = AnalyzePipeline(
        _ListReader([_read_frame(0, 0.0, image)]),
        detector,
        run_id=_RUN,
        output_dir=out,
        analysis_fps=5,
        model_version="custom-test",
        vehicles=VehicleTrackers.from_pilot(analysis_fps=5, tracker_cls=_DistinctTracker),
    )
    pipeline.run()
    rows = pq.read_table(out / "detections.parquet").to_pylist()
    bed = next(row for row in rows if row["class_name"] == "truck_bed")
    assert bed["observation_state"] == "ambiguous"
    assert _xyxy(bed) == pytest.approx(_TIE_BED)
    assert all(row["observation_state"] != "predicted" for row in rows)
    track = next(row for row in pipeline.states.track_states if row.class_name == "truck_bed")
    assert track.state is ObservationState.ambiguous
    assert track.identity_uncertain is True
    assert track.is_observed is True
    assert track.parent_track_id is None
    assert track.xyxy == pytest.approx(list(_TIE_BED))
    trucks = [row for row in pipeline.states.track_states if row.class_name == "haul_truck"]
    assert trucks
    assert all(row.identity_uncertain is False for row in trucks)
    assert all(row.state is ObservationState.observed for row in trucks)


def test_frame_status_keeps_empty_frames_and_omits_unanalyzed(tmp_path: Path) -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    frames = [
        _read_frame(0, 0.0, image),
        _read_frame(1, 0.05, image),
        _read_frame(2, 0.20, image),
    ]
    out = tmp_path / "grid"
    pipeline = AnalyzePipeline(
        _ListReader(frames),
        FakeDetector(class_names=("excavator_body",), scripted=()),
        run_id=_RUN,
        output_dir=out,
        analysis_fps=5,
        model_version="custom-test",
    )
    pipeline.run()
    status = pq.read_table(out / "frame_status.parquet").to_pylist()
    times = [float(row["source_time_s"]) for row in status]
    assert times == pytest.approx([0.0, 0.20])
    assert 0.05 not in times
    assert all(row["analyzed"] is True for row in status)
    assert all(row["num_detections"] == 0 for row in status)
    assert status[0]["analysis_index"] == 0
    assert status[1]["analysis_index"] == 1
    assert set(status[0]) == set(FRAME_STATUS_COLUMNS)
    assert pq.read_table(out / "detections.parquet").to_pylist() == []
    assert pipeline.states.decode_gaps == []


def test_corrupt_frame_is_decode_gap_not_an_idle_interval(tmp_path: Path) -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    out = tmp_path / "gap"
    pipeline = AnalyzePipeline(
        _ListReader([_corrupt_frame(0, 0.0), _read_frame(1, 0.2, image)]),
        FakeDetector(class_names=("excavator_body",), scripted=()),
        run_id=_RUN,
        output_dir=out,
        analysis_fps=5,
        model_version="custom-test",
    )
    pipeline.run()
    status = pq.read_table(out / "frame_status.parquet").to_pylist()
    assert [float(row["source_time_s"]) for row in status] == pytest.approx([0.0, 0.2])
    corrupt = status[0]
    assert corrupt["decode_status"] == "corrupt"
    assert corrupt["analyzed"] is False
    assert corrupt["num_detections"] == 0
    assert corrupt["bucket_observation_state"] is None
    assert status[1]["analyzed"] is True
    assert status[1]["num_detections"] == 0
    assert len(pipeline.states.decode_gaps) == 1
    gap = pipeline.states.decode_gaps[0]
    assert gap.reason is ObservationReason.decode_gap
    assert gap.is_equipment_idle is False
    assert gap.reason is not ObservationReason.no_detection
    assert gap.reason is not ObservationReason.occluded_if_known
    assert gap.source_time_s == pytest.approx(0.0)
    blob = json.dumps(corrupt)
    assert "downtime" not in blob
    assert "простой" not in blob
    assert "idle" not in blob.lower()
    assert pipeline.states.track_states == []
    assert pq.read_table(out / "detections.parquet").to_pylist() == []


def _ledger() -> ObservationLedger:
    return ObservationLedger(detection_floor=_FLOOR, display_threshold=_DISPLAY)


def _dash_frame(stamp: float):
    observation = BucketObservation(
        bucket_observation_id=f"{_RUN}:{_VIDEO}:unset:bucket:0",
        index=None,
        parent_track_id=None,
        association_state=ObservationState.predicted,
        association_score=None,
        xyxy=_PREDICTED,
        is_observed=False,
        reason=ObservationReason.no_detection,
        segments=(),
        last_observed_time_s=0.0,
    )
    bucket_frame = BucketFrame(
        source_time_s=stamp,
        multiple_excavators=False,
        bucket_observation_state=ObservationState.predicted,
        observations=(observation,),
    )
    return _ledger().observe(
        _empty(stamp),
        tracked_rows=(),
        bucket_frame=bucket_frame,
        bed_frame=None,
        lost_vehicles=(),
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        source_time_s=stamp,
        reset_count=1,
    )


def _track_at(pipeline: AnalyzePipeline, stamp: float, class_name: str) -> TrackStateRow:
    matches = [
        row
        for row in pipeline.states.track_states
        if row.class_name == class_name and row.source_time_s == pytest.approx(stamp)
    ]
    assert len(matches) == 1
    return matches[0]


def _watch_analytics(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from excavation_lab import analytics

    calls: list[str] = []
    real = analytics.unavailable_error

    def spy():
        calls.append("unavailable_error")
        return real()

    monkeypatch.setattr(analytics, "unavailable_error", spy)
    return calls


def _empty(stamp: float) -> FrameDetections:
    return _frame((), (), (), stamp)


def _frame(
    names: tuple[str, ...],
    scores: tuple[float, ...],
    boxes: tuple[tuple[float, float, float, float], ...],
    stamp: float,
) -> FrameDetections:
    mapping = {"excavator_body": 0, "bucket": 1, "haul_truck": 2, "truck_bed": 3}
    count = len(names)
    xyxy = np.zeros((count, 4), dtype=np.float64)
    if count:
        xyxy[:] = np.asarray(boxes, dtype=np.float64)
    return FrameDetections(
        timestamp_s=stamp,
        frame_width=640,
        frame_height=480,
        origin=CheckpointOrigin.custom,
        xyxy=xyxy,
        class_id=np.asarray([mapping[name] for name in names], dtype=np.int64),
        class_name=names,
        score=np.asarray(scores, dtype=np.float64) if count else np.zeros((0,), dtype=np.float64),
    )


def _at(rows: list[dict], stamp: float) -> list[dict]:
    return [row for row in rows if abs(float(row["source_time_s"]) - stamp) < 1e-6]


def _xyxy(row: dict) -> list[float]:
    return [float(row[name]) for name in _BOXES]


class _Body:
    def __init__(
        self,
        track_id: str,
        xyxy: tuple[float, float, float, float],
        class_name: str = "excavator_body",
    ) -> None:
        self.track_id = track_id
        self.xyxy = xyxy
        self.class_name = class_name


class _DistinctTracker:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def update(
        self,
        detections: Detections,
        frame: object = None,
        timestamp: float | None = None,
    ) -> Detections:
        del frame, timestamp
        if len(detections) == 0:
            return detections
        confidence = detections.confidence
        class_id = detections.class_id
        replaced = Detections(
            xyxy=np.array(detections.xyxy, copy=True),
            confidence=None if confidence is None else np.array(confidence, copy=True),
            class_id=None if class_id is None else np.array(class_id, copy=True),
            data={key: np.array(value, copy=True) for key, value in detections.data.items()},
        )
        replaced.tracker_id = np.arange(len(detections), dtype=int)
        return replaced

    def reset(self) -> None:
        return None


class _TimedDetector:
    def __init__(self, frames: dict[float, FrameDetections]) -> None:
        self.frames = frames

    def predict(self, frame_rgb: np.ndarray, timestamp_s: float) -> FrameDetections:
        del frame_rgb
        for stamp, frame in self.frames.items():
            if abs(stamp - timestamp_s) < 1e-9:
                return frame
        raise AssertionError(timestamp_s)

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            origin=CheckpointOrigin.custom,
            class_names=("excavator_body", "bucket", "haul_truck", "truck_bed"),
            boxes_are_original=True,
            variant="fake",
            resolution=384,
        )

    def close(self) -> None:
        return None


class _ListReader:
    def __init__(self, frames: list[ReadFrame]) -> None:
        self.frames = frames

    def iterate(self, start_s: float = 0.0, end_s: float | None = None):
        for frame in self.frames:
            stamp = frame.source_time_s
            if stamp is not None and not in_half_open(stamp, start_s, end_s):
                continue
            yield frame


def _read_frame(index: int, time_s: float, image: np.ndarray) -> ReadFrame:
    return ReadFrame(
        video_id=_VIDEO,
        source_frame_index=index,
        source_pts=int(time_s * 1000),
        time_base_numerator=1,
        time_base_denominator=1000,
        source_time_s=time_s,
        span_end_s=None,
        start_offset_s=0.0,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        coded_width=int(image.shape[1]),
        coded_height=int(image.shape[0]),
        rotation_deg=0,
        image=image,
        decode_status=DecodeStatus.ok,
        time_mode=TimeMode.pts,
    )


def _corrupt_frame(index: int, time_s: float) -> ReadFrame:
    return ReadFrame(
        video_id=_VIDEO,
        source_frame_index=index,
        source_pts=int(time_s * 1000),
        time_base_numerator=1,
        time_base_denominator=1000,
        source_time_s=time_s,
        span_end_s=None,
        start_offset_s=0.0,
        width=16,
        height=16,
        coded_width=16,
        coded_height=16,
        rotation_deg=0,
        image=None,
        decode_status=DecodeStatus.corrupt,
        time_mode=TimeMode.pts,
    )
