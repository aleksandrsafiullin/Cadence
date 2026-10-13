"""ByteTrack gets real scores and a 30 fps lost buffer. Parts are not submitted."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from supervision import Detections
from trackers import ByteTrackTracker

from excavation_lab.media.reader import ReadFrame
from excavation_lab.media.timebase import TimeMode, in_half_open
from excavation_lab.models.adapter import (
    CheckpointOrigin,
    FakeDetector,
    FrameDetections,
    ScriptedDetection,
)
from excavation_lab.schemas import DecodeStatus, load_pilot
from excavation_lab.tracking.analyze import AnalyzePipeline
from excavation_lab.tracking.vehicles import (
    EXCAVATOR_BODY_CLASSES,
    HAUL_TRUCK_CLASSES,
    TIMESTAMP_API,
    VehicleTrackers,
    seconds_to_lost_track_buffer,
    tracker_arguments,
)
from excavation_lab.tracking.vehicles import ByteTrackTracker as AdapterTracker


class _RecordingTracker:
    """Stand-in. Returns a zero box so the adapter must keep the detector box."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.updates: list[dict[str, object]] = []
        self.reset_count = 0

    def update(
        self,
        detections: Detections,
        frame: object = None,
        timestamp: float | None = None,
    ) -> Detections:
        confidence = detections.confidence
        self.updates.append(
            {
                "xyxy": np.array(detections.xyxy, copy=True),
                "confidence": None if confidence is None else np.array(confidence, copy=True),
                "frame": frame,
                "timestamp": timestamp,
            }
        )
        if len(detections) == 0:
            return detections
        class_id = detections.class_id
        replaced = Detections(
            xyxy=np.zeros_like(detections.xyxy),
            confidence=None if confidence is None else np.array(confidence, copy=True),
            class_id=None if class_id is None else np.array(class_id, copy=True),
            data={key: np.array(value, copy=True) for key, value in detections.data.items()},
        )
        replaced.tracker_id = np.full(len(detections), 4, dtype=int)
        return replaced

    def reset(self) -> None:
        self.reset_count += 1


class _ListReader:
    def __init__(self, frames: list[ReadFrame]) -> None:
        self.frames = frames

    def iterate(self, start_s: float = 0.0, end_s: float | None = None):
        for frame in self.frames:
            stamp = frame.source_time_s
            if stamp is not None and not in_half_open(stamp, start_s, end_s):
                continue
            yield frame


def test_seconds_to_lost_track_buffer_is_reference_frames() -> None:
    assert seconds_to_lost_track_buffer(2.0) == 60


def test_constructor_receives_60_when_analysis_fps_is_5() -> None:
    trackers = VehicleTrackers.from_pilot(analysis_fps=5, tracker_cls=_RecordingTracker)
    assert trackers.analysis_fps == 5
    assert trackers.haul_truck is not trackers.excavator_body
    for tracker in (trackers.haul_truck, trackers.excavator_body):
        kwargs = tracker.kwargs
        assert kwargs["lost_track_buffer"] == 60
        assert kwargs["lost_track_buffer"] != 10
        assert kwargs["frame_rate"] == pytest.approx(30.0)
        assert kwargs["high_conf_det_threshold"] == pytest.approx(0.35)
        assert kwargs["track_activation_threshold"] == pytest.approx(0.45)
        assert kwargs["minimum_consecutive_frames"] == 2
        assert 5 not in kwargs.values()
        assert 10 not in kwargs.values()


def test_installed_names_and_real_class() -> None:
    assert AdapterTracker is ByteTrackTracker
    assert ByteTrackTracker.__module__.startswith("trackers")
    init = inspect.signature(ByteTrackTracker.__init__).parameters
    update = inspect.signature(ByteTrackTracker.update).parameters
    assert "high_conf_det_threshold" in init
    assert "track_activation_threshold" in init
    assert "minimum_consecutive_frames" in init
    assert "lost_track_buffer" in init
    assert "source_time_s" not in update
    assert "timestamp" in update
    assert TIMESTAMP_API == "timestamp"

    arguments = tracker_arguments(load_pilot().bytetrack)
    assert arguments["lost_track_buffer"] == 60
    tracker = ByteTrackTracker(**arguments)
    assert tracker.maximum_frames_without_update == 60
    assert tracker.maximum_time_without_update == pytest.approx(2.0)
    assert tracker.high_conf_det_threshold == pytest.approx(0.35)
    assert tracker.track_activation_threshold == pytest.approx(0.45)
    assert tracker.minimum_consecutive_frames == 2
    tracker.update(Detections.empty(), timestamp=1.25)


def test_class_lists_are_disjoint_and_exclude_parts() -> None:
    assert HAUL_TRUCK_CLASSES.isdisjoint(EXCAVATOR_BODY_CLASSES)
    assert HAUL_TRUCK_CLASSES == frozenset({"haul_truck"})
    assert EXCAVATOR_BODY_CLASSES == frozenset({"excavator_body"})
    covered = HAUL_TRUCK_CLASSES | EXCAVATOR_BODY_CLASSES
    assert "bucket" not in covered
    assert "truck_bed" not in covered


def test_score_floor_reaches_update_and_display_threshold_does_not_gate() -> None:
    boxes = (
        (0.0, 0.0, 10.0, 10.0),
        (20.0, 0.0, 30.0, 10.0),
        (40.0, 0.0, 50.0, 10.0),
        (0.0, 20.0, 10.0, 30.0),
        (0.0, 40.0, 10.0, 50.0),
        (0.0, 60.0, 10.0, 70.0),
    )
    names = (
        "haul_truck",
        "haul_truck",
        "haul_truck",
        "excavator_body",
        "bucket",
        "truck_bed",
    )
    scores = (0.12, 0.09, 0.34, 0.12, 0.99, 0.80)
    detections = _detections(names, scores, boxes)
    original = np.array(detections.xyxy, copy=True)
    trackers = VehicleTrackers.from_pilot(analysis_fps=5, tracker_cls=_RecordingTracker)
    assert trackers.display_threshold == pytest.approx(0.35)
    assert trackers.detection_floor == pytest.approx(0.10)
    rows = trackers.update(detections, source_time_s=4.25)
    assert np.array_equal(detections.xyxy, original)

    haul = _boxes_and_scores(trackers.haul_truck)
    body = _boxes_and_scores(trackers.excavator_body)
    assert haul["scores"] == pytest.approx((0.12, 0.34))
    assert 0.09 not in haul["scores"]
    assert body["scores"] == pytest.approx((0.12,))
    assert haul["timestamp"] == pytest.approx(4.25)
    assert body["timestamp"] == pytest.approx(4.25)
    assert haul["frame"] is None
    assert body["frame"] is None
    forbidden = {(0.0, 40.0, 10.0, 50.0), (0.0, 60.0, 10.0, 70.0)}
    submitted = {tuple(box) for box in haul["xyxy"]} | {tuple(box) for box in body["xyxy"]}
    assert forbidden.isdisjoint(submitted)
    assert (0.0, 0.0, 10.0, 10.0) in submitted
    assert (0.0, 0.0, 10.0, 10.0) not in {tuple(box) for box in body["xyxy"]}
    assert (0.0, 20.0, 10.0, 30.0) not in {tuple(box) for box in haul["xyxy"]}

    by_name_score = {(row.class_name, round(row.score, 2)): row for row in rows}
    kept = by_name_score[("haul_truck", 0.12)]
    dropped = by_name_score[("haul_truck", 0.09)]
    part = by_name_score[("bucket", 0.99)]
    bed = by_name_score[("truck_bed", 0.80)]
    assert kept.local_track_id == 4
    assert kept.source_time_s == pytest.approx(4.25)
    assert kept.xyxy == pytest.approx((0.0, 0.0, 10.0, 10.0))
    assert dropped.local_track_id is None
    assert part.local_track_id is None
    assert bed.local_track_id is None
    assert all(row.source_time_s == pytest.approx(4.25) for row in rows)


def test_each_tracker_resets() -> None:
    trackers = VehicleTrackers.from_pilot(tracker_cls=_RecordingTracker)
    trackers.haul_truck.reset()
    trackers.excavator_body.reset()
    assert trackers.haul_truck.reset_count == 1
    assert trackers.excavator_body.reset_count == 1
    trackers.reset()
    assert trackers.haul_truck.reset_count == 2
    assert trackers.excavator_body.reset_count == 2


def test_analyze_fills_local_ids_without_replacing_boxes(tmp_path: Path) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    detector = FakeDetector(
        class_names=("haul_truck", "bucket"),
        scripted=(
            ScriptedDetection((1.0, 2.0, 5.0, 6.0), 0, 0.12, "haul_truck"),
            ScriptedDetection((1.0, 1.0, 2.0, 2.0), 1, 0.09, "haul_truck"),
            ScriptedDetection((3.0, 3.0, 4.0, 4.0), 1, 0.99, "bucket"),
        ),
    )
    trackers = VehicleTrackers.from_pilot(analysis_fps=5, tracker_cls=_RecordingTracker)
    out = tmp_path / "tracked"
    pipeline = AnalyzePipeline(
        _ListReader([_read_frame(40, 0.21, image)]),
        detector,
        run_id="tracked",
        output_dir=out,
        analysis_fps=5,
        model_version="custom-test",
        vehicles=trackers,
    )
    pipeline.run()
    haul_updates = [
        item
        for item in trackers.haul_truck.updates
        if item["confidence"] is not None and len(item["confidence"])
    ]
    assert len(haul_updates) == 1
    assert haul_updates[0]["timestamp"] == pytest.approx(0.21)
    assert haul_updates[0]["timestamp"] != 40
    assert haul_updates[0]["frame"] is None
    assert haul_updates[0]["confidence"] == pytest.approx((0.12,))
    body_boxes = [
        item["xyxy"]
        for item in trackers.excavator_body.updates
        if len(item["xyxy"])
    ]
    assert body_boxes == []
    submitted = {tuple(np.round(box, 5)) for box in haul_updates[0]["xyxy"]}
    assert (3.0, 3.0, 4.0, 4.0) not in submitted
    assert (1.0, 1.0, 2.0, 2.0) not in submitted

    rows = pq.read_table(out / "detections.parquet").to_pylist()
    assert len(rows) == 3
    by_score = {round(float(row["score"]), 2): row for row in rows}
    strong = by_score[0.12]
    weak = by_score[0.09]
    bucket = by_score[0.99]
    assert strong["class_name"] == "haul_truck"
    assert strong["track_id_nullable"] == "tracked:clip:unset:haul_truck:4"
    assert [float(strong[name]) for name in ("x1", "y1", "x2", "y2")] == pytest.approx(
        [1.0, 2.0, 5.0, 6.0]
    )
    assert float(strong["source_time_s"]) == pytest.approx(0.21)
    assert weak["track_id_nullable"] is None
    assert bucket["class_name"] == "bucket"
    assert bucket["track_id_nullable"] is None
    assert bucket["parent_track_id_nullable"] is None
    assert [float(bucket[name]) for name in ("x1", "y1", "x2", "y2")] == pytest.approx(
        [3.0, 3.0, 4.0, 4.0]
    )


def _detections(
    names: tuple[str, ...],
    scores: tuple[float, ...],
    boxes: tuple[tuple[float, float, float, float], ...],
) -> FrameDetections:
    return FrameDetections(
        timestamp_s=4.25,
        frame_width=100,
        frame_height=80,
        origin=CheckpointOrigin.custom,
        xyxy=np.asarray(boxes, dtype=np.float64),
        class_id=np.arange(len(names), dtype=np.int64),
        class_name=names,
        score=np.asarray(scores, dtype=np.float64),
    )


def _boxes_and_scores(tracker: _RecordingTracker) -> dict[str, object]:
    nonempty = [item for item in tracker.updates if len(item["xyxy"])]
    assert len(nonempty) == 1
    item = nonempty[0]
    confidence = item["confidence"]
    assert confidence is not None
    return {
        "scores": tuple(float(value) for value in confidence),
        "xyxy": [tuple(float(value) for value in box) for box in item["xyxy"]],
        "timestamp": item["timestamp"],
        "frame": item["frame"],
    }


def _read_frame(index: int, time_s: float, image: np.ndarray) -> ReadFrame:
    return ReadFrame(
        video_id="clip",
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
