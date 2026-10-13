"""Nested truck_bed and bucket stay. Same-class NMS is opt-in."""

from __future__ import annotations

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
    ScriptedDetection,
)
from excavation_lab.schemas import DecodeStatus, load_pilot
from excavation_lab.tracking import analyze as analyze_mod
from excavation_lab.tracking.analyze import AnalyzePipeline
from excavation_lab.tracking.suppress import suppress
from excavation_lab.tracking.vehicles import VehicleTrackers

TRUCK = (0.0, 0.0, 100.0, 100.0)
BED = (5.0, 5.0, 95.0, 95.0)
BODY = (0.0, 0.0, 120.0, 80.0)
BUCKET = (40.0, 20.0, 80.0, 50.0)
TRUCK_HIGH = (1.0, 1.0, 101.0, 101.0)


def test_bed_centered_inside_truck_keeps_both_boxes() -> None:
    assert _center(BED) == pytest.approx(_center(TRUCK))
    assert _inside(BED, TRUCK)
    assert _iou(TRUCK, BED) == pytest.approx(0.81)
    frame = _frame(
        ("haul_truck", "truck_bed"),
        (2, 3),
        (0.90, 0.12),
        (TRUCK, BED),
    )
    original = np.array(frame.xyxy, copy=True)
    for threshold in (None, 0.5):
        result = suppress(frame, same_class_nms_iou=threshold)
        assert result.class_name == ("haul_truck", "truck_bed")
        assert _tuples(result) == [TRUCK, BED]
        assert list(result.class_id) == [2, 3]
        assert result.score[1] == pytest.approx(0.12)
    assert np.array_equal(frame.xyxy, original)


def test_bucket_and_excavator_body_stay_two_rows() -> None:
    assert _inside(BUCKET, BODY)
    assert _iou(BODY, BUCKET) > 0.05
    frame = _frame(
        ("excavator_body", "bucket"),
        (0, 1),
        (0.88, 0.22),
        (BODY, BUCKET),
    )
    result = suppress(frame, same_class_nms_iou=0.05)
    assert result is frame
    assert result.class_name == ("excavator_body", "bucket")
    assert _tuples(result) == [BODY, BUCKET]
    assert list(result.class_id) == [0, 1]


def test_same_class_nms_is_off_by_default_and_keeps_the_bed_when_on() -> None:
    assert load_pilot().same_class_nms_iou is None
    assert _iou(TRUCK, TRUCK_HIGH) > 0.5
    assert _inside(BED, TRUCK_HIGH)
    assert _iou(TRUCK_HIGH, BED) > 0.5
    frame = _frame(
        ("haul_truck", "truck_bed", "haul_truck"),
        (2, 3, 2),
        (0.40, 0.20, 0.90),
        (TRUCK, BED, TRUCK_HIGH),
    )
    original = np.array(frame.xyxy, copy=True)
    names = frame.class_name
    off = suppress(frame, same_class_nms_iou=load_pilot().same_class_nms_iou)
    assert off is frame
    assert off.class_name == names
    assert _tuples(off) == [TRUCK, BED, TRUCK_HIGH]

    on = suppress(frame, same_class_nms_iou=0.5)
    assert on.class_name == ("truck_bed", "haul_truck")
    assert _tuples(on) == [BED, TRUCK_HIGH]
    assert list(on.class_id) == [3, 2]
    assert on.score[0] == pytest.approx(0.20)
    assert on.score[1] == pytest.approx(0.90)
    assert np.array_equal(frame.xyxy, original)
    assert frame.class_name == names


def test_analyze_guards_before_bytetrack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    real = analyze_mod.suppress

    def wrapped(detections: FrameDetections, **kwargs: object) -> FrameDetections:
        order.append("guard")
        return real(detections, **kwargs)

    monkeypatch.setattr(analyze_mod, "suppress", wrapped)

    class _OrderingTracker(_RecordingTracker):
        def update(
            self,
            detections: Detections,
            frame: object = None,
            timestamp: float | None = None,
        ) -> Detections:
            order.append("bytetrack")
            return super().update(detections, frame=frame, timestamp=timestamp)

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    detector = FakeDetector(
        class_names=("excavator_body", "bucket", "haul_truck", "truck_bed"),
        scripted=(
            ScriptedDetection(TRUCK, 2, 0.40, "haul_truck"),
            ScriptedDetection(BED, 3, 0.20, "truck_bed"),
            ScriptedDetection(TRUCK_HIGH, 2, 0.90, "haul_truck"),
        ),
        origin=CheckpointOrigin.custom,
    )
    trackers = VehicleTrackers.from_pilot(analysis_fps=5, tracker_cls=_OrderingTracker)
    out = tmp_path / "guarded"
    pipeline = AnalyzePipeline(
        _ListReader([_read_frame(0, 0.0, image)]),
        detector,
        run_id="nested",
        output_dir=out,
        analysis_fps=5,
        model_version="custom-test",
        vehicles=trackers,
        same_class_nms_iou=0.5,
    )
    pipeline.run()
    assert order == ["guard", "bytetrack", "bytetrack"]

    haul = _nonempty(trackers.haul_truck)
    assert len(haul) == 1
    assert _rows(haul[0]["xyxy"]) == [TRUCK_HIGH]
    confidence = haul[0]["confidence"]
    assert confidence is not None
    assert tuple(float(value) for value in confidence) == pytest.approx((0.90,))
    assert _nonempty(trackers.excavator_body) == []
    submitted = {tuple(np.round(box, 5)) for box in haul[0]["xyxy"]}
    assert BED not in submitted
    assert TRUCK not in submitted

    rows = pq.read_table(out / "detections.parquet").to_pylist()
    assert len(rows) == 2
    by_name = {row["class_name"]: row for row in rows}
    bed = by_name["truck_bed"]
    truck = by_name["haul_truck"]
    assert [float(bed[name]) for name in ("x1", "y1", "x2", "y2")] == pytest.approx(list(BED))
    assert [float(truck[name]) for name in ("x1", "y1", "x2", "y2")] == pytest.approx(
        list(TRUCK_HIGH)
    )
    assert bed["class_id"] == 3
    assert truck["class_id"] == 2
    assert bed["parent_track_id_nullable"] == truck["track_id_nullable"]
    assert bed["track_id_nullable"] is None
    assert truck["track_id_nullable"] == "nested:clip:unset:haul_truck:4"
    assert float(bed["score"]) == pytest.approx(0.20)


class _RecordingTracker:
    """Records the boxes ByteTrack would see. Returns those same boxes."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.updates: list[dict[str, object]] = []

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
            }
        )
        if len(detections) == 0:
            return detections
        class_id = detections.class_id
        replaced = Detections(
            xyxy=np.array(detections.xyxy, copy=True),
            confidence=None if confidence is None else np.array(confidence, copy=True),
            class_id=None if class_id is None else np.array(class_id, copy=True),
            data={key: np.array(value, copy=True) for key, value in detections.data.items()},
        )
        replaced.tracker_id = np.full(len(detections), 4, dtype=int)
        return replaced

    def reset(self) -> None:
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


def _frame(
    names: tuple[str, ...],
    class_ids: tuple[int, ...],
    scores: tuple[float, ...],
    boxes: tuple[tuple[float, float, float, float], ...],
) -> FrameDetections:
    return FrameDetections(
        timestamp_s=1.0,
        frame_width=128,
        frame_height=128,
        origin=CheckpointOrigin.custom,
        xyxy=np.asarray(boxes, dtype=np.float64),
        class_id=np.asarray(class_ids, dtype=np.int64),
        class_name=names,
        score=np.asarray(scores, dtype=np.float64),
    )


def _tuples(detections: FrameDetections) -> list[tuple[float, float, float, float]]:
    return [tuple(float(value) for value in row) for row in detections.xyxy]


def _rows(xyxy: np.ndarray) -> list[tuple[float, float, float, float]]:
    return [tuple(float(value) for value in row) for row in xyxy]


def _nonempty(tracker: _RecordingTracker) -> list[dict[str, object]]:
    return [item for item in tracker.updates if len(item["xyxy"])]


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _inside(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> bool:
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    width = ix2 - ix1
    height = iy2 - iy1
    if width <= 0.0 or height <= 0.0:
        return 0.0
    intersection = width * height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


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
