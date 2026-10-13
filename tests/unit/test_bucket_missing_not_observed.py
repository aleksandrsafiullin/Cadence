"""A missed bucket is missing or a short predicted dash, never an observed box."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from excavation_lab.media.reader import ReadFrame
from excavation_lab.media.timebase import TimeMode, in_half_open
from excavation_lab.models.adapter import (
    CheckpointOrigin,
    FrameDetections,
    ModelMetadata,
)
from excavation_lab.schemas import DecodeStatus, ObservationReason, ObservationState
from excavation_lab.tracking.analyze import AnalyzePipeline
from excavation_lab.tracking.bucket import (
    BucketAssociator,
    BucketFrame,
    BucketObservation,
)
from excavation_lab.tracking.session import public_track_id

_RUN = "run-miss"
_VIDEO = "clip"
_BODY = (100.0, 20.0, 220.0, 100.0)
_BUCKET_0 = (0.0, 0.0, 10.0, 10.0)
_BUCKET_1 = (10.0, 0.0, 20.0, 10.0)
_PREDICTED = (20.0, 0.0, 30.0, 10.0)
_TRUCK = (10.0, 30.0, 90.0, 90.0)


def test_missing_bucket_does_not_invent_an_observed_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _watch_analytics(monkeypatch)
    associator = BucketAssociator.from_pilot()
    body_id = public_track_id(_RUN, _VIDEO, None, "excavator_body", 0)
    detections = _detections(("excavator_body",), (0.95,), (_BODY,), 0.0)
    xyxy = np.array(detections.xyxy, copy=True)
    result = associator.update(
        detections,
        bodies=(_Anchor(body_id, _BODY),),
        source_time_s=0.0,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )
    assert result.bucket_observation_state is ObservationState.missing
    assert result.observations == ()
    assert detections.class_name == ("excavator_body",)
    assert np.array_equal(detections.xyxy, xyxy)
    assert calls == []
    assert "event_id" not in BucketObservation.__dataclass_fields__
    assert "event_id" not in BucketFrame.__dataclass_fields__

    _step(associator, (_BUCKET_0,), 0.0)
    _step(associator, (_BUCKET_1,), 0.2)
    late = _step(associator, (), 0.7)
    assert late.bucket_observation_state is ObservationState.missing
    assert late.observations[0].association_state is ObservationState.missing
    assert late.observations[0].xyxy is None
    assert late.observations[0].is_observed is False
    assert late.observations[0].reason is ObservationReason.no_detection
    assert late.observations[0].reason is not ObservationReason.occluded_if_known
    assert calls == []


def test_predicted_dash_is_absent_from_detection_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _watch_analytics(monkeypatch)
    frames = {
        0.0: _detections(("excavator_body", "bucket"), (0.95, 0.9), (_BODY, _BUCKET_0), 0.0),
        0.2: _detections(("excavator_body", "bucket"), (0.95, 0.9), (_BODY, _BUCKET_1), 0.2),
        0.4: _detections(("excavator_body",), (0.95,), (_BODY,), 0.4),
        0.8: _detections(("excavator_body",), (0.95,), (_BODY,), 0.8),
    }
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
    seen: list[tuple[float, tuple[str, ...]]] = []
    bucket_frames: list[BucketFrame] = []
    original = pipeline.session.bucket.update

    def wrapped(detections: FrameDetections, **kwargs: object) -> BucketFrame:
        stamp = float(kwargs["source_time_s"])
        names = tuple(detections.class_name)
        xyxy = np.array(detections.xyxy, copy=True)
        seen.append((stamp, names))
        result = original(detections, **kwargs)
        assert tuple(detections.class_name) == names
        assert np.array_equal(detections.xyxy, xyxy)
        bucket_frames.append(result)
        return result

    pipeline.session.bucket.update = wrapped  # type: ignore[method-assign]
    pipeline.run()

    assert [names for _stamp, names in seen] == [
        ("excavator_body", "bucket"),
        ("excavator_body", "bucket"),
        ("excavator_body",),
        ("excavator_body",),
    ]
    predicted = bucket_frames[2].observations[0]
    assert predicted.association_state is ObservationState.predicted
    assert predicted.is_observed is False
    assert predicted.xyxy == pytest.approx(_PREDICTED)
    assert predicted.xyxy not in [point.xyxy for segment in predicted.segments for point in segment]
    missing = bucket_frames[3].observations[0]
    assert missing.association_state is ObservationState.missing
    assert missing.xyxy is None
    assert missing.is_observed is False

    rows = pq.read_table(out / "detections.parquet").to_pylist()
    status = pq.read_table(out / "frame_status.parquet").to_pylist()
    assert _at(status, 0.4)[0]["bucket_observation_state"] == "predicted"
    assert _at(status, 0.4)[0]["num_detections"] == 1
    assert _at(status, 0.8)[0]["bucket_observation_state"] == "missing"
    assert _at(status, 0.8)[0]["num_detections"] == 1
    dash_rows = _at(rows, 0.4)
    assert [row["class_name"] for row in dash_rows] == ["excavator_body"]
    assert _xyxy(dash_rows[0]) == pytest.approx(_BODY)
    assert _xyxy(dash_rows[0]) != pytest.approx(_PREDICTED)
    assert all(row["class_name"] != "bucket" for row in dash_rows)
    assert all(_xyxy(row) != pytest.approx(_PREDICTED) for row in rows)
    hole_rows = _at(rows, 0.8)
    assert [row["class_name"] for row in hole_rows] == ["excavator_body"]
    assert calls == []


def test_lost_vehicle_is_not_synthesized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _watch_analytics(monkeypatch)
    frames = {
        0.0: _detections(("haul_truck", "excavator_body"), (0.95, 0.95), (_TRUCK, _BODY), 0.0),
        0.2: _detections(("haul_truck", "excavator_body"), (0.95, 0.95), (_TRUCK, _BODY), 0.2),
        0.4: _detections(("haul_truck", "excavator_body"), (0.95, 0.95), (_TRUCK, _BODY), 0.4),
        0.6: _detections((), (), (), 0.6),
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
    excavators = pipeline.session.vehicles.excavator_body.tracks
    trucks = pipeline.session.vehicles.haul_truck.tracks
    assert excavators
    assert trucks
    assert any(track.time_since_update > 0 for track in excavators)
    assert any(track.time_since_update > 0 for track in trucks)
    rows = pq.read_table(out / "detections.parquet").to_pylist()
    assert _at(rows, 0.6) == []
    held = _at(rows, 0.4)
    by_name = {row["class_name"]: row for row in held}
    assert _xyxy(by_name["excavator_body"]) == pytest.approx(_BODY)
    assert _xyxy(by_name["haul_truck"]) == pytest.approx(_TRUCK)
    assert all(row["class_name"] not in {"bucket"} for row in rows)
    status = _at(pq.read_table(out / "frame_status.parquet").to_pylist(), 0.6)[0]
    assert status["num_detections"] == 0
    assert status["bucket_observation_state"] == "missing"
    assert calls == []


class _Anchor:
    def __init__(self, track_id: str, xyxy: tuple[float, float, float, float]) -> None:
        self.track_id = track_id
        self.xyxy = xyxy
        self.class_name = "excavator_body"


def _watch_analytics(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from excavation_lab import analytics

    # EventRow and TruckVisitRow are schema classes. The gate is functions.
    names = [
        name
        for name, value in vars(analytics).items()
        if not name.startswith("_") and inspect.isfunction(value)
    ]
    assert set(names) == {"unavailable_error"}
    assert not any("event" in name.lower() or "dump" in name.lower() for name in names)
    calls: list[str] = []
    real = analytics.unavailable_error

    def spy():
        calls.append("unavailable_error")
        return real()

    monkeypatch.setattr(analytics, "unavailable_error", spy)
    return calls


def _step(
    associator: BucketAssociator,
    boxes: tuple[tuple[float, float, float, float], ...],
    stamp: float,
) -> BucketFrame:
    names = tuple("bucket" for _box in boxes)
    scores = tuple(0.9 for _box in boxes)
    return associator.update(
        _detections(names, scores, boxes, stamp),
        bodies=(),
        source_time_s=stamp,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )


def _detections(
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
    return [float(row[name]) for name in ("x1", "y1", "x2", "y2")]


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
