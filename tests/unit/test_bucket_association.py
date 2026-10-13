"""Bucket association: normalized distance, ambiguity, trail, extrapolation."""

from __future__ import annotations

from dataclasses import dataclass
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
)
from excavation_lab.schemas import DecodeStatus, ObservationState
from excavation_lab.tracking import analyze as analyze_mod
from excavation_lab.tracking.analyze import AnalyzePipeline
from excavation_lab.tracking.bucket import (
    CONTINUITY_BONUS,
    MAX_NORMALIZED_CENTER_DISTANCE,
    BucketAssociator,
    normalized_center_distance,
    pair_score,
)
from excavation_lab.tracking.session import (
    RESET_SEEK,
    TrackingSession,
    public_track_id,
)
from excavation_lab.tracking.vehicles import VehicleTrackers

_RUN = "run-b"
_VIDEO = "clip"
_PIXEL_GATE = 10.0
_SCORE = 0.9


@dataclass(frozen=True, slots=True)
class _Body:
    track_id: str
    xyxy: tuple[float, float, float, float]
    class_name: str = "excavator_body"
    source_time_s: float = 0.0


def test_distance_is_normalized_by_body_size() -> None:
    """A 10 px gate rejects the large-body pair and accepts the small-body pair."""
    large_body = (0.0, 0.0, 400.0, 200.0)
    large_bucket = (215.0, 95.0, 225.0, 105.0)
    large_pixels = _pixel_distance(large_bucket, large_body)
    large_norm = normalized_center_distance(large_bucket, large_body)
    assert large_pixels > _PIXEL_GATE
    assert large_norm == pytest.approx(large_pixels / 400.0)
    assert large_norm < MAX_NORMALIZED_CENTER_DISTANCE

    small_body = (0.0, 0.0, 4.0, 4.0)
    small_bucket = (10.0, 1.0, 12.0, 3.0)
    small_pixels = _pixel_distance(small_bucket, small_body)
    small_norm = normalized_center_distance(small_bucket, small_body)
    assert small_pixels < _PIXEL_GATE
    assert small_norm == pytest.approx(small_pixels / 4.0)
    assert small_norm > MAX_NORMALIZED_CENTER_DISTANCE

    truck = _Body(
        _track("haul_truck", 0),
        large_bucket,
        class_name="haul_truck",
    )
    excavator = _Body(_track("excavator_body", 0), large_body)
    attached, frame = _step(
        BucketAssociator.from_pilot(),
        ("bucket",),
        (_SCORE,),
        (large_bucket,),
        0.0,
        (truck, excavator),
    )
    observation = attached.observations[0]
    assert observation.parent_track_id == excavator.track_id
    assert observation.parent_track_id != truck.track_id
    assert observation.bucket_observation_id == public_track_id(_RUN, _VIDEO, None, "bucket", 0)
    assert observation.bucket_observation_id != observation.parent_track_id
    assert observation.association_state is ObservationState.observed
    assert frame.class_name == ("bucket",)
    assert _boxes(frame) == [large_bucket]

    rejected = BucketAssociator.from_pilot().update(
        _detections(("bucket",), (_SCORE,), (small_bucket,), 0.0),
        bodies=(_Body(_track("excavator_body", 0), small_body),),
        source_time_s=0.0,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )
    assert rejected.observations[0].parent_track_id is None
    assert rejected.observations[0].bucket_observation_id != _track("excavator_body", 0)


def test_scores_inside_the_margin_do_not_pick_the_first_parent() -> None:
    bucket = (0.0, 0.0, 10.0, 10.0)
    worse = _Body(_track("excavator_body", 1), (-33.0, -45.0, 67.0, 55.0))
    better = _Body(_track("excavator_body", 7), (-40.0, -45.0, 60.0, 55.0))
    worse_norm = normalized_center_distance(bucket, worse.xyxy)
    better_norm = normalized_center_distance(bucket, better.xyxy)
    worse_score = pair_score(_SCORE, worse_norm, continued=False)
    better_score = pair_score(_SCORE, better_norm, continued=False)
    assert worse_norm == pytest.approx(0.12)
    assert better_norm == pytest.approx(0.05)
    assert better_score > worse_score
    assert better_score - worse_score <= 0.1
    # Worse anchor is first. Taking the first parent would keep excavator 1.
    result = BucketAssociator.from_pilot().update(
        _detections(("bucket",), (_SCORE,), (bucket,), 1.0),
        bodies=(worse, better),
        source_time_s=1.0,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )
    observation = result.observations[0]
    assert result.bucket_observation_state is ObservationState.ambiguous
    assert observation.association_state is ObservationState.ambiguous
    assert observation.parent_track_id is None
    assert observation.parent_track_id != worse.track_id
    assert observation.association_score is None
    assert observation.reason is not None
    assert observation.reason.value == "ambiguous_association"


def test_clearer_score_wins_when_it_is_not_first() -> None:
    bucket = (-5.0, -5.0, 5.0, 5.0)
    farther = _Body(_track("excavator_body", 0), (100.0, -50.0, 200.0, 50.0))
    nearer = _Body(_track("excavator_body", 4), (-50.0, -50.0, 50.0, 50.0))
    far_norm = normalized_center_distance(bucket, farther.xyxy)
    near_norm = normalized_center_distance(bucket, nearer.xyxy)
    assert far_norm == pytest.approx(1.5)
    assert near_norm == pytest.approx(0.0)
    assert pair_score(_SCORE, near_norm, continued=False) - pair_score(
        _SCORE, far_norm, continued=False
    ) > 0.1
    result = BucketAssociator.from_pilot().update(
        _detections(("bucket",), (_SCORE,), (bucket,), 0.0),
        bodies=(farther, nearer),
        source_time_s=0.0,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )
    assert result.observations[0].parent_track_id == nearer.track_id
    assert result.observations[0].parent_track_id != farther.track_id


def test_two_excavators_are_not_all_attached_to_the_lower_id() -> None:
    low = _Body(_track("excavator_body", 0), (0.0, 0.0, 100.0, 100.0))
    high = _Body(_track("excavator_body", 9), (400.0, 0.0, 500.0, 100.0))
    near_high = (440.0, 40.0, 460.0, 60.0)
    near_low = (40.0, 40.0, 60.0, 60.0)
    assert int(low.track_id.rsplit(":", 1)[-1]) < int(high.track_id.rsplit(":", 1)[-1])
    result = BucketAssociator.from_pilot().update(
        _detections(("bucket", "bucket"), (_SCORE, _SCORE), (near_high, near_low), 0.0),
        bodies=(low, high),
        source_time_s=0.0,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )
    assert result.multiple_excavators is True
    assert result.diagnostic_flags == ("multiple_excavators",)
    by_index = {item.index: item for item in result.observations}
    assert by_index[0].parent_track_id == high.track_id
    assert by_index[1].parent_track_id == low.track_id
    parents = {item.parent_track_id for item in result.observations}
    assert parents == {low.track_id, high.track_id}
    assert low.track_id not in {item.bucket_observation_id for item in result.observations}
    assert len({item.bucket_observation_id for item in result.observations}) == 2


def test_trail_keeps_five_seconds_and_splits_on_the_gap() -> None:
    associator = BucketAssociator.from_pilot()
    box = (0.0, 0.0, 20.0, 10.0)
    body = _Body(_track("excavator_body", 0), (0.0, 0.0, 80.0, 40.0))
    # 0.5 s stays one segment. The next step is a 0.6 s hole.
    for stamp in (0.0, 0.5, 1.1):
        result, _frame = _step(
            associator,
            ("bucket",),
            (_SCORE,),
            (box,),
            stamp,
            (body,),
        )
    segments = result.observations[0].segments
    assert isinstance(segments, tuple)
    assert len(segments) == 2
    assert [point.source_time_s for point in segments[0]] == pytest.approx([0.0, 0.5])
    assert [point.source_time_s for point in segments[1]] == pytest.approx([1.1])

    window = BucketAssociator.from_pilot()
    last = None
    for stamp in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        last, _frame = _step(window, ("bucket",), (_SCORE,), (box,), stamp, (body,))
    assert last is not None
    kept = [
        point.source_time_s
        for segment in last.observations[0].segments
        for point in segment
    ]
    assert kept == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert 0.0 not in kept


def test_extrapolation_is_predicted_and_stays_out_of_the_trail() -> None:
    associator = BucketAssociator.from_pilot()
    body = _Body(_track("excavator_body", 0), (0.0, 0.0, 80.0, 40.0))
    _step(associator, ("bucket",), (_SCORE,), ((0.0, 0.0, 10.0, 10.0),), 0.0, (body,))
    observed, observed_frame = _step(
        associator,
        ("bucket",),
        (_SCORE,),
        ((10.0, 0.0, 20.0, 10.0),),
        0.2,
        (body,),
    )
    trail = _flat_boxes(observed.observations[0])
    assert _boxes(observed_frame) == [(10.0, 0.0, 20.0, 10.0)]

    predicted, predicted_frame = _step(associator, (), (), (), 0.6, (body,))
    dash = predicted.observations[0]
    assert predicted.bucket_observation_state is ObservationState.predicted
    assert dash.association_state is ObservationState.predicted
    assert dash.is_observed is False
    assert dash.xyxy == pytest.approx((30.0, 0.0, 40.0, 10.0))
    assert dash.index is None
    assert _flat_boxes(dash) == trail
    assert dash.xyxy not in trail
    assert predicted_frame.class_name == ()
    assert "event_id" not in dash.__dataclass_fields__

    missing, _frame = _step(associator, (), (), (), 0.7, (body,))
    assert missing.bucket_observation_state is ObservationState.missing
    assert missing.observations[0].xyxy is None
    assert missing.observations[0].is_observed is False
    assert missing.observations[0].association_state is ObservationState.missing
    assert _flat_boxes(missing.observations[0]) == trail


def test_score_band_and_continuity_across_a_missed_body() -> None:
    body = _Body(_track("excavator_body", 2), (0.0, 0.0, 100.0, 80.0))
    bucket = (45.0, 35.0, 55.0, 45.0)
    observed = _only(
        BucketAssociator.from_pilot(),
        0.35,
        bucket,
        (body,),
    )
    assert observed.association_state is ObservationState.observed
    assert observed.parent_track_id == body.track_id
    weak = _only(BucketAssociator.from_pilot(), 0.10, bucket, (body,))
    assert weak.association_state is ObservationState.weak
    assert weak.parent_track_id == body.track_id
    below = BucketAssociator.from_pilot().update(
        _detections(("bucket",), (0.099,), (bucket,), 0.0),
        bodies=(body,),
        source_time_s=0.0,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )
    assert below.observations == ()
    assert below.bucket_observation_state is ObservationState.missing

    assert CONTINUITY_BONUS > 0.1
    kept = _Body(_track("excavator_body", 2), (60.0, 50.0, 160.0, 150.0))
    other = _Body(_track("excavator_body", 8), (40.0, 50.0, 140.0, 150.0))
    moving = (95.0, 95.0, 105.0, 105.0)
    associator = BucketAssociator.from_pilot()
    first, _frame = _step(associator, ("bucket",), (_SCORE,), (moving,), 0.0, (kept,))
    assert first.observations[0].parent_track_id == kept.track_id
    missed, missed_frame = _step(associator, ("bucket",), (_SCORE,), (moving,), 0.2, ())
    assert missed.observations[0].parent_track_id is None
    assert missed_frame.class_name == ("bucket",)
    assert _boxes(missed_frame) == [moving]
    # ``other`` is first. Equal geometry plus continuity keeps ``kept``.
    assert normalized_center_distance(moving, kept.xyxy) == pytest.approx(
        normalized_center_distance(moving, other.xyxy)
    )
    resumed, _frame = _step(
        associator,
        ("bucket",),
        (_SCORE,),
        (moving,),
        0.4,
        (other, kept),
    )
    assert resumed.multiple_excavators is True
    assert resumed.observations[0].parent_track_id == kept.track_id
    assert resumed.observations[0].parent_track_id != other.track_id


def test_session_reset_drops_the_trail_copy() -> None:
    session = TrackingSession(
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        vehicles=VehicleTrackers.from_pilot(analysis_fps=5),
    )
    body = _Body(_track("excavator_body", 0), (0.0, 0.0, 80.0, 40.0))
    _step(session.bucket, ("bucket",), (_SCORE,), ((0.0, 0.0, 10.0, 10.0),), 0.0, (body,))
    _step(session.bucket, ("bucket",), (_SCORE,), ((10.0, 0.0, 20.0, 10.0),), 0.2, (body,))
    predicted, _frame = _step(session.bucket, (), (), (), 0.4, (body,))
    assert predicted.bucket_observation_state is ObservationState.predicted
    session.reset(RESET_SEEK)
    assert session.last_positions == {}
    cleared, cleared_frame = _step(
        session.bucket,
        (),
        (),
        (),
        0.5,
        session.positions_for("excavator_body"),
    )
    assert cleared.bucket_observation_state is ObservationState.missing
    assert cleared.observations == ()
    assert cleared_frame.class_name == ()
    assert session.bucket.last_frame is cleared


def test_bucket_runs_after_bytetrack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    _MarkTracker.order = order
    real_suppress = analyze_mod.suppress

    def wrapped_suppress(detections: FrameDetections, **kwargs: object) -> FrameDetections:
        order.append("guard")
        return real_suppress(detections, **kwargs)

    monkeypatch.setattr(analyze_mod, "suppress", wrapped_suppress)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    pipeline = AnalyzePipeline(
        _ListReader([_read_frame(0, 0.0, image)]),
        FakeDetector(class_names=("excavator_body",), scripted=()),
        run_id="order",
        output_dir=tmp_path / "order",
        analysis_fps=5,
        model_version="custom-test",
        vehicles=VehicleTrackers.from_pilot(analysis_fps=5, tracker_cls=_MarkTracker),
    )
    real_update = pipeline.session.bucket.update

    def wrapped_update(*args: object, **kwargs: object):
        order.append("bucket")
        return real_update(*args, **kwargs)

    pipeline.session.bucket.update = wrapped_update  # type: ignore[method-assign]
    pipeline.run()
    assert order == ["guard", "bytetrack", "bytetrack", "bucket"]


def test_display_threshold_does_not_drop_a_weak_bucket_row(tmp_path: Path) -> None:
    image = np.zeros((96, 120, 3), dtype=np.uint8)
    body = (0.0, 0.0, 100.0, 80.0)
    weak_box = (45.0, 35.0, 55.0, 45.0)
    low_box = (70.0, 60.0, 90.0, 78.0)
    detector = _TimedDetector(
        {
            0.0: _detections(
                ("excavator_body", "bucket", "bucket"),
                (0.95, 0.20, 0.09),
                (body, weak_box, low_box),
                0.0,
            )
        }
    )
    out = tmp_path / "weak"
    pipeline = AnalyzePipeline(
        _ListReader([_read_frame(0, 0.0, image)]),
        detector,
        run_id="weak-run",
        output_dir=out,
        analysis_fps=5,
        model_version="custom-test",
        vehicles=VehicleTrackers.from_pilot(analysis_fps=5, tracker_cls=_IdTracker),
    )
    pipeline.run()
    rows = pq.read_table(out / "detections.parquet").to_pylist()
    assert len(rows) == 3
    by_score = {round(float(row["score"]), 2): row for row in rows}
    weak = by_score[0.20]
    low = by_score[0.09]
    excavator = by_score[0.95]
    assert weak["class_name"] == "bucket"
    assert [float(weak[name]) for name in ("x1", "y1", "x2", "y2")] == pytest.approx(weak_box)
    assert weak["observation_state"] == "weak"
    assert weak["parent_track_id_nullable"] == "weak-run:clip:unset:excavator_body:4"
    assert weak["track_id_nullable"] is None
    assert weak["parent_track_id_nullable"] != weak["track_id_nullable"]
    assert low["class_name"] == "bucket"
    assert [float(low[name]) for name in ("x1", "y1", "x2", "y2")] == pytest.approx(low_box)
    assert low["observation_state"] is None
    assert low["parent_track_id_nullable"] is None
    assert excavator["class_name"] == "excavator_body"
    assert [float(excavator[name]) for name in ("x1", "y1", "x2", "y2")] == pytest.approx(body)
    link = pipeline.session.bucket.last_frame
    assert link is not None
    assert link.observations[0].bucket_observation_id == "weak-run:clip:unset:bucket:0"
    assert link.observations[0].bucket_observation_id != weak["parent_track_id_nullable"]


def _only(
    associator: BucketAssociator,
    score: float,
    box: tuple[float, float, float, float],
    bodies: tuple[_Body, ...],
):
    result, _frame = _step(associator, ("bucket",), (score,), (box,), 0.0, bodies)
    assert len(result.observations) == 1
    return result.observations[0]


def _step(
    associator: BucketAssociator,
    names: tuple[str, ...],
    scores: tuple[float, ...],
    boxes: tuple[tuple[float, float, float, float], ...],
    stamp: float,
    bodies: tuple[object, ...] | tuple[_Body, ...],
):
    frame = _detections(names, scores, boxes, stamp)
    result = associator.update(
        frame,
        bodies=bodies,
        source_time_s=stamp,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )
    return result, frame


def _track(class_name: str, local_id: int) -> str:
    return public_track_id(_RUN, _VIDEO, None, class_name, local_id)


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


def _boxes(frame: FrameDetections) -> list[tuple[float, float, float, float]]:
    return [tuple(float(value) for value in row) for row in frame.xyxy]


def _flat_boxes(observation) -> list[tuple[float, float, float, float]]:
    return [
        point.xyxy
        for segment in observation.segments
        for point in segment
    ]


def _pixel_distance(
    bucket: tuple[float, float, float, float],
    body: tuple[float, float, float, float],
) -> float:
    return float(
        np.hypot(
            (bucket[0] + bucket[2]) / 2.0 - (body[0] + body[2]) / 2.0,
            (bucket[1] + bucket[3]) / 2.0 - (body[1] + body[3]) / 2.0,
        )
    )


class _TimedDetector:
    def __init__(self, frames: dict[float, FrameDetections]) -> None:
        self.frames = frames

    def predict(self, frame_rgb: np.ndarray, timestamp_s: float) -> FrameDetections:
        del frame_rgb
        for stamp, frame in self.frames.items():
            if abs(stamp - timestamp_s) < 1e-9:
                return frame
        raise AssertionError(timestamp_s)

    def metadata(self):
        from excavation_lab.models.adapter import ModelMetadata

        return ModelMetadata(
            origin=CheckpointOrigin.custom,
            class_names=("excavator_body", "bucket", "haul_truck", "truck_bed"),
            boxes_are_original=True,
            variant="fake",
            resolution=384,
        )

    def close(self) -> None:
        return None


class _MarkTracker:
    """Appends ``bytetrack`` onto the list the order test installs."""

    order: list[str] | None = None

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def update(
        self,
        detections: Detections,
        frame: object = None,
        timestamp: float | None = None,
    ) -> Detections:
        del frame, timestamp
        if type(self).order is not None:
            type(self).order.append("bytetrack")
        return detections

    def reset(self) -> None:
        return None


class _IdTracker:
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
