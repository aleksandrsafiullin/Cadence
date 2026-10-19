"""One bed has one parent. A tie stays null. Two beds may share a truck."""

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
    ScriptedDetection,
)
from excavation_lab.schemas import DecodeStatus, ObservationReason, ObservationState
from excavation_lab.tracking import analyze as analyze_mod
from excavation_lab.tracking.analyze import AnalyzePipeline
from excavation_lab.tracking.bed import (
    BedFrame,
    BedLink,
    BedTruckAssociator,
    bed_containment,
    normalized_center_distance,
    pair_score,
)
from excavation_lab.tracking.export_tables import DETECTIONS_COLUMNS
from excavation_lab.tracking.session import RESET_SEEK, TrackingSession, public_track_id
from excavation_lab.tracking.vehicles import VehicleTrackers

_RUN = "run-bed"
_VIDEO = "clip"
_SCORE = 0.9

# Bed on the left side of a large truck. The partial truck sits further right.
_BED = (80.0, 40.0, 90.0, 50.0)
_TRUCK_B = (0.0, 0.0, 92.0, 100.0)
_TRUCK_A = (84.0, 40.0, 120.0, 50.0)

# Two trucks that both contain the bed. Centers are close, so the scores tie.
_TIE_BED = (40.0, 40.0, 60.0, 60.0)
_TIE_LEFT = (0.0, 0.0, 100.0, 100.0)
_TIE_RIGHT = (10.0, 0.0, 110.0, 100.0)

_SHARED = (0.0, 0.0, 100.0, 100.0)
_BED_CENTER = (40.0, 40.0, 60.0, 60.0)
_BED_NEAR = (42.0, 40.0, 62.0, 60.0)
_BED_EDGE = (5.0, 70.0, 25.0, 90.0)
_BED_OUT = (140.0, 10.0, 180.0, 50.0)


@dataclass(frozen=True, slots=True)
class _Truck:
    track_id: str
    xyxy: tuple[float, float, float, float]
    class_name: str = "haul_truck"


def test_containment_beats_higher_iou() -> None:
    """Truck A wins IoU and sits further right. The parent is the contained truck."""
    associator = BedTruckAssociator.from_pilot()
    assert associator.min_containment == pytest.approx(0.5)
    assert associator.ambiguity_margin == pytest.approx(0.1)
    truck_a = _Truck(_track(1), _TRUCK_A)
    truck_b = _Truck(_track(2), _TRUCK_B)
    assert _iou(_BED, truck_a.xyxy) > _iou(_BED, truck_b.xyxy)
    assert bed_containment(_BED, truck_b.xyxy) > bed_containment(_BED, truck_a.xyxy)
    assert bed_containment(_BED, truck_a.xyxy) >= 0.5
    assert truck_a.xyxy[2] > truck_b.xyxy[2]
    score_a = pair_score(
        bed_containment(_BED, truck_a.xyxy),
        normalized_center_distance(_BED, truck_a.xyxy),
        continued=False,
    )
    score_b = pair_score(
        bed_containment(_BED, truck_b.xyxy),
        normalized_center_distance(_BED, truck_b.xyxy),
        continued=False,
    )
    assert score_b - score_a > 0.1
    frame = _detections(("truck_bed",), (_SCORE,), (_BED,), 0.0)
    original = np.array(frame.xyxy, copy=True)
    names = frame.class_name
    # Higher IoU, and the right-hand box, is first.
    result = associator.update(
        frame,
        trucks=(truck_a, truck_b),
        source_time_s=0.0,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )
    link = result.links[0]
    assert link.parent_track_id == truck_b.track_id
    assert link.parent_track_id != truck_a.track_id
    assert link.ambiguous is False
    assert link.xyxy == pytest.approx(_BED)
    assert frame.class_name == names
    assert np.array_equal(frame.xyxy, original)
    assert "event_id" not in BedLink.__dataclass_fields__
    assert "truck_visit_id" not in BedLink.__dataclass_fields__
    assert "dump" not in BedLink.__dataclass_fields__
    assert "event_id" not in BedFrame.__dataclass_fields__


def test_containment_below_the_floor_is_discarded() -> None:
    bed = (0.0, 0.0, 10.0, 10.0)
    partial = _Truck(_track(1), (5.25, 0.0, 15.25, 10.0))
    kept = _Truck(_track(2), (5.0, 0.0, 15.0, 10.0))
    assert bed_containment(bed, partial.xyxy) == pytest.approx(0.475)
    assert bed_containment(bed, partial.xyxy) < 0.5
    assert bed_containment(bed, kept.xyxy) == pytest.approx(0.5)
    assert _iou(bed, partial.xyxy) > 0.3
    rejected = _one(BedTruckAssociator.from_pilot(), bed, (partial,))
    assert rejected.parent_track_id is None
    assert rejected.ambiguous is False
    assert rejected.association_state is ObservationState.observed
    accepted = _one(BedTruckAssociator.from_pilot(), bed, (kept,))
    assert accepted.parent_track_id == kept.track_id
    assert accepted.ambiguous is False


def test_overlapping_trucks_stay_ambiguous() -> None:
    """Both trucks contain the bed. The right-hand box is first and does not win."""
    right = _Truck(_track(8), _TIE_RIGHT)
    left = _Truck(_track(3), _TIE_LEFT)
    assert right.xyxy[2] > left.xyxy[2]
    assert bed_containment(_TIE_BED, right.xyxy) == pytest.approx(1.0)
    assert bed_containment(_TIE_BED, left.xyxy) == pytest.approx(1.0)
    right_score = pair_score(
        1.0,
        normalized_center_distance(_TIE_BED, right.xyxy),
        continued=False,
    )
    left_score = pair_score(
        1.0,
        normalized_center_distance(_TIE_BED, left.xyxy),
        continued=False,
    )
    assert abs(left_score - right_score) <= 0.1
    link = _one(BedTruckAssociator.from_pilot(), _TIE_BED, (right, left))
    assert link.parent_track_id is None
    assert link.parent_track_id != right.track_id
    assert link.ambiguous is True
    assert link.association_state is ObservationState.ambiguous
    assert link.association_score is None
    assert link.reason is ObservationReason.ambiguous_association
    assert link.xyxy == pytest.approx(_TIE_BED)


def test_two_beds_with_inseparable_scores_are_ambiguous() -> None:
    truck = _Truck(_track(4), _SHARED)
    near_score = pair_score(
        bed_containment(_BED_NEAR, truck.xyxy),
        normalized_center_distance(_BED_NEAR, truck.xyxy),
        continued=False,
    )
    center_score = pair_score(
        bed_containment(_BED_CENTER, truck.xyxy),
        normalized_center_distance(_BED_CENTER, truck.xyxy),
        continued=False,
    )
    assert abs(center_score - near_score) <= 0.1
    result = _beds(
        BedTruckAssociator.from_pilot(),
        (_BED_NEAR, _BED_CENTER),
        (truck,),
    )
    by_index = {item.index: item for item in result.links}
    assert set(by_index) == {0, 1}
    assert by_index[0].parent_track_id is None
    assert by_index[1].parent_track_id is None
    assert by_index[0].ambiguous is True
    assert by_index[1].ambiguous is True
    assert by_index[0].association_state is ObservationState.ambiguous
    assert by_index[1].association_state is ObservationState.ambiguous


def test_second_bed_does_not_drop_the_clearer_parent() -> None:
    """Both beds stay on the truck. A one-bed-per-truck rule drops the first one."""
    truck = _Truck(_track(4), _SHARED)
    edge_score = pair_score(
        bed_containment(_BED_EDGE, truck.xyxy),
        normalized_center_distance(_BED_EDGE, truck.xyxy),
        continued=False,
    )
    center_score = pair_score(
        bed_containment(_BED_CENTER, truck.xyxy),
        normalized_center_distance(_BED_CENTER, truck.xyxy),
        continued=False,
    )
    assert center_score - edge_score > 0.1
    assert bed_containment(_BED_EDGE, truck.xyxy) == pytest.approx(1.0)
    assert bed_containment(_BED_CENTER, truck.xyxy) == pytest.approx(1.0)
    # The weaker bed is first, so a first-wins unique pass would steal the truck.
    result = _beds(
        BedTruckAssociator.from_pilot(),
        (_BED_EDGE, _BED_CENTER),
        (truck,),
    )
    by_index = {item.index: item for item in result.links}
    assert by_index[0].xyxy == pytest.approx(_BED_EDGE)
    assert by_index[1].xyxy == pytest.approx(_BED_CENTER)
    assert by_index[0].parent_track_id == truck.track_id
    assert by_index[1].parent_track_id == truck.track_id
    assert by_index[0].ambiguous is False
    assert by_index[1].ambiguous is False


def test_uncontained_bed_does_not_drop_the_clear_parent() -> None:
    truck = _Truck(_track(4), _SHARED)
    assert bed_containment(_BED_OUT, truck.xyxy) < 0.5
    assert bed_containment(_BED_CENTER, truck.xyxy) == pytest.approx(1.0)
    result = _beds(
        BedTruckAssociator.from_pilot(),
        (_BED_OUT, _BED_CENTER),
        (truck,),
    )
    outside, clear = result.links
    assert outside.xyxy == pytest.approx(_BED_OUT)
    assert outside.parent_track_id is None
    assert outside.ambiguous is False
    assert clear.parent_track_id == truck.track_id
    assert clear.ambiguous is False
    assert clear.xyxy == pytest.approx(_BED_CENTER)


def test_previous_parent_is_a_bonus_not_a_lock() -> None:
    bed = (0.0, 0.0, 10.0, 10.0)
    original = _Truck(_track(1), (0.0, 0.0, 20.0, 20.0))
    shifted = _Truck(_track(1), (4.5, 0.0, 24.5, 20.0))
    other = _Truck(_track(2), (0.0, 0.0, 12.0, 12.0))
    associator = BedTruckAssociator.from_pilot()
    first = _one(associator, bed, (original,), stamp=0.0)
    assert first.parent_track_id == original.track_id
    continued = pair_score(
        bed_containment(bed, shifted.xyxy),
        normalized_center_distance(bed, shifted.xyxy),
        continued=True,
    )
    incoming = pair_score(
        bed_containment(bed, other.xyxy),
        normalized_center_distance(bed, other.xyxy),
        continued=False,
    )
    assert bed_containment(bed, shifted.xyxy) >= 0.5
    assert incoming - continued > 0.1
    moved = _one(associator, bed, (shifted, other), stamp=0.2)
    assert moved.parent_track_id == other.track_id
    assert moved.parent_track_id != shifted.track_id
    assert moved.ambiguous is False


def test_weaker_previous_parent_leaves_the_tie_null() -> None:
    right = _Truck(_track(8), _TIE_RIGHT)
    left = _Truck(_track(3), _TIE_LEFT)
    associator = BedTruckAssociator.from_pilot()
    first = _one(associator, _TIE_BED, (right,), stamp=0.0)
    assert first.parent_track_id == right.track_id
    right_score = pair_score(
        1.0,
        normalized_center_distance(_TIE_BED, right.xyxy),
        continued=True,
    )
    left_score = pair_score(
        1.0,
        normalized_center_distance(_TIE_BED, left.xyxy),
        continued=False,
    )
    assert abs(right_score - left_score) <= 0.1
    tied = _one(associator, _TIE_BED, (left, right), stamp=0.2)
    assert tied.parent_track_id is None
    assert tied.ambiguous is True


def test_truck_miss_keeps_the_copied_parent_without_a_box() -> None:
    left = _Truck(_track(3), _TIE_LEFT)
    right = _Truck(_track(8), _TIE_RIGHT)
    associator = BedTruckAssociator.from_pilot()
    linked = _one(associator, _TIE_BED, (left,), stamp=0.0)
    assert linked.parent_track_id == left.track_id
    frame = _detections(("truck_bed",), (_SCORE,), (_TIE_BED,), 0.2)
    original = np.array(frame.xyxy, copy=True)
    missed = associator.update(
        frame,
        trucks=(),
        source_time_s=0.2,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )
    assert missed.links[0].parent_track_id is None
    assert missed.links[0].xyxy == pytest.approx(_TIE_BED)
    assert frame.class_name == ("truck_bed",)
    assert np.array_equal(frame.xyxy, original)
    assert all(link.xyxy != left.xyxy for link in missed.links)
    resumed = _one(associator, _TIE_BED, (right, left), stamp=0.4)
    assert resumed.parent_track_id == left.track_id
    assert resumed.parent_track_id != right.track_id


def test_reset_clears_the_copied_parent() -> None:
    left = _Truck(_track(3), _TIE_LEFT)
    right = _Truck(_track(8), _TIE_RIGHT)
    session = TrackingSession(
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        vehicles=VehicleTrackers.from_pilot(analysis_fps=5),
    )
    kept = BedTruckAssociator.from_pilot()
    assert _one(session.bed, _TIE_BED, (left,), stamp=0.0).parent_track_id == left.track_id
    assert _one(kept, _TIE_BED, (left,), stamp=0.0).parent_track_id == left.track_id
    session.reset(RESET_SEEK)
    assert session.last_positions == {}
    assert session.bed.last_frame is None
    cleared = _one(session.bed, _TIE_BED, (right, left), stamp=0.2)
    still = _one(kept, _TIE_BED, (right, left), stamp=0.2)
    assert cleared.parent_track_id is None
    assert cleared.ambiguous is True
    assert still.parent_track_id == left.track_id


def test_analyze_writes_the_namespaced_parent_and_leaves_xyxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _watch_analytics(monkeypatch)
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    detector = FakeDetector(
        class_names=("excavator_body", "bucket", "haul_truck", "truck_bed"),
        scripted=(
            ScriptedDetection(_TRUCK_B, 2, 0.95, "haul_truck"),
            ScriptedDetection(_BED, 3, 0.91, "truck_bed"),
        ),
        origin=CheckpointOrigin.custom,
    )
    out = tmp_path / "linked"
    pipeline = AnalyzePipeline(
        _ListReader([_read_frame(0, 0.0, image)]),
        detector,
        run_id=_RUN,
        output_dir=out,
        analysis_fps=5,
        model_version="custom-test",
        vehicles=VehicleTrackers.from_pilot(analysis_fps=5, tracker_cls=_IdTracker),
    )
    pipeline.run()
    rows = pq.read_table(out / "detections.parquet").to_pylist()
    assert len(rows) == 2
    by_name = {row["class_name"]: row for row in rows}
    bed = by_name["truck_bed"]
    truck = by_name["haul_truck"]
    assert [float(bed[name]) for name in ("x1", "y1", "x2", "y2")] == pytest.approx(_BED)
    assert [float(truck[name]) for name in ("x1", "y1", "x2", "y2")] == pytest.approx(_TRUCK_B)
    assert bed["parent_track_id_nullable"] == f"{_RUN}:clip:unset:haul_truck:4"
    assert bed["parent_track_id_nullable"] == truck["track_id_nullable"]
    assert bed["track_id_nullable"] is None
    assert bed["observation_state"] == "observed"
    assert truck["parent_track_id_nullable"] is None
    assert list(bed.keys()) == list(DETECTIONS_COLUMNS)
    assert "event_id" not in bed
    assert "truck_visit_id" not in bed
    assert calls == []


def test_analyze_stores_an_ambiguous_bed_as_null_parent(tmp_path: Path) -> None:
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    detector = FakeDetector(
        class_names=("excavator_body", "bucket", "haul_truck", "truck_bed"),
        scripted=(
            ScriptedDetection(_TIE_BED, 3, 0.91, "truck_bed"),
            ScriptedDetection(_TIE_RIGHT, 2, 0.95, "haul_truck"),
            ScriptedDetection(_TIE_LEFT, 2, 0.94, "haul_truck"),
        ),
        origin=CheckpointOrigin.custom,
    )
    out = tmp_path / "tied"
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
    assert len(rows) == 3
    bed = next(row for row in rows if row["class_name"] == "truck_bed")
    trucks = [row for row in rows if row["class_name"] == "haul_truck"]
    assert [float(bed[name]) for name in ("x1", "y1", "x2", "y2")] == pytest.approx(_TIE_BED)
    assert bed["parent_track_id_nullable"] is None
    assert bed["observation_state"] == "ambiguous"
    assert bed["association_score_nullable"] is None
    assert {row["track_id_nullable"] for row in trucks} == {
        f"{_RUN}:clip:unset:haul_truck:0",
        f"{_RUN}:clip:unset:haul_truck:1",
    }
    assert all(
        tuple(float(row[name]) for name in ("x1", "y1", "x2", "y2")) != _TIE_BED
        for row in trucks
    )


def test_bed_runs_after_the_bucket_on_the_same_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    real_bucket = pipeline.session.bucket.update
    real_bed = pipeline.session.bed.update

    def wrapped_bucket(*args: object, **kwargs: object):
        order.append("bucket")
        return real_bucket(*args, **kwargs)

    def wrapped_bed(*args: object, **kwargs: object):
        order.append("bed")
        return real_bed(*args, **kwargs)

    pipeline.session.bucket.update = wrapped_bucket  # type: ignore[method-assign]
    pipeline.session.bed.update = wrapped_bed  # type: ignore[method-assign]
    pipeline.run()
    assert order == ["guard", "bytetrack", "bytetrack", "bucket", "bed"]


def test_excavator_is_not_a_bed_parent() -> None:
    body = _Truck(
        public_track_id(_RUN, _VIDEO, None, "excavator_body", 0),
        _TRUCK_B,
        class_name="excavator_body",
    )
    link = _one(BedTruckAssociator.from_pilot(), _BED, (body,))
    assert link.parent_track_id is None
    assert link.parent_track_id != body.track_id


def _one(
    associator: BedTruckAssociator,
    box: tuple[float, float, float, float],
    trucks: tuple[_Truck, ...],
    stamp: float = 0.0,
) -> BedLink:
    result = _beds(associator, (box,), trucks, stamp=stamp)
    assert len(result.links) == 1
    return result.links[0]


def _beds(
    associator: BedTruckAssociator,
    boxes: tuple[tuple[float, float, float, float], ...],
    trucks: tuple[_Truck, ...],
    stamp: float = 0.0,
) -> BedFrame:
    frame = _detections(
        tuple("truck_bed" for _box in boxes),
        tuple(_SCORE for _box in boxes),
        boxes,
        stamp,
    )
    original = np.array(frame.xyxy, copy=True)
    names = frame.class_name
    result = associator.update(
        frame,
        trucks=trucks,
        source_time_s=stamp,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )
    assert frame.class_name == names
    assert np.array_equal(frame.xyxy, original)
    return result


def test_catalog_names_gate_the_bed_associator() -> None:
    truck = _Truck(_track(2), _TRUCK_B)
    frame = _detections(("truck_bed",), (_SCORE,), (_BED,), 0.0)
    original = np.array(frame.xyxy, copy=True)
    names = frame.class_name
    blocked = BedTruckAssociator.from_pilot(class_names=frozenset({"shovel"}))
    skipped = blocked.update(
        frame,
        trucks=(truck,),
        source_time_s=0.0,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )
    assert skipped.links == ()
    assert blocked._tracks == []
    assert frame.class_name == names
    assert np.array_equal(frame.xyxy, original)
    assert BedTruckAssociator.from_pilot(class_names=frozenset({"truck_bed"})).enabled() is False
    assert BedTruckAssociator.from_pilot(class_names=frozenset()).enabled() is False
    allowed = BedTruckAssociator.from_pilot(
        class_names=frozenset({"truck_bed", "haul_truck"})
    )
    linked = allowed.update(
        _detections(("truck_bed",), (_SCORE,), (_BED,), 0.0),
        trucks=(truck,),
        source_time_s=0.0,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
    )
    assert linked.links
    assert linked.links[0].parent_track_id == truck.track_id
    assert linked.links[0].xyxy == pytest.approx(_BED)


def _track(local_id: int) -> str:
    return public_track_id(_RUN, _VIDEO, None, "haul_truck", local_id)


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


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = min(left[2], right[2]) - max(left[0], right[0])
    height = min(left[3], right[3]) - max(left[1], right[1])
    if width <= 0.0 or height <= 0.0:
        return 0.0
    intersection = width * height
    union = _area(left) + _area(right) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _area(box: tuple[float, float, float, float]) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def _watch_analytics(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from excavation_lab import analytics

    calls: list[str] = []
    real = analytics.unavailable_error

    def spy():
        calls.append("unavailable_error")
        return real()

    monkeypatch.setattr(analytics, "unavailable_error", spy)
    return calls


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


class _MarkTracker:
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
