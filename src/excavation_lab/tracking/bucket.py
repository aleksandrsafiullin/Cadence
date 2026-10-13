"""Bind bucket detections to the working excavator.

Distance is the center gap divided by ``max(body_width, body_height)``.
A fixed pixel radius is not the gate. Two scores inside
``bucket.ambiguity_margin`` stay ``ambiguous``; the earlier anchor does not
win. A valid ``working_area`` adds ``CONTAINMENT_BONUS`` when the bucket
center and that excavator center both lie in the polygon. The bonus is
applied only after the base scores are already outside the margin, so a
bucket inside the zone does not pick a parent for an ambiguous pair.
There is no rule that the face is on the left. The observed trail is the
last ``trail_s`` seconds, split wherever a gap is longer than ``trail_gap_s``.

The only synthesized box in v0.1 is a bucket dash: the last two observed
bucket points, at most ``extrapolation_s`` (0.4 s), ``predicted``, and not
inserted into the trail. A missed ``excavator_body`` or ``haul_truck`` is
not rebuilt from ByteTrack's lost buffer. ``missing`` means this frame has
no reliable bucket observation. It is not a dump and it has no event id.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from excavation_lab.datasets.zones import (
    CameraZones,
    active_working_area_pixels,
    point_in_polygon,
)
from excavation_lab.models.adapter import FrameDetections
from excavation_lab.schemas import (
    ObservationReason,
    ObservationState,
    PilotConfig,
    load_pilot,
)
from excavation_lab.tracking.vehicles import BUCKET, EXCAVATOR_BODY

# Starting gate in body lengths, not a tuned optimum and not a pixel radius.
MAX_NORMALIZED_CENTER_DISTANCE = 2.0
# Larger than the pilot margin (0.1), so an exact tie can keep the previous
# parent. A gap that is still inside the margin stays ambiguous.
CONTINUITY_BONUS = 0.15
# Applied only when the base ranking is already outside the margin.
# It can move a clear near-tie toward the excavator inside the polygon.
# It cannot turn an ambiguous pair into a parent.
CONTAINMENT_BONUS = 0.25
_TIME_EPS = 1e-9


def normalized_center_distance(
    bucket_xyxy: Sequence[float],
    body_xyxy: Sequence[float],
) -> float:
    """Center distance divided by ``max(body_width, body_height)``.

    A non-positive body size is infinite, so that anchor cannot win.
    """
    bucket = _tuple_box(bucket_xyxy)
    body = _tuple_box(body_xyxy)
    scale = max(body[2] - body[0], body[3] - body[1])
    if not math.isfinite(scale) or scale <= 0.0:
        return math.inf
    bucket_center = _center(bucket)
    body_center = _center(body)
    return math.hypot(bucket_center[0] - body_center[0], bucket_center[1] - body_center[1]) / scale


def pair_score(
    detection_score: float,
    normalized_distance: float,
    *,
    continued: bool,
    contained: bool = False,
) -> float:
    """Rank one bucket against one body.

    Proximity is ``1 / (1 + normalized_distance)``. Continuity adds
    ``CONTINUITY_BONUS`` when this body was the parent chosen last time.
    ``contained`` adds ``CONTAINMENT_BONUS`` when this pair lies in a valid
    working area. Callers refuse that term while the base scores are ambiguous.
    """
    proximity = 1.0 / (1.0 + max(float(normalized_distance), 0.0))
    bonus = CONTINUITY_BONUS if continued else 0.0
    zone_bonus = CONTAINMENT_BONUS if contained else 0.0
    return float(detection_score) * proximity + bonus + zone_bonus


@dataclass(frozen=True, slots=True)
class TrailPoint:
    """One observed bucket box. Predicted dashes are not points."""

    source_time_s: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class BucketObservation:
    """One bucket identity on this frame.

    ``xyxy`` is the detector box when ``is_observed`` is true. A predicted
    dash sets ``xyxy`` and ``is_observed=false`` and is absent from
    ``segments``. ``missing`` leaves ``xyxy`` null. There is no event id.
    """

    bucket_observation_id: str
    index: int | None
    parent_track_id: str | None
    association_state: ObservationState
    association_score: float | None
    xyxy: tuple[float, float, float, float] | None
    is_observed: bool
    reason: ObservationReason | None
    segments: tuple[tuple[TrailPoint, ...], ...]
    last_observed_time_s: float | None


@dataclass(frozen=True, slots=True)
class BucketFrame:
    """Associator output for one source timestamp.

    ``segments`` on each observation is the polyline: a sequence of
    segments. A gap longer than ``trail_gap_s`` starts a new segment.
    """

    source_time_s: float
    multiple_excavators: bool
    bucket_observation_state: ObservationState
    observations: tuple[BucketObservation, ...]

    @property
    def diagnostic_flags(self) -> tuple[str, ...]:
        if self.multiple_excavators:
            return ("multiple_excavators",)
        return ()


@dataclass(frozen=True, slots=True)
class _Anchor:
    track_id: str
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class _Det:
    index: int
    score: float
    xyxy: tuple[float, float, float, float]


@dataclass
class _Track:
    """Trail and last chosen parent. Cleared as a whole by ``reset``."""

    local_id: int
    observation_id: str
    points: list[TrailPoint]
    parent_track_id: str | None


class BucketAssociator:
    """Attach ``bucket`` rows to active ``excavator_body`` tracks.

    The session replaces ``last_positions`` on every frame, including a
    one-frame miss. This object keeps the trail and the last chosen parent
    itself. ``TrackingSession.reset`` calls ``reset`` so that copy does not
    survive a seek, a new video, or an analysis restart.
    """

    def __init__(
        self,
        *,
        trail_s: float,
        trail_gap_s: float,
        extrapolation_s: float,
        ambiguity_margin: float,
        detection_floor: float,
        display_threshold: float,
        max_normalized_distance: float = MAX_NORMALIZED_CENTER_DISTANCE,
    ) -> None:
        self.trail_s = _require_range("trail_s", trail_s, upper=None)
        self.trail_gap_s = _require_range("trail_gap_s", trail_gap_s, upper=None)
        self.extrapolation_s = _require_range("extrapolation_s", extrapolation_s, upper=0.4)
        self.ambiguity_margin = _require_margin(ambiguity_margin)
        self.detection_floor = _require_unit("detection_floor", detection_floor)
        self.display_threshold = _require_unit("display_threshold", display_threshold)
        self.max_normalized_distance = _require_range(
            "max_normalized_distance",
            max_normalized_distance,
            upper=None,
        )
        self._tracks: list[_Track] = []
        self._next_local_id = 0
        self.last_frame: BucketFrame | None = None

    @classmethod
    def from_pilot(cls, pilot: PilotConfig | None = None) -> BucketAssociator:
        loaded = load_pilot() if pilot is None else pilot
        return cls(
            trail_s=loaded.bucket.trail_s,
            trail_gap_s=loaded.bucket.trail_gap_s,
            extrapolation_s=loaded.bucket.extrapolation_s,
            ambiguity_margin=loaded.bucket.ambiguity_margin,
            detection_floor=loaded.detection_floor,
            display_threshold=loaded.display_threshold,
        )

    def reset(self) -> None:
        """Drop the trail, the copied parent, and the last frame.

        The local id counter is not rewound, so a later observation in the
        same run does not reuse an id from before the seek.
        """
        self._tracks.clear()
        self.last_frame = None

    def update(
        self,
        detections: FrameDetections,
        *,
        bodies: Sequence[object],
        source_time_s: float,
        run_id: str,
        video_id: str,
        view_id: str | None,
        zones: CameraZones | None = None,
    ) -> BucketFrame:
        """Associate this frame. ``detections`` is not modified.

        ``bodies`` are the excavators confirmed on this frame. A body that
        is only inside ByteTrack's lost buffer is not in that sequence and
        is not turned into a box. An empty ``bodies`` sequence does not
        fall back to the previous frame's anchors. ``zones`` is optional.
        A stale view (``invalid_for_view``) adds no containment bonus.
        Pixel scale is ``detections.frame_width`` on this call.
        """
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id is required")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("video_id is required")
        stamp = float(source_time_s)
        anchors = _anchors(bodies)
        multiple = len(anchors) >= 2
        working_area = active_working_area_pixels(
            zones,
            frame_width=detections.frame_width,
            frame_height=detections.frame_height,
        )
        self._prune(stamp)
        reliable = _reliable_buckets(
            detections,
            detection_floor=self.detection_floor,
        )
        matched, unmatched_dets, unmatched_tracks = _match(
            self._tracks,
            reliable,
            max_normalized_distance=self.max_normalized_distance,
        )
        observations: list[BucketObservation] = []
        for track, det in matched:
            observations.append(self._observe(track, det, anchors, stamp, working_area))
        for det in unmatched_dets:
            track = self._open_track(run_id, video_id, view_id)
            self._tracks.append(track)
            observations.append(self._observe(track, det, anchors, stamp, working_area))
        for track in unmatched_tracks:
            observations.append(self._miss(track, stamp))
        observations.sort(key=lambda item: (item.index is None, item.index or 0))
        frame = BucketFrame(
            source_time_s=stamp,
            multiple_excavators=multiple,
            bucket_observation_state=_frame_state(observations),
            observations=tuple(observations),
        )
        self.last_frame = frame
        return frame

    def _observe(
        self,
        track: _Track,
        det: _Det,
        anchors: Sequence[_Anchor],
        stamp: float,
        working_area: Sequence[tuple[float, float]] | None,
    ) -> BucketObservation:
        track.points.append(TrailPoint(stamp, det.xyxy))
        previous = track.parent_track_id
        parent, score, ambiguous = _choose(
            det.score,
            det.xyxy,
            anchors,
            previous,
            ambiguity_margin=self.ambiguity_margin,
            max_normalized_distance=self.max_normalized_distance,
            working_area=working_area,
        )
        if anchors:
            track.parent_track_id = None if ambiguous or parent is None else parent
        state = (
            ObservationState.ambiguous
            if ambiguous
            else self._quality(det.score)
        )
        reason = ObservationReason.ambiguous_association if ambiguous else None
        return self._publish(
            track,
            index=det.index,
            parent_track_id=parent,
            association_state=state,
            association_score=None if ambiguous or parent is None else score,
            xyxy=det.xyxy,
            is_observed=True,
            reason=reason,
        )

    def _miss(self, track: _Track, stamp: float) -> BucketObservation:
        predicted = _extrapolate(track.points, stamp, limit_s=self.extrapolation_s)
        if predicted is None:
            return self._publish(
                track,
                index=None,
                parent_track_id=None,
                association_state=ObservationState.missing,
                association_score=None,
                xyxy=None,
                is_observed=False,
                reason=ObservationReason.no_detection,
            )
        # The dash is display-only. It is not a trail point and not a parent choice.
        return self._publish(
            track,
            index=None,
            parent_track_id=None,
            association_state=ObservationState.predicted,
            association_score=None,
            xyxy=predicted,
            is_observed=False,
            reason=ObservationReason.no_detection,
        )

    def _publish(
        self,
        track: _Track,
        *,
        index: int | None,
        parent_track_id: str | None,
        association_state: ObservationState,
        association_score: float | None,
        xyxy: tuple[float, float, float, float] | None,
        is_observed: bool,
        reason: ObservationReason | None,
    ) -> BucketObservation:
        last = track.points[-1].source_time_s if track.points else None
        return BucketObservation(
            bucket_observation_id=track.observation_id,
            index=index,
            parent_track_id=parent_track_id,
            association_state=association_state,
            association_score=association_score,
            xyxy=xyxy,
            is_observed=is_observed,
            reason=reason,
            segments=_segments(track.points, self.trail_gap_s),
            last_observed_time_s=last,
        )

    def _quality(self, score: float) -> ObservationState:
        if score >= self.display_threshold:
            return ObservationState.observed
        return ObservationState.weak

    def _open_track(self, run_id: str, video_id: str, view_id: str | None) -> _Track:
        local_id = self._next_local_id
        self._next_local_id += 1
        return _Track(
            local_id=local_id,
            observation_id=_bucket_observation_id(run_id, video_id, view_id, local_id),
            points=[],
            parent_track_id=None,
        )

    def _prune(self, stamp: float) -> None:
        alive: list[_Track] = []
        for track in self._tracks:
            track.points = [
                point
                for point in track.points
                if stamp - point.source_time_s <= self.trail_s + _TIME_EPS
            ]
            if track.points:
                alive.append(track)
        self._tracks = alive


def _choose(
    detection_score: float,
    bucket_xyxy: tuple[float, float, float, float],
    anchors: Sequence[_Anchor],
    previous_parent: str | None,
    *,
    ambiguity_margin: float,
    max_normalized_distance: float,
    working_area: Sequence[tuple[float, float]] | None = None,
) -> tuple[str | None, float | None, bool]:
    """Return ``(parent_id, score, ambiguous)``.

    Anchors are scored, not taken in input order. A pair inside the margin
    returns no parent. Callers that pass the worse anchor first still get
    ``ambiguous`` when the second score is close. Containment is added only
    after that base decision is already clear, and the margin is checked again.
    """
    ranked: list[tuple[float, str, bool, float]] = []
    for anchor in anchors:
        distance = normalized_center_distance(bucket_xyxy, anchor.xyxy)
        if distance > max_normalized_distance:
            continue
        continued = previous_parent is not None and anchor.track_id == previous_parent
        base = pair_score(detection_score, distance, continued=continued, contained=False)
        inside = _pair_in_working_area(bucket_xyxy, anchor.xyxy, working_area)
        ranked.append((base, anchor.track_id, inside, distance))
    if not ranked:
        return None, None, False
    ranked.sort(key=lambda item: item[0], reverse=True)
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] <= ambiguity_margin:
        return None, None, True
    boosted: list[tuple[float, str]] = []
    for base, track_id, inside, distance in ranked:
        continued = previous_parent is not None and track_id == previous_parent
        score = pair_score(
            detection_score,
            distance,
            continued=continued,
            contained=inside,
        )
        boosted.append((score, track_id))
    boosted.sort(key=lambda item: item[0], reverse=True)
    if len(boosted) > 1 and boosted[0][0] - boosted[1][0] <= ambiguity_margin:
        return None, None, True
    return boosted[0][1], boosted[0][0], False


def _pair_in_working_area(
    bucket_xyxy: tuple[float, float, float, float],
    anchor_xyxy: tuple[float, float, float, float],
    working_area: Sequence[tuple[float, float]] | None,
) -> bool:
    """True when both centers lie in the pixel polygon. An empty area is false."""
    if working_area is None or len(working_area) < 3:
        return False
    bucket_center = _center(bucket_xyxy)
    anchor_center = _center(anchor_xyxy)
    return point_in_polygon(*bucket_center, working_area) and point_in_polygon(
        *anchor_center,
        working_area,
    )


def _match(
    tracks: Sequence[_Track],
    detections: Sequence[_Det],
    *,
    max_normalized_distance: float,
) -> tuple[list[tuple[_Track, _Det]], list[_Det], list[_Track]]:
    """Greedy one-to-one match on normalized center distance to the last point."""
    pairs: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        if not track.points:
            continue
        last = track.points[-1].xyxy
        for det_index, det in enumerate(detections):
            distance = normalized_center_distance(det.xyxy, last)
            if distance <= max_normalized_distance:
                pairs.append((distance, track_index, det_index))
    pairs.sort()
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    matched: list[tuple[_Track, _Det]] = []
    for _distance, track_index, det_index in pairs:
        if track_index in used_tracks or det_index in used_dets:
            continue
        used_tracks.add(track_index)
        used_dets.add(det_index)
        matched.append((tracks[track_index], detections[det_index]))
    unmatched_dets = [
        det for index, det in enumerate(detections) if index not in used_dets
    ]
    unmatched_tracks = [
        track for index, track in enumerate(tracks) if index not in used_tracks
    ]
    return matched, unmatched_dets, unmatched_tracks


def _reliable_buckets(
    detections: FrameDetections,
    *,
    detection_floor: float,
) -> tuple[_Det, ...]:
    """Bucket rows at or above the floor. The display threshold is not applied.

    A score under the floor stays in the detector list and is not a trail
    point. This function does not remove it from ``detections``.
    """
    chosen: list[_Det] = []
    for index, name in enumerate(detections.class_name):
        if name != BUCKET:
            continue
        score = float(detections.score[index])
        if not math.isfinite(score) or score < detection_floor:
            continue
        box = _tuple_box(detections.xyxy[index])
        if not _positive_box(box):
            continue
        chosen.append(_Det(index, score, box))
    return tuple(chosen)


def _anchors(bodies: Sequence[object]) -> tuple[_Anchor, ...]:
    """Copy this frame's excavator boxes. Other classes are ignored.

    The copy is local to the call. The previous frame's anchors are not
    consulted, because ``last_positions`` no longer holds them after a miss.
    """
    anchors: list[_Anchor] = []
    for body in bodies:
        class_name = getattr(body, "class_name", EXCAVATOR_BODY)
        if class_name != EXCAVATOR_BODY:
            continue
        track_id = getattr(body, "track_id", None)
        xyxy = getattr(body, "xyxy", None)
        if not isinstance(track_id, str) or not track_id or xyxy is None:
            continue
        box = _tuple_box(xyxy)
        if not _positive_box(box):
            continue
        anchors.append(_Anchor(track_id, box))
    return tuple(anchors)


def _extrapolate(
    points: Sequence[TrailPoint],
    stamp: float,
    *,
    limit_s: float,
) -> tuple[float, float, float, float] | None:
    """Dash from the last two observed points. Not appended by the caller."""
    if len(points) < 2:
        return None
    previous = points[-2]
    last = points[-1]
    elapsed = stamp - last.source_time_s
    if elapsed <= 0.0 or elapsed > limit_s + _TIME_EPS:
        return None
    history = last.source_time_s - previous.source_time_s
    if history <= _TIME_EPS:
        return None
    start = _center(previous.xyxy)
    end = _center(last.xyxy)
    center = (
        end[0] + (end[0] - start[0]) / history * elapsed,
        end[1] + (end[1] - start[1]) / history * elapsed,
    )
    width = last.xyxy[2] - last.xyxy[0]
    height = last.xyxy[3] - last.xyxy[1]
    return _box_from_center(center, width, height)


def _segments(
    points: Sequence[TrailPoint],
    gap_s: float,
) -> tuple[tuple[TrailPoint, ...], ...]:
    """Polyline as segments. A gap strictly longer than ``gap_s`` splits it."""
    ordered = tuple(sorted(points, key=lambda point: point.source_time_s))
    if not ordered:
        return ()
    groups: list[list[TrailPoint]] = [[ordered[0]]]
    for point in ordered[1:]:
        if point.source_time_s - groups[-1][-1].source_time_s > gap_s:
            groups.append([point])
        else:
            groups[-1].append(point)
    return tuple(tuple(group) for group in groups)


def _frame_state(observations: Sequence[BucketObservation]) -> ObservationState:
    states = [item.association_state for item in observations]
    if ObservationState.ambiguous in states:
        return ObservationState.ambiguous
    if ObservationState.observed in states:
        return ObservationState.observed
    if ObservationState.weak in states:
        return ObservationState.weak
    if ObservationState.predicted in states:
        return ObservationState.predicted
    return ObservationState.missing


def _bucket_observation_id(
    run_id: str,
    video_id: str,
    view_id: str | None,
    local_id: int,
) -> str:
    """Same namespace shape as a vehicle id, with class ``bucket``."""
    if view_id is None or view_id == "":
        token = "unset"
    else:
        token = view_id
    return f"{run_id}:{video_id}:{token}:bucket:{int(local_id)}"


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _box_from_center(
    center: tuple[float, float],
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    return (
        center[0] - width / 2.0,
        center[1] - height / 2.0,
        center[0] + width / 2.0,
        center[1] + height / 2.0,
    )


def _tuple_box(xyxy: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    return (x1, y1, x2, y2)


def _positive_box(box: tuple[float, float, float, float]) -> bool:
    return all(math.isfinite(value) for value in box) and box[2] > box[0] and box[3] > box[1]


def _require_range(name: str, value: float, *, upper: float | None) -> float:
    number = float(value)
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be > 0")
    if upper is not None and number > upper:
        raise ValueError(f"{name} must be <= {upper}")
    return number


def _require_unit(name: str, value: float) -> float:
    number = float(value)
    if isinstance(value, bool) or not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _require_margin(value: float) -> float:
    number = float(value)
    if isinstance(value, bool) or not math.isfinite(number) or number < 0.0 or number >= 1.0:
        raise ValueError("ambiguity_margin must be in [0, 1)")
    return number
