"""Bind each ``truck_bed`` to at most one ``haul_truck``.

The score is bed containment (intersection / bed area), center distance
divided by the truck's longer side, and a bonus when that same pair was
chosen on the previous frame. Pairs under ``bed.min_containment`` are
dropped. IoU is not a term: a large truck can contain the bed and still
lose an IoU contest to a tighter partial overlap.

One bed has one parent. Two beds may share a truck. A parent gap inside
``bed.ambiguity_margin``, or two beds on one truck whose scores are that
close, stores ``parent_track_id`` null and ``ambiguous``. The previous
parent is a bonus, not a lock. This object keeps that parent itself.
``last_positions`` does not survive a one-frame truck miss, and a missing
truck is not rebuilt. A bed box is not a dump and not 3D contact.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from excavation_lab.models.adapter import FrameDetections
from excavation_lab.schemas import (
    ObservationReason,
    ObservationState,
    PilotConfig,
    load_pilot,
)
from excavation_lab.tracking.vehicles import HAUL_TRUCK, TRUCK_BED

# Larger than the pilot margin (0.1), so an exact geometric tie can keep the
# previous parent. A gap that is still inside the margin stays null.
CONTINUITY_BONUS = 0.15
# Match a bed to its own previous box in bed-lengths. Not a parent gate.
_MAX_BED_MATCH_DISTANCE = 2.0


def bed_containment(
    bed_xyxy: Sequence[float],
    truck_xyxy: Sequence[float],
) -> float:
    """Fraction of the bed box that lies inside the truck box.

    This is intersection / bed area, not IoU. A non-positive bed area is 0.
    """
    bed = _tuple_box(bed_xyxy)
    truck = _tuple_box(truck_xyxy)
    area = _area(bed)
    if area <= 0.0:
        return 0.0
    return _intersection(bed, truck) / area


def normalized_center_distance(
    bed_xyxy: Sequence[float],
    truck_xyxy: Sequence[float],
) -> float:
    """Center gap divided by ``max(truck_width, truck_height)``.

    A non-positive truck size is infinite, so that truck cannot win.
    """
    bed = _tuple_box(bed_xyxy)
    truck = _tuple_box(truck_xyxy)
    scale = max(truck[2] - truck[0], truck[3] - truck[1])
    if not math.isfinite(scale) or scale <= 0.0:
        return math.inf
    bed_center = _center(bed)
    truck_center = _center(truck)
    return math.hypot(bed_center[0] - truck_center[0], bed_center[1] - truck_center[1]) / scale


def pair_score(
    containment: float,
    normalized_distance: float,
    *,
    continued: bool,
) -> float:
    """Rank one bed against one truck.

    Proximity is ``1 / (1 + normalized_distance)`` and multiplies containment.
    Continuity adds ``CONTINUITY_BONUS`` when this truck was the parent last
    time. IoU is not an input.
    """
    proximity = 1.0 / (1.0 + max(float(normalized_distance), 0.0))
    bonus = CONTINUITY_BONUS if continued else 0.0
    return float(containment) * proximity + bonus


@dataclass(frozen=True, slots=True)
class BedLink:
    """One bed detection after association.

    ``local_id`` stays on the matched bed. It is not the detector index.
    ``xyxy`` is the detector box, or null when this link is ``missing``.
    ``ambiguous`` is the tie flag. There is no event id, dump flag, or
    truck visit id.
    """

    index: int
    local_id: int
    parent_track_id: str | None
    ambiguous: bool
    association_state: ObservationState
    association_score: float | None
    xyxy: tuple[float, float, float, float] | None
    reason: ObservationReason | None


@dataclass(frozen=True, slots=True)
class BedFrame:
    """Associator output for one source timestamp. Links follow detector rows."""

    source_time_s: float
    links: tuple[BedLink, ...]


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
    """Stable local id, last bed box, and the last committed parent.

    ``reset`` drops the object. It does not rewind ``local_id``.
    """

    local_id: int
    xyxy: tuple[float, float, float, float]
    parent_track_id: str | None


@dataclass
class _Decision:
    track: _Track
    det: _Det
    parent_track_id: str | None
    score: float | None
    ambiguous: bool
    had_anchors: bool


class BedTruckAssociator:
    """Attach ``truck_bed`` rows to ``haul_truck`` tracks confirmed this frame.

    The session replaces ``last_positions`` on every frame, including a
    one-frame miss. This object keeps the previous parent itself.
    ``TrackingSession.reset`` calls ``reset`` so that copy does not survive
    a seek, a new video, or an analysis restart.
    """

    def __init__(
        self,
        *,
        min_containment: float,
        ambiguity_margin: float,
        detection_floor: float,
        display_threshold: float,
    ) -> None:
        self.min_containment = _require_containment(min_containment)
        self.ambiguity_margin = _require_margin(ambiguity_margin)
        self.detection_floor = _require_unit("detection_floor", detection_floor)
        self.display_threshold = _require_unit("display_threshold", display_threshold)
        self._tracks: list[_Track] = []
        self._next_local_id = 0
        self.last_frame: BedFrame | None = None

    @classmethod
    def from_pilot(cls, pilot: PilotConfig | None = None) -> BedTruckAssociator:
        loaded = load_pilot() if pilot is None else pilot
        return cls(
            min_containment=loaded.bed.min_containment,
            ambiguity_margin=loaded.bed.ambiguity_margin,
            detection_floor=loaded.detection_floor,
            display_threshold=loaded.display_threshold,
        )

    def reset(self) -> None:
        """Drop the copied parents and the last frame.

        The local id counter is not rewound, so a later bed in the same
        run does not reuse an id from before the seek.
        """
        self._tracks.clear()
        self.last_frame = None

    def update(
        self,
        detections: FrameDetections,
        *,
        trucks: Sequence[object],
        source_time_s: float,
        run_id: str,
        video_id: str,
        view_id: str | None,
    ) -> BedFrame:
        """Associate this frame. ``detections`` is not modified.

        ``trucks`` are the haul trucks confirmed on this frame. A truck that
        is only inside ByteTrack's lost buffer is not in that sequence and
        is not turned into a box. ``view_id`` is unused: the parent id is
        already namespaced by the session.
        """
        del view_id
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id is required")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("video_id is required")
        stamp = float(source_time_s)
        anchors = _anchors(trucks)
        beds = _bed_detections(detections, detection_floor=self.detection_floor)
        matched, unmatched_dets, _unmatched = _match(self._tracks, beds)
        decisions: list[_Decision] = []
        for track, det in matched:
            decisions.append(self._decide(track, det, anchors))
        for det in unmatched_dets:
            decisions.append(self._decide(self._open_track(det.xyxy), det, anchors))
        _withhold_shared_ties(decisions, self.ambiguity_margin)
        links: list[BedLink] = []
        alive: list[_Track] = []
        for decision in decisions:
            self._remember(decision)
            decision.track.xyxy = decision.det.xyxy
            alive.append(decision.track)
            links.append(self._publish(decision))
        self._tracks = alive
        links.sort(key=lambda item: item.index)
        frame = BedFrame(source_time_s=stamp, links=tuple(links))
        self.last_frame = frame
        return frame

    def _open_track(self, xyxy: tuple[float, float, float, float]) -> _Track:
        local_id = self._next_local_id
        self._next_local_id += 1
        return _Track(local_id=local_id, xyxy=xyxy, parent_track_id=None)

    def _decide(
        self,
        track: _Track,
        det: _Det,
        anchors: Sequence[_Anchor],
    ) -> _Decision:
        parent, score, ambiguous = _choose(
            det.xyxy,
            anchors,
            track.parent_track_id,
            min_containment=self.min_containment,
            ambiguity_margin=self.ambiguity_margin,
        )
        return _Decision(
            track=track,
            det=det,
            parent_track_id=parent,
            score=score,
            ambiguous=ambiguous,
            had_anchors=bool(anchors),
        )

    def _remember(self, decision: _Decision) -> None:
        track = decision.track
        if decision.ambiguous:
            track.parent_track_id = None
            return
        if decision.parent_track_id is not None:
            track.parent_track_id = decision.parent_track_id
            return
        if decision.had_anchors:
            # The trucks on this frame were scored and none won.
            track.parent_track_id = None
            return
        # No truck box this frame. Keep the copied parent. Do not invent a box.

    def _publish(self, decision: _Decision) -> BedLink:
        ambiguous = decision.ambiguous
        parent = None if ambiguous else decision.parent_track_id
        if ambiguous:
            state = ObservationState.ambiguous
            reason = ObservationReason.ambiguous_association
            score = None
        else:
            state = self._quality(decision.det.score)
            reason = None
            score = None if parent is None else decision.score
        return BedLink(
            index=decision.det.index,
            local_id=decision.track.local_id,
            parent_track_id=parent,
            ambiguous=ambiguous,
            association_state=state,
            association_score=score,
            xyxy=decision.det.xyxy,
            reason=reason,
        )

    def _quality(self, score: float) -> ObservationState:
        if score >= self.display_threshold:
            return ObservationState.observed
        return ObservationState.weak


def _choose(
    bed_xyxy: tuple[float, float, float, float],
    anchors: Sequence[_Anchor],
    previous_parent: str | None,
    *,
    min_containment: float,
    ambiguity_margin: float,
) -> tuple[str | None, float | None, bool]:
    """Return ``(parent_id, score, ambiguous)``.

    Anchors are scored, not taken in input order and not by which box sits
    further right. A pair inside the margin returns no parent.
    """
    ranked: list[tuple[float, str]] = []
    for anchor in anchors:
        containment = bed_containment(bed_xyxy, anchor.xyxy)
        if containment < min_containment:
            continue
        distance = normalized_center_distance(bed_xyxy, anchor.xyxy)
        if not math.isfinite(distance):
            continue
        continued = previous_parent is not None and anchor.track_id == previous_parent
        ranked.append(
            (
                pair_score(containment, distance, continued=continued),
                anchor.track_id,
            )
        )
    if not ranked:
        return None, None, False
    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score, best_id = ranked[0]
    if len(ranked) > 1 and best_score - ranked[1][0] <= ambiguity_margin:
        return None, None, True
    return best_id, best_score, False


def _withhold_shared_ties(decisions: list[_Decision], ambiguity_margin: float) -> None:
    """Null every bed on a truck when those bed scores are not separable.

    A separable gap does not drop either bed. One truck may keep both.
    A one-to-one pass that steals the truck from the clearer bed is not
    applied here.
    """
    groups: dict[str, list[_Decision]] = {}
    for decision in decisions:
        parent = decision.parent_track_id
        if decision.ambiguous or parent is None:
            continue
        groups.setdefault(parent, []).append(decision)
    for group in groups.values():
        if len(group) < 2:
            continue
        ranked = sorted(group, key=lambda item: item.score or 0.0, reverse=True)
        best = ranked[0].score or 0.0
        second = ranked[1].score or 0.0
        if best - second > ambiguity_margin:
            continue
        for decision in group:
            decision.parent_track_id = None
            decision.score = None
            decision.ambiguous = True


def _match(
    tracks: Sequence[_Track],
    detections: Sequence[_Det],
) -> tuple[list[tuple[_Track, _Det]], list[_Det], list[_Track]]:
    """Greedy one-to-one match on normalized distance to the previous bed box."""
    pairs: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        for det_index, det in enumerate(detections):
            distance = normalized_center_distance(det.xyxy, track.xyxy)
            if distance <= _MAX_BED_MATCH_DISTANCE:
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
    unmatched_dets = [det for index, det in enumerate(detections) if index not in used_dets]
    unmatched_tracks = [track for index, track in enumerate(tracks) if index not in used_tracks]
    return matched, unmatched_dets, unmatched_tracks


def _bed_detections(
    detections: FrameDetections,
    *,
    detection_floor: float,
) -> tuple[_Det, ...]:
    """Bed rows at or above the floor. The display threshold is not applied.

    A score under the floor stays in the detector list. This function does
    not remove it from ``detections``.
    """
    chosen: list[_Det] = []
    for index, name in enumerate(detections.class_name):
        if name != TRUCK_BED:
            continue
        score = float(detections.score[index])
        if not math.isfinite(score) or score < detection_floor:
            continue
        box = _tuple_box(detections.xyxy[index])
        if not _positive_box(box):
            continue
        chosen.append(_Det(index, score, box))
    return tuple(chosen)


def _anchors(trucks: Sequence[object]) -> tuple[_Anchor, ...]:
    """Copy this frame's haul-truck boxes. Other classes are ignored.

    The copy is local to the call. The previous frame's anchors are not
    consulted, because ``last_positions`` no longer holds them after a miss.
    """
    anchors: list[_Anchor] = []
    for truck in trucks:
        class_name = getattr(truck, "class_name", HAUL_TRUCK)
        if class_name != HAUL_TRUCK:
            continue
        track_id = getattr(truck, "track_id", None)
        xyxy = getattr(truck, "xyxy", None)
        if not isinstance(track_id, str) or not track_id or xyxy is None:
            continue
        box = _tuple_box(xyxy)
        if not _positive_box(box):
            continue
        anchors.append(_Anchor(track_id, box))
    return tuple(anchors)


def _intersection(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = min(left[2], right[2]) - max(left[0], right[0])
    height = min(left[3], right[3]) - max(left[1], right[1])
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return width * height


def _area(box: tuple[float, float, float, float]) -> float:
    width = box[2] - box[0]
    height = box[3] - box[1]
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return width * height


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _tuple_box(xyxy: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    return (x1, y1, x2, y2)


def _positive_box(box: tuple[float, float, float, float]) -> bool:
    return all(math.isfinite(value) for value in box) and box[2] > box[0] and box[3] > box[1]


def _require_containment(value: float) -> float:
    number = float(value)
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0.0 or number > 1.0:
        raise ValueError("min_containment must be in (0, 1]")
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
