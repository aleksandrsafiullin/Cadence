"""Observation states for one analyzed frame.

Scores and association output choose ``observed``, ``weak``, ``predicted``,
``missing``, or ``ambiguous``. A predicted xyxy exists only as a bucket dash
of at most 0.4 s, and only on a track-state row with ``is_observed=false``.
Detection rows stay on the detector boxes. A vehicle that ByteTrack still
holds inside its lost buffer is ``missing`` with null boxes: the Kalman
estimate is not a box and not a detection row.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from excavation_lab.models.adapter import FrameDetections
from excavation_lab.schemas import ObservationReason, ObservationState
from excavation_lab.tracking.bed import BedFrame
from excavation_lab.tracking.bucket import BucketFrame, BucketObservation
from excavation_lab.tracking.session import (
    LostBufferVehicle,
    TrackedRow,
    public_track_id,
)
from excavation_lab.tracking.vehicles import (
    BUCKET,
    EXCAVATOR_BODY,
    HAUL_TRUCK,
    TRUCK_BED,
)

MAX_PREDICTED_S = 0.4
_TIME_EPS = 1e-9
_CANONICAL = frozenset({EXCAVATOR_BODY, BUCKET, HAUL_TRUCK, TRUCK_BED})
_VEHICLES = frozenset({HAUL_TRUCK, EXCAVATOR_BODY})
_SEEN = frozenset(
    {
        ObservationState.observed,
        ObservationState.weak,
        ObservationState.ambiguous,
    }
)


def state_for_score(
    score: float,
    *,
    detection_floor: float,
    display_threshold: float,
    ambiguous: bool = False,
) -> ObservationState | None:
    """Map one detector score to a state.

    Ambiguous association wins even when the score would be ``observed``.
    ``display_threshold`` and above is ``observed``. The floor up to that
    threshold is ``weak``. A score under the floor has no reliable state;
    the raw detection row is still kept by the caller.
    """
    if ambiguous:
        return ObservationState.ambiguous
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    number = float(score)
    if not math.isfinite(number):
        return None
    if number >= display_threshold:
        return ObservationState.observed
    if number >= detection_floor:
        return ObservationState.weak
    return None


def miss_reason(
    *,
    occluded: bool = False,
    out_of_frame: bool = False,
) -> ObservationReason:
    """Reason for a detector miss.

    Occlusion and leaving the frame are recorded only when that flag was
    supplied. The default is ``no_detection``.
    """
    if occluded:
        return ObservationReason.occluded_if_known
    if out_of_frame:
        return ObservationReason.out_of_frame_if_known
    return ObservationReason.no_detection


@dataclass(frozen=True, slots=True)
class TrackStateRow:
    """One in-memory track step. ``export_tables`` writes this row.

    ``xyxy`` is an owned list. Mutating it does not change a detection row.
    It is null for ``missing``. The only non-detector box is a bucket dash.
    """

    run_id: str
    track_id: str
    class_name: str
    source_time_s: float
    state: ObservationState
    last_observed_time_s: float | None
    is_observed: bool
    xyxy: list[float] | None
    parent_track_id: str | None
    identity_uncertain: bool
    reason: ObservationReason | None

    @property
    def x1(self) -> float | None:
        return None if self.xyxy is None else float(self.xyxy[0])

    @property
    def y1(self) -> float | None:
        return None if self.xyxy is None else float(self.xyxy[1])

    @property
    def x2(self) -> float | None:
        return None if self.xyxy is None else float(self.xyxy[2])

    @property
    def y2(self) -> float | None:
        return None if self.xyxy is None else float(self.xyxy[3])


@dataclass(frozen=True, slots=True)
class DecodeGap:
    """A corrupt source span. Not an equipment-idle interval."""

    source_frame_index: int | None
    source_time_s: float | None
    reason: ObservationReason

    @property
    def is_equipment_idle(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class FrameStates:
    """States for one analyzed frame. ``detection_states`` keys are detector indexes."""

    detection_states: dict[int, ObservationState]
    track_states: tuple[TrackStateRow, ...]


class ObservationLedger:
    """Track-state rows for one run. This object does not write parquet."""

    def __init__(
        self,
        *,
        detection_floor: float,
        display_threshold: float,
        extrapolation_s: float = MAX_PREDICTED_S,
    ) -> None:
        self.detection_floor = float(detection_floor)
        self.display_threshold = float(display_threshold)
        self.extrapolation_s = _prediction_limit(extrapolation_s)
        self.track_states: list[TrackStateRow] = []
        self.decode_gaps: list[DecodeGap] = []
        self._last_observed: dict[str, float] = {}
        self._reset_count: int | None = None

    def begin(self, reset_count: int) -> None:
        """Drop rows from a previous ``run`` and forget last-seen times."""
        self.track_states.clear()
        self.decode_gaps.clear()
        self._last_observed.clear()
        self._reset_count = reset_count

    def note_decode_gap(
        self,
        *,
        source_frame_index: int | None,
        source_time_s: float | None,
        reset_count: int,
    ) -> DecodeGap:
        """Record ``decode_gap`` for a corrupt frame. No box and no idle interval."""
        self._sync_reset(reset_count)
        gap = DecodeGap(
            source_frame_index=source_frame_index,
            source_time_s=None if source_time_s is None else float(source_time_s),
            reason=ObservationReason.decode_gap,
        )
        self.decode_gaps.append(gap)
        return gap

    def observe(
        self,
        detections: FrameDetections,
        *,
        tracked_rows: Sequence[TrackedRow],
        bucket_frame: BucketFrame | None,
        bed_frame: BedFrame | None,
        lost_vehicles: Sequence[LostBufferVehicle],
        run_id: str,
        video_id: str,
        view_id: str | None,
        source_time_s: float,
        reset_count: int,
        occluded_track_ids: Collection[str] = (),
        out_of_frame_track_ids: Collection[str] = (),
    ) -> FrameStates:
        """Build this frame's states. ``detections`` is not modified."""
        self._sync_reset(reset_count)
        stamp = float(source_time_s)
        occluded = frozenset(occluded_track_ids)
        outside = frozenset(out_of_frame_track_ids)
        detection_states = _detection_states(
            detections,
            bucket_frame,
            bed_frame,
            detection_floor=self.detection_floor,
            display_threshold=self.display_threshold,
        )
        rows: list[TrackStateRow] = []
        seen_ids: set[str] = set()
        for tracked in tracked_rows:
            if tracked.track_id is None or tracked.class_name not in _VEHICLES:
                continue
            # The detector row exists. Do not also emit a null miss for this id.
            seen_ids.add(tracked.track_id)
            state = detection_states.get(tracked.index)
            if state is None:
                continue
            self._last_observed[tracked.track_id] = stamp
            rows.append(
                _row(
                    run_id=run_id,
                    track_id=tracked.track_id,
                    class_name=tracked.class_name,
                    source_time_s=stamp,
                    state=state,
                    last_observed_time_s=stamp,
                    is_observed=True,
                    xyxy=_copy_box(tracked.xyxy),
                    parent_track_id=None,
                    identity_uncertain=state is ObservationState.ambiguous,
                    reason=_reason_for(state),
                )
            )
        for lost in lost_vehicles:
            if lost.track_id in seen_ids or lost.class_name not in _VEHICLES:
                continue
            rows.append(
                self._missing(
                    run_id=run_id,
                    track_id=lost.track_id,
                    class_name=lost.class_name,
                    source_time_s=stamp,
                    parent_track_id=None,
                    occluded=lost.track_id in occluded,
                    out_of_frame=lost.track_id in outside,
                )
            )
        if bucket_frame is not None:
            for observation in bucket_frame.observations:
                rows.append(
                    self._bucket_row(
                        observation,
                        detection_states=detection_states,
                        run_id=run_id,
                        source_time_s=stamp,
                        occluded=occluded,
                        outside=outside,
                    )
                )
        if bed_frame is not None:
            for link in bed_frame.links:
                state = detection_states.get(link.index)
                if link.ambiguous or link.association_state is ObservationState.ambiguous:
                    state = ObservationState.ambiguous
                elif state is None:
                    state = link.association_state
                # The detector index changes when row order changes. local_id does not.
                track_id = public_track_id(
                    run_id,
                    video_id,
                    view_id,
                    TRUCK_BED,
                    link.local_id,
                )
                seen = state in _SEEN
                if seen:
                    self._last_observed[track_id] = stamp
                xyxy = None
                if state is not ObservationState.missing and link.xyxy is not None:
                    xyxy = _copy_box(link.xyxy)
                rows.append(
                    _row(
                        run_id=run_id,
                        track_id=track_id,
                        class_name=TRUCK_BED,
                        source_time_s=stamp,
                        state=state,
                        last_observed_time_s=stamp if seen else self._last_observed.get(track_id),
                        is_observed=seen,
                        xyxy=xyxy,
                        parent_track_id=link.parent_track_id,
                        identity_uncertain=state is ObservationState.ambiguous,
                        reason=_reason_for(state),
                    )
                )
        self.track_states.extend(rows)
        return FrameStates(detection_states=detection_states, track_states=tuple(rows))

    def _bucket_row(
        self,
        observation: BucketObservation,
        *,
        detection_states: dict[int, ObservationState],
        run_id: str,
        source_time_s: float,
        occluded: frozenset[str],
        outside: frozenset[str],
    ) -> TrackStateRow:
        track_id = observation.bucket_observation_id
        if observation.association_state is ObservationState.predicted:
            dash = _accepted_dash(observation, source_time_s, self.extrapolation_s)
            last = _prior_time(observation, self._last_observed, track_id)
            if dash is None:
                return self._missing(
                    run_id=run_id,
                    track_id=track_id,
                    class_name=BUCKET,
                    source_time_s=source_time_s,
                    parent_track_id=None,
                    occluded=track_id in occluded,
                    out_of_frame=track_id in outside,
                    last_observed_time_s=last,
                )
            return _row(
                run_id=run_id,
                track_id=track_id,
                class_name=BUCKET,
                source_time_s=source_time_s,
                state=ObservationState.predicted,
                last_observed_time_s=last,
                is_observed=False,
                xyxy=dash,
                parent_track_id=None,
                identity_uncertain=False,
                reason=ObservationReason.no_detection,
            )
        state = (
            detection_states.get(observation.index)
            if observation.index is not None
            else None
        )
        if observation.association_state is ObservationState.ambiguous:
            state = ObservationState.ambiguous
        elif state is None:
            state = observation.association_state
        if state in _SEEN and observation.is_observed and observation.xyxy is not None:
            self._last_observed[track_id] = source_time_s
            return _row(
                run_id=run_id,
                track_id=track_id,
                class_name=BUCKET,
                source_time_s=source_time_s,
                state=state,
                last_observed_time_s=source_time_s,
                is_observed=True,
                xyxy=_copy_box(observation.xyxy),
                parent_track_id=observation.parent_track_id,
                identity_uncertain=state is ObservationState.ambiguous,
                reason=_reason_for(state),
            )
        return self._missing(
            run_id=run_id,
            track_id=track_id,
            class_name=BUCKET,
            source_time_s=source_time_s,
            parent_track_id=None,
            occluded=track_id in occluded,
            out_of_frame=track_id in outside,
            last_observed_time_s=_prior_time(observation, self._last_observed, track_id),
        )

    def _missing(
        self,
        *,
        run_id: str,
        track_id: str,
        class_name: str,
        source_time_s: float,
        parent_track_id: str | None,
        occluded: bool,
        out_of_frame: bool,
        last_observed_time_s: float | None = None,
    ) -> TrackStateRow:
        if last_observed_time_s is None:
            last_observed_time_s = self._last_observed.get(track_id)
        return _row(
            run_id=run_id,
            track_id=track_id,
            class_name=class_name,
            source_time_s=source_time_s,
            state=ObservationState.missing,
            last_observed_time_s=last_observed_time_s,
            is_observed=False,
            xyxy=None,
            parent_track_id=parent_track_id,
            identity_uncertain=False,
            reason=miss_reason(occluded=occluded, out_of_frame=out_of_frame),
        )

    def _sync_reset(self, reset_count: int) -> None:
        if self._reset_count != reset_count:
            self._last_observed.clear()
            self._reset_count = reset_count


def _detection_states(
    detections: FrameDetections,
    bucket_frame: BucketFrame | None,
    bed_frame: BedFrame | None,
    *,
    detection_floor: float,
    display_threshold: float,
) -> dict[int, ObservationState]:
    ambiguous = _ambiguous_keys(bucket_frame, bed_frame)
    states: dict[int, ObservationState] = {}
    for index, name in enumerate(detections.class_name):
        if name not in _CANONICAL:
            continue
        state = state_for_score(
            float(detections.score[index]),
            detection_floor=detection_floor,
            display_threshold=display_threshold,
            ambiguous=(name, index) in ambiguous,
        )
        if state is not None:
            states[index] = state
    return states


def _ambiguous_keys(
    bucket_frame: BucketFrame | None,
    bed_frame: BedFrame | None,
) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    if bucket_frame is not None:
        for item in bucket_frame.observations:
            if item.index is None:
                continue
            if item.association_state is ObservationState.ambiguous:
                keys.add((BUCKET, item.index))
    if bed_frame is not None:
        for item in bed_frame.links:
            if item.ambiguous or item.association_state is ObservationState.ambiguous:
                keys.add((TRUCK_BED, item.index))
    return keys


def _accepted_dash(
    observation: BucketObservation,
    stamp: float,
    limit_s: float,
) -> list[float] | None:
    """Bucket dash only, and only while the gap is inside ``limit_s``."""
    if observation.association_state is not ObservationState.predicted:
        return None
    if observation.is_observed or observation.xyxy is None:
        return None
    last = observation.last_observed_time_s
    if last is None:
        return None
    elapsed = stamp - float(last)
    if elapsed <= 0.0 or elapsed > limit_s + _TIME_EPS:
        return None
    return _copy_box(observation.xyxy)


def _prior_time(
    observation: BucketObservation,
    memory: dict[str, float],
    track_id: str,
) -> float | None:
    if observation.last_observed_time_s is not None:
        return float(observation.last_observed_time_s)
    return memory.get(track_id)


def _reason_for(state: ObservationState) -> ObservationReason | None:
    if state is ObservationState.ambiguous:
        return ObservationReason.ambiguous_association
    if state is ObservationState.weak:
        return ObservationReason.low_confidence
    if state is ObservationState.predicted:
        return ObservationReason.no_detection
    if state is ObservationState.missing:
        return ObservationReason.no_detection
    return None


def _row(
    *,
    run_id: str,
    track_id: str,
    class_name: str,
    source_time_s: float,
    state: ObservationState,
    last_observed_time_s: float | None,
    is_observed: bool,
    xyxy: list[float] | None,
    parent_track_id: str | None,
    identity_uncertain: bool,
    reason: ObservationReason | None,
) -> TrackStateRow:
    return TrackStateRow(
        run_id=run_id,
        track_id=track_id,
        class_name=class_name,
        source_time_s=source_time_s,
        state=state,
        last_observed_time_s=last_observed_time_s,
        is_observed=is_observed,
        xyxy=xyxy,
        parent_track_id=parent_track_id,
        identity_uncertain=identity_uncertain,
        reason=reason,
    )


def _copy_box(xyxy: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    return [x1, y1, x2, y2]


def _prediction_limit(extrapolation_s: float) -> float:
    number = float(extrapolation_s)
    if isinstance(extrapolation_s, bool) or not math.isfinite(number) or number <= 0.0:
        raise ValueError("extrapolation_s must be > 0")
    return min(number, MAX_PREDICTED_S)
