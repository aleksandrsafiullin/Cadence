"""Namespaced track ids and the reset the bucket associator joins.

Public ids are ``run:video:view:class:local``. A missing view is the stable
token ``unset`` for the whole run. Scene change in v0.1 is a ``view_id``
change or an explicit ``reset_reason``. There is no shot detector.

``last_positions`` is replaced on every ``update`` with this frame's
confirmed vehicle boxes. A one-frame miss drops the excavator anchor.
``BucketAssociator`` keeps its own trail and clears that copy in ``reset``.
``BedTruckAssociator`` keeps the previous bed parent the same way.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from excavation_lab.models.adapter import FrameDetections
from excavation_lab.tracking.bed import BedTruckAssociator
from excavation_lab.tracking.bucket import BucketAssociator
from excavation_lab.tracking.vehicles import (
    EXCAVATOR_BODY,
    HAUL_TRUCK,
    VehicleAssignment,
    VehicleTrackers,
)

UNSET_VIEW_ID = "unset"

RESET_VIDEO_ID = "video_id"
RESET_SEEK = "seek"
RESET_ANALYSIS = "analysis_restart"
RESET_VIEW_ID = "view_id"

_VEHICLE_CLASSES = (HAUL_TRUCK, EXCAVATOR_BODY)


def view_token(view_id: str | None) -> str:
    """View component of a public track id. Null stays ``unset``."""
    if view_id is None or view_id == "":
        return UNSET_VIEW_ID
    return view_id


def public_track_id(
    run_id: str,
    video_id: str,
    view_id: str | None,
    class_name: str,
    local_track_id: int,
) -> str:
    """Namespace one local tracker id. The local number may repeat across namespaces."""
    return f"{run_id}:{video_id}:{view_token(view_id)}:{class_name}:{int(local_track_id)}"


@dataclass(frozen=True, slots=True)
class TrackedRow:
    """One detector row after the session update.

    ``xyxy`` is the detector box. ``local_track_id`` is the integer embedded
    in ``track_id``. Both are null when the tracker has not confirmed an id.
    """

    index: int
    class_name: str
    score: float
    xyxy: tuple[float, float, float, float]
    source_time_s: float
    local_track_id: int | None
    track_id: str | None


@dataclass(frozen=True, slots=True)
class LostBufferVehicle:
    """Confirmed vehicle still inside the ByteTrack lost buffer.

    There is no box. The Kalman estimate is not read and is not a predicted
    xyxy.
    """

    track_id: str
    class_name: str


@dataclass(frozen=True, slots=True)
class LastPosition:
    """Last detector box of one confirmed vehicle track.

    Cleared in place by ``TrackingSession.reset``. Not a Kalman box.
    """

    class_name: str
    local_track_id: int
    track_id: str
    xyxy: tuple[float, float, float, float]
    source_time_s: float


class TrackingSession:
    """Vehicle trackers plus the anchors bucket and bed association will share."""

    def __init__(
        self,
        *,
        run_id: str,
        video_id: str | None,
        view_id: str | None,
        vehicles: VehicleTrackers,
        bucket: BucketAssociator | None = None,
        bed: BedTruckAssociator | None = None,
        class_names: Collection[str] | None = None,
    ) -> None:
        if not run_id:
            raise ValueError("run_id is required")
        self.run_id = run_id
        self.video_id = video_id
        self.view_id = view_id
        self.vehicles = vehicles
        self.class_names = None if class_names is None else frozenset(class_names)
        self.bucket = (
            bucket
            if bucket is not None
            else BucketAssociator.from_pilot(class_names=self.class_names)
        )
        self.bed = (
            bed if bed is not None else BedTruckAssociator.from_pilot(class_names=self.class_names)
        )
        # In-place updates so a reader that kept the dict sees ``reset``.
        self.last_positions: dict[str, LastPosition] = {}
        self._last_source_time_s: float | None = None
        self._offsets: dict[tuple[str, str, str], int] = {}
        self.reset_count = 0
        self.last_reset_reason: str | None = None

    @property
    def last_source_time_s(self) -> float | None:
        return self._last_source_time_s

    def set_display_threshold(self, value: float) -> None:
        """Store a UI threshold on this run. Does not write parquet or allocate a run.

        Applying the threshold to a new analysis is a new ``enqueue_analyze``
        and therefore a new ``run_id``. This method does not start that job.
        """
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise ValueError("display_threshold must be a finite value from 0 to 1")
        self.vehicles.display_threshold = number

    def restart(self, reset_reason: str | None = None) -> None:
        """Analysis is starting again. An explicit reason is a scene reset."""
        self.reset(_clean_reason(reset_reason, RESET_ANALYSIS))

    def reset(self, reason: str) -> None:
        """Drop both vehicle trackers, last positions, and associator copies.

        ByteTrack starts its own counter at 0. The offset keeps a seek or a
        restart in the same run, video, and view from reusing that number.
        A different video or view may use the same local number; the public
        string differs. A forward pass does not change the offset.
        """
        self._bump_offsets()
        self.vehicles.reset()
        self.last_positions.clear()
        self._last_source_time_s = None
        self.reset_count += 1
        self.last_reset_reason = reason
        self.bucket.reset()
        self.bed.reset()

    def is_backward(self, source_time_s: float) -> bool:
        last = self._last_source_time_s
        if last is None:
            return False
        return float(source_time_s) < last

    def update(
        self,
        detections: FrameDetections,
        *,
        video_id: str,
        view_id: str | None,
        source_time_s: float,
    ) -> tuple[TrackedRow, ...]:
        """Track this frame. Detector boxes on ``detections`` are not replaced."""
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("video_id is required")
        stamp = float(source_time_s)
        self._prepare(video_id, view_id, stamp)
        assigned = self.vehicles.update(detections, stamp)
        self._last_source_time_s = stamp
        rows = tuple(self._publish(row) for row in assigned)
        self._remember(rows)
        return rows

    def positions_for(self, class_name: str) -> tuple[LastPosition, ...]:
        """Current anchors for one class. Empty after ``reset``."""
        return tuple(
            item for item in self.last_positions.values() if item.class_name == class_name
        )

    def lost_buffer_vehicles(self) -> tuple[LostBufferVehicle, ...]:
        """Confirmed vehicles missed this frame and still inside the lost buffer.

        ``tracks`` is internal tracker state. ``get_state_bbox`` and
        ``tracked_objects`` are not read, so a lost vehicle does not become
        a predicted box or a detection row.
        """
        video_id = self.video_id
        if not isinstance(video_id, str) or not video_id:
            return ()
        found: list[LostBufferVehicle] = []
        pairs = (
            (HAUL_TRUCK, self.vehicles.haul_truck),
            (EXCAVATOR_BODY, self.vehicles.excavator_body),
        )
        for class_name, tracker in pairs:
            tracklets = getattr(tracker, "tracks", None)
            if not isinstance(tracklets, list):
                continue
            for tracklet in tracklets:
                local = getattr(tracklet, "tracker_id", None)
                if isinstance(local, bool) or not isinstance(local, int) or local < 0:
                    continue
                missed = getattr(tracklet, "time_since_update", 0)
                if isinstance(missed, bool) or not isinstance(missed, int) or missed <= 0:
                    continue
                published = local + self._offsets.get(self._key(class_name), 0)
                found.append(
                    LostBufferVehicle(
                        track_id=public_track_id(
                            self.run_id,
                            video_id,
                            self.view_id,
                            class_name,
                            published,
                        ),
                        class_name=class_name,
                    )
                )
        found.sort(key=lambda item: (item.class_name, item.track_id))
        return tuple(found)

    def _prepare(self, video_id: str, view_id: str | None, source_time_s: float) -> None:
        # One reset. A new source wins over a seek on the same frame.
        if self.video_id is None:
            self.video_id = video_id
        elif video_id != self.video_id:
            self.reset(RESET_VIDEO_ID)
            self.video_id = video_id
            self.view_id = view_id
            return
        if view_id != self.view_id:
            self.reset(RESET_VIEW_ID)
            self.view_id = view_id
            return
        if self.is_backward(source_time_s):
            self.reset(RESET_SEEK)

    def _publish(self, row: VehicleAssignment) -> TrackedRow:
        local = row.local_track_id
        published: int | None = None
        track_id: str | None = None
        if local is not None:
            published = local + self._offsets.get(self._key(row.class_name), 0)
            video_id = self.video_id
            if video_id is None:
                raise RuntimeError("track id requested before a video id was bound")
            track_id = public_track_id(
                self.run_id,
                video_id,
                self.view_id,
                row.class_name,
                published,
            )
        return TrackedRow(
            index=row.index,
            class_name=row.class_name,
            score=row.score,
            xyxy=row.xyxy,
            source_time_s=row.source_time_s,
            local_track_id=published,
            track_id=track_id,
        )

    def _remember(self, rows: tuple[TrackedRow, ...]) -> None:
        fresh: dict[str, LastPosition] = {}
        for row in rows:
            if row.track_id is None or row.local_track_id is None:
                continue
            if row.class_name not in _VEHICLE_CLASSES:
                continue
            fresh[row.track_id] = LastPosition(
                class_name=row.class_name,
                local_track_id=row.local_track_id,
                track_id=row.track_id,
                xyxy=row.xyxy,
                source_time_s=row.source_time_s,
            )
        self.last_positions.clear()
        self.last_positions.update(fresh)

    def _bump_offsets(self) -> None:
        pairs = (
            (HAUL_TRUCK, self.vehicles.haul_truck),
            (EXCAVATOR_BODY, self.vehicles.excavator_body),
        )
        for class_name, tracker in pairs:
            key = self._key(class_name)
            self._offsets[key] = self._offsets.get(key, 0) + _allocated(tracker)

    def _key(self, class_name: str) -> tuple[str, str, str]:
        video = "" if self.video_id is None else self.video_id
        return (video, view_token(self.view_id), class_name)


def _allocated(tracker: Any) -> int:
    """How many local ids this tracker has issued since it was last reset."""
    value = getattr(tracker, "_next_track_id", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def _clean_reason(value: str | None, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("reset_reason must be a non-empty string")
    return value.strip()
