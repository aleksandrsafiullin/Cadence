"""Two vehicle ByteTrack instances. Parts of a machine are not tracked here.

``haul_truck`` and ``excavator_body`` each get their own tracker. ``bucket``
and ``truck_bed`` are never submitted. Detector boxes are not replaced by a
Kalman box. Local track ids stay integers. ``TrackingSession`` namespaces
them and calls ``reset()`` on both trackers.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from supervision import Detections
from trackers import ByteTrackTracker

from excavation_lab.models.adapter import FrameDetections
from excavation_lab.schemas import ByteTrackConfig, PilotConfig, load_pilot

HAUL_TRUCK = "haul_truck"
EXCAVATOR_BODY = "excavator_body"
BUCKET = "bucket"
TRUCK_BED = "truck_bed"

# One class per tracker. The lists stay disjoint, and neither holds a part.
HAUL_TRUCK_CLASSES: frozenset[str] = frozenset({HAUL_TRUCK})
EXCAVATOR_BODY_CLASSES: frozenset[str] = frozenset({EXCAVATOR_BODY})
EXCLUDED_FROM_VEHICLE_TRACKERS: frozenset[str] = frozenset({BUCKET, TRUCK_BED})

# Installed trackers 2.6.1 names. Project YAML uses the left-hand names.
_ARGUMENT_NAMES: dict[str, str] = {
    "high_confidence": "high_conf_det_threshold",
    "activation": "track_activation_threshold",
    "min_observations": "minimum_consecutive_frames",
    "lost_buffer_s": "lost_track_buffer",
    "reference_fps": "frame_rate",
}

REFERENCE_FPS = 30
_INDEX_KEY = "detection_index"


def _installed_time_parameter() -> str | None:
    """Name of the seconds argument on ``ByteTrackTracker.update``, if any.

    The installed method takes ``timestamp``, not ``source_time_s``. A missing
    seconds argument is ``unsupported``. The frame index is never that argument.
    """
    names = inspect.signature(ByteTrackTracker.update).parameters
    if "source_time_s" in names:
        return "source_time_s"
    if "timestamp" in names:
        return "timestamp"
    return None


INSTALLED_UPDATE_TIME_PARAMETER = _installed_time_parameter()
TIMESTAMP_API = (
    INSTALLED_UPDATE_TIME_PARAMETER
    if INSTALLED_UPDATE_TIME_PARAMETER is not None
    else "unsupported"
)


def seconds_to_lost_track_buffer(seconds: float, reference_fps: float = REFERENCE_FPS) -> int:
    """Convert seconds to ``lost_track_buffer`` at the reference frame rate.

    Two seconds is 60 because the installed buffer counts 30 fps frames.
    It is not ``seconds * analysis_fps`` (that would be 10 at 5 fps).
    """
    frames = float(seconds) * float(reference_fps)
    if not math.isfinite(frames) or frames < 0:
        raise ValueError("lost_track_buffer must be a finite non-negative frame count")
    return round(frames)


def tracker_arguments(config: ByteTrackConfig) -> dict[str, Any]:
    """Constructor kwargs for one ``ByteTrackTracker``.

    ``lost_track_buffer`` uses ``reference_fps`` (30). ``frame_rate`` is that
    same reference, so the library does not scale 60 down by the analysis fps.
    """
    return {
        _ARGUMENT_NAMES["lost_buffer_s"]: seconds_to_lost_track_buffer(
            config.lost_buffer_s,
            config.reference_fps,
        ),
        _ARGUMENT_NAMES["reference_fps"]: float(config.reference_fps),
        _ARGUMENT_NAMES["activation"]: float(config.activation),
        _ARGUMENT_NAMES["min_observations"]: int(config.min_observations),
        _ARGUMENT_NAMES["high_confidence"]: float(config.high_confidence),
    }


@dataclass(frozen=True, slots=True)
class VehicleAssignment:
    """One detector row after the vehicle update.

    ``xyxy`` is the detector box. ``source_time_s`` is kept on the row even
    when ``update`` has no seconds parameter. ``local_track_id`` is a local
    integer, or ``None`` when this class was not tracked or the id is still
    unconfirmed (``-1``).
    """

    index: int
    class_name: str
    score: float
    xyxy: tuple[float, float, float, float]
    source_time_s: float
    local_track_id: int | None


class VehicleTrackers:
    """Independent ByteTrack state for the two vehicle classes."""

    def __init__(
        self,
        *,
        detection_floor: float,
        display_threshold: float,
        bytetrack: ByteTrackConfig,
        analysis_fps: float | None = None,
        tracker_cls: Callable[..., Any] = ByteTrackTracker,
    ) -> None:
        if not HAUL_TRUCK_CLASSES.isdisjoint(EXCAVATOR_BODY_CLASSES):
            raise RuntimeError("vehicle trackers must not share a class")
        overlap = (HAUL_TRUCK_CLASSES | EXCAVATOR_BODY_CLASSES) & EXCLUDED_FROM_VEHICLE_TRACKERS
        if overlap:
            raise RuntimeError(f"vehicle trackers must not include {sorted(overlap)}")
        self.detection_floor = float(detection_floor)
        # Stored for the UI later. Not applied before ``update``.
        self.display_threshold = float(display_threshold)
        # Accepted so a caller can show the analysis rate. Not the buffer scale.
        self.analysis_fps = None if analysis_fps is None else float(analysis_fps)
        self.tracker_arguments = tracker_arguments(bytetrack)
        self.haul_truck = tracker_cls(**self.tracker_arguments)
        self.excavator_body = tracker_cls(**self.tracker_arguments)

    @classmethod
    def from_pilot(
        cls,
        analysis_fps: float | None = None,
        *,
        pilot: PilotConfig | None = None,
        tracker_cls: Callable[..., Any] = ByteTrackTracker,
    ) -> VehicleTrackers:
        loaded = load_pilot() if pilot is None else pilot
        return cls(
            detection_floor=loaded.detection_floor,
            display_threshold=loaded.display_threshold,
            bytetrack=loaded.bytetrack,
            analysis_fps=analysis_fps,
            tracker_cls=tracker_cls,
        )

    def reset(self) -> None:
        """Clear both trackers. Each tracker also exposes its own ``reset()``."""
        self.haul_truck.reset()
        self.excavator_body.reset()

    def update(
        self,
        detections: FrameDetections,
        source_time_s: float,
    ) -> tuple[VehicleAssignment, ...]:
        """Track vehicle classes. Other classes are not submitted.

        A score at or above ``detection_floor`` is passed through, including
        scores under ``display_threshold``. Scores below the floor are not
        passed. Returned boxes are the detector boxes.
        """
        stamp = float(source_time_s)
        local_ids = [None] * len(detections.class_name)
        self._track_class(self.haul_truck, detections, HAUL_TRUCK, stamp, local_ids)
        self._track_class(
            self.excavator_body,
            detections,
            EXCAVATOR_BODY,
            stamp,
            local_ids,
        )
        rows: list[VehicleAssignment] = []
        for index, class_name in enumerate(detections.class_name):
            x1, y1, x2, y2 = (float(value) for value in detections.xyxy[index])
            rows.append(
                VehicleAssignment(
                    index=index,
                    class_name=str(class_name),
                    score=float(detections.score[index]),
                    xyxy=(x1, y1, x2, y2),
                    source_time_s=stamp,
                    local_track_id=local_ids[index],
                )
            )
        return tuple(rows)

    def _track_class(
        self,
        tracker: Any,
        detections: FrameDetections,
        class_name: str,
        source_time_s: float,
        local_ids: list[int | None],
    ) -> None:
        packed = _pack(detections, class_name, self.detection_floor)
        tracked = _call_update(tracker, packed, source_time_s)
        _write_local_ids(tracked, local_ids)


def _pack(detections: FrameDetections, class_name: str, detection_floor: float) -> Detections:
    """Boxes of one class whose score cleared the detection floor.

    The display threshold is not read here. Confidence is the detector score,
    not 1.0 and not the display threshold.
    """
    chosen: list[int] = []
    for index, name in enumerate(detections.class_name):
        if name != class_name:
            continue
        score = float(detections.score[index])
        if not math.isfinite(score) or score < detection_floor:
            continue
        chosen.append(index)
    if not chosen:
        return Detections.empty()
    indices = np.asarray(chosen, dtype=np.int64)
    return Detections(
        xyxy=np.array(detections.xyxy[indices], dtype=np.float64, copy=True),
        confidence=np.array(detections.score[indices], dtype=np.float64, copy=True),
        class_id=np.array(detections.class_id[indices], dtype=np.int64, copy=True),
        data={_INDEX_KEY: indices.copy()},
    )


def _call_update(tracker: Any, packed: Detections, source_time_s: float) -> Detections:
    """Pass source seconds under the installed parameter name.

    ``frame`` is omitted. The image is unused, and the frame index is not a
    substitute for seconds.
    """
    kwargs = _time_kwargs(tracker, source_time_s)
    return tracker.update(packed, **kwargs)


def _time_kwargs(tracker: Any, source_time_s: float) -> dict[str, float]:
    if TIMESTAMP_API == "unsupported":
        return {}
    parameters = inspect.signature(tracker.update).parameters
    accepts_any = any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())
    if TIMESTAMP_API in parameters or accepts_any:
        return {TIMESTAMP_API: float(source_time_s)}
    return {}


def _write_local_ids(tracked: Detections, local_ids: list[int | None]) -> None:
    tracker_ids = tracked.tracker_id
    if tracker_ids is None or len(tracked) == 0:
        return
    raw = tracked.data.get(_INDEX_KEY) if isinstance(tracked.data, Mapping) else None
    if raw is None:
        return
    for slot, tracker_id in enumerate(tracker_ids):
        origin = int(raw[slot])
        if origin < 0 or origin >= len(local_ids):
            continue
        number = int(tracker_id)
        # -1 is the library's unconfirmed id. 0 is a real local id.
        local_ids[origin] = None if number < 0 else number
