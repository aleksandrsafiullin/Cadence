"""Cross-class boxes stay. Same-class NMS is off unless a threshold is set.

A ``truck_bed`` inside a ``haul_truck`` is a real detection. A ``bucket``
stays a bucket when it overlaps ``excavator_body``. Containment is not a
reason to drop the inner class or to rewrite ``class_name``.

Same-class NMS compares boxes that share ``class_name`` and only when
``same_class_nms_iou`` is a finite IoU in ``(0, 1]``. ``configs/pilot.yaml``
stores null for that field in v0.1, so duplicate boxes of one class are
kept. This module does not choose a threshold.
"""

from __future__ import annotations

import math

import numpy as np

from excavation_lab.models.adapter import FrameDetections

_BAD_IOU = "same_class_nms_iou must be in (0, 1] or null."


def suppress(
    detections: FrameDetections,
    *,
    same_class_nms_iou: float | None = None,
) -> FrameDetections:
    """Keep every cross-class row. Optionally drop a same-class duplicate.

    Returned ``class_name`` and ``xyxy`` are the detector values. A bed
    fully inside a truck is kept even when IoU is high. Two boxes of one
    class are both kept when ``same_class_nms_iou`` is null. When it is
    set, the lower score of an overlapping same-class pair is omitted and
    the surviving box is not moved or renamed.
    """
    threshold = _optional_iou(same_class_nms_iou)
    count = len(detections.class_name)
    if threshold is None or count < 2:
        return detections
    kept = _kept_indices(detections, threshold)
    if len(kept) == count:
        return detections
    return _take(detections, kept)


def _optional_iou(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(_BAD_IOU)
    number = float(value)
    if not math.isfinite(number) or number <= 0.0 or number > 1.0:
        raise ValueError(_BAD_IOU)
    return number


def _kept_indices(detections: FrameDetections, threshold: float) -> list[int]:
    """Indices in detector order. Different ``class_name`` values are never compared."""
    by_class: dict[str, list[int]] = {}
    for index, name in enumerate(detections.class_name):
        by_class.setdefault(name, []).append(index)
    dropped: set[int] = set()
    xyxy = detections.xyxy
    scores = detections.score
    for indices in by_class.values():
        if len(indices) < 2:
            continue
        order = sorted(indices, key=lambda index: (-float(scores[index]), index))
        kept_in_class: list[int] = []
        for index in order:
            box = _box(xyxy, index)
            overlapped = any(
                _iou(box, _box(xyxy, other)) >= threshold for other in kept_in_class
            )
            if overlapped:
                dropped.add(index)
                continue
            kept_in_class.append(index)
    return [index for index in range(len(detections.class_name)) if index not in dropped]


def _take(detections: FrameDetections, indices: list[int]) -> FrameDetections:
    chosen = np.asarray(indices, dtype=np.int64)
    return FrameDetections(
        timestamp_s=detections.timestamp_s,
        frame_width=detections.frame_width,
        frame_height=detections.frame_height,
        origin=detections.origin,
        xyxy=np.array(detections.xyxy[chosen], dtype=np.float64, copy=True),
        class_id=np.array(detections.class_id[chosen], dtype=np.int64, copy=True),
        class_name=tuple(detections.class_name[index] for index in indices),
        score=np.array(detections.score[chosen], dtype=np.float64, copy=True),
    )


def _box(xyxy: np.ndarray, index: int) -> tuple[float, float, float, float]:
    row = xyxy[index]
    return (float(row[0]), float(row[1]), float(row[2]), float(row[3]))


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Intersection over union. Containment alone does not yield 1."""
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    width = ix2 - ix1
    height = iy2 - iy1
    if width <= 0.0 or height <= 0.0:
        return 0.0
    intersection = width * height
    left_area = max(left[2] - left[0], 0.0) * max(left[3] - left[1], 0.0)
    right_area = max(right[2] - right[0], 0.0) * max(right[3] - right[1], 0.0)
    union = left_area + right_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union
