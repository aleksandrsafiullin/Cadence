"""Detector boundary. Boxes stay in original-image pixels.

``predict`` does not apply the display score threshold (0.35). Score policy
belongs to a later pass. ``close`` drops the module reference. There is no
process-wide model cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from excavation_lab.errors import LabError
from excavation_lab.models.preprocess import model_rgb_frame

# Spec §13.2. Not applied inside predict.
DISPLAY_SCORE_THRESHOLD = 0.35


class CheckpointOrigin(StrEnum):
    baseline_coco = "baseline_coco"
    custom = "custom"


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Checkpoint identity visible to the caller.

    ``boxes_are_original`` means xyxy is already in the frame passed to
    ``predict``. The letterbox inverse must not run on those boxes.
    ``class_names`` is the checkpoint list in order, not the four project classes.
    """

    origin: CheckpointOrigin
    class_names: tuple[str, ...]
    boxes_are_original: bool
    variant: str
    resolution: int | None = None


@dataclass(frozen=True, slots=True, eq=False)
class FrameDetections:
    """Detections for one frame. xyxy is original-image pixels."""

    timestamp_s: float
    frame_width: int
    frame_height: int
    origin: CheckpointOrigin
    xyxy: NDArray[np.float64]
    class_id: NDArray[np.int64]
    class_name: tuple[str, ...]
    score: NDArray[np.float64]

    def __post_init__(self) -> None:
        count = len(self.class_name)
        if self.xyxy.shape != (count, 4):
            raise ValueError("xyxy must have shape (N, 4)")
        if self.class_id.shape != (count,):
            raise ValueError("class_id must have shape (N,)")
        if self.score.shape != (count,):
            raise ValueError("score must have shape (N,)")


class DetectorAdapter(Protocol):
    def predict(
        self,
        frame_rgb: NDArray[np.uint8],
        timestamp_s: float,
    ) -> FrameDetections: ...

    def metadata(self) -> ModelMetadata: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ScriptedDetection:
    """One canned box.

    ``class_name`` is the checkpoint string when the fake output already
    named the row. ``None`` selects ``class_names[class_id]``.
    """

    xyxy: tuple[float, float, float, float]
    class_id: int
    score: float
    class_name: str | None = None


def class_name_from_checkpoint(
    class_id: int,
    class_names: Sequence[str],
    reported: str | None,
) -> str:
    """Keep the checkpoint's own name.

    A reported name is returned as-is. ``truck`` is not rewritten to a
    project class. When the raw output has no name, the name is
    ``class_names[class_id]`` — the checkpoint list in order, and only for a
    contiguous id. This is not the COCO sparse-id map; ``RFDETR.predict``
    already fills ``class_name`` for that checkpoint.
    """
    if reported is not None:
        return reported
    index = int(class_id)
    if index < 0 or index >= len(class_names):
        raise LabError(
            "unknown_checkpoint_class",
            f"В чекпоинте нет класса с индексом {index}. Имя не подменяется.",
        )
    return class_names[index]


class FakeDetector:
    """In-memory detector. It does not load weights or download them.

    ``locate_red`` returns the half-open box of pixels whose channel 0 is red
    and whose channel 2 is not. Used to check that a display rotation moves
    the box with the pixels.
    """

    def __init__(
        self,
        *,
        class_names: tuple[str, ...] = ("truck",),
        scripted: tuple[ScriptedDetection, ...] = (),
        locate_red: bool = False,
        origin: CheckpointOrigin = CheckpointOrigin.baseline_coco,
        module: object | None = None,
    ) -> None:
        if not class_names:
            raise LabError(
                "checkpoint_class_names_missing",
                "У чекпоинта нет списка классов. Имена не подменяются на классы проекта.",
            )
        self._class_names = tuple(class_names)
        self._scripted = scripted
        self._locate_red = locate_red
        self._origin = origin
        self._module: object | None = object() if module is None else module
        self._resolution = 384
        self.boundary_frames: list[NDArray[np.uint8]] = []

    def predict(
        self,
        frame_rgb: NDArray[np.uint8],
        timestamp_s: float,
    ) -> FrameDetections:
        self._require_open()
        frame = model_rgb_frame(frame_rgb)
        self.boundary_frames = [frame]
        scripted = self._scripted
        if self._locate_red:
            found = _red_xyxy(frame)
            if found is None:
                return _empty_frame(frame, timestamp_s, self._origin)
            scripted = (ScriptedDetection(found, 0, 1.0),)
        named = tuple(
            ScriptedDetection(
                item.xyxy,
                item.class_id,
                item.score,
                class_name_from_checkpoint(
                    item.class_id,
                    self._class_names,
                    item.class_name,
                ),
            )
            for item in scripted
        )
        return _frame_from_scripted(frame, timestamp_s, self._origin, named)

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            origin=self._origin,
            class_names=self._class_names,
            boxes_are_original=True,
            variant="fake",
            resolution=self._resolution,
        )

    def close(self) -> None:
        self._module = None

    def _require_open(self) -> None:
        if self._module is None:
            raise LabError(
                "detector_closed",
                "Детектор закрыт. Откройте модель заново; общий экземпляр не хранится.",
            )


def _frame_from_scripted(
    frame: NDArray[np.uint8],
    timestamp_s: float,
    origin: CheckpointOrigin,
    scripted: tuple[ScriptedDetection, ...],
) -> FrameDetections:
    names: list[str] = []
    rows: list[tuple[float, float, float, float]] = []
    class_ids: list[int] = []
    scores: list[float] = []
    for item in scripted:
        names.append(item.class_name)
        rows.append(item.xyxy)
        class_ids.append(int(item.class_id))
        scores.append(float(item.score))
    count = len(names)
    xyxy = np.zeros((count, 4), dtype=np.float64)
    if count:
        xyxy[:] = np.asarray(rows, dtype=np.float64)
    return FrameDetections(
        timestamp_s=float(timestamp_s),
        frame_width=int(frame.shape[1]),
        frame_height=int(frame.shape[0]),
        origin=origin,
        xyxy=xyxy,
        class_id=np.asarray(class_ids, dtype=np.int64),
        class_name=tuple(names),
        score=np.asarray(scores, dtype=np.float64),
    )


def _empty_frame(
    frame: NDArray[np.uint8],
    timestamp_s: float,
    origin: CheckpointOrigin,
) -> FrameDetections:
    return _frame_from_scripted(frame, timestamp_s, origin, ())


def _red_xyxy(
    frame: NDArray[np.uint8],
) -> tuple[float, float, float, float] | None:
    """Half-open box of saturated red pixels. Channel 0 is red, not channel 2."""
    mask = (frame[:, :, 0] >= 250) & (frame[:, :, 2] <= 5)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )
