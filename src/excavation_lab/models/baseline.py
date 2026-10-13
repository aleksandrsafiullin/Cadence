"""Official RF-DETR weights, behind the detector adapter.

``open_detector(origin="custom")`` without a weights file raises
``CheckpointMissingError`` and does not construct this model. A custom
directory is loaded by :func:`excavation_lab.models.bundle.load_bundle`,
which this function does not call. Unit tests inject a module or use
``FakeDetector``; they do not call :func:`_load_rfdetr_nano`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from excavation_lab.errors import CheckpointMissingError, LabError
from excavation_lab.models.adapter import (
    CheckpointOrigin,
    DetectorAdapter,
    FrameDetections,
    ModelMetadata,
    class_name_from_checkpoint,
)
from excavation_lab.models.preprocess import model_rgb_frame

# Library default is 0.5. Display threshold is 0.35. Neither is applied here.
LIBRARY_SCORE_THRESHOLD = 0.0


class BaselineCoco:
    """``baseline_coco``. Class names stay the checkpoint names, in order."""

    def __init__(self, model: Any) -> None:
        names = _read_class_names(model)
        self._model: Any | None = model
        self._class_names = names
        self._variant = str(getattr(model, "size", None) or "rfdetr-nano")
        self._resolution = _read_resolution(model)
        self.boundary_frames: list[NDArray[np.uint8]] = []

    def predict(
        self,
        frame_rgb: NDArray[np.uint8],
        timestamp_s: float,
    ) -> FrameDetections:
        model = self._model
        if model is None:
            raise LabError(
                "detector_closed",
                "Детектор закрыт. Откройте модель заново; общий экземпляр не хранится.",
            )
        frame = model_rgb_frame(frame_rgb)
        self.boundary_frames = [frame]
        raw = model.predict(
            frame,
            threshold=LIBRARY_SCORE_THRESHOLD,
            include_source_image=False,
        )
        return _frame_from_raw(
            raw,
            frame=frame,
            timestamp_s=timestamp_s,
            origin=CheckpointOrigin.baseline_coco,
            class_names=self._class_names,
        )

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            origin=CheckpointOrigin.baseline_coco,
            class_names=self._class_names,
            boxes_are_original=True,
            variant=self._variant,
            resolution=self._resolution,
        )

    def close(self) -> None:
        self._model = None


def open_detector(
    origin: str,
    weights_path: str | Path | None = None,
    *,
    model: Any | None = None,
) -> DetectorAdapter:
    """Open ``baseline_coco`` or refuse ``custom`` when weights are missing.

    Official weights load only through :func:`_load_rfdetr_nano`. An injected
    ``model`` is for tests and does not touch the network.
    """
    if origin == CheckpointOrigin.custom:
        _require_custom_weights(weights_path)
        raise LabError(
            "custom_bundle_unavailable",
            "Загрузчик custom bundle в этой версии не подключён. "
            "Базовая COCO-модель не используется.",
        )
    if origin != CheckpointOrigin.baseline_coco:
        raise LabError(
            "unknown_origin",
            "Неизвестный режим модели. Допустимы baseline_coco и custom.",
        )
    module = model if model is not None else _load_rfdetr_nano()
    return BaselineCoco(module)


def _load_rfdetr_nano() -> Any:
    """Construct ``RFDETRNano``. This downloads official weights on first use."""
    from rfdetr import RFDETRNano

    return RFDETRNano()


def _require_custom_weights(weights_path: str | Path | None) -> None:
    if weights_path is None or isinstance(weights_path, bool):
        raise CheckpointMissingError()
    if not isinstance(weights_path, (str, Path)):
        raise CheckpointMissingError()
    text = str(weights_path).strip()
    if text == "":
        raise CheckpointMissingError()
    path = Path(text)
    if not path.is_file():
        raise CheckpointMissingError(str(path))


def _read_class_names(model: Any) -> tuple[str, ...]:
    names = getattr(model, "class_names", None)
    if not names:
        raise LabError(
            "checkpoint_class_names_missing",
            "У чекпоинта нет списка классов. Имена не подменяются на классы проекта.",
        )
    return tuple(str(name) for name in names)


def _read_resolution(model: Any) -> int | None:
    direct = getattr(model, "resolution", None)
    if isinstance(direct, int) and not isinstance(direct, bool):
        return direct
    inner = getattr(model, "model", None)
    inner_resolution = getattr(inner, "resolution", None) if inner is not None else None
    if isinstance(inner_resolution, int) and not isinstance(inner_resolution, bool):
        return inner_resolution
    return None


def _frame_from_raw(
    raw: Any,
    *,
    frame: NDArray[np.uint8],
    timestamp_s: float,
    origin: CheckpointOrigin,
    class_names: Sequence[str],
) -> FrameDetections:
    xyxy = np.asarray(raw.xyxy, dtype=np.float64)
    if xyxy.size == 0:
        xyxy = np.zeros((0, 4), dtype=np.float64)
    elif xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise LabError(
            "invalid_detections",
            "Детектор вернул рамки не в формате xyxy.",
        )
    count = int(xyxy.shape[0])
    if count == 0:
        class_id = np.zeros((0,), dtype=np.int64)
        score = np.zeros((0,), dtype=np.float64)
        names: tuple[str, ...] = ()
    else:
        class_id = np.asarray(raw.class_id, dtype=np.int64).reshape(count)
        score = np.asarray(_confidence(raw), dtype=np.float64).reshape(count)
        reported = _reported_names(raw, count)
        names = tuple(
            class_name_from_checkpoint(
                int(class_id[index]),
                class_names,
                None if reported is None else reported[index],
            )
            for index in range(count)
        )
    return FrameDetections(
        timestamp_s=float(timestamp_s),
        frame_width=int(frame.shape[1]),
        frame_height=int(frame.shape[0]),
        origin=origin,
        xyxy=np.array(xyxy, dtype=np.float64, copy=True),
        class_id=class_id,
        class_name=names,
        score=score,
    )


def _confidence(raw: Any) -> Any:
    if getattr(raw, "confidence", None) is not None:
        return raw.confidence
    return raw.score


def _reported_names(raw: Any, count: int) -> list[str] | None:
    """Per-box names from the library output, not the checkpoint's full list."""
    data = getattr(raw, "data", None)
    if not isinstance(data, dict) or "class_name" not in data:
        return None
    names = [str(value) for value in data["class_name"]]
    if len(names) != count:
        raise LabError(
            "invalid_detections",
            "Число имён классов не совпало с числом рамок.",
        )
    return names
