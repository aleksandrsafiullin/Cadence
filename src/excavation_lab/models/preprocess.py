"""Frame geometry in front of RF-DETR 1.11.2.

Installed ``RFDETR.predict`` (``rfdetr/detr.py``) does not letterbox and does
not swap channels.

Color. The docstring requires RGB channel order. A NumPy array is not passed
through ``cv2.cvtColor``. A PIL image whose mode is not RGB is converted to
RGB once inside ``predict``. A negative-stride view (the usual
``[:, :, ::-1]`` BGR-to-RGB slice) is copied so ``torch.from_numpy`` can read
it; that copy does not swap channels again. This module therefore copies a
contiguous RGB frame and does not convert it a second time. ``bgr_to_rgb``
is the single swap for a buffer that is actually BGR.

Resize. ``torchvision.transforms.functional.resize`` stretches the frame to
the square ``(resolution, resolution)`` with ``antialias=False`` (bilinear,
half-pixel centers, no pad). ImageNet normalization follows:
mean ``(0.485, 0.456, 0.406)``, std ``(0.229, 0.224, 0.225)``.

Boxes. ``PostProcess._gather_and_scale_boxes`` multiplies normalized xyxy by
the original width and height (``target_sizes``). Those boxes are already in
original-image pixels. ``boxes_are_original`` is true for that path, and
:func:`inverse_boxes` must not scale them again.

:func:`inverse_boxes` still undoes an explicit letterbox — uniform ``scale``
plus left/top pad — when a caller has boxes on a padded canvas rather than
the library's original-size mapping.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from excavation_lab.errors import ColorOrderError, LabError

# What RF-DETR 1.11.2 predict actually does. Pad is zero: stretch, not letterbox.
INFERENCE_RESIZE = "stretch_square"
INFERENCE_PAD_X = 0.0
INFERENCE_PAD_Y = 0.0
INFERENCE_ANTIALIAS = False
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_INVALID_FRAME = (
    "Кадр для детектора должен быть массивом RGB uint8 размера H×W×3."
)


def model_rgb_frame(frame_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Copy an HWC uint8 RGB frame. Channels are not swapped.

    A negative stride on the channel axis is rejected. That view is the usual
    silent BGR↔RGB reversal. Convert it once with :func:`bgr_to_rgb`, then
    pass the contiguous result here.
    """
    _require_hwc_uint8(frame_rgb)
    if int(frame_rgb.strides[2]) < 0:
        raise ColorOrderError()
    copied = np.array(frame_rgb, dtype=np.uint8, copy=True)
    return np.ascontiguousarray(copied)


def bgr_to_rgb(frame_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Swap BGR to RGB once. Do not call this on a frame that is already RGB."""
    _require_hwc_uint8(frame_bgr)
    return np.ascontiguousarray(frame_bgr[:, :, ::-1])


def inverse_boxes(
    boxes_xyxy: np.ndarray,
    *,
    scale: float,
    pad_x: float,
    pad_y: float,
    boxes_are_original: bool,
) -> NDArray[np.float64]:
    """Map letterboxed xyxy back to the original frame.

    Letterbox places the original frame on a canvas with a uniform ``scale``
    and a left/top margin ``(pad_x, pad_y)``:

    ``x_canvas = x_original * scale + pad_x``
    ``y_canvas = y_original * scale + pad_y``

    The inverse subtracts the pad and divides by ``scale``. When
    ``boxes_are_original`` is true the library (or the caller) already
    returned original-image pixels, and this function returns a copy
    unchanged. ``scale`` and the pads are ignored on that branch.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape (N, 4)")
    if boxes_are_original:
        return np.array(boxes, dtype=np.float64, copy=True)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("scale must be a positive number")
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError("scale must be positive when boxes sit on a letterbox")
    if not math.isfinite(float(pad_x)) or not math.isfinite(float(pad_y)):
        raise ValueError("letterbox pad must be finite")
    restored = np.array(boxes, dtype=np.float64, copy=True)
    restored[:, [0, 2]] -= float(pad_x)
    restored[:, [1, 3]] -= float(pad_y)
    restored /= float(scale)
    return restored


def _require_hwc_uint8(frame: np.ndarray) -> None:
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] < 1
        or frame.shape[1] < 1
    ):
        raise LabError("invalid_frame", _INVALID_FRAME)
