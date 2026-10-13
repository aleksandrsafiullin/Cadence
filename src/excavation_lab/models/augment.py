"""Mild train augmentation. Validation and test stay on preprocess.

v0.1 preset, not a tuned optimum. Bounds live on :data:`MILD_PRESET` and on
the resolved train payload:

- brightness gain in ±0.2
- contrast gain in ±0.2
- a crop that removes at most 10% of each axis (``scale_limit`` 0.1)
- horizontal flip with probability 0.5, which rewrites box x

Contrast is applied around the image mean, then brightness scales the
pixels: ``((pixel - mean) * (1 + contrast) + mean) * (1 + brightness)``,
rounded and clipped to uint8.

A crop window is ``(x0, y0, x1, y1)`` in pixels. Boxes are shifted into that
window and clipped. If any box would have non-positive width or height, the
crop is rejected and the image and boxes are left as they were. A box that
only crosses the edge stays, clipped to a positive area.

Installed rfdetr 1.11.2 cannot express that crop rule. ``aug_config=None``
is torchvision ``RandomHorizontalFlip(p=0.5)``. ``scale_jitter=True`` adds a
resize/crop branch whose crop is far outside ±10%. Named presets in
``rfdetr/datasets/aug_configs.py`` are not used. The train resolver sends
:func:`library_off_switch` so those defaults stay off. This module is what
applies the mild preset, and only when ``split`` is ``train``.

Accepted splits are ``train``, ``validation``, and ``test``.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image

from excavation_lab.models.preprocess import model_rgb_frame

BRIGHTNESS_LIMIT = 0.2
CONTRAST_LIMIT = 0.2
SCALE_LIMIT = 0.1
HORIZONTAL_FLIP_PROBABILITY = 0.5

TRAIN_TRANSFORMS: tuple[str, ...] = (
    "brightness_contrast",
    "scale_crop",
    "horizontal_flip",
)

MILD_PRESET: dict[str, Any] = {
    "preset": "mild",
    "brightness_limit": BRIGHTNESS_LIMIT,
    "contrast_limit": CONTRAST_LIMIT,
    "scale_limit": SCALE_LIMIT,
    "horizontal_flip_p": HORIZONTAL_FLIP_PROBABILITY,
}

_SPLITS = frozenset({"train", "validation", "test"})


def library_off_switch() -> dict[str, Any]:
    """TrainConfig values that disable the installed augmentation defaults.

    ``aug_config={}`` is the empty-dict off switch in ``make_coco_transforms``
    (rfdetr 1.11.2). ``scale_jitter=False`` keeps the direct resize.
    ``multi_scale=False`` is stored as ``MultiScale.OFF``.
    """
    return {
        "aug_config": {},
        "scale_jitter": False,
        "multi_scale": False,
    }


def _eval_body() -> dict[str, Any]:
    body = {"mode": "deterministic_preprocess"}
    body.update(library_off_switch())
    return body


EVAL_AUGMENTATION: dict[str, Any] = _eval_body()


def mild_preset() -> dict[str, Any]:
    """Copy of the v0.1 train preset."""
    return {
        "preset": MILD_PRESET["preset"],
        "brightness_limit": MILD_PRESET["brightness_limit"],
        "contrast_limit": MILD_PRESET["contrast_limit"],
        "scale_limit": MILD_PRESET["scale_limit"],
        "horizontal_flip_p": MILD_PRESET["horizontal_flip_p"],
    }


def eval_augmentation() -> dict[str, Any]:
    """Copy of the deterministic eval payload, including the library off switch."""
    return _eval_body()


@dataclass(frozen=True, slots=True)
class AugmentedSample:
    """Image and boxes after :func:`apply`."""

    image: NDArray[np.uint8]
    boxes: NDArray[np.float64]
    horizontal_flip_applied: bool
    crop_applied: bool


def apply(
    image: NDArray[np.uint8],
    boxes: ArrayLike,
    split: str,
    *,
    seed: int | None = None,
    flip_horizontal: bool | None = None,
    crop_window: tuple[int, int, int, int] | None = None,
    brightness: float | None = None,
    contrast: float | None = None,
) -> AugmentedSample:
    """Augment ``train``. ``validation`` and ``test`` only preprocess.

    Eval ignores ``seed`` and the train overrides. The horizontal flip
    function is not called on those splits.
    """
    if split not in _SPLITS:
        raise ValueError("split must be train, validation, or test")
    frame, owned = _prepare(image, boxes)
    if split != "train":
        return AugmentedSample(
            image=frame,
            boxes=owned,
            horizontal_flip_applied=False,
            crop_applied=False,
        )

    rng = np.random.default_rng(seed)
    brightness_gain = (
        _gain(brightness, BRIGHTNESS_LIMIT, "brightness")
        if brightness is not None
        else float(rng.uniform(-BRIGHTNESS_LIMIT, BRIGHTNESS_LIMIT))
    )
    contrast_gain = (
        _gain(contrast, CONTRAST_LIMIT, "contrast")
        if contrast is not None
        else float(rng.uniform(-CONTRAST_LIMIT, CONTRAST_LIMIT))
    )
    window = crop_window if crop_window is not None else _sample_crop_window(
        int(frame.shape[0]),
        int(frame.shape[1]),
        rng,
    )
    do_flip = (
        flip_horizontal
        if flip_horizontal is not None
        else bool(rng.random() < HORIZONTAL_FLIP_PROBABILITY)
    )

    adjusted = brightness_contrast(
        frame,
        brightness=brightness_gain,
        contrast=contrast_gain,
    )
    cropped, cropped_boxes, crop_applied = crop_to_window(adjusted, owned, window)
    if do_flip:
        flipped, flipped_boxes = horizontal_flip(cropped, cropped_boxes)
        return AugmentedSample(
            image=flipped,
            boxes=flipped_boxes,
            horizontal_flip_applied=True,
            crop_applied=crop_applied,
        )
    return AugmentedSample(
        image=cropped,
        boxes=cropped_boxes,
        horizontal_flip_applied=False,
        crop_applied=crop_applied,
    )


def brightness_contrast(
    image: NDArray[np.uint8],
    *,
    brightness: float,
    contrast: float,
) -> NDArray[np.uint8]:
    """Apply one brightness gain and one contrast gain inside the v0.1 limits."""
    brightness_gain = _gain(brightness, BRIGHTNESS_LIMIT, "brightness")
    contrast_gain = _gain(contrast, CONTRAST_LIMIT, "contrast")
    if brightness_gain == 0.0 and contrast_gain == 0.0:
        return np.array(image, dtype=np.uint8, copy=True)
    pixels = image.astype(np.float64, copy=True)
    mean = float(pixels.mean())
    pixels -= mean
    pixels *= 1.0 + contrast_gain
    pixels += mean
    pixels *= 1.0 + brightness_gain
    np.clip(pixels, 0.0, 255.0, out=pixels)
    return np.rint(pixels).astype(np.uint8)


def crop_to_window(
    image: NDArray[np.uint8],
    boxes: ArrayLike,
    window: tuple[int, int, int, int],
) -> tuple[NDArray[np.uint8], NDArray[np.float64], bool]:
    """Crop to ``window``. Reject the crop when a box would disappear.

    A box that crosses the window is clipped to the intersection. That
    intersection has to stay strictly positive. Otherwise the original image
    and boxes are returned and the third value is false.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("crop expects an H×W×3 image")
    height = int(image.shape[0])
    width = int(image.shape[1])
    owned = _as_boxes(boxes)
    x0, y0, x1, y1 = _check_window(height, width, window)
    if owned.shape[0] == 0:
        cropped = np.ascontiguousarray(image[y0:y1, x0:x1])
        return cropped, np.zeros((0, 4), dtype=np.float64), True

    shifted = np.array(owned, dtype=np.float64, copy=True)
    shifted[:, [0, 2]] -= x0
    shifted[:, [1, 3]] -= y0
    crop_width = float(x1 - x0)
    crop_height = float(y1 - y0)
    shifted[:, [0, 2]] = np.clip(shifted[:, [0, 2]], 0.0, crop_width)
    shifted[:, [1, 3]] = np.clip(shifted[:, [1, 3]], 0.0, crop_height)
    # A zero-area box was outside the window or only touched its edge.
    # Keep every box by refusing the crop instead of dropping one of them.
    if np.any(shifted[:, 2] <= shifted[:, 0]) or np.any(shifted[:, 3] <= shifted[:, 1]):
        return (
            np.array(image, dtype=image.dtype, copy=True),
            np.array(owned, dtype=np.float64, copy=True),
            False,
        )
    cropped = np.ascontiguousarray(image[y0:y1, x0:x1])
    return cropped, shifted, True


def horizontal_flip(
    image: NDArray[np.uint8],
    boxes: ArrayLike,
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    """Mirror the image left-right and rewrite xyxy x coordinates."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("flip expects an H×W×3 image")
    width = int(image.shape[1])
    flipped = np.ascontiguousarray(image[:, ::-1, :])
    owned = _as_boxes(boxes)
    if owned.shape[0] == 0:
        return flipped, owned
    x1 = owned[:, 0].copy()
    x2 = owned[:, 2].copy()
    owned[:, 0] = width - x2
    owned[:, 2] = width - x1
    return flipped, owned


def save_examples(
    directory: str | Path,
    image: NDArray[np.uint8] | None = None,
    boxes: ArrayLike | None = None,
    *,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Write ``before.png`` and ``after.png``. Does not train.

    The after image uses the upper end of the v0.1 gains (+20% brightness
    and contrast), a crop that insets the far edges by the 10% budget, and
    a horizontal flip. ``seed`` is forwarded to :func:`apply`. Those three
    choices are fixed, so the pair does not depend on the draw.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    if image is None:
        source, owned = _synthetic_example()
    else:
        source = image
        owned = _as_boxes(np.zeros((0, 4), dtype=np.float64) if boxes is None else boxes)
    before = model_rgb_frame(source)
    sample = apply(
        before,
        owned,
        "train",
        seed=seed,
        flip_horizontal=True,
        brightness=BRIGHTNESS_LIMIT,
        contrast=CONTRAST_LIMIT,
        crop_window=_example_window(int(before.shape[0]), int(before.shape[1])),
    )
    before_path = target / "before.png"
    after_path = target / "after.png"
    Image.fromarray(np.ascontiguousarray(before)).save(before_path)
    Image.fromarray(np.ascontiguousarray(sample.image)).save(after_path)
    return before_path, after_path


def _prepare(
    image: NDArray[np.uint8],
    boxes: ArrayLike,
) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    return model_rgb_frame(image), _as_boxes(boxes)


def _as_boxes(boxes: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(boxes, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if array.ndim == 1 and array.shape == (4,):
        array = array.reshape(1, 4)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if not np.isfinite(array).all():
        raise ValueError("boxes must be finite")
    return np.array(array, dtype=np.float64, copy=True)


def _gain(value: float, limit: float, name: str) -> float:
    gain = float(value)
    if not math.isfinite(gain) or gain < -limit or gain > limit:
        raise ValueError(f"{name} gain must stay within ±{limit}")
    return gain


def _axis_budget(length: int) -> int:
    return math.floor(length * SCALE_LIMIT)


def _sample_crop_window(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    budget_x = _axis_budget(width)
    budget_y = _axis_budget(height)
    left = int(rng.integers(0, budget_x + 1))
    top = int(rng.integers(0, budget_y + 1))
    right = int(rng.integers(0, budget_x - left + 1))
    bottom = int(rng.integers(0, budget_y - top + 1))
    return (left, top, width - right, height - bottom)


def _check_window(
    height: int,
    width: int,
    window: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if len(window) != 4:
        raise ValueError("crop window must be (x0, y0, x1, y1)")
    x0, y0, x1, y1 = (_pixel(value) for value in window)
    if min(x0, y0) < 0 or x1 > width or y1 > height:
        raise ValueError("crop window is outside the image")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("crop window is empty")
    removed_x = x0 + (width - x1)
    removed_y = y0 + (height - y1)
    if removed_x > _axis_budget(width) or removed_y > _axis_budget(height):
        raise ValueError("crop window exceeds the mild scale limit")
    return x0, y0, x1, y1


def _pixel(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("crop window must be integer pixels")
    if isinstance(value, (int, np.integer)):
        return operator.index(value)
    raise TypeError("crop window must be integer pixels")


def _example_window(height: int, width: int) -> tuple[int, int, int, int]:
    inset_x = _axis_budget(width)
    inset_y = _axis_budget(height)
    return (0, 0, width - inset_x, height - inset_y)


def _synthetic_example() -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :32, 0] = 40
    image[:, 32:, 1] = 180
    image[12:28, 8:24, 2] = 220
    boxes = np.array([[8.0, 12.0, 24.0, 28.0]], dtype=np.float64)
    return image, boxes
