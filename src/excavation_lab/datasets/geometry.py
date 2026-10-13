"""Box geometry for COCO exchange.

Internal boxes are ``xyxy`` pixels ``(x1, y1, x2, y2)``. COCO ``bbox`` is
``xywh`` ``(x, y, width, height)``. Width and height must be positive.
When image width and height are passed, the box must lie inside the image.
This module does not clip a box and does not map classes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def xyxy_to_xywh(
    xyxy: Sequence[float],
    *,
    image_width: float | None = None,
    image_height: float | None = None,
) -> tuple[float, float, float, float]:
    """Convert one internal box to COCO ``[x, y, width, height]``."""
    x1, y1, x2, y2 = _four(xyxy)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("box must have positive width and height")
    _require_inside(x1, y1, x2, y2, image_width, image_height)
    return (x1, y1, width, height)


def xywh_to_xyxy(
    xywh: Sequence[float],
    *,
    image_width: float | None = None,
    image_height: float | None = None,
) -> tuple[float, float, float, float]:
    """Convert one COCO box to internal ``[x1, y1, x2, y2]``."""
    x, y, width, height = _four(xywh)
    if width <= 0 or height <= 0:
        raise ValueError("box must have positive width and height")
    x2 = x + width
    y2 = y + height
    _require_inside(x, y, x2, y2, image_width, image_height)
    return (x, y, x2, y2)


def _four(values: Sequence[float]) -> tuple[float, float, float, float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("box must be four finite coordinates")
    if len(values) != 4:
        raise ValueError("box must be four finite coordinates")
    parsed: list[float] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError("box must be four finite coordinates")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("box must be four finite coordinates")
        parsed.append(number)
    return (parsed[0], parsed[1], parsed[2], parsed[3])


def _require_inside(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: float | None,
    image_height: float | None,
) -> None:
    if image_width is None and image_height is None:
        return
    if image_width is None or image_height is None:
        raise TypeError("image width and height are set together")
    width = _positive_extent(image_width)
    height = _positive_extent(image_height)
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError("box must lie inside the image")


def _positive_extent(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("image width and height must be positive numbers")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("image width and height must be positive")
    return number
