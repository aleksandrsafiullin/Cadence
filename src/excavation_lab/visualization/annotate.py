"""Draw stored observations with supervision.

Boxes and labels go through ``sv.Detections``, ``BoxAnnotator``, and
``RichLabelAnnotator``. A predicted bucket dash is a dashed segment and a
dashed rectangle. It is not a detection row; this module does not write
parquet and does not call ``sv.ByteTrack``.

Colors are a fixed RGB map keyed by ``class_name``. ``truck`` keeps its own
color. Zone polygons are optional. An empty list draws nothing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import supervision as sv
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from excavation_lab.errors import LabError
from excavation_lab.media.encoder import cyrillic_font_file

BASELINE_CAPTION = "Базовая модель, не обучена на наших классах"

# RGB. Supervision draws on a BGR copy; the returned frame is RGB again.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "excavator_body": (230, 126, 34),
    "bucket": (241, 196, 15),
    "haul_truck": (52, 152, 219),
    "truck_bed": (46, 204, 113),
    "truck": (155, 89, 182),
}
FALLBACK_COLOR = (149, 165, 166)
ZONE_COLOR = (127, 140, 141)
_CLASS_ORDER = ("excavator_body", "bucket", "haul_truck", "truck_bed", "truck")
_DASH = 8
_GAP = 5
_THICKNESS = 2


@dataclass(frozen=True, slots=True)
class AnnotateToggles:
    """What the annotated frame includes. Defaults leave every layer on."""

    boxes: bool = True
    class_names: bool = True
    scores: bool = True
    track_ids: bool = True
    bucket_trail: bool = True
    zones: bool = True


@dataclass(frozen=True, slots=True)
class DrawBox:
    """One rectangle. ``predicted`` is display-only and is not a detection."""

    class_name: str
    xyxy: tuple[float, float, float, float]
    score: float | None = None
    track_id: str | None = None
    predicted: bool = False


@dataclass(frozen=True, slots=True)
class TrailSegment:
    """Polyline in pixel centers. ``predicted`` is drawn dashed."""

    class_name: str
    points: tuple[tuple[float, float], ...]
    predicted: bool = False


@dataclass(frozen=True, slots=True)
class ZonePolygon:
    """One camera zone. Fewer than three vertices is rejected."""

    name: str
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or "\x00" in self.name:
            raise LabError("invalid_zone", "У зоны нет имени.")
        if len(self.points) < 3:
            raise LabError(
                "invalid_zone",
                "У зоны меньше трёх вершин. Пустой список зон допустим.",
            )
        for point in self.points:
            if len(point) != 2:
                raise LabError("invalid_zone", "Вершина зоны должна быть парой x, y.")
            for value in point:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise LabError("invalid_zone", "Координата зоны должна быть числом.")
                if not math.isfinite(float(value)):
                    raise LabError("invalid_zone", "Координата зоны должна быть числом.")


def label_text(
    box: DrawBox,
    toggles: AnnotateToggles,
    *,
    titles: Mapping[str, str] | None = None,
    baseline: bool = False,
) -> str | None:
    """Class title, score, and track id. A missing score is omitted.

    Baseline runs keep the checkpoint class name. ``truck`` is not rewritten
    to ``excavator_body`` or to a Russian project title. Custom runs use
    ``titles`` from ``labels.yaml`` when the name is one of the four classes.
    """
    parts: list[str] = []
    if toggles.class_names:
        parts.append(_class_label(box.class_name, titles or {}, baseline=baseline))
    if toggles.scores and _finite_score(box.score) is not None:
        parts.append(f"{_finite_score(box.score):.2f}")
    if toggles.track_ids and isinstance(box.track_id, str) and box.track_id:
        parts.append(box.track_id)
    if not parts:
        return None
    return " ".join(parts)


def bucket_trail(
    points: Sequence[tuple[str, float, tuple[float, float, float, float]]],
    *,
    source_time_s: float,
    trail_s: float,
    trail_gap_s: float,
    predicted_boxes: Sequence[tuple[str, tuple[float, float, float, float]]] = (),
    extrapolation_s: float = 0.4,
) -> list[TrailSegment]:
    """Observed bucket polyline plus a dashed extrapolation.

    ``points`` are detector boxes ``(track_id, source_time_s, xyxy)``. A gap
    longer than ``trail_gap_s`` starts a new solid segment and does not draw
    a straight line across the hole. A predicted box is not inserted into
    those points. Its segment runs from the last observed center, and only
    when that center is within ``extrapolation_s``.
    """
    grouped: dict[str, list[tuple[float, tuple[float, float, float, float]]]] = {}
    for track_id, stamp, xyxy in points:
        if not track_id:
            continue
        if stamp > source_time_s + 1e-9:
            continue
        if source_time_s - stamp > trail_s + 1e-9:
            continue
        grouped.setdefault(track_id, []).append((float(stamp), xyxy))
    segments: list[TrailSegment] = []
    for history in grouped.values():
        history.sort(key=lambda item: item[0])
        for group in _split_gap(history, trail_gap_s):
            if len(group) < 2:
                continue
            segments.append(
                TrailSegment(
                    class_name="bucket",
                    points=tuple(_center(item[1]) for item in group),
                    predicted=False,
                )
            )
    for track_id, xyxy in predicted_boxes:
        history = grouped.get(track_id, ())
        recent = [
            item
            for item in history
            if source_time_s - item[0] <= extrapolation_s + 1e-9
        ]
        if not recent:
            continue
        last = max(recent, key=lambda item: item[0])
        segments.append(
            TrailSegment(
                class_name="bucket",
                points=(_center(last[1]), _center(xyxy)),
                predicted=True,
            )
        )
    return segments


def annotate_rgb(
    image: NDArray[np.uint8],
    boxes: Sequence[DrawBox],
    *,
    toggles: AnnotateToggles | None = None,
    titles: Mapping[str, str] | None = None,
    baseline: bool = False,
    burn_baseline_caption: bool = False,
    trail: Sequence[TrailSegment] = (),
    zones: Sequence[ZonePolygon] = (),
) -> NDArray[np.uint8]:
    """Return a new RGB frame. ``image`` is not modified.

    ``zones`` may be empty. Predicted boxes are dashed and are not added to
    the solid ``Detections`` passed to ``BoxAnnotator``.
    """
    selected = AnnotateToggles() if toggles is None else toggles
    names = {} if titles is None else titles
    _require_rgb(image)
    bgr = np.ascontiguousarray(np.array(image, dtype=np.uint8, copy=True)[:, :, ::-1])
    if selected.zones and zones:
        _draw_zones(bgr, zones)
    if selected.bucket_trail and trail:
        _draw_trail(bgr, trail)
    observed = [box for box in boxes if not box.predicted]
    predicted = [box for box in boxes if box.predicted]
    if selected.boxes:
        _draw_solid_boxes(bgr, observed)
        _draw_dashed_boxes(bgr, predicted)
    _draw_labels(bgr, observed, selected, names, baseline=baseline)
    _draw_labels(bgr, predicted, selected, names, baseline=baseline)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    if burn_baseline_caption and baseline:
        rgb = _burn_caption(rgb, BASELINE_CAPTION)
    return rgb


def _class_label(class_name: str, titles: Mapping[str, str], *, baseline: bool) -> str:
    if baseline:
        return class_name
    title = titles.get(class_name)
    if isinstance(title, str) and title:
        return title
    return class_name


def _finite_score(score: float | None) -> float | None:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    number = float(score)
    if not math.isfinite(number):
        return None
    return number


def _split_gap(
    history: Sequence[tuple[float, tuple[float, float, float, float]]],
    trail_gap_s: float,
) -> list[list[tuple[float, tuple[float, float, float, float]]]]:
    if not history:
        return []
    groups: list[list[tuple[float, tuple[float, float, float, float]]]] = [[history[0]]]
    for item in history[1:]:
        if item[0] - groups[-1][-1][0] > trail_gap_s:
            groups.append([item])
            continue
        groups[-1].append(item)
    return groups


def _center(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)


def _draw_zones(scene: NDArray[np.uint8], zones: Sequence[ZonePolygon]) -> None:
    color = _sv_color(ZONE_COLOR)
    for zone in zones:
        polygon = np.array(zone.points, dtype=np.int32)
        sv.draw_polygon(scene, polygon, color=color, thickness=_THICKNESS)


def _draw_trail(scene: NDArray[np.uint8], trail: Sequence[TrailSegment]) -> None:
    for segment in trail:
        if len(segment.points) < 2:
            continue
        color = _sv_color(CLASS_COLORS.get(segment.class_name, FALLBACK_COLOR))
        pairs = pairwise(segment.points)
        if segment.predicted:
            for start, end in pairs:
                _dashed(scene, start, end, color)
            continue
        for start, end in pairs:
            sv.draw_line(
                scene,
                sv.Point(x=start[0], y=start[1]),
                sv.Point(x=end[0], y=end[1]),
                color=color,
                thickness=_THICKNESS,
            )


def _draw_solid_boxes(scene: NDArray[np.uint8], boxes: Sequence[DrawBox]) -> None:
    for name in _ordered_names(boxes):
        group = [box for box in boxes if box.class_name == name]
        detections = _detections(group)
        sv.BoxAnnotator(color=_sv_color_name(name), thickness=_THICKNESS).annotate(
            scene,
            detections,
        )


def _draw_dashed_boxes(scene: NDArray[np.uint8], boxes: Sequence[DrawBox]) -> None:
    for box in boxes:
        color = _sv_color_name(box.class_name)
        x1, y1, x2, y2 = box.xyxy
        corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        for start, end in pairwise((*corners, corners[0])):
            _dashed(scene, start, end, color)


def _draw_labels(
    scene: NDArray[np.uint8],
    boxes: Sequence[DrawBox],
    toggles: AnnotateToggles,
    titles: Mapping[str, str],
    *,
    baseline: bool,
) -> None:
    labeled = [
        (box, text)
        for box in boxes
        if (text := label_text(box, toggles, titles=titles, baseline=baseline)) is not None
    ]
    if not labeled:
        return
    height = int(scene.shape[0])
    font = str(cyrillic_font_file())
    size = 11 if height < 100 else min(28, max(14, height // 36))
    for name in _ordered_names([box for box, _text in labeled]):
        group = [(box, text) for box, text in labeled if box.class_name == name]
        detections = _detections([box for box, _text in group])
        sv.RichLabelAnnotator(
            color=_sv_color_name(name),
            text_color=sv.Color.BLACK,
            font_path=font,
            font_size=size,
            text_padding=2,
        ).annotate(scene, detections, labels=[text for _box, text in group])


def _detections(boxes: Sequence[DrawBox]) -> sv.Detections:
    xyxy = np.array([box.xyxy for box in boxes], dtype=np.float32)
    class_id = np.zeros((len(boxes),), dtype=np.int64)
    return sv.Detections(xyxy=xyxy, class_id=class_id)


def _ordered_names(boxes: Sequence[DrawBox]) -> list[str]:
    present = {box.class_name for box in boxes}
    ordered = [name for name in _CLASS_ORDER if name in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _dashed(
    scene: NDArray[np.uint8],
    start: tuple[float, float],
    end: tuple[float, float],
    color: sv.Color,
) -> None:
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1.0:
        return
    step_x = (x1 - x0) / length
    step_y = (y1 - y0) / length
    cursor = 0.0
    draw = True
    while cursor < length:
        span = _DASH if draw else _GAP
        nxt = min(length, cursor + span)
        if draw:
            sv.draw_line(
                scene,
                sv.Point(x=x0 + step_x * cursor, y=y0 + step_y * cursor),
                sv.Point(x=x0 + step_x * nxt, y=y0 + step_y * nxt),
                color=color,
                thickness=_THICKNESS,
            )
        draw = not draw
        cursor = nxt


def _sv_color_name(class_name: str) -> sv.Color:
    return _sv_color(CLASS_COLORS.get(class_name, FALLBACK_COLOR))


def _sv_color(rgb: tuple[int, int, int]) -> sv.Color:
    red, green, blue = rgb
    return sv.Color(r=red, g=green, b=blue)


def _burn_caption(rgb: NDArray[np.uint8], text: str) -> NDArray[np.uint8]:
    image = Image.fromarray(np.ascontiguousarray(rgb))
    draw = ImageDraw.Draw(image)
    font = _caption_font(draw, text, image.width, image.height)
    bbox = draw.textbbox((0, 0), text, font=font)
    origin_x = 4
    origin_y = 4
    draw.rectangle(
        (
            origin_x + bbox[0] - 2,
            origin_y + bbox[1] - 2,
            origin_x + bbox[2] + 2,
            origin_y + bbox[3] + 2,
        ),
        fill=(20, 20, 20),
    )
    draw.text((origin_x, origin_y), text, fill=(255, 255, 255), font=font)
    return np.asarray(image, dtype=np.uint8)


def _caption_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    width: int,
    height: int,
) -> ImageFont.FreeTypeFont:
    path = cyrillic_font_file()
    largest = max(10, min(height // 6, 32))
    chosen = ImageFont.truetype(str(path), size=8)
    for size in range(largest, 7, -1):
        font = ImageFont.truetype(str(path), size=size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= width - 8 and bbox[3] - bbox[1] <= height // 3:
            return font
        chosen = font
    return chosen


def _require_rgb(image: NDArray[np.uint8]) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise LabError("invalid_frame", "Кадр аннотации должен быть массивом RGB.")
    if image.dtype != np.uint8:
        raise LabError("invalid_frame", "Кадр аннотации должен быть RGB uint8.")
