"""CFR preview. Duration follows the source interval, not the source FPS.

Output FPS is an argument. Five kept frames spread across one second of
source time are held on that timeline, so the file is about one second,
not ``5/30`` s. A repeated picture is the same observation: the extra
output frames set ``repeated`` and do not allocate a new source frame.

Preview time may start at 0. ``source_time_s`` in ``preview_time_map.csv``
stays on the source clock. A fragment is not rewritten to timestamp 0.

When two kept timestamps differ by more than ``preview_gap_s``, the output
frames between them are a slate («нет наблюдения»). Those rows are not
observations. Annotated boxes are a later export and must reuse this map.
"""

from __future__ import annotations

import csv
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from excavation_lab.errors import DuplicateFrameError, LabError, TimebaseError
from excavation_lab.media.reader import ReadFrame
from excavation_lab.media.timebase import (
    in_half_open,
    require_half_open_interval,
    require_monotonic,
)
from excavation_lab.schemas import DecodeStatus

# More than this gap between kept source timestamps is a missing observation.
DEFAULT_PREVIEW_GAP_S = 0.5
# Set only by the annotated-export cancel test. Production leaves it unset.
EXPORT_HOLD_ENV = "EXCAVATION_LAB_EXPORT_HOLD"
GAP_SLATE_TEXT = "нет наблюдения"
PREVIEW_MAP_COLUMNS = (
    "preview_time_s",
    "source_time_s",
    "source_frame_index",
    "repeated",
    "gap_slate",
)

# Same edge as timebase.in_half_open. A tick 1 µs off a boundary still belongs.
_EDGE_S = Fraction(1, 1_000_000)
_SLATE_RGB = (32, 32, 32)
_SLATE_INK = (240, 240, 240)
_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
)


@dataclass(frozen=True, slots=True, eq=False)
class PreviewFrame:
    """One kept observation. A CFR repeat is not another ``PreviewFrame``."""

    source_frame_index: int
    source_time_s: float
    image: NDArray[np.uint8]

    @classmethod
    def from_read_frame(cls, frame: ReadFrame) -> PreviewFrame:
        """Take the PTS clock and upright RGB image from the reader.

        ``source_time_s`` is the reader's source clock, including the start
        offset. It is not ``frame_index / fps`` and it is not rebased.
        """
        if frame.decode_status is not DecodeStatus.ok:
            raise TimebaseError(
                "Повреждённый кадр не становится наблюдением в preview."
            )
        if (
            frame.image is None
            or frame.source_frame_index is None
            or frame.source_time_s is None
        ):
            raise TimebaseError(
                "Кадр без изображения или времени нельзя записать в preview."
            )
        image = np.array(frame.image, dtype=np.uint8, copy=True)
        return cls(
            source_frame_index=int(frame.source_frame_index),
            source_time_s=float(frame.source_time_s),
            image=np.ascontiguousarray(image),
        )


@dataclass(frozen=True, slots=True)
class PreviewMapRow:
    """One output frame. ``repeated`` is the same observation, not a new one."""

    preview_time_s: float
    source_time_s: float
    source_frame_index: int | None
    repeated: bool
    gap_slate: bool


def write_preview(
    frames: Sequence[PreviewFrame],
    output_path: Path,
    *,
    output_fps: float,
    start_s: float,
    end_s: float,
    preview_gap_s: float = DEFAULT_PREVIEW_GAP_S,
    map_path: Path | None = None,
) -> list[PreviewMapRow]:
    """Write a silent CFR MP4 and ``preview_time_map.csv``.

    ``output_fps`` is the preview rate. It is not the source FPS. The file
    has no audio stream. ``map_path`` defaults to ``preview_time_map.csv``
    beside the MP4.
    """
    rows = build_preview_map(
        frames,
        output_fps=output_fps,
        start_s=start_s,
        end_s=end_s,
        preview_gap_s=preview_gap_s,
    )
    pictures = _pictures(frames, rows)
    destination = Path(output_path)
    _encode_cfr(destination, pictures, _rate(output_fps))
    csv_path = destination.with_name("preview_time_map.csv") if map_path is None else Path(map_path)
    _write_map(csv_path, rows)
    return rows


def build_preview_map(
    frames: Sequence[PreviewFrame],
    *,
    output_fps: float,
    start_s: float,
    end_s: float,
    preview_gap_s: float = DEFAULT_PREVIEW_GAP_S,
) -> list[PreviewMapRow]:
    """Map each output frame onto the half-open source interval.

    The output grid is ``start_s + i / output_fps``. A kept frame covers the
    grid until the next kept timestamp. If that step is longer than
    ``preview_gap_s``, only the output frame on the kept timestamp shows the
    picture. The following frames, up to the next kept timestamp, are the
    slate. The last kept frame is held through ``end_s``: the interval end
    is not a second observation.
    """
    kept = _validate_frames(frames, start_s, end_s)
    rate = _rate(output_fps)
    gap_limit = _gap_limit(preview_gap_s)
    start = _seconds(start_s)
    end = _seconds(end_s)
    count = _output_frame_count(start, end, rate)
    period = Fraction(1) / rate
    large_after = [
        _exceeds_gap(kept[index + 1].source_time_s, kept[index].source_time_s, gap_limit)
        if index + 1 < len(kept)
        else False
        for index in range(len(kept))
    ]
    rows: list[PreviewMapRow] = []
    shown: set[int] = set()
    for index in range(count):
        preview = Fraction(index) / rate
        clock = start + preview
        if not _clock_inside(clock, start, end):
            break
        row = _row_at(
            kept,
            large_after,
            preview=preview,
            clock=clock,
            period=period,
            gap_limit=gap_limit,
            shown=shown,
        )
        rows.append(row)
    if not rows:
        raise TimebaseError(
            "Интервал preview пуст. Нужен полуоткрытый интервал с кадрами внутри."
        )
    return rows


def render_gap_slate(width: int, height: int) -> NDArray[np.uint8]:
    """RGB slate. The text is «нет наблюдения». Not an observation frame."""
    _require_even_size(width, height)
    image = Image.new("RGB", (width, height), _SLATE_RGB)
    draw = ImageDraw.Draw(image)
    font = _fit_font(draw, GAP_SLATE_TEXT, width, height)
    bbox = draw.textbbox((0, 0), GAP_SLATE_TEXT, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    origin_x = (width - text_width) // 2 - bbox[0]
    origin_y = (height - text_height) // 2 - bbox[1]
    draw.text((origin_x, origin_y), GAP_SLATE_TEXT, fill=_SLATE_INK, font=font)
    return np.array(image, dtype=np.uint8, copy=True)


def _pictures(
    frames: Sequence[PreviewFrame],
    rows: Sequence[PreviewMapRow],
) -> list[NDArray[np.uint8]]:
    by_index = {frame.source_frame_index: frame.image for frame in frames}
    height, width = _frame_size(frames[0].image)
    slate = render_gap_slate(width, height)
    pictures: list[NDArray[np.uint8]] = []
    for row in rows:
        if row.gap_slate or row.source_frame_index is None:
            pictures.append(slate)
            continue
        pictures.append(by_index[row.source_frame_index])
    return pictures


def _row_at(
    kept: Sequence[PreviewFrame],
    large_after: Sequence[bool],
    *,
    preview: Fraction,
    clock: Fraction,
    period: Fraction,
    gap_limit: Fraction,
    shown: set[int],
) -> PreviewMapRow:
    active = _active_index(kept, clock)
    if active is None:
        # Before the first kept timestamp. A short lead still shows that
        # picture. A lead longer than the gap threshold is missing time.
        first = kept[0]
        lead = _seconds(first.source_time_s) - clock
        if lead <= gap_limit + _EDGE_S:
            return _observation_row(first, preview, shown)
        return _slate_row(preview, clock)
    frame = kept[active]
    if large_after[active] and not _on_kept_tick(frame.source_time_s, clock, period):
        # The step to the next kept timestamp is the missing duration.
        # Holding the previous picture across it would hide the gap.
        return _slate_row(preview, clock)
    return _observation_row(frame, preview, shown)


def _observation_row(
    frame: PreviewFrame,
    preview: Fraction,
    shown: set[int],
) -> PreviewMapRow:
    repeated = frame.source_frame_index in shown
    shown.add(frame.source_frame_index)
    return PreviewMapRow(
        preview_time_s=float(preview),
        source_time_s=frame.source_time_s,
        source_frame_index=frame.source_frame_index,
        repeated=repeated,
        gap_slate=False,
    )


def _slate_row(preview: Fraction, clock: Fraction) -> PreviewMapRow:
    return PreviewMapRow(
        preview_time_s=float(preview),
        source_time_s=float(clock),
        source_frame_index=None,
        repeated=False,
        gap_slate=True,
    )


def _active_index(kept: Sequence[PreviewFrame], clock: Fraction) -> int | None:
    chosen: int | None = None
    for index, frame in enumerate(kept):
        if _seconds(frame.source_time_s) <= clock + _EDGE_S:
            chosen = index
            continue
        break
    return chosen


def _on_kept_tick(source_time_s: float, clock: Fraction, period: Fraction) -> bool:
    """True on the single output frame that lands on this kept timestamp."""
    stamp = _seconds(source_time_s)
    return stamp - _EDGE_S <= clock < stamp + period - _EDGE_S


def _validate_frames(
    frames: Sequence[PreviewFrame],
    start_s: float,
    end_s: float,
) -> list[PreviewFrame]:
    require_half_open_interval(start_s, end_s)
    if end_s is None or not end_s > start_s:
        raise TimebaseError(
            "Интервал preview должен быть полуоткрытым [начало, конец) "
            "с положительной длительностью."
        )
    if not frames:
        raise TimebaseError(
            "Нет кадров для preview. Пустой интервал не записывается."
        )
    height, width = _frame_size(frames[0].image)
    kept: list[PreviewFrame] = []
    seen: set[int] = set()
    previous_s: float | None = None
    for frame in frames:
        _require_frame(frame, width, height)
        if frame.source_frame_index in seen:
            raise DuplicateFrameError(frame.source_frame_index)
        if not in_half_open(frame.source_time_s, start_s, end_s):
            raise TimebaseError(
                f"Кадр {frame.source_time_s} вне интервала [{start_s}, {end_s}). "
                "Preview не переносит исходное время в ноль."
            )
        if previous_s is not None:
            require_monotonic(previous_s, frame.source_time_s)
            if frame.source_time_s <= previous_s:
                raise TimebaseError(
                    "Временные метки убывают. "
                    "Шкала исходного видео не перезапускается и не идёт назад."
                )
        seen.add(frame.source_frame_index)
        previous_s = frame.source_time_s
        kept.append(frame)
    return kept


def _require_frame(frame: PreviewFrame, width: int, height: int) -> None:
    if (
        isinstance(frame.source_frame_index, bool)
        or not isinstance(frame.source_frame_index, int)
        or frame.source_frame_index < 0
    ):
        raise TimebaseError("Номер исходного кадра preview должен быть неотрицательным целым.")
    if isinstance(frame.source_time_s, bool) or not isinstance(frame.source_time_s, (int, float)):
        raise TimebaseError("Метка времени кадра preview не является конечным числом.")
    if not math.isfinite(float(frame.source_time_s)):
        raise TimebaseError("Метка времени кадра preview не является конечным числом.")
    frame_height, frame_width = _frame_size(frame.image)
    if frame_width != width or frame_height != height:
        raise TimebaseError("Кадры preview должны быть одного размера.")


def _frame_size(image: NDArray[np.uint8]) -> tuple[int, int]:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise TimebaseError("Кадр preview должен быть массивом RGB из трёх каналов.")
    if image.dtype != np.uint8:
        raise TimebaseError("Кадр preview должен быть RGB uint8.")
    height, width = int(image.shape[0]), int(image.shape[1])
    _require_even_size(width, height)
    return height, width


def _require_even_size(width: int, height: int) -> None:
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
        or width % 2
        or height % 2
    ):
        raise TimebaseError(
            "Ширина и высота preview должны быть положительными и чётными."
        )


def _output_frame_count(start: Fraction, end: Fraction, rate: Fraction) -> int:
    count = (end - start) * rate
    # Positive spans only. Half up: an exact integer stays, *.5 goes up.
    rounded = int(count + Fraction(1, 2))
    if rounded < 1:
        raise TimebaseError(
            "Интервал preview пуст. Нужен полуоткрытый интервал с кадрами внутри."
        )
    return rounded


def _clock_inside(clock: Fraction, start: Fraction, end: Fraction) -> bool:
    """Half-open ``[start, end)`` with the same 1 µs edge as the source clock."""
    return clock + _EDGE_S >= start and clock < end


def _exceeds_gap(later_s: float, earlier_s: float, preview_gap_s: Fraction) -> bool:
    delta = _seconds(later_s) - _seconds(earlier_s)
    return delta > preview_gap_s + _EDGE_S


def _encode_cfr(
    path: Path,
    images: Sequence[NDArray[np.uint8]],
    output_fps: Fraction,
) -> None:
    """Mux video only. One PTS tick is one preview frame, not a source frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(images[0].shape[0]), int(images[0].shape[1])
    time_base = Fraction(output_fps.denominator, output_fps.numerator)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=output_fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = time_base
        stream.options = {"preset": "ultrafast", "crf": "18"}
        # Inside the open container. A kill here never returns to mark_completed.
        _pause_when_holding()
        for index, image in enumerate(images):
            owned = np.ascontiguousarray(image, dtype=np.uint8).copy()
            frame = av.VideoFrame.from_ndarray(owned, format="rgb24")
            # reformat returns a new frame and drops pts.
            coded = frame.reformat(format="yuv420p")
            coded.pts = index
            coded.time_base = time_base
            for packet in stream.encode(coded):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def _write_map(path: Path, rows: Sequence[PreviewMapRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PREVIEW_MAP_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            index = "" if row.source_frame_index is None else str(row.source_frame_index)
            writer.writerow(
                {
                    "preview_time_s": _format_seconds(row.preview_time_s),
                    "source_time_s": _format_seconds(row.source_time_s),
                    "source_frame_index": index,
                    "repeated": "true" if row.repeated else "false",
                    "gap_slate": "true" if row.gap_slate else "false",
                }
            )


def _format_seconds(value: float) -> str:
    return f"{value:.6f}"


def _rate(output_fps: float) -> Fraction:
    if isinstance(output_fps, bool) or not isinstance(output_fps, (int, float)):
        raise TimebaseError("Частота preview должна быть положительной.")
    if not math.isfinite(float(output_fps)) or float(output_fps) <= 0:
        raise TimebaseError("Частота preview должна быть положительной.")
    rate = Fraction(float(output_fps)).limit_denominator(100_000)
    if rate <= 0:
        raise TimebaseError("Частота preview должна быть положительной.")
    return rate


def _gap_limit(preview_gap_s: float) -> Fraction:
    if isinstance(preview_gap_s, bool) or not isinstance(preview_gap_s, (int, float)):
        raise TimebaseError("Порог пропуска preview должен быть неотрицательным.")
    if not math.isfinite(float(preview_gap_s)) or float(preview_gap_s) < 0:
        raise TimebaseError("Порог пропуска preview должен быть неотрицательным.")
    return _seconds(float(preview_gap_s))


def _seconds(value: float) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimebaseError("Метка времени preview не является конечным числом.")
    if not math.isfinite(float(value)):
        raise TimebaseError("Метка времени preview не является конечным числом.")
    return Fraction(float(value)).limit_denominator(10_000_000)


def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    width: int,
    height: int,
) -> ImageFont.FreeTypeFont:
    path = _font_file()
    largest = max(8, height // 3)
    chosen = ImageFont.truetype(str(path), size=8)
    for size in range(largest, 7, -1):
        font = ImageFont.truetype(str(path), size=size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= width - 2 and bbox[3] - bbox[1] <= height - 2:
            return font
        chosen = font
    return chosen


def cyrillic_font_file() -> Path:
    """Font that can draw Russian captions, including «нет наблюдения»."""
    return _font_file()


def _pause_when_holding() -> None:
    """Stay inside the open container when the export-hold variable is set.

    The annotated-export worker publishes progress, then enters this
    container. Cancelling during the pause leaves the job ``cancelled`` or
    ``failed`` with ``partial=true``. ``mark_completed`` runs only after
    this function returns and the container closes.
    """
    if os.environ.get(EXPORT_HOLD_ENV) != "1":
        return
    time.sleep(30)


def _font_file() -> Path:
    for path in _FONT_CANDIDATES:
        if path.is_file():
            return path
    raise LabError(
        "preview_font",
        "Не найден шрифт с кириллицей для надписи «нет наблюдения».",
    )
