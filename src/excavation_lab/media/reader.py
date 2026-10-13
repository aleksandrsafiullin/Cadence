"""Streaming PyAV reader. One RGB frame at a time; the clip is not a list.

Seek uses container timestamps in the video stream time base
(``FrameReader`` calls ``InputContainer.seek``). ``backward=True`` stops on
the previous keyframe, and frames with ``source_time_s < start_s`` are
dropped. That preroll is how a keyframe decode works. It is not a fake seek
that starts at timestamp 0 and discards frames when the container can seek.

``tests/fixtures/synthetic.py`` ``write_synthetic_mp4`` writes one leading
keyframe, so a timestamp seek on that fixture returns to time 0 and the
reader still decodes forward. The source timestamps do not change. An
all-keyframe file seeks to the requested tick.

Display rotation: PyAV does not autorotate. ``frame.rotation`` / the
display matrix is counterclockwise degrees, matching ffmpeg autorotate.
``map_coded_image`` and ``map_coded_box`` turn coded pixels into that
upright frame. ``ReadFrame.image`` is already upright RGB; do not map it
again and do not swap channels to BGR.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from numpy.typing import NDArray

from excavation_lab.errors import (
    CorruptMediaError,
    DuplicateFrameError,
    MediaNotFoundError,
    TimebaseError,
)
from excavation_lab.media.timebase import (
    TimeBase,
    TimeMode,
    cfr_fallback_time_s,
    in_half_open,
    require_half_open_interval,
    require_monotonic,
    resolve_time_mode,
    seconds_to_ticks,
    source_time_s,
)
from excavation_lab.schemas import DecodeStatus

_STOP = object()
_MISSING_PTS = (
    "У кадра нет PTS. Время по номеру кадра включается только после "
    "проверки постоянной частоты."
)
_BAD_ROTATION = (
    "Поворот display matrix не кратен 90°. "
    "Кадр нельзя перевести в прямые координаты."
)


@dataclass(eq=False, slots=True)
class _Pass:
    previous_s: float | None = None
    previous_index: int | None = None
    seen: set[int] = field(default_factory=set)
    # Set while every duration matches. A longer frame is not a new period.
    period_ticks: int | None = None
    variable_duration: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class ReadFrame:
    """One decoded frame, or a corrupt span. Not an event and not downtime.

    ``source_time_s`` includes the start offset. ``time_base`` is
    ``numerator/denominator``. ``image`` is upright RGB (``rgb24``) or
    ``None`` when ``decode_status`` is corrupt. ``span_end_s`` is exclusive.
    """

    video_id: str
    source_frame_index: int | None
    source_pts: int | None
    time_base_numerator: int
    time_base_denominator: int
    source_time_s: float | None
    span_end_s: float | None
    start_offset_s: float
    width: int | None
    height: int | None
    coded_width: int | None
    coded_height: int | None
    rotation_deg: int
    image: NDArray[np.uint8] | None
    decode_status: DecodeStatus
    time_mode: TimeMode

    @property
    def time_base(self) -> str:
        return f"{self.time_base_numerator}/{self.time_base_denominator}"

    def __repr__(self) -> str:
        return (
            "ReadFrame("
            f"video_id={self.video_id!r}, "
            f"source_frame_index={self.source_frame_index}, "
            f"source_time_s={self.source_time_s}, "
            f"decode_status={self.decode_status.value}, "
            f"time_mode={self.time_mode.value})"
        )


class FrameReader:
    """Read ``path`` from disk. ``iterate(start_s, end_s)`` is half-open."""

    def __init__(
        self,
        path: Path,
        *,
        video_id: str,
        time_mode: TimeMode = TimeMode.pts,
    ) -> None:
        if not isinstance(video_id, str) or not video_id.strip() or "\x00" in video_id:
            raise TimebaseError("Пустой идентификатор видео.")
        self.path = Path(path)
        self.video_id = video_id
        try:
            self.time_mode = TimeMode(time_mode)
        except ValueError as exc:
            raise TimebaseError(
                "Неизвестный режим времени. Допустимы pts и cfr_fallback."
            ) from exc

    def iterate(
        self,
        start_s: float = 0.0,
        end_s: float | None = None,
    ) -> Iterator[ReadFrame]:
        """Yield frames with ``start_s <= source_time_s < end_s``.

        The iterator does not accumulate frames. Close it to release the
        container, or exhaust it.
        """
        require_half_open_interval(start_s, end_s)
        container: av.container.InputContainer | None = None
        try:
            container = _open_container(self.path)
            fps = _fallback_fps(container, self.path, self.time_mode)
            if fps is not None:
                container.close()
                container = None  # second open may fail; do not close twice
                container = _open_container(self.path)
            yield from _read_frames(
                container,
                path=self.path,
                video_id=self.video_id,
                start_s=start_s,
                end_s=end_s,
                time_mode=self.time_mode,
                fps=fps,
            )
        finally:
            if container is not None:
                container.close()


def decode_packet(packet: av.Packet) -> Iterator[av.VideoFrame]:
    """Yield the frames in one packet. Tests replace this to inject failures."""
    yield from packet.decode()


def upright_size(
    coded_width: int,
    coded_height: int,
    rotation_deg: float,
) -> tuple[int, int]:
    """Display width and height after a display-matrix rotation."""
    _require_size(coded_width, coded_height)
    if _quarter_turns(rotation_deg) in (90, 270):
        return coded_height, coded_width
    return coded_width, coded_height


def map_coded_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    coded_width: int,
    coded_height: int,
    rotation_deg: float,
) -> tuple[float, float, float, float]:
    """Map a coded xyxy box into upright coordinates.

    The box is the continuous rectangle ``[x1, x2) × [y1, y2)`` on the coded
    frame. ``rotation_deg`` is counterclockwise, the same sign as
    ``VideoFrame.rotation`` and ffmpeg autorotate.
    """
    _require_size(coded_width, coded_height)
    quarter = _quarter_turns(rotation_deg)
    if quarter == 0:
        return (float(x1), float(y1), float(x2), float(y2))
    if quarter == 90:
        return (
            float(y1),
            float(coded_width - x2),
            float(y2),
            float(coded_width - x1),
        )
    if quarter == 180:
        return (
            float(coded_width - x2),
            float(coded_height - y2),
            float(coded_width - x1),
            float(coded_height - y1),
        )
    return (
        float(coded_height - y2),
        float(x1),
        float(coded_height - y1),
        float(x2),
    )


def map_coded_image(
    image: NDArray[np.uint8],
    rotation_deg: float,
) -> NDArray[np.uint8]:
    """Rotate a coded RGB array upright. Channels stay RGB."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise TimebaseError(
            "Кадр для поворота должен быть массивом RGB из трёх каналов."
        )
    quarter = _quarter_turns(rotation_deg)
    if quarter == 0:
        copied = np.copy(image)
        return np.ascontiguousarray(copied)
    turns = {90: 1, 180: 2, 270: 3}[quarter]
    rotated = np.rot90(image, turns)
    return np.ascontiguousarray(rotated)


def _open_container(path: Path) -> av.container.InputContainer:
    if not path.is_file():
        raise MediaNotFoundError(str(path))
    try:
        return av.open(str(path))
    except av.error.FFmpegError as exc:
        raise CorruptMediaError(str(path)) from exc


def _fallback_fps(
    container: av.container.InputContainer,
    path: Path,
    time_mode: TimeMode,
) -> float | None:
    """Scan packet durations only when the caller asked for ``cfr_fallback``."""
    if time_mode is not TimeMode.cfr_fallback:
        return None
    stream = _video_stream(container, path)
    time_base = _stream_time_base(stream)
    durations = _packet_durations_s(container, stream, time_base)
    resolve_time_mode(durations, TimeMode.cfr_fallback)
    return 1.0 / durations[0]


def _read_frames(
    container: av.container.InputContainer,
    *,
    path: Path,
    video_id: str,
    start_s: float,
    end_s: float | None,
    time_mode: TimeMode,
    fps: float | None,
) -> Iterator[ReadFrame]:
    stream = _video_stream(container, path)
    time_base = _stream_time_base(stream)
    start_pts = _start_pts(stream)
    start_offset_s = source_time_s(start_pts, time_base, 0.0)
    _seek_to(container, stream, start_s, time_base)
    state = _Pass()
    for packet in container.demux(stream):
        try:
            decoded = decode_packet(packet)
        except EOFError:
            return
        except av.error.FFmpegError:
            action = _place_corrupt(
                state,
                _corrupt_frame(
                    packet,
                    video_id=video_id,
                    time_base=time_base,
                    start_pts=start_pts,
                    start_offset_s=start_offset_s,
                    time_mode=time_mode,
                ),
                start_s,
                end_s,
            )
            if action is _STOP:
                return
            if action is not None:
                yield action
            continue
        iterator = iter(decoded)
        while True:
            try:
                frame = next(iterator)
            except StopIteration:
                break
            except EOFError:
                return
            except av.error.FFmpegError:
                action = _place_corrupt(
                    state,
                    _corrupt_frame(
                        packet,
                        video_id=video_id,
                        time_base=time_base,
                        start_pts=start_pts,
                        start_offset_s=start_offset_s,
                        time_mode=time_mode,
                    ),
                    start_s,
                    end_s,
                )
                if action is _STOP:
                    return
                if action is not None:
                    yield action
                break
            placed = _place_frame(
                state,
                frame,
                video_id=video_id,
                time_base=time_base,
                start_pts=start_pts,
                start_offset_s=start_offset_s,
                start_s=start_s,
                end_s=end_s,
                time_mode=time_mode,
                fps=fps,
            )
            if placed is _STOP:
                return
            if placed is not None:
                yield placed


def _place_corrupt(
    state: _Pass,
    frame: ReadFrame,
    start_s: float,
    end_s: float | None,
) -> ReadFrame | object | None:
    if frame.source_time_s is not None:
        if state.previous_s is not None:
            require_monotonic(state.previous_s, frame.source_time_s)
        if end_s is not None and (
            frame.source_time_s > end_s or _at_end(frame.source_time_s, end_s)
        ):
            return _STOP
        if not in_half_open(frame.source_time_s, start_s, end_s):
            state.previous_s = frame.source_time_s
            return None
        state.previous_s = frame.source_time_s
    return frame


def _place_frame(
    state: _Pass,
    frame: av.VideoFrame,
    *,
    video_id: str,
    time_base: TimeBase,
    start_pts: int,
    start_offset_s: float,
    start_s: float,
    end_s: float | None,
    time_mode: TimeMode,
    fps: float | None,
) -> ReadFrame | object | None:
    if frame.pts is None:
        raise TimebaseError(_MISSING_PTS)
    duration_ticks = _duration_ticks(frame.duration)
    timestamp = source_time_s(
        frame.pts,
        time_base,
        start_offset_s,
        start_pts=start_pts,
    )
    if state.previous_s is not None:
        require_monotonic(state.previous_s, timestamp)
    state.previous_s = timestamp
    if end_s is not None and (timestamp > end_s or _at_end(timestamp, end_s)):
        return _STOP
    if not in_half_open(timestamp, start_s, end_s):
        return None
    index = _source_frame_index(state, frame.pts, start_pts, duration_ticks)
    if index in state.seen:
        raise DuplicateFrameError(index)
    state.seen.add(index)
    state.previous_index = index
    shown = timestamp
    span_end = _pts_span_end(
        frame.pts,
        duration_ticks,
        time_base,
        start_offset_s,
        start_pts,
    )
    if time_mode is TimeMode.cfr_fallback:
        if fps is None:
            raise TimebaseError(
                "Частота кадров для cfr_fallback должна быть положительной."
            )
        # Same clock as source_time_s: offset plus index/fps, not a PTS end.
        shown = cfr_fallback_time_s(index, fps, start_offset_s)
        span_end = cfr_fallback_time_s(index + 1, fps, start_offset_s)
    return _ok_frame(
        frame,
        video_id=video_id,
        index=index,
        time_base=time_base,
        shown_s=shown,
        span_end_s=span_end,
        start_offset_s=start_offset_s,
        time_mode=time_mode,
    )


def _ok_frame(
    frame: av.VideoFrame,
    *,
    video_id: str,
    index: int,
    time_base: TimeBase,
    shown_s: float,
    span_end_s: float | None,
    start_offset_s: float,
    time_mode: TimeMode,
) -> ReadFrame:
    coded = frame.to_ndarray(format="rgb24")
    rotation = int(frame.rotation)
    image = map_coded_image(coded, rotation)
    coded_height, coded_width = int(coded.shape[0]), int(coded.shape[1])
    width, height = upright_size(coded_width, coded_height, rotation)
    if image.shape[0] != height or image.shape[1] != width or image.shape[2] != 3:
        raise TimebaseError("Поворот кадра не совпал с размером upright-кадра.")
    pts = frame.pts
    return ReadFrame(
        video_id=video_id,
        source_frame_index=index,
        source_pts=None if pts is None else int(pts),
        time_base_numerator=time_base.numerator,
        time_base_denominator=time_base.denominator,
        source_time_s=shown_s,
        span_end_s=span_end_s,
        start_offset_s=start_offset_s,
        width=width,
        height=height,
        coded_width=coded_width,
        coded_height=coded_height,
        rotation_deg=rotation,
        image=image,
        decode_status=DecodeStatus.ok,
        time_mode=time_mode,
    )


def _corrupt_frame(
    packet: av.Packet,
    *,
    video_id: str,
    time_base: TimeBase,
    start_pts: int,
    start_offset_s: float,
    time_mode: TimeMode,
) -> ReadFrame:
    pts = packet.pts if isinstance(packet.pts, int) and not isinstance(packet.pts, bool) else None
    timestamp: float | None = None
    span_end: float | None = None
    if pts is not None and pts >= start_pts:
        timestamp = source_time_s(pts, time_base, start_offset_s, start_pts=start_pts)
        duration_ticks = _duration_ticks(packet.duration)
        if duration_ticks is not None:
            span_end = source_time_s(
                pts + duration_ticks,
                time_base,
                start_offset_s,
                start_pts=start_pts,
            )
    return ReadFrame(
        video_id=video_id,
        source_frame_index=None,
        source_pts=pts,
        time_base_numerator=time_base.numerator,
        time_base_denominator=time_base.denominator,
        source_time_s=timestamp,
        span_end_s=span_end,
        start_offset_s=start_offset_s,
        width=None,
        height=None,
        coded_width=None,
        coded_height=None,
        rotation_deg=0,
        image=None,
        decode_status=DecodeStatus.corrupt,
        time_mode=time_mode,
    )


def _pts_span_end(
    pts: int,
    duration_ticks: int | None,
    time_base: TimeBase,
    start_offset_s: float,
    start_pts: int,
) -> float | None:
    if duration_ticks is None:
        return None
    return source_time_s(
        pts + duration_ticks,
        time_base,
        start_offset_s,
        start_pts=start_pts,
    )


def _source_frame_index(
    state: _Pass,
    pts: int,
    start_pts: int,
    duration_ticks: int | None,
) -> int:
    """PTS-derived index only while every duration is the same period.

    Durations 100, 100, 200 ticks are one increasing sequence. The 200-tick
    frame is previous + 1, not ``(pts - start_pts) // 200``.
    """
    if not state.variable_duration and duration_ticks is not None and duration_ticks > 0:
        if state.period_ticks is not None and duration_ticks != state.period_ticks:
            state.variable_duration = True
        else:
            delta = pts - start_pts
            if delta >= 0 and delta % duration_ticks == 0:
                if state.period_ticks is None:
                    state.period_ticks = duration_ticks
                return delta // duration_ticks
            if state.period_ticks is not None:
                state.variable_duration = True
    if state.previous_index is None:
        return 0
    return state.previous_index + 1


def _duration_ticks(duration: int | None) -> int | None:
    if duration is None:
        return None
    if isinstance(duration, bool) or not isinstance(duration, int):
        raise TimebaseError(_MISSING_PTS)
    if duration < 0:
        raise TimebaseError(
            "Отрицательная длительность кадра. Проверьте duration в контейнере."
        )
    if duration == 0:
        return None
    return duration


def _packet_durations_s(
    container: av.container.InputContainer,
    stream: av.VideoStream,
    time_base: TimeBase,
) -> list[float]:
    durations: list[float] = []
    scale = Fraction(time_base.numerator, time_base.denominator)
    for packet in container.demux(stream):
        if packet.duration is None:
            continue
        if packet.duration < 0:
            raise TimebaseError(
                "Отрицательная длительность кадра. Проверьте duration в контейнере."
            )
        if packet.duration == 0:
            continue
        durations.append(float(Fraction(packet.duration) * scale))
    return durations


def _seek_to(
    container: av.container.InputContainer,
    stream: av.VideoStream,
    start_s: float,
    time_base: TimeBase,
) -> None:
    """Seek in ``stream.time_base``. ``start_s == 0`` is already the start."""
    if start_s <= 0:
        return
    offset = seconds_to_ticks(start_s, time_base)
    container.seek(offset, backward=True, any_frame=False, stream=stream)


def _start_pts(stream: av.VideoStream) -> int:
    start = stream.start_time
    if start is None:
        return 0
    if start < 0:
        raise TimebaseError(
            "Временные метки убывают. Начало потока отрицательное."
        )
    return int(start)


def _video_stream(container: av.container.InputContainer, path: Path) -> av.VideoStream:
    streams = container.streams.video
    if not streams:
        raise CorruptMediaError(str(path))
    return streams[0]


def _stream_time_base(stream: av.VideoStream) -> TimeBase:
    raw = stream.time_base
    if raw is None or raw.numerator <= 0 or raw.denominator <= 0:
        raise TimebaseError(
            "Некорректный time base. Нужна положительная дробь числитель/знаменатель."
        )
    return TimeBase(int(raw.numerator), int(raw.denominator))


def _at_end(timestamp: float, end_s: float) -> bool:
    return math.isclose(timestamp, end_s, abs_tol=1e-6, rel_tol=0.0)


def _quarter_turns(rotation_deg: float) -> int:
    if isinstance(rotation_deg, bool) or not isinstance(rotation_deg, (int, float)):
        raise TimebaseError(_BAD_ROTATION)
    if not math.isfinite(float(rotation_deg)):
        raise TimebaseError(_BAD_ROTATION)
    rounded = round(float(rotation_deg))
    if abs(float(rotation_deg) - rounded) > 1e-3:
        raise TimebaseError(_BAD_ROTATION)
    quarter = rounded % 360
    if quarter not in (0, 90, 180, 270):
        raise TimebaseError(_BAD_ROTATION)
    return quarter


def _require_size(width: int, height: int) -> None:
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise TimebaseError("Размер кадра должен быть положительным.")
