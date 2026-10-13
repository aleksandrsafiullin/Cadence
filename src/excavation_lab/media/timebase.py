"""Source clock: PTS, time base, and start offset.

``source_time_s`` is seconds from the start of the source video. It already
includes the start offset. ``cfr_fallback`` is ``start_offset_s + frame_index / fps``
and is used only after a constant-frame-rate check. The default mode is ``pts``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from excavation_lab.errors import TimebaseError

# Half-open edges. A frame 1 µs off an exact boundary still belongs to it.
_BOUNDARY_S = 1e-6
# Durations within 1 ms are one constant rate. 0.2 s then 0.5 s is not.
_CFR_TOLERANCE_S = 1e-3

_DECREASING = (
    "Временные метки убывают. "
    "Шкала исходного видео не перезапускается и не идёт назад."
)
_NEGATIVE_DURATION = (
    "Отрицательная длительность: конец интервала раньше начала. "
    "Интервал полуоткрытый [начало, конец)."
)
_NEGATIVE_FRAME = (
    "Отрицательная длительность кадра. Проверьте duration в контейнере."
)
_BAD_TIME_BASE = (
    "Некорректный time base. Нужна положительная дробь числитель/знаменатель."
)
_CFR_REJECTED = (
    "Переменный шаг кадров нельзя читать как номер_кадра / fps. "
    "Для этой последовательности время остаётся по PTS."
)


class TimeMode(StrEnum):
    """``pts`` is the clock. ``cfr_fallback`` is frame_index / fps."""

    pts = "pts"
    cfr_fallback = "cfr_fallback"


@dataclass(frozen=True, slots=True)
class TimeBase:
    """Rational ``numerator/denominator`` seconds per PTS tick."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.numerator, bool)
            or isinstance(self.denominator, bool)
            or not isinstance(self.numerator, int)
            or not isinstance(self.denominator, int)
            or self.numerator <= 0
            or self.denominator <= 0
        ):
            raise TimebaseError(_BAD_TIME_BASE)

    @property
    def text(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    @classmethod
    def parse(cls, text: str) -> TimeBase:
        if not isinstance(text, str) or text.count("/") != 1:
            raise TimebaseError(_BAD_TIME_BASE)
        numerator_text, denominator_text = text.split("/", 1)
        if (
            not numerator_text.isdigit()
            or not denominator_text.isdigit()
            or numerator_text.startswith("0")
            or denominator_text.startswith("0")
        ):
            # "0" is not a positive integer. "10" is allowed; "01" is not.
            raise TimebaseError(_BAD_TIME_BASE)
        return cls(int(numerator_text), int(denominator_text))


def source_time_s(
    pts: int,
    time_base: TimeBase,
    start_offset_s: float,
    *,
    start_pts: int = 0,
) -> float:
    """Seconds from the start of the source, including ``start_offset_s``.

    ``(pts - start_pts) * time_base + start_offset_s``.

    The offset is added. A fragment whose PTS sits on the 300 s mark stays
    at 300 s. Subtracting the offset would restart that fragment at 0.
    """
    _require_tick(pts, "PTS")
    _require_tick(start_pts, "Начальный PTS")
    if isinstance(start_offset_s, bool) or not isinstance(start_offset_s, (int, float)):
        raise TimebaseError("Смещение начала видео не является конечным числом.")
    if not math.isfinite(float(start_offset_s)):
        raise TimebaseError("Смещение начала видео не является конечным числом.")
    delta = pts - start_pts
    if delta < 0:
        raise TimebaseError(_DECREASING)
    total = Fraction(delta) * Fraction(time_base.numerator, time_base.denominator)
    total += Fraction(float(start_offset_s)).limit_denominator(1_000_000_000)
    if total < 0:
        raise TimebaseError(
            "Отрицательная метка времени. "
            "Фрагмент не обнуляется и не уходит ниже нуля."
        )
    return float(total)


def require_half_open_interval(start_s: float, end_s: float | None) -> None:
    """Reject a negative duration. ``end_s == start_s`` is an empty interval."""
    if isinstance(start_s, bool) or not isinstance(start_s, (int, float)):
        raise TimebaseError("Начало интервала должно быть конечным и неотрицательным.")
    if not math.isfinite(float(start_s)) or float(start_s) < 0:
        raise TimebaseError("Начало интервала должно быть конечным и неотрицательным.")
    if end_s is None:
        return
    if isinstance(end_s, bool) or not isinstance(end_s, (int, float)):
        raise TimebaseError(_NEGATIVE_DURATION)
    if not math.isfinite(float(end_s)):
        raise TimebaseError(_NEGATIVE_DURATION)
    if float(end_s) < float(start_s):
        raise TimebaseError(_NEGATIVE_DURATION)


def require_monotonic(previous_s: float, current_s: float) -> None:
    """Raise when a later timestamp is strictly earlier than the previous one."""
    if current_s < previous_s:
        raise TimebaseError(_DECREASING)


def require_non_negative_duration(duration_s: float) -> None:
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
        raise TimebaseError(_NEGATIVE_FRAME)
    if not math.isfinite(float(duration_s)) or float(duration_s) < 0:
        raise TimebaseError(_NEGATIVE_FRAME)


def in_half_open(source_time_s: float, start_s: float, end_s: float | None) -> bool:
    """True when ``start_s <= source_time_s < end_s``."""
    if _same_instant(source_time_s, start_s) or source_time_s >= start_s:
        if end_s is None:
            return True
        if _same_instant(source_time_s, end_s):
            return False
        return source_time_s < end_s
    return False


def is_constant_frame_rate(
    durations_s: Sequence[float],
    *,
    tolerance_s: float = _CFR_TOLERANCE_S,
) -> bool:
    """True only when every duration is the same positive length.

    A negative duration raises ``TimebaseError``. An empty sequence is not
    constant: nothing has marked the file CFR, so the clock stays on PTS.
    """
    if not durations_s:
        return False
    first: float | None = None
    for duration in durations_s:
        require_non_negative_duration(duration)
        if duration == 0:
            return False
        if first is None:
            first = float(duration)
            continue
        if abs(float(duration) - first) > tolerance_s:
            return False
    return first is not None


def resolve_time_mode(
    durations_s: Sequence[float],
    mode: TimeMode = TimeMode.pts,
) -> TimeMode:
    """Return ``pts`` unless ``cfr_fallback`` was asked for and the file is CFR.

    Asking for the fallback on a variable sequence (0.2 s then 0.5 s) raises.
    """
    chosen = _coerce_mode(mode)
    for duration in durations_s:
        require_non_negative_duration(duration)
    if chosen is TimeMode.pts:
        return TimeMode.pts
    if not is_constant_frame_rate(durations_s):
        raise TimebaseError(_CFR_REJECTED)
    return TimeMode.cfr_fallback


def cfr_fallback_time_s(
    frame_index: int,
    fps: float,
    start_offset_s: float = 0.0,
) -> float:
    """``start_offset_s + frame_index / fps``.

    Valid only after :func:`resolve_time_mode` allows the fallback. The offset
    stays on the clock: a fragment at 300 s does not restart at 0.
    """
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise TimebaseError(
            "Номер кадра для cfr_fallback должен быть неотрицательным целым."
        )
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise TimebaseError("Частота кадров для cfr_fallback должна быть положительной.")
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise TimebaseError("Частота кадров для cfr_fallback должна быть положительной.")
    if isinstance(start_offset_s, bool) or not isinstance(start_offset_s, (int, float)):
        raise TimebaseError("Смещение начала видео не является конечным числом.")
    if not math.isfinite(float(start_offset_s)):
        raise TimebaseError("Смещение начала видео не является конечным числом.")
    total = Fraction(float(start_offset_s)).limit_denominator(1_000_000_000)
    total += Fraction(frame_index) / Fraction(float(fps)).limit_denominator(1_000_000_000)
    if total < 0:
        raise TimebaseError(
            "Отрицательная метка времени. "
            "Фрагмент не обнуляется и не уходит ниже нуля."
        )
    return float(total)


def frame_time_s(
    *,
    pts: int,
    time_base: TimeBase,
    start_offset_s: float,
    durations_s: Sequence[float],
    frame_index: int = 0,
    fps: float | None = None,
    start_pts: int = 0,
    mode: TimeMode = TimeMode.pts,
) -> float:
    """Clock value for one frame. Default mode is PTS, offset included."""
    chosen = resolve_time_mode(durations_s, mode)
    if chosen is TimeMode.cfr_fallback:
        if fps is None:
            raise TimebaseError(
                "Частота кадров для cfr_fallback должна быть положительной."
            )
        return cfr_fallback_time_s(frame_index, fps, start_offset_s)
    return source_time_s(pts, time_base, start_offset_s, start_pts=start_pts)


def seconds_to_ticks(seconds: float, time_base: TimeBase) -> int:
    """Absolute stream timestamp for ``seconds``, in ``time_base`` ticks."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise TimebaseError("Метка времени для seek не является конечным числом.")
    if not math.isfinite(float(seconds)):
        raise TimebaseError("Метка времени для seek не является конечным числом.")
    ticks = Fraction(float(seconds)).limit_denominator(10_000_000)
    ticks *= Fraction(time_base.denominator, time_base.numerator)
    return round(ticks)


def _require_tick(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TimebaseError(f"{label} должен быть целым числом тиков time base.")


def _coerce_mode(mode: TimeMode) -> TimeMode:
    try:
        return TimeMode(mode)
    except ValueError as exc:
        raise TimebaseError(
            "Неизвестный режим времени. Допустимы pts и cfr_fallback."
        ) from exc


def _same_instant(left: float, right: float) -> bool:
    return math.isclose(left, right, abs_tol=_BOUNDARY_S, rel_tol=0.0)
