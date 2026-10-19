"""Operator role blocks. A second belongs to the block that contains it.

``start_s`` and ``end_s`` are seconds on the source clock, including
``start_offset_s``. They are not preview seconds. A player time is converted
before these values are stored.

Intervals are half-open ``[start_s, end_s)``. A second in no block has no
role. Blocks do not use a guard interval.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Literal, Protocol

from pydantic import Field

from excavation_lab.errors import LabError
from excavation_lab.media.timebase import (
    require_half_open_interval as require_timebase_interval,
)
from excavation_lab.media.timebase import (
    require_monotonic,
)
from excavation_lab.schemas import StrictModel
from excavation_lab.storage import require_half_open_interval

RoleName = Literal["train", "validation", "test"]
_ROLES = frozenset({"train", "validation", "test"})
_OVERLAP = "Блоки пересекаются. Одна секунда не относится к двум ролям."
_INVALID_ROLE = "Роль блока должна быть train, validation или test."
# Stated guard between blocks of different roles. A shorter gap is reported, not refused.
GUARD_GAP_S = 10.0
_GAP_TOLERANCE_S = 1e-6


class RoleBlock(StrictModel):
    start_s: float = Field(ge=0)
    end_s: float
    role: RoleName


class _BlockLike(Protocol):
    start_s: object
    end_s: object
    role: object


def validate_role_blocks(blocks: Sequence[_BlockLike]) -> list[RoleBlock]:
    """Reject a backwards interval or any overlap. Touching endpoints is allowed."""
    built: list[RoleBlock] = []
    for block in blocks:
        start_s = block.start_s
        end_s = block.end_s
        role = block.role
        require_half_open_interval(start_s, end_s)
        start = float(start_s)
        end = float(end_s)
        require_timebase_interval(start, end)
        require_monotonic(start, end)
        if role not in _ROLES:
            raise LabError("invalid_role", _INVALID_ROLE)
        built.append(RoleBlock(start_s=start, end_s=end, role=role))
    ordered = sorted(built, key=lambda item: (item.start_s, item.end_s, item.role))
    for index, earlier in enumerate(ordered):
        for later in ordered[index + 1 :]:
            if later.start_s >= earlier.end_s:
                break
            if earlier.start_s < later.end_s and later.start_s < earlier.end_s:
                raise LabError("block_overlap", _OVERLAP)
    return ordered


class GuardGapWarning(StrictModel):
    """Two adjacent blocks of different roles closer than ``GUARD_GAP_S``.

    The freeze is not refused: the operator decides. The pair is reported so
    that an operation crossing the boundary is a visible choice.
    """

    earlier: RoleBlock
    later: RoleBlock
    gap_s: float = Field(ge=0)


def guard_gap_warnings(
    blocks: Sequence[RoleBlock],
    *,
    min_gap_s: float = GUARD_GAP_S,
) -> list[GuardGapWarning]:
    """Adjacent pairs with different roles whose unassigned gap is under ``min_gap_s``.

    Touching blocks of the same role are one block for this purpose and are
    not reported. The input is ordered by start; overlaps were rejected before.
    """
    ordered = sorted(blocks, key=lambda item: (item.start_s, item.end_s, item.role))
    found: list[GuardGapWarning] = []
    for earlier, later in pairwise(ordered):
        if earlier.role == later.role:
            continue
        gap = max(0.0, later.start_s - earlier.end_s)
        if gap + _GAP_TOLERANCE_S < min_gap_s:
            found.append(GuardGapWarning(earlier=earlier, later=later, gap_s=gap))
    return found


def assignment_at(blocks: Sequence[RoleBlock], time_s: float) -> str | None:
    """Role of the block that contains ``time_s``, or ``None``."""
    if isinstance(time_s, bool) or not isinstance(time_s, (int, float)):
        return None
    if not math.isfinite(time_s):
        return None
    found: list[str] = []
    for block in blocks:
        if block.role not in _ROLES:
            continue
        if block.start_s <= time_s < block.end_s:
            found.append(block.role)
    if len(found) > 1:
        raise LabError("block_overlap", _OVERLAP)
    if not found:
        return None
    return found[0]
