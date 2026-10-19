"""Action intervals on the source clock.

Each interval is half-open ``[start_s, end_s)`` and lies inside one role
block. A span that crosses a boundary or sits in a gap is rejected.
Snapping to the frame step is the UI's job; any finite seconds are stored.

``revision`` is the count plus the SHA-256 of the canonical interval list.
``replace_intervals`` refuses a revision that is no longer current.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import Field, ValidationError, field_validator, model_validator

from excavation_lab import storage
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.splits import current_blocks
from excavation_lab.errors import LabError
from excavation_lab.media.timebase import (
    require_half_open_interval as require_timebase_interval,
)
from excavation_lab.media.timebase import require_monotonic
from excavation_lab.schemas import StrictModel
from excavation_lab.storage import require_half_open_interval

_INVALID_INTERVAL = "Интервал должен быть полуоткрытым: end_s больше start_s."
_OVERLAP = "Интервалы действий пересекаются. Одна секунда не относится к двум действиям."
_OUTSIDE = "Интервал не лежит ни в одном блоке роли. Он не обрезается."
_UNKNOWN_ACTION = "Действия «{name}» нет в каталоге."
_UNKNOWN_LABEL = "Интервал действия не найден."
_STALE = "Список интервалов устарел. Замена не записана."
_INVALID_LABEL = "Некорректный интервал действия."
_OPERATOR = "operator"


@dataclass(frozen=True, slots=True)
class LabelRevision:
    """Count of the canonical list plus the SHA-256 hex digest of that list."""

    count: int
    digest: str


class ActionInterval(StrictModel):
    """One labeled action. ``created_at`` is the stored ISO UTC text."""

    label_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    start_s: float
    end_s: float
    action_name: str = Field(min_length=1)
    annotator: str = Field(min_length=1)
    created_at: str

    @field_validator("created_at")
    @classmethod
    def _iso_utc(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("created_at должен быть ISO UTC") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("created_at должен быть ISO UTC")
        return value

    @model_validator(mode="after")
    def _half_open(self) -> ActionInterval:
        if not _finite(self.start_s) or not _finite(self.end_s) or self.end_s <= self.start_s:
            raise ValueError(_INVALID_INTERVAL)
        return self


class IntervalDraft(StrictModel):
    """Replacement row. Missing id, annotator, and time are filled on write."""

    start_s: float
    end_s: float
    action_name: str
    label_id: str | None = None
    annotator: str | None = None
    created_at: str | None = None


def revision(intervals: Sequence[ActionInterval]) -> LabelRevision:
    """Count plus SHA-256 of the canonical JSON list.

    The list is ordered by ``start_s``, ``end_s``, ``label_id``. The digest
    covers that JSON with sorted keys and compact separators.
    """
    ordered = sorted(intervals, key=lambda item: (item.start_s, item.end_s, item.label_id))
    payload = [
        {
            "action_name": item.action_name,
            "annotator": item.annotator,
            "created_at": item.created_at,
            "end_s": item.end_s,
            "label_id": item.label_id,
            "start_s": item.start_s,
            "video_id": item.video_id,
        }
        for item in ordered
    ]
    try:
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LabError("invalid_action_label", _INVALID_LABEL) from exc
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return LabelRevision(count=len(ordered), digest=digest)


def role_of(interval: object, blocks: Sequence[RoleBlock]) -> str | None:
    """Role of the single block that fully contains ``interval``, else ``None``."""
    start_s = getattr(interval, "start_s", None)
    end_s = getattr(interval, "end_s", None)
    if isinstance(interval, Mapping):
        start_s = interval.get("start_s")
        end_s = interval.get("end_s")
    block = _containing_block(start_s, end_s, blocks)
    if block is None:
        return None
    return block.role


def list_intervals(workspace: Path, project_id: str, video_id: str) -> list[ActionInterval]:
    """Intervals on one video, ordered by start. Half-open and non-overlapping."""
    storage.get_video(workspace, project_id, video_id)
    with storage.connect(_db(workspace, project_id)) as connection:
        return _load(connection, project_id, video_id)


def add_interval(
    workspace: Path,
    project_id: str,
    video_id: str,
    start_s: float,
    end_s: float,
    action_name: str,
    annotator: str = _OPERATOR,
) -> ActionInterval:
    """Store one interval inside a single current role block."""
    created = _build_interval(
        video_id=video_id,
        start_s=start_s,
        end_s=end_s,
        action_name=action_name,
        annotator=annotator,
        label_id=uuid4().hex,
        created_at=_utc_now(),
    )
    blocks = current_blocks(workspace, project_id, video_id)
    _require_inside_one_block(created.start_s, created.end_s, blocks)
    _require_not_past_video(
        created.end_s,
        _probed_duration_s(workspace, project_id, video_id),
    )
    with storage.connect(_db(workspace, project_id)) as connection:
        _require_known_action(connection, project_id, created.action_name)
        existing = _load(connection, project_id, video_id)
        _require_no_overlap(created.start_s, created.end_s, existing)
        _insert(connection, project_id, created)
    return created


def update_interval(
    workspace: Path,
    project_id: str,
    label_id: str,
    *,
    start_s: float | None = None,
    end_s: float | None = None,
    action_name: str | None = None,
    annotator: str | None = None,
) -> ActionInterval:
    """Replace fields of one interval. The row stays when the new span is refused."""
    _require_label_id(label_id)
    with storage.connect(_db(workspace, project_id)) as connection:
        current = _find(_load_all(connection, project_id), label_id)
    updated = _build_interval(
        video_id=current.video_id,
        start_s=current.start_s if start_s is None else start_s,
        end_s=current.end_s if end_s is None else end_s,
        action_name=current.action_name if action_name is None else action_name,
        annotator=current.annotator if annotator is None else annotator,
        label_id=current.label_id,
        created_at=current.created_at,
    )
    blocks = current_blocks(workspace, project_id, updated.video_id)
    _require_inside_one_block(updated.start_s, updated.end_s, blocks)
    _require_not_past_video(
        updated.end_s,
        _probed_duration_s(workspace, project_id, updated.video_id),
    )
    with storage.connect(_db(workspace, project_id)) as connection:
        _require_known_action(connection, project_id, updated.action_name)
        others = [
            item
            for item in _load(connection, project_id, updated.video_id)
            if item.label_id != label_id
        ]
        _require_no_overlap(updated.start_s, updated.end_s, others)
        cursor = connection.execute(
            """
            UPDATE action_labels
            SET start_s = ?, end_s = ?, action_name = ?, annotator = ?
            WHERE project_id = ? AND label_id = ?
            """,
            (
                updated.start_s,
                updated.end_s,
                updated.action_name,
                updated.annotator,
                project_id,
                label_id,
            ),
        )
        if cursor.rowcount != 1:
            raise LabError("label_not_found", _UNKNOWN_LABEL)
    return updated


def delete_interval(workspace: Path, project_id: str, label_id: str) -> None:
    """Remove one interval. A missing id is ``label_not_found``."""
    _require_label_id(label_id)
    with storage.connect(_db(workspace, project_id)) as connection:
        cursor = connection.execute(
            """
            DELETE FROM action_labels
            WHERE project_id = ? AND label_id = ?
            """,
            (project_id, label_id),
        )
        if cursor.rowcount != 1:
            raise LabError("label_not_found", _UNKNOWN_LABEL)


def replace_intervals(
    workspace: Path,
    project_id: str,
    video_id: str,
    items: Sequence[ActionInterval | Mapping[str, object]],
    *,
    expected_revision: LabelRevision,
) -> list[ActionInterval]:
    """Replace every interval on ``video_id`` when ``expected_revision`` is current.

    A stale revision writes nothing. The new rows are checked as a set:
    half-open, non-overlapping, each inside one role block.
    """
    if not isinstance(expected_revision, LabelRevision):
        raise LabError("stale_labels", _STALE)
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise LabError("invalid_action_label", _INVALID_LABEL)
    storage.get_video(workspace, project_id, video_id)
    blocks = current_blocks(workspace, project_id, video_id)
    duration_s = _probed_duration_s(workspace, project_id, video_id)
    with storage.connect(_db(workspace, project_id)) as connection:
        current = _load(connection, project_id, video_id)
        if revision(current) != expected_revision:
            raise LabError("stale_labels", _STALE)
        built = [_draft_interval(video_id, item) for item in items]
        _require_unique_ids(built)
        known = _action_names(connection, project_id)
        for item in built:
            if item.action_name not in known:
                raise LabError("unknown_action", _UNKNOWN_ACTION.format(name=item.action_name))
            _require_inside_one_block(item.start_s, item.end_s, blocks)
            _require_not_past_video(item.end_s, duration_s)
        _require_set_disjoint(built)
        connection.execute(
            """
            DELETE FROM action_labels
            WHERE project_id = ? AND video_id = ?
            """,
            (project_id, video_id),
        )
        for item in built:
            _insert(connection, project_id, item)
        return _load(connection, project_id, video_id)


def _db(workspace: Path, project_id: str) -> Path:
    return storage.project_sqlite(workspace, project_id)


def _load(
    connection: sqlite3.Connection,
    project_id: str,
    video_id: str,
) -> list[ActionInterval]:
    rows = connection.execute(
        """
        SELECT label_id, video_id, start_s, end_s, action_name, annotator, created_at
        FROM action_labels
        WHERE project_id = ? AND video_id = ?
        ORDER BY start_s, label_id
        """,
        (project_id, video_id),
    ).fetchall()
    return [_interval_row(row) for row in rows]


def _load_all(connection: sqlite3.Connection, project_id: str) -> list[ActionInterval]:
    rows = connection.execute(
        """
        SELECT label_id, video_id, start_s, end_s, action_name, annotator, created_at
        FROM action_labels
        WHERE project_id = ?
        ORDER BY video_id, start_s, label_id
        """,
        (project_id,),
    ).fetchall()
    return [_interval_row(row) for row in rows]


def _interval_row(row: sqlite3.Row) -> ActionInterval:
    try:
        return ActionInterval(
            label_id=row["label_id"],
            video_id=row["video_id"],
            start_s=row["start_s"],
            end_s=row["end_s"],
            action_name=row["action_name"],
            annotator=row["annotator"],
            created_at=row["created_at"],
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise LabError("invalid_action_label", _INVALID_LABEL) from exc


def _insert(
    connection: sqlite3.Connection,
    project_id: str,
    item: ActionInterval,
) -> None:
    connection.execute(
        """
        INSERT INTO action_labels (
            label_id, project_id, video_id, start_s, end_s,
            action_name, annotator, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item.label_id,
            project_id,
            item.video_id,
            item.start_s,
            item.end_s,
            item.action_name,
            item.annotator,
            item.created_at,
        ),
    )


def _build_interval(
    *,
    video_id: str,
    start_s: object,
    end_s: object,
    action_name: object,
    annotator: object,
    label_id: str,
    created_at: str,
) -> ActionInterval:
    start, end = _finite_interval(start_s, end_s)
    token = _require_action_name(action_name)
    person = _require_annotator(annotator)
    _require_label_id(label_id)
    try:
        return ActionInterval(
            label_id=label_id,
            video_id=video_id,
            start_s=start,
            end_s=end,
            action_name=token,
            annotator=person,
            created_at=created_at,
        )
    except ValidationError as exc:
        raise LabError("invalid_action_label", _INVALID_LABEL) from exc


def _draft_interval(
    video_id: str,
    item: ActionInterval | Mapping[str, object],
) -> ActionInterval:
    if isinstance(item, ActionInterval):
        if item.video_id != video_id:
            raise LabError("invalid_action_label", _INVALID_LABEL)
        return item
    if not isinstance(item, Mapping):
        raise LabError("invalid_action_label", _INVALID_LABEL)
    try:
        draft = IntervalDraft.model_validate(dict(item))
    except ValidationError as exc:
        raise LabError("invalid_action_label", _INVALID_LABEL) from exc
    return _build_interval(
        video_id=video_id,
        start_s=draft.start_s,
        end_s=draft.end_s,
        action_name=draft.action_name,
        annotator=_OPERATOR if draft.annotator is None else draft.annotator,
        label_id=uuid4().hex if draft.label_id is None else draft.label_id,
        created_at=_utc_now() if draft.created_at is None else draft.created_at,
    )


def _finite_interval(start_s: object, end_s: object) -> tuple[float, float]:
    if (
        isinstance(start_s, bool)
        or isinstance(end_s, bool)
        or not isinstance(start_s, (int, float))
        or not isinstance(end_s, (int, float))
    ):
        raise LabError("invalid_interval", _INVALID_INTERVAL)
    start = float(start_s)
    end = float(end_s)
    require_half_open_interval(start, end)
    require_timebase_interval(start, end)
    require_monotonic(start, end)
    return start, end


def _require_action_name(name: object) -> str:
    if not isinstance(name, str) or not name.strip() or "\x00" in name:
        raise LabError("invalid_action_name", "Имя действия — английский токен.")
    return name.strip()


def _require_annotator(annotator: object) -> str:
    if not isinstance(annotator, str) or "\x00" in annotator or not annotator.strip():
        raise LabError("invalid_action_label", "Нужно имя разметчика.")
    return annotator.strip()


def _require_label_id(label_id: object) -> None:
    if (
        not isinstance(label_id, str)
        or not label_id
        or label_id in {".", ".."}
        or "/" in label_id
        or "\\" in label_id
        or "\x00" in label_id
    ):
        raise LabError("invalid_action_label", _INVALID_LABEL)


def _require_known_action(
    connection: sqlite3.Connection,
    project_id: str,
    name: str,
) -> None:
    if name not in _action_names(connection, project_id):
        raise LabError("unknown_action", _UNKNOWN_ACTION.format(name=name))


def _action_names(connection: sqlite3.Connection, project_id: str) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM action_classes WHERE project_id = ?",
        (project_id,),
    ).fetchall()
    return {str(row["name"]) for row in rows}


def _require_no_overlap(
    start_s: float,
    end_s: float,
    others: Sequence[ActionInterval],
) -> None:
    for item in others:
        if start_s < item.end_s and item.start_s < end_s:
            raise LabError("action_overlap", _OVERLAP)


def _require_set_disjoint(items: Sequence[ActionInterval]) -> None:
    ordered = sorted(items, key=lambda item: (item.start_s, item.end_s, item.label_id))
    for index, earlier in enumerate(ordered):
        for later in ordered[index + 1 :]:
            if later.start_s >= earlier.end_s:
                break
            if earlier.start_s < later.end_s and later.start_s < earlier.end_s:
                raise LabError("action_overlap", _OVERLAP)


def _require_unique_ids(items: Sequence[ActionInterval]) -> None:
    ids = [item.label_id for item in items]
    if len(ids) != len(set(ids)):
        raise LabError("invalid_action_label", _INVALID_LABEL)


def _find(items: Sequence[ActionInterval], label_id: str) -> ActionInterval:
    matches = [item for item in items if item.label_id == label_id]
    if len(matches) != 1:
        raise LabError("label_not_found", _UNKNOWN_LABEL)
    return matches[0]


def _containing_block(
    start_s: object,
    end_s: object,
    blocks: Sequence[RoleBlock],
) -> RoleBlock | None:
    if not _finite(start_s) or not _finite(end_s):
        return None
    start = float(start_s)
    end = float(end_s)
    containing = [
        block
        for block in blocks
        if block.start_s <= start and end <= block.end_s
    ]
    if len(containing) != 1:
        return None
    return containing[0]


def _probed_duration_s(workspace: Path, project_id: str, video_id: str) -> float | None:
    duration_s = storage.get_video(workspace, project_id, video_id)["duration_s"]
    if duration_s is None or isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
        return None
    if not math.isfinite(float(duration_s)):
        return None
    return float(duration_s)


def _require_not_past_video(end_s: float, duration_s: float | None) -> None:
    """Refuse ``end_s`` past the probed video length.

    A null ``duration_s`` is not a length. No duration is invented, and the
    interval stays subject only to the block rules.
    """
    if duration_s is None:
        return
    if end_s > duration_s:
        raise LabError(
            "interval_past_end",
            f"Конец интервала позже конца видео ({_format_seconds(duration_s)} с).",
        )


def _require_inside_one_block(
    start_s: float,
    end_s: float,
    blocks: Sequence[RoleBlock],
) -> RoleBlock:
    block = _containing_block(start_s, end_s, blocks)
    if block is not None:
        return block
    overlapping = [
        item
        for item in blocks
        if item.start_s < end_s and start_s < item.end_s
    ]
    if not overlapping:
        raise LabError("action_outside_block", _OUTSIDE)
    boundary = _first_boundary(start_s, end_s, blocks)
    raise LabError(
        "action_crosses_block",
        f"Интервал пересекает границу блока {_format_seconds(boundary)} с. Он не обрезается.",
    )


def _first_boundary(start_s: float, end_s: float, blocks: Sequence[RoleBlock]) -> float:
    found: list[float] = []
    for block in blocks:
        if start_s < block.start_s < end_s:
            found.append(float(block.start_s))
        if start_s < block.end_s < end_s:
            found.append(float(block.end_s))
    if found:
        return min(found)
    return float(min(blocks, key=lambda item: item.start_s).end_s)


def _format_seconds(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
