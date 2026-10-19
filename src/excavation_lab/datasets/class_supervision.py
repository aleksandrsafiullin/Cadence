"""Per-class supervision for one labeled frame.

A missing box is a negative only when the frame was reviewed at or after
that class was added. A later class is unsupervised. ``reviewed_at is
None`` is not earlier than every ``added_at``. Frame-level ``is_negative``
and ``label_state`` stay as ``validate_frame`` set them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from excavation_lab.datasets.catalog import StoredCatalog
from excavation_lab.datasets.labels import LabelRecord
from excavation_lab.schemas import ReviewStatus


@dataclass(frozen=True, slots=True)
class ClassSupervision:
    """Every catalog name is in exactly one of these sets."""

    positives: frozenset[str]
    negatives: frozenset[str]
    unsupervised: frozenset[str]


def class_supervision(record: LabelRecord, catalog: StoredCatalog) -> ClassSupervision:
    """Split ``catalog`` names for ``record``.

    Supervision needs a reviewed trainable frame and both timestamps, with
    ``reviewed_at >= added_at``. A box of that class is a positive. No box
    is a negative. Unreviewed, excluded, and unlabelable frames supervise
    nothing. This does not rewrite boxes.
    """
    reviewed_at = _reviewed_instant(record.reviewed_at)
    present = {box.class_name for box in record.boxes}
    can_supervise = (
        reviewed_at is not None
        and record.trainable
        and record.review_status is ReviewStatus.reviewed
    )
    positives: set[str] = set()
    negatives: set[str] = set()
    unsupervised: set[str] = set()
    for item in catalog.classes:
        if can_supervise and reviewed_at is not None and reviewed_at >= item.added_at:
            if item.name in present:
                positives.add(item.name)
            else:
                negatives.add(item.name)
        else:
            unsupervised.add(item.name)
    return ClassSupervision(
        positives=frozenset(positives),
        negatives=frozenset(negatives),
        unsupervised=frozenset(unsupervised),
    )


def _reviewed_instant(reviewed_at: datetime | None) -> datetime | None:
    """Return the review instant. ``None`` stays missing.

    A naive datetime is refused. It is not treated as earlier than every
    ``added_at``.
    """
    if reviewed_at is None:
        return None
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise ValueError("reviewed_at must be timezone-aware")
    return reviewed_at
