"""Annotation rules for one catalog.

A reviewed frame with zero boxes is ``reviewed_empty`` (a negative).
An unreviewed frame is stored and is not a negative.
An excluded frame is stored and is not trainable.
``visibility=unlabelable`` forces ``review_status=excluded``.

Class ids come from the catalog passed into ``validate_frame``. Id 0 is
a class, not background. A note never adds a box. ``iscrowd`` other than
0 is refused and is not rewritten as ``partial``. Nested boxes are kept.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import Field, model_validator

from excavation_lab.datasets.catalog import StoredCatalog
from excavation_lab.errors import LabError, UnknownCategoryError, UnsupportedCrowdError
from excavation_lab.schemas import (
    AnnotatedFrame,
    AnnotationBox,
    CatalogClass,
    CatalogConfig,
    ReviewStatus,
    StrictModel,
    Visibility,
)


class LabelState(StrEnum):
    """Derived frame state. This is not a fourth ``review_status``."""

    reviewed_empty = "reviewed_empty"
    reviewed = "reviewed"
    unreviewed = "unreviewed"
    excluded = "excluded"


class BoxDraft(StrictModel):
    """One incoming box. ``category_id`` is not the internal class id.

    ``class_id``, when set, must already be that catalog's id for the name.
    """

    category_name: str
    category_id: int | None = None
    class_id: int | None = None
    visibility: Visibility = Visibility.clear
    truncated: bool = False
    instance_local_id: str | None = Field(default=None, min_length=1)
    x1: float
    y1: float
    x2: float
    y2: float
    note: str | None = None
    iscrowd: int = 0


class FrameDraft(StrictModel):
    """One incoming frame, before review rules are applied."""

    video_id: str = Field(min_length=1)
    source_frame_index: int = Field(ge=0)
    review_status: ReviewStatus
    visibility: Visibility | None = None
    annotator: str | None = Field(default=None, min_length=1)
    reviewed_at: datetime | None = None
    note: str | None = None
    boxes: list[BoxDraft] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Decision:
    review_status: ReviewStatus
    label_state: LabelState
    trainable: bool
    is_negative: bool


def _decision(
    review_status: ReviewStatus,
    visibility: Visibility | None,
    boxes: Sequence[AnnotationBox],
) -> _Decision:
    """Detector-training role of one stored frame.

    Unlabelable forces excluded. Reviewed with no boxes is a negative.
    Unreviewed is stored and is not a negative. Excluded is stored and
    is not trainable. Ignore-region loss is not applied.
    """
    unlabelable = visibility is Visibility.unlabelable or any(
        box.visibility is Visibility.unlabelable for box in boxes
    )
    if unlabelable or review_status is ReviewStatus.excluded:
        return _Decision(ReviewStatus.excluded, LabelState.excluded, False, False)
    if review_status is ReviewStatus.unreviewed:
        return _Decision(ReviewStatus.unreviewed, LabelState.unreviewed, False, False)
    if not boxes:
        return _Decision(ReviewStatus.reviewed, LabelState.reviewed_empty, True, True)
    return _Decision(ReviewStatus.reviewed, LabelState.reviewed, True, False)


class LabelRecord(AnnotatedFrame):
    """Stored frame label. Derived flags follow the review rules."""

    label_state: LabelState = LabelState.unreviewed
    trainable: bool = False
    is_negative: bool = False

    @model_validator(mode="after")
    def enforce_annotation_rules(self) -> Self:
        # Mutate and return self. ``__init__`` drops a replacement instance.
        decision = _decision(self.review_status, self.visibility, self.boxes)
        self.review_status = decision.review_status
        self.label_state = decision.label_state
        self.trainable = decision.trainable
        self.is_negative = decision.is_negative
        return self


def lookup_class(
    category_name: str,
    catalog: StoredCatalog | CatalogConfig,
    *,
    frame_id: str | None = None,
) -> CatalogClass:
    """Return the class named ``category_name`` in ``catalog``.

    Matching is exact. An unknown name, including ``dump`` or ``"0"``,
    raises ``UnknownCategoryError`` and is not snapped to a nearby class.
    """
    if isinstance(category_name, bool) or not isinstance(category_name, str):
        raise UnknownCategoryError(str(category_name), frame_id)
    name = category_name.strip()
    for item in catalog.classes:
        if item.name == name:
            return item
    raise UnknownCategoryError(name, frame_id)


def _reject_crowd(iscrowd: object, frame_id: str) -> None:
    if iscrowd == 0 and not isinstance(iscrowd, bool):
        return
    shown = (
        iscrowd if isinstance(iscrowd, int) and not isinstance(iscrowd, bool) else None
    )
    raise UnsupportedCrowdError(frame_id, shown)


def _accept_box(
    box: BoxDraft,
    frame_id: str,
    catalog: StoredCatalog | CatalogConfig,
) -> AnnotationBox:
    """One visible-extent box.

    ``box.category_id`` is not read. A COCO id of 0 does not select a class
    and does not mean background. The stored id is the catalog id for the
    name. A different ``class_id`` is rejected and the name stays.
    """
    # Crowd is not occlusion. Do not rewrite it as visibility=partial.
    _reject_crowd(box.iscrowd, frame_id)
    found = lookup_class(box.category_name, catalog, frame_id=frame_id)
    if box.class_id is not None and box.class_id != found.id:
        raise LabError(
            "class_id_mismatch",
            f"Кадр {frame_id}: класс «{found.name}» хранится с id {found.id}, "
            f"не {box.class_id}. Имя не подменяется.",
        )
    return AnnotationBox(
        class_name=found.name,
        class_id=found.id,
        visibility=box.visibility,
        truncated=box.truncated,
        instance_local_id=box.instance_local_id,
        x1=box.x1,
        y1=box.y1,
        x2=box.x2,
        y2=box.y2,
        note=box.note,
    )


def validate_frame(
    frame: FrameDraft | Mapping[str, Any],
    catalog: StoredCatalog | CatalogConfig,
) -> LabelRecord:
    """Accept one frame under the review rules and ``catalog``.

    The returned record is the stored label. Notes are copied and never
    add a box, including when the note says the bucket is hidden.
    Every accepted box is kept. The catalog is an argument.
    """
    draft = frame if isinstance(frame, FrameDraft) else FrameDraft.model_validate(frame)
    frame_id = f"{draft.video_id}:{draft.source_frame_index}"
    boxes = [_accept_box(box, frame_id, catalog) for box in draft.boxes]
    return LabelRecord(
        video_id=draft.video_id,
        source_frame_index=draft.source_frame_index,
        review_status=draft.review_status,
        visibility=draft.visibility,
        annotator=draft.annotator,
        reviewed_at=draft.reviewed_at,
        note=draft.note,
        boxes=boxes,
    )
