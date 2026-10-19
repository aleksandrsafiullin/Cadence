"""Per-class negatives. Synthetic frames, not pit-video accuracy."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta, timezone

import pytest

from excavation_lab.datasets.catalog import (
    StoredCatalog,
    StoredCatalogClass,
    catalog_from_seed,
)
from excavation_lab.datasets.class_supervision import (
    ClassSupervision,
    class_supervision,
)
from excavation_lab.datasets.labels import LabelState, validate_frame
from excavation_lab.schemas import ReviewStatus

_SEED = ("excavator_body", "bucket", "haul_truck", "truck_bed")


def test_later_class_is_unsupervised_and_the_old_box_stays() -> None:
    added = datetime(2026, 1, 1, tzinfo=UTC)
    reviewed = datetime(2026, 6, 1, tzinfo=UTC)
    shovel_at = datetime(2026, 9, 1, tzinfo=UTC)
    old = _catalog(added)
    record = validate_frame(
        _frame(reviewed_at=reviewed, boxes=[_box("excavator_body")]),
        old,
    )
    before = record.model_dump()
    boxes = tuple(record.boxes)
    catalog = _catalog(added, shovel_at=shovel_at)
    result = class_supervision(record, catalog)
    assert record.model_dump() == before
    assert tuple(record.boxes) == boxes
    assert [box.class_name for box in record.boxes] == ["excavator_body"]
    assert result.positives == frozenset({"excavator_body"})
    assert result.negatives == frozenset({"bucket", "haul_truck", "truck_bed"})
    assert result.unsupervised == frozenset({"shovel"})
    assert record.is_negative is False
    _assert_partition(result, catalog)


def test_reviewed_empty_does_not_make_a_later_class_a_negative() -> None:
    added = datetime(2026, 1, 1, tzinfo=UTC)
    reviewed = datetime(2026, 6, 1, tzinfo=UTC)
    catalog = _catalog(added, shovel_at=datetime(2026, 9, 1, tzinfo=UTC))
    record = validate_frame(_frame(reviewed_at=reviewed, boxes=[]), catalog)
    result = class_supervision(record, catalog)
    assert record.label_state is LabelState.reviewed_empty
    assert record.is_negative is True
    assert record.boxes == []
    assert result.positives == frozenset()
    assert result.negatives == frozenset(_SEED)
    assert result.unsupervised == frozenset({"shovel"})
    _assert_partition(result, catalog)


def test_equal_timestamp_with_no_box_is_a_negative() -> None:
    instant = datetime(2026, 4, 1, 8, 0, tzinfo=UTC)
    catalog = _catalog(instant, shovel_at=instant)
    record = validate_frame(_frame(reviewed_at=instant, boxes=[]), catalog)
    result = class_supervision(record, catalog)
    assert record.reviewed_at == catalog.classes[-1].added_at
    assert "shovel" in result.negatives
    assert result.negatives == frozenset({*_SEED, "shovel"})
    assert result.positives == frozenset()
    assert result.unsupervised == frozenset()
    _assert_partition(result, catalog)


def test_offsets_compare_as_instants() -> None:
    added = datetime(2026, 6, 1, 12, 0, tzinfo=timezone(timedelta(hours=5)))
    reviewed = datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
    catalog = _catalog(added, shovel_at=added)
    assert catalog.classes[-1].added_at == datetime(2026, 6, 1, 7, 0, tzinfo=UTC)
    record = validate_frame(_frame(reviewed_at=reviewed, boxes=[]), catalog)
    result = class_supervision(record, catalog)
    assert "shovel" in result.negatives
    assert result.unsupervised == frozenset()
    _assert_partition(result, catalog)


def test_shovel_box_is_a_positive_when_the_review_is_not_before_added_at() -> None:
    added = datetime(2026, 1, 1, tzinfo=UTC)
    shovel_at = datetime(2026, 3, 1, tzinfo=UTC)
    reviewed = datetime(2026, 5, 1, tzinfo=UTC)
    catalog = _catalog(added, shovel_at=shovel_at)
    record = validate_frame(_frame(reviewed_at=reviewed, boxes=[_box("shovel")]), catalog)
    assert record.boxes[0].class_name == "shovel"
    assert record.boxes[0].class_id == 4
    result = class_supervision(record, catalog)
    assert result.positives == frozenset({"shovel"})
    assert "shovel" not in result.negatives
    assert result.negatives == frozenset(_SEED)
    assert result.unsupervised == frozenset()
    _assert_partition(result, catalog)


def test_missing_reviewed_at_leaves_every_class_unsupervised() -> None:
    added = datetime(2020, 1, 1, tzinfo=UTC)
    catalog = _catalog(added, shovel_at=added)
    names = frozenset({*_SEED, "shovel"})
    empty = validate_frame(_frame(review_status="reviewed", boxes=[]), catalog)
    assert empty.reviewed_at is None
    assert empty.label_state is LabelState.reviewed_empty
    assert empty.is_negative is True
    empty_result = class_supervision(empty, catalog)
    assert empty_result.unsupervised == names
    assert empty_result.negatives == frozenset()
    assert empty_result.positives == frozenset()

    boxed = validate_frame(_frame(boxes=[_box("bucket")]), catalog)
    boxed_result = class_supervision(boxed, catalog)
    assert boxed_result.unsupervised == names
    assert boxed_result.negatives == frozenset()
    assert boxed_result.positives == frozenset()
    _assert_partition(empty_result, catalog)
    _assert_partition(boxed_result, catalog)


def test_naive_reviewed_at_is_not_earlier_than_added_at() -> None:
    catalog = _catalog(datetime(2026, 1, 1, tzinfo=UTC))
    record = validate_frame(
        _frame(reviewed_at=datetime(2020, 1, 1), boxes=[]),  # noqa: DTZ001
        catalog,
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        class_supervision(record, catalog)
    assert record.is_negative is True
    assert record.boxes == []


def test_unreviewed_is_not_a_negative_of_any_class() -> None:
    instant = datetime(2026, 1, 1, tzinfo=UTC)
    catalog = _catalog(instant, shovel_at=instant)
    record = validate_frame(
        _frame(
            review_status="unreviewed",
            reviewed_at=instant,
            boxes=[_box("bucket")],
        ),
        catalog,
    )
    assert record.label_state is LabelState.unreviewed
    assert record.is_negative is False
    assert record.trainable is False
    result = class_supervision(record, catalog)
    assert result.negatives == frozenset()
    assert result.positives == frozenset()
    assert result.unsupervised == frozenset({*_SEED, "shovel"})
    _assert_partition(result, catalog)


def test_excluded_and_unlabelable_stay_not_trainable() -> None:
    instant = datetime(2026, 1, 1, tzinfo=UTC)
    catalog = _catalog(instant)
    excluded = validate_frame(
        _frame(
            review_status="excluded",
            reviewed_at=instant,
            boxes=[_box("bucket")],
        ),
        catalog,
    )
    assert excluded.review_status is ReviewStatus.excluded
    assert excluded.label_state is LabelState.excluded
    assert excluded.trainable is False
    assert excluded.is_negative is False
    excluded_result = class_supervision(excluded, catalog)
    assert excluded_result.negatives == frozenset()
    assert excluded_result.positives == frozenset()
    assert excluded_result.unsupervised == frozenset(_SEED)

    unlabelable = validate_frame(
        _frame(
            review_status="reviewed",
            visibility="unlabelable",
            reviewed_at=instant,
            boxes=[],
        ),
        catalog,
    )
    assert unlabelable.review_status is ReviewStatus.excluded
    assert unlabelable.label_state is LabelState.excluded
    assert unlabelable.label_state is not LabelState.reviewed_empty
    assert unlabelable.trainable is False
    assert unlabelable.is_negative is False
    hidden = class_supervision(unlabelable, catalog)
    assert hidden.negatives == frozenset()
    assert hidden.positives == frozenset()
    assert hidden.unsupervised == frozenset(_SEED)
    _assert_partition(excluded_result, catalog)
    _assert_partition(hidden, catalog)


def test_reviewed_empty_with_timestamps_is_a_negative_of_each_seed_class() -> None:
    added = datetime(2026, 1, 1, tzinfo=UTC)
    reviewed = datetime(2026, 2, 1, tzinfo=UTC)
    catalog = _catalog(added)
    record = validate_frame(_frame(reviewed_at=reviewed, boxes=[]), catalog)
    assert record.label_state is LabelState.reviewed_empty
    assert record.review_status is ReviewStatus.reviewed
    assert record.is_negative is True
    assert record.trainable is True
    assert record.boxes == []
    result = class_supervision(record, catalog)
    assert result.negatives == frozenset(_SEED)
    assert result.positives == frozenset()
    assert result.unsupervised == frozenset()
    _assert_partition(result, catalog)


def test_rule_does_not_special_case_a_name() -> None:
    import excavation_lab.datasets.class_supervision as module

    source = inspect.getsource(module)
    assert "shovel" not in source
    assert "starlette" not in source.casefold()
    assert "isoformat" not in source


def _catalog(added_at: datetime, *, shovel_at: datetime | None = None) -> StoredCatalog:
    seed = catalog_from_seed()
    classes: list[StoredCatalogClass] = [
        item.model_copy(update={"added_at": added_at}) for item in seed.classes
    ]
    if shovel_at is not None:
        classes.append(
            StoredCatalogClass(
                id=4,
                name="shovel",
                title="Лопата",
                box="Видимая лопата.",
                added_at=shovel_at,
            )
        )
    return seed.model_copy(update={"classes": classes})


def _assert_partition(result: ClassSupervision, catalog: StoredCatalog) -> None:
    names = {item.name for item in catalog.classes}
    assert result.positives | result.negatives | result.unsupervised == names
    assert result.positives.isdisjoint(result.negatives)
    assert result.positives.isdisjoint(result.unsupervised)
    assert result.negatives.isdisjoint(result.unsupervised)


def _box(category_name: str) -> dict[str, object]:
    return {
        "category_name": category_name,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 30.0,
        "y2": 40.0,
        "iscrowd": 0,
    }


def _frame(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "video_id": "pit",
        "source_frame_index": 4,
        "review_status": "reviewed",
        "boxes": [],
    }
    payload.update(overrides)
    return payload
