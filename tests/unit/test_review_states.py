"""Review states for detector labels.

Synthetic frames. They check review rules, not pit-video accuracy.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from excavation_lab.datasets.labels import (
    LabelRecord,
    LabelState,
    lookup_class,
    validate_frame,
)
from excavation_lab.errors import UnknownCategoryError, UnsupportedCrowdError
from excavation_lab.schemas import ReviewStatus, Visibility, labels_path, repo_root


def test_reviewed_frame_with_zero_boxes_is_reviewed_empty() -> None:
    record = validate_frame(_frame(review_status="reviewed", boxes=[]))
    assert record.label_state is LabelState.reviewed_empty
    assert record.label_state.value == "reviewed_empty"
    assert record.review_status is ReviewStatus.reviewed
    assert record.boxes == []
    assert record.is_negative is True
    assert record.trainable is True


def test_unreviewed_frame_with_zero_boxes_is_not_a_negative() -> None:
    record = validate_frame(_frame(review_status="unreviewed", boxes=[]))
    assert record.label_state is LabelState.unreviewed
    assert record.review_status is ReviewStatus.unreviewed
    assert record.boxes == []
    assert record.is_negative is False
    assert record.trainable is False
    assert record.label_state is not LabelState.reviewed_empty

    with_box = validate_frame(
        _frame(
            review_status="unreviewed",
            boxes=[_box("haul_truck")],
        )
    )
    assert len(with_box.boxes) == 1
    assert with_box.is_negative is False
    assert with_box.trainable is False
    assert with_box.label_state is LabelState.unreviewed


def test_excluded_frame_is_stored_and_not_trainable() -> None:
    record = validate_frame(
        _frame(
            review_status="excluded",
            note="пыль закрывает забой",
            boxes=[_box("bucket")],
        )
    )
    assert isinstance(record, LabelRecord)
    assert record.review_status is ReviewStatus.excluded
    assert record.label_state is LabelState.excluded
    assert record.trainable is False
    assert record.is_negative is False
    assert len(record.boxes) == 1
    assert record.note == "пыль закрывает забой"

    empty = validate_frame(_frame(review_status="excluded", boxes=[]))
    assert empty.label_state is LabelState.excluded
    assert empty.label_state is not LabelState.reviewed_empty
    assert empty.is_negative is False
    assert empty.trainable is False
    assert empty.boxes == []


def test_unknown_category_dump_is_not_mapped_to_bucket_or_class_zero() -> None:
    assert lookup_class("excavator_body").id == 0
    assert lookup_class("bucket").id == 1
    assert lookup_class("bucket").name.value == "bucket"

    with pytest.raises(UnknownCategoryError) as caught:
        lookup_class("dump", frame_id="pit:4")
    error = caught.value
    assert isinstance(error, UnknownCategoryError)
    assert error.code == "unknown_category"
    assert error.category_name == "dump"
    assert error.class_id is None
    assert error.category_name != "bucket"
    assert error.class_id != 0
    assert "bucket" not in error.message
    assert "pit:4" in error.message

    with pytest.raises(UnknownCategoryError) as from_frame:
        validate_frame(
            _frame(
                boxes=[
                    _box("dump", category_id=0, x1=1, y1=1, x2=8, y2=8),
                ]
            )
        )
    frame_error = from_frame.value
    assert frame_error.category_name == "dump"
    assert frame_error.class_id is None
    assert frame_error.class_id != 0
    assert frame_error.category_name != "bucket"
    assert frame_error.frame_id == "pit:4"


def test_class_lookup_by_name_uses_labels_yaml() -> None:
    path = labels_path()
    assert path == repo_root() / "configs" / "labels.yaml"
    assert path.is_file()
    found = {
        lookup_class(name).name.value: lookup_class(name).id
        for name in (
            "excavator_body",
            "bucket",
            "haul_truck",
            "truck_bed",
        )
    }
    assert found == {
        "excavator_body": 0,
        "bucket": 1,
        "haul_truck": 2,
        "truck_bed": 3,
    }
    assert found["excavator_body"] == 0
    with pytest.raises(UnknownCategoryError) as background:
        lookup_class("background")
    assert background.value.class_id is None
    with pytest.raises(UnknownCategoryError) as zero_name:
        lookup_class("0")
    assert zero_name.value.category_name == "0"
    assert zero_name.value.class_id is None

    named_bucket = validate_frame(
        _frame(boxes=[_box("bucket", category_id=0, x1=1, y1=1, x2=5, y2=5)])
    )
    assert named_bucket.boxes[0].class_name.value == "bucket"
    assert named_bucket.boxes[0].class_id == 1

    named_body = validate_frame(
        _frame(boxes=[_box("excavator_body", category_id=99, x1=1, y1=1, x2=5, y2=5)])
    )
    assert named_body.boxes[0].class_name.value == "excavator_body"
    assert named_body.boxes[0].class_id == 0


def test_iscrowd_raises_unsupported_crowd_not_partial() -> None:
    with pytest.raises(UnsupportedCrowdError) as caught:
        validate_frame(
            _frame(
                boxes=[
                    _box(
                        "bucket",
                        visibility="partial",
                        iscrowd=1,
                        x1=1,
                        y1=1,
                        x2=12,
                        y2=12,
                    )
                ]
            )
        )
    error = caught.value
    assert error.code == "unsupported_crowd"
    assert error.iscrowd == 1
    assert error.visibility is None
    assert error.frame_id == "pit:4"
    assert "partial" not in error.message
    assert "iscrowd=1" in error.message

    with pytest.raises(UnsupportedCrowdError) as dump_crowd:
        validate_frame(
            _frame(
                boxes=[
                    _box("dump", category_id=0, iscrowd=1, visibility="partial"),
                ]
            )
        )
    assert isinstance(dump_crowd.value, UnsupportedCrowdError)
    assert dump_crowd.value.iscrowd == 1
    assert dump_crowd.value.visibility is None

    with pytest.raises(UnsupportedCrowdError) as other:
        validate_frame(_frame(boxes=[_box("bucket", iscrowd=2)]))
    assert other.value.iscrowd == 2
    assert other.value.visibility is None

    partial = validate_frame(
        _frame(
            review_status="reviewed",
            boxes=[_box("bucket", visibility="partial", iscrowd=0)],
        )
    )
    assert partial.boxes[0].visibility is Visibility.partial
    assert partial.review_status is ReviewStatus.reviewed
    assert partial.label_state is LabelState.reviewed
    assert partial.trainable is True
    assert len(partial.boxes) == 1


def test_hidden_bucket_note_does_not_add_a_box() -> None:
    notes = (
        "ковш полностью скрыт за самосвалом",
        "the bucket is hidden",
        "hidden bucket, do not invent a box",
    )
    for note in notes:
        record = validate_frame(_frame(review_status="reviewed", note=note, boxes=[]))
        assert record.boxes == []
        assert record.label_state is LabelState.reviewed_empty
        assert record.note == note

    body_only = validate_frame(
        _frame(
            review_status="reviewed",
            note="bucket is hidden behind the truck",
            boxes=[_box("excavator_body", x1=4, y1=4, x2=40, y2=50)],
        )
    )
    assert [box.class_name.value for box in body_only.boxes] == ["excavator_body"]
    assert len(body_only.boxes) == 1


def test_unlabelable_frame_is_excluded_from_detector_training() -> None:
    record = validate_frame(
        _frame(
            review_status="reviewed",
            visibility="unlabelable",
            note="ковш скрыт, границы не указать",
            boxes=[],
        )
    )
    assert record.review_status is ReviewStatus.excluded
    assert record.label_state is LabelState.excluded
    assert record.visibility is Visibility.unlabelable
    assert record.trainable is False
    assert record.is_negative is False
    assert record.boxes == []
    assert record.label_state is not LabelState.reviewed_empty

    with_box = validate_frame(
        _frame(
            review_status="reviewed",
            boxes=[_box("bucket", visibility="unlabelable")],
        )
    )
    assert with_box.review_status is ReviewStatus.excluded
    assert with_box.trainable is False
    assert with_box.is_negative is False
    assert len(with_box.boxes) == 1
    assert with_box.boxes[0].visibility is Visibility.unlabelable
    assert with_box.boxes[0].class_name.value == "bucket"


def test_nested_truck_bed_inside_haul_truck_is_valid() -> None:
    record = validate_frame(
        _frame(
            review_status="reviewed",
            boxes=[
                _box("haul_truck", x1=10, y1=20, x2=210, y2=180),
                _box(
                    "truck_bed",
                    x1=80,
                    y1=40,
                    x2=190,
                    y2=150,
                    truncated=True,
                    instance_local_id="bed-a",
                ),
            ],
        )
    )
    assert record.label_state is LabelState.reviewed
    assert record.trainable is True
    assert len(record.boxes) == 2
    by_name = {box.class_name.value: box for box in record.boxes}
    assert set(by_name) == {"haul_truck", "truck_bed"}
    outer = by_name["haul_truck"]
    inner = by_name["truck_bed"]
    assert outer.class_id == 2
    assert inner.class_id == 3
    assert inner.x1 >= outer.x1
    assert inner.y1 >= outer.y1
    assert inner.x2 <= outer.x2
    assert inner.y2 <= outer.y2
    assert inner.truncated is True
    assert inner.instance_local_id == "bed-a"


def test_annotation_attributes_round_trip() -> None:
    reviewed_at = datetime(2026, 10, 10, 8, 30, tzinfo=UTC)
    record = validate_frame(
        _frame(
            review_status="reviewed",
            annotator="alex",
            reviewed_at=reviewed_at,
            note="кадр проверен",
            boxes=[
                _box(
                    "bucket",
                    truncated=True,
                    instance_local_id="b-1",
                    note="виден край ковша",
                    visibility="partial",
                    x1=3,
                    y1=4,
                    x2=33,
                    y2=44,
                )
            ],
        )
    )
    assert record.annotator == "alex"
    assert record.reviewed_at == reviewed_at
    assert record.note == "кадр проверен"
    assert record.boxes[0].truncated is True
    assert record.boxes[0].instance_local_id == "b-1"
    assert record.boxes[0].note == "виден край ковша"

    restored = LabelRecord.model_validate(record.model_dump())
    assert restored == record
    assert restored.annotator == "alex"
    assert restored.reviewed_at == reviewed_at
    assert restored.note == "кадр проверен"
    assert restored.boxes[0].truncated is True
    assert restored.boxes[0].instance_local_id == "b-1"
    assert restored.boxes[0].note == "виден край ковша"

    from_json = LabelRecord.model_validate(record.model_dump(mode="json"))
    assert from_json == record
    assert from_json.annotator == "alex"
    assert from_json.reviewed_at == reviewed_at
    assert from_json.note == "кадр проверен"
    assert from_json.boxes[0].truncated is True
    assert from_json.boxes[0].instance_local_id == "b-1"
    assert from_json.boxes[0].note == "виден край ковша"


def test_annotation_guide_is_russian_and_covers_the_rules() -> None:
    path = repo_root() / "docs" / "ANNOTATION_GUIDE.md"
    text = path.read_text(encoding="utf-8")
    assert path.name == "ANNOTATION_GUIDE.md"
    for name in ("excavator_body", "bucket", "haul_truck", "truck_bed"):
        assert name in text
    for token in (
        "reviewed_empty",
        "unreviewed",
        "excluded",
        "partial",
        "unlabelable",
        "полный ковш",
        "пустой ковш",
        "подъезд",
        "разгрузка",
        "пересменка",
    ):
        assert token in text
    folded = text.casefold()
    assert "циклы разгрузки не размечаются" in folded
    assert "будут размеч" not in folded
    assert "разметим циклы" not in folded
    cyrillic = sum(1 for char in text if "А" <= char <= "я" or char in "Ёё")
    latin = sum(1 for char in text if "A" <= char <= "Z" or "a" <= char <= "z")
    assert cyrillic > latin


def _box(category_name: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "category_name": category_name,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 30.0,
        "y2": 40.0,
        "iscrowd": 0,
    }
    payload.update(overrides)
    return payload


def _frame(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "video_id": "pit",
        "source_frame_index": 4,
        "review_status": "reviewed",
        "boxes": [],
    }
    payload.update(overrides)
    return payload
