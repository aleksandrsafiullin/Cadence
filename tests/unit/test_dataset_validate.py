"""Snapshot checks before training.

Synthetic JPEGs. They check the report, not pit-video accuracy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from typer.testing import CliRunner

from excavation_lab import storage
from excavation_lab.cli import app
from excavation_lab.datasets.catalog import catalog_from_seed, list_catalogs
from excavation_lab.datasets.coco_import import build_class_mapping
from excavation_lab.datasets.labels import LabelRecord
from excavation_lab.datasets.validate import validate_dataset
from excavation_lab.schemas import (
    AnnotationBox,
    DetectionClassName,
    ReviewStatus,
    Visibility,
)

_RUNNER = CliRunner()
_WIDTH = 64
_HEIGHT = 48
_ALL = ("excavator_body", "bucket", "haul_truck", "truck_bed")
_NO_BUCKET = ("excavator_body", "haul_truck", "truck_bed")
_XYWH = {
    "excavator_body": (1.0, 1.0, 8.0, 8.0),
    "bucket": (12.0, 1.0, 8.0, 8.0),
    "haul_truck": (24.0, 1.0, 8.0, 8.0),
    "truck_bed": (36.0, 1.0, 8.0, 8.0),
}


def _cyrillic(text: str) -> bool:
    return any("а" <= char.lower() <= "я" or char.lower() == "ё" for char in text)


@dataclass(frozen=True)
class _Spec:
    image_file: str
    index: int
    split: str
    group: str
    review: str
    classes: tuple[str, ...] = ()
    directory: str | None = None
    extra_boxes: tuple[tuple[str, tuple[float, float, float, float]], ...] = ()
    video_id: str = "vid"


def _class_id(name: str) -> int:
    return int(build_class_mapping(catalog_from_seed())["by_name"][name]["coco_category_id"])


def _payload(index: int) -> bytes:
    return f"synthetic-jpeg-{index:04d}-not-a-real-image".encode()


def _categories() -> list[dict[str, object]]:
    mapping = build_class_mapping(catalog_from_seed())["by_name"]
    rows = [
        {"id": int(spec["coco_category_id"]), "name": name, "supercategory": "none"}
        for name, spec in mapping.items()
    ]
    rows.sort(key=lambda item: int(item["id"]))
    return rows


def _label_box(name: str, xywh: tuple[float, float, float, float]) -> AnnotationBox:
    x, y, width, height = xywh
    return AnnotationBox(
        class_name=DetectionClassName(name),
        class_id=_class_id(name),
        visibility=Visibility.clear,
        x1=x,
        y1=y,
        x2=x + width,
        y2=y + height,
    )


def _record(spec: _Spec) -> LabelRecord:
    boxes = [_label_box(name, _XYWH[name]) for name in spec.classes]
    return LabelRecord(
        video_id=spec.video_id,
        source_frame_index=spec.index,
        review_status=ReviewStatus(spec.review),
        boxes=boxes,
    )


def _coco_box(
    ann_id: int,
    image_id: int,
    name: str,
    xywh: tuple[float, float, float, float],
) -> dict[str, object]:
    x, y, width, height = xywh
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": _class_id(name),
        "bbox": [x, y, width, height],
        "area": width * height,
        "iscrowd": 0,
    }


def _dump(path: Path, document: object) -> None:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_case(tmp_path: Path, specs: list[_Spec]) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    project = workspace / "projects" / "pilot"
    snapshot = project / "dataset_versions" / "dsvalidate"
    batch = project / "annotation_batches" / "vid__train"
    images = batch / "images"
    images.mkdir(parents=True)
    frames: list[dict[str, object]] = []
    by_dir: dict[str, list[_Spec]] = {"train": [], "valid": [], "test": []}
    for spec in specs:
        payload = _payload(spec.index)
        (images / spec.image_file).write_bytes(payload)
        frames.append(
            {
                "image_file": spec.image_file,
                "video_id": spec.video_id,
                "source_frame_index": spec.index,
                "source_time_s": float(spec.index),
                "episode_id": "",
                "split": spec.split,
                "split_group_id": spec.group,
                "image_sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "width": _WIDTH,
                "height": _HEIGHT,
                "coco_subset": "",
                "flagged": [],
                "record": _record(spec).model_dump(mode="json"),
            }
        )
        if spec.directory is None:
            continue
        by_dir[spec.directory].append(spec)
        destination = snapshot / spec.directory
        destination.mkdir(parents=True, exist_ok=True)
        (destination / spec.image_file).write_bytes(payload)
    categories = _categories()
    for directory, items in by_dir.items():
        split_dir = snapshot / directory
        split_dir.mkdir(parents=True, exist_ok=True)
        coco_images: list[dict[str, object]] = []
        annotations: list[dict[str, object]] = []
        next_ann = 1
        for image_id, spec in enumerate(items, start=1):
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": spec.image_file,
                    "width": _WIDTH,
                    "height": _HEIGHT,
                }
            )
            for name in spec.classes:
                annotations.append(_coco_box(next_ann, image_id, name, _XYWH[name]))
                next_ann += 1
            for name, xywh in spec.extra_boxes:
                annotations.append(_coco_box(next_ann, image_id, name, xywh))
                next_ann += 1
        _dump(
            split_dir / "_annotations.coco.json",
            {
                "categories": categories,
                "images": coco_images,
                "annotations": annotations,
            },
        )
    dataset_hash = "a" * 64
    _dump(snapshot / "class_mapping.json", build_class_mapping(catalog_from_seed()))
    _dump(snapshot / "split_manifest.json", {"schema_version": 1})
    _dump(
        snapshot / "dataset_manifest.json",
        {
            "schema_version": 1,
            "dataset_id": "dsvalidate",
            "project_id": "pilot",
            "dataset_hash": dataset_hash,
        },
    )
    _dump(
        batch / "imported_labels.json",
        {
            "schema_version": 1,
            "batch_id": "vid__train",
            "project_id": "pilot",
            "video_id": "vid",
            "frames": frames,
        },
    )
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            """
            INSERT INTO dataset_versions (
                dataset_id, project_id, content_hash, path, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "dsvalidate",
                "pilot",
                dataset_hash,
                str(snapshot.resolve()),
                "2026-01-01T00:00:00+00:00",
            ),
        )
    return workspace, snapshot


def _names(directory: Path) -> set[str]:
    return {path.name for path in directory.iterdir()}


def test_missing_class_does_not_copy_from_test(tmp_path: Path) -> None:
    workspace, snapshot = _write_case(
        tmp_path,
        [
            _Spec("vid_train.jpg", 1, "train", "g-train", "reviewed", _NO_BUCKET, "train"),
            _Spec("vid_valid.jpg", 2, "validation", "g-val", "reviewed", _NO_BUCKET, "valid"),
            _Spec("vid_test.jpg", 3, "test", "g-test", "reviewed", ("bucket",), "test"),
            _Spec("vid_open.jpg", 4, "train", "g-open", "unreviewed", ("bucket",)),
            _Spec("vid_empty.jpg", 5, "train", "g-train", "reviewed", (), "train"),
            _Spec("vid_skip.jpg", 6, "train", "g-skip", "excluded", ("bucket",)),
        ],
    )
    train_before = _names(snapshot / "train")
    valid_before = _names(snapshot / "valid")
    test_bytes = (snapshot / "test" / "vid_test.jpg").read_bytes()
    result = validate_dataset(workspace, "pilot")
    assert result.ok is False
    missing = {
        (item.split, item.class_name)
        for item in result.errors
        if item.code == "missing_class"
    }
    assert missing == {("train", "bucket"), ("validation", "bucket")}
    assert result.negatives == result.reviewed_empty == 1
    assert result.unreviewed == 1
    assert result.excluded == 1
    assert _names(snapshot / "train") == train_before
    assert _names(snapshot / "valid") == valid_before
    assert "vid_test.jpg" not in train_before
    assert not (snapshot / "train" / "vid_test.jpg").exists()
    assert not (snapshot / "valid" / "vid_test.jpg").exists()
    assert (snapshot / "test" / "vid_test.jpg").read_bytes() == test_bytes
    invoked = _RUNNER.invoke(
        app,
        [
            "dataset",
            "validate",
            "--project",
            "pilot",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert invoked.exit_code != 0
    payload = json.loads(invoked.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "missing_class"
    assert payload["error"]["code"].isascii()
    assert _cyrillic(payload["error"]["message"])
    listed = {
        (item["split"], item["class_name"])
        for item in payload["data"]["errors"]
        if item["code"] == "missing_class"
    }
    assert listed == {("train", "bucket"), ("validation", "bucket")}


def test_box_past_the_edge_is_reported_and_not_rewritten(tmp_path: Path) -> None:
    workspace, snapshot = _write_case(
        tmp_path,
        [
            _Spec(
                "vid_edge.jpg",
                11,
                "train",
                "g-train",
                "reviewed",
                _ALL,
                "train",
                extra_boxes=(("excavator_body", (0.0, 0.0, 80.0, 4.0)),),
            ),
            _Spec("vid_valid.jpg", 12, "validation", "g-val", "reviewed", _ALL, "valid"),
            _Spec("vid_test.jpg", 13, "test", "g-test", "reviewed", _ALL, "test"),
        ],
    )
    coco_path = snapshot / "train" / "_annotations.coco.json"
    before = coco_path.read_bytes()
    result = validate_dataset(workspace, "pilot")
    assert coco_path.read_bytes() == before
    boxes = [
        annotation["bbox"]
        for annotation in json.loads(before)["annotations"]
    ]
    assert [0.0, 0.0, 80.0, 4.0] in boxes
    matched = [
        item
        for item in result.errors
        if item.code == "invalid_bbox"
    ]
    assert len(matched) == 1
    assert matched[0].image_file == "vid_edge.jpg"
    assert matched[0].source_frame_index == 11
    assert result.ok is False


def test_split_group_crossover_names_a_frame(tmp_path: Path) -> None:
    workspace, _snapshot = _write_case(
        tmp_path,
        [
            _Spec("vid_train.jpg", 1, "train", "g-shared", "reviewed", _ALL, "train"),
            _Spec("vid_cross.jpg", 2, "train", "g-shared", "reviewed", _ALL, "valid"),
            _Spec("vid_test.jpg", 3, "test", "g-test", "reviewed", _ALL, "test"),
        ],
    )
    result = validate_dataset(workspace, "pilot")
    matched = [item for item in result.errors if item.code == "split_group_crossover"]
    assert len(matched) == 1
    issue = matched[0]
    assert issue.image_file == "vid_cross.jpg"
    assert issue.source_frame_index == 2
    assert "g-shared" in issue.message
    assert "validation" in issue.message
    assert result.ok is False


def test_same_group_id_on_two_videos_is_not_crossover(tmp_path: Path) -> None:
    workspace, _snapshot = _write_case(
        tmp_path,
        [
            _Spec(
                "a_train.jpg",
                1,
                "train",
                "g-train",
                "reviewed",
                _ALL,
                "train",
                video_id="video-a",
            ),
            _Spec(
                "b_valid.jpg",
                2,
                "validation",
                "g-train",
                "reviewed",
                _ALL,
                "valid",
                video_id="video-b",
            ),
            _Spec(
                "a_test.jpg",
                3,
                "test",
                "g-test",
                "reviewed",
                _ALL,
                "test",
                video_id="video-a",
            ),
        ],
    )
    result = validate_dataset(workspace, "pilot")
    assert result.errors == []
    assert result.ok is True


def test_unlabeled_test_warns_and_unreviewed_is_not_a_negative(tmp_path: Path) -> None:
    workspace, _snapshot = _write_case(
        tmp_path,
        [
            _Spec("vid_train.jpg", 1, "train", "g-train", "reviewed", _ALL, "train"),
            _Spec("vid_empty.jpg", 2, "train", "g-train", "reviewed", (), "train"),
            _Spec("vid_valid.jpg", 3, "validation", "g-val", "reviewed", _ALL, "valid"),
            _Spec("vid_open.jpg", 4, "train", "g-open", "unreviewed", ()),
            _Spec("vid_boxed.jpg", 5, "train", "g-boxed", "unreviewed", ("bucket",)),
            _Spec("vid_skip.jpg", 6, "train", "g-skip", "excluded", ("bucket",)),
        ],
    )
    result = validate_dataset(workspace, "pilot")
    assert result.ok is True
    assert result.errors == []
    assert [item.code for item in result.warnings] == ["test_not_labeled"]
    assert result.reviewed == 3
    assert result.unreviewed == 2
    assert result.excluded == 1
    assert result.reviewed_empty == 1
    assert result.negatives == 1
    assert result.reviewed + result.unreviewed + result.excluded == 6
    invoked = _RUNNER.invoke(
        app,
        [
            "dataset",
            "validate",
            "--project",
            "pilot",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert invoked.exit_code == 0
    payload = json.loads(invoked.stdout)
    assert payload["ok"] is True
    data = payload["data"]
    assert data["warnings"][0]["code"] == "test_not_labeled"
    assert data["errors"] == []
    assert data["negatives"] == data["reviewed_empty"] == 1
    assert data["unreviewed"] == 2
    assert _cyrillic(data["warnings"][0]["message"])


def test_hash_mismatch_names_the_frame(tmp_path: Path) -> None:
    workspace, snapshot = _write_case(
        tmp_path,
        [
            _Spec("vid_train.jpg", 1, "train", "g-train", "reviewed", _ALL, "train"),
            _Spec("vid_valid.jpg", 2, "validation", "g-val", "reviewed", _ALL, "valid"),
            _Spec("vid_test.jpg", 3, "test", "g-test", "reviewed", _ALL, "test"),
        ],
    )
    target = snapshot / "train" / "vid_train.jpg"
    batch = (
        workspace
        / "projects"
        / "pilot"
        / "annotation_batches"
        / "vid__train"
        / "images"
        / "vid_train.jpg"
    )
    original = batch.read_bytes()
    target.write_bytes(b"tampered-bytes")
    result = validate_dataset(workspace, "pilot")
    assert target.read_bytes() == b"tampered-bytes"
    assert batch.read_bytes() == original
    matched = [item for item in result.errors if item.code == "image_hash_mismatch"]
    assert len(matched) == 1
    assert matched[0].image_file == "vid_train.jpg"
    assert matched[0].source_frame_index == 1
    assert result.ok is False


def test_validate_module_does_not_score_or_copy() -> None:
    source = Path(validate_dataset.__code__.co_filename).read_text(encoding="utf-8")
    assert "pycocotools" not in source
    assert "streamlit" not in source
    assert "shutil" not in source
    assert "mAP" not in source
    assert "_REQUIRED_CLASSES" not in source
    assert "DetectionClassName" not in source
