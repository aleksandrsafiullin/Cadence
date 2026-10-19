"""Snapshot supervision. Synthetic frames, not pit-video accuracy."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from excavation_lab import storage
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import (
    StoredCatalog,
    StoredCatalogClass,
    _commit,
    add_class,
    catalog_from_seed,
    list_catalogs,
    read_catalog,
)
from excavation_lab.datasets.class_supervision import class_supervision
from excavation_lab.datasets.labels import validate_frame
from excavation_lab.datasets.snapshot import snapshot_dataset
from excavation_lab.datasets.splits import freeze_split, write_blocks
from excavation_lab.datasets.validate import validate_dataset
from excavation_lab.errors import CheckpointMissingError
from excavation_lab.ui.copy import EN, RU

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = datetime(2026, 3, 1, tzinfo=UTC)
_T2 = datetime(2026, 6, 1, tzinfo=UTC)
_T3 = datetime(2026, 8, 1, tzinfo=UTC)
_BLOCKS = (
    RoleBlock(start_s=0, end_s=60, role="train"),
    RoleBlock(start_s=72, end_s=92, role="validation"),
    RoleBlock(start_s=104, end_s=124, role="test"),
)
_WIDTH = 80
_HEIGHT = 60


def test_old_frame_is_not_a_negative_of_a_later_class(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    catalog = _excavation_with_shovel(workspace)
    _freeze(workspace, "pilot")
    labels = _write_frames(
        workspace,
        "pilot",
        [
            _stored(
                "vid_before.jpg",
                1,
                "train",
                "g-train",
                "train-ep",
                10,
                _labeled(catalog, 1, "reviewed", _T1, [_box("bucket")]),
            ),
            _stored(
                "vid_after.jpg",
                2,
                "train",
                "g-train",
                "train-ep",
                20,
                _labeled(catalog, 2, "reviewed", _T3, [_box("haul_truck")]),
            ),
            _stored(
                "vid_open.jpg",
                3,
                "train",
                "g-train",
                "train-ep",
                30,
                _labeled(catalog, 3, "unreviewed", _T3, [_box("bucket")]),
            ),
        ],
    )
    label_bytes = labels.read_bytes()
    catalog_bytes = (workspace / "catalogs" / "excavation" / "catalog.yaml").read_bytes()
    result = snapshot_dataset(workspace, "pilot")
    assert labels.read_bytes() == label_bytes
    root = Path(result.path)
    manifest = _json(root / "dataset_manifest.json")
    mapping = _json(root / "class_mapping.json")
    assert manifest["catalog_id"] == "excavation"
    assert manifest["catalog_hash"] == hashlib.sha256(catalog_bytes).hexdigest()
    assert manifest["weights_id"] is None
    assert manifest["negative_authority"] == (
        "class_supervision.json is the authority for negative. "
        "A missing COCO box is not, by itself, a negative."
    )
    assert set(mapping["by_name"]) == {
        "excavator_body",
        "bucket",
        "haul_truck",
        "truck_bed",
        "shovel",
    }
    assert mapping["by_name"]["shovel"]["internal_id"] == 4
    assert mapping["by_name"]["bucket"]["coco_category_id"] == 1
    assert "dump" not in mapping["by_name"]
    train = _json(root / "train" / "_annotations.coco.json")
    names = {item["id"]: item["name"] for item in train["categories"]}
    assert {image["file_name"] for image in train["images"]} == {
        "vid_before.jpg",
        "vid_after.jpg",
    }
    assert not (root / "train" / "vid_open.jpg").exists()
    assert not (root / "valid" / "vid_before.jpg").exists()
    before_image = _image(train, "vid_before.jpg")
    after_image = _image(train, "vid_after.jpg")
    assert _annotation_names(train, names, before_image["id"]) == ["bucket"]
    assert _annotation_names(train, names, after_image["id"]) == ["haul_truck"]
    sidecar = _json(root / "train" / "class_supervision.json")
    before = _same_record(sidecar, "vid_before.jpg")
    after = _same_record(sidecar, "vid_after.jpg")
    assert before["image_id"] == before_image["id"]
    assert after["image_id"] == after_image["id"]
    assert "shovel" in before["unsupervised"]
    assert "shovel" not in before["negatives"]
    assert "bucket" in before["positives"]
    assert "excavator_body" in before["negatives"]
    assert "shovel" in after["negatives"]
    assert "shovel" not in after["unsupervised"]
    assert "shovel" not in after["positives"]
    assert "vid_open.jpg" not in sidecar["images"]
    stored = json.loads(label_bytes)
    by_file = {frame["image_file"]: frame for frame in stored["frames"]}
    _matches_rule(before, by_file["vid_before.jpg"]["record"], catalog)
    _matches_rule(after, by_file["vid_after.jpg"]["record"], catalog)


def test_next_snapshot_does_not_rewrite_the_previous_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    catalog = _excavation_with_shovel(workspace)
    _freeze(workspace, "pilot")
    labels = _write_frames(
        workspace,
        "pilot",
        [
            _stored(
                "vid_before.jpg",
                1,
                "train",
                "g-train",
                "train-ep",
                10,
                _labeled(catalog, 1, "reviewed", _T1, [_box("bucket")]),
            ),
        ],
    )
    label_bytes = labels.read_bytes()
    first = snapshot_dataset(workspace, "pilot")
    frozen = _tree(Path(first.path))
    add_class(workspace, "excavation", "loader", "Погрузчик", "Видимый погрузчик.")
    second = snapshot_dataset(workspace, "pilot")
    assert second.dataset_id != first.dataset_id
    assert _tree(Path(first.path)) == frozen
    assert "loader" not in _json(Path(first.path) / "class_mapping.json")["by_name"]
    assert "loader" in _json(Path(second.path) / "class_mapping.json")["by_name"]
    assert labels.read_bytes() == label_bytes


def test_weights_id_is_one_published_bundle_or_null(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    catalog = _excavation_with_shovel(workspace)
    _freeze(workspace, "pilot")
    _write_frames(
        workspace,
        "pilot",
        [
            _stored(
                "vid_before.jpg",
                1,
                "train",
                "g-train",
                "train-ep",
                10,
                _labeled(catalog, 1, "reviewed", _T1, [_box("bucket")]),
            ),
        ],
    )
    _remember_model(workspace, "pilot", "pit-model", "2026-01-02T00:00:00+00:00")
    assert not (workspace / "weights").exists()
    first = snapshot_dataset(workspace, "pilot")
    assert _json(Path(first.path) / "dataset_manifest.json")["weights_id"] == "pit-model"
    assert not (workspace / "weights").exists()
    frozen = _tree(Path(first.path))
    _remember_model(workspace, "pilot", "other-model", "2026-02-02T00:00:00+00:00")
    second = snapshot_dataset(workspace, "pilot")
    assert _json(Path(second.path) / "dataset_manifest.json")["weights_id"] is None
    assert _tree(Path(first.path)) == frozen
    assert not (workspace / "weights" / "pit-model" / "weights.pth").exists()


def test_shovel_catalog_does_not_require_seed_classes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    shovel = _solo_shovel(workspace)
    _freeze(workspace, "gap")
    _write_frames(
        workspace,
        "gap",
        [
            _stored(
                "vid_empty.jpg",
                1,
                "train",
                "g-train",
                "train-ep",
                10,
                _labeled(shovel, 1, "reviewed", _T3, []),
            ),
            _stored(
                "vid_shovel.jpg",
                2,
                "validation",
                "g-val",
                "val-ep",
                80,
                _labeled(shovel, 2, "reviewed", _T3, [_box("shovel")]),
            ),
        ],
    )
    gap = snapshot_dataset(workspace, "gap")
    gap_report = validate_dataset(workspace, "gap")
    missing = {
        (item.split, item.class_name)
        for item in gap_report.errors
        if item.code == "missing_class"
    }
    assert missing == {("train", "shovel")}
    assert all(item.class_name != "excavator_body" for item in gap_report.errors)
    names = set(_json(Path(gap.path) / "class_mapping.json")["by_name"])
    assert names == {"shovel"}

    _freeze(workspace, "full")
    _write_frames(
        workspace,
        "full",
        [
            _stored(
                "vid_train.jpg",
                1,
                "train",
                "g-train",
                "train-ep",
                10,
                _labeled(shovel, 1, "reviewed", _T3, [_box("shovel")]),
            ),
            _stored(
                "vid_valid.jpg",
                2,
                "validation",
                "g-val",
                "val-ep",
                80,
                _labeled(shovel, 2, "reviewed", _T3, [_box("shovel")]),
            ),
        ],
    )
    full = snapshot_dataset(workspace, "full")
    report = validate_dataset(workspace, "full")
    assert [item.code for item in report.errors] == []
    assert report.ok is True
    assert set(_json(Path(full.path) / "class_mapping.json")["by_name"]) == {"shovel"}
    valid = _json(Path(full.path) / "valid" / "class_supervision.json")
    row = _same_record(valid, "vid_valid.jpg")
    assert row["positives"] == ["shovel"]
    assert row["negatives"] == []
    assert "excavator_body" not in row["negatives"]
    assert "excavator_body" not in row["unsupervised"]


def test_bundle_sentences_follow_the_selected_catalog() -> None:
    assert "four classes" not in EN.custom_needs_bundle
    assert "four classes" not in RU.custom_needs_bundle
    assert "четырьмя" not in RU.custom_needs_bundle
    assert "catalog" in EN.custom_needs_bundle
    assert "каталога" in RU.custom_needs_bundle
    assert "not substituted" in EN.custom_needs_bundle
    assert "не подставляется" in RU.custom_needs_bundle
    for error in (CheckpointMissingError(), CheckpointMissingError("weights.pth")):
        assert "четырьмя" not in error.message
        assert "четыре" not in error.message
        assert "COCO" in error.message
        assert "каталога" in error.message
    assert "weights.pth" in CheckpointMissingError("weights.pth").message
    snapshot_source = Path(snapshot_dataset.__code__.co_filename).read_text(encoding="utf-8")
    assert "run_test" not in snapshot_source
    assert "catalog_from_seed" not in snapshot_source
    assert "load_labels" not in snapshot_source
    assert "starlette" not in snapshot_source.casefold()


def _excavation_with_shovel(workspace: Path) -> StoredCatalog:
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    seed = catalog_from_seed()
    classes = [item.model_copy(update={"added_at": _T0}) for item in seed.classes]
    classes.append(
        StoredCatalogClass(
            id=4,
            name="shovel",
            title="Лопата",
            box="Видимая лопата.",
            added_at=_T2,
        )
    )
    _commit(
        workspace / "catalogs" / "excavation" / "catalog.yaml",
        seed.model_copy(update={"classes": classes}),
    )
    return read_catalog(workspace, "excavation")


def _solo_shovel(workspace: Path) -> StoredCatalog:
    list_catalogs(workspace)
    seed = catalog_from_seed()
    solo = seed.model_copy(
        update={
            "classes": [
                StoredCatalogClass(
                    id=0,
                    name="shovel",
                    title="Лопата",
                    box="Видимая лопата.",
                    added_at=_T0,
                )
            ]
        }
    )
    _commit(workspace / "catalogs" / "pit" / "catalog.yaml", solo)
    return read_catalog(workspace, "pit")


def _freeze(workspace: Path, project_id: str) -> None:
    if project_id != "pilot":
        storage.create_project(project_id, workspace, catalog_id="pit")
    clip = workspace / f"{project_id}-clip.mp4"
    clip.write_bytes(b"synthetic-not-a-video")
    storage.insert_video(workspace, project_id, "vid", path=clip, name="clip.mp4")
    freeze_split(workspace, project_id, write_blocks(workspace, project_id, "vid", list(_BLOCKS)))


def _labeled(
    catalog: StoredCatalog,
    index: int,
    review: str,
    reviewed_at: datetime,
    boxes: list[dict[str, object]],
) -> dict[str, object]:
    record = validate_frame(
        {
            "video_id": "vid",
            "source_frame_index": index,
            "review_status": review,
            "reviewed_at": reviewed_at,
            "boxes": boxes,
        },
        catalog,
    )
    return record.model_dump(mode="json")


def _stored(
    image_file: str,
    index: int,
    split: str,
    group: str,
    episode_id: str,
    time_s: float,
    record: dict[str, object],
) -> dict[str, object]:
    payload = f"synthetic-jpeg-{image_file}".encode()
    return {
        "image_file": image_file,
        "payload": payload,
        "video_id": "vid",
        "source_frame_index": index,
        "source_time_s": time_s,
        "episode_id": episode_id,
        "split": split,
        "split_group_id": group,
        "record": record,
    }


def _write_frames(workspace: Path, project_id: str, frames: list[dict[str, object]]) -> Path:
    batch = workspace / "projects" / project_id / "annotation_batches" / "vid__train"
    images = batch / "images"
    images.mkdir(parents=True)
    stored: list[dict[str, object]] = []
    for frame in frames:
        payload = frame["payload"]
        assert isinstance(payload, bytes)
        image_file = frame["image_file"]
        assert isinstance(image_file, str)
        (images / image_file).write_bytes(payload)
        stored.append(
            {
                "image_file": image_file,
                "video_id": frame["video_id"],
                "source_frame_index": frame["source_frame_index"],
                "source_time_s": frame["source_time_s"],
                "episode_id": frame["episode_id"],
                "split": frame["split"],
                "split_group_id": frame["split_group_id"],
                "image_sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "width": _WIDTH,
                "height": _HEIGHT,
                "coco_subset": "",
                "flagged": [],
                "record": frame["record"],
            }
        )
    path = batch / "imported_labels.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "batch_id": "vid__train",
                "project_id": project_id,
                "video_id": "vid",
                "frames": stored,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _remember_model(workspace: Path, project_id: str, model_id: str, created_at: str) -> None:
    with storage.connect(storage.project_sqlite(workspace, project_id)) as connection:
        connection.execute(
            """
            INSERT INTO model_versions (
                model_id, project_id, dataset_id, variant, path, created_at
            ) VALUES (?, ?, NULL, ?, ?, ?)
            """,
            (
                model_id,
                project_id,
                "nano",
                str(workspace / "weights" / model_id),
                created_at,
            ),
        )


def _box(category_name: str) -> dict[str, object]:
    return {
        "category_name": category_name,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 30.0,
        "y2": 40.0,
        "iscrowd": 0,
    }


def _json(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _image(coco: dict[str, object], file_name: str) -> dict[str, object]:
    images = coco["images"]
    assert isinstance(images, list)
    found = next(item for item in images if item["file_name"] == file_name)
    assert isinstance(found, dict)
    return found


def _annotation_names(
    coco: dict[str, object],
    names: dict[object, object],
    image_id: object,
) -> list[object]:
    annotations = coco["annotations"]
    assert isinstance(annotations, list)
    return [
        names[item["category_id"]]
        for item in annotations
        if item["image_id"] == image_id
    ]


def _same_record(document: dict[str, object], file_name: str) -> dict[str, object]:
    images = document["images"]
    assert isinstance(images, dict)
    by_name = images[file_name]
    assert isinstance(by_name, dict)
    by_id = images[str(by_name["image_id"])]
    assert by_id == by_name
    assert by_name["file_name"] == file_name
    return by_name


def _matches_rule(
    sidecar: dict[str, object],
    raw_record: object,
    catalog: StoredCatalog,
) -> None:
    from excavation_lab.datasets.labels import LabelRecord

    record = LabelRecord.model_validate(raw_record)
    supervision = class_supervision(record, catalog)
    assert sidecar["positives"] == sorted(supervision.positives)
    assert sidecar["negatives"] == sorted(supervision.negatives)
    assert sidecar["unsupervised"] == sorted(supervision.unsupervised)


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
