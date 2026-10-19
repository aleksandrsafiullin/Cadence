"""CVAT COCO import and the RF-DETR snapshot.

Synthetic images. They check the exchange, not pit-video accuracy.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from excavation_lab import storage
from excavation_lab.cli import app
from excavation_lab.datasets import coco_import as import_mod
from excavation_lab.datasets import snapshot as snapshot_mod
from excavation_lab.datasets import splits as split_mod
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import (
    add_class,
    list_catalogs,
    require_project_catalog,
)
from excavation_lab.datasets.class_supervision import class_supervision
from excavation_lab.datasets.coco_import import (
    DEFAULT_MAX_UNPACK_BYTES,
    import_coco,
)
from excavation_lab.datasets.cvat_export import MANIFEST_COLUMNS
from excavation_lab.datasets.geometry import xywh_to_xyxy
from excavation_lab.datasets.labels import LabelRecord
from excavation_lab.datasets.snapshot import snapshot_dataset
from excavation_lab.datasets.splits import freeze_split, write_blocks
from excavation_lab.errors import (
    LabError,
    SplitNotFrozenError,
    UnknownCategoryError,
    UnsupportedCrowdError,
)

_RUNNER = CliRunner()
_BLOCKS = (
    RoleBlock(start_s=0, end_s=60, role="train"),
    RoleBlock(start_s=72, end_s=92, role="validation"),
    RoleBlock(start_s=104, end_s=124, role="test"),
)


def _calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                found.add(func.id)
            elif isinstance(func, ast.Attribute):
                found.add(func.attr)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                found.add(alias.name)
    return found


def _jpeg(width: int, height: int, tone: int) -> bytes:
    image = np.full((height, width, 3), tone, dtype=np.uint8)
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    assert ok
    return encoded.tobytes()


def _manifest(rows: list[dict[str, str]]) -> bytes:
    fields = [*MANIFEST_COLUMNS, "size_bytes"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _row(
    *,
    image_file: str,
    payload: bytes,
    index: int,
    episode_id: str,
    split: str,
    group: str,
    time_s: float,
    width: int,
    height: int,
    sha256: str | None = None,
    size_bytes: int | None = None,
) -> dict[str, str]:
    return {
        "image_file": image_file,
        "video_id": "vid",
        "source_frame_index": str(index),
        "source_pts": str(index),
        "time_base": "1/1000",
        "source_time_s": f"{time_s:.6f}",
        "episode_id": episode_id,
        "split": split,
        "split_group_id": group,
        "image_sha256": sha256 or hashlib.sha256(payload).hexdigest(),
        "width": str(width),
        "height": str(height),
        "size_bytes": str(len(payload) if size_bytes is None else size_bytes),
    }


def _json(document: object) -> bytes:
    return (json.dumps(document, ensure_ascii=False) + "\n").encode("utf-8")


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            info = zipfile.ZipInfo(filename=name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.date_time = (2026, 1, 1, 0, 0, 0)
            archive.writestr(info, payload)


def _frozen_project(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    clip = workspace / "clip.mp4"
    clip.write_bytes(b"synthetic-not-a-video")
    storage.insert_video(workspace, "pilot", "vid", path=clip, name="clip.mp4")
    freeze_split(workspace, "pilot", write_blocks(workspace, "pilot", "vid", list(_BLOCKS)))
    frozen = split_mod.require_frozen(workspace, "pilot", "vid")
    assert [(block.role, block.start_s, block.end_s) for block in frozen.blocks] == [
        ("train", 0.0, 60.0),
        ("validation", 72.0, 92.0),
        ("test", 104.0, 124.0),
    ]
    return workspace


def _batch(workspace: Path, batch_id: str) -> Path:
    return workspace / "projects" / "pilot" / "annotation_batches" / batch_id


def test_roundtrip_keeps_xyxy_and_category_name(tmp_path: Path) -> None:
    workspace = _frozen_project(tmp_path)
    width, height = 80, 60
    xyxy = (10.0, 12.0, 40.0, 36.0)
    xywh = (10.0, 12.0, 30.0, 24.0)
    payload = _jpeg(width, height, 40)
    image_file = "vid_3.jpg"
    zip_path = workspace / "incoming.zip"
    _write_zip(
        zip_path,
        {
            "frames_manifest.csv": _manifest(
                [
                    _row(
                        image_file=image_file,
                        payload=payload,
                        index=3,
                        episode_id="val-ep",
                        split="validation",
                        group="g-val",
                        time_s=80,
                        width=width,
                        height=height,
                    )
                ]
            ),
            "annotations/instances_default.json": _json(
                {
                    "categories": [{"id": 99, "name": "bucket", "supercategory": ""}],
                    "images": [
                        {
                            "id": 1,
                            "file_name": f"images/{image_file}",
                            "width": width,
                            "height": height,
                            "attributes": {"review_status": "reviewed"},
                        }
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 99,
                            "bbox": list(xywh),
                            "area": 30 * 24,
                            "iscrowd": 0,
                            "attributes": {
                                "visibility": "partial",
                                "truncated": True,
                                "instance_local_id": "b-1",
                                "note": "край",
                                "mystery_attr": "do-not-drop",
                            },
                            "keeper": "stay",
                        }
                    ],
                }
            ),
            f"images/{image_file}": payload,
        },
    )
    imported = _RUNNER.invoke(
        app,
        [
            "dataset",
            "import-coco",
            "--project",
            "pilot",
            "--batch",
            "vid__validation",
            "--file",
            str(zip_path),
            "--workspace",
            str(workspace),
        ],
    )
    assert imported.exit_code == 0, imported.output
    batch = _batch(workspace, "vid__validation")
    assert (batch / "source.zip").read_bytes() == zip_path.read_bytes()
    labels = json.loads((batch / "imported_labels.json").read_text(encoding="utf-8"))
    box = labels["frames"][0]["record"]["boxes"][0]
    assert box["class_name"] == "bucket"
    assert box["class_id"] == 1
    sidecar = json.loads((batch / "attributes_sidecar.json").read_text(encoding="utf-8"))
    attrs = sidecar["frames"][0]["boxes"][0]["attributes"]
    assert attrs["mystery_attr"] == "do-not-drop"
    assert attrs["keeper"] == "stay"
    assert attrs["visibility"] == "partial"
    assert attrs["truncated"] is True
    assert attrs["instance_local_id"] == "b-1"
    assert attrs["note"] == "край"
    snapped = _RUNNER.invoke(
        app,
        [
            "dataset",
            "snapshot",
            "--project",
            "pilot",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert snapped.exit_code == 0, snapped.output
    data = json.loads(snapped.stdout)["data"]
    root = Path(data["path"])
    assert root.name == data["dataset_id"]
    assert root.parent.name == "dataset_versions"
    assert sorted(path.name for path in root.iterdir()) == [
        "class_mapping.json",
        "dataset_manifest.json",
        "split_manifest.json",
        "test",
        "train",
        "valid",
    ]
    assert not (root / "val").exists()
    assert (root / "valid" / image_file).is_file()
    coco = json.loads((root / "valid" / "_annotations.coco.json").read_text(encoding="utf-8"))
    names = {item["id"]: item["name"] for item in coco["categories"]}
    assert coco["images"][0]["file_name"] == image_file
    annotation = coco["annotations"][0]
    assert names[annotation["category_id"]] == "bucket"
    assert annotation["category_id"] == 1
    assert 99 not in names
    assert xywh_to_xyxy(annotation["bbox"]) == xyxy
    assert "mystery_attr" not in annotation
    train = json.loads((root / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert train["images"] == []
    mapping = json.loads((root / "class_mapping.json").read_text(encoding="utf-8"))
    assert mapping["by_name"]["bucket"]["internal_id"] == 1
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["dataset_hash"] == data["dataset_hash"]
    assert len(manifest["dataset_hash"]) == 64
    assert manifest["split_content_hash"] == frozen["content_hash"]
    again = snapshot_dataset(workspace, "pilot")
    assert again.dataset_hash == manifest["dataset_hash"]
    assert again.dataset_id != data["dataset_id"]


def test_train_coco_keeps_reviewed_frames_only(tmp_path: Path) -> None:
    workspace = _frozen_project(tmp_path)
    width, height = 64, 48
    specs = (
        ("vid_empty.jpg", 1, "reviewed", None, 10),
        ("vid_open.jpg", 2, "unreviewed", (99, "bucket", [4, 4, 10, 8]), 12),
        ("vid_skip.jpg", 3, "excluded", (0, "excavator_body", [1, 1, 6, 6]), 14),
        ("vid_truck.jpg", 4, "reviewed", (5, "haul_truck", [2, 3, 12, 9]), 16),
    )
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    rows: list[dict[str, str]] = []
    members: dict[str, bytes] = {}
    categories = {
        99: "bucket",
        0: "excavator_body",
        5: "haul_truck",
    }
    next_annotation = 1
    for image_file, index, review, box, tone in specs:
        payload = _jpeg(width, height, tone)
        members[f"images/{image_file}"] = payload
        rows.append(
            _row(
                image_file=image_file,
                payload=payload,
                index=index,
                episode_id="train-ep",
                split="train",
                group="g-train",
                time_s=float(index),
                width=width,
                height=height,
            )
        )
        images.append(
            {
                "id": index,
                "file_name": f"images/{image_file}",
                "width": width,
                "height": height,
                "attributes": {"review_status": review},
            }
        )
        if box is not None:
            category_id, _name, bbox = box
            annotations.append(
                {
                    "id": next_annotation,
                    "image_id": index,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            next_annotation += 1
    members["frames_manifest.csv"] = _manifest(rows)
    members["annotations/instances_train.json"] = _json(
        {
            "categories": [
                {"id": category_id, "name": name, "supercategory": ""}
                for category_id, name in categories.items()
            ],
            "images": images,
            "annotations": annotations,
        }
    )
    zip_path = workspace / "train.zip"
    _write_zip(zip_path, members)
    result = import_coco(workspace, "pilot", "vid__train", zip_path)
    assert result.reviewed_empty == 1
    assert result.flagged_unreviewed == 1
    assert result.stored_excluded == 1
    labels = json.loads((Path(result.labels_path)).read_text(encoding="utf-8"))
    by_file = {frame["image_file"]: frame for frame in labels["frames"]}
    assert by_file["vid_empty.jpg"]["record"]["label_state"] == "reviewed_empty"
    assert by_file["vid_empty.jpg"]["record"]["review_status"] == "reviewed"
    assert by_file["vid_empty.jpg"]["record"]["boxes"] == []
    assert by_file["vid_empty.jpg"]["record"]["is_negative"] is True
    assert by_file["vid_open.jpg"]["flagged"] == ["unreviewed"]
    assert by_file["vid_open.jpg"]["record"]["is_negative"] is False
    assert by_file["vid_open.jpg"]["record"]["trainable"] is False
    assert by_file["vid_open.jpg"]["record"]["boxes"][0]["class_id"] == 1
    assert by_file["vid_skip.jpg"]["record"]["review_status"] == "excluded"
    assert by_file["vid_skip.jpg"]["record"]["trainable"] is False
    assert by_file["vid_skip.jpg"]["record"]["boxes"][0]["class_name"] == "excavator_body"
    assert by_file["vid_skip.jpg"]["record"]["boxes"][0]["class_id"] == 0
    batch_images = Path(result.labels_path).parent / "images"
    assert (batch_images / "vid_open.jpg").is_file()
    assert (batch_images / "vid_skip.jpg").is_file()
    snapshot = snapshot_dataset(workspace, "pilot")
    train = json.loads(
        (Path(snapshot.path) / "train" / "_annotations.coco.json").read_text(encoding="utf-8")
    )
    names = {item["id"]: item["name"] for item in train["categories"]}
    assert {image["file_name"] for image in train["images"]} == {"vid_empty.jpg", "vid_truck.jpg"}
    empty_id = next(image["id"] for image in train["images"] if image["file_name"] == "vid_empty.jpg")
    assert all(annotation["image_id"] != empty_id for annotation in train["annotations"])
    assert len(train["annotations"]) == 1
    assert names[train["annotations"][0]["category_id"]] == "haul_truck"
    assert train["annotations"][0]["category_id"] == 2
    train_dir = Path(snapshot.path) / "train"
    assert (train_dir / "vid_empty.jpg").is_file()
    assert (train_dir / "vid_truck.jpg").is_file()
    assert not (train_dir / "vid_open.jpg").exists()
    assert not (train_dir / "vid_skip.jpg").exists()


def test_coco_subset_test_follows_manifest_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _frozen_project(tmp_path)
    calls: list[str] = []

    def _spy(*_args: object, **_kwargs: object) -> None:
        calls.append("propose_split")

    monkeypatch.setattr(split_mod, "propose_split", _spy)
    for module in (import_mod, snapshot_mod):
        if hasattr(module, "propose_split"):
            monkeypatch.setattr(module, "propose_split", _spy)
    width, height = 64, 48
    payload = _jpeg(width, height, 90)
    image_file = "vid_1.jpg"
    zip_path = workspace / "subset.zip"
    _write_zip(
        zip_path,
        {
            "frames_manifest.csv": _manifest(
                [
                    _row(
                        image_file=image_file,
                        payload=payload,
                        index=1,
                        episode_id="train-ep",
                        split="train",
                        group="g-train",
                        time_s=10,
                        width=width,
                        height=height,
                    )
                ]
            ),
            "annotations/instances_test.json": _json(
                {
                    "categories": [{"id": 99, "name": "bucket", "supercategory": ""}],
                    "images": [
                        {
                            "id": 1,
                            "file_name": f"images/test/{image_file}",
                            "width": width,
                            "height": height,
                            "attributes": {"review_status": "reviewed"},
                        }
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 99,
                            "bbox": [2, 2, 8, 6],
                            "area": 48,
                            "iscrowd": 0,
                        }
                    ],
                }
            ),
            f"images/test/{image_file}": payload,
        },
    )
    imported = import_coco(workspace, "pilot", "vid__train", zip_path)
    labels = json.loads(Path(imported.labels_path).read_text(encoding="utf-8"))
    assert labels["frames"][0]["coco_subset"] == "test"
    assert labels["frames"][0]["split"] == "train"
    snapshot = snapshot_dataset(workspace, "pilot")
    root = Path(snapshot.path)
    assert (root / "train" / image_file).is_file()
    assert not (root / "test" / image_file).exists()
    assert not (root / "val").exists()
    assert calls == []
    for module in (import_mod, snapshot_mod):
        assert "propose_split" not in _calls(Path(module.__file__))


def test_snapshot_without_manifest_is_split_not_frozen(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    clip = workspace / "clip.mp4"
    clip.write_bytes(b"synthetic-not-a-video")
    storage.insert_video(workspace, "pilot", "vid", path=clip, name="clip.mp4")
    with pytest.raises(SplitNotFrozenError) as caught:
        snapshot_dataset(workspace, "pilot")
    assert caught.value.video_id == "vid"
    result = _RUNNER.invoke(
        app,
        [
            "dataset",
            "snapshot",
            "--project",
            "pilot",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "split_not_frozen"


@pytest.mark.parametrize("name", ["../evil.jpg", "/etc/passwd", "images/../../passwd"])
def test_zip_rejects_parent_and_absolute(tmp_path: Path, name: str) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    zip_path = workspace / "bad.zip"
    _write_zip(zip_path, {name: b"x"})
    with pytest.raises(LabError) as caught:
        import_coco(workspace, "pilot", "vid__train", zip_path)
    assert caught.value.code == "zip_member"
    assert list((workspace / "projects" / "pilot" / "annotation_batches").iterdir()) == []


def test_zip_rejects_uncompressed_total_above_cap(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    zip_path = workspace / "bulky.zip"
    _write_zip(zip_path, {"images/a.jpg": b"y" * 80})
    with pytest.raises(LabError) as caught:
        import_coco(workspace, "pilot", "vid__train", zip_path, max_unpack_bytes=40)
    assert caught.value.code == "zip_too_large"
    assert DEFAULT_MAX_UNPACK_BYTES == 2 * 1024**3
    assert list((workspace / "projects" / "pilot" / "annotation_batches").iterdir()) == []


def test_import_rejects_dump_category_id_zero(tmp_path: Path) -> None:
    workspace = _frozen_project(tmp_path)
    payload = _jpeg(32, 24, 12)
    zip_path = workspace / "dump.zip"
    _write_zip(zip_path, _single_members(payload, category_id=0, category_name="dump"))
    with pytest.raises(UnknownCategoryError) as caught:
        import_coco(workspace, "pilot", "vid__train", zip_path)
    assert caught.value.category_name == "dump"
    assert caught.value.class_id is None
    assert not (_batch(workspace, "vid__train") / "imported_labels.json").exists()


def test_import_maps_shovel_by_name_not_category_id(tmp_path: Path) -> None:
    workspace = _frozen_project(tmp_path)
    add_class(workspace, "excavation", "shovel", "Лопата", "Видимая лопата.")
    catalog = require_project_catalog(workspace, "pilot")
    shovel_id = next(item.id for item in catalog.classes if item.name == "shovel")
    assert shovel_id == 4
    payload = _jpeg(32, 24, 12)
    zip_path = workspace / "shovel.zip"
    _write_zip(zip_path, _single_members(payload, category_id=0, category_name="shovel"))
    result = import_coco(workspace, "pilot", "vid__train", zip_path)
    labels = json.loads(Path(result.labels_path).read_text(encoding="utf-8"))
    box = labels["frames"][0]["record"]["boxes"][0]
    assert box["class_name"] == "shovel"
    assert box["class_id"] == shovel_id
    assert box["class_name"] != "excavator_body"


def test_four_excavation_names_keep_catalog_ids_after_shovel(tmp_path: Path) -> None:
    workspace = _frozen_project(tmp_path)
    add_class(workspace, "excavation", "shovel", "Лопата", "Видимая лопата.")
    payload = _jpeg(32, 24, 30)
    categories = (
        (0, "bucket"),
        (7, "excavator_body"),
        (4, "haul_truck"),
        (8, "truck_bed"),
    )
    boxes = (
        (0, [1.0, 1.0, 4.0, 4.0]),
        (7, [6.0, 6.0, 4.0, 4.0]),
        (4, [12.0, 2.0, 4.0, 4.0]),
        (8, [2.0, 12.0, 4.0, 4.0]),
    )
    zip_path = workspace / "four.zip"
    _write_zip(zip_path, _labeled_members(payload, categories=categories, boxes=boxes))
    result = import_coco(workspace, "pilot", "vid__train", zip_path)
    labels = json.loads(Path(result.labels_path).read_text(encoding="utf-8"))
    stored = {
        box["class_name"]: box["class_id"] for box in labels["frames"][0]["record"]["boxes"]
    }
    assert stored == {
        "bucket": 1,
        "excavator_body": 0,
        "haul_truck": 2,
        "truck_bed": 3,
    }
    assert "shovel" not in stored


def test_old_zip_keeps_pre_shovel_review(tmp_path: Path) -> None:
    workspace = _frozen_project(tmp_path)
    add_class(workspace, "excavation", "shovel", "Лопата", "Видимая лопата.")
    catalog = require_project_catalog(workspace, "pilot")
    shovel = next(item for item in catalog.classes if item.name == "shovel")
    seed_at = next(item.added_at for item in catalog.classes if item.name == "bucket")
    assert shovel.added_at > seed_at
    reviewed_dt = seed_at + (shovel.added_at - seed_at) / 2
    assert seed_at < reviewed_dt < shovel.added_at
    reviewed = reviewed_dt.isoformat()
    payload = _jpeg(32, 24, 40)
    first = workspace / "with-shovel.zip"
    _write_zip(
        first,
        _labeled_members(
            payload,
            categories=((0, "shovel"), (99, "bucket")),
            boxes=((0, [8.0, 8.0, 4.0, 4.0]), (99, [2.0, 2.0, 4.0, 4.0])),
            reviewed_at=reviewed,
            note="старая заметка",
        ),
    )
    import_coco(workspace, "pilot", "vid__train", first)
    before = _stored_record(workspace)
    later = (shovel.added_at + timedelta(hours=1)).isoformat()
    second = workspace / "old-four.zip"
    _write_zip(
        second,
        _labeled_members(
            payload,
            categories=(
                (0, "excavator_body"),
                (1, "bucket"),
                (2, "haul_truck"),
                (3, "truck_bed"),
            ),
            boxes=((1, [1.0, 1.0, 5.0, 5.0]),),
            reviewed_at=later,
            note="импорт",
        ),
    )
    import_coco(workspace, "pilot", "vid__train", second)
    after = _stored_record(workspace)
    assert after["reviewed_at"] == before["reviewed_at"]
    assert after["reviewed_at"] != later
    assert _named_box(after, "shovel") == _named_box(before, "shovel")
    assert _named_box(after, "bucket") != _named_box(before, "bucket")
    assert _named_box(after, "bucket")["x1"] == 1.0
    supervision = class_supervision(LabelRecord.model_validate(after), catalog)
    assert "shovel" not in supervision.negatives
    assert "shovel" in supervision.unsupervised


def test_post_shovel_review_is_left_stored(tmp_path: Path) -> None:
    workspace = _frozen_project(tmp_path)
    add_class(workspace, "excavation", "shovel", "Лопата", "Видимая лопата.")
    catalog = require_project_catalog(workspace, "pilot")
    shovel = next(item for item in catalog.classes if item.name == "shovel")
    reviewed = shovel.added_at.isoformat()
    payload = _jpeg(32, 24, 50)
    first = workspace / "reviewed-after.zip"
    _write_zip(
        first,
        _labeled_members(
            payload,
            categories=((0, "shovel"), (99, "bucket")),
            boxes=((0, [8.0, 8.0, 4.0, 4.0]), (99, [2.0, 2.0, 3.0, 3.0])),
            reviewed_at=reviewed,
            note="не трогать",
        ),
    )
    import_coco(workspace, "pilot", "vid__train", first)
    before = _stored_record(workspace)
    second = workspace / "old-four.zip"
    _write_zip(
        second,
        _labeled_members(
            payload,
            categories=(
                (0, "excavator_body"),
                (1, "bucket"),
                (2, "haul_truck"),
                (3, "truck_bed"),
            ),
            boxes=((1, [1.0, 1.0, 5.0, 5.0]),),
            reviewed_at=(shovel.added_at + timedelta(days=1)).isoformat(),
            note="замена",
        ),
    )
    import_coco(workspace, "pilot", "vid__train", second)
    after = _stored_record(workspace)
    assert after == before
    assert _named_box(after, "shovel")["class_name"] == "shovel"
    assert after["note"] == "не трогать"
    stored_review = datetime.fromisoformat(str(after["reviewed_at"]))
    assert stored_review == shovel.added_at


def test_import_rejects_iscrowd(tmp_path: Path) -> None:
    workspace = _frozen_project(tmp_path)
    payload = _jpeg(32, 24, 12)
    members = _single_members(payload, category_id=99, category_name="bucket")
    document = json.loads(members["annotations/instances_default.json"])
    document["annotations"][0]["iscrowd"] = 1
    document["annotations"][0]["attributes"] = {"visibility": "partial"}
    members["annotations/instances_default.json"] = _json(document)
    zip_path = workspace / "crowd.zip"
    _write_zip(zip_path, members)
    with pytest.raises(UnsupportedCrowdError) as caught:
        import_coco(workspace, "pilot", "vid__train", zip_path)
    assert caught.value.iscrowd == 1
    assert caught.value.visibility is None


def test_import_checks_hash_size_and_duplicate_ids(tmp_path: Path) -> None:
    workspace = _frozen_project(tmp_path)
    payload = _jpeg(32, 24, 12)
    members = _single_members(payload, category_id=99, category_name="bucket")
    text = members["frames_manifest.csv"].decode("utf-8")
    members["frames_manifest.csv"] = text.replace(
        hashlib.sha256(payload).hexdigest(),
        "0" * 64,
    ).encode("utf-8")
    zip_path = workspace / "hash.zip"
    _write_zip(zip_path, members)
    with pytest.raises(LabError) as caught:
        import_coco(workspace, "pilot", "vid__train", zip_path)
    assert caught.value.code == "image_hash_mismatch"
    assert "vid:1" in caught.value.message

    members = _single_members(payload, category_id=99, category_name="bucket", size_bytes=len(payload) - 1)
    zip_path = workspace / "size.zip"
    _write_zip(zip_path, members)
    with pytest.raises(LabError) as caught:
        import_coco(workspace, "pilot", "vid__train", zip_path)
    assert caught.value.code == "image_size_mismatch"

    members = _single_members(payload, category_id=99, category_name="bucket")
    document = json.loads(members["annotations/instances_default.json"])
    document["images"].append(dict(document["images"][0]))
    members["annotations/instances_default.json"] = _json(document)
    zip_path = workspace / "dup-image.zip"
    _write_zip(zip_path, members)
    with pytest.raises(LabError) as caught:
        import_coco(workspace, "pilot", "vid__train", zip_path)
    assert caught.value.code == "duplicate_image_id"

    members = _single_members(payload, category_id=99, category_name="bucket")
    document = json.loads(members["annotations/instances_default.json"])
    document["annotations"].append(dict(document["annotations"][0]))
    members["annotations/instances_default.json"] = _json(document)
    zip_path = workspace / "dup-ann.zip"
    _write_zip(zip_path, members)
    with pytest.raises(LabError) as caught:
        import_coco(workspace, "pilot", "vid__train", zip_path)
    assert caught.value.code == "duplicate_annotation_id"


def test_import_rejects_missing_image_and_split_crossover(tmp_path: Path) -> None:
    workspace = _frozen_project(tmp_path)
    payload = _jpeg(32, 24, 12)
    members = _single_members(payload, category_id=99, category_name="bucket")
    del members["images/vid_1.jpg"]
    zip_path = workspace / "missing.zip"
    _write_zip(zip_path, members)
    with pytest.raises(LabError) as caught:
        import_coco(workspace, "pilot", "vid__train", zip_path)
    assert caught.value.code == "image_missing"
    assert "vid:1" in caught.value.message

    other = _jpeg(32, 24, 30)
    first = _row(
        image_file="vid_1.jpg",
        payload=payload,
        index=1,
        episode_id="train-ep",
        split="train",
        group="g-train",
        time_s=10,
        width=32,
        height=24,
    )
    second = _row(
        image_file="vid_2.jpg",
        payload=other,
        index=2,
        episode_id="val-ep",
        split="validation",
        group="g-train",
        time_s=80,
        width=32,
        height=24,
    )
    zip_path = workspace / "cross.zip"
    _write_zip(
        zip_path,
        {
            "frames_manifest.csv": _manifest([first, second]),
            "images/vid_1.jpg": payload,
            "images/vid_2.jpg": other,
        },
    )
    with pytest.raises(LabError) as caught:
        import_coco(workspace, "pilot", "vid__train", zip_path)
    assert caught.value.code == "split_group_crossover"
    assert "g-train" in caught.value.message


def test_import_rejects_a_box_outside_the_image(tmp_path: Path) -> None:
    workspace = _frozen_project(tmp_path)
    payload = _jpeg(32, 24, 12)
    members = _single_members(payload, category_id=99, category_name="bucket")
    document = json.loads(members["annotations/instances_default.json"])
    document["annotations"][0]["bbox"] = [0, 0, 33, 1]
    members["annotations/instances_default.json"] = _json(document)
    zip_path = workspace / "outside.zip"
    _write_zip(zip_path, members)
    with pytest.raises(LabError) as caught:
        import_coco(workspace, "pilot", "vid__train", zip_path)
    assert caught.value.code == "invalid_bbox"
    assert not (_batch(workspace, "vid__train") / "imported_labels.json").exists()


def _stored_record(workspace: Path) -> dict[str, object]:
    labels = json.loads(
        (_batch(workspace, "vid__train") / "imported_labels.json").read_text(encoding="utf-8")
    )
    return labels["frames"][0]["record"]


def _named_box(record: dict[str, object], name: str) -> dict[str, object]:
    boxes = record["boxes"]
    assert isinstance(boxes, list)
    return next(box for box in boxes if box["class_name"] == name)


def _labeled_members(
    payload: bytes,
    *,
    categories: tuple[tuple[int, str], ...],
    boxes: tuple[tuple[int, list[float]], ...],
    reviewed_at: str | None = None,
    note: str | None = None,
) -> dict[str, bytes]:
    image_file = "vid_1.jpg"
    width, height = 32, 24
    attributes: dict[str, object] = {"review_status": "reviewed"}
    if reviewed_at is not None:
        attributes["reviewed_at"] = reviewed_at
    if note is not None:
        attributes["note"] = note
    return {
        "frames_manifest.csv": _manifest(
            [
                _row(
                    image_file=image_file,
                    payload=payload,
                    index=1,
                    episode_id="train-ep",
                    split="train",
                    group="g-train",
                    time_s=10,
                    width=width,
                    height=height,
                )
            ]
        ),
        "annotations/instances_default.json": _json(
            {
                "categories": [
                    {"id": category_id, "name": name, "supercategory": ""}
                    for category_id, name in categories
                ],
                "images": [
                    {
                        "id": 1,
                        "file_name": f"images/{image_file}",
                        "width": width,
                        "height": height,
                        "attributes": attributes,
                    }
                ],
                "annotations": [
                    {
                        "id": annotation_id,
                        "image_id": 1,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                    for annotation_id, (category_id, bbox) in enumerate(boxes, start=1)
                ],
            }
        ),
        f"images/{image_file}": payload,
    }


def _single_members(
    payload: bytes,
    *,
    category_id: int,
    category_name: str,
    size_bytes: int | None = None,
) -> dict[str, bytes]:
    image_file = "vid_1.jpg"
    width, height = 32, 24
    return {
        "frames_manifest.csv": _manifest(
            [
                _row(
                    image_file=image_file,
                    payload=payload,
                    index=1,
                    episode_id="train-ep",
                    split="train",
                    group="g-train",
                    time_s=10,
                    width=width,
                    height=height,
                    size_bytes=size_bytes,
                )
            ]
        ),
        "annotations/instances_default.json": _json(
            {
                "categories": [
                    {"id": category_id, "name": category_name, "supercategory": ""}
                ],
                "images": [
                    {
                        "id": 1,
                        "file_name": f"images/{image_file}",
                        "width": width,
                        "height": height,
                        "attributes": {"review_status": "reviewed"},
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": category_id,
                        "bbox": [1, 1, 4, 4],
                        "area": 16,
                        "iscrowd": 0,
                    }
                ],
            }
        ),
        f"images/{image_file}": payload,
    }
