"""Write an RF-DETR dataset directory from imported labels and a frozen split.

The directory for validation is ``valid``, matching ``build_roboflow_from_coco``.
A CVAT subset name is not a split. Train COCO contains only reviewed frames
that are not excluded. ``class_supervision.json`` in each split is the
authority for a negative. A missing COCO box is not one by itself.
This module does not propose a split, does not open ``weights.pth``, and
does not train.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from excavation_lab.datasets.blocks import assignment_at as role_at
from excavation_lab.datasets.catalog import (
    StoredCatalog,
    require_catalog_file,
    require_project_catalog,
)
from excavation_lab.datasets.class_supervision import class_supervision
from excavation_lab.datasets.coco_import import (
    ImportedFrame,
    build_class_mapping,
    load_imported_frames,
)
from excavation_lab.datasets.geometry import xyxy_to_xywh
from excavation_lab.datasets.labels import LabelRecord, LabelState
from excavation_lab.datasets.splits import (
    FrozenSplit,
    canonical_json_bytes,
    frozen_manifest_path,
    require_frozen,
)
from excavation_lab.errors import LabError, SplitNotFrozenError
from excavation_lab.schemas import SCHEMA_VERSION, ReviewStatus, SplitName, StrictModel
from excavation_lab.storage import connect, project_directory, project_sqlite

# The sidecar, not a missing COCO box, decides a negative.
_NEGATIVE_AUTHORITY = (
    "class_supervision.json is the authority for negative. "
    "A missing COCO box is not, by itself, a negative."
)
_SUPERVISION_FILE = "class_supervision.json"

# RF-DETR reads ``valid``. The internal split name stays ``validation``.
RFDETR_SPLIT_DIRECTORIES: dict[SplitName, str] = {
    SplitName.train: "train",
    SplitName.validation: "valid",
    SplitName.test: "test",
}


class SnapshotResult(StrictModel):
    """One ``dataset_versions/<dataset_id>`` tree."""

    dataset_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: int = SCHEMA_VERSION


def snapshot_dataset(workspace: Path, project_id: str) -> SnapshotResult:
    """Freeze-check every video, then write ``train`` / ``valid`` / ``test``.

    Missing ``split_manifest.json`` raises ``SplitNotFrozenError``.
    """
    project_dir = project_directory(workspace, project_id)
    video_ids = _video_ids(workspace, project_id)
    if not video_ids:
        raise LabError(
            "video_not_found",
            "В проекте нет видео. Снимок датасета не создаётся.",
        )
    frozen = {
        video_id: require_frozen(workspace, project_id, video_id) for video_id in video_ids
    }
    frames = load_imported_frames(workspace, project_id)
    _reject_duplicates(frames)
    _reject_group_crossover(frames)
    for frame in frames:
        _assert_frozen_split(frozen, frame)
    catalog = require_project_catalog(workspace, project_id)
    catalog_id = _project_catalog_id(workspace, project_id)
    catalog_file = require_catalog_file(workspace, catalog_id)
    catalog_hash = hashlib.sha256(catalog_file.read_bytes()).hexdigest()
    mapping = build_class_mapping(catalog)
    weights_id = _selected_weights_id(workspace, project_id)
    grouped: dict[SplitName, list[ImportedFrame]] = {split: [] for split in SplitName}
    for frame in frames:
        if _in_training_coco(frame.record):
            grouped[frame.split].append(frame)
    for items in grouped.values():
        items.sort(key=lambda item: (item.video_id, item.source_frame_index, item.image_file))
    coco = {
        RFDETR_SPLIT_DIRECTORIES[split]: _coco_document(items, mapping)
        for split, items in grouped.items()
    }
    supervision = {
        RFDETR_SPLIT_DIRECTORIES[split]: _supervision_document(items, catalog)
        for split, items in grouped.items()
    }
    split_hashes = {video_id: item.content_hash for video_id, item in frozen.items()}
    split_content_hash = _combined_split_hash(split_hashes)
    dataset_hash = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "catalog_hash": catalog_hash,
                "catalog_id": catalog_id,
                "class_mapping": mapping,
                "class_supervision": supervision,
                "split_content_hashes": split_hashes,
                "splits": coco,
                "weights_id": weights_id,
            }
        )
    ).hexdigest()
    dataset_id = uuid.uuid4().hex
    destination = _write_tree(
        project_dir,
        dataset_id=dataset_id,
        project_id=project_id,
        video_ids=video_ids,
        workspace=workspace,
        grouped=grouped,
        coco=coco,
        mapping=mapping,
        supervision=supervision,
        catalog_id=catalog_id,
        catalog_hash=catalog_hash,
        weights_id=weights_id,
        dataset_hash=dataset_hash,
        split_content_hash=split_content_hash,
        split_hashes=split_hashes,
    )
    _record_dataset(workspace, project_id, dataset_id, dataset_hash, destination)
    return SnapshotResult(
        dataset_id=dataset_id,
        project_id=project_id,
        path=str(destination),
        dataset_hash=dataset_hash,
        split_content_hash=split_content_hash,
        schema_version=SCHEMA_VERSION,
    )


def format_snapshot_summary(result: SnapshotResult) -> str:
    """Russian operator text: path, dataset hash, frozen split hash."""
    return "\n".join(
        [
            result.path,
            f"dataset_hash: {result.dataset_hash}",
            f"split_content_hash: {result.split_content_hash}",
        ]
    )


def _in_training_coco(record: LabelRecord) -> bool:
    """Reviewed frames only. Unreviewed and excluded stay out of train COCO."""
    if record.review_status is not ReviewStatus.reviewed:
        return False
    if record.label_state is LabelState.excluded:
        return False
    return record.trainable is True


def _video_ids(workspace: Path, project_id: str) -> list[str]:
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT video_id FROM videos
            WHERE project_id = ?
            ORDER BY video_id
            """,
            (project_id,),
        ).fetchall()
    return [str(row["video_id"]) for row in rows]


def _reject_duplicates(frames: list[ImportedFrame]) -> None:
    seen: set[tuple[str, int]] = set()
    for frame in frames:
        key = (frame.video_id, frame.source_frame_index)
        if key in seen:
            raise LabError(
                "duplicate_frame",
                f"Кадр {frame.video_id}:{frame.source_frame_index} импортирован дважды.",
            )
        seen.add(key)


def _reject_group_crossover(frames: list[ImportedFrame]) -> None:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for frame in frames:
        if not frame.split_group_id:
            continue
        grouped[(frame.video_id, frame.split_group_id)].add(frame.split.value)
    for (video_id, group_id), splits in grouped.items():
        if len(splits) > 1:
            joined = ", ".join(sorted(splits))
            raise LabError(
                "split_group_crossover",
                f"Группа «{group_id}» видео «{video_id}» попала в несколько частей: {joined}.",
            )


def _assert_frozen_split(
    frozen_by_video: dict[str, FrozenSplit],
    frame: ImportedFrame,
) -> None:
    frozen = frozen_by_video.get(frame.video_id)
    if frozen is None:
        raise SplitNotFrozenError(frame.video_id)
    expected = role_at(frozen.blocks, frame.source_time_s)
    if expected is None:
        raise LabError(
            "frame_unassigned",
            f"Кадр {frame.video_id}:{frame.source_frame_index} не попадает "
            "в зафиксированное разбиение. Split не пересчитывается.",
        )
    if frame.split.value != expected:
        raise LabError(
            "split_mismatch",
            f"Кадр {frame.video_id}:{frame.source_frame_index} не совпадает "
            "с зафиксированным разбиением. Split не пересчитывается.",
        )


def _combined_split_hash(split_hashes: dict[str, str]) -> str:
    if len(split_hashes) == 1:
        return next(iter(split_hashes.values()))
    return hashlib.sha256(
        canonical_json_bytes({"split_content_hashes": split_hashes})
    ).hexdigest()


def _coco_document(frames: list[ImportedFrame], mapping: dict[str, Any]) -> dict[str, Any]:
    by_name: dict[str, dict[str, int]] = mapping["by_name"]
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, frame in enumerate(frames, start=1):
        if frame.image_file in seen_names:
            raise LabError(
                "duplicate_image",
                f"Файл «{frame.image_file}» повторяется в одной части снимка.",
            )
        seen_names.add(frame.image_file)
        images.append(
            {
                "id": index,
                "file_name": frame.image_file,
                "width": frame.width,
                "height": frame.height,
            }
        )
        for box in frame.record.boxes:
            spec = by_name.get(box.class_name)
            if (
                spec is None
                or spec["internal_id"] != box.class_id
                or spec["coco_category_id"] != box.class_id
            ):
                raise LabError(
                    "class_map",
                    f"Кадр {frame.video_id}:{frame.source_frame_index}: класс "
                    f"«{box.class_name}» не совпадает с class_mapping.json.",
                )
            try:
                x, y, width, height = xyxy_to_xywh(
                    (box.x1, box.y1, box.x2, box.y2),
                    image_width=frame.width,
                    image_height=frame.height,
                )
            except (TypeError, ValueError) as exc:
                raise LabError(
                    "invalid_bbox",
                    f"Кадр {frame.video_id}:{frame.source_frame_index}, "
                    f"файл {frame.image_file}: рамка вне изображения "
                    "или с неположительным размером.",
                ) from exc
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": index,
                    "category_id": spec["coco_category_id"],
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
    categories = [
        {"id": spec["coco_category_id"], "name": name, "supercategory": "none"}
        for name, spec in by_name.items()
    ]
    categories.sort(key=lambda item: int(item["id"]))
    return {
        "info": {
            "description": "Excavation Video Lab detection COCO",
            "version": str(SCHEMA_VERSION),
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }


def _supervision_document(
    frames: list[ImportedFrame],
    catalog: StoredCatalog,
) -> dict[str, Any]:
    """One record per trainable image, keyed by image id and by file name.

    Both keys are the same record. Image ids match ``_coco_document``.
    """
    images: dict[str, dict[str, Any]] = {}
    for index, frame in enumerate(frames, start=1):
        supervision = class_supervision(frame.record, catalog)
        record = {
            "file_name": frame.image_file,
            "image_id": index,
            "negatives": sorted(supervision.negatives),
            "positives": sorted(supervision.positives),
            "unsupervised": sorted(supervision.unsupervised),
        }
        images[str(index)] = record
        images[frame.image_file] = record
    return {"images": images, "schema_version": SCHEMA_VERSION}


def _write_tree(
    project_dir: Path,
    *,
    dataset_id: str,
    project_id: str,
    video_ids: list[str],
    workspace: Path,
    grouped: dict[SplitName, list[ImportedFrame]],
    coco: dict[str, dict[str, Any]],
    mapping: dict[str, Any],
    supervision: dict[str, dict[str, Any]],
    catalog_id: str,
    catalog_hash: str,
    weights_id: str | None,
    dataset_hash: str,
    split_content_hash: str,
    split_hashes: dict[str, str],
) -> Path:
    versions = project_dir / "dataset_versions"
    versions.mkdir(parents=True, exist_ok=True)
    partial = versions / f".{dataset_id}.partial"
    final = versions / dataset_id
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    try:
        for split, directory_name in RFDETR_SPLIT_DIRECTORIES.items():
            if directory_name == "val":
                raise LabError(
                    "invalid_layout",
                    "Каталог снимка называется valid, не val.",
                )
            split_dir = partial / directory_name
            split_dir.mkdir()
            for frame in grouped[split]:
                target = split_dir / frame.image_file
                shutil.copyfile(frame.jpeg_path, target)
            _write_json(split_dir / "_annotations.coco.json", coco[directory_name])
            _write_json(split_dir / _SUPERVISION_FILE, supervision[directory_name])
        _write_json(partial / "class_mapping.json", mapping)
        _write_split_manifest(partial / "split_manifest.json", workspace, project_id, video_ids)
        _write_json(
            partial / "dataset_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_id": dataset_id,
                "project_id": project_id,
                "dataset_hash": dataset_hash,
                "split_content_hash": split_content_hash,
                "split_content_hashes": split_hashes,
                "catalog_id": catalog_id,
                "catalog_hash": catalog_hash,
                "weights_id": weights_id,
                "class_supervision_file": _SUPERVISION_FILE,
                "negative_authority": _NEGATIVE_AUTHORITY,
            },
        )
        os.replace(partial, final)
    except Exception:
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
        raise
    return final.resolve()


def _write_split_manifest(
    destination: Path,
    workspace: Path,
    project_id: str,
    video_ids: list[str],
) -> None:
    paths = [frozen_manifest_path(workspace, project_id, video_id) for video_id in video_ids]
    if len(paths) == 1:
        destination.write_bytes(paths[0].read_bytes())
        return
    document = {
        "schema_version": SCHEMA_VERSION,
        "videos": {
            video_id: json.loads(path.read_text(encoding="utf-8"))
            for video_id, path in zip(video_ids, paths, strict=True)
        },
    }
    _write_json(destination, document)


def _write_json(path: Path, document: dict[str, Any]) -> None:
    text = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _project_catalog_id(workspace: Path, project_id: str) -> str:
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        row = connection.execute("SELECT catalog_id FROM projects LIMIT 1").fetchone()
    catalog_id = None if row is None else row["catalog_id"]
    if not isinstance(catalog_id, str) or not catalog_id.strip():
        raise LabError(
            "catalog_missing",
            f"У проекта «{project_id}» нет каталога.",
        )
    return catalog_id


def _selected_weights_id(workspace: Path, project_id: str) -> str | None:
    """The one published bundle id in ``model_versions``, or null.

    Zero rows and more than one row are not a selection. The path is not
    opened and ``weights.pth`` is not read.
    """
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT model_id FROM model_versions
            WHERE project_id = ?
            ORDER BY created_at, model_id
            """,
            (project_id,),
        ).fetchall()
    if len(rows) != 1:
        return None
    model_id = rows[0]["model_id"]
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    return model_id


def _record_dataset(
    workspace: Path,
    project_id: str,
    dataset_id: str,
    dataset_hash: str,
    destination: Path,
) -> None:
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO dataset_versions (
                dataset_id, project_id, content_hash, path, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                project_id,
                dataset_hash,
                str(destination),
                datetime.now(UTC).isoformat(),
            ),
        )
