"""Write an RF-DETR dataset directory from imported labels and a frozen split.

The directory for validation is ``valid``, matching ``build_roboflow_from_coco``.
A CVAT subset name is not a split. Train COCO contains only reviewed frames
that are not excluded. This module does not propose a split and does not train.
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
    mapping = build_class_mapping()
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
    split_hashes = {video_id: item.content_hash for video_id, item in frozen.items()}
    split_content_hash = _combined_split_hash(split_hashes)
    dataset_hash = hashlib.sha256(
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "class_mapping": mapping,
                "split_content_hashes": split_hashes,
                "splits": coco,
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
    if frame.episode_id:
        matches = [
            episode for episode in frozen.episodes if episode.episode_id == frame.episode_id
        ]
        if len(matches) != 1:
            raise LabError(
                "episode_not_found",
                f"Кадр {frame.video_id}:{frame.source_frame_index}: эпизод "
                f"«{frame.episode_id}» не найден в зафиксированном разбиении. "
                "Split не пересчитывается.",
            )
        episode = matches[0]
        expected_split, expected_group = episode.split, episode.split_group_id
    else:
        expected_split, expected_group = _block_assignment(frozen, frame)
    if frame.split is not expected_split or frame.split_group_id != expected_group:
        raise LabError(
            "split_mismatch",
            f"Кадр {frame.video_id}:{frame.source_frame_index} не совпадает "
            "с зафиксированным разбиением. Split не пересчитывается.",
        )


def _block_assignment(frozen: FrozenSplit, frame: ImportedFrame) -> tuple[SplitName, str]:
    for block in frozen.blocks:
        if block.role not in {"train", "validation", "test"}:
            continue
        if block.start_s <= frame.source_time_s < block.end_s:
            return SplitName(block.role), ""
    raise LabError(
        "frame_unassigned",
        f"Кадр {frame.video_id}:{frame.source_frame_index} не попадает "
        "в зафиксированное разбиение. Split не пересчитывается.",
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
            spec = by_name.get(box.class_name.value)
            if (
                spec is None
                or spec["internal_id"] != box.class_id
                or spec["coco_category_id"] != box.class_id
            ):
                raise LabError(
                    "class_map",
                    f"Кадр {frame.video_id}:{frame.source_frame_index}: класс "
                    f"«{box.class_name.value}» не совпадает с class_mapping.json.",
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
