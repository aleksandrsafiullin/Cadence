"""Import a CVAT COCO zip into stored labels.

Split comes from ``frames_manifest.csv``. A CVAT subset directory or an
``instances_<subset>.json`` name does not move a frame. Category ids are
not internal ids: the category ``name`` is matched to the project catalog.
``iscrowd`` other than 0 is refused. Attributes COCO training files do not
carry are written to ``attributes_sidecar.json``.

A frame already stored keeps its ``reviewed_at``. The import clock is not
written onto that field. Boxes of catalog classes the ZIP does not list
stay, and that missing class is not stored as a negative. A review at or
after ``added_at`` of such a class is left as stored: boxes, ``reviewed_at``,
and the note. The stored frame is not deleted.

A frame already in this batch and absent from the ZIP stays, including
its JPEG. The ZIP replaces only the frames it contains.

This module does not propose a split and does not start training.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import sqlite3
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from pydantic import Field, ValidationError

from excavation_lab.datasets.blocks import assignment_at as role_at
from excavation_lab.datasets.catalog import (
    StoredCatalog,
    StoredCatalogClass,
    require_project_catalog,
)
from excavation_lab.datasets.cvat_export import (
    MANIFEST_COLUMNS,
    _parse_batch_id,
    safe_member_name,
)
from excavation_lab.datasets.geometry import xywh_to_xyxy
from excavation_lab.datasets.label_store import (
    FrameWrite,
    _frame_label_document,
    _frame_sidecar_document,
    batch_directory,
    upsert_frames,
)
from excavation_lab.datasets.labels import (
    BoxDraft,
    FrameDraft,
    LabelRecord,
    LabelState,
    validate_frame,
)
from excavation_lab.datasets.splits import FrozenSplit, require_frozen
from excavation_lab.errors import (
    LabError,
    UnknownCategoryError,
    UnsupportedCrowdError,
)
from excavation_lab.schemas import (
    SCHEMA_VERSION,
    AnnotationBox,
    CatalogConfig,
    ReviewStatus,
    SplitName,
    StrictModel,
    Visibility,
)
from excavation_lab.storage import (
    connect,
    get_video,
    project_directory,
    project_sqlite,
    resolve_project_path,
)

# 2 GiB. Tests pass a smaller cap and do not allocate this.
DEFAULT_MAX_UNPACK_BYTES = 2 * 1024**3

_COCO_IMAGE_KEYS = frozenset(
    {
        "id",
        "file_name",
        "width",
        "height",
        "license",
        "flickr_url",
        "coco_url",
        "date_captured",
        "attributes",
    }
)
_COCO_ANNOTATION_KEYS = frozenset(
    {
        "id",
        "image_id",
        "category_id",
        "bbox",
        "area",
        "iscrowd",
        "segmentation",
        "keypoints",
        "num_keypoints",
        "attributes",
    }
)
_ZIP_TOO_LARGE = (
    "Суммарный размер файлов в архиве больше допустимого. Архив не распакован."
)
_SKIP_JSON = frozenset({"predictions.coco.json", "batch.json"})


class ImportResult(StrictModel):
    """What ``import-coco`` wrote from one ZIP. The original zip is ``source_zip``.

    ``image_count`` is how many frames the ZIP contributed. It is not the
    number of frames stored in the batch afterwards. Frames already in the
    batch and absent from the ZIP stay.
    """

    batch_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    source_zip: str = Field(min_length=1)
    sidecar_path: str = Field(min_length=1)
    labels_path: str = Field(min_length=1)
    image_count: int = Field(
        ge=1,
        description="Frames the ZIP contributed, not the total stored in the batch.",
    )
    reviewed_empty: int = Field(ge=0)
    flagged_unreviewed: int = Field(ge=0)
    stored_excluded: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class _ManifestRow:
    image_file: str
    video_id: str
    source_frame_index: int
    source_pts: int
    time_base: str
    source_time_s: float
    episode_id: str
    split: SplitName
    split_group_id: str
    image_sha256: str
    width: int
    height: int
    size_bytes: int | None

    @property
    def frame_id(self) -> str:
        return f"{self.video_id}:{self.source_frame_index}"


@dataclass(frozen=True, slots=True)
class _Built:
    row: _ManifestRow
    payload: bytes
    coco_subset: str
    image_attributes: dict[str, Any]
    sidecar_boxes: list[dict[str, Any]]
    record: LabelRecord
    flagged: list[str]


@dataclass(frozen=True, slots=True)
class ImportedFrame:
    """One stored frame. ``coco_subset`` is not the split."""

    image_file: str
    video_id: str
    source_frame_index: int
    episode_id: str
    split: SplitName
    split_group_id: str
    source_time_s: float
    image_sha256: str
    size_bytes: int
    width: int
    height: int
    coco_subset: str
    jpeg_path: Path
    flagged: tuple[str, ...]
    record: LabelRecord


def build_class_mapping(catalog: StoredCatalog | CatalogConfig) -> dict[str, Any]:
    """Name → catalog id. The COCO category id in a snapshot is this id.

    The installed RF-DETR loader remaps category ids on its own. Id 0 is a
    class, not background. The catalog is an argument.
    """
    by_name = {
        item.name: {
            "internal_id": item.id,
            "coco_category_id": item.id,
        }
        for item in catalog.classes
    }
    return {"schema_version": SCHEMA_VERSION, "by_name": by_name}


def resolve_category_name(
    category_name: str,
    category_id: int,
    *,
    catalog: StoredCatalog | CatalogConfig,
    frame_id: str | None = None,
) -> int:
    """Return the catalog id for ``category_name``.

    ``category_id`` is not the internal id. A known name keeps that catalog
    id. An unknown name raises ``UnknownCategoryError`` and is not snapped
    to a neighbor or to id 0.
    """
    if isinstance(category_id, bool) or not isinstance(category_id, int):
        raise LabError(
            "invalid_category_id",
            "category_id должен быть целым. Класс выбирается по имени."
            + (f" Кадр {frame_id}." if frame_id else ""),
        )
    if isinstance(category_name, bool) or not isinstance(category_name, str):
        raise UnknownCategoryError(str(category_name), frame_id)
    name = category_name.strip()
    found = build_class_mapping(catalog)["by_name"].get(name)
    if found is None:
        raise UnknownCategoryError(name, frame_id)
    return int(found["internal_id"])


def import_coco(
    workspace: Path,
    project_id: str,
    batch_id: str,
    file: Path,
    *,
    max_unpack_bytes: int = DEFAULT_MAX_UNPACK_BYTES,
) -> ImportResult:
    """Check one CVAT COCO zip and store labels for a later snapshot.

    Every image must exist. Its byte size, when ``size_bytes`` is in the
    manifest, and its SHA-256 must match. ``image_id`` and ``annotation_id``
    are unique. One ``split_group_id`` has one split.
    """
    video_id, batch_split = _parse_batch_id(batch_id)
    if isinstance(max_unpack_bytes, bool) or not isinstance(max_unpack_bytes, int):
        raise LabError("zip_too_large", _ZIP_TOO_LARGE)
    if max_unpack_bytes < 0:
        raise LabError("zip_too_large", _ZIP_TOO_LARGE)
    project_dir = project_directory(workspace, project_id)
    source = _resolve_zip(workspace, project_id, file)
    members = _read_zip(source, max_unpack_bytes)
    rows = _parse_manifest(members, video_id)
    _reject_group_crossover(rows)
    images, annotations, categories = _merge_coco(members)
    _pair_manifest(rows, images)
    get_video(workspace, project_id, video_id)
    frozen = require_frozen(workspace, project_id, video_id)
    catalog = require_project_catalog(workspace, project_id)
    built = [
        _build_frame(
            row,
            images[row.image_file],
            annotations,
            categories,
            members,
            frozen,
            catalog,
        )
        for row in rows
    ]
    stored_frames, stored_sidecars = _indexed_imports(project_dir)
    zip_names = {name.strip() for name in categories.values()}
    built = [
        _merge_stored(
            item,
            stored_frames.get((item.row.video_id, item.row.source_frame_index)),
            stored_sidecars.get((item.row.video_id, item.row.source_frame_index)),
            catalog,
            zip_names,
        )
        for item in built
    ]
    batch_dir = batch_directory(workspace, project_id, video_id, batch_split)
    _publish(workspace, project_id, video_id, batch_split, built, frozen)
    source_zip = batch_dir / "source.zip"
    _copy_source(source, source_zip)
    _record_batch(workspace, project_id, batch_id, video_id, source_zip)
    flagged = sum(1 for item in built if "unreviewed" in item.flagged)
    excluded = sum(
        1 for item in built if item.record.review_status is ReviewStatus.excluded
    )
    empty = sum(
        1 for item in built if item.record.label_state is LabelState.reviewed_empty
    )
    return ImportResult(
        batch_id=batch_id,
        project_id=project_id,
        video_id=video_id,
        source_zip=str(source_zip.resolve()),
        sidecar_path=str((batch_dir / "attributes_sidecar.json").resolve()),
        labels_path=str((batch_dir / "imported_labels.json").resolve()),
        image_count=len(built),
        reviewed_empty=empty,
        flagged_unreviewed=flagged,
        stored_excluded=excluded,
    )


def format_import_summary(result: ImportResult) -> str:
    """Russian operator text plus the stored zip path.

    The first count is the ZIP, not every frame now stored in the batch.
    """
    return "\n".join(
        [
            f"Кадров из архива: {result.image_count}.",
            f"Проверенных пустых: {result.reviewed_empty}.",
            f"Непроверенных: {result.flagged_unreviewed}.",
            f"Исключённых: {result.stored_excluded}.",
            result.source_zip,
        ]
    )


def load_imported_frames(workspace: Path, project_id: str) -> list[ImportedFrame]:
    """Read every imported batch. Export-only batches are skipped."""
    root = project_directory(workspace, project_id) / "annotation_batches"
    if not root.is_dir():
        return []
    frames: list[ImportedFrame] = []
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        labels_path = child / "imported_labels.json"
        if not labels_path.is_file():
            continue
        frames.extend(_load_batch(labels_path))
    return frames


def _resolve_zip(workspace: Path, project_id: str, file: Path) -> Path:
    resolved = resolve_project_path(workspace, project_id, file)
    if not resolved.is_file():
        raise LabError("file_not_found", f"Файл не найден: {resolved}.")
    return resolved


def _read_zip(path: Path, max_unpack_bytes: int) -> dict[str, bytes]:
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise LabError("zip_invalid", "Файл не является ZIP-архивом.") from exc
    with archive:
        planned: list[tuple[str, zipfile.ZipInfo]] = []
        declared = 0
        for info in archive.infolist():
            safe = _safe_zip_entry(info.filename)
            if safe is None:
                continue
            if info.file_size < 0:
                raise LabError("zip_too_large", _ZIP_TOO_LARGE)
            declared += int(info.file_size)
            if declared > max_unpack_bytes:
                raise LabError("zip_too_large", _ZIP_TOO_LARGE)
            planned.append((safe, info))
        members: dict[str, bytes] = {}
        actual = 0
        for safe, info in planned:
            if safe in members:
                raise LabError("zip_member", "В архиве повторяется имя файла.")
            payload = archive.read(info)
            actual += len(payload)
            if actual > max_unpack_bytes:
                raise LabError("zip_too_large", _ZIP_TOO_LARGE)
            members[safe] = payload
    return members


def _safe_zip_entry(name: str) -> str | None:
    """Return a relative member name. Directory entries are skipped."""
    if not isinstance(name, str):
        raise LabError(
            "zip_member",
            "Имя файла в архиве должно быть относительным и не содержать «..».",
        )
    if name.endswith("/"):
        safe_member_name(name[:-1])
        return None
    return safe_member_name(name)


def _parse_manifest(members: Mapping[str, bytes], video_id: str) -> list[_ManifestRow]:
    payload = members.get("frames_manifest.csv")
    if payload is None:
        raise LabError(
            "manifest_missing",
            "В архиве нет frames_manifest.csv. Split берётся из него, а не из subset CVAT.",
        )
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LabError(
            "invalid_manifest",
            "frames_manifest.csv должен быть в кодировке UTF-8.",
        ) from exc
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = list(reader.fieldnames or [])
    missing = [column for column in MANIFEST_COLUMNS if column not in fieldnames]
    if missing:
        raise LabError(
            "invalid_manifest",
            "В frames_manifest.csv нет столбца «" + missing[0] + "».",
        )
    rows: list[_ManifestRow] = []
    seen_files: set[str] = set()
    seen_indexes: set[int] = set()
    for raw in reader:
        row = _manifest_row(raw, video_id, "size_bytes" in fieldnames)
        if row.image_file in seen_files:
            raise LabError(
                "duplicate_image",
                f"Файл «{row.image_file}» повторяется в манифесте.",
            )
        if row.source_frame_index in seen_indexes:
            raise LabError(
                "duplicate_frame",
                f"Кадр {row.frame_id} повторяется в манифесте.",
            )
        seen_files.add(row.image_file)
        seen_indexes.add(row.source_frame_index)
        rows.append(row)
    if not rows:
        raise LabError("batch_empty", "В манифесте нет кадров.")
    return rows


def _manifest_row(raw: Mapping[str, str | None], video_id: str, has_size: bool) -> _ManifestRow:
    image_file = _cell(raw, "image_file")
    if (
        not image_file
        or "/" in image_file
        or "\\" in image_file
        or image_file in {".", ".."}
    ):
        raise LabError(
            "invalid_image_name",
            "Имя изображения в манифесте должно быть одним файлом без «..».",
        )
    row_video = _cell(raw, "video_id")
    if row_video != video_id:
        raise LabError(
            "video_mismatch",
            f"Кадр {image_file} относится к видео «{row_video}», пакет — к «{video_id}».",
        )
    index = _plain_int(_cell(raw, "source_frame_index"), f"{image_file}: source_frame_index")
    if index < 0:
        raise LabError("invalid_manifest", f"Кадр {video_id}:{index}: отрицательный индекс.")
    frame_id = f"{video_id}:{index}"
    split_text = _cell(raw, "split")
    try:
        split = SplitName(split_text)
    except ValueError as exc:
        raise LabError(
            "invalid_split",
            f"Кадр {frame_id}: split «{split_text}» не входит в train, validation, test.",
        ) from exc
    size_bytes: int | None = None
    if has_size:
        size_text = _cell(raw, "size_bytes")
        size_bytes = _plain_int(size_text, f"{frame_id}: size_bytes")
        if size_bytes < 0:
            raise LabError(
                "image_size_mismatch",
                f"Кадр {frame_id}, файл {image_file}: байтовый размер в манифесте отрицательный.",
            )
    return _ManifestRow(
        image_file=image_file,
        video_id=video_id,
        source_frame_index=index,
        source_pts=_plain_int(_cell(raw, "source_pts"), f"{frame_id}: source_pts"),
        time_base=_cell(raw, "time_base"),
        source_time_s=_plain_float(_cell(raw, "source_time_s"), frame_id),
        episode_id=_cell(raw, "episode_id"),
        split=split,
        split_group_id=_cell(raw, "split_group_id"),
        image_sha256=_sha256(_cell(raw, "image_sha256"), frame_id),
        width=_positive_int(_cell(raw, "width"), frame_id, "width"),
        height=_positive_int(_cell(raw, "height"), frame_id, "height"),
        size_bytes=size_bytes,
    )


def _reject_group_crossover(rows: list[_ManifestRow]) -> None:
    grouped: dict[str, set[str]] = {}
    for row in rows:
        if not row.split_group_id:
            continue
        grouped.setdefault(row.split_group_id, set()).add(row.split.value)
    for group_id, splits in grouped.items():
        if len(splits) > 1:
            joined = ", ".join(sorted(splits))
            raise LabError(
                "split_group_crossover",
                f"Группа «{group_id}» попала в несколько частей: {joined}.",
            )


def _merge_coco(
    members: Mapping[str, bytes],
) -> tuple[dict[str, tuple[dict[str, Any], str, int]], dict[int, list[dict[str, Any]]], dict[int, str]]:
    documents = _coco_documents(members)
    images: dict[str, tuple[dict[str, Any], str, int]] = {}
    annotations: dict[int, list[dict[str, Any]]] = {}
    categories: dict[int, str] = {}
    seen_image_ids: set[int] = set()
    seen_annotation_ids: set[int] = set()
    for member_name, document in documents:
        file_subset = _subset_from_coco_name(member_name)
        for category in _as_list(document.get("categories"), member_name, "categories"):
            if not isinstance(category, dict):
                raise LabError("coco_invalid", f"Категория в {member_name} должна быть объектом.")
            category_id = _json_int(category.get("id"), f"category id в {member_name}")
            name = category.get("name")
            if not isinstance(name, str) or not name.strip():
                raise LabError("coco_invalid", f"У категории {category_id} нет имени.")
            previous = categories.get(category_id)
            if previous is not None and previous != name:
                raise LabError(
                    "coco_invalid",
                    f"category_id {category_id} назван и «{previous}», и «{name}».",
                )
            categories[category_id] = name
        for image in _as_list(document.get("images"), member_name, "images"):
            if not isinstance(image, dict):
                raise LabError("coco_invalid", f"Изображение в {member_name} должно быть объектом.")
            image_id = _json_int(image.get("id"), f"image_id в {member_name}")
            if image_id in seen_image_ids:
                raise LabError("duplicate_image_id", f"Повторяется image_id {image_id}.")
            seen_image_ids.add(image_id)
            file_name = image.get("file_name")
            if not isinstance(file_name, str) or not file_name:
                raise LabError("coco_invalid", f"У image_id {image_id} нет file_name.")
            basename = PurePosixPath(file_name.replace("\\", "/")).name
            if basename in images:
                raise LabError("duplicate_image", f"Файл «{basename}» повторяется в COCO.")
            path_subset = _subset_from_image_name(file_name)
            images[basename] = (image, path_subset or file_subset, image_id)
            annotations[image_id] = []
        for annotation in _as_list(document.get("annotations"), member_name, "annotations"):
            if not isinstance(annotation, dict):
                raise LabError("coco_invalid", f"Аннотация в {member_name} должна быть объектом.")
            annotation_id = _json_int(annotation.get("id"), f"annotation_id в {member_name}")
            if annotation_id in seen_annotation_ids:
                raise LabError(
                    "duplicate_annotation_id",
                    f"Повторяется annotation_id {annotation_id}.",
                )
            seen_annotation_ids.add(annotation_id)
            image_id = _json_int(annotation.get("image_id"), f"annotation {annotation_id}")
            bucket = annotations.get(image_id)
            if bucket is None:
                raise LabError(
                    "unknown_image",
                    f"Аннотация {annotation_id} ссылается на неизвестный image_id {image_id}.",
                )
            bucket.append(annotation)
    names: dict[str, int] = {}
    for category_id, name in categories.items():
        if name in names:
            raise LabError("coco_invalid", f"Имя категории «{name}» повторяется.")
        names[name] = category_id
    return images, annotations, categories


def _coco_documents(members: Mapping[str, bytes]) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    for name in sorted(members):
        if PurePosixPath(name).name in _SKIP_JSON or not name.endswith(".json"):
            continue
        try:
            document = json.loads(members[name].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LabError("coco_invalid", f"Файл {name} не является JSON.") from exc
        if not isinstance(document, dict):
            continue
        if {"images", "annotations", "categories"} <= set(document):
            found.append((name, document))
    if not found:
        raise LabError(
            "coco_missing",
            "В архиве нет COCO с images, annotations и categories.",
        )
    return found


def _pair_manifest(
    rows: list[_ManifestRow],
    images: Mapping[str, tuple[dict[str, Any], str, int]],
) -> None:
    manifest_names = {row.image_file for row in rows}
    coco_names = set(images)
    missing = sorted(manifest_names - coco_names)
    extra = sorted(coco_names - manifest_names)
    if missing:
        raise LabError("image_missing", f"В COCO нет изображения «{missing[0]}».")
    if extra:
        raise LabError(
            "unknown_image",
            f"Изображения «{extra[0]}» нет в frames_manifest.csv.",
        )


def _build_frame(
    row: _ManifestRow,
    image_entry: tuple[dict[str, Any], str, int],
    annotations: Mapping[int, list[dict[str, Any]]],
    categories: Mapping[int, str],
    members: Mapping[str, bytes],
    frozen: FrozenSplit,
    catalog: StoredCatalog | CatalogConfig,
) -> _Built:
    _match_frozen(frozen, row)
    image, subset, image_id = image_entry
    member_name, payload = _locate_image(members, row, str(image.get("file_name", "")))
    _check_image_bytes(row, image, payload)
    if not subset:
        subset = _subset_from_image_name(member_name)
    image_attributes = _collect_attributes(image, _COCO_IMAGE_KEYS)
    anns = annotations[image_id]
    review = _review_status(row, image_attributes, anns)
    drafts: list[BoxDraft] = []
    sidecar_boxes: list[dict[str, Any]] = []
    resolved_ids: list[int] = []
    for annotation in anns:
        _reject_crowd(annotation, row)
        category_id = _json_int(annotation.get("category_id"), f"кадр {row.frame_id}")
        name = categories.get(category_id)
        if name is None:
            raise LabError(
                "unknown_category",
                f"Кадр {row.frame_id}: category_id {category_id} нет в categories.",
            )
        resolved_ids.append(
            resolve_category_name(
                name,
                category_id,
                catalog=catalog,
                frame_id=row.frame_id,
            )
        )
        attrs = _collect_attributes(annotation, _COCO_ANNOTATION_KEYS)
        x1, y1, x2, y2 = _xyxy(annotation, row)
        drafts.append(
            BoxDraft(
                category_name=name,
                category_id=category_id,
                visibility=_visibility(attrs, row),
                truncated=_truncated(attrs, row),
                instance_local_id=_instance_id(attrs),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                note=_optional_text(attrs.get("note")),
                iscrowd=0,
            )
        )
        sidecar_boxes.append(
            {
                "annotation_id": _json_int(annotation.get("id"), f"кадр {row.frame_id}"),
                "source_category_id": category_id,
                "attributes": attrs,
            }
        )
    record = validate_frame(
        FrameDraft(
            video_id=row.video_id,
            source_frame_index=row.source_frame_index,
            review_status=review,
            visibility=None,
            annotator=_optional_text(image_attributes.get("annotator")),
            reviewed_at=_reviewed_at(image_attributes.get("reviewed_at"), row),
            note=_optional_text(image_attributes.get("note")),
            boxes=drafts,
        ),
        catalog,
    )
    for box, internal_id in zip(record.boxes, resolved_ids, strict=True):
        if box.class_id != internal_id:
            raise LabError(
                "class_map",
                f"Кадр {row.frame_id}: внутренний id не совпал с именем класса.",
            )
    flagged = ["unreviewed"] if record.label_state is LabelState.unreviewed else []
    return _Built(
        row=row,
        payload=payload,
        coco_subset=subset,
        image_attributes=image_attributes,
        sidecar_boxes=sidecar_boxes,
        record=record,
        flagged=flagged,
    )


def _merge_stored(
    item: _Built,
    stored: ImportedFrame | None,
    sidecar: dict[str, Any] | None,
    catalog: StoredCatalog,
    zip_names: set[str],
) -> _Built:
    """Keep a stored review when this ZIP does not know a later class.

    ``reviewed_at`` is the stored instant, never the import clock. A review
    at or after ``added_at`` of a catalog class missing from the ZIP is left
    untouched. An earlier review keeps boxes of those missing classes and
    still takes the ZIP's boxes for names the ZIP does contain.
    """
    if stored is None:
        return item
    missing = [entry for entry in catalog.classes if entry.name not in zip_names]
    if _reviewed_after_missing_class(stored.record.reviewed_at, missing):
        return _keep_stored_frame(item, stored, sidecar)
    catalog_names = {entry.name for entry in catalog.classes}
    kept = [
        box
        for box in stored.record.boxes
        if box.class_name in catalog_names and box.class_name not in zip_names
    ]
    incoming = [box for box in item.record.boxes if box.class_name in zip_names]
    record = validate_frame(
        FrameDraft(
            video_id=stored.record.video_id,
            source_frame_index=stored.record.source_frame_index,
            review_status=stored.record.review_status,
            visibility=stored.record.visibility,
            annotator=stored.record.annotator,
            reviewed_at=stored.record.reviewed_at,
            note=stored.record.note,
            boxes=[_draft_from_box(box) for box in (*incoming, *kept)],
        ),
        catalog,
    )
    return replace(item, record=record, flagged=_flagged(record))


def _keep_stored_frame(
    item: _Built,
    stored: ImportedFrame,
    sidecar: dict[str, Any] | None,
) -> _Built:
    attributes = item.image_attributes
    boxes = item.sidecar_boxes
    if sidecar is not None:
        raw_attributes = sidecar.get("attributes")
        raw_boxes = sidecar.get("boxes")
        if isinstance(raw_attributes, dict):
            attributes = dict(raw_attributes)
        if isinstance(raw_boxes, list) and all(isinstance(box, dict) for box in raw_boxes):
            boxes = [dict(box) for box in raw_boxes]
    return replace(
        item,
        record=stored.record,
        flagged=_flagged(stored.record),
        image_attributes=attributes,
        sidecar_boxes=boxes,
    )


def _reviewed_after_missing_class(
    reviewed_at: datetime | None,
    missing: list[StoredCatalogClass],
) -> bool:
    """True when this review saw a catalog class the ZIP does not list."""
    if reviewed_at is None or reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        return False
    instant = reviewed_at.astimezone(UTC)
    return any(instant >= entry.added_at for entry in missing)


def _draft_from_box(box: AnnotationBox) -> BoxDraft:
    return BoxDraft(
        category_name=box.class_name,
        class_id=box.class_id,
        visibility=box.visibility,
        truncated=box.truncated,
        instance_local_id=box.instance_local_id,
        x1=box.x1,
        y1=box.y1,
        x2=box.x2,
        y2=box.y2,
        note=box.note,
        iscrowd=0,
    )


def _flagged(record: LabelRecord) -> list[str]:
    if record.label_state is LabelState.unreviewed:
        return ["unreviewed"]
    return []


def _indexed_imports(
    project_dir: Path,
) -> tuple[dict[tuple[str, int], ImportedFrame], dict[tuple[str, int], dict[str, Any]]]:
    root = project_dir / "annotation_batches"
    frames: dict[tuple[str, int], ImportedFrame] = {}
    sidecars: dict[tuple[str, int], dict[str, Any]] = {}
    if not root.is_dir():
        return frames, sidecars
    for child in sorted(path for path in root.iterdir() if path.is_dir() and not path.is_symlink()):
        labels_path = child / "imported_labels.json"
        if labels_path.is_file() and not labels_path.is_symlink():
            for frame in _load_batch(labels_path):
                frames[(frame.video_id, frame.source_frame_index)] = frame
        side_path = child / "attributes_sidecar.json"
        if side_path.is_file() and not side_path.is_symlink():
            sidecars.update(_index_sidecar(side_path))
    return frames, sidecars


def _index_sidecar(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError("invalid_import", f"Файл {path.name} повреждён.") from exc
    raw_frames = document.get("frames") if isinstance(document, dict) else None
    if not isinstance(raw_frames, list):
        raise LabError("invalid_import", f"Файл {path.name} повреждён.")
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in raw_frames:
        if not isinstance(raw, dict):
            raise LabError("invalid_import", f"Файл {path.name} повреждён.")
        video_id = raw.get("video_id")
        index = raw.get("source_frame_index")
        if not isinstance(video_id, str) or isinstance(index, bool) or not isinstance(index, int):
            raise LabError("invalid_import", f"Файл {path.name} повреждён.")
        found[(video_id, index)] = raw
    return found


def _match_frozen(frozen: FrozenSplit, row: _ManifestRow) -> None:
    expected = _expected_assignment(frozen, row)
    if row.split.value != expected:
        raise LabError(
            "split_mismatch",
            f"Кадр {row.frame_id}: манифест указывает {row.split.value}, "
            f"зафиксировано {expected}. Split не пересчитывается.",
        )


def _expected_assignment(frozen: FrozenSplit, row: _ManifestRow) -> str:
    role = role_at(frozen.blocks, row.source_time_s)
    if role is None:
        raise LabError(
            "frame_unassigned",
            f"Кадр {row.frame_id} не попадает в зафиксированное разбиение. "
            "Split не пересчитывается.",
        )
    return role


def _locate_image(
    members: Mapping[str, bytes],
    row: _ManifestRow,
    coco_file_name: str,
) -> tuple[str, bytes]:
    matches = [
        (name, payload)
        for name, payload in members.items()
        if PurePosixPath(name).name == row.image_file
        and PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg"}
    ]
    if not matches:
        raise LabError(
            "image_missing",
            f"Кадр {row.frame_id}, файл {row.image_file}: изображения нет в архиве.",
        )
    wanted = coco_file_name.replace("\\", "/")
    for name, payload in matches:
        if name == wanted:
            return name, payload
    if len(matches) == 1:
        return matches[0]
    raise LabError(
        "duplicate_image",
        f"Кадр {row.frame_id}: в архиве несколько файлов «{row.image_file}».",
    )


def _check_image_bytes(row: _ManifestRow, image: Mapping[str, Any], payload: bytes) -> None:
    digest = hashlib.sha256(payload).hexdigest()
    if digest != row.image_sha256:
        raise LabError(
            "image_hash_mismatch",
            f"Кадр {row.frame_id}, файл {row.image_file}: "
            "SHA-256 не совпадает с манифестом.",
        )
    if row.size_bytes is not None and row.size_bytes != len(payload):
        raise LabError(
            "image_size_mismatch",
            f"Кадр {row.frame_id}, файл {row.image_file}: "
            f"байтовый размер {len(payload)} не совпадает с манифестом ({row.size_bytes}).",
        )
    width, height = _jpeg_size(payload, row)
    if width != row.width or height != row.height:
        raise LabError(
            "image_size_mismatch",
            f"Кадр {row.frame_id}, файл {row.image_file}: "
            f"размер JPEG {width}×{height} не совпадает с манифестом "
            f"{row.width}×{row.height}.",
        )
    coco_width = _json_int(image.get("width"), f"ширина кадра {row.frame_id}")
    coco_height = _json_int(image.get("height"), f"высота кадра {row.frame_id}")
    if coco_width != row.width or coco_height != row.height:
        raise LabError(
            "image_size_mismatch",
            f"Кадр {row.frame_id}, файл {row.image_file}: "
            "ширина или высота в COCO не совпадает с манифестом.",
        )


def _jpeg_size(payload: bytes, row: _ManifestRow) -> tuple[int, int]:
    array = np.frombuffer(payload, dtype=np.uint8).copy()
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise LabError(
            "frame_unreadable",
            f"Кадр {row.frame_id}, файл {row.image_file}: JPEG не декодируется.",
        )
    height, width = int(image.shape[0]), int(image.shape[1])
    return width, height


def _xyxy(annotation: Mapping[str, Any], row: _ManifestRow) -> tuple[float, float, float, float]:
    bbox = annotation.get("bbox")
    try:
        return xywh_to_xyxy(bbox, image_width=row.width, image_height=row.height)
    except (TypeError, ValueError) as exc:
        raise LabError(
            "invalid_bbox",
            f"Кадр {row.frame_id}, файл {row.image_file}: "
            "рамка вне изображения или с неположительным размером.",
        ) from exc


def _reject_crowd(annotation: Mapping[str, Any], row: _ManifestRow) -> None:
    if "iscrowd" not in annotation:
        return
    iscrowd = annotation.get("iscrowd")
    if iscrowd == 0 and not isinstance(iscrowd, bool):
        return
    shown = iscrowd if isinstance(iscrowd, int) and not isinstance(iscrowd, bool) else None
    raise UnsupportedCrowdError(row.frame_id, shown)


def _review_status(
    row: _ManifestRow,
    image_attributes: Mapping[str, Any],
    annotations: list[dict[str, Any]],
) -> ReviewStatus:
    found: list[object] = []
    if "review_status" in image_attributes:
        found.append(image_attributes["review_status"])
    for annotation in annotations:
        attrs = _collect_attributes(annotation, _COCO_ANNOTATION_KEYS)
        if "review_status" in attrs:
            found.append(attrs["review_status"])
    if not found:
        return ReviewStatus.unreviewed
    parsed: list[ReviewStatus] = []
    for item in found:
        if not isinstance(item, str):
            raise LabError(
                "invalid_review",
                f"Кадр {row.frame_id}: review_status должен быть строкой.",
            )
        try:
            parsed.append(ReviewStatus(item))
        except ValueError as exc:
            raise LabError(
                "invalid_review",
                f"Кадр {row.frame_id}: неизвестный review_status «{item}».",
            ) from exc
    if len(set(parsed)) != 1:
        raise LabError(
            "review_conflict",
            f"Кадр {row.frame_id}: review_status на изображении и рамках не совпадает.",
        )
    return parsed[0]


def _visibility(attributes: Mapping[str, Any], row: _ManifestRow) -> Visibility:
    if "visibility" not in attributes:
        return Visibility.clear
    value = attributes["visibility"]
    if not isinstance(value, str):
        raise LabError(
            "invalid_visibility",
            f"Кадр {row.frame_id}: visibility должен быть строкой.",
        )
    try:
        return Visibility(value)
    except ValueError as exc:
        raise LabError(
            "invalid_visibility",
            f"Кадр {row.frame_id}: неизвестный visibility «{value}».",
        ) from exc


def _truncated(attributes: Mapping[str, Any], row: _ManifestRow) -> bool:
    if "truncated" not in attributes:
        return False
    value = attributes["truncated"]
    if isinstance(value, bool):
        return value
    if value in {0, 1} and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise LabError(
        "invalid_attribute",
        f"Кадр {row.frame_id}: truncated должен быть true или false.",
    )


def _instance_id(attributes: Mapping[str, Any]) -> str | None:
    value = attributes.get("instance_local_id")
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _optional_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _reviewed_at(value: object, row: _ManifestRow) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise LabError(
            "invalid_attribute",
            f"Кадр {row.frame_id}: reviewed_at должен быть строкой времени.",
        )
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise LabError(
            "invalid_attribute",
            f"Кадр {row.frame_id}: reviewed_at не разбирается.",
        ) from exc


def _collect_attributes(obj: Mapping[str, Any], standard: frozenset[str]) -> dict[str, Any]:
    """Copy attribute keys, including ones COCO training files cannot carry."""
    merged = _coerce_attributes(obj.get("attributes"))
    for key, value in obj.items():
        if key in standard or key in merged:
            continue
        merged[key] = value
    return merged


def _coerce_attributes(raw: object) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):
        converted: dict[str, Any] = {}
        extras: list[Any] = []
        for item in raw:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                converted[item["name"]] = item.get("value")
            else:
                extras.append(item)
        if extras:
            converted["unknown_items"] = extras
        return converted
    return {"unknown_value": raw}


def _publish(
    workspace: Path,
    project_id: str,
    video_id: str,
    split: SplitName,
    built: list[_Built],
    frozen: FrozenSplit,
) -> None:
    upsert_frames(
        workspace,
        project_id,
        video_id,
        split,
        [_write_from_built(item) for item in built],
        frozen=frozen,
        batch_status="imported",
    )


def _write_from_built(item: _Built) -> FrameWrite:
    row = item.row
    return FrameWrite(
        image_file=row.image_file,
        source_frame_index=row.source_frame_index,
        source_pts=row.source_pts,
        time_base=row.time_base,
        source_time_s=row.source_time_s,
        episode_id=row.episode_id,
        split=row.split,
        split_group_id=row.split_group_id,
        width=row.width,
        height=row.height,
        payload=item.payload,
        record=item.record,
        coco_subset=item.coco_subset,
        image_attributes=item.image_attributes,
        sidecar_boxes=item.sidecar_boxes,
    )


def _frame_document(item: _Built) -> dict[str, Any]:
    return _frame_label_document(_write_from_built(item), item.row.video_id)


def _sidecar_frame(item: _Built) -> dict[str, Any]:
    return _frame_sidecar_document(_write_from_built(item), item.row.video_id)


def _load_batch(labels_path: Path) -> list[ImportedFrame]:
    try:
        document = json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError(
            "invalid_import",
            f"Файл {labels_path.name} повреждён.",
        ) from exc
    if not isinstance(document, dict) or not isinstance(document.get("frames"), list):
        raise LabError("invalid_import", f"Файл {labels_path.name} повреждён.")
    frames: list[ImportedFrame] = []
    for raw in document["frames"]:
        if not isinstance(raw, dict):
            raise LabError("invalid_import", f"Файл {labels_path.name} повреждён.")
        frames.append(_load_frame(labels_path, raw))
    return frames


def _load_frame(labels_path: Path, raw: Mapping[str, Any]) -> ImportedFrame:
    image_file = raw.get("image_file")
    if not isinstance(image_file, str) or "/" in image_file or "\\" in image_file:
        raise LabError("invalid_import", "В импорте испорчено имя изображения.")
    jpeg = (labels_path.parent / "images" / image_file).resolve()
    root = labels_path.parent.resolve()
    if not jpeg.is_relative_to(root) or not jpeg.is_file():
        video_id = raw.get("video_id")
        index = raw.get("source_frame_index")
        raise LabError(
            "image_missing",
            f"Кадр {video_id}:{index}, файл {image_file}: изображения нет в пакете.",
        )
    payload = jpeg.read_bytes()
    digest = raw.get("image_sha256")
    if hashlib.sha256(payload).hexdigest() != digest:
        raise LabError(
            "image_hash_mismatch",
            f"Файл {image_file}: SHA-256 не совпадает с импортом.",
        )
    if len(payload) != raw.get("size_bytes"):
        raise LabError(
            "image_size_mismatch",
            f"Файл {image_file}: байтовый размер не совпадает с импортом.",
        )
    try:
        record = LabelRecord.model_validate(raw.get("record"))
    except ValidationError as exc:
        raise LabError("invalid_import", f"Файл {image_file}: разметка повреждена.") from exc
    flagged = raw.get("flagged")
    if not isinstance(flagged, list) or any(not isinstance(item, str) for item in flagged):
        raise LabError("invalid_import", f"Файл {image_file}: список flagged испорчен.")
    try:
        split = SplitName(str(raw.get("split")))
    except ValueError as exc:
        raise LabError("invalid_import", f"Файл {image_file}: split испорчен.") from exc
    return ImportedFrame(
        image_file=image_file,
        video_id=str(raw.get("video_id")),
        source_frame_index=int(raw["source_frame_index"]),
        episode_id=str(raw.get("episode_id") or ""),
        split=split,
        split_group_id=str(raw.get("split_group_id") or ""),
        source_time_s=float(raw["source_time_s"]),
        image_sha256=str(digest),
        size_bytes=int(raw["size_bytes"]),
        width=int(raw["width"]),
        height=int(raw["height"]),
        coco_subset=str(raw.get("coco_subset") or ""),
        jpeg_path=jpeg,
        flagged=tuple(flagged),
        record=record,
    )


def _copy_source(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".partial")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _record_batch(
    workspace: Path,
    project_id: str,
    batch_id: str,
    video_id: str,
    destination: Path,
) -> None:
    db_path = project_sqlite(workspace, project_id)
    created_at = datetime.now(UTC).isoformat()
    with connect(db_path) as connection:
        try:
            connection.execute(
                """
                INSERT INTO annotation_batches (
                    batch_id, project_id, video_id, status, path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    video_id = excluded.video_id,
                    status = excluded.status,
                    path = excluded.path
                """,
                (
                    batch_id,
                    project_id,
                    video_id,
                    "imported",
                    str(destination),
                    created_at,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise LabError("video_not_found", "Видео не найдено в этом проекте.") from exc


def _subset_from_coco_name(name: str) -> str:
    stem = PurePosixPath(name).stem
    if stem.startswith("instances_"):
        return stem.removeprefix("instances_")
    return ""


def _subset_from_image_name(file_name: str) -> str:
    """CVAT subset directory, if the path has one. ``images/<file>`` has none."""
    parts = PurePosixPath(file_name.replace("\\", "/")).parts
    if len(parts) >= 3 and parts[0] == "images":
        return parts[1]
    if len(parts) >= 2 and parts[0] != "images":
        return parts[0]
    return ""


def _as_list(value: object, member_name: str, key: str) -> list[Any]:
    if not isinstance(value, list):
        raise LabError("coco_invalid", f"В {member_name} поле {key} должно быть списком.")
    return value


def _cell(raw: Mapping[str, str | None], key: str) -> str:
    value = raw.get(key)
    if value is None:
        return ""
    return value.strip()


def _plain_int(text: str, label: str) -> int:
    if text.startswith(("+", "-")):
        body = text[1:]
        sign = -1 if text[0] == "-" else 1
    else:
        body = text
        sign = 1
    if not body.isdigit():
        raise LabError("invalid_manifest", f"Поле {label} должно быть целым.")
    return sign * int(body)


def _positive_int(text: str, frame_id: str, label: str) -> int:
    number = _plain_int(text, f"{frame_id}: {label}")
    if number <= 0:
        raise LabError(
            "invalid_manifest",
            f"Кадр {frame_id}: {label} должен быть положительным.",
        )
    return number


def _plain_float(text: str, frame_id: str) -> float:
    try:
        number = float(text)
    except ValueError as exc:
        raise LabError(
            "invalid_manifest",
            f"Кадр {frame_id}: source_time_s должен быть числом.",
        ) from exc
    if number < 0 or not math.isfinite(number):
        raise LabError(
            "invalid_manifest",
            f"Кадр {frame_id}: source_time_s должен быть конечным и неотрицательным.",
        )
    return number


def _sha256(text: str, frame_id: str) -> str:
    lowered = text.strip().lower()
    if len(lowered) != 64 or any(char not in "0123456789abcdef" for char in lowered):
        raise LabError(
            "invalid_manifest",
            f"Кадр {frame_id}: image_sha256 должен быть 64 шестнадцатеричными знаками.",
        )
    return lowered


def _json_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabError("coco_invalid", f"{label} должен быть целым.")
    return value
