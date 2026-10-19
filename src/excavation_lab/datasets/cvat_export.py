"""Export one frozen split's sampled frames as a CVAT batch zip.

The archive is an annotation batch, not an RF-DETR training snapshot.
There is no ``_annotations.coco.json`` at the root. ``split`` is the
frozen block role. A haul episode id is not written. This module does
not propose or rewrite a split.

``labels.yaml`` and the class table in the instructions are the project
catalog at export time, including each class ``box`` phrase. ``title``
and ``box`` are copied as text. A class with no boxes yet is still listed.

JPEG bytes come from the sampler cache when the file matches the sample
hash. Otherwise the reader seeks to that frame and encodes it. PTS and the
time base come from the sample row when it recorded them; a legacy row
without PTS is read from the file the same way. Images are not sent to a
network API.

Validation and test never include ``predictions.coco.json``, including
when the caller asks for predictions. A train file is written only on
request, and ``info.description`` contains ``not_ground_truth``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sqlite3
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from pydantic import Field, ValidationError

from excavation_lab.datasets.catalog import (
    StoredCatalog,
    StoredCatalogClass,
    require_project_catalog,
)
from excavation_lab.datasets.geometry import xyxy_to_xywh
from excavation_lab.datasets.label_store import frozen_fields_for
from excavation_lab.datasets.sampler import SampleResult, SampleRow, decode_frame_at
from excavation_lab.datasets.splits import FrozenSplit, require_frozen
from excavation_lab.errors import LabError, MediaNotFoundError, TimebaseError
from excavation_lab.media.reader import FrameReader
from excavation_lab.schemas import (
    SCHEMA_VERSION,
    DetectionClassName,
    SplitName,
    StrictModel,
    load_labels,
    repo_root,
)
from excavation_lab.storage import connect, get_video, project_directory, project_sqlite

MANIFEST_COLUMNS: tuple[str, ...] = (
    "image_file",
    "video_id",
    "source_frame_index",
    "source_pts",
    "time_base",
    "source_time_s",
    "episode_id",
    "split",
    "split_group_id",
    "image_sha256",
    "width",
    "height",
)

_NOT_GROUND_TRUTH = "not_ground_truth"
_JPEG_QUALITY = 95
_HELD_OUT = frozenset({SplitName.validation, SplitName.test})
_BATCH_KIND = "annotation_batch"
_TEMPLATE_NAME = "annotation_instructions.md"

_WITHOUT_PREDICTIONS = (
    "Пакет {split} выгружается без предсказаний модели. "
    "Части validation и test не получают предразметку."
)
_NOT_GROUND_TRUTH_MESSAGE = (
    "Файл predictions.coco.json — not_ground_truth и не является эталонной разметкой."
)


class PredictionBox(StrictModel):
    """One optional pre-annotation. Coordinates are internal xyxy pixels."""

    source_frame_index: int = Field(ge=0)
    class_name: DetectionClassName
    x1: float
    y1: float
    x2: float
    y2: float
    score: float | None = Field(default=None, ge=0, le=1)


class CvatExportResult(StrictModel):
    """What ``export-cvat`` wrote. ``predictions_included`` is false for held-out splits."""

    batch_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    split: SplitName
    zip_path: str = Field(min_length=1)
    predictions_included: bool
    uncompressed_bytes: int = Field(ge=0)
    image_count: int = Field(ge=1)
    message: str = ""


@dataclass(frozen=True, slots=True)
class _Chosen:
    row: SampleRow
    split: SplitName
    split_group_id: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class _Shipped:
    image_file: str
    video_id: str
    source_frame_index: int
    source_pts: int
    time_base: str
    source_time_s: float
    episode_id: str
    split: str
    split_group_id: str
    image_sha256: str
    width: int
    height: int
    jpeg: bytes


def export_cvat(
    workspace: Path,
    project_id: str,
    batch_id: str,
    *,
    include_predictions: bool = False,
    predictions: Sequence[PredictionBox | Mapping[str, Any]] | None = None,
) -> CvatExportResult:
    """Write ``annotation_batch.zip`` for one frozen split.

    ``batch_id`` is ``<video_id>__train``, ``<video_id>__validation``, or
    ``<video_id>__test``. Frames are the sampled rows whose frozen episode
    (or frozen block) has that split.
    """
    video_id, split = _parse_batch_id(batch_id)
    project_dir = project_directory(workspace, project_id)
    video = get_video(workspace, project_id, video_id)
    catalog = require_project_catalog(workspace, project_id)
    frozen = require_frozen(workspace, project_id, video_id)
    sample = _load_sample(project_dir, project_id, video_id)
    if sample.split_content_hash != frozen.content_hash:
        raise LabError(
            "sample_stale",
            "Отбор сделан по другому зафиксированному разбиению. Повторите отбор.",
        )
    chosen = _choose(sample, frozen, split)
    if not chosen:
        raise LabError(
            "batch_empty",
            "В отобранных кадрах нет кадров этой части разбиения.",
        )
    shipped = _ship_frames(video, chosen, _cache_dir(project_dir, video_id))
    write_predictions = include_predictions and split not in _HELD_OUT
    boxes = _checked_predictions(predictions) if write_predictions else ()
    members = _members(
        shipped,
        catalog=catalog,
        write_predictions=write_predictions,
        boxes=boxes,
    )
    batch_body = _batch_document(
        batch_id=batch_id,
        project_id=project_id,
        video_id=video_id,
        split=split,
        split_content_hash=frozen.content_hash,
        image_count=len(shipped),
        predictions_included=write_predictions,
        other_bytes=sum(len(payload) for _, payload in members),
    )
    members.append(("batch.json", batch_body))
    destination = _write_zip(project_dir, batch_id, members)
    uncompressed = sum(len(payload) for _, payload in members)
    _record_batch(workspace, project_id, batch_id, video_id, destination)
    message = _operator_message(split, write_predictions)
    return CvatExportResult(
        batch_id=batch_id,
        project_id=project_id,
        video_id=video_id,
        split=split,
        zip_path=str(destination),
        predictions_included=write_predictions,
        uncompressed_bytes=uncompressed,
        image_count=len(shipped),
        message=message,
    )


def format_export_summary(result: CvatExportResult) -> str:
    """Russian operator text plus the zip path."""
    lines = [result.message] if result.message else []
    lines.append(result.zip_path)
    return "\n".join(lines)


def add_zip_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    """Add one relative member. ``..`` and absolute names are refused."""
    safe = safe_member_name(name)
    info = zipfile.ZipInfo(filename=safe)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.date_time = (2026, 1, 1, 0, 0, 0)
    archive.writestr(info, data)


def safe_member_name(name: str) -> str:
    """Return a relative posix name, or raise ``zip_member``."""
    if not isinstance(name, str) or not name or "\x00" in name:
        raise LabError(
            "zip_member",
            "Имя файла в архиве должно быть относительным и не содержать «..».",
        )
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or _has_drive(normalized):
        raise LabError(
            "zip_member",
            "Имя файла в архиве должно быть относительным и не содержать «..».",
        )
    path = PurePosixPath(normalized)
    if path.is_absolute() or normalized.endswith("/"):
        raise LabError(
            "zip_member",
            "Имя файла в архиве должно быть относительным и не содержать «..».",
        )
    if any(part in {"", ".", ".."} for part in path.parts):
        raise LabError(
            "zip_member",
            "Имя файла в архиве должно быть относительным и не содержать «..».",
        )
    return normalized


def _operator_message(split: SplitName, write_predictions: bool) -> str:
    if split in _HELD_OUT:
        return _WITHOUT_PREDICTIONS.format(split=split.value)
    if write_predictions:
        return _NOT_GROUND_TRUTH_MESSAGE
    return ""


def _parse_batch_id(batch_id: str) -> tuple[str, SplitName]:
    if (
        not isinstance(batch_id, str)
        or not batch_id
        or batch_id in {".", ".."}
        or "/" in batch_id
        or "\\" in batch_id
        or "\x00" in batch_id
    ):
        raise LabError("invalid_batch_id", "Недопустимый идентификатор пакета.")
    video_id, separator, split_name = batch_id.rpartition("__")
    if separator != "__" or not video_id or "/" in video_id or "\\" in video_id:
        raise LabError(
            "unknown_batch",
            "Идентификатор пакета имеет вид «<video_id>__train», "
            "«<video_id>__validation» или «<video_id>__test».",
        )
    try:
        split = SplitName(split_name)
    except ValueError as exc:
        raise LabError(
            "unknown_batch",
            "Идентификатор пакета имеет вид «<video_id>__train», "
            "«<video_id>__validation» или «<video_id>__test».",
        ) from exc
    return video_id, split


def _has_drive(name: str) -> bool:
    head = name.split("/", 1)[0]
    return len(head) >= 2 and head[0].isalpha() and head[1] == ":"


def _load_sample(project_dir: Path, project_id: str, video_id: str) -> SampleResult:
    path = project_dir / "samples" / video_id / "sample_manifest.json"
    if not path.is_file():
        raise LabError(
            "sample_missing",
            "Нет отобранных кадров для этого видео. Сначала выполните отбор.",
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError(
            "sample_missing",
            "Файл отбора кадров повреждён. Повторите отбор.",
        ) from exc
    try:
        sample = SampleResult.model_validate(document)
    except ValidationError as exc:
        raise LabError(
            "sample_missing",
            "Файл отбора кадров повреждён. Повторите отбор.",
        ) from exc
    if sample.project_id != project_id or sample.video_id != video_id:
        raise LabError(
            "sample_missing",
            "Файл отбора относится к другому проекту или видео.",
        )
    return sample


def _cache_dir(project_dir: Path, video_id: str) -> Path:
    return project_dir / "samples" / video_id / "images"


def _choose(sample: SampleResult, frozen: FrozenSplit, split: SplitName) -> list[_Chosen]:
    chosen: list[_Chosen] = []
    for row in sample.rows:
        fields = _frozen_fields(frozen, row)
        if fields is None or fields[0] is not split:
            continue
        frozen_split, group_id, start_s, end_s = fields
        chosen.append(
            _Chosen(
                row=row,
                split=frozen_split,
                split_group_id=group_id,
                start_s=start_s,
                end_s=end_s,
            )
        )
    chosen.sort(key=lambda item: item.row.source_frame_index)
    return chosen


def _frozen_fields(
    frozen: FrozenSplit,
    row: SampleRow,
) -> tuple[SplitName, str, float, float] | None:
    return frozen_fields_for(frozen, row.source_time_s)


def _ship_frames(
    video: Mapping[str, Any],
    chosen: Sequence[_Chosen],
    cache_dir: Path,
) -> list[_Shipped]:
    media = _media_path(video.get("path"))
    video_id = str(video["video_id"])
    cached = {
        item.row.source_frame_index: payload
        for item in chosen
        if (payload := _cached_jpeg(cache_dir, item.row)) is not None
    }
    observed = _read_selected(media, video_id, chosen, set(cached))
    shipped: list[_Shipped] = []
    for item in chosen:
        index = item.row.source_frame_index
        meta = observed[index]
        payload = cached.get(index)
        if payload is None:
            if meta.image is None:
                raise LabError(
                    "frame_missing",
                    f"Кадр {index} не декодирован. Подставной кадр не записывается.",
                )
            payload = _encode_jpeg(meta.image)
        width, height = _jpeg_size(payload)
        # The split is the block role. A leftover haul episode id is left blank.
        shipped.append(
            _Shipped(
                image_file=item.row.image_file,
                video_id=item.row.video_id,
                source_frame_index=index,
                source_pts=meta.source_pts,
                time_base=meta.time_base,
                source_time_s=meta.source_time_s,
                episode_id="",
                split=item.split.value,
                split_group_id=item.split_group_id,
                image_sha256=hashlib.sha256(payload).hexdigest(),
                width=width,
                height=height,
                jpeg=payload,
            )
        )
    return shipped


@dataclass(frozen=True, slots=True)
class _Observed:
    source_pts: int
    time_base: str
    source_time_s: float
    image: np.ndarray | None


def _read_selected(
    media: Path,
    video_id: str,
    chosen: Sequence[_Chosen],
    cached_indexes: set[int],
) -> dict[int, _Observed]:
    """Clock and, when the cache misses, pixels for every chosen row.

    A sample row that carries ``source_pts`` and has a matching cached JPEG
    needs no read at all. A legacy row (no PTS) or a cache miss seeks to that
    one frame; the block windows are never decoded in full.
    """
    found: dict[int, _Observed] = {}
    reader: FrameReader | None = None
    for item in chosen:
        row = item.row
        index = row.source_frame_index
        if index in found:
            continue
        if row.source_pts is not None and row.time_base is not None and index in cached_indexes:
            found[index] = _Observed(
                source_pts=row.source_pts,
                time_base=row.time_base,
                source_time_s=float(row.source_time_s),
                image=None,
            )
            continue
        if reader is None:
            reader = FrameReader(media, video_id=video_id)
        frame = decode_frame_at(
            reader,
            source_frame_index=index,
            source_time_s=row.source_time_s,
            source_pts=row.source_pts,
            limit_s=item.end_s,
        )
        if frame is None or frame.source_frame_index != index:
            raise LabError(
                "frame_missing",
                f"Кадр {index} не прочитан. Подставной кадр не записывается.",
            )
        if frame.source_pts is None or frame.source_time_s is None:
            raise TimebaseError(
                f"У кадра {index} нет PTS. Подставное время не записывается."
            )
        if not isinstance(frame.source_pts, int) or isinstance(frame.source_pts, bool):
            raise TimebaseError(
                f"У кадра {index} нет PTS. Подставное время не записывается."
            )
        image = None
        if index not in cached_indexes:
            if frame.image is None:
                raise LabError(
                    "frame_missing",
                    f"Кадр {index} не декодирован. Подставной кадр не записывается.",
                )
            image = np.ascontiguousarray(frame.image).copy()
        found[index] = _Observed(
            source_pts=frame.source_pts,
            time_base=frame.time_base,
            source_time_s=float(frame.source_time_s),
            image=image,
        )
    return found


def _cached_jpeg(cache_dir: Path, row: SampleRow) -> bytes | None:
    """Sampler JPEG whose bytes match ``image_sha256``. A mismatch is a miss."""
    if "/" in row.image_file or "\\" in row.image_file or row.image_file in {".", ".."}:
        return None
    path = cache_dir / row.image_file
    if not path.is_file():
        return None
    resolved = path.resolve()
    root = cache_dir.resolve()
    if not resolved.is_relative_to(root):
        return None
    payload = resolved.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row.image_sha256:
        return None
    return payload


def _encode_jpeg(image: np.ndarray) -> bytes:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise LabError("frame_unreadable", "Кадр для JPEG должен быть RGB uint8.")
    bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY],
    )
    if not ok:
        raise LabError("frame_unreadable", "Не удалось записать JPEG.")
    return encoded.tobytes()


def _jpeg_size(payload: bytes) -> tuple[int, int]:
    array = np.frombuffer(payload, dtype=np.uint8).copy()
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise LabError("frame_unreadable", "JPEG в пакете не декодируется.")
    height, width = int(image.shape[0]), int(image.shape[1])
    return width, height


def _media_path(stored: object) -> Path:
    if not isinstance(stored, str) or not stored:
        raise LabError(
            "video_path_missing",
            "У видео нет локального файла. Зарегистрируйте путь к MP4, MOV или MKV.",
        )
    path = Path(stored)
    if not path.is_file():
        raise MediaNotFoundError(str(path))
    return path


def _checked_predictions(
    predictions: Sequence[PredictionBox | Mapping[str, Any]] | None,
) -> tuple[PredictionBox, ...]:
    if not predictions:
        return ()
    boxes: list[PredictionBox] = []
    for item in predictions:
        if isinstance(item, PredictionBox):
            boxes.append(item)
            continue
        try:
            boxes.append(PredictionBox.model_validate(item))
        except ValidationError as exc:
            raise LabError(
                "invalid_prediction",
                "Предсказание задано некорректно и в пакет не записывается.",
            ) from exc
    return tuple(boxes)


def _members(
    shipped: Sequence[_Shipped],
    *,
    catalog: StoredCatalog,
    write_predictions: bool,
    boxes: Sequence[PredictionBox],
) -> list[tuple[str, bytes]]:
    members: list[tuple[str, bytes]] = [
        ("frames_manifest.csv", _manifest_csv(shipped)),
        ("labels.yaml", _labels_yaml(catalog)),
        (_TEMPLATE_NAME, _instructions(catalog)),
    ]
    if write_predictions:
        members.append(("predictions.coco.json", _predictions_coco(shipped, boxes, catalog)))
    for frame in shipped:
        members.append((f"images/{frame.image_file}", frame.jpeg))
    return members


def _manifest_csv(shipped: Sequence[_Shipped]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(MANIFEST_COLUMNS),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for frame in shipped:
        writer.writerow(
            {
                "image_file": frame.image_file,
                "video_id": frame.video_id,
                "source_frame_index": str(frame.source_frame_index),
                "source_pts": str(frame.source_pts),
                "time_base": frame.time_base,
                "source_time_s": f"{frame.source_time_s:.6f}",
                "episode_id": frame.episode_id,
                "split": frame.split,
                "split_group_id": frame.split_group_id,
                "image_sha256": frame.image_sha256,
                "width": str(frame.width),
                "height": str(frame.height),
            }
        )
    return buffer.getvalue().encode("utf-8")


def _predictions_coco(
    shipped: Sequence[_Shipped],
    boxes: Sequence[PredictionBox],
    catalog: StoredCatalog,
) -> bytes:
    by_index = {frame.source_frame_index: frame for frame in shipped}
    image_ids = {
        frame.source_frame_index: image_id
        for image_id, frame in enumerate(shipped, start=1)
    }
    classes = _catalog_classes(catalog)
    categories = [{"id": item.id, "name": item.name} for item in classes]
    category_ids = {item.name: item.id for item in classes}
    images = [
        {
            "id": image_ids[frame.source_frame_index],
            "file_name": f"images/{frame.image_file}",
            "width": frame.width,
            "height": frame.height,
        }
        for frame in shipped
    ]
    ordered = sorted(boxes, key=lambda box: (box.source_frame_index, box.class_name.value, box.x1))
    annotations: list[dict[str, Any]] = []
    for box in ordered:
        frame = by_index.get(box.source_frame_index)
        if frame is None:
            raise LabError(
                "prediction_frame",
                f"Предсказание относится к кадру {box.source_frame_index}, "
                "которого нет в пакете.",
            )
        try:
            x, y, width, height = xyxy_to_xywh((box.x1, box.y1, box.x2, box.y2))
        except ValueError as exc:
            raise LabError(
                "invalid_prediction",
                "Рамка предсказания должна иметь положительные ширину и высоту.",
            ) from exc
        category_id = category_ids.get(box.class_name.value)
        if category_id is None:
            raise LabError(
                "unknown_class",
                f"Класса «{box.class_name.value}» нет в каталоге проекта. "
                "Предсказание в пакет не записывается.",
            )
        annotation: dict[str, Any] = {
            "id": len(annotations) + 1,
            "image_id": image_ids[frame.source_frame_index],
            "category_id": category_id,
            "bbox": [x, y, width, height],
            "area": width * height,
            "iscrowd": 0,
        }
        if box.score is not None:
            annotation["score"] = box.score
        annotations.append(annotation)
    document = {
        "info": {
            "description": (
                "not_ground_truth: предразметка модели, не эталонная разметка"
            ),
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return _json_bytes(document)


def _batch_document(
    *,
    batch_id: str,
    project_id: str,
    video_id: str,
    split: SplitName,
    split_content_hash: str,
    image_count: int,
    predictions_included: bool,
    other_bytes: int,
) -> bytes:
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": _BATCH_KIND,
        "batch_id": batch_id,
        "project_id": project_id,
        "video_id": video_id,
        "split": split.value,
        "split_content_hash": split_content_hash,
        "image_count": image_count,
        "predictions_included": predictions_included,
        "uncompressed_bytes": other_bytes,
    }
    payload = _json_bytes(document)
    for _ in range(8):
        total = other_bytes + len(payload)
        if document["uncompressed_bytes"] == total:
            return payload
        document["uncompressed_bytes"] = total
        payload = _json_bytes(document)
    raise LabError(
        "batch_json",
        "Не удалось записать размер архива в batch.json.",
    )


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    text = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
    return (text + "\n").encode("utf-8")


def _catalog_classes(catalog: StoredCatalog) -> list[StoredCatalogClass]:
    return sorted(catalog.classes, key=lambda item: item.id)


def _labels_yaml(catalog: StoredCatalog) -> bytes:
    """Annotation schema at export time. ``added_at`` stays out of the ZIP."""
    lines = [f"schema_version: {catalog.schema_version}", "classes:"]
    for item in _catalog_classes(catalog):
        lines.append(f"  - id: {item.id}")
        lines.append(f"    name: {item.name}")
        lines.append(f"    title: {_yaml_string(item.title)}")
        lines.append(f"    box: {_yaml_string(item.box)}")
    lines.append("visibility:")
    for item in catalog.visibility:
        lines.append(f"  - {item.value}")
    lines.append("review_status:")
    for item in catalog.review_status:
        lines.append(f"  - {item.value}")
    lines.append("attributes:")
    for item in catalog.attributes:
        lines.append(f"  - {item}")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _instructions(catalog: StoredCatalog) -> bytes:
    """Shared rules from the template, class table from the catalog rows."""
    classes = _catalog_classes(catalog)
    text = _read_repo_file(_template_path()).decode("utf-8")
    kept_newline = text.endswith("\n")
    lines = text.splitlines()
    start, end = _class_table_span(lines)
    lines[start : end + 1] = _class_section(classes).splitlines()
    rendered = "\n".join(lines)
    if not _same_seed_classes(classes):
        rendered = rendered.replace(" (тот же набор, что в `configs/labels.yaml`)", "")
    if len(classes) != 4:
        rendered = rendered.replace(
            "В этой версии размечаются только четыре класса объектов.",
            "В этой версии размечаются классы каталога проекта на момент выгрузки.",
        )
        rendered = rendered.replace(
            "В четыре класса детекции не входят действия и состояния:",
            "В классы детекции не входят действия и состояния:",
        )
    if kept_newline and not rendered.endswith("\n"):
        rendered += "\n"
    return rendered.encode("utf-8")


def _class_section(classes: Sequence[StoredCatalogClass]) -> str:
    heading = "## Четыре класса" if len(classes) == 4 else "## Классы каталога"
    rows = [
        heading,
        "",
        "| ID | Имя | Подпись | Что входит в рамку |",
        "|---|---|---|---|",
    ]
    for item in classes:
        rows.append(f"| {item.id} | `{item.name}` | {item.title} | {item.box} |")
    return "\n".join(rows)


def _class_table_span(lines: Sequence[str]) -> tuple[int, int]:
    start: int | None = None
    for index, line in enumerate(lines):
        if not line.startswith("## "):
            continue
        if index + 2 < len(lines) and lines[index + 2].startswith("| ID |"):
            start = index
            break
    if start is None:
        raise LabError(
            "template_missing",
            "В инструкции нет таблицы классов. Пакет не записывается.",
        )
    end = start + 1
    while end + 1 < len(lines) and lines[end + 1].startswith("|"):
        end += 1
    if end == start + 1:
        raise LabError(
            "template_missing",
            "В инструкции нет таблицы классов. Пакет не записывается.",
        )
    return start, end


def _same_seed_classes(classes: Sequence[StoredCatalogClass]) -> bool:
    seed = load_labels().classes
    if len(classes) != len(seed):
        return False
    for item, label in zip(classes, seed, strict=True):
        if item.id != label.id or item.name != label.name.value:
            return False
        if item.title != label.title or item.box != label.box:
            return False
    return True


def _template_path() -> Path:
    return repo_root() / "templates" / _TEMPLATE_NAME


def _read_repo_file(path: Path) -> bytes:
    if not path.is_file():
        raise LabError(
            "template_missing",
            f"Нет файла {path.name} для пакета разметки.",
        )
    return path.read_bytes()


def _write_zip(project_dir: Path, batch_id: str, members: Sequence[tuple[str, bytes]]) -> Path:
    directory = project_dir / "annotation_batches" / batch_id
    resolved = directory.resolve()
    if not resolved.is_relative_to(project_dir.resolve()):
        raise LabError("invalid_batch_id", "Недопустимый идентификатор пакета.")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "annotation_batch.zip"
    temporary = directory / "annotation_batch.zip.partial"
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for name, payload in members:
                add_zip_member(archive, name, payload)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination.resolve()


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
                    "exported",
                    str(destination),
                    created_at,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise LabError(
                "video_not_found",
                "Видео не найдено в этом проекте.",
            ) from exc
