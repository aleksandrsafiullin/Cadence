"""Per-frame label store for one video split.

``annotation_batches/<video_id>__<split>/`` holds ``imported_labels.json``,
``attributes_sidecar.json``, and ``images/``. CVAT import and the in-app
labeler both upsert frames here. ``sample_rows_with_role`` lists sampled
frames with the frozen role. ``save_frame_label`` writes one frame from
the sampled JPEG. A save writes only the incoming JPEGs. Other frames
stay in both JSON documents and their image files are not opened.

Existing documents are parsed before any JPEG is written. Incoming bytes
go to a filename the current JSON does not cite. The sidecar is replaced
first and the labels file last, because loaders trust the labels file.
After both replaces, every regular file in ``images/`` that the new pair
does not name is removed. A crash before the labels replace leaves the
previous labels file and the JPEG it names readable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from pydantic import ValidationError

from excavation_lab.datasets.blocks import assignment_at as role_at
from excavation_lab.datasets.catalog import require_project_catalog
from excavation_lab.datasets.labels import (
    BoxDraft,
    FrameDraft,
    LabelRecord,
    LabelState,
    validate_frame,
)
from excavation_lab.datasets.sampler import SampleResult, SampleRow
from excavation_lab.datasets.splits import FrozenSplit, require_frozen
from excavation_lab.errors import LabError
from excavation_lab.schemas import SCHEMA_VERSION, ReviewStatus, SplitName, Visibility
from excavation_lab.storage import connect, project_directory, project_sqlite

_BATCH_STATUSES = frozenset({"imported", "labelled"})
_BATCH_LOCKS: dict[str, threading.Lock] = {}
_BATCH_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True, slots=True)
class FrameWrite:
    """One incoming frame. ``payload`` is the JPEG bytes for ``image_file``."""

    image_file: str
    source_frame_index: int
    source_pts: int
    time_base: str
    source_time_s: float
    episode_id: str
    split: SplitName
    split_group_id: str
    width: int
    height: int
    payload: bytes
    record: LabelRecord
    coco_subset: str
    image_attributes: dict[str, Any]
    sidecar_boxes: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class UpsertResult:
    """What ``upsert_frames`` stored. ``frame_count`` includes frames already there."""

    batch_id: str
    project_id: str
    video_id: str
    split: SplitName
    status: str
    labels_path: str
    sidecar_path: str
    frame_count: int


@dataclass(frozen=True, slots=True)
class SampleRowWithRole:
    """One sampled frame and the frozen role at its source second.

    ``role`` is ``None`` when that second lies outside every block. The row
    is still listed. ``save_frame_label`` does not write it.
    """

    row: SampleRow
    role: SplitName | None


def batch_directory(
    workspace: Path,
    project_id: str,
    video_id: str,
    split: SplitName | str,
) -> Path:
    """``annotation_batches/<video_id>__<split>`` under the project."""
    checked = _as_split(split)
    leaf = _batch_leaf(video_id, checked)
    project_dir = project_directory(workspace, project_id)
    directory = project_dir / "annotation_batches" / leaf
    resolved = directory.resolve()
    if not resolved.is_relative_to(project_dir.resolve()):
        raise LabError("invalid_batch_id", "Недопустимый идентификатор пакета.")
    return directory


def frozen_fields_for(
    frozen: FrozenSplit,
    source_time_s: float,
) -> tuple[SplitName, str, float, float] | None:
    """Split copied from the block that contains the source second.

    Role blocks have no ``split_group_id``. The second field stays empty.
    """
    role = role_at(frozen.blocks, source_time_s)
    if role is None:
        return None
    for block in frozen.blocks:
        if block.role == role and block.start_s <= source_time_s < block.end_s:
            return SplitName(role), "", block.start_s, block.end_s
    return None


def upsert_frames(
    workspace: Path,
    project_id: str,
    video_id: str,
    split: SplitName | str,
    writes: Sequence[FrameWrite],
    *,
    frozen: FrozenSplit,
    batch_status: str,
) -> UpsertResult:
    """Merge ``writes`` into the batch for ``(video_id, split)``.

    Each write must lie on ``split`` in ``frozen``. Incoming bytes are stored
    under a name the current JSON does not cite. ``batch_status`` is
    ``imported`` or ``labelled``.
    """
    checked = _as_split(split)
    status = _require_status(batch_status)
    incoming = _prepare_writes(writes)
    batch_dir = batch_directory(workspace, project_id, video_id, checked)
    with _lock_for(batch_dir):
        return _upsert_locked(
            workspace,
            project_id,
            video_id,
            checked,
            batch_dir,
            incoming,
            frozen,
            status,
        )


def sample_rows_with_role(
    workspace: Path,
    project_id: str,
    video_id: str,
) -> list[SampleRowWithRole]:
    """Join ``sample_manifest.json`` rows with ``frozen_fields_for``.

    Missing manifest is an empty list. A second outside every block is
    returned with ``role=None``.
    """
    frozen = require_frozen(workspace, project_id, video_id)
    result = _load_sample_result(workspace, project_id, video_id)
    if result is None:
        return []
    listed: list[SampleRowWithRole] = []
    for row in result.rows:
        fields = frozen_fields_for(frozen, row.source_time_s)
        role = None if fields is None else fields[0]
        listed.append(SampleRowWithRole(row=row, role=role))
    return listed


def save_frame_label(
    workspace: Path,
    project_id: str,
    video_id: str,
    source_frame_index: int,
    *,
    review_status: ReviewStatus | str,
    boxes: Sequence[BoxDraft],
    annotator: str | None,
    visibility: Visibility | None = None,
    note: str | None = None,
    now: datetime | None = None,
) -> LabelRecord:
    """Store one sampled frame's review and boxes.

    The split is the frozen role at the sample's source second. ``source_pts``
    and ``time_base`` are copied from that sample row. A row without them is
    refused. Nothing is derived from ``frame_index / fps``.
    """
    frozen = require_frozen(workspace, project_id, video_id)
    index = _require_sample_index(source_frame_index)
    frame_id = f"{video_id}:{index}"
    row = _sample_row(workspace, project_id, video_id, index, frame_id)
    # Legacy samples omit PTS. Do not invent it from frame_index / fps.
    if row.source_pts is None or row.time_base is None:
        raise LabError(
            "sample_without_pts",
            f"Кадр {frame_id}: в отборе нет source_pts. "
            "Повторите отбор кадров. Время не выводится из номера кадра.",
        )
    fields = frozen_fields_for(frozen, row.source_time_s)
    if fields is None:
        raise LabError(
            "frame_unassigned",
            f"Кадр {frame_id} не попадает ни в один блок. Кадр не записывается.",
        )
    split, split_group_id, _start_s, _end_s = fields
    image_file, payload = _sampled_jpeg(workspace, project_id, video_id, row, frame_id)
    width, height = _jpeg_size(payload, frame_id, image_file)
    drafts = _box_drafts(boxes, frame_id)
    for box in drafts:
        _reject_box(box, frame_id, width, height)
    status = _effective_review(_as_review(review_status), visibility, drafts)
    reviewed_at = None if status is ReviewStatus.unreviewed else _aware_instant(now)
    catalog = require_project_catalog(workspace, project_id)
    record = validate_frame(
        FrameDraft(
            video_id=video_id,
            source_frame_index=index,
            review_status=status,
            visibility=visibility,
            annotator=_annotator(annotator),
            reviewed_at=reviewed_at,
            note=note,
            boxes=drafts,
        ),
        catalog,
    )
    upsert_frames(
        workspace,
        project_id,
        video_id,
        split,
        [
            FrameWrite(
                image_file=image_file,
                source_frame_index=index,
                source_pts=row.source_pts,
                time_base=row.time_base,
                source_time_s=row.source_time_s,
                episode_id=row.episode_id,
                split=split,
                split_group_id=split_group_id,
                width=width,
                height=height,
                payload=payload,
                record=record,
                coco_subset="",
                image_attributes={},
                sidecar_boxes=[],
            )
        ],
        frozen=frozen,
        batch_status="labelled",
    )
    return record


def _lock_for(batch_dir: Path) -> threading.Lock:
    key = str(batch_dir.resolve())
    with _BATCH_LOCKS_GUARD:
        lock = _BATCH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _BATCH_LOCKS[key] = lock
        return lock


def _upsert_locked(
    workspace: Path,
    project_id: str,
    video_id: str,
    split: SplitName,
    batch_dir: Path,
    writes: Sequence[FrameWrite],
    frozen: FrozenSplit,
    batch_status: str,
) -> UpsertResult:
    for item in writes:
        _require_frame_split(item, split, frozen)
    batch_id = _batch_leaf(video_id, split)
    images_dir = batch_dir / "images"
    labels_path = batch_dir / "imported_labels.json"
    sidecar_path = batch_dir / "attributes_sidecar.json"
    batch_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    _reject_foreign_image_dir(images_dir, batch_dir)
    # Parse both documents before any JPEG write. A corrupt file must not
    # replace bytes that the previous JSON still names.
    existing_labels = _read_frames(labels_path)
    existing_sidecar = _read_frames(sidecar_path)
    taken = _image_names(existing_labels) | _image_names(existing_sidecar)
    staged: list[FrameWrite] = []
    for item in writes:
        stored_name = _unused_image_name(item.image_file, item.payload, taken)
        taken.add(stored_name)
        staged.append(replace(item, image_file=stored_name))
    for item in staged:
        _replace_bytes(images_dir / item.image_file, item.payload)
    label_frames = _merge_frames(
        existing_labels,
        [_frame_label_document(item, video_id) for item in staged],
        labels_path.name,
    )
    sidecar_frames = _merge_frames(
        existing_sidecar,
        [_frame_sidecar_document(item, video_id) for item in staged],
        sidecar_path.name,
    )
    _commit_documents(
        labels_path,
        sidecar_path,
        {
            "schema_version": SCHEMA_VERSION,
            "batch_id": batch_id,
            "project_id": project_id,
            "video_id": video_id,
            "split_content_hash": frozen.content_hash,
            "frames": label_frames,
        },
        {
            "schema_version": SCHEMA_VERSION,
            "batch_id": batch_id,
            "frames": sidecar_frames,
        },
    )
    _delete_uncited(
        images_dir,
        _image_names(label_frames) | _image_names(sidecar_frames),
    )
    stored_labels = str(labels_path.resolve())
    _upsert_batch_row(
        workspace,
        project_id,
        batch_id,
        video_id,
        batch_status,
        stored_labels,
    )
    return UpsertResult(
        batch_id=batch_id,
        project_id=project_id,
        video_id=video_id,
        split=split,
        status=batch_status,
        labels_path=stored_labels,
        sidecar_path=str(sidecar_path.resolve()),
        frame_count=len(label_frames),
    )


def _prepare_writes(writes: Sequence[FrameWrite]) -> list[FrameWrite]:
    if not writes:
        raise LabError("batch_empty", "В пакете нет кадров.")
    by_index: dict[int, FrameWrite] = {}
    order: list[int] = []
    files: dict[str, int] = {}
    for item in writes:
        if not isinstance(item, FrameWrite):
            raise LabError("invalid_import", "Кадр разметки должен быть FrameWrite.")
        image_file = _require_image_name(item.image_file)
        index = _require_index(item.source_frame_index, image_file)
        if not isinstance(item.payload, (bytes, bytearray)):
            raise LabError(
                "invalid_import",
                f"Кадр {image_file}: нет байтов JPEG.",
            )
        previous = files.get(image_file)
        if previous is not None and previous != index:
            raise LabError(
                "duplicate_image",
                f"Файл «{image_file}» повторяется в пакете.",
            )
        files[image_file] = index
        if index not in by_index:
            order.append(index)
        by_index[index] = item
    return [by_index[index] for index in order]


def _require_image_name(image_file: str) -> str:
    if (
        not isinstance(image_file, str)
        or not image_file
        or image_file in {".", ".."}
        or "/" in image_file
        or "\\" in image_file
        or "\x00" in image_file
    ):
        raise LabError(
            "invalid_image_name",
            "Имя изображения должно быть одним файлом без «..».",
        )
    return image_file


def _require_frame_split(write: FrameWrite, batch_split: SplitName, frozen: FrozenSplit) -> None:
    write_split = write.split if isinstance(write.split, SplitName) else _as_split(write.split)
    fields = frozen_fields_for(frozen, write.source_time_s)
    role = fields[0] if fields is not None else None
    if write_split is not batch_split or role is not batch_split:
        found = role.value if role is not None else "вне блоков"
        raise LabError(
            "frame_split_mismatch",
            f"Кадр {write.source_frame_index}: роль {found}, пакет {batch_split.value}. "
            "Кадр в эту часть не записывается.",
        )


def _require_index(index: object, image_file: str) -> int:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise LabError(
            "invalid_manifest",
            f"Кадр {image_file}: source_frame_index должен быть целым ≥ 0.",
        )
    return index


def _reject_foreign_image_dir(images_dir: Path, batch_dir: Path) -> None:
    if images_dir.is_symlink():
        raise LabError("invalid_batch_id", "Недопустимый идентификатор пакета.")
    resolved = images_dir.resolve()
    if not resolved.is_relative_to(batch_dir.resolve()):
        raise LabError("invalid_batch_id", "Недопустимый идентификатор пакета.")


def _replace_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_frames(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError("invalid_import", f"Файл {path.name} повреждён.") from exc
    if not isinstance(document, dict):
        raise LabError("invalid_import", f"Файл {path.name} повреждён.")
    frames = document.get("frames")
    if not isinstance(frames, list):
        raise LabError("invalid_import", f"Файл {path.name} повреждён.")
    parsed: list[dict[str, Any]] = []
    for raw in frames:
        if not isinstance(raw, dict):
            raise LabError("invalid_import", f"Файл {path.name} повреждён.")
        _entry_key(raw, path.name)
        image_file = raw.get("image_file")
        if (
            not isinstance(image_file, str)
            or not image_file
            or image_file in {".", ".."}
            or "/" in image_file
            or "\\" in image_file
            or "\x00" in image_file
        ):
            raise LabError("invalid_import", f"Файл {path.name} повреждён.")
        parsed.append(raw)
    return parsed


def _image_names(frames: Sequence[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for raw in frames:
        image_file = raw.get("image_file")
        if isinstance(image_file, str):
            names.add(image_file)
    return names


def _unused_image_name(requested: str, payload: bytes, taken: set[str]) -> str:
    """A JPEG name the current documents do not cite.

    The first save keeps ``requested``. A resave gets a content-hash suffix
    so the live file named by the previous JSON is not replaced.
    """
    if requested not in taken:
        return requested
    stem = PurePosixPath(requested).stem
    suffix = PurePosixPath(requested).suffix
    if suffix.lower() not in {".jpg", ".jpeg"}:
        suffix = ".jpg"
    digest = hashlib.sha256(payload).hexdigest()[:16]
    candidate = f"{stem}.{digest}{suffix}"
    generation = 2
    while candidate in taken:
        candidate = f"{stem}.{digest}.{generation}{suffix}"
        generation += 1
    return _require_image_name(candidate)


def _delete_uncited(images_dir: Path, cited: set[str]) -> None:
    """Remove regular files in ``images/`` that the new pair does not name.

    A crash can leave a JPEG no document cites. Unlink errors are ignored.
    Symlinks are not followed, and nothing outside this directory is touched.
    """
    try:
        entries = list(os.scandir(images_dir))
    except OSError:
        return
    root = images_dir.resolve()
    for entry in entries:
        try:
            name = entry.name
            if (
                not name
                or name in cited
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
                or "\x00" in name
            ):
                continue
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                continue
            path = Path(entry.path)
            if path.parent.resolve() != root:
                continue
            path.unlink()
        except OSError:
            continue


def _merge_frames(
    existing: Sequence[Mapping[str, Any]],
    incoming: Sequence[Mapping[str, Any]],
    filename: str,
) -> list[dict[str, Any]]:
    order: list[tuple[str, int]] = []
    stored: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in existing:
        key = _entry_key(raw, filename)
        if key not in stored:
            order.append(key)
        stored[key] = dict(raw)
    for raw in incoming:
        key = _entry_key(raw, filename)
        if key not in stored:
            order.append(key)
        stored[key] = dict(raw)
    return [stored[key] for key in order]


def _entry_key(raw: Mapping[str, Any], filename: str) -> tuple[str, int]:
    video_id = raw.get("video_id")
    index = raw.get("source_frame_index")
    if not isinstance(video_id, str) or isinstance(index, bool) or not isinstance(index, int):
        raise LabError("invalid_import", f"Файл {filename} повреждён.")
    return video_id, index


def _frame_label_document(write: FrameWrite, video_id: str) -> dict[str, Any]:
    if isinstance(write.split, SplitName):
        split = write.split.value
    else:
        split = _as_split(write.split).value
    return {
        "image_file": write.image_file,
        "video_id": video_id,
        "source_frame_index": write.source_frame_index,
        "source_pts": write.source_pts,
        "time_base": write.time_base,
        "source_time_s": write.source_time_s,
        "episode_id": write.episode_id,
        "split": split,
        "split_group_id": write.split_group_id,
        "image_sha256": hashlib.sha256(write.payload).hexdigest(),
        "size_bytes": len(write.payload),
        "width": write.width,
        "height": write.height,
        "coco_subset": write.coco_subset,
        "flagged": _flagged(write.record),
        "record": write.record.model_dump(mode="json"),
    }


def _frame_sidecar_document(write: FrameWrite, video_id: str) -> dict[str, Any]:
    return {
        "image_file": write.image_file,
        "video_id": video_id,
        "source_frame_index": write.source_frame_index,
        "review_status": write.record.review_status.value,
        "flagged": _flagged(write.record),
        "coco_subset": write.coco_subset,
        "attributes": dict(write.image_attributes),
        "boxes": [dict(box) for box in write.sidecar_boxes],
    }


def _flagged(record: LabelRecord) -> list[str]:
    if record.label_state is LabelState.unreviewed:
        return ["unreviewed"]
    return []


def _commit_documents(
    labels_path: Path,
    sidecar_path: Path,
    labels: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> None:
    """Replace the sidecar, then the labels file. Loaders trust labels.

    A failed labels replace puts the previous sidecar back. A leftover
    ``.bak`` is removed here and is not written over the live pair.
    """
    labels_tmp = labels_path.with_name(labels_path.name + ".tmp")
    sidecar_tmp = sidecar_path.with_name(sidecar_path.name + ".tmp")
    backup = sidecar_path.with_name(sidecar_path.name + ".bak")
    labels_backup = labels_path.with_name(labels_path.name + ".bak")
    _drop_stale_backup(backup)
    _drop_stale_backup(labels_backup)
    had_sidecar = sidecar_path.is_file() and not sidecar_path.is_symlink()
    sidecar_replaced = False
    try:
        labels_tmp.write_text(_json_text(labels), encoding="utf-8")
        sidecar_tmp.write_text(_json_text(sidecar), encoding="utf-8")
        if had_sidecar:
            if backup.is_symlink():
                raise LabError(
                    "invalid_import",
                    "Копия attributes_sidecar.json.bak — ссылка. Она не читается.",
                )
            backup.write_bytes(sidecar_path.read_bytes())
        os.replace(sidecar_tmp, sidecar_path)
        sidecar_replaced = True
        os.replace(labels_tmp, labels_path)
    except Exception:
        if (
            sidecar_replaced
            and had_sidecar
            and not backup.is_symlink()
            and backup.is_file()
        ):
            os.replace(backup, sidecar_path)
        elif sidecar_replaced and not had_sidecar and not sidecar_path.is_symlink():
            sidecar_path.unlink(missing_ok=True)
        labels_tmp.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)
        _drop_stale_backup(backup)
        raise
    labels_tmp.unlink(missing_ok=True)
    sidecar_tmp.unlink(missing_ok=True)
    _drop_stale_backup(backup)


def _drop_stale_backup(path: Path) -> None:
    """Remove a leftover backup. A symlink is unlinked, not followed."""
    try:
        info = path.lstat()
    except OSError:
        return
    if not stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
        return
    try:
        path.unlink()
    except OSError:
        return


def _json_text(document: Mapping[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _upsert_batch_row(
    workspace: Path,
    project_id: str,
    batch_id: str,
    video_id: str,
    status: str,
    destination: str,
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
                (batch_id, project_id, video_id, status, destination, created_at),
            )
        except sqlite3.IntegrityError as exc:
            raise LabError("video_not_found", "Видео не найдено в этом проекте.") from exc


def _as_split(split: SplitName | str) -> SplitName:
    if isinstance(split, SplitName):
        return split
    try:
        return SplitName(split)
    except ValueError as exc:
        raise LabError(
            "unknown_batch",
            "Идентификатор пакета имеет вид «<video_id>__train», "
            "«<video_id>__validation» или «<video_id>__test».",
        ) from exc


def _require_status(batch_status: str) -> str:
    if batch_status not in _BATCH_STATUSES:
        raise LabError(
            "invalid_batch_status",
            "Статус пакета должен быть imported или labelled.",
        )
    return batch_status


def _batch_leaf(video_id: str, split: SplitName) -> str:
    if (
        not isinstance(video_id, str)
        or not video_id
        or video_id in {".", ".."}
        or "/" in video_id
        or "\\" in video_id
        or "\x00" in video_id
    ):
        raise LabError("invalid_batch_id", "Недопустимый идентификатор пакета.")
    return f"{video_id}__{split.value}"


def _sample_paths(workspace: Path, project_id: str, video_id: str) -> tuple[Path, Path]:
    root = project_directory(workspace, project_id) / "samples" / video_id
    return root / "sample_manifest.json", root / "images"


def _load_sample_result(
    workspace: Path,
    project_id: str,
    video_id: str,
) -> SampleResult | None:
    path, _images = _sample_paths(workspace, project_id, video_id)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        result = SampleResult.model_validate(document)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise LabError(
            "invalid_manifest",
            "Файл sample_manifest.json повреждён.",
        ) from exc
    if result.video_id != video_id or result.project_id != project_id:
        raise LabError(
            "video_mismatch",
            f"Отбор относится к видео «{result.video_id}», запрошено «{video_id}».",
        )
    return result


def _sample_row(
    workspace: Path,
    project_id: str,
    video_id: str,
    index: int,
    frame_id: str,
) -> SampleRow:
    result = _load_sample_result(workspace, project_id, video_id)
    if result is None:
        raise LabError(
            "frame_not_sampled",
            f"Кадр {frame_id} нет в отборе. Сначала отберите кадры.",
        )
    for row in result.rows:
        if row.source_frame_index == index:
            return row
    raise LabError(
        "frame_not_sampled",
        f"Кадр {frame_id} нет в отборе. Сначала отберите кадры.",
    )


def _require_sample_index(index: object) -> int:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise LabError(
            "frame_not_sampled",
            "Номер кадра должен быть целым ≥ 0. Кадр не записан.",
        )
    return index


def _sampled_jpeg(
    workspace: Path,
    project_id: str,
    video_id: str,
    row: SampleRow,
    frame_id: str,
) -> tuple[str, bytes]:
    image_file = f"{video_id}_{row.source_frame_index}.jpg"
    _manifest, images = _sample_paths(workspace, project_id, video_id)
    path = images / image_file
    if row.image_file != image_file or not path.is_file():
        raise LabError(
            "image_missing",
            f"Кадр {frame_id}, файл {image_file}: изображения нет в отборе.",
        )
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != row.image_sha256:
        raise LabError(
            "image_hash_mismatch",
            f"Кадр {frame_id}, файл {image_file}: "
            "SHA-256 не совпадает с манифестом отбора.",
        )
    return image_file, payload


def _jpeg_size(payload: bytes, frame_id: str, image_file: str) -> tuple[int, int]:
    array = np.frombuffer(payload, dtype=np.uint8).copy()
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise LabError(
            "frame_unreadable",
            f"Кадр {frame_id}, файл {image_file}: JPEG не декодируется.",
        )
    return int(image.shape[1]), int(image.shape[0])


def _box_drafts(boxes: Sequence[BoxDraft], frame_id: str) -> list[BoxDraft]:
    if isinstance(boxes, (str, bytes)) or not isinstance(boxes, Sequence):
        raise LabError(
            "invalid_box",
            f"Кадр {frame_id}: рамка вне изображения или с неположительным размером.",
        )
    drafts: list[BoxDraft] = []
    for box in boxes:
        if isinstance(box, BoxDraft):
            drafts.append(box)
            continue
        try:
            drafts.append(BoxDraft.model_validate(box))
        except ValidationError as exc:
            raise LabError(
                "invalid_box",
                f"Кадр {frame_id}: рамка вне изображения или с неположительным размером.",
            ) from exc
    return drafts


def _reject_box(box: BoxDraft, frame_id: str, width: int, height: int) -> None:
    coords = (box.x1, box.y1, box.x2, box.y2)
    finite = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in coords
    )
    inside = (
        finite
        and box.x1 >= 0
        and box.y1 >= 0
        and box.x2 <= width
        and box.y2 <= height
        and box.x2 > box.x1
        and box.y2 > box.y1
    )
    if not inside:
        raise LabError(
            "invalid_box",
            f"Кадр {frame_id}: рамка вне изображения или с неположительным размером.",
        )


def _effective_review(
    review_status: ReviewStatus,
    visibility: Visibility | None,
    boxes: Sequence[BoxDraft],
) -> ReviewStatus:
    if visibility is Visibility.unlabelable:
        return ReviewStatus.excluded
    if any(box.visibility is Visibility.unlabelable for box in boxes):
        return ReviewStatus.excluded
    return review_status


def _as_review(value: ReviewStatus | str) -> ReviewStatus:
    if isinstance(value, ReviewStatus):
        return value
    try:
        return ReviewStatus(value)
    except ValueError as exc:
        raise LabError(
            "invalid_review",
            "Статус проверки должен быть unreviewed, reviewed или excluded.",
        ) from exc


def _aware_instant(now: datetime | None) -> datetime:
    if now is None:
        stamp = datetime.now(UTC)
    elif isinstance(now, datetime) and now.tzinfo is not None and now.utcoffset() is not None:
        stamp = now.astimezone(UTC)
    else:
        raise LabError(
            "invalid_reviewed_at",
            "Время проверки должно быть с часовым поясом. Наивное время не записывается.",
        )
    return stamp


def _annotator(annotator: str | None) -> str | None:
    if annotator is None:
        return None
    if not isinstance(annotator, str):
        raise LabError(
            "invalid_annotator",
            "Имя разметчика должно быть строкой.",
        )
    text = annotator.strip()
    return text or None
