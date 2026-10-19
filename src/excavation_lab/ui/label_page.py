"""In-app box labeler view-model.

No Starlette. Routes call these functions and ``label_store.save_frame_label``.
An unreviewed frame is stored and is not a negative. Suggestions are returned
to the page and are not written until the operator accepts them and saves.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import ValidationError

from excavation_lab import storage
from excavation_lab.datasets.catalog import require_project_catalog
from excavation_lab.datasets.coco_import import load_imported_frames
from excavation_lab.datasets.label_store import sample_rows_with_role, save_frame_label
from excavation_lab.datasets.labels import BoxDraft, LabelRecord, LabelState
from excavation_lab.errors import LabError, PathNotAllowedError
from excavation_lab.schemas import ReviewStatus, SplitName, Visibility
from excavation_lab.ui.copy import Catalog
from excavation_lab.ui.data_page import ActionNote, split_label

_DEFAULT_ANNOTATOR = "operator"
_ANNOTATOR_FILE = "annotator.txt"
_ANNOTATOR_LIMIT = 80
# The stored line is at most 80 characters plus a newline. One extra byte
# is enough to see that the file continues; the rest is not read.
_ANNOTATOR_READ_LIMIT = 82
_MAX_BOXES = 200
_JPEG_NAME = re.compile(r"^[\w.-]+\.jpe?g$", re.IGNORECASE)
_SPLITS = ("train", "validation", "test")
_NOT_FOUND = frozenset(
    {
        "video_not_found",
        "frame_not_found",
        "frame_not_sampled",
        "image_missing",
    }
)
_MESSAGE_ATTR = {
    "split_not_frozen": "label_not_frozen",
    "frame_not_sampled": "label_frame_not_sampled",
    "video_not_found": "label_unknown_video",
    "invalid_video_id": "label_unknown_video",
    "frame_not_found": "label_unknown_frame",
    "invalid_label": "label_invalid_json",
    "invalid_review": "label_invalid_review",
    "invalid_visibility": "label_invalid_visibility",
    "invalid_box": "label_invalid_box",
    "unknown_category": "label_unknown_class",
    "class_id_mismatch": "label_unknown_class",
    "frame_unassigned": "label_frame_unassigned",
    "image_missing": "label_image_missing",
    "invalid_annotator": "label_annotator_invalid",
    "unsupported_crowd": "label_unsupported_crowd",
    "catalog_missing": "catalog_required",
    "no_project": "no_project",
    "unsupported_media_type": "label_unsupported_media_type",
    "cross_site": "label_cross_site",
    "not_train_frame": "label_not_train_frame",
    "job_running": "label_job_running",
    "bundle_missing": "label_bundle_missing",
}
_SYNTHETIC_ROLE = "synthetic_wiring"
_BUNDLE_DOCS = (
    "model_config.json",
    "validation_metrics.json",
    "resolved_train_config.json",
)
_ACTIVE_JOBS = frozenset({"queued", "running", "cancel_requested"})


@dataclass(frozen=True, slots=True)
class LabelFrame:
    """One sampled frame merged with its stored label, if any."""

    index: int
    source_time_s: float
    split: str | None
    image_url: str
    review_status: str
    label_state: str
    box_count: int


def frames_for_video(
    workspace: Path,
    project_id: str,
    video_id: str,
) -> list[LabelFrame]:
    """Sample rows for ``video_id`` with stored review state.

    A row with no stored record is ``unreviewed`` and is not a negative.
    ``total`` for :func:`progress` is this list, not frames that were never sampled.
    """
    listed, _stored, _files = _collect(workspace, project_id, video_id)
    return listed


def frame_payload(
    workspace: Path,
    project_id: str,
    video_id: str,
    index: int,
    book: Catalog,
) -> dict[str, Any]:
    """One frame for the labeler. Raises ``LabError`` when it cannot be shown.

    Boxes are image pixels. ``suggestions_allowed`` is true only for a train
    frame when a non-synthetic custom bundle exists and no job is active.
    This function does not run the model.
    """
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise LabError("frame_not_found", "Кадр не найден в отборе этого видео.")
    listed, stored, files = _collect(workspace, project_id, video_id)
    position = next((i for i, frame in enumerate(listed) if frame.index == index), None)
    if position is None:
        if not listed:
            raise LabError(
                "frame_not_sampled",
                "Этого кадра нет в отборе. Сначала отберите кадры.",
            )
        raise LabError("frame_not_found", "Кадр не найден в отборе этого видео.")
    current = listed[position]
    record = stored.get(index)
    image_file = files.get(index)
    if image_file is None:
        raise LabError("image_missing", "Изображения этого кадра нет в отборе.")
    width, height = _image_size(workspace, project_id, video_id, image_file)
    role = current.split
    suggestion_block, suggestion_bundles = _suggestion_state(workspace, project_id, role)
    prev_index = listed[position - 1].index if position > 0 else None
    next_index = listed[position + 1].index if position + 1 < len(listed) else None
    return {
        "video_id": video_id,
        "index": index,
        "source_time_s": current.source_time_s,
        "split": role,
        "split_label": book.not_calculated if role is None else split_label(role, book),
        "width": width,
        "height": height,
        "image_url": current.image_url,
        "review_status": current.review_status,
        "label_state": current.label_state,
        "trainable": False if record is None else record.trainable,
        "is_negative": False if record is None else record.is_negative,
        "visibility": _visibility_token(None if record is None else record.visibility),
        "note": None if record is None else record.note,
        "annotator": None if record is None else record.annotator,
        "annotator_name": annotator_name(workspace),
        "reviewed_at": _instant(None if record is None else record.reviewed_at),
        "box_count": current.box_count,
        "boxes": [] if record is None else [_box_payload(box) for box in record.boxes],
        "classes": _class_rows(workspace, project_id),
        "suggestions_allowed": suggestion_block is None,
        "suggestion_block": suggestion_block,
        "suggestion_reason": (
            None if suggestion_block is None else message_for(book, suggestion_block)
        ),
        "suggestion_bundles": [
            {"model_id": model_id} for model_id, _path in suggestion_bundles
        ],
        "prev_index": prev_index,
        "next_index": next_index,
        "progress": _progress_from(listed),
    }


def save_frame(
    workspace: Path,
    project_id: str,
    video_id: str,
    index: int,
    body: Mapping[str, Any],
    book: Catalog,
) -> dict[str, Any] | ActionNote:
    """Validate one JSON body and store it through ``save_frame_label`` only."""
    try:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise LabError("frame_not_found", "Кадр не найден в отборе этого видео.")
        if not isinstance(body, Mapping):
            raise LabError("invalid_label", "Нужны review_status и список boxes.")
        review, drafts, visibility, note = _parsed_body(
            workspace,
            project_id,
            video_id,
            index,
            body,
        )
        save_frame_label(
            workspace,
            project_id,
            video_id,
            index,
            review_status=review,
            boxes=drafts,
            annotator=annotator_name(workspace),
            visibility=visibility,
            note=note,
        )
        return frame_payload(workspace, project_id, video_id, index, book)
    except LabError as exc:
        return ActionNote(text=message_for(book, exc.code), code=exc.code)
    except ValidationError:
        return ActionNote(text=message_for(book, "invalid_box"), code="invalid_box")


def suggest(
    workspace: Path,
    project_id: str,
    video_id: str,
    index: int,
    model_id: str,
    *,
    detector_cache: Any,
    loader: Callable[[Path], Any] | None = None,
    book: Catalog | None = None,
) -> list[dict[str, Any]] | ActionNote:
    """Boxes for one train frame. Nothing is written.

    Validation and test return ``not_train_frame``. An active job closes the
    cached adapter and returns ``job_running``. A missing, baseline, or
    synthetic bundle returns ``bundle_missing``. Kept boxes are catalog names
    at or above the pilot display threshold, each with ``suggested`` true.
    Names outside the catalog increment ``detector_cache.dropped``.
    """
    catalog = book if book is not None else _default_book()
    detector_cache.dropped = 0
    if _job_active(workspace):
        _close_cache(detector_cache)
        return _note(catalog, "job_running")
    try:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise LabError("frame_not_found", "Кадр не найден в отборе этого видео.")
        listed, _stored, files = _collect(workspace, project_id, video_id)
        current = next((frame for frame in listed if frame.index == index), None)
        if current is None:
            if not listed:
                raise LabError(
                    "frame_not_sampled",
                    "Этого кадра нет в отборе. Сначала отберите кадры.",
                )
            raise LabError("frame_not_found", "Кадр не найден в отборе этого видео.")
        if current.split != SplitName.train.value:
            return _note(catalog, "not_train_frame")
        directory = _listed_bundle(workspace, project_id, model_id)
        if directory is None or _bundle_refused(directory):
            return _note(catalog, "bundle_missing")
        if _job_active(workspace):
            _close_cache(detector_cache)
            return _note(catalog, "job_running")
        adapter = _cached_adapter(detector_cache, directory, loader or _default_loader)
        if _job_active(workspace):
            _close_cache(detector_cache)
            return _note(catalog, "job_running")
        image_file = files.get(index)
        if image_file is None:
            raise LabError("image_missing", "Изображения этого кадра нет в отборе.")
        frame = _image_rgb(workspace, project_id, video_id, image_file)
        detections = adapter.predict(frame, float(current.source_time_s))
        names = {item.name for item in require_project_catalog(workspace, project_id).classes}
        boxes, dropped = _suggestion_boxes(detections, names, _display_floor())
        detector_cache.dropped = dropped
        return boxes
    except LabError as exc:
        return _note(catalog, exc.code)


class SuggestionCache:
    """One detector. Tests and the page server both call ``close``."""

    def __init__(self) -> None:
        self.adapter: Any = None
        self.key: str | None = None
        self.dropped: int = 0

    def close(self) -> None:
        adapter = self.adapter
        self.adapter = None
        self.key = None
        if adapter is not None:
            adapter.close()


def progress(
    workspace: Path,
    project_id: str,
    video_id: str,
) -> dict[str, dict[str, int]]:
    """Per-split counts. ``total`` is the sampled frames in that role."""
    return _progress_from(frames_for_video(workspace, project_id, video_id))


def format_progress(counts: Mapping[str, Mapping[str, int]], book: Catalog) -> str:
    """One sentence per split, using ``book.label_progress``."""
    names = {
        "train": book.split_name_train,
        "validation": book.split_name_validation,
        "test": book.split_name_test,
        "unassigned": book.block_unassigned,
    }
    parts: list[str] = []
    for split, title in names.items():
        row = counts[split]
        parts.append(
            book.label_progress.format(
                split=title,
                reviewed=row["reviewed"],
                unreviewed=row["unreviewed"],
                excluded=row["excluded"],
                total=row["total"],
            )
        )
    return " ".join(parts)


def annotator_name(workspace: Path) -> str:
    """Preference from ``ui/annotator.txt``, or ``operator``. Does not write."""
    text = _read_annotator(workspace)
    if text is None:
        return _DEFAULT_ANNOTATOR
    try:
        return _clean_annotator(text)
    except LabError:
        return _DEFAULT_ANNOTATOR


def set_annotator(workspace: Path, name: str) -> str:
    """Atomically store one annotator line next to theme and locale."""
    cleaned = _clean_annotator(name)
    _write_annotator(workspace, cleaned)
    return cleaned


def parse_frame_token(value: object) -> int:
    """A decimal frame index from a query string. Otherwise ``frame_not_found``."""
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    raise LabError("frame_not_found", "Кадр не найден в отборе этого видео.")


def message_for(book: Catalog, code: str) -> str:
    """Locale sentence for a stable error code."""
    attr = _MESSAGE_ATTR.get(code, "label_save_failed")
    return str(getattr(book, attr))


def http_status(code: str) -> int:
    """404 when the video or frame is unknown. Other refusals are 400."""
    if code in _NOT_FOUND:
        return 404
    return 400


def script_json(payload: Mapping[str, Any]) -> str:
    """JSON for a ``<script type="application/json">`` body."""
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c")


def _collect(
    workspace: Path,
    project_id: str,
    video_id: str,
) -> tuple[list[LabelFrame], dict[int, LabelRecord], dict[int, str]]:
    _require_video(workspace, project_id, video_id)
    rows = sample_rows_with_role(workspace, project_id, video_id)
    stored = _stored_records(workspace, project_id, video_id)
    listed: list[LabelFrame] = []
    files: dict[int, str] = {}
    for item in rows:
        index = item.row.source_frame_index
        record = stored.get(index)
        role = None if item.role is None else item.role.value
        if record is None:
            # No stored record is unreviewed. It is not a negative and not trainable.
            review = ReviewStatus.unreviewed.value
            state = LabelState.unreviewed.value
            box_count = 0
        else:
            review = record.review_status.value
            state = record.label_state.value
            box_count = len(record.boxes)
        files[index] = item.row.image_file
        listed.append(
            LabelFrame(
                index=index,
                source_time_s=item.row.source_time_s,
                split=role,
                image_url=_image_url(video_id, item.row.image_file),
                review_status=review,
                label_state=state,
                box_count=box_count,
            )
        )
    return listed, stored, files


def _stored_records(
    workspace: Path,
    project_id: str,
    video_id: str,
) -> dict[int, LabelRecord]:
    found: dict[int, LabelRecord] = {}
    for frame in load_imported_frames(workspace, project_id):
        if frame.video_id == video_id:
            found[frame.source_frame_index] = frame.record
    return found


def _progress_from(frames: Sequence[LabelFrame]) -> dict[str, dict[str, int]]:
    counts = {
        split: {"reviewed": 0, "unreviewed": 0, "excluded": 0, "total": 0}
        for split in (*_SPLITS, "unassigned")
    }
    for frame in frames:
        # A second outside every block is not a split. It still counts.
        bucket = counts[frame.split] if frame.split in counts else counts["unassigned"]
        bucket["total"] += 1
        if frame.review_status == ReviewStatus.reviewed.value:
            bucket["reviewed"] += 1
        elif frame.review_status == ReviewStatus.excluded.value:
            bucket["excluded"] += 1
        else:
            bucket["unreviewed"] += 1
    return counts


def _suggestion_state(
    workspace: Path,
    project_id: str,
    split: str | None,
) -> tuple[str | None, list[tuple[str, Path]]]:
    """Block code and the bundles the select may offer.

    Train is the only split. A synthetic bundle is omitted the same way a
    baseline row is: it is not a custom detector for this catalog.
    """
    bundles = _eligible_bundles(workspace, project_id)
    if split != SplitName.train.value:
        return "not_train_frame", bundles
    if _job_active(workspace):
        return "job_running", bundles
    if not bundles:
        return "bundle_missing", bundles
    return None, bundles


def _job_active(workspace: Path) -> bool:
    from excavation_lab.jobs import reopen

    return any(job.status in _ACTIVE_JOBS for job in reopen(workspace))


def _eligible_bundles(workspace: Path, project_id: str) -> list[tuple[str, Path]]:
    return [
        (model_id, path)
        for model_id, path, refused in _bundle_rows(workspace, project_id)
        if not refused
    ]


def _listed_bundle(workspace: Path, project_id: str, model_id: str) -> Path | None:
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    token = model_id.strip()
    for found, path, _refused in _bundle_rows(workspace, project_id):
        if found == token:
            return path
    return None


def _bundle_rows(workspace: Path, project_id: str) -> list[tuple[str, Path, bool]]:
    found: list[tuple[str, Path, bool]] = []
    try:
        db_path = storage.project_sqlite(workspace, project_id)
    except LabError:
        return found
    try:
        with storage.connect(db_path) as connection:
            rows = connection.execute(
                """
                SELECT model_id, path FROM model_versions
                WHERE project_id = ?
                ORDER BY created_at, model_id
                """,
                (project_id,),
            ).fetchall()
    except (sqlite3.Error, LabError):
        return found
    seen: set[str] = set()
    for row in rows:
        model_id = row["model_id"]
        raw_path = row["path"]
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            continue
        if not isinstance(raw_path, str) or not raw_path:
            continue
        directory = Path(raw_path)
        weights = directory / "weights.pth"
        if weights.is_symlink() or not weights.is_file():
            continue
        seen.add(model_id)
        found.append((model_id, directory, _bundle_refused(directory)))
    return found


def _bundle_refused(directory: Path) -> bool:
    """Baseline is absent from ``model_versions``. Synthetic wiring is refused too."""
    for name in _BUNDLE_DOCS:
        document = _read_json_object(directory / name)
        if document is None:
            continue
        if document.get("synthetic") is True:
            return True
        if document.get("metric_role") == _SYNTHETIC_ROLE:
            return True
    return False


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def _display_floor() -> float:
    from excavation_lab.schemas import load_pilot

    return float(load_pilot().display_threshold)


def _default_loader(path: Path) -> Any:
    from excavation_lab.models.bundle import load_bundle

    return load_bundle(path)


def _cached_adapter(cache: Any, directory: Path, loader: Callable[[Path], Any]) -> Any:
    key = str(directory.resolve())
    current = cache.adapter
    if current is not None and cache.key == key:
        return current
    _close_cache(cache)
    opened = loader(directory)
    cache.adapter = opened
    cache.key = key
    return opened


def _close_cache(cache: Any) -> None:
    cache.close()


def _suggestion_boxes(
    detections: Any,
    names: set[str],
    floor: float,
) -> tuple[list[dict[str, Any]], int]:
    dropped = 0
    boxes: list[dict[str, Any]] = []
    xyxy = detections.xyxy
    for index, name in enumerate(detections.class_name):
        score = float(detections.score[index])
        if not math.isfinite(score) or score < floor:
            continue
        if not isinstance(name, str) or name not in names:
            dropped += 1
            continue
        x1, y1, x2, y2 = (float(value) for value in xyxy[index])
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        boxes.append(
            {
                "class_name": name,
                "score": score,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "suggested": True,
            }
        )
    return boxes, dropped


def _note(book: Catalog, code: str) -> ActionNote:
    return ActionNote(text=message_for(book, code), code=code)


def _default_book() -> Catalog:
    from excavation_lab.ui.copy import catalog

    return catalog("ru")


def _class_rows(workspace: Path, project_id: str) -> list[dict[str, Any]]:
    catalog = require_project_catalog(workspace, project_id)
    ordered = sorted(catalog.classes, key=lambda item: (item.id, item.name))
    rows: list[dict[str, Any]] = []
    for item in ordered:
        key = str(len(rows) + 1) if len(rows) < 9 else None
        rows.append(
            {
                "id": item.id,
                "name": item.name,
                "title": item.title,
                "box": item.box,
                "key": key,
            }
        )
    return rows


def _parsed_body(
    workspace: Path,
    project_id: str,
    video_id: str,
    index: int,
    body: Mapping[str, Any],
) -> tuple[ReviewStatus, list[BoxDraft], Visibility | None, str | None]:
    if "review_status" not in body or "boxes" not in body:
        raise LabError("invalid_label", "Нужны review_status и список boxes.")
    review = _review(body["review_status"])
    visibility = _optional_visibility(body.get("visibility")) if "visibility" in body else None
    note = _optional_text(body.get("note"), "invalid_label") if "note" in body else None
    raw_boxes = body["boxes"]
    if not isinstance(raw_boxes, list):
        raise LabError("invalid_label", "Нужны review_status и список boxes.")
    if len(raw_boxes) > _MAX_BOXES:
        raise LabError("invalid_label", "Нужны review_status и список boxes.")
    listed, _stored, files = _collect(workspace, project_id, video_id)
    if not any(frame.index == index for frame in listed):
        if not listed:
            raise LabError(
                "frame_not_sampled",
                "Этого кадра нет в отборе. Сначала отберите кадры.",
            )
        raise LabError("frame_not_found", "Кадр не найден в отборе этого видео.")
    image_file = files.get(index)
    if image_file is None:
        raise LabError("image_missing", "Изображения этого кадра нет в отборе.")
    width, height = _image_size(workspace, project_id, video_id, image_file)
    drafts = [_draft_box(box, width, height) for box in raw_boxes]
    return review, drafts, visibility, note


def _draft_box(box: object, width: int, height: int) -> BoxDraft:
    if not isinstance(box, Mapping):
        raise LabError("invalid_label", "Нужны review_status и список boxes.")
    name = box["class_name"] if "class_name" in box else box.get("category_name")
    if (
        "class_name" in box
        and "category_name" in box
        and box["class_name"] != box["category_name"]
    ):
        raise LabError("invalid_label", "Нужны review_status и список boxes.")
    if not isinstance(name, str) or not name.strip():
        raise LabError("unknown_category", "Имени класса нет в каталоге проекта.")
    if "iscrowd" in box and (isinstance(box["iscrowd"], bool) or box["iscrowd"] != 0):
        raise LabError("unsupported_crowd", "iscrowd, отличное от 0, не записывается.")
    visibility = Visibility.clear
    if "visibility" in box and box["visibility"] is not None:
        visibility = _required_visibility(box["visibility"])
    truncated = False
    if "truncated" in box and box["truncated"] is not None:
        if not isinstance(box["truncated"], bool):
            raise LabError("invalid_label", "Нужны review_status и список boxes.")
        truncated = box["truncated"]
    instance = None
    if "instance_local_id" in box:
        instance = _optional_text(box["instance_local_id"], "invalid_label")
    note = None
    if "note" in box:
        note = _optional_text(box["note"], "invalid_label")
    x1, y1, x2, y2 = _box_inside_image(
        _finite(box.get("x1")),
        _finite(box.get("y1")),
        _finite(box.get("x2")),
        _finite(box.get("y2")),
        width,
        height,
    )
    return BoxDraft(
        category_name=name.strip(),
        visibility=visibility,
        truncated=truncated,
        instance_local_id=instance,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        note=note,
    )


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError(
            "invalid_box",
            "Рамка вне изображения или с неположительным размером.",
        )
    number = float(value)
    if not math.isfinite(number):
        raise LabError(
            "invalid_box",
            "Рамка вне изображения или с неположительным размером.",
        )
    return number


def _box_inside_image(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Refuse a non-positive box, or one that is mostly outside, then clamp.

    The comparison uses the raw edges. A box whose overlap with the image is
    under half its original area is ``invalid_box``. A one- or two-pixel
    overflow still overlaps enough and is clamped to the image.
    """
    span_x = x2 - x1
    span_y = y2 - y1
    if span_x <= 0 or span_y <= 0:
        raise LabError(
            "invalid_box",
            "Рамка вне изображения или с неположительным размером.",
        )
    overlap_x = min(x2, float(width)) - max(x1, 0.0)
    overlap_y = min(y2, float(height)) - max(y1, 0.0)
    if overlap_x <= 0 or overlap_y <= 0 or overlap_x * overlap_y < 0.5 * span_x * span_y:
        raise LabError(
            "invalid_box",
            "Рамка вне изображения или с неположительным размером.",
        )
    return (
        _clamp(x1, width),
        _clamp(y1, height),
        _clamp(x2, width),
        _clamp(y2, height),
    )


def _clamp(value: float, limit: int) -> float:
    if value < 0:
        return 0.0
    if value > limit:
        return float(limit)
    return value


def _review(value: object) -> ReviewStatus:
    if not isinstance(value, str):
        raise LabError(
            "invalid_review",
            "Статус проверки: unreviewed, reviewed или excluded.",
        )
    try:
        return ReviewStatus(value)
    except ValueError as exc:
        raise LabError(
            "invalid_review",
            "Статус проверки: unreviewed, reviewed или excluded.",
        ) from exc


def _required_visibility(value: object) -> Visibility:
    if not isinstance(value, str):
        raise LabError("invalid_visibility", "Видимость: clear, partial или unlabelable.")
    try:
        return Visibility(value)
    except ValueError as exc:
        raise LabError(
            "invalid_visibility",
            "Видимость: clear, partial или unlabelable.",
        ) from exc


def _optional_visibility(value: object) -> Visibility | None:
    if value is None:
        return None
    return _required_visibility(value)


def _optional_text(value: object, code: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LabError(code, "Нужны review_status и список boxes.")
    text = value.strip()
    return text or None


def _box_payload(box: Any) -> dict[str, Any]:
    return {
        "class_name": box.class_name,
        "visibility": box.visibility.value,
        "truncated": box.truncated,
        "instance_local_id": box.instance_local_id,
        "note": box.note,
        "x1": box.x1,
        "y1": box.y1,
        "x2": box.x2,
        "y2": box.y2,
    }


def _visibility_token(value: Visibility | None) -> str | None:
    if value is None:
        return None
    return value.value


def _instant(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        return value.isoformat()
    return value.astimezone(UTC).isoformat()


def _image_url(video_id: str, image_file: str) -> str:
    return "/data/frame?" + "&".join(
        f"{_url_part(key)}={_url_part(value)}"
        for key, value in (("video", video_id), ("name", image_file))
    )


def _url_part(value: str) -> str:
    pieces: list[str] = []
    for char in value:
        if char.isascii() and (char.isalnum() or char in "-._~"):
            pieces.append(char)
            continue
        pieces.append("".join(f"%{byte:02X}" for byte in char.encode("utf-8")))
    return "".join(pieces)


def _image_size(
    workspace: Path,
    project_id: str,
    video_id: str,
    image_file: str,
) -> tuple[int, int]:
    image = _image_bgr(workspace, project_id, video_id, image_file)
    return int(image.shape[1]), int(image.shape[0])


def _image_rgb(
    workspace: Path,
    project_id: str,
    video_id: str,
    image_file: str,
) -> Any:
    from excavation_lab.models.preprocess import bgr_to_rgb

    return bgr_to_rgb(_image_bgr(workspace, project_id, video_id, image_file))


def _image_bgr(
    workspace: Path,
    project_id: str,
    video_id: str,
    image_file: str,
) -> Any:
    if _unsafe_leaf(video_id) or _unsafe_jpeg(image_file):
        raise LabError("image_missing", "Изображения этого кадра нет в отборе.")
    project = storage.project_directory(workspace, project_id).resolve()
    images = (project / "samples" / video_id / "images").resolve()
    if not images.is_relative_to(project):
        raise LabError("image_missing", "Изображения этого кадра нет в отборе.")
    path = images / image_file
    if path.is_symlink():
        raise LabError("image_missing", "Изображения этого кадра нет в отборе.")
    resolved = path.resolve()
    if (
        resolved.parent != images
        or not resolved.is_file()
        or resolved.suffix.lower() not in {".jpg", ".jpeg"}
    ):
        raise LabError("image_missing", "Изображения этого кадра нет в отборе.")
    array = np.frombuffer(resolved.read_bytes(), dtype=np.uint8).copy()
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise LabError("frame_unreadable", "JPEG не декодируется.")
    return image


def _require_video(workspace: Path, project_id: str, video_id: str) -> None:
    if _unsafe_leaf(video_id):
        raise LabError("video_not_found", "Видео не найдено в этом проекте.")
    try:
        storage.get_video(workspace, project_id, video_id)
    except LabError as exc:
        if exc.code == "video_not_found":
            raise
        raise LabError("video_not_found", "Видео не найдено в этом проекте.") from exc


def _unsafe_leaf(value: str) -> bool:
    """Same leaf rules as the data-page frame route."""
    if not isinstance(value, str) or not value or value in {".", ".."}:
        return True
    if value != value.strip():
        return True
    if any(char in value for char in "/\\\x00"):
        return True
    return ".." in value


def _unsafe_jpeg(value: str) -> bool:
    if _unsafe_leaf(value):
        return True
    return _JPEG_NAME.fullmatch(value) is None


def _clean_annotator(name: object) -> str:
    if not isinstance(name, str):
        raise LabError("invalid_annotator", "Укажите имя разметчика одной строкой.")
    text = name.strip()
    if not text or len(text) > _ANNOTATOR_LIMIT or text in {".", ".."} or ".." in text:
        raise LabError("invalid_annotator", "Укажите имя разметчика одной строкой.")
    if any(ord(char) < 32 or char in "/\\" for char in text):
        raise LabError("invalid_annotator", "Укажите имя разметчика одной строкой.")
    return text


def _read_annotator(workspace: Path) -> str | None:
    root = Path(workspace).resolve()
    path = root / "ui" / _ANNOTATOR_FILE
    try:
        escaped = (path.is_symlink() or path.exists()) and not path.resolve().is_relative_to(
            root
        )
        if escaped or not path.is_file():
            return None
        with path.open("rb") as handle:
            raw = handle.read(_ANNOTATOR_READ_LIMIT)
        return raw.decode("utf-8")
    except (OSError, UnicodeError):
        return None


def _write_annotator(workspace: Path, name: str) -> None:
    root = Path(workspace).resolve()
    ui_dir = root / "ui"
    _require_inside(ui_dir, root)
    target = ui_dir / _ANNOTATOR_FILE
    if target.exists() or target.is_symlink():
        _require_inside(target, root)
    ui_dir.mkdir(parents=True, exist_ok=True)
    directory = _require_inside(ui_dir, root)
    destination = directory / _ANNOTATOR_FILE
    if destination.exists() or destination.is_symlink():
        _require_inside(destination, root)
    fd, tmp_name = tempfile.mkstemp(prefix=".annotator-", suffix=".tmp", dir=directory)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            _require_inside(tmp_path, root)
            handle.write(name + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _require_inside(destination, root)
        os.replace(tmp_path, destination)
        final = destination.resolve()
        if not final.is_relative_to(root):
            destination.unlink(missing_ok=True)
            raise PathNotAllowedError(str(final))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _require_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise PathNotAllowedError(str(resolved))
    return resolved
