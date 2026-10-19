"""Check a snapshot before training.

This is dataset validation, not detection metrics and not the RF-DETR ``valid`` split.
Train and validation each need one reviewed box of every class. Images in
``test`` are not copied over to fill a gap. An unlabeled test split is the
warning ``test_not_labeled`` and does not fail the command by itself.

``reviewed``, ``unreviewed``, and ``excluded`` are separate counts.
``reviewed`` includes empty reviewed frames. ``reviewed_empty`` is that
subset and is the only negative. An unreviewed frame is not a negative.
A box that fails the check is left as it was written.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from excavation_lab.datasets.coco_import import ImportedFrame, load_imported_frames
from excavation_lab.datasets.geometry import xywh_to_xyxy
from excavation_lab.datasets.labels import LabelState
from excavation_lab.datasets.snapshot import RFDETR_SPLIT_DIRECTORIES
from excavation_lab.errors import LabError
from excavation_lab.schemas import (
    SCHEMA_VERSION,
    ReviewStatus,
    SplitName,
    StrictModel,
)
from excavation_lab.storage import connect, project_sqlite

_HEX = re.compile(r"^[0-9a-f]{64}$")
_NOT_FOUND = (
    "Снимок датасета не найден. Сначала соберите снимок командой dataset snapshot."
)


class ValidationIssue(StrictModel):
    """One snapshot problem. Row errors name ``image_file`` or a frame index."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    split: str | None = None
    class_name: str | None = None
    image_file: str | None = None
    source_frame_index: int | None = Field(default=None, ge=0)


class DatasetValidation(StrictModel):
    """Report for one ``dataset_versions/<id>`` tree.

    ``reviewed`` follows ``review_status`` and includes empty reviewed frames.
    ``negatives`` equals ``reviewed_empty``. Unreviewed frames stay out of it.
    """

    schema_version: int = SCHEMA_VERSION
    dataset_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewed: int = Field(ge=0)
    unreviewed: int = Field(ge=0)
    excluded: int = Field(ge=0)
    reviewed_empty: int = Field(ge=0)
    negatives: int = Field(ge=0)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    errors: list[ValidationIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every row is valid and train and validation cover all classes."""
        return not self.errors

    def failure_error(self) -> LabError:
        """English code and Russian message for a non-zero exit."""
        row_errors = [item for item in self.errors if item.code != "missing_class"]
        if row_errors:
            first = row_errors[0]
            return LabError(first.code, first.message)
        missing = [item for item in self.errors if item.code == "missing_class"]
        if not missing:
            return LabError("internal_error", "Внутренняя ошибка.")
        if len(missing) == 1:
            return LabError("missing_class", missing[0].message)
        listed = ", ".join(f"{item.split} «{item.class_name}»" for item in missing)
        return LabError(
            "missing_class",
            "В train или validation нет проверенной рамки: "
            f"{listed}. Кадры из test не копируются.",
        )


def validate_dataset(workspace: Path, project_id: str) -> DatasetValidation:
    """Validate the latest snapshot for ``project_id``. Does not train."""
    snapshot = _latest_snapshot(workspace, project_id)
    frames = load_imported_frames(workspace, project_id)
    return validate_snapshot(snapshot, frames, project_id=project_id)


def validate_snapshot(
    snapshot: Path,
    frames: Sequence[ImportedFrame],
    *,
    project_id: str,
) -> DatasetValidation:
    """Check JPEGs, hashes, boxes, class names, and split groups.

    Does not rewrite a box and does not copy an image from ``test``.
    """
    root = snapshot.resolve()
    manifest = _read_object(root / "dataset_manifest.json", "dataset_manifest.json")
    dataset_id, dataset_hash = _manifest_fields(manifest, project_id)
    if not (root / "split_manifest.json").is_file():
        raise LabError("invalid_snapshot", "В снимке нет split_manifest.json.")
    mapping = _class_ids(_read_object(root / "class_mapping.json", "class_mapping.json"))
    by_file, duplicate_errors = _index_frames(frames)
    errors: list[ValidationIssue] = list(duplicate_errors)
    present: dict[SplitName, set[str]] = {item: set() for item in SplitName}
    groups: dict[tuple[str, str], dict[SplitName, ImportedFrame]] = {}
    for split, directory_name in RFDETR_SPLIT_DIRECTORIES.items():
        _scan_split(
            root / directory_name,
            split,
            directory_name,
            mapping,
            by_file,
            errors,
            present[split],
            groups,
        )
    _append_crossovers(groups, errors)
    # Missing train or validation classes stay missing. test is not a donor.
    _append_missing_classes(present, _catalog_names(mapping), errors)
    warnings: list[ValidationIssue] = []
    if not present[SplitName.test]:
        warnings.append(
            ValidationIssue(
                code="test_not_labeled",
                message=(
                    "Часть test не размечена. Это предупреждение "
                    "и не отказ проверки перед обучением."
                ),
                split=SplitName.test.value,
            )
        )
    counts = _counts(frames)
    return DatasetValidation(
        dataset_id=dataset_id,
        project_id=project_id,
        path=str(root),
        dataset_hash=dataset_hash,
        warnings=warnings,
        errors=errors,
        **counts,
    )


def format_validation_summary(result: DatasetValidation) -> str:
    """Russian operator text: counts, then warnings and errors."""
    lines = [
        result.path,
        f"Проверенных: {result.reviewed}.",
        f"Непроверенных: {result.unreviewed}.",
        f"Исключённых: {result.excluded}.",
        f"Проверенных пустых: {result.reviewed_empty}.",
        f"Отрицательных: {result.negatives}.",
    ]
    lines.extend(f"{item.code}: {item.message}" for item in result.warnings)
    lines.extend(f"{item.code}: {item.message}" for item in result.errors)
    return "\n".join(lines)


def _latest_snapshot(workspace: Path, project_id: str) -> Path:
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT path FROM dataset_versions
            WHERE project_id = ?
            ORDER BY created_at DESC, dataset_id DESC
            LIMIT 1
            """,
            (project_id,),
        ).fetchone()
    path_text = None if row is None else row["path"]
    if not isinstance(path_text, str) or not path_text:
        raise LabError("dataset_not_found", _NOT_FOUND)
    snapshot = Path(path_text)
    if not snapshot.is_dir():
        raise LabError("dataset_not_found", f"Каталог снимка не найден: {snapshot}.")
    return snapshot


def _counts(frames: Sequence[ImportedFrame]) -> dict[str, int]:
    """Count review states. Unreviewed is not added to ``negatives``."""
    reviewed = 0
    unreviewed = 0
    excluded = 0
    reviewed_empty = 0
    negatives = 0
    for frame in frames:
        record = frame.record
        if record.review_status is ReviewStatus.reviewed:
            reviewed += 1
        elif record.review_status is ReviewStatus.unreviewed:
            unreviewed += 1
        elif record.review_status is ReviewStatus.excluded:
            excluded += 1
        if record.label_state is LabelState.reviewed_empty:
            reviewed_empty += 1
        if record.is_negative:
            negatives += 1
    return {
        "reviewed": reviewed,
        "unreviewed": unreviewed,
        "excluded": excluded,
        "reviewed_empty": reviewed_empty,
        "negatives": negatives,
    }


def _index_frames(
    frames: Sequence[ImportedFrame],
) -> tuple[dict[str, ImportedFrame], list[ValidationIssue]]:
    by_file: dict[str, ImportedFrame] = {}
    errors: list[ValidationIssue] = []
    for frame in frames:
        previous = by_file.get(frame.image_file)
        if previous is not None:
            errors.append(
                _issue(
                    "duplicate_image",
                    (
                        f"{_where(frame)} повторяется в импорте. "
                        "Хеш сверяется с первой записью."
                    ),
                    frame=frame,
                )
            )
            continue
        by_file[frame.image_file] = frame
    return by_file, errors


def _scan_split(
    split_dir: Path,
    split: SplitName,
    directory_name: str,
    mapping: Mapping[str, int],
    by_file: Mapping[str, ImportedFrame],
    errors: list[ValidationIssue],
    present: set[str],
    groups: dict[tuple[str, str], dict[SplitName, ImportedFrame]],
) -> None:
    if not split_dir.is_dir():
        raise LabError(
            "invalid_layout",
            f"В снимке нет каталога {directory_name}.",
        )
    label = f"{directory_name}/_annotations.coco.json"
    document = _read_object(split_dir / "_annotations.coco.json", label)
    images = _as_list(document.get("images"), label, "images")
    annotations = _as_list(document.get("annotations"), label, "annotations")
    categories = _as_list(document.get("categories"), label, "categories")
    names_by_id, bad_categories = _category_index(categories, mapping, label)
    images_by_id = _index_images(split_dir, split, images, by_file, errors, groups)
    seen_bad: set[int] = set()
    for annotation in annotations:
        _scan_annotation(
            annotation,
            split,
            images_by_id,
            names_by_id,
            bad_categories,
            by_file,
            errors,
            present,
            seen_bad,
        )
    for category_id, (code, name) in bad_categories.items():
        if category_id in seen_bad:
            continue
        frame, image_file = _first_image(images_by_id, by_file)
        errors.append(
            _issue(
                code,
                f"{_where(frame, image_file)}: {_category_text(code, name)}",
                split=split,
                frame=frame,
                image_file=image_file,
            )
        )


def _scan_annotation(
    annotation: object,
    split: SplitName,
    images_by_id: Mapping[int, dict[str, Any]],
    names_by_id: Mapping[int, str],
    bad_categories: Mapping[int, tuple[str, str]],
    by_file: Mapping[str, ImportedFrame],
    errors: list[ValidationIssue],
    present: set[str],
    seen_bad: set[int],
) -> None:
    if not isinstance(annotation, dict):
        errors.append(
            _issue(
                "coco_invalid",
                f"В части {split.value} аннотация не является объектом.",
                split=split,
            )
        )
        return
    image, image_file, frame = _annotation_image(annotation, images_by_id, by_file)
    category_id = annotation.get("category_id")
    if isinstance(category_id, bool) or not isinstance(category_id, int):
        errors.append(
            _issue(
                "unknown_category",
                f"{_where(frame, image_file)}: category_id не является целым.",
                split=split,
                frame=frame,
                image_file=image_file,
            )
        )
        return
    if category_id in bad_categories:
        seen_bad.add(category_id)
        code, name = bad_categories[category_id]
        errors.append(
            _issue(
                code,
                f"{_where(frame, image_file)}: {_category_text(code, name)}",
                split=split,
                frame=frame,
                image_file=image_file,
            )
        )
        return
    if image is None:
        errors.append(
            _issue(
                "coco_invalid",
                f"{_where(frame, image_file)}: аннотация ссылается на неизвестный кадр.",
                split=split,
                frame=frame,
                image_file=image_file,
            )
        )
        return
    name = names_by_id.get(category_id)
    if name is None:
        errors.append(
            _issue(
                "unknown_category",
                (
                    f"{_where(frame, image_file)}: категория {category_id} "
                    "отсутствует в class_mapping.json."
                ),
                split=split,
                frame=frame,
                image_file=image_file,
            )
        )
        return
    if not _box_inside(annotation.get("bbox"), image):
        errors.append(
            _issue(
                "invalid_bbox",
                (
                    f"{_where(frame, image_file)}: рамка вне изображения "
                    "или с неположительным размером. Рамка не исправляется."
                ),
                split=split,
                frame=frame,
                image_file=image_file,
            )
        )
        return
    file_ok = image.get("_file_ok") is True
    if file_ok and frame is not None and frame.record.review_status is ReviewStatus.reviewed:
        present.add(name)


def _annotation_image(
    annotation: Mapping[str, Any],
    images_by_id: Mapping[int, dict[str, Any]],
    by_file: Mapping[str, ImportedFrame],
) -> tuple[dict[str, Any] | None, str | None, ImportedFrame | None]:
    image_id = annotation.get("image_id")
    if isinstance(image_id, bool) or not isinstance(image_id, int):
        return None, None, None
    image = images_by_id.get(image_id)
    if image is None:
        return None, None, None
    image_file = image.get("_plain_name")
    file_name = image_file if isinstance(image_file, str) else None
    frame = by_file.get(file_name) if file_name is not None else None
    return image, file_name, frame


def _box_inside(bbox: object, image: Mapping[str, Any]) -> bool:
    """True when ``bbox`` lies inside the COCO image. Does not clip it."""
    size = _image_size(image)
    if size is None or isinstance(bbox, (str, bytes)) or not isinstance(bbox, Sequence):
        return False
    width, height = size
    try:
        xywh_to_xyxy(bbox, image_width=width, image_height=height)
    except (TypeError, ValueError):
        return False
    return True


def _image_size(image: Mapping[str, Any]) -> tuple[float, float] | None:
    width = image.get("width")
    height = image.get("height")
    if isinstance(width, bool) or isinstance(height, bool):
        return None
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        return None
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        return None
    return float(width), float(height)


def _index_images(
    split_dir: Path,
    split: SplitName,
    images: Sequence[object],
    by_file: Mapping[str, ImportedFrame],
    errors: list[ValidationIssue],
    groups: dict[tuple[str, str], dict[SplitName, ImportedFrame]],
) -> dict[int, dict[str, Any]]:
    images_by_id: dict[int, dict[str, Any]] = {}
    seen_names: set[str] = set()
    root = split_dir.resolve()
    for image in images:
        if not isinstance(image, dict):
            errors.append(
                _issue(
                    "coco_invalid",
                    f"В части {split.value} изображение не является объектом.",
                    split=split,
                )
            )
            continue
        image_id = image.get("id")
        file_name = _plain_name(image.get("file_name"))
        frame = by_file.get(file_name) if file_name is not None else None
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            errors.append(
                _issue(
                    "coco_invalid",
                    f"{_where(frame, file_name)}: image id не является целым.",
                    split=split,
                    frame=frame,
                    image_file=file_name,
                )
            )
            continue
        if image_id in images_by_id:
            errors.append(
                _issue(
                    "duplicate_image",
                    f"{_where(frame, file_name)} повторяется в части {split.value}.",
                    split=split,
                    frame=frame,
                    image_file=file_name,
                )
            )
            continue
        stored = dict(image)
        stored["_plain_name"] = file_name
        stored["_file_ok"] = False
        images_by_id[image_id] = stored
        if file_name is None:
            raw_name = image.get("file_name")
            shown = raw_name if isinstance(raw_name, str) else None
            errors.append(
                _issue(
                    "image_missing",
                    f"Файл «{shown}» в части {split.value} имеет недопустимое имя.",
                    split=split,
                    image_file=shown,
                )
            )
            continue
        if file_name in seen_names:
            errors.append(
                _issue(
                    "duplicate_image",
                    f"{_where(frame, file_name)} повторяется в части {split.value}.",
                    split=split,
                    frame=frame,
                    image_file=file_name,
                )
            )
            continue
        seen_names.add(file_name)
        stored["_file_ok"] = _check_jpeg(root, split_dir, split, file_name, frame, errors)
        if frame is None:
            continue
        if frame.split_group_id:
            # Same id on another video is not a leak. Snapshot uses this pair.
            key = (frame.video_id, frame.split_group_id)
            groups.setdefault(key, {}).setdefault(split, frame)
    return images_by_id


def _check_jpeg(
    root: Path,
    split_dir: Path,
    split: SplitName,
    file_name: str,
    frame: ImportedFrame | None,
    errors: list[ValidationIssue],
) -> bool:
    path = split_dir / file_name
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if not resolved.is_relative_to(root) or not resolved.is_file():
        errors.append(
            _issue(
                "image_missing",
                f"{_where(frame, file_name)}: JPEG нет в снимке.",
                split=split,
                frame=frame,
                image_file=file_name,
            )
        )
        return False
    if frame is None:
        errors.append(
            _issue(
                "frame_not_imported",
                f"Файл {file_name}: кадра нет в импорте, хеш не с чем сравнить.",
                split=split,
                image_file=file_name,
            )
        )
        return True
    digest = _file_sha256(resolved)
    if digest != frame.image_sha256:
        errors.append(
            _issue(
                "image_hash_mismatch",
                f"{_where(frame)}: SHA-256 не совпадает с импортом.",
                split=split,
                frame=frame,
            )
        )
        return True
    return True


def _append_crossovers(
    groups: Mapping[tuple[str, str], Mapping[SplitName, ImportedFrame]],
    errors: list[ValidationIssue],
) -> None:
    for video_id, group_id in sorted(groups):
        located = groups[(video_id, group_id)]
        if len(located) < 2:
            continue
        ordered = [located[split] for split in SplitName if split in located]
        frame = ordered[-1]
        joined = ", ".join(split.value for split in SplitName if split in located)
        errors.append(
            _issue(
                "split_group_crossover",
                (
                    f"Группа «{group_id}» видео «{video_id}» попала в несколько частей: "
                    f"{joined}. {_where(frame)}."
                ),
                frame=frame,
            )
        )


def _catalog_names(mapping: Mapping[str, int]) -> tuple[str, ...]:
    """Class names required by this snapshot, in catalog id order."""
    return tuple(name for name, _class_id in sorted(mapping.items(), key=lambda item: item[1]))


def _append_missing_classes(
    present: Mapping[SplitName, set[str]],
    required: Sequence[str],
    errors: list[ValidationIssue],
) -> None:
    for split in (SplitName.train, SplitName.validation):
        found = present[split]
        for name in required:
            if name in found:
                continue
            errors.append(
                ValidationIssue(
                    code="missing_class",
                    message=(
                        f"В части {split.value} нет проверенной рамки класса «{name}». "
                        "Кадры из test не копируются."
                    ),
                    split=split.value,
                    class_name=name,
                )
            )


def _category_index(
    categories: Sequence[object],
    mapping: Mapping[str, int],
    label: str,
) -> tuple[dict[int, str], dict[int, tuple[str, str]]]:
    names_by_id: dict[int, str] = {}
    bad: dict[int, tuple[str, str]] = {}
    for category in categories:
        if not isinstance(category, dict):
            raise LabError("invalid_snapshot", f"В {label} категория не является объектом.")
        name = category.get("name")
        category_id = category.get("id")
        if (
            not isinstance(name, str)
            or isinstance(category_id, bool)
            or not isinstance(category_id, int)
        ):
            raise LabError(
                "invalid_snapshot",
                f"В {label} у категории нет имени или целого id.",
            )
        if name not in mapping:
            bad[category_id] = ("unknown_category", name)
            continue
        if category_id != mapping[name]:
            bad[category_id] = ("class_map", name)
            continue
        names_by_id[category_id] = name
    return names_by_id, bad


def _category_text(code: str, name: str) -> str:
    if code == "class_map":
        return f"категория «{name}» не совпадает с class_mapping.json."
    return f"категория «{name}» отсутствует в class_mapping.json."


def _first_image(
    images_by_id: Mapping[int, Mapping[str, Any]],
    by_file: Mapping[str, ImportedFrame],
) -> tuple[ImportedFrame | None, str | None]:
    for image in images_by_id.values():
        file_name = image.get("_plain_name")
        if not isinstance(file_name, str):
            continue
        return by_file.get(file_name), file_name
    return None, None


def _manifest_fields(document: Mapping[str, Any], project_id: str) -> tuple[str, str]:
    dataset_id = document.get("dataset_id")
    dataset_hash = document.get("dataset_hash")
    owner = document.get("project_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise LabError("invalid_snapshot", "В dataset_manifest.json нет dataset_id.")
    if not isinstance(dataset_hash, str) or _HEX.fullmatch(dataset_hash) is None:
        raise LabError("invalid_snapshot", "В dataset_manifest.json нет dataset_hash.")
    if owner != project_id:
        raise LabError(
            "dataset_project_mismatch",
            f"Снимок относится к проекту «{owner}», а не «{project_id}».",
        )
    return dataset_id, dataset_hash


def _class_ids(document: Mapping[str, Any]) -> dict[str, int]:
    by_name = document.get("by_name")
    if not isinstance(by_name, dict) or not by_name:
        raise LabError("invalid_snapshot", "В class_mapping.json нет by_name.")
    found: dict[str, int] = {}
    for name, spec in by_name.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            raise LabError("invalid_snapshot", "class_mapping.json повреждён.")
        category_id = spec.get("coco_category_id")
        if isinstance(category_id, bool) or not isinstance(category_id, int):
            raise LabError(
                "invalid_snapshot",
                f"В class_mapping.json у «{name}» нет coco_category_id.",
            )
        found[name] = category_id
    return found


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise LabError("invalid_snapshot", f"В снимке нет файла {label}.")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LabError("invalid_snapshot", f"Файл {label} не читается.") from exc
    if not isinstance(document, dict):
        raise LabError("invalid_snapshot", f"Файл {label} не является JSON-объектом.")
    return document


def _as_list(value: object, label: str, key: str) -> list[Any]:
    if not isinstance(value, list):
        raise LabError("invalid_snapshot", f"В {label} поле {key} должно быть списком.")
    return value


def _plain_name(value: object) -> str | None:
    if not isinstance(value, str) or value in {"", ".", ".."}:
        return None
    if Path(value).name != value:
        return None
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _where(frame: ImportedFrame | None, image_file: str | None = None) -> str:
    if frame is not None:
        return f"Кадр {frame.video_id}:{frame.source_frame_index}, файл {frame.image_file}"
    if image_file:
        return f"Файл {image_file}"
    return "Кадр"


def _issue(
    code: str,
    message: str,
    *,
    split: SplitName | None = None,
    frame: ImportedFrame | None = None,
    image_file: str | None = None,
) -> ValidationIssue:
    file_name = image_file
    if file_name is None and frame is not None:
        file_name = frame.image_file
    index = None if frame is None else frame.source_frame_index
    return ValidationIssue(
        code=code,
        message=message,
        split=None if split is None else split.value,
        image_file=file_name,
        source_frame_index=index,
    )
