"""Workspace class catalogs.

Files live at ``workspace/catalogs/<catalog_id>/catalog.yaml``, not under
``workspace/projects/``. ``excavation`` is the seed copied from
``configs/labels.yaml`` on the first list or read.

``added_at`` is a timezone-aware UTC datetime. Compare instants with
``<`` and ``==``. A shared string prefix is not an earlier class.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn
from urllib.parse import quote

from pydantic import ValidationError, field_validator, model_validator

from excavation_lab.errors import LabError
from excavation_lab.schemas import (
    CATALOG_CLASS_LIMIT,
    CatalogClass,
    CatalogConfig,
    ReviewStatus,
    VersionedModel,
    Visibility,
    load_labels,
    read_yaml_mapping,
)

SEED_CATALOG_ID = "excavation"
_CATALOG_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CLASS_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class StoredCatalogClass(CatalogClass):
    """Catalog class plus the instant it was added.

    Id 0 is a real class, not background. ``added_at`` stays on this row
    when ``title`` or ``box`` changes.
    """

    added_at: datetime

    @field_validator("added_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("added_at must be timezone-aware")
        return value.astimezone(UTC)


class StoredCatalog(VersionedModel):
    """One catalog file. Class rules match ``CatalogConfig``."""

    classes: list[StoredCatalogClass]
    visibility: list[Visibility]
    review_status: list[ReviewStatus]
    attributes: list[str]

    @model_validator(mode="after")
    def _same_rules_as_catalog(self) -> StoredCatalog:
        try:
            CatalogConfig(
                schema_version=self.schema_version,
                classes=[
                    CatalogClass(
                        id=item.id,
                        name=item.name,
                        title=item.title,
                        box=item.box,
                    )
                    for item in self.classes
                ],
                visibility=self.visibility,
                review_status=self.review_status,
                attributes=list(self.attributes),
            )
        except ValidationError as exc:
            message = exc.errors()[0]["msg"]
            if message.startswith("Value error, "):
                message = message.removeprefix("Value error, ")
            raise ValueError(message) from exc
        return self


def list_catalogs(workspace: Path) -> list[str]:
    """Return catalog ids. Creates the ``excavation`` seed when it is absent."""
    root = _workspace(workspace)
    _ensure_seed(root)
    catalog_root = root / "catalogs"
    found: list[str] = []
    if not catalog_root.is_dir():
        return found
    for child in catalog_root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        if _CATALOG_ID_RE.fullmatch(child.name) is None:
            continue
        if (child / "catalog.yaml").is_file() and not (child / "catalog.yaml").is_symlink():
            found.append(child.name)
    return sorted(found)


def read_catalog(workspace: Path, catalog_id: str) -> StoredCatalog:
    """Read one catalog. The first read also creates the ``excavation`` seed."""
    root = _workspace(workspace)
    _require_catalog_id(catalog_id)
    _ensure_seed(root)
    return _read(_catalog_file(root, catalog_id), catalog_id)


def catalog_from_seed() -> StoredCatalog:
    """In-memory excavation seed from ``configs/labels.yaml``.

    A project does not receive this when its catalog id is missing.
    Callers that want the seed pass the object in.
    """
    labels = load_labels()
    added_at = datetime.now(UTC)
    return StoredCatalog(
        schema_version=labels.schema_version,
        classes=[
            StoredCatalogClass(
                id=item.id,
                name=item.name.value,
                title=item.title,
                box=item.box,
                added_at=added_at,
            )
            for item in labels.classes
        ],
        visibility=list(labels.visibility),
        review_status=list(labels.review_status),
        attributes=list(labels.attributes),
    )


def require_catalog_file(workspace: Path, catalog_id: str) -> Path:
    """Return ``catalog.yaml`` for an id that is already stored.

    Does not create the excavation seed and does not copy the file.
    """
    root = _workspace(workspace)
    path = _catalog_file(root, catalog_id)
    if not path.is_file() or path.is_symlink():
        raise LabError(
            "catalog_missing",
            f"Каталога «{catalog_id}» нет. Проект его не создаёт.",
        )
    return path


def require_project_catalog(workspace: Path, project_id: str) -> StoredCatalog:
    """Read the catalog named by the project's SQLite row.

    A null id, a missing column, or a missing catalog file raises.
    This does not open the seed file to fill a gap.
    """
    from excavation_lab.storage import connect, project_sqlite

    root = _workspace(workspace)
    db_path = project_sqlite(root, project_id)
    if not db_path.is_file() or db_path.is_symlink():
        raise LabError(
            "catalog_missing",
            f"У проекта «{project_id}» нет базы с каталогом.",
        )
    try:
        with connect(db_path) as connection:
            row = connection.execute("SELECT catalog_id FROM projects LIMIT 1").fetchone()
    except sqlite3.OperationalError as exc:
        raise LabError(
            "catalog_missing",
            f"У проекта «{project_id}» нет catalog_id.",
        ) from exc
    catalog_id = None if row is None else row["catalog_id"]
    if not isinstance(catalog_id, str) or not catalog_id.strip():
        raise LabError(
            "catalog_missing",
            f"У проекта «{project_id}» не указан каталог.",
        )
    path = _catalog_file(root, catalog_id)
    if not path.is_file() or path.is_symlink():
        raise LabError("catalog_missing", f"Каталога «{catalog_id}» нет.")
    return _read(path, catalog_id)


def create_catalog(
    workspace: Path,
    catalog_id: str,
    rows: Sequence[tuple[str, str, str]],
) -> StoredCatalog:
    """Write a new catalog from operator rows.

    ``title`` and ``box`` are stored as given. The seed classes are not
    copied in. Visibility, review status, and attributes come from the
    in-memory seed shape, not from a project.
    """
    root = _workspace(workspace)
    _require_new_catalog_id(catalog_id)
    destination = _catalog_file(root, catalog_id)
    if destination.is_file():
        raise LabError(
            "catalog_exists",
            f"Каталог «{catalog_id}» уже есть. Он не перезаписывается.",
        )
    if not rows:
        raise LabError("catalog_empty", "В новом каталоге нет классов.")
    shape = catalog_from_seed()
    added_at = datetime.now(UTC)
    classes: list[StoredCatalogClass] = []
    for index, (name, title, box) in enumerate(rows):
        _require_class_token(name)
        try:
            classes.append(
                StoredCatalogClass(
                    id=index,
                    name=name,
                    title=title,
                    box=box,
                    added_at=added_at,
                )
            )
        except ValidationError as exc:
            raise LabError("catalog_invalid", _validation_message(catalog_id, exc)) from exc
    try:
        catalog = StoredCatalog(
            schema_version=shape.schema_version,
            classes=classes,
            visibility=list(shape.visibility),
            review_status=list(shape.review_status),
            attributes=list(shape.attributes),
        )
    except ValidationError as exc:
        raise LabError("catalog_invalid", _validation_message(catalog_id, exc)) from exc
    _commit(destination, catalog)
    return _read(destination, catalog_id)


def copy_catalog(workspace: Path, source_id: str, catalog_id: str) -> StoredCatalog:
    """Write ``catalog_id`` as a copy of ``source_id``. The source file stays."""
    root = _workspace(workspace)
    _require_catalog_id(source_id)
    _require_new_catalog_id(catalog_id)
    _ensure_seed(root)
    source = _read(_catalog_file(root, source_id), source_id)
    destination = _catalog_file(root, catalog_id)
    if destination.is_file():
        raise LabError(
            "catalog_exists",
            f"Каталог «{catalog_id}» уже есть. Он не перезаписывается.",
        )
    _commit(destination, source)
    return _read(destination, catalog_id)


def add_class(
    workspace: Path,
    catalog_id: str,
    name: str,
    title: str,
    box: str,
) -> StoredCatalog:
    """Append one class with the smallest unused id ``>= 0``.

    Existing ids stay. A duplicate name or a 17th class is not written.
    """
    root = _workspace(workspace)
    _require_catalog_id(catalog_id)
    _require_class_token(name)
    _ensure_seed(root)
    path = _catalog_file(root, catalog_id)
    current = _read(path, catalog_id)
    if any(item.name == name for item in current.classes):
        raise LabError(
            "duplicate_class_name",
            f"Имя «{name}» уже есть в каталоге. Повтор не записывается.",
        )
    if len(current.classes) >= CATALOG_CLASS_LIMIT:
        raise LabError(
            "catalog_full",
            f"В каталоге не больше {CATALOG_CLASS_LIMIT} классов. Новый класс не записан.",
        )
    try:
        added = StoredCatalogClass(
            id=_smallest_unused_id(current.classes),
            name=name,
            title=title,
            box=box,
            added_at=datetime.now(UTC),
        )
    except ValidationError as exc:
        raise LabError("catalog_invalid", _validation_message(catalog_id, exc)) from exc
    updated = _rebuild(current, [*current.classes, added], catalog_id)
    _commit(path, updated)
    return _read(path, catalog_id)


def edit_class(
    workspace: Path,
    catalog_id: str,
    name: str,
    *,
    title: str | None = None,
    box: str | None = None,
) -> StoredCatalog:
    """Change ``title`` and/or ``box``. ``id``, ``name``, and ``added_at`` stay."""
    root = _workspace(workspace)
    _require_catalog_id(catalog_id)
    _ensure_seed(root)
    path = _catalog_file(root, catalog_id)
    current = _read(path, catalog_id)
    index = _find(current, name)
    if title is None and box is None:
        raise LabError(
            "invalid_edit",
            "Можно изменить только подпись и фразу рамки.",
        )
    classes = list(current.classes)
    changes: dict[str, str] = {}
    if title is not None:
        changes["title"] = title
    if box is not None:
        changes["box"] = box
    classes[index] = classes[index].model_copy(update=changes)
    updated = _rebuild(current, classes, catalog_id)
    _commit(path, updated)
    return _read(path, catalog_id)


def rename_class(
    workspace: Path,
    catalog_id: str,
    name: str,
    new_name: str,
) -> StoredCatalog:
    """Rename ``name`` when no box and no saved weights use it.

    There is no ``in_use`` argument. A name that appears is not written.
    """
    root = _workspace(workspace)
    _require_catalog_id(catalog_id)
    _require_class_token(new_name)
    _ensure_seed(root)
    path = _catalog_file(root, catalog_id)
    current = _read(path, catalog_id)
    index = _find(current, name)
    if any(item.name == new_name for item in current.classes):
        raise LabError(
            "duplicate_class_name",
            f"Имя «{new_name}» уже есть в каталоге. Повтор не записывается.",
        )
    if _name_in_use(root, catalog_id, name):
        raise LabError(
            "class_in_use",
            f"Класс «{name}» есть в разметке или в сохранённых весах. "
            "Файл каталога не изменён.",
        )
    classes = list(current.classes)
    classes[index] = classes[index].model_copy(update={"name": new_name})
    updated = _rebuild(current, classes, catalog_id)
    _commit(path, updated)
    return _read(path, catalog_id)


def delete_class(workspace: Path, catalog_id: str, name: str) -> NoReturn:
    """Refuse to delete a class. The catalog file is not written.

    There is no ``in_use`` argument. A name that appears is reported as in use.
    """
    root = _workspace(workspace)
    _require_catalog_id(catalog_id)
    _ensure_seed(root)
    path = _catalog_file(root, catalog_id)
    current = _read(path, catalog_id)
    _find(current, name)
    if _name_in_use(root, catalog_id, name):
        raise LabError(
            "class_in_use",
            f"Класс «{name}» есть в разметке или в сохранённых весах. "
            "Файл каталога не изменён.",
        )
    raise LabError(
        "class_delete_refused",
        "Удаление класса не выполняется. Файл каталога не изменён.",
    )


def _workspace(workspace: Path) -> Path:
    return Path(workspace).resolve()


def _require_catalog_id(catalog_id: str) -> None:
    if not isinstance(catalog_id, str) or _CATALOG_ID_RE.fullmatch(catalog_id) is None:
        raise LabError(
            "invalid_catalog_id",
            "Идентификатор каталога — латинский токен: строчная буква, затем цифры или _.",
        )


def _require_new_catalog_id(catalog_id: str) -> None:
    _require_catalog_id(catalog_id)
    if catalog_id == SEED_CATALOG_ID:
        raise LabError(
            "reserved_catalog_id",
            "Идентификатор excavation занят встроенным каталогом.",
        )


def _require_class_token(name: str) -> None:
    if not isinstance(name, str) or _CLASS_TOKEN_RE.fullmatch(name) is None:
        raise LabError(
            "invalid_class_name",
            "Имя класса — латинский токен: строчная буква, затем цифры или _.",
        )


def _catalog_file(workspace: Path, catalog_id: str) -> Path:
    _require_catalog_id(catalog_id)
    root = (workspace / "catalogs").resolve()
    path = (root / catalog_id / "catalog.yaml").resolve()
    if not path.is_relative_to(root):
        raise LabError(
            "invalid_catalog_id",
            "Идентификатор каталога — латинский токен: строчная буква, затем цифры или _.",
        )
    return path


def _ensure_seed(workspace: Path) -> None:
    path = _catalog_file(workspace, SEED_CATALOG_ID)
    if path.is_file():
        return
    catalog = catalog_from_seed()
    if path.is_file():
        return
    _commit(path, catalog)


def _read(path: Path, catalog_id: str) -> StoredCatalog:
    if not path.is_file() or path.is_symlink():
        raise LabError("catalog_missing", f"Каталога «{catalog_id}» нет.")
    try:
        return StoredCatalog.model_validate(read_yaml_mapping(path))
    except ValidationError as exc:
        raise LabError("catalog_invalid", _validation_message(catalog_id, exc)) from exc


def _rebuild(
    current: StoredCatalog,
    classes: list[StoredCatalogClass],
    catalog_id: str,
) -> StoredCatalog:
    try:
        return StoredCatalog(
            schema_version=current.schema_version,
            classes=classes,
            visibility=list(current.visibility),
            review_status=list(current.review_status),
            attributes=list(current.attributes),
        )
    except ValidationError as exc:
        raise LabError("catalog_invalid", _validation_message(catalog_id, exc)) from exc


def _commit(path: Path, catalog: StoredCatalog) -> None:
    try:
        checked = StoredCatalog.model_validate(catalog.model_dump(mode="json"))
    except ValidationError as exc:
        raise LabError(
            "catalog_invalid",
            _validation_message(path.parent.name, exc),
        ) from exc
    _atomic_write(path, _render(checked))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _render(catalog: StoredCatalog) -> str:
    lines = [f"schema_version: {catalog.schema_version}", "classes:"]
    for item in catalog.classes:
        lines.append(f"  - id: {item.id}")
        lines.append(f"    name: {item.name}")
        lines.append(f"    title: {_quote(item.title)}")
        lines.append(f"    box: {_quote(item.box)}")
        lines.append(f"    added_at: {_quote(_format_added_at(item.added_at))}")
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
    return "\n".join(lines)


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _format_added_at(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _smallest_unused_id(classes: list[StoredCatalogClass]) -> int:
    used = {item.id for item in classes}
    candidate = 0
    while candidate in used:
        candidate += 1
    return candidate


def _find(catalog: StoredCatalog, name: str) -> int:
    matches = [index for index, item in enumerate(catalog.classes) if item.name == name]
    if len(matches) != 1:
        raise LabError("unknown_class", f"В каталоге нет класса «{name}».")
    return matches[0]


def _validation_message(catalog_id: str, exc: ValidationError) -> str:
    message = exc.errors()[0]["msg"]
    if message.startswith("Value error, "):
        message = message.removeprefix("Value error, ")
    return f"Каталог «{catalog_id}» не принят: {message}"


def _name_in_use(workspace: Path, catalog_id: str, name: str) -> bool:
    # Rename and delete are the only callers. Add and title/box edit do not scan.
    # A project whose catalog_id names another catalog is skipped.
    # A missing column, a null id, or an unreadable database still counts.
    return _labels_have_name(workspace, catalog_id, name) or _weights_have_name(
        workspace, name
    )


def _labels_have_name(workspace: Path, catalog_id: str, name: str) -> bool:
    projects = workspace / "projects"
    if not projects.is_dir():
        return False
    for project_dir in _child_dirs(projects):
        referenced = _referenced_catalog_id(project_dir)
        if referenced is not None and referenced != catalog_id:
            continue
        batches = project_dir / "annotation_batches"
        if not batches.is_dir():
            continue
        for batch_dir in _child_dirs(batches):
            path = batch_dir / "imported_labels.json"
            if not path.is_file() or path.is_symlink():
                continue
            if name in _label_class_names(_read_json(path)):
                return True
    return False


def _weights_have_name(workspace: Path, name: str) -> bool:
    root = workspace / "weights"
    if not root.is_dir():
        return False
    for model_dir in _child_dirs(root):
        path = model_dir / "class_mapping.json"
        if not path.is_file() or path.is_symlink():
            continue
        if name in _mapping_class_names(_read_json(path)):
            return True
    return False


def _referenced_catalog_id(project_dir: Path) -> str | None:
    """Catalog id stored on the project, if that column exists.

    ``None`` means the database or the column is absent, or the value is
    null. The caller then scans the project's boxes.
    """
    db_path = project_dir / "project.sqlite"
    if not db_path.is_file() or db_path.is_symlink():
        return None
    try:
        connection = sqlite3.connect(_read_only_uri(db_path), uri=True)
    except sqlite3.Error:
        return None
    try:
        columns = connection.execute("PRAGMA table_info(projects)").fetchall()
        names = {row[1] for row in columns}
        if "catalog_id" not in names:
            return None
        row = connection.execute("SELECT catalog_id FROM projects LIMIT 1").fetchone()
    except sqlite3.Error:
        return None
    finally:
        connection.close()
    if row is None or not isinstance(row[0], str) or not row[0]:
        return None
    return row[0]


def _read_only_uri(path: Path) -> str:
    return "file:" + quote(path.resolve().as_posix(), safe="/:") + "?mode=ro"


def _child_dirs(root: Path) -> list[Path]:
    return sorted(
        (child for child in root.iterdir() if child.is_dir() and not child.is_symlink()),
        key=lambda child: child.name,
    )


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LabError(
            "catalog_scan_failed",
            f"Не удалось прочитать {path.name}. Файл каталога не изменён.",
        ) from exc


def _label_class_names(document: object) -> set[str]:
    if not isinstance(document, dict):
        return set()
    frames = document.get("frames")
    if not isinstance(frames, list):
        return set()
    names: set[str] = set()
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        for box in _frame_boxes(frame):
            if not isinstance(box, dict):
                continue
            class_name = box.get("class_name")
            if isinstance(class_name, str):
                names.add(class_name)
    return names


def _frame_boxes(frame: dict[str, Any]) -> list[object]:
    record = frame.get("record")
    if isinstance(record, dict) and isinstance(record.get("boxes"), list):
        return list(record["boxes"])
    boxes = frame.get("boxes")
    if isinstance(boxes, list):
        return list(boxes)
    return []


def _mapping_class_names(document: object) -> set[str]:
    if not isinstance(document, dict):
        return set()
    names: set[str] = set()
    by_name = document.get("by_name")
    if isinstance(by_name, dict):
        names.update(key for key in by_name if isinstance(key, str))
    class_names = document.get("class_names")
    if isinstance(class_names, list):
        names.update(item for item in class_names if isinstance(item, str))
    return names
