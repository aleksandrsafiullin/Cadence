"""Per-project action catalog stored in project SQLite.

``name`` is an English token. ``title`` is the operator text passed in.
Seed titles are not read from ``excavation_lab.ui``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field, ValidationError, field_validator, model_validator

from excavation_lab import storage
from excavation_lab.errors import LabError
from excavation_lab.schemas import StrictModel

SEED_ACTIONS: tuple[str, ...] = (
    "digging",
    "swing_loaded",
    "dumping",
    "swing_empty",
    "idle",
)
ACTION_CLASS_LIMIT = 16
_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_INVALID_NAME = "Имя действия — английский токен: строчная буква, затем цифры или _."
_INVALID_TITLE = "У действия должна быть непустая подпись."
_UNKNOWN = "В каталоге нет действия «{name}»."
_DUPLICATE = "Имя «{name}» уже есть в каталоге действий. Повтор не записывается."
_FULL = (
    f"В каталоге действий не больше {ACTION_CLASS_LIMIT} классов. "
    "Новый класс не записан."
)
_IN_USE = (
    "Действие «{name}» есть в разметке или в class_order модели. Каталог не изменён."
)
_BUNDLE_UNREADABLE = (
    "Не читается action_bundle.json в {path}. Каталог действий не изменён."
)


class ActionClass(StrictModel):
    """One action class. Id 0 is a real class. ``added_at`` is timezone-aware UTC."""

    id: int = Field(ge=0)
    name: str
    title: str = Field(min_length=1)
    added_at: datetime

    @field_validator("name", mode="before")
    @classmethod
    def _token(cls, value: object) -> str:
        if isinstance(value, str) and _TOKEN.fullmatch(value) is not None:
            return value
        raise ValueError(_INVALID_NAME)

    @field_validator("added_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("added_at должен быть временем UTC")
        return value.astimezone(UTC)


class ActionCatalog(StrictModel):
    """One project's action classes. Between 1 and 16, unique id and name."""

    classes: list[ActionClass]

    @model_validator(mode="after")
    def _bounds(self) -> ActionCatalog:
        count = len(self.classes)
        if count < 1 or count > ACTION_CLASS_LIMIT:
            raise ValueError(
                f"в каталоге действий от 1 до {ACTION_CLASS_LIMIT} классов"
            )
        ids = [item.id for item in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError("повторяющийся id действия")
        names = [item.name for item in self.classes]
        if len(names) != len(set(names)):
            raise ValueError("повторяющееся имя действия")
        return self


def list_actions(workspace: Path, project_id: str) -> list[ActionClass]:
    """Action classes for one project, ordered by id. Empty before the first seed."""
    with storage.connect(_db(workspace, project_id)) as connection:
        return _load(connection, project_id)


def add_action(
    workspace: Path,
    project_id: str,
    name: str,
    title: str,
) -> ActionClass:
    """Append one class with the smallest unused id ``>= 0``. Existing ids stay."""
    token = _require_name(name)
    text = _require_title(title)
    with storage.connect(_db(workspace, project_id)) as connection:
        existing = _load(connection, project_id)
        if any(item.name == token for item in existing):
            raise LabError("duplicate_action_name", _DUPLICATE.format(name=token))
        if len(existing) >= ACTION_CLASS_LIMIT:
            raise LabError("action_catalog_full", _FULL)
        added = ActionClass(
            id=_smallest_unused_id(existing),
            name=token,
            title=text,
            added_at=datetime.now(UTC),
        )
        _require_catalog([*existing, added])
        _insert(connection, project_id, added)
    return added


def edit_title(
    workspace: Path,
    project_id: str,
    name: str,
    title: str,
) -> ActionClass:
    """Change ``title``. ``id``, ``name``, and ``added_at`` stay."""
    token = _require_name(name)
    text = _require_title(title)
    with storage.connect(_db(workspace, project_id)) as connection:
        current = _find(_load(connection, project_id), token)
        updated = current.model_copy(update={"title": text})
        cursor = connection.execute(
            """
            UPDATE action_classes
            SET title = ?
            WHERE project_id = ? AND name = ?
            """,
            (updated.title, project_id, token),
        )
        if cursor.rowcount != 1:
            raise LabError("unknown_action", _UNKNOWN.format(name=token))
    return updated


def rename_action(
    workspace: Path,
    project_id: str,
    name: str,
    new_name: str,
) -> ActionClass:
    """Rename ``name`` when no label and no bundle ``class_order`` uses it.

    ``id`` stays. A name listed in ``action_bundle.json`` is refused even
    when no ``action_labels`` row uses it.
    """
    token = _require_name(name)
    renamed = _require_name(new_name)
    with storage.connect(_db(workspace, project_id)) as connection:
        existing = _load(connection, project_id)
        current = _find(existing, token)
        if renamed != token and any(item.name == renamed for item in existing):
            raise LabError("duplicate_action_name", _DUPLICATE.format(name=renamed))
        _refuse_if_in_use(connection, project_id, token)
        updated = current.model_copy(update={"name": renamed})
        classes = [
            updated if item.name == token else item
            for item in existing
        ]
        _require_catalog(classes)
        cursor = connection.execute(
            """
            UPDATE action_classes
            SET name = ?
            WHERE project_id = ? AND name = ?
            """,
            (renamed, project_id, token),
        )
        if cursor.rowcount != 1:
            raise LabError("unknown_action", _UNKNOWN.format(name=token))
    return updated


def delete_action(workspace: Path, project_id: str, name: str) -> None:
    """Delete one class that no label and no bundle ``class_order`` uses."""
    token = _require_name(name)
    with storage.connect(_db(workspace, project_id)) as connection:
        _find(_load(connection, project_id), token)
        _refuse_if_in_use(connection, project_id, token)
        cursor = connection.execute(
            """
            DELETE FROM action_classes
            WHERE project_id = ? AND name = ?
            """,
            (project_id, token),
        )
        if cursor.rowcount != 1:
            raise LabError("unknown_action", _UNKNOWN.format(name=token))


def seed_actions(
    workspace: Path,
    project_id: str,
    titles: Mapping[str, str],
) -> list[ActionClass]:
    """Insert any missing seed class. Titles already stored are left as entered.

    ``titles`` maps a seed ``name`` to the operator title. Names already in
    the catalog are skipped, so a second seed does not rename them.
    """
    if not isinstance(titles, Mapping):
        raise LabError("invalid_action_title", _INVALID_TITLE)
    with storage.connect(_db(workspace, project_id)) as connection:
        existing = _load(connection, project_id)
        have = {item.name for item in existing}
        missing = [name for name in SEED_ACTIONS if name not in have]
        if not missing:
            return existing
        if len(existing) + len(missing) > ACTION_CLASS_LIMIT:
            raise LabError("action_catalog_full", _FULL)
        used = {item.id for item in existing}
        added_at = datetime.now(UTC)
        created: list[ActionClass] = []
        for name in missing:
            class_id = 0
            while class_id in used:
                class_id += 1
            used.add(class_id)
            created.append(
                ActionClass(
                    id=class_id,
                    name=name,
                    title=_title_for(name, titles),
                    added_at=added_at,
                )
            )
        _require_catalog([*existing, *created])
        for item in created:
            _insert(connection, project_id, item)
        return _load(connection, project_id)


def _db(workspace: Path, project_id: str) -> Path:
    return storage.project_sqlite(workspace, project_id)


def _load(connection: sqlite3.Connection, project_id: str) -> list[ActionClass]:
    rows = connection.execute(
        """
        SELECT id, name, title, added_at
        FROM action_classes
        WHERE project_id = ?
        ORDER BY id
        """,
        (project_id,),
    ).fetchall()
    return [_class_row(row) for row in rows]


def _class_row(row: sqlite3.Row) -> ActionClass:
    try:
        return ActionClass(
            id=row["id"],
            name=row["name"],
            title=row["title"],
            added_at=datetime.fromisoformat(str(row["added_at"])),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise LabError(
            "invalid_action",
            "Строка каталога действий не читается.",
        ) from exc


def _insert(
    connection: sqlite3.Connection,
    project_id: str,
    item: ActionClass,
) -> None:
    try:
        connection.execute(
            """
            INSERT INTO action_classes (project_id, id, name, title, added_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (project_id, item.id, item.name, item.title, item.added_at.isoformat()),
        )
    except sqlite3.IntegrityError as exc:
        raise LabError("duplicate_action_name", _DUPLICATE.format(name=item.name)) from exc


def _find(classes: Sequence[ActionClass], name: str) -> ActionClass:
    matches = [item for item in classes if item.name == name]
    if len(matches) != 1:
        raise LabError("unknown_action", _UNKNOWN.format(name=name))
    return matches[0]


def _smallest_unused_id(classes: Sequence[ActionClass]) -> int:
    used = {item.id for item in classes}
    candidate = 0
    while candidate in used:
        candidate += 1
    return candidate


def _require_catalog(classes: Sequence[ActionClass]) -> None:
    try:
        ActionCatalog(classes=list(classes))
    except ValidationError as exc:
        message = exc.errors()[0]["msg"]
        if message.startswith("Value error, "):
            message = message.removeprefix("Value error, ")
        code = "action_catalog_full" if "16" in message else "invalid_action"
        raise LabError(code, message) from exc


def _require_name(name: object) -> str:
    if not isinstance(name, str):
        raise LabError("invalid_action_name", _INVALID_NAME)
    token = name.strip()
    if _TOKEN.fullmatch(token) is None:
        raise LabError("invalid_action_name", _INVALID_NAME)
    return token


def _require_title(title: object) -> str:
    if not isinstance(title, str) or "\x00" in title or not title.strip():
        raise LabError("invalid_action_title", _INVALID_TITLE)
    return title.strip()


def _title_for(name: str, titles: Mapping[str, str]) -> str:
    if name not in titles:
        raise LabError(
            "invalid_action_title",
            f"Для действия «{name}» не передана подпись.",
        )
    return _require_title(titles[name])


def _refuse_if_in_use(
    connection: sqlite3.Connection,
    project_id: str,
    name: str,
) -> None:
    labeled = connection.execute(
        """
        SELECT 1 FROM action_labels
        WHERE project_id = ? AND action_name = ?
        LIMIT 1
        """,
        (project_id, name),
    ).fetchone()
    if labeled is not None:
        raise LabError("action_in_use", _IN_USE.format(name=name))
    rows = connection.execute(
        "SELECT path FROM action_models WHERE project_id = ?",
        (project_id,),
    ).fetchall()
    for row in rows:
        if name in _class_order(Path(row["path"])):
            raise LabError("action_in_use", _IN_USE.format(name=name))


def _class_order(bundle_dir: Path) -> list[str]:
    path = bundle_dir / "action_bundle.json"
    if path.is_symlink():
        raise LabError(
            "action_bundle_unreadable",
            _BUNDLE_UNREADABLE.format(path=bundle_dir),
        )
    if not path.is_file():
        return []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LabError(
            "action_bundle_unreadable",
            _BUNDLE_UNREADABLE.format(path=bundle_dir),
        ) from exc
    if not isinstance(document, dict):
        raise LabError("action_bundle_unreadable", _BUNDLE_UNREADABLE.format(path=bundle_dir))
    order = document.get("class_order")
    if order is None:
        return []
    if not isinstance(order, list) or any(not isinstance(item, str) for item in order):
        raise LabError("action_bundle_unreadable", _BUNDLE_UNREADABLE.format(path=bundle_dir))
    return order
