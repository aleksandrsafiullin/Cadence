"""Workspace project id and one-shot flash.

Both files live in ``<workspace>/ui/``. A symlink that resolves outside the
workspace is refused. Reads do not create directories or files. Flash is one
JSON object replaced in place, not a log of notices.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from excavation_lab.errors import PathNotAllowedError
from excavation_lab.storage import validate_project_name

_PROJECT_FILE = "project.txt"
_FLASH_FILE = "flash.json"


def project_path(workspace: Path) -> Path:
    """``<workspace>/ui/project.txt``. Does not create the file."""
    return Path(workspace) / "ui" / _PROJECT_FILE


def flash_path(workspace: Path) -> Path:
    """``<workspace>/ui/flash.json``. Does not create the file."""
    return Path(workspace) / "ui" / _FLASH_FILE


def choose_project_id(
    workspace: Path,
    projects: Sequence[Mapping[str, str]],
) -> str | None:
    """Stored id when it is listed, otherwise the only project.

    Several projects and an empty file stay unselected. This does not write.
    """
    stored = read_project_file(workspace)
    ids = [item["project_id"] for item in projects]
    if stored in ids:
        return stored
    if len(ids) == 1:
        return ids[0]
    return None


def read_project_file(workspace: Path) -> str | None:
    """Return the stored project id, or ``None`` when the file is unusable.

    A missing file, a blank line, or a symlink that leaves the workspace is
    ``None``. This does not create directories and does not write.
    """
    text = _read_text(workspace, _PROJECT_FILE)
    if text is None:
        return None
    value = text.strip()
    if not value or any(char.isspace() for char in value):
        return None
    if "/" in value or "\\" in value:
        return None
    return value


def write_project_file(workspace: Path, project_id: str) -> None:
    """Atomically write one project id.

    The temp file is created in ``ui/`` and ``os.replace``d onto ``project.txt``.
    After resolve, that file must stay inside the resolved workspace. A symlink
    that leaves the root is refused. The id must pass ``validate_project_name``.
    """
    name = validate_project_name(project_id)
    _atomic_write(workspace, _PROJECT_FILE, name + "\n", ".project-")


def read_flash(workspace: Path) -> dict[str, Any] | None:
    """Return the flash object, or ``None`` when the file is unusable.

    A missing file, invalid JSON, a JSON value that is not an object, or a
    symlink that leaves the workspace is ``None``. This does not create the file.
    """
    text = _read_text(workspace, _FLASH_FILE)
    if text is None:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if type(payload) is not dict:
        return None
    return payload


def write_flash(workspace: Path, payload: dict[str, Any]) -> None:
    """Replace ``flash.json`` with one JSON object.

    A list or any other top-level JSON value is refused. The previous file is
    not read and is not appended to.
    """
    if type(payload) is not dict:
        raise TypeError("flash payload must be one JSON object")
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    _atomic_write(workspace, _FLASH_FILE, body + "\n", ".flash-")


def clear_flash(workspace: Path) -> None:
    """Remove ``flash.json`` when it stays inside the workspace.

    A missing file returns. A symlink that leaves the workspace is refused and
    is not followed.
    """
    root = Path(workspace).resolve()
    path = root / "ui" / _FLASH_FILE
    if not path.exists() and not path.is_symlink():
        return
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise PathNotAllowedError(str(resolved))
    path.unlink()


def consume_flash(workspace: Path) -> dict[str, Any] | None:
    """Read the flash object and remove the file.

    ``None`` leaves the directory untouched. A symlink that leaves the workspace
    is not deleted.
    """
    payload = read_flash(workspace)
    if payload is None:
        return None
    clear_flash(workspace)
    return payload


def _read_text(workspace: Path, filename: str) -> str | None:
    root = Path(workspace).resolve()
    path = root / "ui" / filename
    try:
        escaped = (path.is_symlink() or path.exists()) and not path.resolve().is_relative_to(
            root
        )
        if escaped:
            return None
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _atomic_write(workspace: Path, filename: str, text: str, prefix: str) -> None:
    root = Path(workspace).resolve()
    ui_dir = root / "ui"
    _require_inside(ui_dir, root)
    target = ui_dir / filename
    if target.exists() or target.is_symlink():
        _require_inside(target, root)
    ui_dir.mkdir(parents=True, exist_ok=True)
    directory = _require_inside(ui_dir, root)
    if not directory.is_dir():
        raise PathNotAllowedError(str(directory))
    destination = directory / filename
    if destination.exists() or destination.is_symlink():
        _require_inside(destination, root)
    fd, tmp_name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            _require_inside(tmp_path, root)
            handle.write(text)
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
