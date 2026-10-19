"""Workspace locale: ``ru`` or ``en``, and the APPEARANCE language switch."""

from __future__ import annotations

import os
import tempfile
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Literal

from excavation_lab.errors import PathNotAllowedError
from excavation_lab.ui import copy as ui_copy

LocaleName = Literal["ru", "en"]

_PAGE_IDS: tuple[str, ...] = (
    "video",
    "data",
    "training",
    "analysis",
    "quality",
    "actions",
    "events",
)
_LABEL_TO_LOCALE: dict[str, LocaleName] = {
    ui_copy.LANGUAGE_RUSSIAN: "ru",
    ui_copy.LANGUAGE_ENGLISH: "en",
}
_WRITE_ERROR_KEY = "lab_locale_write_error"


def preference_path(workspace: Path) -> Path:
    """``<workspace>/ui/locale.txt``. Does not create the file."""
    return Path(workspace) / "ui" / "locale.txt"


def read_locale_file(workspace: Path) -> LocaleName | None:
    """Return ``ru`` or ``en`` from the preference file.

    A missing file, an invalid line, or a symlink that leaves the workspace
    is ``None``. This does not create directories and does not write.
    """
    root = Path(workspace).resolve()
    path = root / "ui" / "locale.txt"
    try:
        escaped = (path.is_symlink() or path.exists()) and not path.resolve().is_relative_to(
            root
        )
        if escaped:
            return None
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    value = text.strip()
    if value == "ru":
        return "ru"
    if value == "en":
        return "en"
    return None


def resolve_locale(
    workspace: Path,
    session: MutableMapping[str, Any] | None = None,
) -> LocaleName:
    """Session ``lab_locale``, then the preference file, then ``ru``.

    A valid file value is stored in the session. An invalid file is ``ru``
    and is not stored. A missing file does not write ``lab_locale``.
    """
    state: MutableMapping[str, Any] = {} if session is None else session
    current = state.get("lab_locale")
    if current == "ru":
        return "ru"
    if current == "en":
        return "en"
    stored = read_locale_file(workspace)
    if stored is None:
        return "ru"
    state["lab_locale"] = stored
    return stored


def write_locale_file(workspace: Path, locale: str) -> None:
    """Atomically write one ``ru`` or ``en`` line.

    The temp file is created in ``ui/`` and ``os.replace``d onto ``locale.txt``.
    After resolve, that file must stay inside the resolved workspace. A symlink
    that leaves the root is refused.
    """
    name = _locale_name(locale)
    root = Path(workspace).resolve()
    ui_dir = root / "ui"
    _require_inside(ui_dir, root)
    target = ui_dir / "locale.txt"
    if target.exists() or target.is_symlink():
        _require_inside(target, root)
    ui_dir.mkdir(parents=True, exist_ok=True)
    directory = _require_inside(ui_dir, root)
    if not directory.is_dir():
        raise PathNotAllowedError(str(directory))
    destination = directory / "locale.txt"
    if destination.exists() or destination.is_symlink():
        _require_inside(destination, root)
    fd, tmp_name = tempfile.mkstemp(prefix=".locale-", suffix=".tmp", dir=directory)
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


def commit_locale_choice(
    workspace: Path,
    label: object,
    session: MutableMapping[str, Any],
) -> None:
    """Map a Русский/English label into the session and write ``locale.txt``.

    A failed write keeps the previous valid locale, or the file, or ``ru``,
    and records the active catalog's write-failed sentence. It does not raise.
    Other session keys are left as they are.
    """
    if not isinstance(label, str):
        return
    locale = _LABEL_TO_LOCALE.get(label)
    if locale is None:
        return
    previous = session.get("lab_locale")
    try:
        write_locale_file(workspace, locale)
    except Exception:  # noqa: BLE001 — a failed preference write stays on the page
        kept = previous if previous == "ru" or previous == "en" else None
        if kept is None:
            stored = read_locale_file(workspace)
            kept = stored if stored is not None else "ru"
        session["lab_locale"] = kept
        session["lab_locale_choice"] = _label_for(kept)
        session[_WRITE_ERROR_KEY] = ui_copy.catalog(kept).locale_write_failed
        return
    session["lab_locale"] = locale
    if _WRITE_ERROR_KEY in session:
        del session[_WRITE_ERROR_KEY]


def page_id_for(text: str) -> str:
    """Map a page id or either catalog title onto one id.

    Unknown text raises ``ValueError``. No lead is invented here.
    """
    if text in _PAGE_IDS:
        return text
    for code in ("ru", "en"):
        book = ui_copy.catalog(code)
        for page_id, title in zip(_PAGE_IDS, book.page_titles, strict=True):
            if text == title:
                return page_id
    raise ValueError(f"unknown page: {text!r}")


def title_for(book: ui_copy.Catalog, page_id: str) -> str:
    """Active catalog title for one page id."""
    try:
        index = _PAGE_IDS.index(page_id)
    except ValueError:
        raise ValueError(f"unknown page: {page_id!r}") from None
    return book.page_titles[index]


def _locale_name(locale: object) -> LocaleName:
    if locale == "ru":
        return "ru"
    if locale == "en":
        return "en"
    raise ValueError(f"unknown locale: {locale!r}")


def _label_for(locale: LocaleName) -> str:
    if locale == "en":
        return ui_copy.LANGUAGE_ENGLISH
    return ui_copy.LANGUAGE_RUSSIAN


def _require_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise PathNotAllowedError(str(resolved))
    return resolved
