"""Dark-default theme: stylesheets, ``data-theme``, and a workspace preference."""

from __future__ import annotations

import os
import tempfile
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Literal

from excavation_lab.errors import PathNotAllowedError
from excavation_lab.ui import copy as ui_copy

ThemeName = Literal["dark", "light"]

_LABEL_TO_THEME: dict[str, ThemeName] = {
    ui_copy.THEME_LIGHT: "light",
    ui_copy.THEME_DARK: "dark",
}
_WRITE_ERROR_KEY = "lab_theme_write_error"


def preference_path(workspace: Path) -> Path:
    """``<workspace>/ui/theme.txt``. Does not create the file."""
    return Path(workspace) / "ui" / "theme.txt"


def read_theme_file(workspace: Path) -> ThemeName | None:
    """Return ``dark`` or ``light`` from the preference file.

    A missing file, an invalid line, or a symlink that leaves the workspace
    is ``None``. This does not create directories and does not write.
    """
    root = Path(workspace).resolve()
    path = root / "ui" / "theme.txt"
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
    if value == "dark":
        return "dark"
    if value == "light":
        return "light"
    return None


def resolve_theme(
    workspace: Path,
    session: MutableMapping[str, Any] | None = None,
) -> ThemeName:
    """Session ``lab_theme``, then the preference file, then ``dark``.

    A valid file value is stored in the session. An invalid file is ``dark``
    and is not stored.
    """
    state: MutableMapping[str, Any] = {} if session is None else session
    current = state.get("lab_theme")
    if current == "dark":
        return "dark"
    if current == "light":
        return "light"
    stored = read_theme_file(workspace)
    if stored is None:
        return "dark"
    state["lab_theme"] = stored
    return stored


def write_theme_file(workspace: Path, theme: str) -> None:
    """Atomically write one ``dark`` or ``light`` line.

    The temp file is created in ``ui/`` and ``os.replace``d onto ``theme.txt``.
    After resolve, that file must stay inside the resolved workspace. A symlink
    that leaves the root is refused.
    """
    name = _theme_name(theme)
    root = Path(workspace).resolve()
    ui_dir = root / "ui"
    _require_inside(ui_dir, root)
    target = ui_dir / "theme.txt"
    if target.exists() or target.is_symlink():
        _require_inside(target, root)
    ui_dir.mkdir(parents=True, exist_ok=True)
    directory = _require_inside(ui_dir, root)
    if not directory.is_dir():
        raise PathNotAllowedError(str(directory))
    destination = directory / "theme.txt"
    if destination.exists() or destination.is_symlink():
        _require_inside(destination, root)
    fd, tmp_name = tempfile.mkstemp(prefix=".theme-", suffix=".tmp", dir=directory)
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


def commit_theme_choice(
    workspace: Path,
    label: object,
    session: MutableMapping[str, Any],
) -> None:
    """Map a Light/Dark label into the session and write ``theme.txt``.

    A failed write stores the previous valid theme, or the file, or ``dark``,
    and records ``THEME_WRITE_FAILED``. It does not raise.
    """
    if not isinstance(label, str):
        return
    theme = _LABEL_TO_THEME.get(label)
    if theme is None:
        return
    previous = session.get("lab_theme")
    try:
        write_theme_file(workspace, theme)
    except Exception:  # noqa: BLE001 — a failed preference write stays on the page
        kept = previous if previous == "dark" or previous == "light" else None
        if kept is None:
            stored = read_theme_file(workspace)
            kept = stored if stored is not None else "dark"
        session["lab_theme"] = kept
        session["lab_appearance"] = _label_for(kept)
        session[_WRITE_ERROR_KEY] = ui_copy.THEME_WRITE_FAILED
        return
    session["lab_theme"] = theme
    if _WRITE_ERROR_KEY in session:
        del session[_WRITE_ERROR_KEY]


# Same statements for dark and light. The theme name is not interpolated here.
# href="" is the default page. Other hrefs are the url_path segment.
# Streamlit writes href after the link node exists, so one frame is not always
# enough. The observer re-marks when that attribute arrives. It does not watch
# aria-current, or the write would loop.
_MARK_CURRENT_PAGE: tuple[str, ...] = (
    "(function () {",
    "  function markCurrent(again) {",
    '    var links = document.querySelectorAll(\'[data-testid="stPageLink-NavLink"]\');',
    "    if (!links.length) {",
    "      if (!again) {",
    "        requestAnimationFrame(function () {",
    "          markCurrent(true);",
    "        });",
    "      }",
    "      return;",
    "    }",
    '    var segment = location.pathname.split("/").pop();',
    "    links.forEach(function (link) {",
    '      var href = link.getAttribute("href");',
    '      var current = href === "" ? segment === "" : segment === href;',
    "      if (current) {",
    '        link.setAttribute("aria-current", "page");',
    "      } else {",
    '        link.removeAttribute("aria-current");',
    "      }",
    "    });",
    "  }",
    "  markCurrent(false);",
    "  var root = document.documentElement;",
    "  if (root.labNavMark) {",
    "    root.labNavMark.disconnect();",
    "  }",
    "  var observer = new MutationObserver(function () {",
    "    markCurrent(true);",
    "  });",
    "  observer.observe(document.body, {",
    "    childList: true,",
    "    subtree: true,",
    "    attributes: true,",
    '    attributeFilter: ["href"],',
    "  });",
    "  root.labNavMark = observer;",
    "})();",
)


def theme_script_lines(theme: str) -> list[str]:
    """JavaScript statements for the single theme script.

    The first line sets ``data-theme`` from a dark/light whitelist.
    The rest marks one sidebar link current from ``getAttribute("href")``
    and the last segment of ``location.pathname``.
    """
    name = _theme_name(theme)
    return [
        'document.documentElement.setAttribute("data-theme", "' + name + '");',
        *_MARK_CURRENT_PAGE,
    ]


def _theme_name(theme: object) -> ThemeName:
    if theme == "dark":
        return "dark"
    if theme == "light":
        return "light"
    raise ValueError(f"unknown theme: {theme!r}")


def _label_for(theme: ThemeName) -> str:
    if theme == "light":
        return ui_copy.THEME_LIGHT
    return ui_copy.THEME_DARK


def _require_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise PathNotAllowedError(str(resolved))
    return resolved
