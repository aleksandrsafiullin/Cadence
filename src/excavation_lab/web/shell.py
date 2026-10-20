"""Shared page shell for routes that are not top-navigation entries."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path

from starlette.requests import Request

from excavation_lab import storage
from excavation_lab.datasets.catalog import SEED_CATALOG_ID, list_catalogs, read_catalog
from excavation_lab.errors import LabError
from excavation_lab.ui.copy import LANGUAGE_ENGLISH, LANGUAGE_RUSSIAN, catalog
from excavation_lab.ui.locale import resolve_locale
from excavation_lab.ui.navigation import PAGE_ROUTES
from excavation_lab.ui.project_home import display_rows
from excavation_lab.ui.theme import resolve_theme
from excavation_lab.ui.workflow import project_stages
from excavation_lab.web.preferences import choose_project_id

_NAV_PATHS = frozenset(path for _page_id, path, _attr in PAGE_ROUTES)
# Theme, locale, and project forms may return to these. They are not nav entries.
RETURN_PATHS = _NAV_PATHS | {"/", "/label"}
_EMPTY_ATTR = {
    "data": "data_empty",
    "training": "training_empty",
    "analysis": "analysis_empty",
    "quality": "quality_empty",
}
_TITLE_ATTR = {page_id: attr for page_id, _path, attr in PAGE_ROUTES}
_QUIET_NOTICE = frozenset({"video", "data", "training", "analysis", "quality", "actions"})


def page_shell(request: Request, page_id: str, flash: dict[str, str]) -> dict[str, object]:
    """Sidebar, theme, locale, and catalog strings for one page.

    ``page_id="label"`` uses the labeler title and marks «Данные и разметка» current.
    """
    workspace = _workspace(request)
    theme_name = resolve_theme(workspace, session={})
    locale_name = resolve_locale(workspace, session={})
    book = catalog(locale_name)
    projects, project_error = _projects(workspace)
    project_id = _selected_project(workspace, projects)
    project_name = _project_name(projects, project_id)
    current_path = request.url.path if request.url.path in RETURN_PATHS else "/"
    if request.url.path == "/label":
        current_path = (
            label_return_query(list(request.query_params.multi_items())) or "/label"
        )
    # The labeler is not a nav entry. The data link stays current.
    marked = "data" if page_id == "label" else page_id
    links = [
        {
            "title": getattr(book, attr),
            "path": path,
            "current": current == marked,
        }
        for current, path, attr in PAGE_ROUTES
    ]
    empty_attr = _EMPTY_ATTR.get(page_id)
    marked_stage = "data" if page_id == "label" else page_id
    stages = [
        {
            "stage_id": stage.stage_id,
            "title": stage.title,
            "path": stage.path,
            "state": stage.state,
            "caption": stage.caption,
            "status": stage.status,
            "here": stage.stage_id == marked_stage,
        }
        for stage in project_stages(workspace, project_id, book)
    ]
    if page_id == "label":
        page_title = book.label_page_title
        lead = ""
    else:
        page_title = getattr(book, _TITLE_ATTR[page_id])
        lead = book.lead(page_id)
    return {
        "lang": locale_name,
        "theme": theme_name,
        "product_title": book.product_title,
        "skip_label": "К содержанию" if locale_name == "ru" else "Skip to content",
        "kicker_project": book.kicker_project,
        "kicker_workflow": book.kicker_workflow,
        "kicker_appearance": book.kicker_appearance,
        "project_label": book.project_label,
        "project_help": book.project_help,
        "no_project": book.no_project,
        "projects": projects,
        "project_id": project_id,
        "project_name": project_name,
        "back_to_projects": book.back_to_projects,
        "project_error": project_error,
        "nav_links": links,
        "stages": stages,
        "current_path": current_path,
        "theme_label": book.theme_label,
        "theme_light": book.theme_light,
        "theme_dark": book.theme_dark,
        "theme_display_only": book.theme_display_only,
        "locale_label": book.locale_label,
        "locale_caption": book.locale_caption,
        "language_russian": LANGUAGE_RUSSIAN,
        "language_english": LANGUAGE_ENGLISH,
        "notice": None if page_id in _QUIET_NOTICE else _sidebar_notice(flash),
        "page_title": page_title,
        "lead": lead,
        "empty_state": getattr(book, empty_attr) if empty_attr else "",
        "events_unavailable": book.events_unavailable,
        "events_unavailable_caption": book.events_unavailable_caption,
        "events_v02_caption": book.events_v02_caption,
        "event_indicator_labels": book.event_indicator_labels,
        "not_calculated": book.not_calculated,
        "catalogs": _catalog_ids(workspace),
        "catalog_classes": _custom_classes(workspace, project_id),
        "create_project_heading": book.create_project_heading,
        "create_project_name": book.create_project_name,
        "create_project_button": book.create_project_button,
        "catalog_existing": book.catalog_existing,
        "catalog_new": book.catalog_new,
        "catalog_id_label": book.catalog_id_label,
        "catalog_copy_label": book.catalog_copy_label,
        "catalog_copy_none": book.catalog_copy_none,
        "catalog_none": book.catalog_none,
        "add_class_row": book.add_class_row,
        "class_name_column": book.class_name_column,
        "class_title_column": book.class_title_column,
        "class_box_column": book.class_box_column,
        "catalog_classes_heading": book.catalog_classes_heading,
        "rename_project_label": book.rename_project_label,
        "rename_project_button": book.rename_project_button,
        "delete_project_button": book.delete_project_button,
        "delete_project_confirm": book.delete_project_confirm,
    }


def home_context(request: Request) -> dict[str, object]:
    """Project list. No workflow nav and no stepper."""
    workspace = _workspace(request)
    theme_name = resolve_theme(workspace, session={})
    locale_name = resolve_locale(workspace, session={})
    book = catalog(locale_name)
    projects, _project_error = _projects(workspace)
    project_id = _selected_project(workspace, projects) or ""
    raw_names = {
        item["project_id"]: item["name"] or ""
        for item in storage.list_project_summaries(workspace)
    }
    rows: list[dict[str, str]] = []
    for item in display_rows(workspace, book):
        row = dict(item)
        row["rename_value"] = raw_names.get(item["project_id"], "")
        rows.append(row)
    return {
        "lang": locale_name,
        "theme": theme_name,
        "product_title": book.product_title,
        "skip_label": "К содержанию" if locale_name == "ru" else "Skip to content",
        "kicker_appearance": book.kicker_appearance,
        "current_path": "/",
        "project_id": project_id,
        "rows": rows,
        "home_title": book.home_title,
        "home_lead": book.home_lead,
        "home_empty": book.home_empty,
        "home_current": book.home_current,
        "open_project": book.open_project,
        "home_name_column": book.home_name_column,
        "home_id_column": book.home_id_column,
        "home_catalog_column": book.home_catalog_column,
        "home_created_column": book.home_created_column,
        "home_videos_column": book.home_videos_column,
        "home_episodes_column": book.home_episodes_column,
        "home_datasets_column": book.home_datasets_column,
        "home_models_column": book.home_models_column,
        "theme_label": book.theme_label,
        "theme_light": book.theme_light,
        "theme_dark": book.theme_dark,
        "theme_display_only": book.theme_display_only,
        "locale_label": book.locale_label,
        "locale_caption": book.locale_caption,
        "language_russian": LANGUAGE_RUSSIAN,
        "language_english": LANGUAGE_ENGLISH,
        "catalogs": _catalog_ids(workspace),
        "create_project_heading": book.create_project_heading,
        "create_project_name": book.create_project_name,
        "create_project_name_hint": book.create_project_name_hint,
        "create_project_button": book.create_project_button,
        "project_manager_close": book.project_manager_close,
        "catalog_existing": book.catalog_existing,
        "catalog_new": book.catalog_new,
        "catalog_id_label": book.catalog_id_label,
        "catalog_copy_label": book.catalog_copy_label,
        "catalog_copy_none": book.catalog_copy_none,
        "catalog_none": book.catalog_none,
        "add_class_row": book.add_class_row,
        "class_name_column": book.class_name_column,
        "class_title_column": book.class_title_column,
        "class_box_column": book.class_box_column,
        "catalog_classes_heading": book.catalog_classes_heading,
        "rename_project_label": book.rename_project_label,
        "rename_project_button": book.rename_project_button,
        "delete_project_button": book.delete_project_button,
        "delete_project_confirm": book.delete_project_confirm,
    }


def label_return_target(value: str | None) -> str | None:
    """``/label?video=`` plus an optional all-digit ``frame``, or nothing.

    Any other query key, a repeated key, or an unsafe video token is refused.
    The result is rebuilt. The raw string is not copied through.
    """
    if not isinstance(value, str) or not value.startswith("/label?"):
        return None
    if any(char in value for char in "\r\n\\#"):
        return None
    query = value.removeprefix("/label?")
    if not query or "?" in query:
        return None
    items: list[tuple[str, str]] = []
    for part in query.split("&"):
        if part.count("=") != 1:
            return None
        key, raw = part.split("=", 1)
        if not key or "%" in raw or "+" in raw:
            return None
        items.append((key, raw))
    return label_return_query(items)


def label_return_query(items: Sequence[tuple[str, str]]) -> str | None:
    """Canonical label return path from already-split query pairs."""
    if not items:
        return None
    seen: dict[str, str] = {}
    for key, raw in items:
        if key not in {"video", "frame"} or key in seen:
            return None
        seen[key] = raw
    video = seen.get("video")
    if video is None or not _safe_video_token(video):
        return None
    frame = seen.get("frame")
    if "frame" in seen and (frame is None or not frame.isascii() or not frame.isdigit()):
        return None
    if frame is None:
        return "/label?video=" + video
    return f"/label?video={video}&frame={frame}"


def _safe_video_token(value: str) -> bool:
    if not value or value in {".", ".."} or ".." in value:
        return False
    if any(ord(char) < 32 or char in "/\\;\x00" for char in value):
        return False
    return all(char.isascii() and (char.isalnum() or char in "-._~") for char in value)


def _workspace(request: Request) -> Path:
    return Path(request.app.state.workspace)


def _projects(workspace: Path) -> tuple[list[dict[str, str]], str | None]:
    try:
        found = storage.list_projects(workspace)
    except LabError as exc:
        return [], exc.message
    return [
        {"project_id": item["project_id"], "name": item["name"]} for item in found
    ], None


def _selected_project(workspace: Path, projects: list[dict[str, str]]) -> str | None:
    return choose_project_id(workspace, projects)


def _project_name(projects: list[dict[str, str]], project_id: str | None) -> str:
    if project_id is None:
        return ""
    for item in projects:
        if item["project_id"] == project_id:
            return item["name"]
    return ""


def _sidebar_notice(flash: dict[str, str]) -> str | None:
    for key in ("error", "notice"):
        value = flash.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _catalog_ids(workspace: Path) -> list[str]:
    try:
        return list_catalogs(workspace)
    except LabError:
        return []


def _custom_classes(workspace: Path, project_id: str | None) -> list[dict[str, str]]:
    catalog_id = _stored_catalog_id(workspace, project_id)
    if catalog_id is None or catalog_id == SEED_CATALOG_ID:
        return []
    try:
        stored = read_catalog(workspace, catalog_id)
    except LabError:
        return []
    return [
        {"name": item.name, "title": item.title, "box": item.box}
        for item in stored.classes
    ]


def _stored_catalog_id(workspace: Path, project_id: str | None) -> str | None:
    if not isinstance(project_id, str) or not project_id:
        return None
    try:
        db_path = storage.project_sqlite(workspace, project_id)
        with storage.connect(db_path) as connection:
            row = connection.execute(
                "SELECT catalog_id FROM projects LIMIT 1"
            ).fetchone()
    except (LabError, OSError, sqlite3.Error):
        return None
    if row is None:
        return None
    value = row["catalog_id"]
    if isinstance(value, str) and value.strip():
        return value
    return None
