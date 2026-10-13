"""Starlette app. Pages call the same services as the CLI."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.applications import Starlette
from starlette.datastructures import FormData
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, RedirectResponse
from starlette.routing import BaseRoute, Mount, Route
from starlette.staticfiles import StaticFiles

from excavation_lab import storage
from excavation_lab.errors import LabError
from excavation_lab.ui.copy import LANGUAGE_ENGLISH, LANGUAGE_RUSSIAN, catalog
from excavation_lab.ui.locale import resolve_locale, write_locale_file
from excavation_lab.ui.navigation import PAGE_ROUTES as _PAGE_ROUTES
from excavation_lab.ui.theme import resolve_theme, write_theme_file
from excavation_lab.ui.tokens import LAB_THEME_CSS
from excavation_lab.web.analysis_routes import analysis_context, analysis_routes
from excavation_lab.web.data_routes import data_context, data_routes
from excavation_lab.web.preferences import (
    choose_project_id,
    consume_flash,
    write_flash,
    write_project_file,
)
from excavation_lab.web.quality_routes import quality_context, quality_routes
from excavation_lab.web.routes import job_routes, video_context, video_routes
from excavation_lab.web.training_routes import training_context, training_routes

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

_ALLOWED_PATHS = frozenset(path for _page_id, path, _attr in _PAGE_ROUTES)
_EMPTY_ATTR = {
    "data": "data_empty",
    "training": "training_empty",
    "analysis": "analysis_empty",
    "quality": "quality_empty",
}
_ANALYSIS_COOKIES = (
    ("analysis_time_cookie", "lab_analysis_time"),
    ("analysis_mode_cookie", "lab_analysis_mode"),
    ("analysis_run_cookie", "lab_analysis_run"),
    ("analysis_view_cookie", "lab_analysis_view"),
)
_QUALITY_COOKIES = (
    ("quality_run_cookie", "lab_quality_run"),
    ("quality_run_a_cookie", "lab_quality_run_a"),
    ("quality_run_b_cookie", "lab_quality_run_b"),
)
_THEMES = frozenset({"dark", "light"})
_LOCALES = frozenset({"ru", "en"})
_TITLE_ATTR = {page_id: attr for page_id, _path, attr in _PAGE_ROUTES}


def theme_stylesheet(_request: Request) -> FileResponse:
    """Serve ``ui/lab_theme.css`` in place. The file is not copied here."""
    return FileResponse(LAB_THEME_CSS, media_type="text/css")


def _jinja() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html"]),
    )


def _get(page_id: str):
    async def endpoint(request: Request) -> HTMLResponse:
        return _render(request, page_id)

    endpoint.__name__ = f"get_{page_id}"
    return endpoint


async def _post_project(request: Request) -> RedirectResponse:
    form = await _fields(request)
    target = _safe_next(form.get("next"))
    chosen = form.get("project")
    if chosen:
        workspace = _workspace(request)
        try:
            projects = storage.list_projects(workspace)
        except LabError:
            projects = []
        if any(item["project_id"] == chosen for item in projects):
            try:
                write_project_file(workspace, chosen)
            except Exception:  # noqa: BLE001 — a failed preference write stays on the page
                locale_name = resolve_locale(workspace, session={})
                _stash_notice(request, catalog(locale_name).project_write_failed)
    return _redirect(target)


async def _post_theme(request: Request) -> RedirectResponse:
    form = await _fields(request)
    target = _safe_next(form.get("next"))
    theme_value = form.get("theme")
    if theme_value in _THEMES:
        workspace = _workspace(request)
        try:
            write_theme_file(workspace, theme_value)
        except Exception:  # noqa: BLE001 — a failed preference write stays on the page
            locale_name = resolve_locale(workspace, session={})
            _stash_notice(request, catalog(locale_name).theme_write_failed)
    return _redirect(target)


async def _post_locale(request: Request) -> RedirectResponse:
    form = await _fields(request)
    target = _safe_next(form.get("next"))
    locale_value = form.get("locale")
    if locale_value in _LOCALES:
        workspace = _workspace(request)
        try:
            write_locale_file(workspace, locale_value)
        except Exception:  # noqa: BLE001 — a failed preference write stays on the page
            locale_name = resolve_locale(workspace, session={})
            _stash_notice(request, catalog(locale_name).locale_write_failed)
    return _redirect(target)


def create_app(workspace: Path) -> Starlette:
    """Build the page server for one workspace."""
    templates = _jinja()
    routes: list[BaseRoute] = [
        Route(path, _get(page_id), methods=["GET"]) for page_id, path, _attr in _PAGE_ROUTES
    ]
    routes.extend(video_routes())
    routes.extend(data_routes())
    routes.extend(training_routes())
    routes.extend(analysis_routes())
    routes.extend(quality_routes())
    routes.extend(job_routes())
    routes.extend(
        [
            Route("/project", _post_project, methods=["POST"]),
            Route("/theme", _post_theme, methods=["POST"]),
            Route("/locale", _post_locale, methods=["POST"]),
            Route("/static/lab_theme.css", theme_stylesheet, methods=["GET"]),
            Mount("/static", app=StaticFiles(directory=STATIC_DIR), name="static"),
        ]
    )
    application = Starlette(routes=routes)
    application.state.workspace = Path(workspace)
    application.state.templates = templates
    application.state.pending_notice = None
    return application


def _render(request: Request, page_id: str) -> HTMLResponse:
    flash = _take_flash(request)
    context = _context(request, page_id, flash)
    selected_video = None
    if page_id == "events":
        name = "events.html"
    elif page_id == "video":
        extra = video_context(request, context, flash)
        selected_video = extra.get("selected_video_id")
        context.update(extra)
        name = "video.html"
    elif page_id == "data":
        extra = data_context(request, context, flash)
        selected_video = extra.get("selected_video_id")
        context.update(extra)
        name = "data.html"
    elif page_id == "training":
        extra = training_context(request, context, flash)
        context.update(extra)
        name = "training.html"
    elif page_id == "analysis":
        extra = analysis_context(request, context, flash)
        selected_video = extra.get("selected_video_id")
        context.update(extra)
        name = "analysis.html"
    elif page_id == "quality":
        extra = quality_context(request, context, flash)
        context.update(extra)
        name = "quality.html"
    else:
        name = "page.html"
    templates: Environment = request.app.state.templates
    markup = templates.get_template(name).render(**context)
    response = HTMLResponse(markup)
    if isinstance(selected_video, str) and selected_video and not _bad_cookie(selected_video):
        response.set_cookie(
            "lab_video",
            selected_video,
            path="/",
            httponly=True,
            samesite="lax",
        )
    for key, cookie_name in _ANALYSIS_COOKIES:
        _set_cookie(response, cookie_name, context.get(key))
    for key, cookie_name in _QUALITY_COOKIES:
        _set_cookie(response, cookie_name, context.get(key))
    return response


def _context(request: Request, page_id: str, flash: dict[str, str]) -> dict[str, object]:
    workspace = _workspace(request)
    theme_name = resolve_theme(workspace, session={})
    locale_name = resolve_locale(workspace, session={})
    book = catalog(locale_name)
    projects, project_error = _projects(workspace)
    project_id = _selected_project(workspace, projects)
    current_path = request.url.path if request.url.path in _ALLOWED_PATHS else "/"
    links = [
        {
            "title": getattr(book, attr),
            "path": path,
            "current": current == page_id,
        }
        for current, path, attr in _PAGE_ROUTES
    ]
    empty_attr = _EMPTY_ATTR.get(page_id)
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
        "project_error": project_error,
        "nav_links": links,
        "current_path": current_path,
        "theme_label": book.theme_label,
        "theme_light": book.theme_light,
        "theme_dark": book.theme_dark,
        "theme_display_only": book.theme_display_only,
        "locale_label": book.locale_label,
        "locale_caption": book.locale_caption,
        "language_russian": LANGUAGE_RUSSIAN,
        "language_english": LANGUAGE_ENGLISH,
        "notice": (
            None
            if page_id in {"video", "data", "training", "analysis", "quality"}
            else _sidebar_notice(flash)
        ),
        "page_title": getattr(book, _TITLE_ATTR[page_id]),
        "lead": book.lead(page_id),
        "empty_state": getattr(book, empty_attr) if empty_attr else "",
        "events_unavailable": book.events_unavailable,
        "events_unavailable_caption": book.events_unavailable_caption,
        "events_v02_caption": book.events_v02_caption,
        "event_indicator_labels": book.event_indicator_labels,
        "not_calculated": book.not_calculated,
    }


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


def _stash_notice(request: Request, text: str) -> None:
    try:
        write_flash(_workspace(request), {"error": text})
    except Exception:  # noqa: BLE001 — the page still has to show the sentence
        request.app.state.pending_notice = text


def _take_flash(request: Request) -> dict[str, str]:
    found: dict[str, str] = {}
    try:
        payload = consume_flash(_workspace(request))
    except (LabError, OSError):
        payload = None
    if isinstance(payload, dict):
        for key in ("error", "notice", "video_id"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                found[key] = value
    pending = getattr(request.app.state, "pending_notice", None)
    request.app.state.pending_notice = None
    if "error" not in found and isinstance(pending, str) and pending:
        found["error"] = pending
    return found


def _sidebar_notice(flash: dict[str, str]) -> str | None:
    for key in ("error", "notice"):
        value = flash.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _bad_cookie(value: str) -> bool:
    return any(char in value for char in "\r\n;")


def _set_cookie(response: HTMLResponse, name: str, value: object) -> None:
    """Remember one analysis choice. A measured ``0`` is a real value."""
    if not isinstance(value, str) or value == "" or _bad_cookie(value):
        return
    response.set_cookie(name, value, path="/", httponly=True, samesite="lax")


def _workspace(request: Request) -> Path:
    return Path(request.app.state.workspace)


def _safe_next(value: str | None) -> str:
    if value in _ALLOWED_PATHS:
        return value
    return "/"


def _redirect(target: str) -> RedirectResponse:
    return RedirectResponse(target, status_code=303)


async def _fields(request: Request) -> dict[str, str]:
    form: FormData = await request.form()
    found: dict[str, str] = {}
    for key in ("theme", "locale", "project", "next"):
        value = form.get(key)
        if isinstance(value, str):
            found[key] = value
    return found
