"""Starlette app. Pages call the same services as the CLI."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from starlette.applications import Starlette
from starlette.datastructures import FormData
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from starlette.routing import BaseRoute, Mount, Route
from starlette.staticfiles import StaticFiles

from excavation_lab import storage
from excavation_lab.datasets.catalog import (
    add_class,
    copy_catalog,
    create_catalog,
)
from excavation_lab.errors import LabError
from excavation_lab.ui.copy import catalog
from excavation_lab.ui.locale import resolve_locale, write_locale_file
from excavation_lab.ui.navigation import PAGE_ROUTES as _PAGE_ROUTES
from excavation_lab.ui.theme import write_theme_file
from excavation_lab.ui.tokens import LAB_THEME_CSS
from excavation_lab.web.actions_routes import actions_context, actions_routes
from excavation_lab.web.analysis_routes import analysis_context, analysis_routes
from excavation_lab.web.data_routes import data_context, data_routes
from excavation_lab.web.label_routes import label_context, label_routes
from excavation_lab.web.preferences import (
    clear_project_file,
    consume_flash,
    read_project_file,
    write_flash,
    write_project_file,
)
from excavation_lab.web.quality_routes import quality_context, quality_routes
from excavation_lab.web.routes import job_routes, video_context, video_routes
from excavation_lab.web.shell import (
    RETURN_PATHS,
    home_context,
    label_return_target,
    page_shell,
)
from excavation_lab.web.training_routes import training_context, training_routes

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

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


def theme_stylesheet(_request: Request) -> FileResponse:
    """Serve ``ui/lab_theme.css`` in place. The file is not copied here."""
    return FileResponse(LAB_THEME_CSS, media_type="text/css")


def _jinja() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html"]),
    )


def _get(page_id: str):
    async def endpoint(request: Request) -> Response:
        return _render(request, page_id)

    endpoint.__name__ = f"get_{page_id}"
    return endpoint


async def get_home(request: Request) -> HTMLResponse:
    """Project list. Not a navigation entry and not the video page."""
    flash = _take_flash(request)
    context = home_context(request)
    error = flash.get("error")
    context["home_error"] = error if isinstance(error, str) else ""
    draft = flash.get("draft_name")
    context["create_draft"] = draft if isinstance(draft, str) else ""
    templates: Environment = request.app.state.templates
    markup = templates.get_template("home.html").render(**context)
    return HTMLResponse(markup)


async def _post_project(request: Request) -> RedirectResponse:
    form = await _fields(request)
    chosen = form.get("project", "").strip()
    workspace = _workspace(request)
    book = catalog(resolve_locale(workspace, session={}))
    if not chosen:
        _stash_notice(request, book.project_name_required)
        return _redirect("/")
    try:
        storage.project_sqlite(workspace, chosen)
    except LabError as exc:
        _stash_notice(request, exc.message)
        return _redirect("/")
    try:
        write_project_file(workspace, chosen)
    except Exception:  # noqa: BLE001 — a failed preference write stays on home
        _stash_notice(request, book.project_write_failed)
        return _redirect("/")
    return _redirect("/video")


async def _post_project_create(request: Request) -> RedirectResponse:
    form = await request.form()
    workspace = _workspace(request)
    book = catalog(resolve_locale(workspace, session={}))
    draft = _token(form.get("name"))
    try:
        name = _posted_project_name(form)
        # A blank choice is not the seed. create_project runs only with an id.
        catalog_id = _posted_catalog(workspace, form)
        record = storage.create_project(name, workspace, catalog_id=catalog_id)
    except _ChoiceError as exc:
        _stash_notice(request, getattr(book, exc.key), draft_name=draft)
        return _redirect("/")
    except LabError as exc:
        _stash_notice(request, exc.message, draft_name=draft)
        return _redirect("/")
    try:
        write_project_file(workspace, record["project_id"])
    except Exception:  # noqa: BLE001 — a failed preference write stays on home
        _stash_notice(request, book.project_write_failed)
        return _redirect("/")
    return _redirect("/video")


async def _post_project_rename(request: Request) -> RedirectResponse:
    form = await request.form()
    workspace = _workspace(request)
    book = catalog(resolve_locale(workspace, session={}))
    try:
        name = _posted_project_name(form)
        storage.rename_project(_token(form.get("project")), name, workspace)
    except _ChoiceError as exc:
        _stash_notice(request, getattr(book, exc.key))
        return _redirect("/")
    except LabError as exc:
        _stash_notice(request, exc.message)
        return _redirect("/")
    return _redirect("/")


async def _post_project_delete(request: Request) -> RedirectResponse:
    form = await request.form()
    workspace = _workspace(request)
    book = catalog(resolve_locale(workspace, session={}))
    if _token(form.get("confirm")) != "yes":
        _stash_notice(request, book.delete_confirm_required)
        return _redirect("/")
    project_id = _token(form.get("project"))
    try:
        storage.delete_project(project_id, workspace)
    except LabError as exc:
        _stash_notice(request, exc.message)
        return _redirect("/")
    try:
        if read_project_file(workspace) == project_id:
            clear_project_file(workspace)
    except Exception:  # noqa: BLE001 — a failed preference write stays on home
        _stash_notice(request, book.project_write_failed)
    return _redirect("/")


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
        Route("/", get_home, methods=["GET"]),
    ]
    routes.extend(
        Route(path, _get(page_id), methods=["GET"]) for page_id, path, _attr in _PAGE_ROUTES
    )
    routes.extend(video_routes())
    routes.extend(data_routes())
    routes.extend(label_routes())
    routes.extend(training_routes())
    routes.extend(analysis_routes())
    routes.extend(quality_routes())
    routes.extend(actions_routes())
    routes.extend(job_routes())
    routes.extend(
        [
            Route("/project", _post_project, methods=["POST"]),
            Route("/project/create", _post_project_create, methods=["POST"]),
            Route("/project/rename", _post_project_rename, methods=["POST"]),
            Route("/project/delete", _post_project_delete, methods=["POST"]),
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


def _render(request: Request, page_id: str) -> Response:
    if not _project_entered(request):
        return RedirectResponse("/", status_code=303)
    flash = _take_flash(request)
    context = page_shell(request, page_id, flash)
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
    elif page_id == "actions":
        extra = actions_context(request, context, flash)
        selected_video = extra.get("selected_video_id")
        context.update(extra)
        name = "actions.html"
    elif page_id == "label":
        extra = label_context(request, context)
        selected_video = extra.get("selected_video_id")
        context.update(extra)
        name = "label.html"
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


def _stash_notice(request: Request, text: str, *, draft_name: str = "") -> None:
    payload: dict[str, str] = {"error": text}
    if draft_name:
        payload["draft_name"] = draft_name
    try:
        write_flash(_workspace(request), payload)
    except Exception:  # noqa: BLE001 — the page still has to show the sentence
        request.app.state.pending_notice = text


def _take_flash(request: Request) -> dict[str, str]:
    found: dict[str, str] = {}
    try:
        payload = consume_flash(_workspace(request))
    except (LabError, OSError):
        payload = None
    if isinstance(payload, dict):
        for key in ("error", "notice", "video_id", "draft_name"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                found[key] = value
    pending = getattr(request.app.state, "pending_notice", None)
    request.app.state.pending_notice = None
    if "error" not in found and isinstance(pending, str) and pending:
        found["error"] = pending
    return found


def _bad_cookie(value: str) -> bool:
    return any(char in value for char in "\r\n;")


def _set_cookie(response: HTMLResponse, name: str, value: object) -> None:
    """Remember one analysis choice. A measured ``0`` is a real value."""
    if not isinstance(value, str) or value == "" or _bad_cookie(value):
        return
    response.set_cookie(name, value, path="/", httponly=True, samesite="lax")


def _workspace(request: Request) -> Path:
    return Path(request.app.state.workspace)


def _project_entered(request: Request) -> bool:
    """True when ``project.txt`` names a listed project. Does not write."""
    workspace = _workspace(request)
    try:
        projects = storage.list_projects(workspace)
    except LabError:
        return False
    stored = read_project_file(workspace)
    return any(item["project_id"] == stored for item in projects)


def _safe_next(value: str | None) -> str:
    if value in RETURN_PATHS:
        return value
    target = label_return_target(value)
    if target is not None:
        return target
    return "/"


def _redirect(target: str) -> RedirectResponse:
    return RedirectResponse(target, status_code=303)


class _ChoiceError(Exception):
    def __init__(self, key: str) -> None:
        self.key = key


def _text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _token(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _posted_project_name(form: FormData) -> str:
    name = _token(form.get("name"))
    if not name:
        raise _ChoiceError("project_name_required")
    return name


def _posted_catalog(workspace: Path, form: FormData) -> str:
    mode = form.get("catalog_mode")
    if mode == "existing":
        catalog_id = _token(form.get("catalog_id"))
        if not catalog_id:
            raise _ChoiceError("catalog_required")
        return catalog_id
    if mode != "new":
        raise _ChoiceError("catalog_required")
    catalog_id = _token(form.get("new_catalog_id"))
    if not catalog_id:
        raise _ChoiceError("catalog_id_required")
    rows = _posted_rows(form)
    source = _token(form.get("copy_from"))
    if source:
        copy_catalog(workspace, source, catalog_id)
        for class_name, title, box in rows:
            add_class(workspace, catalog_id, class_name, title, box)
        return catalog_id
    if not rows:
        raise _ChoiceError("catalog_required")
    create_catalog(workspace, catalog_id, rows)
    return catalog_id


def _posted_rows(form: FormData) -> list[tuple[str, str, str]]:
    names = _row_text(form, "class_name")
    titles = _row_text(form, "class_title")
    boxes = _row_text(form, "class_box")
    if names is None or titles is None or boxes is None:
        raise _ChoiceError("class_row_incomplete")
    if not (len(names) == len(titles) == len(boxes)):
        raise _ChoiceError("class_row_incomplete")
    rows: list[tuple[str, str, str]] = []
    for name, title, box in zip(names, titles, boxes, strict=True):
        blank = name.strip() == "" and title.strip() == "" and box.strip() == ""
        if blank:
            continue
        if name.strip() == "" or title.strip() == "" or box.strip() == "":
            raise _ChoiceError("class_row_incomplete")
        rows.append((name.strip(), title, box))
    return rows


def _row_text(form: FormData, key: str) -> list[str] | None:
    found: list[str] = []
    for item in form.getlist(key):
        if not isinstance(item, str):
            return None
        found.append(item)
    return found


async def _fields(request: Request) -> dict[str, str]:
    form: FormData = await request.form()
    found: dict[str, str] = {}
    for key in ("theme", "locale", "project", "next"):
        value = form.get(key)
        if isinstance(value, str):
            found[key] = value
    return found
