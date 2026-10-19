"""Video routes and job progress. Page helpers stay in ``ui``."""

from __future__ import annotations

import json
import math
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from excavation_lab import storage
from excavation_lab.errors import LabError
from excavation_lab.jobs import PROGRESS_FILENAME, run_directory, worker_job
from excavation_lab.media.probe import add_video
from excavation_lab.ui import training_page
from excavation_lab.ui.charts import loss_chart
from excavation_lab.ui.copy import Catalog, catalog
from excavation_lab.ui.locale import resolve_locale
from excavation_lab.ui.status import status_markup, status_tone
from excavation_lab.ui.theme import resolve_theme
from excavation_lab.ui.video_page import (
    allowed_directory_caption,
    assemble_player,
    job_status_line,
    register_chosen_file,
    request_baseline_run,
    video_choice_label,
)
from excavation_lab.web.choose_file import ChooseFileFailed, choose_video_path
from excavation_lab.web.preferences import choose_project_id, write_flash
from excavation_lab.web.range_files import media_file, traversal

_COOKIE = "lab_video"
_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv"})


def job_routes() -> list[Route]:
    return [
        Route("/jobs/{job_id}/progress", get_job_progress, methods=["GET"]),
    ]


def video_routes() -> list[Route]:
    return [
        Route("/video/add", post_video_add, methods=["POST"]),
        Route("/video/choose", post_video_choose, methods=["POST"]),
        Route("/video/baseline", post_video_baseline, methods=["POST"]),
        Route("/media/{project_id}/{video_id:path}", get_media, methods=["GET", "HEAD"]),
    ]


def video_context(
    request: Request,
    base: dict[str, object],
    flash: dict[str, str],
) -> dict[str, object]:
    """Template fields for ``video.html``. Re-probe runs inside ``assemble_player``."""
    workspace = _workspace(request)
    book = catalog(str(base.get("lang") or "ru"))
    project_raw = base.get("project_id")
    project_id = project_raw if isinstance(project_raw, str) and project_raw else None
    state = assemble_player(workspace, project_id, book, _wanted_video(request, flash))
    options = [
        {
            "id": item.get("video_id"),
            "label": video_choice_label(item),
            "selected": item.get("video_id") == state.selected_id,
        }
        for item in state.videos
    ]
    directory = ""
    if project_id is not None and not state.has_video and state.list_error is None:
        try:
            directory = allowed_directory_caption(workspace, project_id, book)
        except LabError as exc:
            directory = exc.message
    shown = state.shown
    raw_path = shown.get("path") if state.has_video and shown is not None else ""
    stored_path = raw_path if isinstance(raw_path, str) else ""
    return {
        "flash_error": flash.get("error"),
        "flash_notice": visible_notice(flash.get("notice"), book),
        "list_error": state.list_error,
        "has_video": state.has_video,
        "add_local_video": book.add_local_video,
        "allowed_caption": directory,
        "local_path_label": book.local_path_label,
        "local_path": stored_path,
        "choose_file_button": book.choose_file_button,
        "need_project": book.need_project,
        "video_options": options,
        "choice_label": book.video_choice,
        "media_url": (
            f"/media/{project_id}/{state.selected_id}"
            if project_id and state.selected_id and state.playable is not None
            else ""
        ),
        "play_error": state.play_error,
        "playback_failed": book.playback_failed,
        "probe_note": state.probe_note,
        "metadata_lines": state.metadata,
        "rotation_text": state.rotation,
        "offset_text": state.offset,
        "interval_start_label": book.interval_start,
        "interval_end_label": book.interval_end,
        "interval_help": book.interval_help,
        "start_value": bound_text(state.interval_start),
        "end_value": bound_text(state.interval_end),
        "selected_video_id": state.selected_id,
        "blocked": state.blocked,
        "blocked_message": book.job_already_running if state.blocked else "",
        "baseline_button": book.baseline_button,
        "baseline_caption": book.baseline_caption,
        "baseline_help": book.baseline_help,
        "runs_heading": book.runs_heading,
        "refresh_runs": book.refresh_runs,
        "runs_empty": book.runs_empty,
        "runs_no_boxes": book.runs_no_boxes,
        "run_rows": [
            status_markup(job_status_line(job, book), status_tone(job.status.value))
            for job in state.jobs
        ],
    }


def visible_notice(notice: str | None, book: Catalog) -> str | None:
    """Keep an add notice or a queued-baseline notice. Drop every other flash."""
    if not isinstance(notice, str) or not notice:
        return None
    if notice == book.video_added or notice.startswith(book.baseline_queued):
        return notice
    return None


async def post_video_add(request: Request) -> RedirectResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    form = await request.form()
    if project_id is None:
        _remember(request, {"error": book.need_project})
        return _redirect()
    raw_path = form.get("local_path")
    text = raw_path.strip() if isinstance(raw_path, str) else ""
    if not text:
        _remember(request, {"error": book.path_required})
        return _redirect()
    try:
        added = add_video(workspace, project_id, Path(text), copy=False)
    except LabError as exc:
        _remember(request, {"error": exc.message})
        return _redirect()
    video_id = str(added["video_id"])
    _remember(request, {"notice": book.video_added, "video_id": video_id})
    return _redirect(video_id)


async def post_video_choose(request: Request) -> RedirectResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    if project_id is None:
        _remember(request, {"error": book.need_project})
        return _redirect()
    try:
        chosen = choose_video_path(prompt=book.choose_file_prompt)
    except ChooseFileFailed:
        _remember(request, {"error": book.choose_failed})
        return _redirect()
    if chosen is None:
        return _redirect()
    if chosen.suffix.lower() not in _VIDEO_SUFFIXES:
        _remember(request, {"error": book.upload_bad_name})
        return _redirect()
    try:
        video_id = register_chosen_file(workspace, project_id, chosen)
    except LabError as exc:
        _remember(request, {"error": exc.message})
        return _redirect()
    _remember(request, {"notice": book.video_added, "video_id": video_id})
    return _redirect(video_id)


async def post_video_baseline(request: Request) -> RedirectResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    form = await request.form()
    video_raw = form.get("video_id")
    video_id = video_raw if isinstance(video_raw, str) else ""
    if project_id is None or not video_id:
        _remember(request, {"error": book.need_project})
        return _redirect()
    outcome = request_baseline_run(
        workspace,
        project_id,
        video_id,
        _number(form.get("start_s")),
        _number(form.get("end_s")),
        book,
    )
    payload: dict[str, str] = {"video_id": video_id}
    if outcome.error:
        payload["error"] = outcome.error
    elif outcome.job_id:
        payload["notice"] = f"{book.baseline_queued} {outcome.job_id}."
    _remember(request, payload)
    return _redirect(video_id)


async def get_media(request: Request) -> Response:
    """Bytes for one registered id. A query string is not a filesystem path."""
    project_id = str(request.path_params.get("project_id", ""))
    video_id = str(request.path_params.get("video_id", ""))
    raw_path = str(request.scope.get("path", ""))
    if traversal(raw_path, project_id, video_id):
        return Response(status_code=404)
    try:
        return media_file(_workspace(request), project_id, video_id)
    except FileNotFoundError:
        return Response(status_code=404)


def _project_id(workspace: Path) -> str | None:
    try:
        projects = storage.list_projects(workspace)
    except LabError:
        return None
    rows = [{"project_id": item["project_id"], "name": item["name"]} for item in projects]
    return choose_project_id(workspace, rows)


def _book(workspace: Path) -> Catalog:
    return catalog(resolve_locale(workspace, session={}))


def _workspace(request: Request) -> Path:
    return Path(request.app.state.workspace)


def _wanted_video(request: Request, flash: dict[str, str]) -> str | None:
    flashed = flash.get("video_id")
    if flashed:
        return flashed
    query = request.query_params.get("video")
    if isinstance(query, str) and query and not _bad_token(query):
        return query
    cookie = request.cookies.get(_COOKIE)
    if isinstance(cookie, str) and cookie and not _bad_token(cookie):
        return cookie
    return None


def _remember(request: Request, payload: dict[str, str]) -> None:
    try:
        write_flash(_workspace(request), payload)
    except Exception:  # noqa: BLE001 — the page still has to show the sentence
        text = payload.get("error")
        if isinstance(text, str) and text:
            request.app.state.pending_notice = text


def _redirect(video_id: str | None = None) -> RedirectResponse:
    response = RedirectResponse("/", status_code=303)
    if isinstance(video_id, str) and video_id and not _bad_token(video_id):
        response.set_cookie(_COOKIE, video_id, path="/", httponly=True, samesite="lax")
    return response


def _number(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def bound_text(value: float | None) -> str | None:
    """Input value. ``None`` stays empty; ``0`` stays ``0``."""
    if value is None:
        return None
    return format(float(value), "g")


def _bad_token(value: str) -> bool:
    return any(char in value for char in "\r\n;")


async def get_job_progress(request: Request) -> Response:
    """Job row plus ``progress.json``. A query string is not a run directory."""
    job_id = request.path_params.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        return Response(status_code=404)
    workspace = _workspace(request)
    try:
        found = training_page.reopen(workspace)
    except LabError:
        return Response(status_code=404)
    match = next((job for job in found if job.job_id == job_id), None)
    if match is None:
        return Response(status_code=404)
    try:
        owner, _stored, _record = worker_job(workspace, job_id)
        run_dir = run_directory(workspace, owner, job_id)
    except LabError:
        return Response(status_code=404)
    view = _training_log_view(workspace, owner, job_id, run_dir)
    if training_page.should_draw_chart(view, match.error_code):
        chart = loss_chart(
            view.epoch_marks,
            view.losses,
            resolve_theme(workspace, session={}),
        ).to_dict()
    else:
        chart = None
    payload = {
        "job_id": match.job_id,
        "status": match.status.value,
        "orphaned": match.orphaned,
        "partial": match.partial,
        "analyzed_end_s": _positive_seconds(match.analyzed_end_s),
        "error_code": match.error_code,
        "line": job_status_line(match, _book(workspace)),
        "progress": _read_progress_file(run_dir),
        "chart": chart,
    }
    return JSONResponse(payload)


def _positive_seconds(value: object) -> float | None:
    """A real positive row value. Zero and a missing stamp stay unknown."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _training_log_view(
    workspace: Path,
    project_id: str,
    job_id: str,
    run_dir: Path,
) -> training_page.TrainingLogView:
    """The log the training page would chart. No request file means no curve."""
    request = run_dir / "train_request.json"
    try:
        if request.is_symlink() or not request.is_file():
            return training_page.TrainingLogView.absent()
    except OSError:
        return training_page.TrainingLogView.absent()
    return training_page._view_for_run(workspace, project_id, job_id, run_dir)


def _read_progress_file(run_dir: Path) -> dict[str, object] | None:
    """Parse ``PROGRESS_FILENAME``. Missing or unreadable stays ``None``."""
    path = run_dir / PROGRESS_FILENAME
    try:
        if path.is_symlink() or not path.is_file():
            return None
        if path.stat().st_size > 1_048_576:
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded
