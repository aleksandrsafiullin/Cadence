"""Action page and JSON routes. Storage goes through ``ui.actions_page``."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from excavation_lab import storage
from excavation_lab.actions.catalog import list_actions
from excavation_lab.actions.labels import list_intervals
from excavation_lab.datasets.splits import FrozenSplit
from excavation_lab.errors import LabError
from excavation_lab.ui import actions_page
from excavation_lab.ui.actions_page import ActionWrite
from excavation_lab.ui.copy import Catalog, action_seed_titles, catalog
from excavation_lab.ui.data_page import frame_step_text, load_split_view
from excavation_lab.ui.locale import resolve_locale
from excavation_lab.ui.video_page import video_choice_label
from excavation_lab.web.data_routes import _book, _project_id, _wanted_video, _workspace
from excavation_lab.web.preferences import write_flash

_VIDEO_COOKIE = "lab_video"


def actions_routes() -> list[Route]:
    return [
        Route("/actions/class", post_action_class, methods=["POST"]),
        Route("/actions/seed", post_action_seed, methods=["POST"]),
        Route("/actions/intervals", get_action_intervals, methods=["GET"]),
        Route("/actions/intervals", post_action_intervals, methods=["POST"]),
        Route("/actions/ready", get_action_ready, methods=["GET"]),
        Route("/actions/job", get_action_job, methods=["GET"]),
        Route("/actions/train", post_action_train, methods=["POST"]),
        Route("/actions/cancel", post_action_cancel, methods=["POST"]),
        Route("/actions/infer", post_action_infer, methods=["POST"]),
        Route("/actions/export", get_action_export, methods=["GET"]),
        Route("/actions/evaluate", post_action_evaluate, methods=["POST"]),
    ]


def actions_context(
    request: Request,
    base: Mapping[str, object],
    flash: Mapping[str, str],
) -> dict[str, object]:
    """Template fields for ``actions.html``. The same video cookie as «Данные»."""
    workspace = _workspace(request)
    book = catalog(str(base.get("lang") or "ru"))
    project_raw = base.get("project_id")
    project_id = project_raw if isinstance(project_raw, str) and project_raw else None
    shared: dict[str, object] = {
        "flash_error": flash.get("error"),
        "flash_notice": flash.get("notice"),
        "book": book,
        "list_error": None,
        "show_empty": project_id is None,
        "show_catalog": False,
        "show_editor": False,
        "selected_video_id": None,
        "video_options": [],
        "catalog_rows": [],
        "coverage_rows": [],
        "interval_rows": [],
        "timeline_pieces": [],
        "block_editor": {},
        "blocks_frozen": False,
        "split_load_error": None,
        "steps": actions_page.workflow_steps(catalog_count=0, has_video=False, text=book),
        "media_url": "",
        "block_clock": False,
        "block_frame_step": None,
    }
    if project_id is None:
        return shared
    try:
        videos = storage.list_videos(workspace, project_id)
    except LabError as exc:
        shared["list_error"] = f"{exc.code}: {exc.message}"
        return shared
    rows = actions_page.catalog_rows(workspace, project_id)
    shared["catalog_rows"] = rows
    shared["show_catalog"] = True
    if not videos:
        shared["steps"] = actions_page.workflow_steps(
            catalog_count=len(rows),
            has_video=False,
            text=book,
        )
        return shared
    video_id = _wanted_video(request, flash, videos)
    video = next(item for item in videos if str(item["video_id"]) == video_id)
    shared.update(_editor(request, workspace, project_id, video, videos, book))
    shared["show_editor"] = True
    shared["selected_video_id"] = video_id
    return shared


async def post_action_class(request: Request) -> JSONResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    if not _json_content_type(request):
        return _media_type(book)
    project_id = _project_id(workspace)
    if project_id is None:
        return _stated("no_project", book.need_project)
    body = await _json_body(request)
    if body is None:
        return _stated("invalid_action_label", book.action_invalid_interval)
    return _write_response(actions_page.save_class(workspace, project_id, body))


async def post_action_seed(request: Request) -> JSONResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    if not _json_content_type(request):
        return _media_type(book)
    project_id = _project_id(workspace)
    if project_id is None:
        return _stated("no_project", book.need_project)
    body = await _json_body(request)
    if body is None:
        return _stated("invalid_action_label", book.action_invalid_interval)
    locale_name = resolve_locale(workspace, session={})
    result = actions_page.seed_catalog(
        workspace,
        project_id,
        action_seed_titles[locale_name],
    )
    return _write_response(result)


async def get_action_intervals(request: Request) -> JSONResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    if project_id is None:
        return _stated("no_project", book.need_project)
    video_id = _video_query(request)
    if video_id is None:
        return _stated("video_not_found", book.video_choice)
    try:
        payload = actions_page.intervals_document(workspace, project_id, video_id, book)
    except LabError as exc:
        return _stated(exc.code, exc.message)
    return JSONResponse({"ok": True, **payload})


async def post_action_intervals(request: Request) -> JSONResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    if not _json_content_type(request):
        return _media_type(book)
    project_id = _project_id(workspace)
    if project_id is None:
        return _stated("no_project", book.need_project)
    video_id = _video_query(request)
    if video_id is None:
        return _stated("video_not_found", book.video_choice)
    body = await _json_body(request)
    if body is None:
        return _stated("invalid_action_label", book.action_invalid_interval)
    result = actions_page.save_intervals(workspace, project_id, video_id, body, book)
    return _write_response(result)


def _editor(
    request: Request,
    workspace: Path,
    project_id: str,
    video: Mapping[str, object],
    videos: Sequence[Mapping[str, object]],
    book: Catalog,
) -> dict[str, object]:
    video_id = str(video["video_id"])
    document, load_error = load_split_view(workspace, project_id, video_id, book)
    frozen = isinstance(document, FrozenSplit)
    blocks = () if document is None else tuple(document.blocks)
    try:
        intervals = list_intervals(workspace, project_id, video_id)
    except LabError as exc:
        intervals = []
        load_error = load_error or f"{exc.code}: {exc.message}"
    classes = list_actions(workspace, project_id)
    payload = actions_page.timeline_payload(
        video,
        blocks,
        intervals,
        book,
        frozen=frozen,
        classes=classes,
    )
    duration_s = payload["duration_s"]
    offset_s = payload["offset_s"]
    clock_known = isinstance(duration_s, float) and isinstance(offset_s, float)
    fps = payload["fps"] if isinstance(payload["fps"], float) else None
    return {
        "video_options": [
            {
                "id": str(item["video_id"]),
                "label": video_choice_label(item),
                "selected": str(item["video_id"]) == video_id,
            }
            for item in videos
        ],
        "media_url": f"/media/{project_id}/{video_id}",
        "block_clock": clock_known,
        "block_frame_step": frame_step_text(fps),
        "blocks_frozen": frozen,
        "split_load_error": load_error,
        "timeline_pieces": actions_page.role_strip(blocks, video, book),
        "block_editor": payload,
        "interval_rows": payload["intervals"],
        "coverage_rows": actions_page.coverage_rows(classes, intervals, blocks),
        "steps": actions_page.workflow_steps(
            catalog_count=len(classes),
            has_video=True,
            text=book,
        ),
        "train_panel": actions_page.train_panel(
            workspace,
            project_id,
            video,
            blocks,
            book,
            selected_runs=actions_page.parse_run_ids(request.query_params.getlist("runs")),
            model_id=_query_token(request, "model"),
        ),
        "results_panel": actions_page.results_panel(
            workspace,
            project_id,
            video,
            blocks,
            intervals,
            classes,
            book,
            model_id=_query_token(request, "model"),
            inference_id=_query_token(request, "inference"),
        ),
    }


def _write_response(result: ActionWrite) -> JSONResponse:
    if result.note is not None and result.note.code:
        return JSONResponse(
            {"ok": False, "error": result.note.text, "code": result.note.code},
            status_code=actions_page.http_status(result.note.code),
        )
    return JSONResponse({"ok": True, **(result.payload or {})})


def _stated(code: str, text: str) -> JSONResponse:
    return JSONResponse(
        {"ok": False, "error": text, "code": code},
        status_code=actions_page.http_status(code),
    )


def _media_type(book: Catalog) -> JSONResponse:
    return JSONResponse(
        {
            "ok": False,
            "error": book.label_unsupported_media_type,
            "code": "unsupported_media_type",
        },
        status_code=415,
    )


def _json_content_type(request: Request) -> bool:
    return request.headers.get("content-type", "").lower().startswith("application/json")


async def _json_body(request: Request) -> object | None:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    return body


async def get_action_ready(request: Request) -> JSONResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    if project_id is None:
        return _stated("no_project", book.need_project)
    document = actions_page.readiness_document(
        workspace,
        project_id,
        actions_page.parse_run_ids(request.query_params.getlist("runs")),
    )
    return JSONResponse(document)


async def get_action_job(request: Request) -> JSONResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    if project_id is None:
        return _stated("no_project", book.need_project)
    job_id = request.query_params.get("id") or ""
    try:
        payload = actions_page.action_job_document(workspace, project_id, job_id)
    except LabError as exc:
        return _stated(exc.code, exc.message)
    return JSONResponse(payload)


async def post_action_train(request: Request) -> JSONResponse | RedirectResponse:
    from excavation_lab.web.label_routes import release_suggestion_adapter

    release_suggestion_adapter(request.app)
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    if project_id is None:
        return _stated("no_project", book.need_project)
    if _json_content_type(request):
        body = await _json_body(request)
        if not isinstance(body, dict):
            return _stated("invalid_action_train", "Нужна конфигурация обучения действий.")
        raw_ids = body.get("run_ids")
        ids = [item for item in raw_ids if isinstance(item, str)] if isinstance(raw_ids, list) else []
        result = actions_page.start_action_train(workspace, project_id, ids, body)
        return _write_response(result)
    form = await request.form()
    ids = actions_page.parse_run_ids(_form_list(form, "runs"))
    result = actions_page.start_action_train(workspace, project_id, ids, _train_fields(form))
    return _redirect_write(request, result, _text(form.get("video")))


async def post_action_cancel(request: Request) -> JSONResponse | RedirectResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    if project_id is None:
        return _stated("no_project", book.need_project)
    if _json_content_type(request):
        body = await _json_body(request)
        job_id = "" if not isinstance(body, dict) else body.get("job_id")
        if not isinstance(job_id, str):
            job_id = ""
        result = actions_page.cancel_action_job(workspace, project_id, job_id)
        return _write_response(result)
    form = await request.form()
    job_id = _text(form.get("job_id")) or ""
    result = actions_page.cancel_action_job(workspace, project_id, job_id)
    return _redirect_write(request, result, _text(form.get("video")))


async def post_action_infer(request: Request) -> JSONResponse | RedirectResponse:
    from excavation_lab.web.label_routes import release_suggestion_adapter

    release_suggestion_adapter(request.app)
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    if project_id is None:
        return _stated("no_project", book.need_project)
    if _json_content_type(request):
        body = await _json_body(request)
        if not isinstance(body, dict):
            return _stated("invalid_action_infer", "Инференс действий выполняется на одном прогоне.")
        run_id = body.get("run_id")
        model_id = body.get("action_model_id")
        result = actions_page.start_action_infer(
            workspace,
            project_id,
            run_id if isinstance(run_id, str) else "",
            model_id if isinstance(model_id, str) else "",
            body,
        )
        return _write_response(result)
    form = await request.form()
    result = actions_page.start_action_infer(
        workspace,
        project_id,
        _text(form.get("run_id")) or "",
        _text(form.get("action_model_id")) or "",
        _train_fields(form),
    )
    return _redirect_write(request, result, _text(form.get("video")))


async def get_action_export(request: Request) -> Response:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    if project_id is None:
        return _stated("no_project", book.need_project)
    job_id = request.query_params.get("job") or ""
    try:
        payload = actions_page.action_export_bytes(workspace, project_id, job_id)
    except LabError as exc:
        return _stated(exc.code, exc.message)
    return Response(
        payload,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="action_intervals.csv"'},
    )


async def post_action_evaluate(request: Request) -> JSONResponse | RedirectResponse:
    from excavation_lab.actions.evaluate import evaluate_actions

    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    if project_id is None:
        return _stated("no_project", book.need_project)
    if _json_content_type(request):
        body = await _json_body(request)
        if not isinstance(body, dict):
            return _stated("invalid_split", "split должен быть validation или test.")
        infer_job_id = body.get("infer_job_id")
        split = body.get("split") if isinstance(body.get("split"), str) else "validation"
        final = body.get("final") is True
        video_id = body.get("video") if isinstance(body.get("video"), str) else None
    else:
        form = await request.form()
        infer_job_id = _text(form.get("infer_job_id"))
        split = _text(form.get("split")) or "validation"
        final = _text(form.get("final")) in {"1", "true", "on"}
        video_id = _text(form.get("video"))
        body = None
    if not isinstance(infer_job_id, str) or not infer_job_id:
        return _stated("job_not_found", "Задание инференса действий не найдено.")
    try:
        report = evaluate_actions(
            workspace,
            project_id,
            infer_job_id,
            split=split,
            final=final,
        )
    except LabError as exc:
        if body is None:
            _remember(request, {"error": exc.message})
            return _actions_redirect(video_id, inference_id=infer_job_id)
        return _stated(exc.code, exc.message)
    if body is None:
        _remember(request, {"notice": book.action_evaluate_validation})
        return _actions_redirect(video_id, inference_id=infer_job_id)
    return JSONResponse({"ok": True, "report": report.to_dict()})


def _video_query(request: Request) -> str | None:
    return _query_token(request, "video")


def _query_token(request: Request, name: str) -> str | None:
    value = request.query_params.get(name)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or any(char in text for char in "\r\n;/\\\x00") or text in {".", ".."}:
        return None
    if ".." in text:
        return None
    return text


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _form_list(form: object, key: str) -> list[str]:
    getter = getattr(form, "getlist", None)
    if not callable(getter):
        return []
    values = getter(key)
    if not isinstance(values, list):
        return []
    return [item for item in values if isinstance(item, str)]


def _train_fields(form: object) -> dict[str, object]:
    getter = getattr(form, "get", None)
    fields: dict[str, object] = {}
    for key in ("epochs", "window_s", "min_confidence", "min_duration_s"):
        raw = getter(key) if callable(getter) else None
        fields[key] = _text(raw)
    return fields


def _redirect_write(
    request: Request,
    result: ActionWrite,
    video_id: str | None,
) -> RedirectResponse:
    if result.note is not None and result.note.text:
        _remember(request, {"error": result.note.text})
    elif result.payload and result.payload.get("job_id"):
        _remember(request, {"notice": str(result.payload["job_id"])})
    return _actions_redirect(video_id)


def _actions_redirect(
    video_id: str | None,
    inference_id: str | None = None,
) -> RedirectResponse:
    pairs: list[tuple[str, str]] = []
    if isinstance(video_id, str) and video_id and _plain_token(video_id):
        pairs.append(("video", video_id))
    if isinstance(inference_id, str) and inference_id and _plain_token(inference_id):
        pairs.append(("inference", inference_id))
    query = "&".join(f"{key}={value}" for key, value in pairs)
    target = "/actions" if not query else "/actions?" + query
    response = RedirectResponse(target, status_code=303)
    if pairs and pairs[0][0] == "video":
        response.set_cookie(
            _VIDEO_COOKIE,
            pairs[0][1],
            path="/",
            httponly=True,
            samesite="lax",
        )
    return response


def _plain_token(value: str) -> bool:
    if not value or value in {".", ".."} or ".." in value:
        return False
    return not any(char in value for char in "\r\n;/\\\x00")


def _remember(request: Request, payload: dict[str, str]) -> None:
    try:
        write_flash(_workspace(request), payload)
    except Exception:  # noqa: BLE001 — the page still has to show the sentence
        text = payload.get("error")
        if isinstance(text, str) and text:
            request.app.state.pending_notice = text
