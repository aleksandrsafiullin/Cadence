"""Label page and JSON routes. Storage goes through ``ui.label_page``."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from excavation_lab import storage
from excavation_lab.errors import LabError
from excavation_lab.ui import label_page
from excavation_lab.ui.copy import Catalog, catalog
from excavation_lab.ui.data_page import ActionNote
from excavation_lab.ui.video_page import video_choice_label
from excavation_lab.web.data_routes import _book, _project_id, _query, _workspace
from excavation_lab.web.preferences import write_flash

_LOCK_GUARD = threading.Lock()
_LOCK_TYPE = type(_LOCK_GUARD)


def label_routes() -> list[Route]:
    return [
        Route("/label", get_label, methods=["GET"]),
        Route("/label/frames", get_label_frames, methods=["GET"]),
        Route("/label/frame", get_label_frame, methods=["GET"]),
        Route("/label/frame", post_label_frame, methods=["POST"]),
        Route("/label/suggest", post_label_suggest, methods=["POST"]),
        Route("/label/annotator", post_label_annotator, methods=["POST"]),
    ]


async def get_label(request: Request) -> Response:
    from excavation_lab.web.app import _render

    response = _render(request, "label")
    if response.status_code == 303:
        return response
    status = getattr(request.state, "label_http_status", 200)
    if status != 200:
        response.status_code = int(status)
    return response


async def get_label_frames(request: Request) -> JSONResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    if project_id is None:
        return _failure(book, "no_project")
    video_id = _video_query(request)
    if video_id is None:
        return _failure(book, "video_not_found")
    try:
        listed = label_page.frames_for_video(workspace, project_id, video_id)
        counts = label_page.progress(workspace, project_id, video_id)
    except LabError as exc:
        return _failure(book, exc.code)
    return JSONResponse(
        {
            "ok": True,
            "video_id": video_id,
            "frames": [_frame_json(frame) for frame in listed],
            "progress": counts,
        }
    )


async def get_label_frame(request: Request) -> JSONResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    if project_id is None:
        return _failure(book, "no_project")
    video_id = _video_query(request)
    if video_id is None:
        return _failure(book, "video_not_found")
    try:
        index = label_page.parse_frame_token(request.query_params.get("frame"))
        payload = label_page.frame_payload(workspace, project_id, video_id, index, book)
    except LabError as exc:
        return _failure(book, exc.code)
    return JSONResponse({"ok": True, **payload})


async def post_label_frame(request: Request) -> JSONResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    if not _json_content_type(request):
        return _stated_failure(book, "unsupported_media_type", 415)
    project_id = _project_id(workspace)
    if project_id is None:
        return _failure(book, "no_project")
    video_id = _video_query(request)
    if video_id is None:
        return _failure(book, "video_not_found")
    try:
        index = label_page.parse_frame_token(request.query_params.get("frame"))
    except LabError as exc:
        return _failure(book, exc.code)
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeError, ValueError):
        return _failure(book, "invalid_label")
    if not isinstance(body, dict):
        return _failure(book, "invalid_label")
    result = label_page.save_frame(workspace, project_id, video_id, index, body, book)
    if isinstance(result, ActionNote):
        code = result.code or "label_save_failed"
        return JSONResponse(
            {"ok": False, "error": result.text, "code": code},
            status_code=label_page.http_status(code),
        )
    return JSONResponse({"ok": True, **result})


def post_label_suggest(request: Request) -> JSONResponse:
    """Sync handler. Starlette runs it in the threadpool, so ``predict`` is not on the loop."""
    workspace = _workspace(request)
    book = _book(workspace)
    if not _json_content_type(request):
        return _stated_failure(book, "unsupported_media_type", 415)
    project_id = _project_id(workspace)
    if project_id is None:
        return _failure(book, "no_project")
    video_id = _video_query(request)
    if video_id is None:
        return _failure(book, "video_not_found")
    try:
        index = label_page.parse_frame_token(request.query_params.get("frame"))
    except LabError as exc:
        return _failure(book, exc.code)
    try:
        body = _request_json(request)
    except (json.JSONDecodeError, UnicodeError, ValueError):
        return _failure(book, "invalid_label")
    if not isinstance(body, dict):
        return _failure(book, "invalid_label")
    model_id = body.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        return _failure(book, "bundle_missing")
    try:
        result, dropped = suggest_holding_lock(
            request.app,
            workspace,
            project_id,
            video_id,
            index,
            model_id.strip(),
            book,
        )
    except LabError as exc:
        return _failure(book, exc.code)
    if isinstance(result, ActionNote):
        code = result.code or "label_save_failed"
        return JSONResponse(
            {"ok": False, "error": result.text, "code": code},
            status_code=label_page.http_status(code),
        )
    return JSONResponse({"ok": True, "boxes": result, "dropped": dropped})


def _request_json(request: Request) -> object:
    """Read JSON from the event loop while this handler runs in the threadpool."""
    from anyio.from_thread import run

    return run(request.json)


def suggestion_lock(app: Any) -> threading.Lock:
    """The lock shared by ``suggest_holding_lock`` and ``release_suggestion_adapter``."""
    current = getattr(app.state, "suggestion_lock", None)
    if isinstance(current, _LOCK_TYPE):
        return current
    with _LOCK_GUARD:
        current = getattr(app.state, "suggestion_lock", None)
        if isinstance(current, _LOCK_TYPE):
            return current
        created = threading.Lock()
        app.state.suggestion_lock = created
        return created


def suggest_holding_lock(
    app: Any,
    workspace: Path,
    project_id: str,
    video_id: str,
    index: int,
    model_id: str,
    book: Catalog,
    *,
    loader: Any = None,
) -> tuple[list[dict[str, Any]] | ActionNote, int]:
    """Run ``label_page.suggest`` while holding ``app.state.suggestion_lock``.

    ``predict`` stays inside the lock, so ``release_suggestion_adapter`` waits
    and the in-flight response is already complete when the adapter closes.
    """
    chosen = loader if loader is not None else getattr(app.state, "suggestion_loader", None)
    lock = suggestion_lock(app)
    with lock:
        cache = _StateCache(app)
        result = label_page.suggest(
            workspace,
            project_id,
            video_id,
            index,
            model_id,
            detector_cache=cache,
            loader=chosen,
            book=book,
        )
        return result, int(cache.dropped)


def release_suggestion_adapter(app: Any) -> None:
    """Close the page-server detector. Call this before ``enqueue_*``."""
    lock = suggestion_lock(app)
    with lock:
        _close_suggestion_adapter(app)


def _close_suggestion_adapter(app: Any) -> None:
    adapter = getattr(app.state, "suggestion_adapter", None)
    app.state.suggestion_adapter = None
    app.state.suggestion_adapter_key = None
    if adapter is not None:
        adapter.close()


class _StateCache:
    """One adapter stored on ``app.state.suggestion_adapter``."""

    def __init__(self, app: Any) -> None:
        self._app = app
        self.dropped = 0

    @property
    def adapter(self) -> Any:
        return getattr(self._app.state, "suggestion_adapter", None)

    @adapter.setter
    def adapter(self, value: Any) -> None:
        self._app.state.suggestion_adapter = value

    @property
    def key(self) -> str | None:
        return getattr(self._app.state, "suggestion_adapter_key", None)

    @key.setter
    def key(self, value: str | None) -> None:
        self._app.state.suggestion_adapter_key = value

    def close(self) -> None:
        _close_suggestion_adapter(self._app)


async def post_label_annotator(request: Request) -> Response:
    workspace = _workspace(request)
    book = _book(workspace)
    if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
        return _stated_failure(book, "cross_site", 403)
    form = await request.form()
    video_id = _form_video(form.get("video")) or _video_query(request)
    frame = _form_frame(form.get("frame"))
    if frame is None:
        raw_frame = request.query_params.get("frame")
        frame = raw_frame if isinstance(raw_frame, str) else None
    try:
        label_page.set_annotator(workspace, _form_text(form.get("annotator")))
    except LabError as exc:
        _remember(request, label_page.message_for(book, exc.code))
    return _back(video_id, frame)


def label_context(request: Request, base: dict[str, object]) -> dict[str, object]:
    """Page fields. Unknown video or frame sets ``request.state.label_http_status``."""
    request.state.label_http_status = 200
    book = catalog(str(base.get("lang") or "ru"))
    workspace = Path(request.app.state.workspace)
    selected = _video_query(request)
    project_raw = base.get("project_id")
    project_id = project_raw if isinstance(project_raw, str) and project_raw else None
    context = _empty(book, workspace, selected)
    if project_id is None or selected is None:
        if _rejected_video(request):
            request.state.label_http_status = 404
            context["label_error"] = book.label_unknown_video
        return context
    try:
        index = _page_index(request, workspace, project_id, selected)
        payload = label_page.frame_payload(workspace, project_id, selected, index, book)
    except LabError as exc:
        if label_page.http_status(exc.code) == 404:
            request.state.label_http_status = 404
        context["label_error"] = label_page.message_for(book, exc.code)
        context["selected_video_id"] = selected
        return context
    video = storage.get_video(workspace, project_id, selected)
    context.update(_ready(book, selected, video_choice_label(video), payload))
    return context


def _empty(book: Catalog, workspace: Path, selected: str | None) -> dict[str, object]:
    return {
        "book": book,
        "label_loading": book.label_loading,
        "label_back_to_data": book.label_back_to_data,
        "label_back_href": _data_href(selected),
        "label_copy_json": label_page.script_json(_label_copy(book)),
        "reviewed_at_text": "",
        "frame_annotator": "",
        "selected_video_id": selected,
        "label_ready": False,
        "label_error": None,
        "label_state_json": label_page.script_json({"ok": False}),
        "video_label": "",
        "split_badge": "",
        "split_token": "",
        "source_time_text": "",
        "frame_index": None,
        "progress_line": "",
        "annotator_value": label_page.annotator_name(workspace),
        "classes": [],
        "prev_href": None,
        "next_href": None,
        "review_status": "",
        "image_url": "",
        "image_width": 0,
        "image_height": 0,
        "frame_note": "",
        "suggestions_allowed": False,
        "suggestion_reason": "",
        "suggestion_bundles": [],
    }


def _ready(
    book: Catalog,
    video_id: str,
    video_label: str,
    payload: dict[str, Any],
) -> dict[str, object]:
    index = payload["index"]
    prev_index = payload["prev_index"]
    next_index = payload["next_index"]
    role = payload["split"]
    return {
        "label_ready": True,
        "label_error": None,
        "label_state_json": label_page.script_json(payload),
        "video_label": video_label,
        "split_badge": payload["split_label"],
        "split_token": "" if role is None else role,
        "source_time_text": _time_text(payload["source_time_s"], book),
        "frame_index": index,
        "progress_line": label_page.format_progress(payload["progress"], book),
        "annotator_value": payload["annotator_name"],
        "classes": payload["classes"],
        "prev_href": _label_href(video_id, prev_index),
        "next_href": _label_href(video_id, next_index),
        "review_status": payload["review_status"],
        "image_url": payload["image_url"],
        "image_width": payload["width"],
        "image_height": payload["height"],
        "frame_note": payload["note"] or "",
        "selected_video_id": video_id,
        "reviewed_at_text": payload["reviewed_at"] or "",
        "frame_annotator": payload["annotator"] or "",
        "label_copy_json": label_page.script_json(_label_copy(book)),
        "suggestions_allowed": bool(payload["suggestions_allowed"]),
        "suggestion_reason": payload["suggestion_reason"] or "",
        "suggestion_bundles": payload["suggestion_bundles"],
    }


def _page_index(request: Request, workspace: Path, project_id: str, video_id: str) -> int:
    raw = request.query_params.get("frame")
    if raw is None:
        listed = label_page.frames_for_video(workspace, project_id, video_id)
        if not listed:
            raise LabError(
                "frame_not_sampled",
                "Этого кадра нет в отборе. Сначала отберите кадры.",
            )
        return listed[0].index
    return label_page.parse_frame_token(raw)


def _time_text(seconds: object, book: Catalog) -> str:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return book.not_calculated
    return f"{float(seconds):.3f} {book.seconds_suffix}"


def _data_href(video_id: str | None) -> str:
    """Data page, labeling step. The hash is not part of the query."""
    if video_id is None:
        return "/data#step-label"
    return "/data?" + _query((("video", video_id),)) + "#step-label"


def _label_copy(book: Catalog) -> dict[str, object]:
    """Strings the labeler script formats. Not part of the frame payload."""
    return {
        "confirm_empty": book.label_confirm_empty,
        "progress": book.label_progress,
        "seconds": book.seconds_suffix,
        "frame_index": book.label_frame_index,
        "splits": {
            "train": book.split_name_train,
            "validation": book.split_name_validation,
            "test": book.split_name_test,
            "unassigned": book.block_unassigned,
        },
        "suggest_accept": book.label_suggest_accept,
        "suggest_accept_all": book.label_suggest_accept_all,
        "suggest_reject_all": book.label_suggest_reject_all,
        "suggest_dropped": book.label_suggest_dropped,
    }


def _label_href(video_id: str, index: object) -> str | None:
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    return "/label?" + _query((("video", video_id), ("frame", str(index))))


def _frame_json(frame: label_page.LabelFrame) -> dict[str, object]:
    return {
        "index": frame.index,
        "source_time_s": frame.source_time_s,
        "split": frame.split,
        "image_url": frame.image_url,
        "review_status": frame.review_status,
        "label_state": frame.label_state,
        "box_count": frame.box_count,
    }


def _failure(book: Catalog, code: str) -> JSONResponse:
    return _stated_failure(book, code, label_page.http_status(code))


def _stated_failure(book: Catalog, code: str, status: int) -> JSONResponse:
    return JSONResponse(
        {"ok": False, "error": label_page.message_for(book, code), "code": code},
        status_code=status,
    )


def _json_content_type(request: Request) -> bool:
    return request.headers.get("content-type", "").lower().startswith("application/json")


def _remember(request: Request, text: str) -> None:
    try:
        write_flash(_workspace(request), {"error": text})
    except Exception:  # noqa: BLE001 — the page still has to show the sentence
        request.app.state.pending_notice = text


def _back(video_id: str | None, frame: str | None) -> RedirectResponse:
    pairs: list[tuple[str, str]] = []
    if video_id is not None:
        pairs.append(("video", video_id))
    if frame is not None and frame.isascii() and frame.isdigit():
        pairs.append(("frame", frame))
    target = "/label" if not pairs else "/label?" + _query(tuple(pairs))
    return RedirectResponse(target, status_code=303)


def _rejected_video(request: Request) -> bool:
    raw = request.query_params.get("video")
    if not isinstance(raw, str) or not raw.strip():
        return False
    text = raw.strip()
    if _bad_cookie(text):
        return False
    return _unsafe_leaf(text)


def _video_query(request: Request) -> str | None:
    video = request.query_params.get("video")
    if not isinstance(video, str):
        return None
    text = video.strip()
    if not text or _bad_cookie(text) or _unsafe_leaf(text):
        return None
    return text


def _form_video(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or _bad_cookie(text) or _unsafe_leaf(text):
        return None
    return text


def _form_frame(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.isascii() and text.isdigit():
        return text
    return None


def _form_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value


def _bad_cookie(value: str) -> bool:
    return any(char in value for char in "\r\n;")


def _unsafe_leaf(value: str) -> bool:
    if not value or value in {".", ".."}:
        return True
    if value != value.strip():
        return True
    if any(char in value for char in "/\\\x00"):
        return True
    return ".." in value
