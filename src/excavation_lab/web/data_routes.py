"""Data routes. Page helpers stay in ``ui.data_page``."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from excavation_lab import storage
from excavation_lab.datasets.splits import FrozenSplit, propose_for_project
from excavation_lab.errors import LabError
from excavation_lab.schemas import load_pilot
from excavation_lab.ui import data_page
from excavation_lab.ui.copy import Catalog, catalog
from excavation_lab.ui.locale import resolve_locale
from excavation_lab.ui.video_page import video_choice_label
from excavation_lab.web import uploads
from excavation_lab.web.preferences import choose_project_id, write_flash

_COOKIE = "lab_video"
_PLANS = ("diagnostic", "content", "expanded")
_SPLITS = ("train", "validation", "test")
_JPEG_NAME = re.compile(r"^[\w.-]+\.jpe?g$", re.IGNORECASE)


def data_routes() -> list[Route]:
    return [
        Route("/data/blocks", post_data_blocks, methods=["POST"]),
        Route("/data/episodes", post_data_episodes, methods=["POST"]),
        Route("/data/episode", post_data_episode, methods=["POST"]),
        Route("/data/propose", post_data_propose, methods=["POST"]),
        Route("/data/freeze", post_data_freeze, methods=["POST"]),
        Route("/data/sample", post_data_sample, methods=["POST"]),
        Route("/data/class", post_data_class, methods=["POST"]),
        Route("/data/export", post_data_export, methods=["POST"]),
        Route("/data/import", post_data_import, methods=["POST"]),
        Route("/data/snapshot", post_data_snapshot, methods=["POST"]),
        Route("/data/batch", get_data_batch, methods=["GET"]),
        Route("/data/frame", get_data_frame, methods=["GET"]),
    ]


def data_context(
    request: Request,
    base: dict[str, object],
    flash: dict[str, str],
) -> dict[str, object]:
    """Template fields for ``data.html``. Splits stay in the dataset service."""
    workspace = _workspace(request)
    book = catalog(str(base.get("lang") or "ru"))
    project_raw = base.get("project_id")
    project_id = project_raw if isinstance(project_raw, str) and project_raw else None
    shared = {
        "flash_error": flash.get("error"),
        "flash_notice": flash.get("notice"),
        "book": book,
        "list_error": None,
        "show_empty": False,
        "show_editor": False,
        "selected_video_id": None,
        "video_options": [],
        "snapshot_enabled": False,
        "snapshot_reason": book.snapshot_locked,
        "review_text": data_page.review_count_text(None, book),
        "label_counts_video": book.not_calculated,
        "label_counts_project": book.not_calculated,
        "review_error": None,
        "snapshot_other": None,
        "review_warnings": [],
        "review_errors": [],
    }
    if project_id is None:
        shared["show_empty"] = True
        return shared
    gate = data_page.snapshot_gate(workspace, project_id, book)
    shared["snapshot_enabled"] = gate.enabled
    shared["snapshot_reason"] = gate.reason
    try:
        videos = storage.list_videos(workspace, project_id)
    except LabError as exc:
        shared["list_error"] = f"{exc.code}: {exc.message}"
        return shared
    if not videos:
        shared["show_empty"] = True
        return shared
    video_id = _wanted_video(request, flash, videos)
    shared.update(_editor(workspace, project_id, video_id, videos, book, gate))
    shared["show_editor"] = True
    shared["selected_video_id"] = video_id
    return shared


async def post_data_blocks(request: Request) -> JSONResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    if project_id is None:
        return _episode_json(book.data_empty)
    try:
        body = await request.json()
    except (UnicodeError, ValueError):
        body = None
    if not isinstance(body, dict):
        body = {}
    video_id = body.get("video_id")
    if not isinstance(video_id, str):
        video_id = ""
    result = data_page.save_role_blocks(
        workspace,
        project_id,
        video_id,
        body.get("blocks"),
        text=book,
    )
    if result.note.code:
        return _episode_json(result.note.text)
    return JSONResponse({"ok": True, "blocks": [dict(row) for row in result.blocks]})


async def post_data_episodes(request: Request) -> JSONResponse:
    """Refuse. Episodes are not the split. Nothing is written."""
    del request
    message = "refused"
    try:
        propose_for_project(Path("."), "", "")
    except LabError as exc:
        message = exc.message
    return _episode_json(message)


async def post_data_episode(request: Request) -> RedirectResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    form = await request.form()
    video_id = _text(form.get("video_id"))
    if project_id is None:
        _remember(request, {"error": book.data_empty})
        return _redirect()
    if not video_id:
        _remember(request, {"error": book.video_choice})
        return _redirect()
    start_s = _float(form.get("start_s"))
    end_s = _float(form.get("end_s"))
    if start_s is None or end_s is None:
        note = data_page.ActionNote(text=book.episode_boundary_error, code="invalid_episode")
    else:
        note = data_page.store_episode(
            workspace,
            project_id,
            video_id,
            episode_id=_text(form.get("episode_id")),
            split_group_id=_text(form.get("split_group_id")),
            start_s=start_s,
            end_s=end_s,
            label=_text(form.get("label")),
            concurrent=_checked(form.get("concurrent")),
            text=book,
        )
    _remember(request, _note_payload(note, video_id))
    return _redirect(video_id)


async def post_data_propose(request: Request) -> RedirectResponse:
    return await _video_action(request, data_page.propose_recorded)


async def post_data_freeze(request: Request) -> RedirectResponse:
    return await _video_action(request, data_page.freeze_recorded_proposal)


async def post_data_sample(request: Request) -> RedirectResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    form = await request.form()
    video_id = _text(form.get("video_id"))
    if project_id is None:
        _remember(request, {"error": book.data_empty})
        return _redirect()
    if not video_id:
        _remember(request, {"error": book.video_choice})
        return _redirect()
    spacing_s = _spacing(form.get("spacing_s"))
    mode = _text(form.get("sample_mode"))
    if mode == "plan":
        plan = _text(form.get("plan")) or None
        note = data_page.run_sample(
            workspace,
            project_id,
            video_id,
            plan=plan,
            split=None,
            count=None,
            spacing_s=spacing_s,
        )
    elif mode == "count":
        split = _text(form.get("split")) or None
        count = _count(form.get("count"))
        note = data_page.run_sample(
            workspace,
            project_id,
            video_id,
            plan=None,
            split=split,
            count=count,
            spacing_s=spacing_s,
        )
    else:
        note = data_page.ActionNote(text=book.sample_rule, code="sample_args")
    _remember(request, _note_payload(note, video_id))
    return _redirect(video_id)


async def post_data_class(request: Request) -> RedirectResponse:
    """Append one catalog class. A snapshot does not close this."""
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    form = await request.form()
    video_id = _text(form.get("video_id"))
    if project_id is None:
        _remember(request, {"error": book.catalog_required})
        return _redirect(video_id)
    note = data_page.add_catalog_class(
        workspace,
        project_id,
        _text(form.get("class_name")),
        _text(form.get("title")),
        _text(form.get("box")),
        book,
    )
    _remember(request, _note_payload(note, video_id))
    return _redirect(video_id)


async def post_data_export(request: Request) -> RedirectResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    form = await request.form()
    video_id = _text(form.get("video_id"))
    if project_id is None:
        _remember(request, {"error": book.data_empty})
        return _redirect()
    if not video_id:
        _remember(request, {"error": book.video_choice})
        return _redirect()
    note = data_page.run_export(
        workspace,
        project_id,
        video_id,
        _text(form.get("split")),
        include_predictions=_checked(form.get("include_predictions")),
    )
    _remember(request, _note_payload(note, video_id))
    return _redirect(video_id)


async def post_data_import(request: Request) -> Response:
    """Store the ZIP on disk, then ``run_import``. The route does not open it."""
    if uploads.declared_over_cap(request.headers.get("content-length")):
        return Response(status_code=413)
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    if project_id is None:
        _remember(request, {"error": book.data_empty})
        return _redirect()
    streamed = await uploads.stream_import(request, workspace, project_id, book)
    video_id = _text(streamed.fields.get("video_id"))
    if streamed.status == 413:
        return Response(status_code=413)
    if streamed.error:
        _remember(request, _error_payload(streamed.error, video_id))
        return _redirect(video_id)
    if streamed.missing or streamed.empty or streamed.destination is None:
        _remember(request, _error_payload(book.import_path_required, video_id))
        return _redirect(video_id)
    split = _text(streamed.fields.get("split"))
    if not video_id or split not in _SPLITS:
        streamed.destination.unlink(missing_ok=True)
        _remember(request, _error_payload(book.import_path_required, video_id))
        return _redirect(video_id)
    note = data_page.run_import(
        workspace,
        project_id,
        video_id,
        split,
        str(streamed.destination),
        book,
    )
    _remember(request, _note_payload(note, video_id))
    return _redirect(video_id)


async def post_data_snapshot(request: Request) -> RedirectResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    if project_id is None:
        _remember(request, {"error": book.snapshot_locked})
        return _redirect()
    note = data_page.run_snapshot(workspace, project_id, book)
    _remember(request, _note_payload(note, ""))
    return _redirect()


async def get_data_batch(request: Request) -> Response:
    """Attachment for one exported ``annotation_batch.zip`` under the project."""
    workspace = _workspace(request)
    project_id = _project_id(workspace)
    if project_id is None:
        return Response(status_code=404)
    video_id = request.query_params.get("video", "")
    split = request.query_params.get("split", "")
    try:
        path = _batch_zip(workspace, project_id, video_id, split)
    except FileNotFoundError:
        return Response(status_code=404)
    return FileResponse(
        path,
        media_type="application/zip",
        filename="annotation_batch.zip",
    )


async def get_data_frame(request: Request) -> Response:
    """One sampled JPEG named by the sample manifest. Other files stay unserved."""
    workspace = _workspace(request)
    project_id = _project_id(workspace)
    if project_id is None:
        return Response(status_code=404)
    video_id = request.query_params.get("video", "")
    name = request.query_params.get("name", "")
    if _unsafe_leaf(video_id) or _unsafe_jpeg(name):
        return Response(status_code=404)
    try:
        paths, _error = data_page.contact_jpegs(workspace, project_id, video_id)
        project = storage.project_directory(workspace, project_id).resolve()
    except LabError:
        return Response(status_code=404)
    match = next((path for path in paths if path.name == name), None)
    if match is None:
        return Response(status_code=404)
    try:
        resolved = match.resolve()
        allowed = storage.resolve_project_path(workspace, project_id, resolved)
    except LabError:
        return Response(status_code=404)
    images_root = (project / "samples" / video_id / "images").resolve()
    if (
        allowed != resolved
        or not resolved.is_file()
        or not images_root.is_relative_to(project)
        or not resolved.is_relative_to(images_root)
        or resolved.suffix.lower() not in {".jpg", ".jpeg"}
    ):
        return Response(status_code=404)
    return FileResponse(resolved, media_type="image/jpeg")


async def _video_action(
    request: Request,
    action: Callable[[Path, str, str, Catalog], data_page.ActionNote],
) -> RedirectResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    form = await request.form()
    video_id = _text(form.get("video_id"))
    if project_id is None:
        _remember(request, {"error": book.data_empty})
        return _redirect()
    if not video_id:
        _remember(request, {"error": book.video_choice})
        return _redirect()
    note = action(workspace, project_id, video_id, book)
    _remember(request, _note_payload(note, video_id))
    return _redirect(video_id)


def _editor(
    workspace: Path,
    project_id: str,
    video_id: str,
    videos: Sequence[Mapping[str, object]],
    book: Catalog,
    gate: data_page.SnapshotGate,
) -> dict[str, object]:
    document, load_error = data_page.load_split_view(workspace, project_id, video_id, book)
    video = next(item for item in videos if str(item["video_id"]) == video_id)
    frozen = isinstance(document, FrozenSplit)
    editor = data_page.block_editor_payload(video, document, frozen=frozen, text=book)
    duration_s = editor["duration_s"]
    offset_s = editor["offset_s"]
    fps = editor["fps"]
    duration_known = isinstance(duration_s, float)
    offset_known = isinstance(offset_s, float)
    clock_known = duration_known and offset_known
    fps_known = isinstance(fps, float)
    guards: list[str] = []
    blocks: list[str] = []
    limitation = None
    independence_line = None
    split_hash_line = None
    split_state = "missing"
    split_state_heading = book.proposal_missing
    if document is not None:
        guards = data_page.guard_lines(document, book)
        blocks = data_page.block_lines(document, book)
        limitation = data_page.limitation_text(document)
        independence_line = f"{book.independence_label}: {document.independence.value}."
        if isinstance(document, FrozenSplit):
            split_state = "frozen"
            split_state_heading = book.frozen_heading
            split_hash_line = f"{book.split_hash}: {document.content_hash}"
        else:
            split_state = "proposal"
            split_state_heading = book.proposal_heading
    paths, contact_error = data_page.contact_jpegs(workspace, project_id, video_id, book)
    result, review_error = data_page.load_review(workspace, project_id)
    if review_error and review_error.startswith("dataset_not_found:"):
        review_error = None
    blocked_id = data_page.first_unfrozen_video_id(workspace, project_id)
    snapshot_other = None
    if blocked_id and blocked_id != video_id:
        blocked = next(
            (item for item in videos if str(item["video_id"]) == blocked_id),
            None,
        )
        if blocked is not None:
            snapshot_other = book.snapshot_other.replace(
                "{name}",
                video_choice_label(blocked),
            )
    warnings: list[str] = []
    errors: list[str] = []
    if result is not None and review_error is None:
        warnings = [data_page.validation_issue_line(item) for item in result.warnings]
        errors = [data_page.validation_issue_line(item) for item in result.errors]
    options = []
    if len(videos) > 1:
        options = [
            {
                "id": str(item["video_id"]),
                "label": video_choice_label(item),
                "selected": str(item["video_id"]) == video_id,
            }
            for item in videos
        ]
    block_duration = duration_s if clock_known and isinstance(duration_s, float) else None
    block_offset = offset_s if clock_known and isinstance(offset_s, float) else None
    role_blocks = () if document is None else document.blocks
    label_catalog_id, label_classes = data_page.labeling_classes(workspace, project_id)
    return {
        "video_options": options,
        "media_url": f"/media/{project_id}/{video_id}",
        "block_clock": clock_known,
        "block_duration_s": duration_s if duration_known else None,
        "block_offset_s": offset_s if offset_known else None,
        "block_duration_attr": data_page.duration_attr(block_duration),
        "block_offset_attr": data_page.offset_attr(block_offset),
        "timeline_pieces": data_page.timeline_pieces(
            role_blocks,
            block_offset,
            block_duration,
            book,
        ),
        "block_fps": fps if fps_known else None,
        "block_frame_step": data_page.frame_step_text(fps if fps_known else None),
        "blocks_frozen": frozen,
        "block_editor": editor,
        "split_load_error": load_error,
        "split_state": split_state,
        "split_state_heading": split_state_heading,
        "episode_label_lines": data_page.stored_episode_labels(
            workspace, project_id, video_id, book
        ),
        "split_hash_line": split_hash_line,
        "independence_line": independence_line,
        "limitation": limitation,
        "guard_lines": guards,
        "block_lines": blocks,
        "plan_options": [
            {"id": plan_id, "label": book.sample_plan_labels[plan_id]} for plan_id in _PLANS
        ],
        "splits": _SPLITS,
        "spacing_value": format(float(load_pilot().sample_spacing_s), "g"),
        "plan_count_text": data_page.plan_count_text(book),
        "contact_names": [path.name for path in paths],
        "contact_error": contact_error,
        "review_text": data_page.review_count_text(result, book),
        "label_counts_video": data_page.format_store_counts(
            data_page.store_counts(workspace, project_id, video_id),
            book,
        ),
        "label_counts_project": data_page.format_store_counts(
            data_page.store_counts(workspace, project_id),
            book,
        ),
        "review_error": review_error,
        "snapshot_other": snapshot_other,
        "review_warnings": warnings,
        "review_errors": errors,
        "steps": data_page.workflow_steps(
            episode_count=len(role_blocks),
            split_state=split_state,
            jpeg_count=len(paths),
            review_ready=gate.review_ready,
            gate=gate,
            snapshot_exists=data_page.snapshot_written(workspace, project_id),
            label_caption=data_page.label_step_caption(
                workspace, project_id, video_id, book
            ),
            text=book,
        ),
        "split_choices": [
            {"value": name, "label": data_page.split_label(name, book)} for name in _SPLITS
        ],
        "contact_frames": [
            {
                "name": path.name,
                "src": "/data/frame?" + _query((("video", video_id), ("name", path.name))),
            }
            for path in paths
        ],
        "label_open_href": "/label?" + _query((("video", video_id),)),
        "label_links": [
            {
                "index": _source_frame_index(path.name, video_id),
                "href": _label_href(video_id, path.name),
            }
            for path in paths
        ],
        "label_catalog_id": label_catalog_id,
        "label_classes": label_classes,
        "batch_links": [
            {
                "split": split,
                "label": data_page.split_label(split, book),
                "href": "/data/batch?" + _query((("video", video_id), ("split", split))),
            }
            for split in _ready_batches(workspace, project_id, video_id)
        ],
    }


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


def _wanted_video(
    request: Request,
    flash: dict[str, str],
    videos: Sequence[Mapping[str, object]],
) -> str:
    ids = [str(item["video_id"]) for item in videos]
    allowed = set(ids)
    for candidate in (
        flash.get("video_id"),
        request.query_params.get("video"),
        request.cookies.get(_COOKIE),
    ):
        if isinstance(candidate, str) and candidate in allowed and not _bad_token(candidate):
            return candidate
    return ids[0]


def _remember(request: Request, payload: dict[str, str]) -> None:
    try:
        write_flash(_workspace(request), payload)
    except Exception:  # noqa: BLE001 — the page still has to show the sentence
        text = payload.get("error")
        if isinstance(text, str) and text:
            request.app.state.pending_notice = text


def _episode_json(message: str) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=400)


def _redirect(video_id: str | None = None) -> RedirectResponse:
    response = RedirectResponse("/data", status_code=303)
    if isinstance(video_id, str) and video_id and not _bad_token(video_id):
        response.set_cookie(_COOKIE, video_id, path="/", httponly=True, samesite="lax")
    return response


def _note_payload(note: data_page.ActionNote, video_id: str) -> dict[str, str]:
    payload: dict[str, str] = {}
    if video_id and not _bad_token(video_id):
        payload["video_id"] = video_id
    text = note.text.strip()
    if note.code:
        prefix = f"{note.code}:"
        if text.startswith(prefix):
            shown = text
        elif text:
            shown = f"{note.code}: {text}"
        else:
            shown = note.code
        payload["error"] = shown
        return payload
    if text:
        payload["notice"] = text
    return payload


def _error_payload(text: str, video_id: str) -> dict[str, str]:
    payload = {"error": text}
    if video_id and not _bad_token(video_id):
        payload["video_id"] = video_id
    return payload


def _text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _float(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _count(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        number = int(text)
    except ValueError:
        return None
    return number


def _spacing(value: object) -> float:
    default = float(load_pilot().sample_spacing_s)
    number = _float(value)
    if number is None or number <= 0:
        return default
    return number


def _checked(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"1", "true", "on", "yes"}


def _bad_token(value: str) -> bool:
    return any(char in value for char in "\r\n;")


def _source_frame_index(name: str, video_id: str) -> int | None:
    """Frame index from ``<video_id>_<index>.jpg``. The template does not parse names."""
    lowered = name.lower()
    if lowered.endswith(".jpeg"):
        stem = name[: -len(".jpeg")]
    elif lowered.endswith(".jpg"):
        stem = name[: -len(".jpg")]
    else:
        return None
    prefix = f"{video_id}_"
    if not stem.startswith(prefix):
        return None
    tail = stem[len(prefix) :]
    if not tail.isascii() or not tail.isdigit():
        return None
    return int(tail)


def _label_href(video_id: str, name: str) -> str:
    index = _source_frame_index(name, video_id)
    if index is None:
        return "/label?" + _query((("video", video_id),))
    return "/label?" + _query((("video", video_id), ("frame", str(index))))


def _query(pairs: Sequence[tuple[str, str]]) -> str:
    return "&".join(f"{_url_part(key)}={_url_part(value)}" for key, value in pairs)


def _url_part(value: str) -> str:
    pieces: list[str] = []
    for char in value:
        if char.isascii() and (char.isalnum() or char in "-._~"):
            pieces.append(char)
            continue
        pieces.append("".join(f"%{byte:02X}" for byte in char.encode("utf-8")))
    return "".join(pieces)


def _unsafe_leaf(value: str) -> bool:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        return True
    if value != value.strip():
        return True
    if any(char in value for char in "/\\\x00"):
        return True
    return ".." in value


def _unsafe_jpeg(value: str) -> bool:
    if _unsafe_leaf(value):
        return True
    return _JPEG_NAME.fullmatch(value) is None


def _batch_zip(workspace: Path, project_id: str, video_id: str, split: str) -> Path:
    """The exported zip for one video and split, only if it stays in the project."""
    if split not in _SPLITS or _unsafe_leaf(video_id):
        raise FileNotFoundError(video_id)
    try:
        project = storage.project_directory(workspace, project_id).resolve()
    except LabError as exc:
        raise FileNotFoundError(video_id) from exc
    leaf = f"{video_id}__{split}"
    batch_dir = (project / "annotation_batches" / leaf).resolve()
    if not batch_dir.is_relative_to(project):
        raise FileNotFoundError(video_id)
    resolved = (batch_dir / "annotation_batch.zip").resolve()
    if resolved.parent != batch_dir or resolved.name != "annotation_batch.zip":
        raise FileNotFoundError(video_id)
    if not resolved.is_file() or not resolved.is_relative_to(batch_dir):
        raise FileNotFoundError(video_id)
    try:
        allowed = storage.resolve_project_path(workspace, project_id, resolved)
    except LabError as exc:
        raise FileNotFoundError(video_id) from exc
    if allowed != resolved or not allowed.is_relative_to(project):
        raise FileNotFoundError(video_id)
    return allowed


def _ready_batches(workspace: Path, project_id: str, video_id: str) -> list[str]:
    ready: list[str] = []
    for split in _SPLITS:
        try:
            _batch_zip(workspace, project_id, video_id, split)
        except FileNotFoundError:
            continue
        ready.append(split)
    return ready
