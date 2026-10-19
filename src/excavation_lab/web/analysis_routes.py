"""Analysis routes. Page helpers stay in ``ui.analysis_page``."""

from __future__ import annotations

import base64
import math
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from starlette.requests import Request
from starlette.responses import FileResponse, RedirectResponse, Response
from starlette.routing import Route

from excavation_lab import storage
from excavation_lab.errors import LabError
from excavation_lab.hardware import official_nano_weights_path, weights_cache_dir
from excavation_lab.jobs import reopen
from excavation_lab.schemas import ZONE_NAMES, JobRecord, load_pilot
from excavation_lab.ui import analysis_page
from excavation_lab.ui.copy import Catalog, catalog
from excavation_lab.ui.locale import resolve_locale
from excavation_lab.ui.status import status_tone
from excavation_lab.ui.video_page import job_status_line, video_choice_label
from excavation_lab.visualization.annotate import AnnotateToggles
from excavation_lab.web.preferences import choose_project_id, write_flash
from excavation_lab.web.range_files import traversal

_VIDEO_COOKIE = "lab_video"
_TIME_COOKIE = "lab_analysis_time"
_MODE_COOKIE = "lab_analysis_mode"
_RUN_COOKIE = "lab_analysis_run"
_VIEW_COOKIE = "lab_analysis_view"
_MODES = frozenset({"baseline", "custom"})
_TIME_EPS = 1e-3
_BASELINE_MODEL_ID = "baseline_coco"
_FRAME_QUERY = frozenset({"video_id", "source_time_s"})
_TOGGLE_FIELDS = (
    ("boxes", "toggle_boxes"),
    ("class_names", "toggle_classes"),
    ("scores", "toggle_scores"),
    ("track_ids", "toggle_ids"),
    ("bucket_trail", "toggle_trail"),
    ("zones", "toggle_zones"),
)


def analysis_routes() -> list[Route]:
    return [
        Route("/analysis/start", post_analysis_start, methods=["POST"]),
        Route("/analysis/export", post_analysis_export, methods=["POST"]),
        Route("/analysis/zones", post_analysis_zones, methods=["POST"]),
        Route("/analysis/threshold", post_analysis_threshold, methods=["POST"]),
        Route("/analysis/frame", get_analysis_frame, methods=["GET"]),
        Route(
            "/runs/{project_id}/{run_id}/video",
            get_run_video,
            methods=["GET", "HEAD"],
        ),
    ]


def analysis_context(
    request: Request,
    base: dict[str, object],
    flash: dict[str, str],
) -> dict[str, object]:
    """Template fields for ``analysis.html``. Start stays in ``request_analysis``."""
    workspace = _workspace(request)
    book = catalog(str(base.get("lang") or "ru"))
    project_raw = base.get("project_id")
    project_id = project_raw if isinstance(project_raw, str) and project_raw else None
    shared = _shell(book, flash)
    if project_id is None:
        return shared
    try:
        videos = storage.list_videos(workspace, project_id)
    except LabError as exc:
        shared["list_error"] = exc.message
        return shared
    if not videos:
        shared["no_video"] = True
        return shared
    video = _choose_video(request, videos)
    return _filled(request, workspace, project_id, video, videos, book, flash, shared)


def encode_rgb_jpeg(image: np.ndarray) -> bytes:
    """One JPEG from an upright RGB array.

    OpenCV's JPEG encoder reads BGR. This writer keeps channel 0 as red.
    """
    array = np.ascontiguousarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected upright uint8 RGB")
    buffer = BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


async def post_analysis_start(request: Request) -> RedirectResponse:
    from excavation_lab.web.label_routes import release_suggestion_adapter

    release_suggestion_adapter(request.app)
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    form = await request.form()
    video = _posted_video(workspace, project_id, form)
    mode = _posted_mode(form)
    if project_id is None or video is None:
        _remember(request, {"error": book.need_project})
        return _redirect(mode=mode)
    draft = _draft_from_form(form)
    if draft is None:
        _remember(request, {"error": book.thresholds_unset})
        return _redirect(video_id=video["video_id"], mode=mode)
    outcome = analysis_page.request_analysis(
        workspace,
        project_id,
        str(video["video_id"]),
        _model_id(workspace, project_id, form, mode),
        _analysis_fps(form),
        _posted_number(_field(form, "start_s")),
        _posted_number(_field(form, "end_s")),
        view_id=_view_id(form),
        thresholds=draft,
        text=book,
    )
    _remember(request, _attempt_payload(outcome))
    return _redirect(video_id=str(video["video_id"]), mode=mode)


async def post_analysis_export(request: Request) -> RedirectResponse:
    """Toggles and zone rows. They are not sent to ``request_analysis``."""
    from excavation_lab.web.label_routes import release_suggestion_adapter

    release_suggestion_adapter(request.app)
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    form = await request.form()
    video = _posted_video(workspace, project_id, form)
    mode = _posted_mode(form)
    run_id = _text(_field(form, "run_id"))
    if project_id is None or video is None:
        _remember(request, {"error": book.need_project})
        return _redirect(mode=mode)
    video_id = str(video["video_id"])
    if not _known_run(workspace, project_id, video_id, run_id):
        _remember(request, {"error": book.export_no_run})
        return _redirect(video_id=video_id, mode=mode, run_id=run_id or None)
    outcome = analysis_page.request_export(
        workspace,
        project_id,
        run_id,
        toggles=_toggles(form),
        zone_rows=_zone_rows(form),
        width=video.get("width"),
        height=video.get("height"),
        text=book,
    )
    _remember(request, _attempt_payload(outcome))
    return _redirect(video_id=video_id, mode=mode, run_id=run_id, view_id=_view_id(form))


async def post_analysis_zones(request: Request) -> RedirectResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    form = await request.form()
    video = _posted_video(workspace, project_id, form)
    mode = _posted_mode(form)
    view_id = _view_id(form) or ""
    video_id = None if video is None else str(video["video_id"])
    if project_id is None or video is None:
        _remember(request, {"error": book.need_project})
        return _redirect(mode=mode)
    if not view_id:
        _remember(request, {"error": book.zone_view_required})
        return _redirect(video_id=video_id, mode=mode)
    try:
        analysis_page.save_zone_table(
            workspace,
            project_id,
            view_id,
            _zone_rows(form),
            book,
        )
    except LabError as exc:
        _remember(request, {"error": exc.message})
        return _redirect(video_id=video_id, mode=mode, view_id=view_id)
    _remember(request, {"notice": book.zone_saved})
    return _redirect(video_id=video_id, mode=mode, view_id=view_id)


async def post_analysis_threshold(request: Request) -> RedirectResponse:
    """Validate the draft. The finished run's parquet stays where it is."""
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    form = await request.form()
    video = _posted_video(workspace, project_id, form)
    mode = _posted_mode(form)
    run_id = _text(_field(form, "run_id")) or None
    video_id = None if video is None else str(video["video_id"])
    draft = _draft_from_form(form)
    if project_id is None or video is None:
        _remember(request, {"error": book.need_project})
        return _redirect(mode=mode)
    if draft is None:
        _remember(request, {"error": book.thresholds_unset})
        return _redirect(video_id=video_id, mode=mode, run_id=run_id)
    finished = _run_directory(workspace, project_id, video_id, run_id)
    message = analysis_page.note_threshold_edit(draft, finished_run=finished, text=book)
    key = "notice" if analysis_page._threshold_failure(draft) is None else "error"
    _remember(request, {key: message})
    return _redirect(video_id=video_id, mode=mode, run_id=run_id)


async def get_analysis_frame(request: Request) -> Response:
    """One JPEG for ``video_id`` and ``source_time_s``. A path query is refused."""
    if set(request.query_params.keys()) - _FRAME_QUERY:
        return Response(status_code=400)
    video_id = request.query_params.get("video_id")
    raw_time = request.query_params.get("source_time_s")
    stamp = _posted_number(raw_time)
    if not isinstance(video_id, str) or not video_id or _unsafe_token(video_id) or stamp is None:
        return Response(status_code=400)
    workspace = _workspace(request)
    project_id = _project_id(workspace)
    if project_id is None:
        return Response(status_code=404)
    try:
        video = storage.get_video(workspace, project_id, video_id)
        path = _video_file(workspace, project_id, video)
    except LabError:
        return Response(status_code=404)
    if path is None:
        return Response(status_code=404)
    image = analysis_page.read_source_frame(path, video_id, stamp)
    if image is None:
        return Response(status_code=404)
    try:
        body = encode_rgb_jpeg(image)
    except ValueError:
        return Response(status_code=404)
    return Response(body, media_type="image/jpeg")


async def get_run_video(request: Request) -> Response:
    """Range response for ``playback_file``. Does not walk source frames."""
    project_id = str(request.path_params.get("project_id", ""))
    run_id = str(request.path_params.get("run_id", ""))
    raw_path = str(request.scope.get("path", ""))
    if "path" in request.query_params or traversal(raw_path, project_id, run_id):
        return Response(status_code=404)
    workspace = _workspace(request)
    found = analysis_page.playback_file(workspace, project_id, run_id)
    if found is None:
        return Response(status_code=404)
    try:
        root = storage.project_directory(workspace, project_id).resolve()
    except LabError:
        return Response(status_code=404)
    resolved = found.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        return Response(status_code=404)
    return FileResponse(resolved, media_type="video/mp4")


def _filled(
    request: Request,
    workspace: Path,
    project_id: str,
    video: storage.VideoRecord,
    videos: Sequence[storage.VideoRecord],
    book: Catalog,
    flash: dict[str, str],
    shared: dict[str, object],
) -> dict[str, object]:
    video_id = str(video["video_id"])
    mode = _selected_mode(request)
    options = analysis_page.list_bundle_options(workspace, project_id)
    bundle_ids = [item.model_id for item in options]
    posted_bundle = _text(request.query_params.get("bundle"))
    bundle = posted_bundle if posted_bundle in bundle_ids else (bundle_ids[0] if bundle_ids else "")
    bundles = [
        {"id": item.model_id, "label": analysis_page.bundle_option_label(item, book)}
        for item in options
    ]
    start, end = analysis_page.interval_defaults(video)
    draft = analysis_page.threshold_draft_from_pilot()
    jobs, list_error = _jobs(workspace)
    if list_error and shared.get("list_error") is None:
        shared["list_error"] = list_error
    runs = analysis_page.list_finished_runs(workspace, project_id, video_id)
    run = _selected_run(request, runs)
    snapshot = None if run is None else analysis_page.load_snapshot(run.directory)
    times = [] if snapshot is None else analysis_page._analyzed_times(snapshot.frame_rows)
    selected_time = _selected_time(request, times)
    header = analysis_page.analysis_header(snapshot, selected_time, book)
    enabled = analysis_page.start_enabled([job.status for job in jobs])
    blocking = next(
        (job for job in jobs if not analysis_page.start_enabled([job.status])),
        None,
    )
    view_id = _selected_view(request, video)
    zone_rows = analysis_page._initial_zone_rows(workspace, project_id, view_id)
    review = _review(workspace, project_id, video, snapshot, selected_time)
    play_url = ""
    if run is not None and analysis_page.playback_file(workspace, project_id, run.run_id):
        play_url = f"/runs/{project_id}/{run.run_id}/video"
    preview = _zone_preview(zone_rows, video, book)
    body = dict(shared)
    body.update(
        {
            "show_empty": False,
            "no_video": False,
            "show_form": True,
            "header_lines": analysis_page.header_lines(header, book),
            "selected_video_id": video_id,
            "video_options": _video_options(videos, video_id),
            "mode": mode,
            "bundles": bundles,
            "bundle": bundle,
            "weights_warning": _weights_warning(workspace, book, mode),
            "interval_known": start is not None and end is not None,
            "start_value": _shown_number(start),
            "end_value": _shown_number(end),
            "fps_value": _shown_number(load_pilot().analysis_fps),
            "high_value": _shown_number(draft.high_confidence),
            "activation_value": _shown_number(draft.activation),
            "lost_value": _shown_number(draft.lost_buffer_s),
            "min_value": str(draft.min_observations),
            "display_value": _shown_number(draft.display_threshold),
            "display_floor": _shown_number(load_pilot().detection_floor),
            "view_id": view_id,
            "zone_names": list(ZONE_NAMES),
            "zone_rows": _zone_fields(zone_rows),
            "zone_preview_url": preview["url"],
            "zone_preview_error": preview["error"],
            "zone_size_unknown": preview["size_unknown"],
            "toggles": [
                {"name": name, "label": getattr(book, label)} for name, label in _TOGGLE_FIELDS
            ],
            "start_enabled": enabled,
            "export_enabled": enabled and run is not None,
            "blocked_message": None if enabled else book.job_already_running,
            "run_options": [
                {"id": item.run_id, "selected": run is not None and item.run_id == run.run_id}
                for item in runs
            ],
            "run_id": "" if run is None else run.run_id,
            "time_options": _time_options(times, selected_time, book),
            "frame_list_missing": snapshot is not None and not times,
            "play_url": play_url,
            "player_waiting": None if play_url else book.player_waiting,
            "baseline_run": run is not None and run.model_origin == _BASELINE_MODEL_ID,
            "poll": blocking is not None,
            "status_tone": status_tone(blocking.status.value if blocking is not None else None),
            "status_value": blocking.status.value if blocking is not None else "",
            "status_job_id": blocking.job_id if blocking is not None else "",
            "status_line": (
                job_status_line(blocking, book) if blocking is not None else ""
            ),
            "job_rows": [
                {"tone": status_tone(job.status.value), "line": job_status_line(job, book)}
                for job in jobs
                if blocking is None or job.job_id != blocking.job_id
            ],
            "analysis_time_cookie": (
                None if selected_time is None else format(float(selected_time), "g")
            ),
            "analysis_mode_cookie": mode,
            "analysis_run_cookie": None if run is None else run.run_id,
            "analysis_view_cookie": view_id or None,
        }
    )
    body.update(review)
    return body


def _shell(book: Catalog, flash: dict[str, str]) -> dict[str, object]:
    header = analysis_page.analysis_header(None, text=book)
    return {
        "book": book,
        "flash_error": flash.get("error"),
        "flash_notice": flash.get("notice"),
        "list_error": None,
        "show_empty": True,
        "no_video": False,
        "show_form": False,
        "header_lines": analysis_page.header_lines(header, book),
        "selected_video_id": None,
        "analysis_time_cookie": None,
        "analysis_mode_cookie": None,
        "analysis_run_cookie": None,
        "analysis_view_cookie": None,
    }


def _review(
    workspace: Path,
    project_id: str,
    video: storage.VideoRecord,
    snapshot: analysis_page.RunSnapshot | None,
    source_time_s: float | None,
) -> dict[str, object]:
    if snapshot is None or source_time_s is None:
        return {
            "review_ready": False,
            "frame_url": "",
            "detection_rows": [],
            "predicted_rows": [],
        }
    found = analysis_page.review_frame(
        snapshot.detection_rows,
        snapshot.track_rows,
        source_time_s,
    )
    path = _video_file(workspace, project_id, video)
    image = None if path is None else analysis_page.read_source_frame(
        path,
        str(video["video_id"]),
        source_time_s,
    )
    frame_url = ""
    if image is not None:
        # Ids and the formatted stamp have no query metacharacters.
        frame_url = (
            "/analysis/frame?video_id="
            + str(video["video_id"])
            + "&source_time_s="
            + format(float(source_time_s), "g")
        )
    return {
        "review_ready": True,
        "frame_url": frame_url,
        "detection_rows": list(found.detections),
        "predicted_rows": list(found.predicted),
    }


def _zone_preview(
    rows: Sequence[Mapping[str, object]],
    video: Mapping[str, object],
    book: Catalog,
) -> dict[str, object]:
    width = analysis_page._positive_int(video.get("width"))
    height = analysis_page._positive_int(video.get("height"))
    if width is None or height is None:
        return {"url": "", "error": None, "size_unknown": book.zone_size_unknown}
    if analysis_page._out_of_range(rows):
        return {"url": "", "error": book.zone_coordinate_preview, "size_unknown": None}
    image = analysis_page.zone_preview_rgb(rows, width=width, height=height)
    if image is None:
        return {"url": "", "error": None, "size_unknown": None}
    try:
        encoded = base64.standard_b64encode(encode_rgb_jpeg(image)).decode("ascii")
    except ValueError:
        return {"url": "", "error": None, "size_unknown": None}
    return {"url": f"data:image/jpeg;base64,{encoded}", "error": None, "size_unknown": None}


def _zone_fields(rows: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    found = [
        {
            "zone": str(row["zone"]),
            "x": _shown_number(row.get("x")),
            "y": _shown_number(row.get("y")),
        }
        for row in rows
        if isinstance(row.get("zone"), str)
    ]
    blanks = 4 if len(found) < 4 else 1
    found.extend({"zone": "", "x": "", "y": ""} for _ in range(blanks))
    return found


def _time_options(
    times: Sequence[float],
    selected: float | None,
    book: Catalog,
) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    for stamp in times:
        options.append(
            {
                "value": format(float(stamp), "g"),
                "label": analysis_page._format_time(stamp, book),
                "selected": selected is not None and abs(float(stamp) - selected) <= _TIME_EPS,
            }
        )
    return options


def _weights_warning(workspace: Path, book: Catalog, mode: str) -> str | None:
    if mode == "custom":
        return None
    if official_nano_weights_path(workspace).is_file():
        return None
    return book.weights_warning(weights_cache_dir(workspace))


def _jobs(workspace: Path) -> tuple[list[JobRecord], str | None]:
    try:
        return list(reopen(workspace)), None
    except LabError as exc:
        return [], exc.message


def _video_options(
    videos: Sequence[storage.VideoRecord],
    selected_id: str,
) -> list[dict[str, object]]:
    if len(videos) < 2:
        return []
    return [
        {
            "id": item["video_id"],
            "label": video_choice_label(item),
            "selected": item["video_id"] == selected_id,
        }
        for item in videos
    ]


def _choose_video(
    request: Request,
    videos: Sequence[storage.VideoRecord],
) -> storage.VideoRecord:
    wanted = _token(request.query_params.get("video")) or _token(request.cookies.get(_VIDEO_COOKIE))
    if wanted:
        for item in videos:
            if item["video_id"] == wanted:
                return item
    return videos[0]


def _selected_mode(request: Request) -> str:
    query = _text(request.query_params.get("mode"))
    if query in _MODES:
        return query
    cookie = _text(request.cookies.get(_MODE_COOKIE))
    if cookie in _MODES:
        return cookie
    return "baseline"


def _selected_run(
    request: Request,
    runs: Sequence[analysis_page.FinishedRun],
) -> analysis_page.FinishedRun | None:
    if not runs:
        return None
    wanted = _token(request.query_params.get("run")) or _token(request.cookies.get(_RUN_COOKIE))
    if wanted:
        for item in runs:
            if item.run_id == wanted:
                return item
    return runs[-1]


def _selected_time(request: Request, times: Sequence[float]) -> float | None:
    if not times:
        return None
    raw = request.query_params.get("source_time_s")
    if not isinstance(raw, str) or not raw.strip():
        raw = request.cookies.get(_TIME_COOKIE)
    matched = _match_time(raw, times)
    if matched is not None:
        return matched
    return float(times[0])


def _selected_view(request: Request, video: Mapping[str, object]) -> str:
    query = _token(request.query_params.get("view"))
    if query is not None:
        return query
    cookie = _token(request.cookies.get(_VIEW_COOKIE))
    if cookie is not None:
        return cookie
    stored = video.get("view_id")
    if isinstance(stored, str) and stored.strip() and not _unsafe_token(stored.strip()):
        return stored.strip()
    return ""


def _match_time(raw: object, times: Sequence[float]) -> float | None:
    number = _posted_number(raw)
    if number is None:
        return None
    for stamp in times:
        if abs(float(stamp) - number) <= _TIME_EPS:
            return float(stamp)
    return None


def _model_id(workspace: Path, project_id: str, form: object, mode: str) -> str | None:
    if mode != "custom":
        return _BASELINE_MODEL_ID
    bundle = _text(_field(form, "bundle"))
    if bundle and bundle in analysis_page.list_bundles(workspace, project_id):
        return bundle
    return None


def _draft_from_form(form: object) -> analysis_page.ThresholdDraft | None:
    high = _posted_number(_field(form, "high_confidence"))
    activation = _posted_number(_field(form, "activation"))
    lost = _posted_number(_field(form, "lost_buffer_s"))
    minimum = _posted_number(_field(form, "min_observations"))
    display = _posted_number(_field(form, "display_threshold"))
    if None in {high, activation, lost, minimum, display}:
        return None
    assert high is not None
    assert activation is not None
    assert lost is not None
    assert minimum is not None
    assert display is not None
    return analysis_page.ThresholdDraft(
        high_confidence=high,
        activation=activation,
        lost_buffer_s=lost,
        reference_fps=load_pilot().bytetrack.reference_fps,
        min_observations=int(minimum),
        display_threshold=display,
    )


def _analysis_fps(form: object) -> float:
    parsed = _posted_number(_field(form, "analysis_fps"))
    if parsed is None or parsed <= 0:
        return float(load_pilot().analysis_fps)
    return parsed


def _toggles(form: object) -> AnnotateToggles:
    flags = {name: _flag(form, name) for name, _label in _TOGGLE_FIELDS}
    return AnnotateToggles(
        boxes=flags["boxes"],
        class_names=flags["class_names"],
        scores=flags["scores"],
        track_ids=flags["track_ids"],
        bucket_trail=flags["bucket_trail"],
        zones=flags["zones"],
    )


def _zone_rows(form: object) -> list[dict[str, object]]:
    names = _strings(form, "zone")
    xs = _strings(form, "x")
    ys = _strings(form, "y")
    rows: list[dict[str, object]] = []
    for name, raw_x, raw_y in zip(names, xs, ys, strict=False):
        x = _posted_number(raw_x)
        y = _posted_number(raw_y)
        if x is None or y is None:
            continue
        rows.append({"zone": name, "x": x, "y": y})
    return analysis_page.rows_from_editor(rows)


def _posted_video(
    workspace: Path,
    project_id: str | None,
    form: object,
) -> storage.VideoRecord | None:
    if project_id is None:
        return None
    try:
        videos = storage.list_videos(workspace, project_id)
    except LabError:
        return None
    if not videos:
        return None
    posted = _text(_field(form, "video_id"))
    if posted:
        for item in videos:
            if item["video_id"] == posted:
                return item
        return None
    if len(videos) == 1:
        return videos[0]
    return None


def _known_run(workspace: Path, project_id: str, video_id: str, run_id: str) -> bool:
    if not run_id:
        return False
    return any(
        item.run_id == run_id
        for item in analysis_page.list_finished_runs(workspace, project_id, video_id)
    )


def _run_directory(
    workspace: Path,
    project_id: str,
    video_id: str | None,
    run_id: str | None,
) -> Path | None:
    if video_id is None or not run_id:
        return None
    for item in analysis_page.list_finished_runs(workspace, project_id, video_id):
        if item.run_id == run_id:
            return item.directory
    return None


def _video_file(
    workspace: Path,
    project_id: str,
    video: Mapping[str, object],
) -> Path | None:
    raw = video.get("path")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return storage.resolve_project_path(workspace, project_id, Path(raw))
    except LabError:
        return None


def _attempt_payload(attempt: analysis_page.AnalysisAttempt) -> dict[str, str]:
    payload: dict[str, str] = {}
    if attempt.error:
        payload["error"] = attempt.error
    if attempt.notice:
        text = attempt.notice
        if attempt.job_id:
            text = f"{text} {attempt.job_id}."
        payload["notice"] = text
    return payload


def _view_id(form: object) -> str | None:
    token = _text(_field(form, "view_id"))
    if not token or _unsafe_token(token):
        return None
    return token


def _posted_mode(form: object) -> str | None:
    value = _text(_field(form, "mode"))
    if value in _MODES:
        return value
    return None


def _flag(form: object, name: str) -> bool:
    return _text(_field(form, name)) in {"1", "on", "true", "yes"}


def _posted_number(value: object) -> float | None:
    """A finite number. Empty stays missing. ``0`` stays ``0``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def _shown_number(value: object) -> str:
    number = _posted_number(value)
    if number is None:
        return ""
    return format(number, "g")


def _strings(form: object, name: str) -> list[str]:
    getter = getattr(form, "getlist", None)
    if not callable(getter):
        return []
    return [item for item in getter(name) if isinstance(item, str)]


def _field(form: object, name: str) -> object:
    getter = getattr(form, "get", None)
    if not callable(getter):
        return None
    return getter(name)


def _text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or _unsafe_token(text):
        return None
    return text


def _unsafe_token(value: str) -> bool:
    if value in {".", ".."}:
        return True
    return any(char in value for char in "/\\\x00\r\n;")


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


def _remember(request: Request, payload: dict[str, str]) -> None:
    try:
        write_flash(_workspace(request), payload)
    except Exception:  # noqa: BLE001 — the page still has to show the sentence
        text = payload.get("error")
        if isinstance(text, str) and text:
            request.app.state.pending_notice = text


def _redirect(
    *,
    video_id: str | None = None,
    mode: str | None = None,
    run_id: str | None = None,
    view_id: str | None = None,
) -> RedirectResponse:
    response = RedirectResponse("/analysis", status_code=303)
    _stamp(response, _VIDEO_COOKIE, video_id)
    _stamp(response, _MODE_COOKIE, mode if mode in _MODES else None)
    _stamp(response, _RUN_COOKIE, run_id)
    _stamp(response, _VIEW_COOKIE, view_id)
    return response


def _stamp(response: RedirectResponse, name: str, value: str | None) -> None:
    if not isinstance(value, str) or not value or _unsafe_token(value):
        return
    response.set_cookie(name, value, path="/", httponly=True, samesite="lax")
