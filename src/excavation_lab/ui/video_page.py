"""«Видео». Load a local file, play it, start baseline.

The page calls ``add_video``, ``reprobe_video``, and ``enqueue_analyze``.
Probe data stays in the media service. This module does not run ffprobe
itself and does not construct a detector. A second baseline click hits the
workspace job lock. Routes import these helpers. ``render`` remains for the
Streamlit entry until that UI is removed.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from excavation_lab import storage
from excavation_lab.errors import (
    JobAlreadyRunningError,
    LabError,
    MediaNotFoundError,
    PathNotAllowedError,
    TimebaseError,
    map_os_error,
)
from excavation_lab.hardware import official_nano_weights_path
from excavation_lab.jobs import reopen
from excavation_lab.media.probe import add_video, format_probe_line, reprobe_video
from excavation_lab.media.timebase import require_half_open_interval
from excavation_lab.schemas import JobRecord, JobStatus, load_pilot
from excavation_lab.tracking.analyze import enqueue_analyze
from excavation_lab.ui import copy as ui_copy

# A MOV/MKV above this size is played from the short preview. An MP4 does
# not use the bound: ``player_file`` serves the registered file.
_DIRECT_PLAYBACK_MAX_BYTES = 200 * 1024 * 1024
_PREVIEW_WINDOW_S = 2.0
_PREVIEW_FPS = 5.0
_PREVIEW_FRAME_CAP = 8
_BASELINE_MODEL_ID = "baseline_coco"
_BLOCKING = frozenset(
    {
        JobStatus.queued,
        JobStatus.running,
        JobStatus.cancel_requested,
    }
)
_UPLOAD_SUFFIXES = frozenset({".mp4", ".mov", ".mkv"})
_UPLOAD_FORBIDDEN = frozenset('<>:"|?*')
_UPLOAD_STEM_LIMIT = 180
_MIME = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
}


@dataclass(frozen=True, slots=True)
class BaselineAttempt:
    """Result of one baseline click. ``job_id`` is set only after the lock."""

    job_id: str | None = None
    notice: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PlayerState:
    """One video response. ``shown`` is the stored row, including a mismatch."""

    videos: tuple[Mapping[str, object], ...] = ()
    selected_id: str | None = None
    shown: Mapping[str, object] | None = None
    probe_note: str | None = None
    list_error: str | None = None
    playable: Path | None = None
    caption: str | None = None
    play_error: str | None = None
    metadata: tuple[str, ...] = ()
    rotation: str | None = None
    offset: str | None = None
    interval_start: float | None = None
    interval_end: float | None = None
    blocked: bool = False
    jobs: tuple[JobRecord, ...] = ()

    @property
    def has_video(self) -> bool:
        return self.shown is not None and self.list_error is None


def allowed_directory_caption(
    workspace: Path,
    project_id: str,
    text: ui_copy.Catalog | None = None,
) -> str:
    """Name the directories ``allowed_media_roots`` returns.

    The workspace root is the primary allowed directory. Extra roots are
    named by their folder. The page does not fetch a remote clip.
    No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    roots = storage.allowed_media_roots(workspace, project_id)
    base = Path(workspace).resolve()
    extras = [
        Path(root).resolve().name
        for root in roots
        if Path(root).resolve() != base
    ]
    if not extras:
        return catalog.allowed_directory
    return catalog.allowed_directory_extras(*extras)


def metadata_lines(
    video: Mapping[str, object],
    device: object,
    text: ui_copy.Catalog | None = None,
) -> list[str]:
    """Probe fields. A missing measurement is «не рассчитано», not zero.

    No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    nominal = _fps_text(video.get("fps_nominal"), catalog)
    average = _fps_text(video.get("fps_average"), catalog)
    return [
        f"{catalog.meta_duration}: {_duration_text(video.get('duration_s'), catalog)}.",
        (
            f"{catalog.meta_fps}: {catalog.meta_fps_nominal} {nominal}, "
            f"{catalog.meta_fps_average} {average}."
        ),
        (
            f"{catalog.meta_frame}: "
            f"{_frame_text(video.get('width'), video.get('height'), catalog)}."
        ),
        f"{catalog.meta_codec}: {_known_text(video.get('codec'), catalog)}.",
        f"{catalog.meta_sha}: {_known_text(video.get('sha256'), catalog)}.",
        f"{catalog.meta_device}: {_device_text(device, catalog)}.",
    ]


def rotation_line(
    video: Mapping[str, object],
    text: ui_copy.Catalog | None = None,
) -> str:
    catalog = _resolved(text)
    return (
        f"{catalog.rotation_label}: "
        f"{_rotation_text(video.get('rotation_deg'), catalog)}."
    )


def interval_bounds(video: Mapping[str, object]) -> tuple[float | None, float | None]:
    """Half-open source window ``[start_offset, start_offset + duration)``.

    ``duration_s`` is the container length, not the timestamp of the last
    frame. A file whose offset is 300 and whose duration is 10 runs from
    300 to 310. A missing or negative offset is not a widget default:
    ``number_input`` rejects a value below ``min_value``.
    """
    start = _known_offset(video.get("start_offset_s"))
    duration = _known_positive(video.get("duration_s"))
    if start is None or duration is None:
        return start, None
    return start, start + duration


def offset_line(
    video: Mapping[str, object],
    text: ui_copy.Catalog | None = None,
) -> str:
    """Measured ``start_offset_s``. Missing stays «не рассчитано»; 0 stays 0.

    No catalog argument stays Russian (``300 с``).
    """
    catalog = _resolved(text)
    raw = video.get("start_offset_s")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        shown = catalog.not_calculated
    else:
        number = float(raw)
        shown = (
            _seconds(number, catalog, precise=False)
            if math.isfinite(number)
            else catalog.not_calculated
        )
    return f"{catalog.offset_label}: {shown}."


def video_choice_label(video: Mapping[str, object]) -> str:
    """Selectbox text. ``video_id`` keeps two files with one name apart."""
    raw_name = video.get("name")
    name = raw_name.strip() if isinstance(raw_name, str) else ""
    raw_id = video.get("video_id")
    video_id = raw_id.strip() if isinstance(raw_id, str) else ""
    if name and video_id:
        return f"{name} ({video_id})"
    return name or video_id


def register_chosen_file(workspace: Path, project_id: str, path: Path) -> str:
    """Register a chosen file in place and return its video id.

    A file outside the allowed roots appends its parent to the project's
    extra roots. If that append is committed and ``add_video`` does not
    return a new row, the previous extra-root list is restored. The
    filesystem root is refused. The file is not copied. A path that is
    already registered returns that video id.
    """
    resolved = Path(path).resolve()
    if resolved.suffix.lower() not in _UPLOAD_SUFFIXES:
        raise LabError("invalid_video", ui_copy.UPLOAD_BAD_NAME)
    if not resolved.is_file():
        raise MediaNotFoundError(str(resolved))
    previous_roots: list[Path] | None = None
    try:
        storage.resolve_project_path(workspace, project_id, resolved)
    except PathNotAllowedError:
        parent = resolved.parent
        # ``/`` as an extra root would allow every path on the machine.
        if parent == Path(parent.anchor):
            raise
        current = storage.allowed_media_roots(workspace, project_id)
        remembered = current[1:]
        storage.replace_extra_roots(workspace, project_id, [*remembered, parent])
        previous_roots = remembered
    try:
        added = add_video(workspace, project_id, resolved, copy=False)
    except Exception as exc:
        if isinstance(exc, LabError) and exc.code == "duplicate_video":
            existing = _video_id_at_path(workspace, project_id, resolved)
            if existing is not None:
                return existing
        if previous_roots is not None:
            storage.replace_extra_roots(workspace, project_id, previous_roots)
        raise
    return added["video_id"]


def _video_id_at_path(workspace: Path, project_id: str, resolved: Path) -> str | None:
    target = str(resolved)
    for video in storage.list_videos(workspace, project_id):
        raw = video["path"]
        if not raw:
            continue
        if raw == target or Path(raw).resolve() == resolved:
            return video["video_id"]
    return None


def upload_rejection(size: int, text: ui_copy.Catalog | None = None) -> str | None:
    """Empty browser upload. No write. Size is not capped here.

    No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        return catalog.upload_empty
    return None


def store_browser_upload(
    workspace: Path,
    project_id: str,
    filename: str,
    payload: bytes,
    text: ui_copy.Catalog | None = None,
) -> Mapping[str, object]:
    """Write one browser upload into project media, then ``add_video``."""
    catalog = _resolved(text)
    rejection = upload_rejection(len(payload), catalog)
    if rejection is not None:
        raise LabError("upload_rejected", rejection)
    leaf = _upload_leaf(filename, catalog)
    directory = storage.project_directory(workspace, project_id) / "media"
    destination = directory / leaf
    stored = storage.resolve_project_path(workspace, project_id, destination)
    if stored.exists():
        raise LabError("duplicate_video", catalog.duplicate_video_message())
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with map_os_error(str(stored)):
            stored.write_bytes(payload)
        return add_video(workspace, project_id, stored)
    except Exception:
        stored.unlink(missing_ok=True)
        raise  # registration failed; the upload must not stay in media


def request_baseline_run(
    workspace: Path,
    project_id: str,
    video_id: str,
    start_s: object,
    end_s: object,
    text: ui_copy.Catalog | None = None,
) -> BaselineAttempt:
    """Queue ``baseline_coco`` through the job lock.

    Does not construct a detector. Missing official weights return an
    empty attempt and do not enqueue. A running job returns the lock
    error instead of a second ``enqueue_analyze``.
    No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    try:
        start, end = _interval(start_s, end_s, catalog)
    except TimebaseError as exc:
        return BaselineAttempt(error=exc.message)
    if job_blocks(workspace):
        return BaselineAttempt(error=JobAlreadyRunningError().message)
    if not official_nano_weights_path(workspace).is_file():
        return BaselineAttempt()
    try:
        fps = load_pilot().analysis_fps
        enqueued = enqueue_analyze(
            workspace,
            project_id,
            video_id,
            _BASELINE_MODEL_ID,
            fps,
            start_s=start,
            end_s=end,
        )
    except LabError as exc:
        return BaselineAttempt(error=exc.message)
    return BaselineAttempt(job_id=enqueued.job_id)


def job_blocks(workspace: Path) -> bool:
    """True while a queued, running, or cancel-requested job holds the lock."""
    return any(job.status in _BLOCKING for job in reopen(workspace))


def playback_path(
    workspace: Path,
    project_id: str,
    video: storage.VideoRecord,
    text: ui_copy.Catalog | None = None,
) -> Path:
    """File the player may open.

    Streamlit's media store reads a local file into RAM. A source at or
    under the direct-playback bound is played from disk. A longer file
    stays on disk and the player receives a short CFR preview.
    """
    catalog = _resolved(text)
    source = _source_path(video)
    size = video["size_bytes"]
    if (
        source.is_file()
        and isinstance(size, int)
        and not isinstance(size, bool)
        and 0 < size <= _DIRECT_PLAYBACK_MAX_BYTES
    ):
        return source
    return _short_preview(workspace, project_id, video, source, catalog)


def player_file(
    workspace: Path,
    project_id: str,
    video: storage.VideoRecord,
    text: ui_copy.Catalog | None = None,
) -> tuple[Path, str]:
    """Playable path and caption kind.

    An ``.mp4`` is the registered file. ``playback_path`` and
    ``write_preview`` are not called. Any other suffix is whatever
    ``playback_path`` already returns: the source (``playback_file``) or
    the short preview (``playback_preview``).
    """
    catalog = _resolved(text)
    source = _source_path(video)
    if source.suffix.lower() == ".mp4":
        return source, "playback_file"
    # An escaping symlink must fail here. playback_path opens the file and
    # can write a preview; that must not run for a path outside the roots.
    storage.resolve_project_path(workspace, project_id, source)
    playable = playback_path(workspace, project_id, video, catalog)
    if playable.resolve() == source.resolve():
        return playable, "playback_file"
    return playable, "playback_preview"


def playback_device() -> str | None:
    """Device string already shown by ``metadata_lines``. No detector."""
    from excavation_lab import hardware as machine

    try:
        return machine.select_inference_device()
    except LabError:
        return None


def shown_video(
    workspace: Path,
    project_id: str,
    video: Mapping[str, object],
) -> tuple[Mapping[str, object], str | None]:
    """Re-probe once. A mismatch is reported and the stored row stays."""
    video_id = video.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        return video, None
    try:
        report = reprobe_video(workspace, project_id, video_id)
    except LabError as exc:
        return video, exc.message
    note = format_probe_line(report) if report["mismatch"] else None
    return report["video"], note


def assemble_player(
    workspace: Path,
    project_id: str | None,
    catalog: ui_copy.Catalog,
    selected_id: str | None,
) -> PlayerState:
    """Metadata, player, and runs for one GET. Re-probe runs once."""
    if project_id is None:
        return PlayerState()
    try:
        found = storage.list_videos(workspace, project_id)
    except LabError as exc:
        return PlayerState(list_error=exc.message)
    if not found:
        return PlayerState()
    selected = _pick_video(found, selected_id)
    shown, note = shown_video(workspace, project_id, selected)
    playable: Path | None = None
    caption: str | None = None
    play_error: str | None = None
    try:
        path, kind = player_file(workspace, project_id, shown, catalog)  # type: ignore[arg-type]
    except LabError as exc:
        play_error = exc.message
        caption = catalog.playback_failed
    else:
        if path.is_file():
            try:
                storage.resolve_project_path(workspace, project_id, path)
            except LabError as exc:
                play_error = exc.message
                caption = catalog.playback_failed
            else:
                playable = path
                caption = getattr(catalog, kind)
        else:
            play_error = catalog.playback_failed
    start, end = interval_bounds(shown)
    raw_id = shown.get("video_id")
    chosen = raw_id if isinstance(raw_id, str) and raw_id else None
    return PlayerState(
        videos=tuple(found),
        selected_id=chosen,
        shown=shown,
        probe_note=note,
        playable=playable,
        caption=caption,
        play_error=play_error,
        metadata=tuple(metadata_lines(shown, playback_device(), catalog)),
        rotation=rotation_line(shown, catalog),
        offset=offset_line(shown, catalog),
        interval_start=start,
        interval_end=end,
        blocked=job_blocks(workspace) or not official_nano_weights_path(workspace).is_file(),
        jobs=tuple(reopen(workspace)),
    )


def job_status_line(job: JobRecord, text: ui_copy.Catalog | None = None) -> str:
    """Status text from the job row. No boxes and no invented end time.

    No catalog argument stays Russian. A missing status is the catalog's
    not-calculated phrase.
    """
    catalog = _resolved(text)
    label = catalog.job_status_labels.get(job.status.value, catalog.not_calculated)
    parts = [f"{job.job_id}: {label}"]
    if job.orphaned:
        parts.append(catalog.process_lost)
    if job.partial:
        parts.append(catalog.partial_result)
    end = _known_positive(job.analyzed_end_s)
    if end is not None:
        parts.append(_seconds(end, catalog, precise=True))
    if job.error_code:
        parts.append(job.error_code)
    return ". ".join(parts) + "."


def _resolved(text: ui_copy.Catalog | None) -> ui_copy.Catalog:
    if text is not None:
        return text
    return ui_copy.catalog("ru")


def _listed_videos(
    workspace: Path,
    project_id: str | None,
) -> tuple[list[storage.VideoRecord], str | None]:
    if project_id is None:
        return [], None
    try:
        return storage.list_videos(workspace, project_id), None
    except LabError as exc:
        return [], exc.message


def _pick_video(
    videos: list[storage.VideoRecord],
    selected_id: str | None,
) -> storage.VideoRecord:
    if isinstance(selected_id, str) and selected_id:
        for item in videos:
            if item["video_id"] == selected_id:
                return item
    return videos[0]


def _interval(
    start_s: object,
    end_s: object,
    catalog: ui_copy.Catalog,
) -> tuple[float, float]:
    if (
        start_s is None
        or end_s is None
        or isinstance(start_s, bool)
        or isinstance(end_s, bool)
        or not isinstance(start_s, (int, float))
        or not isinstance(end_s, (int, float))
    ):
        raise TimebaseError(catalog.interval_required)
    start = float(start_s)
    end = float(end_s)
    require_half_open_interval(start, end)
    return start, end


def _short_preview(
    workspace: Path,
    project_id: str,
    video: storage.VideoRecord,
    source: Path,
    catalog: ui_copy.Catalog,
) -> Path:
    from excavation_lab.media.encoder import PreviewFrame, write_preview
    from excavation_lab.media.reader import FrameReader
    from excavation_lab.schemas import DecodeStatus

    # Resolve before mkdir and before any open. Path.mkdir follows a symlink,
    # so media/ must already be the real directory inside the allowed roots.
    resolved_source = storage.resolve_project_path(workspace, project_id, source)
    media = storage.resolve_project_path(
        workspace,
        project_id,
        storage.project_directory(workspace, project_id) / "media",
    )
    if not resolved_source.is_file():
        raise LabError("media_not_found", catalog.playback_failed)
    start = interval_bounds(video)[0]
    duration = _known_positive(video.get("duration_s"))
    if start is None or duration is None:
        raise LabError("invalid_video", catalog.playback_failed)
    end = start + min(duration, _PREVIEW_WINDOW_S)
    require_half_open_interval(start, end)
    directory = media / "previews"
    # mkdir(exist_ok=True) follows an existing directory symlink. Resolve
    # that child first so a previews link outside the roots gets no files.
    if directory.exists() or directory.is_symlink():
        directory = storage.resolve_project_path(workspace, project_id, directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{video['video_id']}.mp4"
    map_path = directory / f"{video['video_id']}.csv"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    reader = FrameReader(resolved_source, video_id=video["video_id"])
    kept: list[PreviewFrame] = []
    step = 1.0 / _PREVIEW_FPS
    next_tick = start
    for frame in reader.iterate(start, end):
        stamp = frame.source_time_s
        if frame.decode_status is not DecodeStatus.ok or stamp is None:
            continue
        if stamp + 1e-6 < next_tick:
            continue
        kept.append(PreviewFrame.from_read_frame(frame))
        next_tick = stamp + step
        if len(kept) >= _PREVIEW_FRAME_CAP:
            break
    if not kept:
        raise LabError("preview_empty", catalog.playback_failed)
    temporary = directory / f".{video['video_id']}.part.mp4"
    temporary_map = directory / f".{video['video_id']}.part.csv"
    try:
        write_preview(
            kept,
            temporary,
            output_fps=_PREVIEW_FPS,
            start_s=start,
            end_s=end,
            map_path=temporary_map,
        )
        os.replace(temporary, destination)
        os.replace(temporary_map, map_path)
    except LabError:
        temporary.unlink(missing_ok=True)
        temporary_map.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        temporary_map.unlink(missing_ok=True)
        raise LabError("preview_failed", catalog.playback_failed) from exc
    return destination


def _source_path(video: Mapping[str, object]) -> Path:
    raw = video.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return Path()
    return Path(raw)


def _upload_leaf(filename: str, catalog: ui_copy.Catalog) -> str:
    """One media-folder leaf. Spaces, brackets, and other letters stay.

    Slashes and ``..`` do not. The suffix must be mp4, mov, or mkv.
    """
    if (
        not isinstance(filename, str)
        or not filename
        or filename != Path(filename).name
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or any(ord(char) < 32 for char in filename)
        or any(char in _UPLOAD_FORBIDDEN for char in filename)
    ):
        raise LabError("invalid_video", catalog.upload_bad_name)
    suffix = Path(filename).suffix.lower()
    stem = Path(filename).stem.strip().rstrip(".")
    if suffix not in _UPLOAD_SUFFIXES or not stem or stem in {".", ".."}:
        raise LabError("invalid_video", catalog.upload_bad_name)
    if len(stem) > _UPLOAD_STEM_LIMIT:
        stem = stem[:_UPLOAD_STEM_LIMIT].rstrip(" .")
    if not stem:
        raise LabError("invalid_video", catalog.upload_bad_name)
    return f"{stem}{suffix}"


def _mime(path: Path) -> str:
    return _MIME.get(path.suffix.lower(), "video/mp4")


def _duration_text(value: object, catalog: ui_copy.Catalog) -> str:
    number = _known_positive(value)
    if number is None:
        return catalog.not_calculated
    return _seconds(number, catalog, precise=True)


def _seconds(number: float, catalog: ui_copy.Catalog, *, precise: bool) -> str:
    shown = f"{number:.3f}" if precise else f"{number:g}"
    return f"{shown} {catalog.seconds_suffix}"


def _fps_text(value: object, catalog: ui_copy.Catalog) -> str:
    number = _known_positive(value)
    if number is None:
        return catalog.not_calculated
    return f"{number:g}"


def _frame_text(width: object, height: object, catalog: ui_copy.Catalog) -> str:
    wide = _known_positive(width)
    high = _known_positive(height)
    if wide is None or high is None:
        return catalog.not_calculated
    return f"{int(wide)}×{int(high)}"


def _rotation_text(value: object, catalog: ui_copy.Catalog) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return catalog.not_calculated
    number = float(value)
    if not math.isfinite(number):
        return catalog.not_calculated
    return f"{number:g}°"


def _device_text(device: object, catalog: ui_copy.Catalog) -> str:
    if isinstance(device, str) and device.strip():
        return device.strip()
    return catalog.not_calculated


def _known_text(value: object, catalog: ui_copy.Catalog) -> str:
    if not isinstance(value, str):
        return catalog.not_calculated
    text = value.strip()
    if not text:
        return catalog.not_calculated
    return text


def _known_positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _known_offset(value: object) -> float | None:
    """Finite ``start_offset_s`` at or above 0, including a stored 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number
