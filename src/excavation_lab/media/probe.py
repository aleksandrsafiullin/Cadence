"""ffprobe import. Container metadata only; frames are not decoded.

``source_url`` is stored as provenance. This module does not fetch it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

from excavation_lab.errors import (
    CorruptMediaError,
    FfmpegMissingError,
    LabError,
    MediaNotFoundError,
    PathNotAllowedError,
)
from excavation_lab.storage import (
    VideoRecord,
    allowed_media_roots,
    get_video,
    insert_probed_video,
    project_directory,
    resolve_allowed_path,
    resolve_project_path,
    update_probed_video,
)

# One read is a chunk. The file is never loaded with ``read_bytes``.
_HASH_CHUNK_BYTES = 1024 * 1024
_TIME_BASE_RE = re.compile(r"^[1-9][0-9]*/[1-9][0-9]*$")
_MISMATCH_TITLES = {
    "sha256": "SHA-256",
    "size_bytes": "размер",
}


class VideoMetadata(TypedDict):
    video_id: str
    name: str
    sha256: str
    size_bytes: int
    duration_s: float
    width: int
    height: int
    rotation_deg: float | None
    sar: str | None
    dar: str | None
    codec: str
    fps_nominal: float | None
    fps_average: float | None
    time_base: str
    start_offset_s: float | None
    source_url: str | None
    path: str


class ProbeReport(TypedDict):
    mismatch: bool
    mismatched_fields: list[str]
    video: VideoRecord
    observed: VideoMetadata


def file_sha256(path: Path) -> str:
    """SHA-256 of ``path``, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_executable() -> str:
    """``ffprobe`` from ``PATH``. No hardcoded binary location."""
    found = shutil.which("ffprobe")
    if found is None:
        raise FfmpegMissingError()
    return found


def probe_media(
    path: Path,
    *,
    roots: Sequence[Path],
    video_id: str | None = None,
    source_url: str | None = None,
    name: str | None = None,
) -> VideoMetadata:
    """Probe one file inside ``roots`` and return §6.1 metadata.

    Duration is the container duration from ffprobe. Rotation and SAR stay
    null when the container does not carry them.
    """
    resolved = resolve_allowed_path(path, roots)
    if not resolved.is_file():
        raise MediaNotFoundError(str(resolved))
    payload = _run_ffprobe(resolved)
    parsed = _parse_ffprobe(payload, resolved)
    size_bytes = resolved.stat().st_size
    if size_bytes <= 0:
        raise CorruptMediaError(str(resolved))
    leaf = resolved.name if name is None else name
    if not isinstance(leaf, str) or not leaf.strip() or "\x00" in leaf:
        raise LabError("invalid_video", "Некорректное имя видео.")
    return VideoMetadata(
        video_id=_allocate_video_id(video_id),
        name=leaf,
        sha256=file_sha256(resolved),
        size_bytes=size_bytes,
        duration_s=parsed["duration_s"],
        width=parsed["width"],
        height=parsed["height"],
        rotation_deg=parsed["rotation_deg"],
        sar=parsed["sar"],
        dar=parsed["dar"],
        codec=parsed["codec"],
        fps_nominal=parsed["fps_nominal"],
        fps_average=parsed["fps_average"],
        time_base=parsed["time_base"],
        start_offset_s=parsed["start_offset_s"],
        source_url=_clean_source_url(source_url),
        path=str(resolved),
    )


def add_video(
    workspace: Path,
    project_id: str,
    path: Path,
    *,
    copy: bool = False,
    source_url: str | None = None,
    video_id: str | None = None,
) -> VideoMetadata:
    """Register ``path`` in the project. Copy only when ``copy`` is true.

    ``source_url`` is written to the row and is not requested over the network.
    """
    roots = allowed_media_roots(workspace, project_id)
    source = resolve_allowed_path(path, roots)
    if not source.is_file():
        raise MediaNotFoundError(str(source))
    registered = source
    if copy:
        registered = _copy_into_project_media(workspace, project_id, source)
    try:
        metadata = probe_media(
            registered,
            roots=roots,
            video_id=video_id,
            source_url=source_url,
            name=source.name,
        )
        insert_probed_video(workspace, project_id, metadata)
    except Exception:
        if copy:
            registered.unlink(missing_ok=True)
        raise
    return metadata


def reprobe_video(workspace: Path, project_id: str, video_id: str) -> ProbeReport:
    """Re-probe the registered file. A changed size or SHA-256 is a mismatch.

    A mismatch is reported and the stored row is left as it was. A match
    refreshes the other probe columns.
    """
    stored = get_video(workspace, project_id, video_id)
    raw_path = stored["path"]
    if not raw_path:
        raise MediaNotFoundError()
    observed = probe_media(
        Path(raw_path),
        roots=allowed_media_roots(workspace, project_id),
        video_id=video_id,
        source_url=stored["source_url"],
        name=stored["name"],
    )
    mismatched: list[str] = []
    if stored["size_bytes"] is not None and stored["size_bytes"] != observed["size_bytes"]:
        mismatched.append("size_bytes")
    if stored["sha256"] is not None and stored["sha256"] != observed["sha256"]:
        mismatched.append("sha256")
    if not mismatched:
        update_probed_video(workspace, project_id, observed)
        stored = get_video(workspace, project_id, video_id)
    return ProbeReport(
        mismatch=bool(mismatched),
        mismatched_fields=mismatched,
        video=stored,
        observed=observed,
    )


def format_video_line(metadata: VideoMetadata) -> str:
    fps = metadata["fps_nominal"]
    fps_text = "не рассчитано" if fps is None else f"{fps:g}"
    rotation = metadata["rotation_deg"]
    rotation_text = "не рассчитано" if rotation is None else f"{rotation:g}°"
    return (
        f"Видео «{metadata['name']}» ({metadata['video_id']}). "
        f"Длительность: {metadata['duration_s']:.3f} с. "
        f"Кадры: {fps_text} кадр/с. "
        f"Размер кадра: {metadata['width']}×{metadata['height']}. "
        f"Поворот: {rotation_text}. "
        f"Путь: {metadata['path']}"
    )


def format_probe_line(report: ProbeReport) -> str:
    if not report["mismatch"]:
        return format_video_line(report["observed"])
    titles = [_MISMATCH_TITLES[field] for field in report["mismatched_fields"]]
    joined = " и ".join(titles)
    return (
        f"Файл изменился: не совпадают {joined}. "
        "Зарегистрированные метаданные не перезаписаны. "
        f"Путь: {report['observed']['path']}"
    )


def ffprobe_json(path: Path) -> dict[str, Any]:
    """Run ffprobe with the same argument array as media registration."""
    return _run_ffprobe(path)


def _run_ffprobe(path: Path) -> dict[str, Any]:
    command = [
        ffprobe_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise CorruptMediaError(str(path))
    try:
        text = completed.stdout.decode("utf-8")
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorruptMediaError(str(path)) from exc
    if not isinstance(payload, dict):
        raise CorruptMediaError(str(path))
    return payload


def _parse_ffprobe(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    stream = _video_stream(payload, path)
    format_info = payload.get("format")
    if not isinstance(format_info, dict):
        raise CorruptMediaError(str(path))
    # Container duration only. Do not walk packets or use frame_index / fps.
    duration_s = _positive_float(format_info.get("duration"))
    if duration_s is None:
        raise CorruptMediaError(str(path))
    width = _positive_int(stream.get("width"))
    height = _positive_int(stream.get("height"))
    codec = stream.get("codec_name")
    time_base = stream.get("time_base")
    if (
        width is None
        or height is None
        or not isinstance(codec, str)
        or not codec.strip()
        or not isinstance(time_base, str)
        or _TIME_BASE_RE.fullmatch(time_base) is None
    ):
        raise CorruptMediaError(str(path))
    return {
        "duration_s": duration_s,
        "width": width,
        "height": height,
        "rotation_deg": _rotation_deg(stream),
        "sar": _stream_ratio(stream, "sample_aspect_ratio", ("sample_aspect_ratio", "SAR")),
        "dar": _stream_ratio(stream, "display_aspect_ratio", ("display_aspect_ratio", "DAR")),
        "codec": codec.strip(),
        "fps_nominal": _fps(stream.get("r_frame_rate")),
        "fps_average": _fps(stream.get("avg_frame_rate")),
        "time_base": time_base,
        "start_offset_s": _start_offset(format_info),
    }


def _video_stream(payload: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise CorruptMediaError(str(path))
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "video":
            return stream
    raise CorruptMediaError(str(path))


def _rotation_deg(stream: Mapping[str, Any]) -> float | None:
    """Display-matrix or rotate tag. Absent means null, not 0°."""
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if not isinstance(item, Mapping):
                continue
            kind = item.get("side_data_type")
            if not isinstance(kind, str) or "rotation" not in item:
                continue
            lowered = kind.casefold()
            if "display matrix" in lowered or lowered == "rotation":
                return _finite_float(item.get("rotation"))
    tags = stream.get("tags")
    if isinstance(tags, Mapping):
        for key in ("rotate", "rotation"):
            if key in tags:
                return _finite_float(tags.get(key))
    return None


def _stream_ratio(
    stream: Mapping[str, Any],
    field: str,
    tag_names: tuple[str, ...],
) -> str | None:
    raw = stream.get(field)
    parsed = _ratio(raw)
    if parsed is not None:
        return parsed
    if isinstance(raw, str) and raw.strip():
        return None
    tags = stream.get("tags")
    if not isinstance(tags, Mapping):
        return None
    for name in tag_names:
        if name in tags:
            return _ratio(tags.get(name))
    return None


def _ratio(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.upper() in {"N/A", "NA"} or text in {"0:1", "0:0"}:
        return None
    return text


def _fps(value: object) -> float | None:
    """ffprobe rational such as ``30/1``. ``0/0`` is unknown, not 0."""
    if not isinstance(value, str):
        return None
    if "/" not in value:
        return _positive_float(value)
    numerator_text, denominator_text = value.split("/", 1)
    numerator = _finite_float(numerator_text)
    denominator = _finite_float(denominator_text)
    if (
        numerator is None
        or denominator is None
        or denominator == 0
        or numerator <= 0
    ):
        return None
    rate = numerator / denominator
    if not math.isfinite(rate) or rate <= 0:
        return None
    return rate


def _start_offset(format_info: Mapping[str, Any]) -> float | None:
    if "start_time" not in format_info:
        return None
    return _finite_float(format_info.get("start_time"))


def _copy_into_project_media(workspace: Path, project_id: str, source: Path) -> Path:
    media = _project_media_dir(workspace, project_id)
    destination = _unique_file(media, source.name)
    try:
        with source.open("rb") as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=_HASH_CHUNK_BYTES)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return resolve_project_path(workspace, project_id, destination)


def _project_media_dir(workspace: Path, project_id: str) -> Path:
    root = Path(workspace).resolve()
    media = (project_directory(workspace, project_id) / "media").resolve()
    if not media.is_relative_to(root):
        raise PathNotAllowedError(str(media))
    if not media.is_dir():
        raise MediaNotFoundError(str(media))
    return media


def _unique_file(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    for index in range(1, 1000):
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise LabError(
        "duplicate_video",
        "Не удалось подобрать имя копии в каталоге media.",
    )


def _allocate_video_id(video_id: str | None) -> str:
    if video_id is None:
        return "v_" + uuid.uuid4().hex
    if not isinstance(video_id, str):
        raise LabError("invalid_video_id", "Недопустимый идентификатор видео.")
    return video_id


def _clean_source_url(source_url: str | None) -> str | None:
    if source_url is None:
        return None
    if not isinstance(source_url, str) or "\x00" in source_url:
        raise LabError("invalid_source_url", "Некорректный исходный URL.")
    text = source_url.strip()
    return text or None


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _positive_float(value: object) -> float | None:
    number = _finite_float(value)
    if number is None or number <= 0:
        return None
    return number


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
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
