"""Serve one registered video. The URL carries ids, never a filesystem path."""

from __future__ import annotations

from pathlib import Path

from starlette.responses import FileResponse

from excavation_lab import storage
from excavation_lab.errors import LabError
from excavation_lab.ui.video_page import _mime, player_file


def traversal(path: str, project_id: str, video_id: str) -> bool:
    """True when the URL or either id leaves the media route."""
    decoded = _percent_dots(path)
    if any(part == ".." for part in decoded.split("/")):
        return True
    if any(part == ".." for part in decoded.split("\\")):
        return True
    return _unsafe_id(_percent_dots(project_id)) or _unsafe_id(_percent_dots(video_id))


def media_file(workspace: Path, project_id: str, video_id: str) -> FileResponse:
    """Range response for the file ``player_file`` selects.

    The registered id is resolved through allowed roots. A symlink that
    leaves those roots raises ``FileNotFoundError``.
    """
    if _unsafe_id(project_id) or _unsafe_id(video_id):
        raise FileNotFoundError(video_id)
    try:
        video = storage.get_video(workspace, project_id, video_id)
        path, _kind = player_file(workspace, project_id, video)
        resolved = storage.resolve_project_path(workspace, project_id, path)
    except LabError as exc:
        raise FileNotFoundError(video_id) from exc
    if not resolved.is_file():
        raise FileNotFoundError(video_id)
    return FileResponse(resolved, media_type=_mime(resolved))


def _percent_dots(value: str) -> str:
    """Turn ``%2e`` / ``%2E`` into dots, including a second encoding pass."""
    text = value
    for _ in range(2):
        folded = text.replace("%2e", ".").replace("%2E", ".")
        if folded == text:
            break
        text = folded
    return text


def _unsafe_id(value: str) -> bool:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        return True
    return "/" in value or "\\" in value or "\x00" in value or ".." in value
