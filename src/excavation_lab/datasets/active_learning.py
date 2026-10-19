"""Train-only active-learning samples from saved detections.

The selector reads a frozen split and does not call ``propose_split``.
A frame is train only when the block that contains its source second is
train. A stored episode id does not choose the role. A second in no block
is not train.

Boxes come from ``detections.parquet`` or an evaluation ``matches.json``.
Nothing here runs a detector or writes ``_annotations.coco.json``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq

from excavation_lab.datasets.catalog import require_project_catalog
from excavation_lab.datasets.splits import FrozenSplit, require_frozen
from excavation_lab.datasets.zones import (
    ZONE_VALID,
    CameraZones,
    load_view_zones_file,
    point_in_polygon,
    views_directory,
)
from excavation_lab.errors import LabError
from excavation_lab.models.quality import _error_rows, _shown_split
from excavation_lab.schemas import DetectionClassName, load_pilot
from excavation_lab.storage import (
    get_video,
    project_directory,
    recorded_model_directory,
)

if TYPE_CHECKING:
    from excavation_lab.datasets.sampler import SampleResult

ACTIVE_REASONS = (
    "false_positive",
    "coordinate_jump",
    "ambiguous_bed",
    "bucket_missing_in_working_area",
    "low_confidence",
)
ZONE_MISSING = "zone_missing"
BUCKET_REASON = "bucket_missing_in_working_area"
_BASELINE_IDS = frozenset({"baseline", "baseline_coco"})
_TIME_EPS = 1e-3
_EXCAVATOR = DetectionClassName.excavator_body.value
_BED = DetectionClassName.truck_bed.value
_MODEL_REQUIRED = (
    "Нет обученной модели и нет сохранённых детекций. Рамки не создаются."
)
_DETECTIONS_MISSING = (
    "Нет сохранённых детекций и нет файла сопоставления. Рамки не создаются."
)
_ACTIVE_SPLIT = "Active learning дополняет только train."
_ACTIVE_ARGS = "Стратегия active не сочетается с планом или дополнительным кадром."
_ACTIVE_COUNT = "Число кадров должно быть целым и больше нуля."


@dataclass(frozen=True, slots=True)
class ActivePick:
    video_id: str
    source_frame_index: int
    source_time_s: float
    selection_reason: str
    model_version: str


@dataclass(frozen=True, slots=True)
class ActiveSelection:
    picks: tuple[ActivePick, ...]
    reason_reports: dict[str, str]


@dataclass(frozen=True, slots=True)
class _EncodedPicks:
    """Written rows plus what the sampler reports about the read."""

    rows: list[Any] = field(default_factory=list)
    decode_failed: list[int] = field(default_factory=list)
    tick_pass: str | None = None
    tick_pass_reason: str | None = None


@dataclass
class _Frame:
    video_id: str
    index: int
    time_s: float
    episodes: set[str | None] = field(default_factory=set)
    reasons: set[str] = field(default_factory=set)
    conflict: bool = False


@dataclass(frozen=True, slots=True)
class _Obs:
    run_id: str
    video_id: str
    index: int
    time_s: float
    track_id: str
    xyxy: tuple[float, float, float, float]
    frame_width: int
    frame_height: int


def sample_active(
    workspace: Path,
    project_id: str,
    *,
    model_id: str,
    video_id: str | None = None,
    split: str | None = None,
    count: int | None = None,
    plan: str | None = None,
    spacing_s: float | None = None,
    extras: Sequence[object] | None = None,
) -> SampleResult:
    """Write train JPEGs for failure frames. The frozen manifest is only read."""
    from excavation_lab.datasets import sampler as sampler_mod

    _require_active_args(split=split, count=count, plan=plan, extras=extras)
    checked_model = _require_model_id(model_id)
    resolved_id = sampler_mod._resolve_video_id(workspace, project_id, video_id)
    frozen = require_frozen(workspace, project_id, resolved_id)
    project_dir = project_directory(workspace, project_id)
    evidence = _load_evidence(project_dir, checked_model, resolved_id)
    has_bundle = _has_custom_bundle(workspace, project_id, checked_model)
    if not has_bundle and not evidence.detection_files:
        raise LabError("model_required", _MODEL_REQUIRED)
    if not evidence.detection_files and not evidence.match_files:
        raise LabError("detections_missing", _DETECTIONS_MISSING)
    pilot = load_pilot()
    video = get_video(workspace, project_id, resolved_id)
    zones = _zones_for(project_dir, _view_ids(evidence, video.get("view_id")))
    rule_names: frozenset[str] = frozenset()
    if checked_model not in _BASELINE_IDS:
        catalog = require_project_catalog(workspace, project_id)
        rule_names = frozenset(item.name for item in catalog.classes)
    selection = select_train_frames(
        frozen,
        video_id=resolved_id,
        model_id=checked_model,
        detections=evidence.detections,
        frame_rows=evidence.frames,
        track_rows=evidence.tracks,
        false_positives=evidence.false_positives,
        zones_by_view=zones,
        video_view_id=_text(video.get("view_id")),
        jump_norm=pilot.jump_norm,
        detection_floor=pilot.detection_floor,
        display_threshold=pilot.display_threshold,
        class_names=rule_names,
    )
    return _write_selection(
        sampler_mod,
        workspace,
        project_id,
        frozen,
        resolved_id,
        video,
        selection,
        count=count,
        spacing_s=spacing_s,
    )


def select_train_frames(
    frozen: FrozenSplit,
    *,
    video_id: str,
    model_id: str,
    detections: Sequence[Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
    track_rows: Sequence[Mapping[str, Any]],
    false_positives: Sequence[Mapping[str, Any]],
    zones_by_view: Mapping[str, CameraZones | None],
    video_view_id: str | None,
    jump_norm: float,
    detection_floor: float,
    display_threshold: float,
    class_names: Collection[str] | None = None,
) -> ActiveSelection:
    """Pick train frames. ``jump_norm`` is the pilot value, with no second default.

    ``bucket_missing_in_working_area`` is emitted only when ``bucket`` is in
    ``class_names``. ``ambiguous_bed`` is emitted only when ``truck_bed`` is.
    ``None`` keeps both, so a caller that has not loaded a catalog is unchanged.
    """
    _require_positive(jump_norm, "Порог скачка координат должен быть положительным числом из конфига.")
    _require_unit_interval(detection_floor, "Нижний порог детекции должен быть числом из конфига.")
    _require_unit_interval(display_threshold, "Порог показа должен быть числом из конфига.")
    names = None if class_names is None else frozenset(class_names)
    bed_on = names is None or "truck_bed" in names
    bucket_on = names is None or "bucket" in names
    frames: dict[tuple[str, int], _Frame] = {}
    status = _status_index(frame_rows, video_id, model_id)
    kept_dets = [
        row
        for row in detections
        if _row_video(row) == video_id and _row_model(row, model_id) and _analyzed(row, status)
    ]
    for row in kept_dets:
        _remember(frames, row, video_id)
        if _low_confidence(row, detection_floor, display_threshold):
            _add_reason(frames, row, video_id, "low_confidence")
        if bed_on and _ambiguous_bed_detection(row):
            _add_reason(frames, row, video_id, "ambiguous_bed")
    if bed_on:
        for row in track_rows:
            if not _ambiguous_bed_track(row):
                continue
            located = _track_frame(row, status, video_id)
            if located is None:
                continue
            _add_located(frames, located, "ambiguous_bed")
    for row in false_positives:
        if _row_video(row, default=video_id) != video_id:
            continue
        if not _row_model(row, model_id):
            continue
        _add_reason(frames, row, video_id, "false_positive")
    _mark_jumps(frames, kept_dets, status, video_id, jump_norm)
    reports: dict[str, str] = {}
    if bucket_on and _mark_missing_buckets(
        frozen,
        frames,
        kept_dets,
        status,
        video_id,
        zones_by_view,
        video_view_id,
    ):
        reports[BUCKET_REASON] = ZONE_MISSING
    picks = _train_picks(frozen, frames, video_id, model_id)
    return ActiveSelection(picks=tuple(picks), reason_reports=reports)


def _require_active_args(
    *,
    split: str | None,
    count: int | None,
    plan: str | None,
    extras: Sequence[object] | None,
) -> None:
    if plan is not None or extras:
        raise LabError("sample_args", _ACTIVE_ARGS)
    if split != "train":
        raise LabError("sample_split", _ACTIVE_SPLIT)
    if count is None:
        return
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise LabError("sample_args", _ACTIVE_COUNT)


def _require_model_id(model_id: str) -> str:
    if (
        not isinstance(model_id, str)
        or not model_id.strip()
        or model_id in {".", ".."}
        or "/" in model_id
        or "\\" in model_id
        or "\x00" in model_id
    ):
        raise LabError("invalid_model", "Недопустимый идентификатор модели.")
    return model_id.strip()


def _has_custom_bundle(workspace: Path, project_id: str, model_id: str) -> bool:
    if model_id in _BASELINE_IDS:
        return False
    recorded = recorded_model_directory(workspace, project_id, model_id)
    if recorded is None:
        return False
    return _regular_file(recorded / "weights.pth")


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


@dataclass(frozen=True, slots=True)
class _Evidence:
    detections: tuple[Mapping[str, Any], ...]
    frames: tuple[Mapping[str, Any], ...]
    tracks: tuple[Mapping[str, Any], ...]
    false_positives: tuple[Mapping[str, Any], ...]
    detection_files: int
    match_files: int


def _load_evidence(project_dir: Path, model_id: str, video_id: str) -> _Evidence:
    detections: list[Mapping[str, Any]] = []
    frames: list[Mapping[str, Any]] = []
    tracks: list[Mapping[str, Any]] = []
    false_positives: list[Mapping[str, Any]] = []
    detection_files = 0
    match_files = 0
    runs = project_dir / "runs"
    if runs.is_symlink() or not runs.is_dir():
        return _Evidence((), (), (), (), 0, 0)
    for child in sorted(runs.iterdir(), key=lambda item: item.name):
        if child.is_symlink() or not child.is_dir():
            continue
        run_model = _sidecar_model(child)
        loaded = _load_parquet(child / "detections.parquet")
        if loaded is not None and _table_matches(loaded, model_id, run_model):
            detection_files += 1
            detections.extend(_tagged(loaded, child.name, model_id, run_model))
        loaded_frames = _load_parquet(child / "frame_status.parquet")
        if loaded_frames is not None and _table_matches(loaded_frames, model_id, run_model):
            frames.extend(_tagged(loaded_frames, child.name, model_id, run_model))
        loaded_tracks = _load_parquet(child / "track_states.parquet")
        if loaded_tracks is not None and _tracks_for_run(child.name, detections, frames):
            tracks.extend(_tagged(loaded_tracks, child.name, model_id, run_model))
        matched = _load_false_positives(child, model_id, video_id)
        if matched is not None:
            match_files += 1
            false_positives.extend(matched)
    return _Evidence(
        tuple(detections),
        tuple(frames),
        tuple(tracks),
        tuple(false_positives),
        detection_files,
        match_files,
    )


def _table_matches(
    rows: Sequence[Mapping[str, Any]],
    model_id: str,
    run_model: str | None,
) -> bool:
    versions = {
        version
        for row in rows
        if isinstance((version := row.get("model_version")), str) and version
    }
    if model_id in versions:
        return True
    if versions:
        return False
    return run_model == model_id


def _tagged(
    rows: Sequence[Mapping[str, Any]],
    run_id: str,
    model_id: str,
    run_model: str | None,
) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for row in rows:
        if not _row_model(row, model_id, run_model=run_model):
            continue
        copied = dict(row)
        if not isinstance(copied.get("run_id"), str) or not copied.get("run_id"):
            copied["run_id"] = run_id
        if not isinstance(copied.get("model_version"), str) or not copied.get("model_version"):
            copied["model_version"] = model_id
        tagged.append(copied)
    return tagged


def _tracks_for_run(
    run_id: str,
    detections: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
) -> bool:
    return any(row.get("run_id") == run_id for row in detections) or any(
        row.get("run_id") == run_id for row in frames
    )


def _load_parquet(path: Path) -> list[dict[str, Any]] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        rows = pq.read_table(path).to_pylist()
    except Exception as exc:
        raise LabError(
            "invalid_detections",
            "Файл детекций повреждён. Рамки не подставляются.",
        ) from exc
    if not isinstance(rows, list):
        raise LabError("invalid_detections", "Файл детекций повреждён. Рамки не подставляются.")
    return [row for row in rows if isinstance(row, dict)]


def _load_false_positives(
    directory: Path,
    model_id: str,
    video_id: str,
) -> list[dict[str, Any]] | None:
    path = directory / "matches.json"
    if path.is_symlink() or not path.is_file():
        return None
    document = _read_json(path)
    named = _document_model(document)
    if named is None:
        named = _sidecar_model(directory)
    if named != model_id:
        return None
    default_video = document.get("video_id") if isinstance(document, Mapping) else None
    rows: list[dict[str, Any]] = []
    for kind, row in _error_rows(document):
        if kind != "false_positive":
            continue
        copied = dict(row)
        if not isinstance(copied.get("video_id"), str) or not copied.get("video_id"):
            if isinstance(default_video, str) and default_video:
                copied["video_id"] = default_video
            else:
                copied["video_id"] = video_id
        copied["model_version"] = model_id
        copied.pop("split", None)
        rows.append(copied)
    return rows


def _document_model(document: object) -> str | None:
    if not isinstance(document, Mapping):
        return None
    return _text(document.get("model_id")) or _text(document.get("model_version"))


def _sidecar_model(directory: Path) -> str | None:
    for name in ("metrics.json", "run.json", "analyze_request.json"):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            continue
        return _document_model(_read_json(path))
    return None


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LabError(
            "invalid_match",
            "Файл сопоставления повреждён. Рамки не подставляются.",
        ) from exc


def _status_index(
    frame_rows: Sequence[Mapping[str, Any]],
    video_id: str,
    model_id: str,
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    found: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in frame_rows:
        if _row_video(row) != video_id or not _row_model(row, model_id):
            continue
        index = _index(row.get("source_frame_index"))
        if index is None:
            continue
        found[(_run_id(row), video_id, index)] = row
    return found


def _analyzed(row: Mapping[str, Any], status: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> bool:
    index = _index(row.get("source_frame_index"))
    video_id = _row_video(row)
    if index is None or video_id is None:
        return False
    frame = status.get((_run_id(row), video_id, index))
    if frame is None:
        return True
    return frame.get("analyzed") is not False


def _remember(frames: dict[tuple[str, int], _Frame], row: Mapping[str, Any], video_id: str) -> _Frame | None:
    index = _index(row.get("source_frame_index"))
    time_s = _finite(row.get("source_time_s"))
    if index is None or time_s is None or time_s < 0:
        return None
    key = (video_id, index)
    current = frames.get(key)
    episode = _text(row.get("episode_id"))
    if current is None:
        current = _Frame(video_id=video_id, index=index, time_s=time_s)
        frames[key] = current
    elif abs(current.time_s - time_s) > _TIME_EPS:
        current.conflict = True
    current.episodes.add(episode)
    return current


def _add_reason(
    frames: dict[tuple[str, int], _Frame],
    row: Mapping[str, Any],
    video_id: str,
    reason: str,
) -> None:
    current = _remember(frames, row, video_id)
    if current is not None:
        current.reasons.add(reason)


def _add_located(
    frames: dict[tuple[str, int], _Frame],
    located: tuple[Mapping[str, Any], str],
    reason: str,
) -> None:
    row, video_id = located
    _add_reason(frames, row, video_id, reason)


def _mark_jumps(
    frames: dict[tuple[str, int], _Frame],
    detections: Sequence[Mapping[str, Any]],
    status: Mapping[tuple[str, str, int], Mapping[str, Any]],
    video_id: str,
    jump_norm: float,
) -> None:
    grouped: dict[tuple[str, str], list[_Obs]] = defaultdict(list)
    for row in detections:
        obs = _observation(row, video_id, status)
        if obs is None:
            continue
        grouped[(obs.run_id, obs.track_id)].append(obs)
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (item.time_s, item.index))
        for previous, current in pairwise(ordered):
            if current.time_s <= previous.time_s + _TIME_EPS:
                continue
            if not _jumped(previous, current, jump_norm):
                continue
            holder = frames.get((video_id, current.index))
            if holder is not None and not holder.conflict:
                holder.reasons.add("coordinate_jump")


def _observation(
    row: Mapping[str, Any],
    video_id: str,
    status: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> _Obs | None:
    track_id = _text(row.get("track_id_nullable")) or _text(row.get("track_id"))
    index = _index(row.get("source_frame_index"))
    time_s = _finite(row.get("source_time_s"))
    box = _xyxy(row)
    width = _extent(row.get("frame_width")) or _extent(row.get("width"))
    height = _extent(row.get("frame_height")) or _extent(row.get("height"))
    if (
        track_id is None
        or index is None
        or time_s is None
        or box is None
        or width is None
        or height is None
        or not _analyzed(row, status)
    ):
        return None
    return _Obs(
        run_id=_run_id(row),
        video_id=video_id,
        index=index,
        time_s=time_s,
        track_id=track_id,
        xyxy=box,
        frame_width=width,
        frame_height=height,
    )


def _jumped(previous: _Obs, current: _Obs, jump_norm: float) -> bool:
    """Normalized center motion above ``jump_norm`` times the previous box size.

    Box size is the diagonal in the same normalized frame coordinates as the
    center. The threshold is not stored a second time in this module.
    """
    size = _box_size(previous.xyxy, previous.frame_width, previous.frame_height)
    motion = _center_motion(previous, current)
    if size is None or motion is None or size <= 0:
        return False
    return motion > jump_norm * size


def _box_size(
    xyxy: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> float | None:
    x1, y1, x2, y2 = xyxy
    width = (x2 - x1) / frame_width
    height = (y2 - y1) / frame_height
    if width <= 0 or height <= 0:
        return None
    return math.hypot(width, height)


def _center_motion(previous: _Obs, current: _Obs) -> float | None:
    start = _center(previous.xyxy, previous.frame_width, previous.frame_height)
    end = _center(current.xyxy, current.frame_width, current.frame_height)
    if start is None or end is None:
        return None
    return math.hypot(end[0] - start[0], end[1] - start[1])


def _center(
    xyxy: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
) -> tuple[float, float] | None:
    x1, y1, x2, y2 = xyxy
    if min(frame_width, frame_height) <= 0 or x2 <= x1 or y2 <= y1:
        return None
    if not all(math.isfinite(value) for value in xyxy):
        return None
    return ((x1 + x2) / 2 / frame_width, (y1 + y2) / 2 / frame_height)


def _mark_missing_buckets(
    frozen: FrozenSplit,
    frames: dict[tuple[str, int], _Frame],
    detections: Sequence[Mapping[str, Any]],
    status: Mapping[tuple[str, str, int], Mapping[str, Any]],
    video_id: str,
    zones_by_view: Mapping[str, CameraZones | None],
    video_view_id: str | None,
) -> bool:
    """Return whether a train frame skipped the bucket reason for lack of a zone."""
    zone_gap = False
    by_index: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in detections:
        index = _index(row.get("source_frame_index"))
        if index is not None:
            by_index[index].append(row)
    for frame in status.values():
        if frame.get("bucket_observation_state") != "missing":
            continue
        if frame.get("analyzed") is False:
            continue
        index = _index(frame.get("source_frame_index"))
        time_s = _finite(frame.get("source_time_s"))
        if index is None or time_s is None:
            continue
        holder = _remember(frames, frame, video_id)
        if holder is None or holder.conflict or not _is_train(frozen, holder):
            continue
        view_id = _text(frame.get("view_id")) or video_view_id
        zones = None if view_id is None else zones_by_view.get(view_id)
        if zones is None or zones.status != ZONE_VALID or len(zones.working_area) < 3:
            zone_gap = True
            continue
        if any(_excavator_in_zone(row, zones) for row in by_index.get(index, ())):
            holder.reasons.add(BUCKET_REASON)
    return zone_gap


def _excavator_in_zone(row: Mapping[str, Any], zones: CameraZones) -> bool:
    if row.get("class_name") != _EXCAVATOR or zones.status != ZONE_VALID:
        return False
    if len(zones.working_area) < 3:
        return False
    box = _xyxy(row)
    width = _extent(row.get("frame_width")) or _extent(row.get("width"))
    height = _extent(row.get("frame_height")) or _extent(row.get("height"))
    if box is None or width is None or height is None:
        return False
    center = _center(box, width, height)
    if center is None:
        return False
    return point_in_polygon(center[0], center[1], zones.working_area)


def _train_picks(
    frozen: FrozenSplit,
    frames: dict[tuple[str, int], _Frame],
    video_id: str,
    model_id: str,
) -> list[ActivePick]:
    picks: list[ActivePick] = []
    for frame in frames.values():
        if frame.video_id != video_id or frame.conflict or not frame.reasons:
            continue
        if not _is_train(frozen, frame):
            continue
        picks.append(
            ActivePick(
                video_id=frame.video_id,
                source_frame_index=frame.index,
                source_time_s=frame.time_s,
                selection_reason=_primary(frame.reasons),
                model_version=model_id,
            )
        )
    picks.sort(key=lambda item: (item.source_time_s, item.source_frame_index))
    return picks


def _is_train(frozen: FrozenSplit, frame: _Frame) -> bool:
    return _shown_split(frozen, None, frame.time_s) == "train"


def _primary(reasons: set[str]) -> str:
    for name in ACTIVE_REASONS:
        if name in reasons:
            return name
    raise LabError("sample_internal", "У кадра нет причины отбора. Кадр не записывается.")


def _low_confidence(row: Mapping[str, Any], floor: float, display: float) -> bool:
    state = row.get("observation_state")
    if state == "weak":
        return True
    if state == "observed":
        return False
    score = _finite(row.get("score"))
    if score is None:
        return False
    return floor <= score < display


def _ambiguous_bed_detection(row: Mapping[str, Any]) -> bool:
    return row.get("class_name") == _BED and row.get("observation_state") == "ambiguous"


def _ambiguous_bed_track(row: Mapping[str, Any]) -> bool:
    if row.get("class_name") != _BED:
        return False
    if row.get("identity_uncertain") is True:
        return True
    return row.get("state") == "ambiguous"


def _track_frame(
    row: Mapping[str, Any],
    status: Mapping[tuple[str, str, int], Mapping[str, Any]],
    video_id: str,
) -> tuple[Mapping[str, Any], str] | None:
    time_s = _finite(row.get("source_time_s"))
    if time_s is None:
        return None
    run_id = _run_id(row)
    for (_run, found_video, _index), frame in status.items():
        if _run != run_id or found_video != video_id:
            continue
        if frame.get("analyzed") is False:
            continue
        stamp = _finite(frame.get("source_time_s"))
        if stamp is None or abs(stamp - time_s) > _TIME_EPS:
            continue
        return frame, video_id
    return None


def _zones_for(project_dir: Path, view_ids: set[str]) -> dict[str, CameraZones | None]:
    return {view_id: _load_zone(project_dir, view_id) for view_id in view_ids}


def _view_ids(evidence: _Evidence, video_view: object) -> set[str]:
    found: set[str] = set()
    video_text = _text(video_view)
    if video_text is not None:
        found.add(video_text)
    for row in (*evidence.frames, *evidence.detections):
        view_id = _text(row.get("view_id"))
        if view_id is not None:
            found.add(view_id)
    return found


def _load_zone(project_dir: Path, view_id: str) -> CameraZones | None:
    for directory in (project_dir / "views", views_directory()):
        path = directory / f"{view_id}.yaml"
        if path.is_symlink() or not path.is_file():
            continue
        zones = load_view_zones_file(path)
        if zones.view_id != view_id or zones.status != ZONE_VALID:
            return None
        if len(zones.working_area) < 3:
            return None
        return zones
    return None


def _write_selection(
    sampler_mod: Any,
    workspace: Path,
    project_id: str,
    frozen: FrozenSplit,
    video_id: str,
    video: Mapping[str, Any],
    selection: ActiveSelection,
    *,
    count: int | None,
    spacing_s: float | None,
) -> SampleResult:
    spacing = sampler_mod._spacing_or_config(spacing_s)
    limited = selection.picks if count is None else selection.picks[:count]
    requested_train = len(selection.picks) if count is None else count
    manifest_path, images_dir = sampler_mod._artifact_paths(
        project_directory(workspace, project_id),
        video_id,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _EncodedPicks()
    if limited:
        encoded = _encode_picks(
            sampler_mod,
            frozen,
            video_id,
            video,
            limited,
            images_dir,
            spacing,
        )
    rows = encoded.rows
    from excavation_lab.schemas import SplitCounts

    requested, written, shortfall = _counts(requested_train, len(rows))
    result = sampler_mod.SampleResult(
        project_id=project_id,
        video_id=video_id,
        split_content_hash=frozen.content_hash,
        plan=None,
        spacing_s=spacing,
        requested=SplitCounts(train=requested, validation=0, test=0),
        written=SplitCounts(train=written, validation=0, test=0),
        shortfall=SplitCounts(train=shortfall, validation=0, test=0),
        rows=rows,
        manifest_path=str(manifest_path),
        images_dir=str(images_dir),
        reason_reports=dict(selection.reason_reports),
        sample_schema_version=sampler_mod.SAMPLE_SCHEMA_VERSION,
        decode_failed=encoded.decode_failed,
        tick_pass=encoded.tick_pass,
        tick_pass_reason=encoded.tick_pass_reason,
    )
    manifest_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _encode_picks(
    sampler_mod: Any,
    frozen: FrozenSplit,
    video_id: str,
    video: Mapping[str, Any],
    picks: Sequence[ActivePick],
    images_dir: Path,
    spacing: float,
) -> _EncodedPicks:
    from excavation_lab.media.reader import FrameReader
    from excavation_lab.schemas import SplitName

    media = sampler_mod._media_path(video.get("path"))
    reader = FrameReader(media, video_id=video_id)
    windows = sampler_mod._windows(frozen, SplitName.train)
    prefer_pts = sampler_mod.probe_is_cfr(video)
    collected = sampler_mod._collect_ticks(
        reader,
        frozen,
        SplitName.train,
        windows,
        prefer_pts=prefer_pts,
    )
    if not prefer_pts:
        collected = sampler_mod.TickPass(
            collected.ticks, False, sampler_mod._PROBE_NOT_CFR
        )
    ticks = {tick.source_frame_index: tick for tick in collected.ticks}
    selected = []
    for pick in picks:
        tick = ticks.get(pick.source_frame_index)
        if tick is None or abs(tick.source_time_s - pick.source_time_s) > _TIME_EPS:
            raise LabError(
                "frame_missing",
                f"Кадр {pick.source_frame_index} не входит в train. "
                "Подставной кадр не записывается.",
            )
        if _shown_split(frozen, None, tick.source_time_s) != "train":
            raise LabError(
                "frame_missing",
                f"Кадр {pick.source_frame_index} не входит в train. "
                "Подставной кадр не записывается.",
            )
        selected.append(
            sampler_mod.SelectedTick(
                video_id=tick.video_id,
                source_frame_index=tick.source_frame_index,
                source_time_s=tick.source_time_s,
                episode_id=tick.episode_id,
                split=SplitName.train,
                selection_reason=pick.selection_reason,
                strategy="active",
                model_version=pick.model_version,
                source_pts=tick.source_pts,
                span_end_s=tick.span_end_s,
            )
        )
    sampler_mod.require_sample_spacing(selected, spacing)
    images_dir.mkdir(parents=True, exist_ok=True)
    outcome = sampler_mod._write_chosen(
        reader,
        {SplitName.train: windows},
        selected,
        images_dir,
    )
    # A tick that did not decode is reported in ``decode_failed`` and counted
    # as shortfall, the same as the plain sampler. No neighbour replaces it.
    return _EncodedPicks(
        rows=sampler_mod._rows_with_warnings(outcome.encoded),
        decode_failed=list(outcome.decode_failed),
        tick_pass=collected.name,
        tick_pass_reason=collected.reason if not collected.used_pts else None,
    )


def _counts(requested_train: int, written_train: int) -> tuple[int, int, int]:
    if written_train > requested_train:
        raise LabError("sample_internal", "Отбор вернул больше кадров, чем запрошено.")
    return requested_train, written_train, requested_train - written_train


def _row_video(row: Mapping[str, Any], default: str | None = None) -> str | None:
    return _text(row.get("video_id")) or default


def _row_model(
    row: Mapping[str, Any],
    model_id: str,
    *,
    run_model: str | None = None,
) -> bool:
    version = _text(row.get("model_version"))
    if version is not None:
        return version == model_id
    if run_model is not None:
        return run_model == model_id
    return True


def _run_id(row: Mapping[str, Any]) -> str:
    return _text(row.get("run_id")) or ""


def _xyxy(row: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    if all(_finite(row.get(name)) is not None for name in ("x1", "y1", "x2", "y2")):
        box = tuple(float(row[name]) for name in ("x1", "y1", "x2", "y2"))
    else:
        raw = row.get("xyxy")
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            return None
        if any(_finite(value) is None for value in raw):
            return None
        box = tuple(float(value) for value in raw)
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _index(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _extent(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _require_positive(value: float, message: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError("sample_args", message)
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise LabError("sample_args", message)


def _require_unit_interval(value: float, message: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError("sample_args", message)
    number = float(value)
    if not math.isfinite(number) or number < 0 or number > 1:
        raise LabError("sample_args", message)
