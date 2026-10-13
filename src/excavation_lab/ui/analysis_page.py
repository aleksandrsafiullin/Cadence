"""«Анализ». Start and export go through the job lock.

The page calls ``enqueue_analyze`` and ``enqueue_video_export``. It does not
construct a detector and does not load weights. Long playback is one encoded
MP4. One ``source_time_s`` is a single frame plus a table built only from
``detections.parquet``.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageDraw
from pydantic import ValidationError

from excavation_lab import storage
from excavation_lab.datasets.zones import (
    load_view_zones_file,
    polygon_pixels,
    views_directory,
)
from excavation_lab.errors import (
    JobAlreadyRunningError,
    LabError,
    TimebaseError,
    map_os_error,
)
from excavation_lab.hardware import (
    official_nano_weights_path,
    weights_cache_dir,
)
from excavation_lab.jobs import PROGRESS_FILENAME, run_directory
from excavation_lab.media.reader import FrameReader
from excavation_lab.media.timebase import require_half_open_interval
from excavation_lab.schemas import (
    ZONE_NAMES,
    JobStatus,
    ViewConfig,
    ZoneVertex,
    load_pilot,
)
from excavation_lab.tracking.analyze import (
    enqueue_analyze,
    validate_analysis_thresholds,
)
from excavation_lab.tracking.export_tables import (
    DETECTIONS_NAME,
    FRAME_STATUS_NAME,
    TRACK_STATES_NAME,
)
from excavation_lab.ui import copy as ui_copy
from excavation_lab.ui.video_page import interval_bounds, job_blocks
from excavation_lab.visualization.annotate import AnnotateToggles, ZonePolygon
from excavation_lab.visualization.export_video import (
    ANNOTATED_VIDEO,
    enqueue_video_export,
)

_BASELINE_MODEL_ID = "baseline_coco"
_TIME_EPS = 1e-3
_BLOCKING = frozenset(
    {
        JobStatus.queued,
        JobStatus.running,
        JobStatus.cancel_requested,
    }
)
_DETECTION_FIELDS = (
    "class_name",
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "track_id_nullable",
    "observation_state",
    "source_time_s",
)
_PREDICTED_FIELDS = (
    "class_name",
    "state",
    "x1_nullable",
    "y1_nullable",
    "x2_nullable",
    "y2_nullable",
    "is_observed",
    "source_time_s",
)
_TOGGLES = (
    ("show_boxes", "toggle_boxes", "boxes"),
    ("show_classes", "toggle_classes", "class_names"),
    ("show_scores", "toggle_scores", "scores"),
    ("show_ids", "toggle_ids", "track_ids"),
    ("show_trail", "toggle_trail", "bucket_trail"),
    ("show_zones", "toggle_zones", "zones"),
)
_FILL = (210, 210, 210)
_BACKGROUND = (16, 16, 16)
_DELTA_KEYS = ("edited_rows", "added_rows", "deleted_rows")


@dataclass(frozen=True, slots=True)
class AnalysisHeader:
    """Three header fields. Missing stays the catalog not-calculated phrase."""

    source_time: str
    model_version: str
    observed: str


@dataclass(frozen=True, slots=True)
class ObservedShare:
    """Observed-frame fraction. ``scope`` says bucket or all classes."""

    text: str
    scope: str | None


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Rows read from one finished run. Predicted boxes stay on ``track_rows``."""

    frame_rows: tuple[Mapping[str, Any], ...]
    detection_rows: tuple[Mapping[str, Any], ...]
    track_rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class FrameReview:
    """Detection rows and the separate predicted list for one source time."""

    detections: tuple[dict[str, Any], ...]
    predicted: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class FinishedRun:
    run_id: str
    directory: Path
    video_id: str
    model_origin: str | None


@dataclass(frozen=True, slots=True)
class ThresholdDraft:
    """ByteTrack and display thresholds for a future run, not a saved one."""

    high_confidence: float
    activation: float
    lost_buffer_s: float
    reference_fps: int
    min_observations: int
    display_threshold: float

    def payload(self) -> dict[str, Any]:
        return {
            "display_threshold": float(self.display_threshold),
            "bytetrack": {
                "high_confidence": float(self.high_confidence),
                "activation": float(self.activation),
                "lost_buffer_s": float(self.lost_buffer_s),
                "reference_fps": int(self.reference_fps),
                "min_observations": int(self.min_observations),
            },
        }


@dataclass(frozen=True, slots=True)
class AnalysisAttempt:
    """One Start or export click. ``job_id`` is set only after the lock."""

    job_id: str | None = None
    notice: str | None = None
    error: str | None = None


def _resolved(text: ui_copy.Catalog | None) -> ui_copy.Catalog:
    if text is not None:
        return text
    return ui_copy.catalog("ru")


def analysis_header(
    snapshot: RunSnapshot | None,
    source_time_s: float | None = None,
    text: ui_copy.Catalog | None = None,
) -> AnalysisHeader:
    """Source time, model version, and observed fraction for the header.

    No catalog argument stays Russian. A missing time is the catalog
    not-calculated phrase. A measured ``0`` stays ``0``.
    """
    catalog = _resolved(text)
    if snapshot is None:
        missing = catalog.not_calculated
        return AnalysisHeader(missing, missing, missing)
    return AnalysisHeader(
        _format_time(source_time_s, catalog),
        _model_version(snapshot.frame_rows, source_time_s, catalog),
        observed_share(snapshot.frame_rows, snapshot.detection_rows, catalog).text,
    )


def header_lines(
    header: AnalysisHeader,
    text: ui_copy.Catalog | None = None,
) -> list[str]:
    """No catalog argument stays Russian."""
    catalog = _resolved(text)
    return [
        f"{catalog.source_time}: {header.source_time}.",
        f"{catalog.model_version}: {header.model_version}.",
        f"{catalog.observed_share}: {header.observed}.",
    ]


def observed_share(
    frame_rows: Sequence[Mapping[str, Any]],
    detection_rows: Sequence[Mapping[str, Any]],
    text: ui_copy.Catalog | None = None,
) -> ObservedShare:
    """``count(observed) / count(analyzed frames)``.

    Skipped frames (``analyzed`` false) are left out of both counts. A missing
    measurement is the catalog not-calculated phrase, not zero. Labels come
    from the catalog. No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    analyzed: list[tuple[float, Mapping[str, Any]]] = []
    for row in frame_rows:
        if not _is_analyzed(row):
            continue
        stamp = _finite(row.get("source_time_s"))
        if stamp is None:
            return ObservedShare(catalog.not_calculated, None)
        analyzed.append((stamp, row))
    if not analyzed:
        return ObservedShare(catalog.not_calculated, None)
    states: list[str] = []
    for _stamp, row in analyzed:
        state = _state_text(row.get("bucket_observation_state"))
        if state is None:
            return _all_class_share(analyzed, detection_rows, catalog)
        states.append(state)
    observed = sum(state == "observed" for state in states)
    return _share(observed, len(states), "bucket", catalog)


def review_frame(
    detection_rows: Sequence[Mapping[str, Any]],
    track_rows: Sequence[Mapping[str, Any]],
    source_time_s: float,
) -> FrameReview:
    """Split one source time.

    The detection tuple is copied from detection rows only. A predicted box
    that exists only on a track-state row stays in ``predicted``.
    """
    return FrameReview(
        detections=tuple(detection_table(detection_rows, source_time_s)),
        predicted=tuple(predicted_list(track_rows, source_time_s)),
    )


def detection_table(
    rows: Sequence[Mapping[str, Any]],
    source_time_s: float,
) -> list[dict[str, Any]]:
    """Columns from ``detections.parquet``. Predicted states are dropped."""
    kept: list[dict[str, Any]] = []
    for row in _rows_at(rows, source_time_s):
        if _state_text(row.get("observation_state")) == "predicted":
            continue
        kept.append({name: _plain(row.get(name)) for name in _DETECTION_FIELDS})
    return kept


def predicted_list(
    rows: Sequence[Mapping[str, Any]],
    source_time_s: float,
) -> list[dict[str, Any]]:
    """Track-state rows whose ``state`` is ``predicted`` and not observed."""
    kept: list[dict[str, Any]] = []
    for row in _rows_at(rows, source_time_s):
        if _state_text(row.get("state")) != "predicted":
            continue
        if _as_bool(row.get("is_observed")) is not False:
            continue
        kept.append({name: _plain(row.get(name)) for name in _PREDICTED_FIELDS})
    return kept


def interval_defaults(video: Mapping[str, object]) -> tuple[float | None, float | None]:
    """Same source clock as the video page: offset plus duration, half-open."""
    return interval_bounds(video)


def threshold_draft_from_pilot() -> ThresholdDraft:
    pilot = load_pilot()
    track = pilot.bytetrack
    return ThresholdDraft(
        high_confidence=track.high_confidence,
        activation=track.activation,
        lost_buffer_s=track.lost_buffer_s,
        reference_fps=track.reference_fps,
        min_observations=track.min_observations,
        display_threshold=pilot.display_threshold,
    )


def _threshold_failure(draft: ThresholdDraft) -> LabError | None:
    """Validate ``draft`` in memory. Does not open or replace a finished run."""
    try:
        body = draft.payload()
        validate_analysis_thresholds(body["display_threshold"], body["bytetrack"])
    except LabError as exc:
        return exc
    return None


def note_threshold_edit(
    draft: ThresholdDraft,
    finished_run: Path | None = None,
    text: ui_copy.Catalog | None = None,
) -> str:
    """Validate ``draft`` in memory.

    ``finished_run`` is not opened and its parquet is not replaced. Using the
    draft takes a new ``enqueue_analyze``. A service ``LabError`` stays the
    service text. No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    if finished_run is not None:
        Path(finished_run)
    failure = _threshold_failure(draft)
    if failure is not None:
        return failure.message
    return catalog.threshold_new_run


def load_snapshot(directory: Path) -> RunSnapshot | None:
    """Read the three tables. A missing detection file is an unknown run."""
    root = Path(directory)
    frames = root / FRAME_STATUS_NAME
    detections = root / DETECTIONS_NAME
    if not frames.is_file() or not detections.is_file():
        return None
    tracks = root / TRACK_STATES_NAME
    try:
        frame_rows = _read_parquet(frames)
        detection_rows = _read_parquet(detections)
        track_rows = _read_parquet(tracks) if tracks.is_file() else []
    except (OSError, pa.ArrowInvalid, ValueError):
        return None
    return RunSnapshot(
        frame_rows=tuple(frame_rows),
        detection_rows=tuple(detection_rows),
        track_rows=tuple(track_rows),
    )


def list_finished_runs(workspace: Path, project_id: str, video_id: str) -> list[FinishedRun]:
    """Runs on disk for this video. Newest last. Does not open weights."""
    root = storage.project_directory(workspace, project_id) / "runs"
    if not root.is_dir() or root.is_symlink():
        return []
    found: list[FinishedRun] = []
    for child in root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        if not (child / DETECTIONS_NAME).is_file() or not (child / FRAME_STATUS_NAME).is_file():
            continue
        document = _read_json(child / "run.json")
        if document is None or document.get("video_id") != video_id:
            continue
        origin = document.get("model_origin")
        found.append(
            FinishedRun(
                run_id=child.name,
                directory=child,
                video_id=video_id,
                model_origin=origin if isinstance(origin, str) else None,
            )
        )
    found.sort(key=lambda item: item.directory.stat().st_mtime)
    return found


def playback_file(workspace: Path, project_id: str, run_id: str) -> Path | None:
    """The encoded MP4 for one run, or none. Never a frame sequence."""
    path = storage.project_directory(workspace, project_id) / "exports" / run_id / ANNOTATED_VIDEO
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        return None
    return path


def list_bundles(workspace: Path, project_id: str) -> list[str]:
    """Custom bundle ids that already have ``weights.pth``. Does not load them."""
    found: list[str] = []
    models = storage.project_directory(workspace, project_id) / "models"
    if models.is_dir() and not models.is_symlink():
        for child in sorted(models.iterdir()):
            if child.is_dir() and not child.is_symlink() and (child / "weights.pth").is_file():
                found.append(child.name)
    try:
        db_path = storage.project_sqlite(workspace, project_id)
    except LabError:
        return _unique(found)
    try:
        with storage.connect(db_path) as connection:
            rows = connection.execute(
                """
                SELECT model_id, path FROM model_versions
                WHERE project_id = ?
                ORDER BY created_at, model_id
                """,
                (project_id,),
            ).fetchall()
    except (sqlite3.Error, LabError):
        return _unique(found)
    for row in rows:
        model_id = row["model_id"]
        raw_path = row["path"]
        if not isinstance(model_id, str) or not model_id:
            continue
        if isinstance(raw_path, str) and Path(raw_path).joinpath("weights.pth").is_file():
            found.append(model_id)
    return _unique(found)


def request_analysis(
    workspace: Path,
    project_id: str,
    video_id: str,
    model_id: str | None,
    analysis_fps: float,
    start_s: object,
    end_s: object,
    *,
    view_id: str | None = None,
    thresholds: ThresholdDraft | None = None,
    text: ui_copy.Catalog | None = None,
) -> AnalysisAttempt:
    """Enqueue one analyze job, or refuse without a second start.

    Does not construct a detector. Baseline enqueue requires the cached Nano
    file. Custom mode without a bundle does not fall back and does not invent
    a result. A service ``LabError`` stays the service text. No catalog
    argument stays Russian.
    """
    catalog = _resolved(text)
    try:
        start, end = _interval(start_s, end_s, catalog)
    except TimebaseError as exc:
        return AnalysisAttempt(error=exc.message)
    if job_blocks(workspace):
        return AnalysisAttempt(error=JobAlreadyRunningError().message)
    if not model_id:
        return AnalysisAttempt(error=catalog.custom_needs_bundle)
    if model_id == _BASELINE_MODEL_ID and not official_nano_weights_path(workspace).is_file():
        notice = catalog.weights_warning(weights_cache_dir(workspace))
        return AnalysisAttempt(notice=notice)
    display: float | None = None
    bytetrack: dict[str, Any] | None = None
    if thresholds is not None:
        body = thresholds.payload()
        display = body["display_threshold"]
        bytetrack = body["bytetrack"]
        try:
            validate_analysis_thresholds(display, bytetrack)
        except LabError as exc:
            return AnalysisAttempt(error=exc.message)
    try:
        enqueued = enqueue_analyze(
            workspace,
            project_id,
            video_id,
            model_id,
            analysis_fps,
            start_s=start,
            end_s=end,
            view_id=view_id,
            display_threshold=display,
            bytetrack=bytetrack,
        )
    except LabError as exc:
        return AnalysisAttempt(error=exc.message)
    return AnalysisAttempt(job_id=enqueued.job_id, notice=catalog.analysis_queued)


def request_export(
    workspace: Path,
    project_id: str,
    run_id: str,
    *,
    toggles: AnnotateToggles,
    zone_rows: Sequence[Mapping[str, object]],
    width: object,
    height: object,
    text: ui_copy.Catalog | None = None,
) -> AnalysisAttempt:
    """Enqueue annotated export. Does not construct a detector or encode here.

    A service ``LabError`` stays the service text. No catalog argument stays
    Russian.
    """
    catalog = _resolved(text)
    if job_blocks(workspace):
        return AnalysisAttempt(error=JobAlreadyRunningError().message)
    try:
        zones = zones_for_export(
            zone_rows,
            width,
            height,
            enabled=toggles.zones,
            text=catalog,
        )
    except LabError as exc:
        return AnalysisAttempt(error=exc.message)
    try:
        enqueued = enqueue_video_export(
            workspace,
            project_id,
            run_id,
            toggles=toggles,
            zones=zones,
        )
    except LabError as exc:
        return AnalysisAttempt(error=exc.message)
    return AnalysisAttempt(job_id=enqueued.job_id, notice=catalog.export_queued)


def zones_for_export(
    rows: Sequence[Mapping[str, object]],
    width: object,
    height: object,
    *,
    enabled: bool,
    text: ui_copy.Catalog | None = None,
) -> tuple[ZonePolygon, ...]:
    """Pixel polygons for the encoder. Unit coordinates stay unit until here.

    No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    if not enabled:
        return ()
    if _out_of_range(rows):
        raise LabError("invalid_zone", catalog.zone_coordinate_file)
    grouped = _group_zone_rows(rows)
    if not grouped:
        return ()
    frame_width = _positive_int(width)
    frame_height = _positive_int(height)
    if frame_width is None or frame_height is None:
        raise LabError("invalid_frame", catalog.zone_export_size)
    zones: list[ZonePolygon] = []
    for name in ZONE_NAMES:
        vertices = grouped.get(name, ())
        if len(vertices) < 3:
            continue
        pixels = polygon_pixels(vertices, frame_width=frame_width, frame_height=frame_height)
        zones.append(ZonePolygon(name=name, points=pixels))
    return tuple(zones)


def zone_preview_rgb(
    rows: Sequence[Mapping[str, object]],
    *,
    width: int,
    height: int,
) -> np.ndarray | None:
    """One preview image. Vertices stay in the unit square until this draw.

    A polygon is drawn wherever its vertices are. ``x < 0.5`` is not a face.
    """
    if _positive_int(width) is None or _positive_int(height) is None:
        return None
    if _out_of_range(rows):
        return None
    grouped = _group_zone_rows(rows)
    scene = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(scene)
    drew = False
    for name in ZONE_NAMES:
        vertices = grouped.get(name, ())
        if len(vertices) < 3:
            continue
        pixels = polygon_pixels(vertices, frame_width=width, frame_height=height)
        draw.polygon(list(pixels), fill=_FILL)
        drew = True
    if not drew:
        return None
    return np.asarray(scene)


def save_zone_table(
    workspace: Path,
    project_id: str,
    view_id: str,
    rows: Sequence[Mapping[str, object]],
    text: ui_copy.Catalog | None = None,
) -> Path:
    """Write ``<project>/views/<view_id>.yaml``. Does not touch ``configs/views``.

    No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    token = _view_token(view_id, catalog)
    if _out_of_range(rows):
        raise LabError("invalid_zone", catalog.zone_coordinate_file)
    payload: dict[str, list[ZoneVertex]] = {name: [] for name in ZONE_NAMES}
    for name, vertices in _group_zone_rows(rows).items():
        payload[name] = [ZoneVertex(x=x, y=y) for x, y in vertices]
    try:
        view = ViewConfig(view_id=token, **payload)
    except ValidationError as exc:
        raise LabError("invalid_zone", catalog.zone_three_vertices) from exc
    directory = (storage.project_directory(workspace, project_id) / "views").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = (directory / f"{token}.yaml").resolve()
    if destination.parent != directory:
        raise LabError("invalid_view", catalog.invalid_view_id)
    text = yaml.safe_dump(view.model_dump(mode="json"), allow_unicode=True, sort_keys=False)
    fd, tmp_name = tempfile.mkstemp(prefix=".view-", suffix=".yaml", dir=directory)
    tmp_path = Path(tmp_name)
    try:
        with map_os_error(str(destination)):
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp_path, destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return destination


def read_source_frame(path: Path, video_id: str, source_time_s: float) -> np.ndarray | None:
    """One upright RGB frame near ``source_time_s``. Does not keep the clip."""
    stamp = _finite(source_time_s)
    if stamp is None or not path.is_file():
        return None
    start = max(0.0, stamp - 0.05)
    end = stamp + 0.2
    if end <= start:
        return None
    reader = FrameReader(path, video_id=video_id)
    held: np.ndarray | None = None
    best: float | None = None
    try:
        for frame in reader.iterate(start, end):
            if frame.image is None or frame.source_time_s is None:
                continue
            delta = abs(frame.source_time_s - stamp)
            if best is None or delta < best:
                held = frame.image.copy()
                best = delta
            if delta <= _TIME_EPS:
                break
    except LabError:
        return None
    return held


def start_enabled(statuses: Sequence[JobStatus]) -> bool:
    """False while a queued, running, or cancel-requested job holds the lock."""
    return not any(status in _BLOCKING for status in statuses)


def _is_editor_delta(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(key in value for key in _DELTA_KEYS)
    return any(getattr(value, key, None) is not None for key in _DELTA_KEYS)


def _apply_editor_delta(seed: object, state: object) -> list[dict[str, object]]:
    rows = _copy_seed_rows(seed)
    for index, changes in _edited_cells(state).items():
        if index < 0 or index >= len(rows):
            continue
        updated = dict(rows[index])
        for column, cell in changes.items():
            updated[str(column)] = cell
        rows[index] = updated
    for index in sorted(_deleted_indexes(state), reverse=True):
        if 0 <= index < len(rows):
            del rows[index]
    for row in _added_row_maps(state):
        rows.append(dict(row))
    return rows


def _copy_seed_rows(seed: object) -> list[dict[str, object]]:
    if isinstance(seed, pd.DataFrame):
        records = seed.to_dict("records")
    elif isinstance(seed, list):
        records = seed
    else:
        records = []
    return [dict(row) for row in records if isinstance(row, Mapping)]


def _edited_cells(state: object) -> dict[int, Mapping[str, object]]:
    raw = _delta_field(state, "edited_rows")
    if not isinstance(raw, Mapping):
        return {}
    found: dict[int, Mapping[str, object]] = {}
    for key, changes in raw.items():
        index = _row_index(key)
        if index is None or not isinstance(changes, Mapping):
            continue
        found[index] = changes
    return found


def _deleted_indexes(state: object) -> list[int]:
    raw = _delta_field(state, "deleted_rows")
    if not isinstance(raw, list):
        return []
    indexes: list[int] = []
    for item in raw:
        index = _row_index(item)
        if index is not None:
            indexes.append(index)
    return indexes


def _added_row_maps(state: object) -> list[Mapping[str, object]]:
    raw = _delta_field(state, "added_rows")
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, Mapping)]


def _delta_field(state: object, name: str) -> object:
    if isinstance(state, Mapping):
        return state.get(name)
    return getattr(state, name, None)


def _row_index(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def rows_from_editor(value: object) -> list[dict[str, object]]:
    if isinstance(value, pd.DataFrame):
        records = value.to_dict("records")
    elif isinstance(value, list):
        records = value
    else:
        return []
    cleaned: list[dict[str, object]] = []
    for row in records:
        if isinstance(row, Mapping):
            cleaned.extend(_clean_one(row))
    return cleaned


def _clean_one(row: Mapping[str, object]) -> list[dict[str, object]]:
    name = row.get("zone")
    if not isinstance(name, str) or name not in ZONE_NAMES:
        return []
    x = _finite(row.get("x"))
    y = _finite(row.get("y"))
    if x is None or y is None:
        return []
    return [{"zone": name, "x": x, "y": y}]


def _initial_zone_rows(workspace: Path, project_id: str, view_id: str) -> list[dict[str, object]]:
    token = view_id.strip()
    if not token or "/" in token or "\\" in token or token in {".", ".."}:
        return []
    directories = (
        storage.project_directory(workspace, project_id) / "views",
        views_directory(),
    )
    for directory in directories:
        path = directory / f"{token}.yaml"
        if path.is_symlink() or not path.is_file():
            continue
        try:
            zones = load_view_zones_file(path)
        except LabError:
            return []
        if zones.view_id != token:
            return []
        rows: list[dict[str, object]] = []
        for name in ZONE_NAMES:
            for x, y in zones.polygon(name):
                rows.append({"zone": name, "x": float(x), "y": float(y)})
        return rows
    return []


def _group_zone_rows(
    rows: Sequence[Mapping[str, object]] | pd.DataFrame,
) -> dict[str, tuple[tuple[float, float], ...]]:
    grouped: dict[str, list[tuple[float, float]]] = {name: [] for name in ZONE_NAMES}
    for row in _zone_records(rows):
        name = row.get("zone")
        x = _finite(row.get("x"))
        y = _finite(row.get("y"))
        if not isinstance(name, str) or name not in ZONE_NAMES or x is None or y is None:
            continue
        if not _in_unit(x) or not _in_unit(y):
            continue
        grouped[name].append((x, y))
    return {name: tuple(points) for name, points in grouped.items() if points}


def _zone_records(
    rows: Sequence[Mapping[str, object]] | pd.DataFrame,
) -> list[dict[str, object]]:
    if isinstance(rows, pd.DataFrame):
        return rows_from_editor(rows)
    return rows_from_editor(list(rows))


def _out_of_range(rows: Sequence[Mapping[str, object]] | pd.DataFrame) -> bool:
    for row in _zone_records(rows):
        x = _finite(row.get("x"))
        y = _finite(row.get("y"))
        if x is None or y is None:
            continue
        if not _in_unit(x) or not _in_unit(y):
            return True
    return False


def _in_unit(value: float) -> bool:
    return 0.0 <= value <= 1.0


def _progress_caption(
    workspace: Path,
    project_id: str,
    job_id: str,
    catalog: ui_copy.Catalog,
) -> str:
    try:
        run_dir = run_directory(workspace, project_id, job_id)
    except LabError:
        return f"{catalog.progress_time}: {catalog.not_calculated}."
    return f"{catalog.progress_time}: {_format_time(_progress_stamp(run_dir), catalog)}."


def _progress_stamp(run_dir: Path) -> float | None:
    payload = _read_json(run_dir / PROGRESS_FILENAME)
    if payload is None:
        return None
    stamp = _finite(payload.get("analyzed_end_s"))
    if stamp is not None:
        return stamp
    frames = payload.get("frames")
    if isinstance(frames, list) and frames and isinstance(frames[-1], Mapping):
        return _finite(frames[-1].get("source_time_s"))
    return None


def _analyzed_times(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    found: list[float] = []
    for row in rows:
        if not _is_analyzed(row):
            continue
        stamp = _finite(row.get("source_time_s"))
        if stamp is None:
            continue
        if any(abs(stamp - previous) <= _TIME_EPS for previous in found):
            continue
        found.append(stamp)
    return found


def _all_class_share(
    analyzed: Sequence[tuple[float, Mapping[str, Any]]],
    detection_rows: Sequence[Mapping[str, Any]],
    catalog: ui_copy.Catalog,
) -> ObservedShare:
    observed: list[float] = []
    for row in detection_rows:
        if _state_text(row.get("observation_state")) != "observed":
            continue
        stamp = _finite(row.get("source_time_s"))
        if stamp is None:
            continue
        for known, _frame in analyzed:
            if abs(known - stamp) <= _TIME_EPS and known not in observed:
                observed.append(known)
                break
    return _share(len(observed), len(analyzed), "all", catalog)


def _share(
    observed: int,
    total: int,
    scope: str,
    catalog: ui_copy.Catalog,
) -> ObservedShare:
    if total <= 0:
        return ObservedShare(catalog.not_calculated, None)
    label = catalog.observed_bucket if scope == "bucket" else catalog.observed_all
    if observed <= 0:
        body = catalog.observed_none.format(total=total)
    else:
        body = catalog.observed_of.format(observed=observed, total=total)
    return ObservedShare(f"{body}, {label}", scope)


def _model_version(
    rows: Sequence[Mapping[str, Any]],
    source_time_s: float | None,
    catalog: ui_copy.Catalog,
) -> str:
    analyzed = [row for row in rows if _is_analyzed(row)]
    if source_time_s is not None:
        for row in analyzed:
            stamp = _finite(row.get("source_time_s"))
            if stamp is None or abs(stamp - float(source_time_s)) > _TIME_EPS:
                continue
            version = row.get("model_version")
            if isinstance(version, str) and version.strip():
                return version.strip()
        return catalog.not_calculated
    versions = [
        row["model_version"].strip()
        for row in analyzed
        if isinstance(row.get("model_version"), str) and row["model_version"].strip()
    ]
    if len(set(versions)) == 1:
        return versions[0]
    return catalog.not_calculated


def _rows_at(rows: Sequence[Mapping[str, Any]], source_time_s: float) -> list[Mapping[str, Any]]:
    stamp = _finite(source_time_s)
    if stamp is None:
        return []
    return [
        row
        for row in rows
        if (found := _finite(row.get("source_time_s"))) is not None and abs(found - stamp) <= _TIME_EPS
    ]


def _is_analyzed(row: Mapping[str, Any]) -> bool:
    value = row.get("analyzed")
    if isinstance(value, bool):
        return value
    return _as_bool(value) is True


def _state_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    raw = getattr(value, "value", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _as_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _as_bool(item())
    return None


def _plain(value: object) -> object:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    item = getattr(value, "item", None)
    if callable(item):
        return _plain(item())
    return value


def _format_time(value: object, catalog: ui_copy.Catalog) -> str:
    number = _finite(value)
    if number is None:
        return catalog.not_calculated
    return f"{number:g} {catalog.seconds_suffix}"


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


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


def _view_token(view_id: str, catalog: ui_copy.Catalog) -> str:
    if (
        not isinstance(view_id, str)
        or not view_id.strip()
        or view_id.strip() in {".", ".."}
        or "/" in view_id
        or "\\" in view_id
        or "\x00" in view_id
    ):
        raise LabError("invalid_view", catalog.invalid_view_id)
    return view_id.strip()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _unique(values: list[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen
