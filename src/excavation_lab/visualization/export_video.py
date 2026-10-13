"""Annotated MP4 for one analysis run.

The picture is the analysis grid, encoded by ``media/encoder.py`` on the
source interval. Five kept frames spread across one second stay about one
second long. They are not written at the source frame rate. A repeated
picture is the same observation. A large gap is the encoder slate
«нет наблюдения» and is not a new observation.

Predicted boxes are read from ``track_states`` (``is_observed`` false) and
drawn dashed. This module does not write ``detections.parquet``.

The parent enqueues a job and does not construct a detector. The child
opened by ``jobs`` encodes. ``mark_completed`` runs only after
``write_preview`` returns (the encoder container has closed) and ffprobe
can read the file.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from excavation_lab.errors import CorruptMediaError, LabError, MediaNotFoundError
from excavation_lab.jobs import (
    get_job,
    mark_completed,
    mark_failed,
    start_job,
    worker_job,
    write_progress,
)
from excavation_lab.media.encoder import (
    DEFAULT_PREVIEW_GAP_S,
    PreviewFrame,
    write_preview,
)
from excavation_lab.media.probe import ffprobe_json
from excavation_lab.media.reader import FrameReader
from excavation_lab.media.timebase import in_half_open, require_half_open_interval
from excavation_lab.schemas import (
    SCHEMA_VERSION,
    DecodeStatus,
    JobStatus,
    load_labels,
    load_pilot,
)
from excavation_lab.storage import (
    get_analysis_run,
    get_video,
    project_directory,
    record_analysis_export,
)
from excavation_lab.tracking.export_tables import (
    DETECTIONS_NAME,
    FRAME_STATUS_NAME,
    TRACK_STATES_NAME,
)
from excavation_lab.visualization.annotate import (
    BASELINE_CAPTION,
    AnnotateToggles,
    DrawBox,
    TrailSegment,
    ZonePolygon,
    annotate_rgb,
    bucket_trail,
)

# Preview rate. Not the source FPS and not the analysis FPS.
DEFAULT_EXPORT_FPS = 10.0
ANNOTATED_VIDEO = "annotated.mp4"
ANNOTATED_SIDECAR = "annotated.json"
EXPORT_REQUEST = "export_request.json"
RUN_RECORD_NAME = "run.json"
_BASELINE = "baseline_coco"
_PREDICTED = "predicted"
_BUCKET = "bucket"
_START_WAIT_S = 10.0
_TIME_MATCH_S = 1e-3


@dataclass(frozen=True, slots=True)
class AnnotatedExport:
    """Files written after the encoder closed and ffprobe read the MP4."""

    video_path: Path
    map_path: Path
    sidecar_path: Path
    export_fps: float
    encode_ms: float | None
    baseline: bool
    caption: str | None

    def payload(self) -> dict[str, Any]:
        return {
            "video": str(self.video_path),
            "preview_time_map": str(self.map_path),
            "sidecar": str(self.sidecar_path),
            "export_fps": self.export_fps,
            "encode_ms": self.encode_ms,
            "baseline": self.baseline,
            "caption": self.caption,
        }


@dataclass(frozen=True, slots=True)
class ExportEnqueue:
    """What the parent returns. The detector is not part of it."""

    job_id: str
    status: str
    run_id: str
    export_fps: float
    output: str

    def payload(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "run_id": self.run_id,
            "export_fps": self.export_fps,
            "output": self.output,
        }


def encode_annotated_video(
    *,
    video_path: Path,
    video_id: str,
    tables_dir: Path,
    output_path: Path,
    start_s: float,
    end_s: float,
    export_fps: float = DEFAULT_EXPORT_FPS,
    baseline: bool,
    toggles: AnnotateToggles | None = None,
    zones: Sequence[ZonePolygon] = (),
    map_path: Path | None = None,
) -> AnnotatedExport:
    """Annotate the analyzed frames and write a CFR preview.

    ``write_preview`` returns only after its container closes. ffprobe then
    has to read the file. This function does not mark a job completed and
    does not write a detection row.
    """
    fps = _require_fps(export_fps)
    require_half_open_interval(start_s, end_s)
    selected = AnnotateToggles() if toggles is None else toggles
    zone_list = tuple(zones)
    tables = Path(tables_dir)
    frames = _analyzed_rows(tables, start_s, end_s)
    detections = _read_rows(tables / DETECTIONS_NAME)
    tracks = _read_rows(tables / TRACK_STATES_NAME)
    _reject_predicted_detections(detections)
    kept = _annotated_frames(
        Path(video_path),
        video_id,
        frames,
        detections,
        tracks,
        start_s=start_s,
        end_s=end_s,
        baseline=baseline,
        toggles=selected,
        zones=zone_list,
    )
    destination = Path(output_path)
    csv_path = destination.with_name("preview_time_map.csv") if map_path is None else Path(map_path)
    started = time.perf_counter()
    write_preview(
        kept,
        destination,
        output_fps=fps,
        start_s=start_s,
        end_s=end_s,
        preview_gap_s=DEFAULT_PREVIEW_GAP_S,
        map_path=csv_path,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    encode_ms = elapsed_ms if elapsed_ms > 0.0 else None
    _require_playable(destination)
    caption = BASELINE_CAPTION if baseline else None
    sidecar = destination.with_name(ANNOTATED_SIDECAR)
    _write_sidecar(
        sidecar,
        baseline=baseline,
        caption=caption,
        export_fps=fps,
        encode_ms=encode_ms,
        video_name=destination.name,
        map_name=csv_path.name,
    )
    return AnnotatedExport(
        video_path=destination,
        map_path=csv_path,
        sidecar_path=sidecar,
        export_fps=fps,
        encode_ms=encode_ms,
        baseline=baseline,
        caption=caption,
    )


def enqueue_video_export(
    workspace: Path,
    project_id: str,
    run_id: str,
    *,
    export_fps: float | None = None,
    toggles: AnnotateToggles | None = None,
    zones: Sequence[ZonePolygon] = (),
) -> ExportEnqueue:
    """Queue one annotated export. Does not construct a detector."""
    selected = AnnotateToggles() if toggles is None else toggles
    fps = DEFAULT_EXPORT_FPS if export_fps is None else _require_fps(export_fps)
    zone_list = tuple(zones)
    run = get_analysis_run(workspace, project_id, run_id)
    video = get_video(workspace, project_id, str(run["video_id"]))
    _media_path(video)
    duration = _video_duration(video)
    step = 0.2 if 0.2 < duration else duration / 2.0
    output = project_directory(workspace, project_id) / "exports" / run_id / ANNOTATED_VIDEO

    def write_request(run_dir: Path) -> None:
        _atomic_json(
            run_dir / EXPORT_REQUEST,
            {
                "run_id": run_id,
                "export_fps": fps,
                "toggles": {
                    "boxes": selected.boxes,
                    "class_names": selected.class_names,
                    "scores": selected.scores,
                    "track_ids": selected.track_ids,
                    "bucket_trail": selected.bucket_trail,
                    "zones": selected.zones,
                },
                "zones": [
                    {"name": zone.name, "points": [list(point) for point in zone.points]}
                    for zone in zone_list
                ],
            },
        )

    job = start_job(
        workspace,
        project_id,
        kind="export",
        worker="export",
        video_duration_s=duration,
        step_s=step,
        on_run_dir=write_request,
    )
    return ExportEnqueue(
        job_id=job.job_id,
        status=job.status.value,
        run_id=run_id,
        export_fps=fps,
        output=str(output),
    )


def run_export_job(workspace: Path, job_id: str) -> int:
    """Worker entry. Encodes from stored tables. Does not open a detector."""
    root = Path(workspace).resolve()
    project_id, run_dir, _record = worker_job(root, job_id)
    if not _wait_until_running(root, project_id, job_id):
        record = get_job(root, project_id, job_id)
        if record.status in {JobStatus.queued, JobStatus.running}:
            _fail(root, project_id, job_id, "worker_start_timeout")
        return 1
    try:
        request = _read_request(run_dir)
        run_id = str(request["run_id"])
        run = get_analysis_run(root, project_id, run_id)
        video = get_video(root, project_id, str(run["video_id"]))
        media = _media_path(video)
        start_s, end_s = _interval(run, video)
        tables = project_directory(root, project_id) / "runs" / str(run["job_id"])
        frames = _analyzed_rows(tables, start_s, end_s)
        last = float(frames[-1]["source_time_s"])
        duration = _video_duration(video)
        write_progress(
            run_dir,
            {
                "job_id": job_id,
                "video_duration_s": duration,
                "analyzed_end_s": last,
                "frames": [{"source_time_s": last}],
                "stage": "encoding",
            },
        )
        if get_job(root, project_id, job_id).status != JobStatus.running:
            return 0
        destination = project_directory(root, project_id) / "exports" / run_id / ANNOTATED_VIDEO
        result = encode_annotated_video(
            video_path=media,
            video_id=str(video["video_id"]),
            tables_dir=tables,
            output_path=destination,
            start_s=start_s,
            end_s=end_s,
            export_fps=float(request["export_fps"]),
            baseline=run["model_origin"] == _BASELINE,
            toggles=_toggles(request),
            zones=_zones(request),
        )
        if get_job(root, project_id, job_id).status != JobStatus.running:
            return 0
        try:
            mark_completed(root, project_id, job_id, [result.video_path, result.map_path])
        except LabError as exc:
            if exc.code == "job_not_completable":
                return 0
            raise
        record_analysis_export(
            root,
            project_id,
            run_id,
            export_fps=result.export_fps,
            encode_ms=result.encode_ms,
        )
        _patch_run_json(tables, result)
        return 0
    except LabError as exc:
        _fail(root, project_id, job_id, exc.code)
        return 1
    except Exception as exc:  # noqa: BLE001 — the job row keeps the failure
        _fail(root, project_id, job_id, _exception_text(exc))
        return 1


def format_video_export_summary(enqueued: ExportEnqueue) -> str:
    fps = f"{enqueued.export_fps:g}"
    return (
        f"Аннотированное видео поставлено в очередь: {enqueued.job_id}. "
        f"Прогон {enqueued.run_id}. Частота ролика {fps} кадр/с."
    )


def _annotated_frames(
    video_path: Path,
    video_id: str,
    frames: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    tracks: Sequence[Mapping[str, Any]],
    *,
    start_s: float,
    end_s: float,
    baseline: bool,
    toggles: AnnotateToggles,
    zones: Sequence[ZonePolygon],
) -> list[PreviewFrame]:
    wanted = {int(row["source_frame_index"]): row for row in frames}
    by_index: dict[int, list[Mapping[str, Any]]] = {}
    for row in detections:
        index = row.get("source_frame_index")
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        by_index.setdefault(index, []).append(row)
    bucket_points = _bucket_points(detections, tracks)
    pilot = load_pilot()
    titles = _titles(baseline)
    kept: list[PreviewFrame] = []
    first = True
    reader = FrameReader(video_path, video_id=video_id)
    for frame in reader.iterate(start_s, end_s):
        index = frame.source_frame_index
        if index not in wanted or frame.image is None or frame.source_time_s is None:
            continue
        if frame.decode_status is not DecodeStatus.ok:
            continue
        stamp = float(frame.source_time_s)
        boxes, trail = _overlay(
            by_index.get(int(index), ()),
            tracks,
            stamp,
            bucket_points,
            trail_s=pilot.bucket.trail_s,
            trail_gap_s=pilot.bucket.trail_gap_s,
            extrapolation_s=pilot.bucket.extrapolation_s,
        )
        image = annotate_rgb(
            frame.image,
            boxes,
            toggles=toggles,
            titles=titles,
            baseline=baseline,
            burn_baseline_caption=first and baseline,
            trail=trail,
            zones=zones,
        )
        kept.append(PreviewFrame(int(index), stamp, image))
        first = False
        del wanted[index]
    if wanted:
        missing = min(wanted)
        raise LabError(
            "missing_frame",
            f"В видео нет кадра {missing}, который есть в frame_status.",
        )
    kept.sort(key=lambda item: (item.source_time_s, item.source_frame_index))
    return kept


def _overlay(
    detections: Sequence[Mapping[str, Any]],
    tracks: Sequence[Mapping[str, Any]],
    stamp: float,
    bucket_points: Sequence[tuple[str, float, tuple[float, float, float, float]]],
    *,
    trail_s: float,
    trail_gap_s: float,
    extrapolation_s: float,
) -> tuple[list[DrawBox], list[TrailSegment]]:
    boxes: list[DrawBox] = []
    for row in detections:
        if row.get("observation_state") == _PREDICTED:
            continue
        xyxy = _box(row, ("x1", "y1", "x2", "y2"))
        if xyxy is None:
            continue
        track_id = row.get("track_id_nullable")
        boxes.append(
            DrawBox(
                class_name=str(row.get("class_name") or ""),
                xyxy=xyxy,
                score=_optional_float(row.get("score")),
                track_id=track_id if isinstance(track_id, str) and track_id else None,
                predicted=False,
            )
        )
    predicted: list[tuple[str, tuple[float, float, float, float]]] = []
    for row in tracks:
        if not _same_time(row.get("source_time_s"), stamp):
            continue
        if row.get("state") != _PREDICTED or row.get("is_observed") is not False:
            continue
        xyxy = _box(
            row,
            ("x1_nullable", "y1_nullable", "x2_nullable", "y2_nullable"),
        )
        if xyxy is None:
            continue
        track_id = row.get("track_id")
        if not isinstance(track_id, str) or not track_id:
            continue
        boxes.append(
            DrawBox(
                class_name=str(row.get("class_name") or ""),
                xyxy=xyxy,
                score=None,
                track_id=track_id,
                predicted=True,
            )
        )
        if row.get("class_name") == _BUCKET:
            predicted.append((track_id, xyxy))
    trail = bucket_trail(
        bucket_points,
        source_time_s=stamp,
        trail_s=trail_s,
        trail_gap_s=trail_gap_s,
        predicted_boxes=predicted,
        extrapolation_s=extrapolation_s,
    )
    return boxes, trail


def _bucket_points(
    detections: Sequence[Mapping[str, Any]],
    tracks: Sequence[Mapping[str, Any]],
) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    """Observed bucket centers. A null detection track id uses the track row.

    ByteTrack does not own ``bucket``. The observation id lives on
    ``track_states``. A predicted row is not a trail point.
    """
    points: list[tuple[str, float, tuple[float, float, float, float]]] = []
    for row in detections:
        if row.get("class_name") != _BUCKET or row.get("observation_state") == _PREDICTED:
            continue
        stamp = _optional_float(row.get("source_time_s"))
        xyxy = _box(row, ("x1", "y1", "x2", "y2"))
        if stamp is None or xyxy is None:
            continue
        track_id = row.get("track_id_nullable")
        if not isinstance(track_id, str) or not track_id:
            track_id = _observed_bucket_id(tracks, stamp, xyxy)
        if not isinstance(track_id, str) or not track_id:
            continue
        points.append((track_id, stamp, xyxy))
    return points


def _observed_bucket_id(
    tracks: Sequence[Mapping[str, Any]],
    stamp: float,
    xyxy: tuple[float, float, float, float],
) -> str | None:
    for row in tracks:
        if row.get("class_name") != _BUCKET or row.get("is_observed") is not True:
            continue
        if row.get("state") == _PREDICTED or not _same_time(row.get("source_time_s"), stamp):
            continue
        other = _box(row, ("x1_nullable", "y1_nullable", "x2_nullable", "y2_nullable"))
        if other is None or not _same_box(other, xyxy):
            continue
        track_id = row.get("track_id")
        if isinstance(track_id, str) and track_id:
            return track_id
    return None


def _analyzed_rows(tables: Path, start_s: float, end_s: float) -> list[dict[str, Any]]:
    rows = _read_rows(tables / FRAME_STATUS_NAME)
    kept: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        if row.get("analyzed") is not True or row.get("decode_status") != DecodeStatus.ok.value:
            continue
        index = row.get("source_frame_index")
        stamp = _optional_float(row.get("source_time_s"))
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or stamp is None:
            continue
        if not in_half_open(stamp, start_s, end_s):
            continue
        if index in seen:
            raise LabError(
                "duplicate_frame",
                f"Кадр {index} уже есть в frame_status. Повтор не становится наблюдением.",
            )
        seen.add(index)
        kept.append(dict(row))
    kept.sort(key=lambda item: (float(item["source_time_s"]), int(item["source_frame_index"])))
    if not kept:
        raise LabError(
            "empty_export",
            "В интервале нет проанализированных кадров. Ролик не записывается.",
        )
    previous: float | None = None
    for row in kept:
        stamp = float(row["source_time_s"])
        if previous is not None and stamp <= previous:
            raise LabError(
                "timebase",
                "Временные метки анализа убывают. Шкала исходного видео не идёт назад.",
            )
        previous = stamp
    return kept


def _reject_predicted_detections(rows: Sequence[Mapping[str, Any]]) -> None:
    """A predicted box in the detection table is a corrupt run, not a drawing."""
    for row in rows:
        if row.get("observation_state") == _PREDICTED:
            raise LabError(
                "predicted_detection",
                "В detections.parquet есть предсказанная рамка. "
                "Она должна оставаться в track_states с is_observed=false.",
            )


def _titles(baseline: bool) -> dict[str, str]:
    if baseline:
        return {}
    loaded = load_labels()
    return {item.name.value: item.title for item in loaded.classes}


def _require_playable(path: Path) -> None:
    payload = ffprobe_json(path)
    streams = payload.get("streams")
    video = []
    if isinstance(streams, list):
        video = [
            item
            for item in streams
            if isinstance(item, dict) and item.get("codec_type") == "video"
        ]
    duration = _container_duration(payload.get("format"))
    if len(video) != 1 or duration is None:
        raise CorruptMediaError(str(path))


def _container_duration(format_info: object) -> float | None:
    if not isinstance(format_info, dict):
        return None
    raw = format_info.get("duration")
    if isinstance(raw, str):
        try:
            number = float(raw)
        except ValueError:
            return None
    elif isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    else:
        number = float(raw)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _write_sidecar(
    path: Path,
    *,
    baseline: bool,
    caption: str | None,
    export_fps: float,
    encode_ms: float | None,
    video_name: str,
    map_name: str,
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "baseline": baseline,
            "caption": caption,
            "export_fps": export_fps,
            "encode_ms": encode_ms,
            "video": video_name,
            "preview_time_map": map_name,
        },
    )


def _patch_run_json(tables: Path, result: AnnotatedExport) -> None:
    path = tables / RUN_RECORD_NAME
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError("invalid_run", "run.json повреждён.") from exc
    if not isinstance(payload, dict):
        raise LabError("invalid_run", "run.json должен быть объектом.")
    payload["export_fps"] = result.export_fps
    payload["encode_ms"] = result.encode_ms
    _atomic_json(path, payload)


def _read_request(run_dir: Path) -> dict[str, Any]:
    path = run_dir / EXPORT_REQUEST
    if not path.is_file():
        raise LabError("missing_artifact", "В задании нет export_request.json.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError("invalid_export", "export_request.json повреждён.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("run_id"), str):
        raise LabError("invalid_export", "export_request.json должен содержать run_id.")
    _require_fps(payload.get("export_fps"))
    return payload


def _toggles(request: Mapping[str, Any]) -> AnnotateToggles:
    raw = request.get("toggles")
    if raw is None:
        return AnnotateToggles()
    if not isinstance(raw, dict):
        raise LabError("invalid_export", "Переключатели аннотации должны быть объектом.")
    return AnnotateToggles(
        boxes=_flag(raw, "boxes"),
        class_names=_flag(raw, "class_names"),
        scores=_flag(raw, "scores"),
        track_ids=_flag(raw, "track_ids"),
        bucket_trail=_flag(raw, "bucket_trail"),
        zones=_flag(raw, "zones"),
    )


def _flag(raw: Mapping[str, Any], name: str) -> bool:
    if name not in raw:
        return True
    value = raw[name]
    if not isinstance(value, bool):
        raise LabError(
            "invalid_export",
            "Переключатель аннотации должен быть true или false.",
        )
    return value


def _zones(request: Mapping[str, Any]) -> tuple[ZonePolygon, ...]:
    raw = request.get("zones", [])
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise LabError("invalid_zone", "Список зон должен быть массивом.")
    zones: list[ZonePolygon] = []
    for item in raw:
        if not isinstance(item, dict):
            raise LabError("invalid_zone", "Зона должна быть объектом.")
        name = item.get("name")
        points = item.get("points")
        if not isinstance(name, str) or not isinstance(points, list):
            raise LabError("invalid_zone", "У зоны нет имени или вершин.")
        parsed: list[tuple[float, float]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise LabError("invalid_zone", "Вершина зоны должна быть парой x, y.")
            try:
                parsed.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError) as exc:
                raise LabError("invalid_zone", "Координата зоны должна быть числом.") from exc
        zones.append(ZonePolygon(name=name, points=tuple(parsed)))
    return tuple(zones)


def _interval(run: Mapping[str, Any], video: Mapping[str, Any]) -> tuple[float, float]:
    start = _optional_float(run.get("start_s"))
    if start is None:
        raise LabError("invalid_export", "У прогона нет начала интервала.")
    end = _optional_float(run.get("end_s"))
    if end is None:
        end = _video_duration(video)
    require_half_open_interval(start, end)
    return start, end


def _media_path(video: Mapping[str, Any]) -> Path:
    raw = video.get("path")
    if not isinstance(raw, str) or not raw:
        raise MediaNotFoundError()
    path = Path(raw)
    if not path.is_file():
        raise MediaNotFoundError(str(path))
    return path


def _video_duration(video: Mapping[str, Any]) -> float:
    duration = _optional_float(video.get("duration_s"))
    if duration is None or duration <= 0.0:
        raise LabError(
            "missing_duration",
            "Длительность видео не рассчитана. Конец интервала не подставляется нулём.",
        )
    return duration


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise LabError("missing_artifact", f"В прогоне нет файла {path.name}.")
    return pq.read_table(path).to_pylist()


def _box(
    row: Mapping[str, Any],
    names: tuple[str, str, str, str],
) -> tuple[float, float, float, float] | None:
    values = [_optional_float(row.get(name)) for name in names]
    if any(item is None for item in values):
        return None
    x1, y1, x2, y2 = values
    assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
    return (x1, y1, x2, y2)


def _same_box(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return all(abs(a - b) <= 1e-3 for a, b in zip(left, right, strict=True))


def _same_time(value: object, stamp: float) -> bool:
    number = _optional_float(value)
    if number is None:
        return False
    return abs(number - stamp) <= _TIME_MATCH_S


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _require_fps(value: object) -> float:
    number = _optional_float(value)
    if number is None or number <= 0.0:
        raise LabError("invalid_export", "Частота ролика должна быть положительной.")
    return number


def _wait_until_running(workspace: Path, project_id: str, job_id: str) -> bool:
    deadline = time.monotonic() + _START_WAIT_S
    while time.monotonic() < deadline:
        record = get_job(workspace, project_id, job_id)
        if record.status == JobStatus.running:
            return True
        if record.status in {
            JobStatus.failed,
            JobStatus.cancelled,
            JobStatus.completed,
            JobStatus.cancel_requested,
        }:
            return False
        time.sleep(0.01)
    return False


def _fail(workspace: Path, project_id: str, job_id: str, code: str) -> None:
    mark_failed(workspace, project_id, job_id, code)


def _exception_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    if not text:
        return "export_failed"
    return text[:2000]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)
