"""Detector inference on a source interval.

Frames are read from ``media/reader.py`` and passed to
``DetectorAdapter.predict`` one at a time. The grid picks which decoded
frames are analyzed. Each chosen frame keeps the reader's ``source_time_s``.
That timestamp is not ``analysis_index / analysis_fps``.

``analysis_fps``, the source FPS, and ``export_fps`` are separate fields on
the run row. ``export_fps`` stays null until an annotated export exists.
Decode and encode timings stay null until those stages are wrapped.

The parent enqueues a job and does not construct a detector. The child
opened by ``jobs`` loads the model. After ``predict``, ``suppress`` keeps
every cross-class box, then one ``TrackingSession`` runs class-split
vehicle ByteTrack and namespaces track ids. ``BucketAssociator`` and
``BedTruckAssociator`` run on that same frame, after the session update.
An empty detector result is a frame row with no boxes. A predicted bucket
dash is not one of those boxes, and a bed link does not add one either.
Observation states run after both associators. A predicted box stays on
the track-state rows with ``is_observed=false`` and in
``track_states.parquet``. Rows flush by time chunk inside ``export_tables``.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from numpy.typing import NDArray
from pydantic import ValidationError

from excavation_lab.errors import (
    CheckpointMissingError,
    LabError,
    MediaNotFoundError,
)
from excavation_lab.jobs import (
    get_job,
    mark_completed,
    mark_failed,
    start_job,
    worker_job,
    write_progress,
)
from excavation_lab.media.reader import FrameReader, ReadFrame
from excavation_lab.media.timebase import in_half_open, require_half_open_interval
from excavation_lab.models.adapter import (
    CheckpointOrigin,
    DetectorAdapter,
    FakeDetector,
    FrameDetections,
)
from excavation_lab.schemas import (
    SCHEMA_VERSION,
    ByteTrackConfig,
    DecodeStatus,
    JobStatus,
    ObservationState,
    load_pilot,
)
from excavation_lab.storage import (
    VideoRecord,
    connect,
    get_video,
    insert_analysis_run,
    project_directory,
    project_sqlite,
)
from excavation_lab.tracking.bed import BedFrame
from excavation_lab.tracking.bucket import BucketFrame
from excavation_lab.tracking.export_tables import (
    DETECTIONS_COLUMNS,
    FRAME_STATUS_COLUMNS,
    AnalysisTableWriter,
    track_state_record,
    write_analysis_tables,
)
from excavation_lab.tracking.session import RESET_SEEK, TrackedRow, TrackingSession
from excavation_lab.tracking.states import FrameStates, ObservationLedger
from excavation_lab.tracking.suppress import suppress
from excavation_lab.tracking.vehicles import (
    BUCKET,
    EXCAVATOR_BODY,
    HAUL_TRUCK,
    TRUCK_BED,
    VehicleTrackers,
)

REQUEST_NAME = "analyze_request.json"
RUN_RECORD_NAME = "run.json"
FAKE_DETECTOR_ENV = "EXCAVATION_LAB_FAKE_DETECTOR"
_BASELINE_IDS = frozenset({"baseline", "baseline_coco"})
_START_WAIT_S = 10.0
_EDGE = Fraction(1, 1_000_000)
_BAD_FPS = "Частота анализа должна быть положительным числом кадров в секунду."


class _Cancelled(Exception):
    """The job was cancelled while frames were still being read."""


class _FrameSource(Protocol):
    def iterate(
        self,
        start_s: float = 0.0,
        end_s: float | None = None,
    ) -> Iterator[ReadFrame]: ...


@dataclass
class AnalyzeEnqueue:
    """What the parent returns. The detector is not part of it."""

    job_id: str
    status: str
    video_id: str
    model_id: str
    model_origin: str
    analysis_fps: float
    source_fps: float | None
    export_fps: None
    start_s: float
    end_s: float | None

    def payload(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "video_id": self.video_id,
            "model_id": self.model_id,
            "model_origin": self.model_origin,
            "analysis_fps": self.analysis_fps,
            "source_fps": self.source_fps,
            "export_fps": self.export_fps,
            "start_s": self.start_s,
            "end_s": self.end_s,
        }


class AnalyzePipeline:
    """Stream one interval. At most one frame image is retained."""

    def __init__(
        self,
        reader: _FrameSource,
        detector: DetectorAdapter,
        *,
        run_id: str,
        output_dir: Path,
        analysis_fps: float,
        model_version: str,
        start_s: float = 0.0,
        end_s: float | None = None,
        view_id: str | None = None,
        reset_reason: str | None = None,
        on_recorded: Callable[[float], None] | None = None,
        vehicles: VehicleTrackers | None = None,
        same_class_nms_iou: float | None = None,
    ) -> None:
        self.reader = reader
        self.detector = detector
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.analysis_fps = _require_fps(analysis_fps)
        self.model_version = model_version
        self.start_s = float(start_s)
        self.end_s = None if end_s is None else float(end_s)
        self.view_id = view_id
        self.reset_reason = _optional_reset_reason(reset_reason)
        self.on_recorded = on_recorded
        self.vehicles = (
            vehicles
            if vehicles is not None
            else VehicleTrackers.from_pilot(analysis_fps=self.analysis_fps)
        )
        self.same_class_nms_iou = _configured_nms_iou(same_class_nms_iou)
        self.session = TrackingSession(
            run_id=run_id,
            video_id=None,
            view_id=view_id,
            vehicles=self.vehicles,
        )
        self.states = ObservationLedger(
            detection_floor=self.vehicles.detection_floor,
            display_threshold=self.vehicles.display_threshold,
            extrapolation_s=self.session.bucket.extrapolation_s,
        )
        self._frame_states: FrameStates | None = None
        self.max_retained_frames = 0
        self._current_image: NDArray[Any] | None = None
        self._held_frames = 0
        self._tracked_rows: tuple[TrackedRow, ...] = ()
        self._bucket_frame: BucketFrame | None = None
        self._bed_frame: BedFrame | None = None
        self.tables: AnalysisTableWriter | None = None
        require_half_open_interval(self.start_s, self.end_s)

    def run(self) -> tuple[Path, Path, Path]:
        """Read, predict, and write the three parquet files.

        The open chunk lives in ``export_tables``. This object does not keep
        a list of every frame, and it does not keep the images.
        """
        opened = write_analysis_tables(self.output_dir)
        if not isinstance(opened, AnalysisTableWriter):
            raise LabError("invalid_table", "Не удалось открыть запись таблиц анализа.")
        self.tables = opened
        tables = opened
        self.session.restart(self.reset_reason)
        self.states.begin(self.session.reset_count)
        step = Fraction(1) / Fraction(self.analysis_fps).limit_denominator(1_000_000)
        due = _first_tick(self.start_s, step)
        analysis_index = 0
        try:
            return self._consume(tables, due, step, analysis_index)
        except _Cancelled:
            end = tables.last_source_time_s
            if end is None:
                tables.close()
            else:
                tables.close(partial=True, analyzed_end_s=end)
            raise
        except Exception:
            tables.abort()
            raise

    def _consume(
        self,
        tables: AnalysisTableWriter,
        due: Fraction,
        step: Fraction,
        analysis_index: int,
    ) -> tuple[Path, Path, Path]:
        self._consume_frames(tables, due, step, analysis_index)
        return tables.close()

    def _consume_frames(
        self,
        tables: AnalysisTableWriter,
        due: Fraction,
        step: Fraction,
        analysis_index: int,
    ) -> None:
        for frame in self.reader.iterate(self.start_s, self.end_s):
            timestamp = frame.source_time_s
            # A backward source_time_s is a seek. The media reader still
            # rejects a backward PTS inside one file. Rewind the grid so
            # this frame can be selected; the session resets on that call.
            # The stamp stays source_time_s, not the frame index.
            backward = timestamp is not None and self.session.is_backward(timestamp)
            if backward:
                due = _tick_not_after(timestamp, step)
            if frame.decode_status is DecodeStatus.corrupt or frame.image is None:
                if backward:
                    self.session.reset(RESET_SEEK)
                # A corrupt span is a decode gap. It is not downtime and it
                # does not invent a box.
                if frame.decode_status is DecodeStatus.corrupt:
                    self.states.note_decode_gap(
                        source_frame_index=frame.source_frame_index,
                        source_time_s=frame.source_time_s,
                        reset_count=self.session.reset_count,
                    )
                tables.add_frame(
                    _frame_row(
                        frame,
                        run_id=self.run_id,
                        view_id=self.view_id,
                        analyzed=False,
                        model_version=self.model_version,
                        inference_ms=None,
                        num_detections=0,
                        analysis_index=None,
                    )
                )
                self._notify(timestamp)
                continue
            if timestamp is None or not in_half_open(timestamp, self.start_s, self.end_s):
                if backward:
                    self.session.reset(RESET_SEEK)
                continue
            if timestamp + 1e-6 < float(due):
                # The grid did not select this source frame. No zero row.
                if backward:
                    self.session.reset(RESET_SEEK)
                continue
            self._hold(frame.image)
            try:
                started = time.perf_counter()
                # source_time_s, not analysis_index / analysis_fps.
                detections = self.detector.predict(frame.image, timestamp)
                inference_ms = (time.perf_counter() - started) * 1000.0
                detections = self._after_predict(frame, detections)
            finally:
                self._release()
            count = len(detections.class_name)
            bucket_frame = self._bucket_frame
            tables.add_frame(
                _frame_row(
                    frame,
                    run_id=self.run_id,
                    view_id=self.view_id,
                    analyzed=True,
                    model_version=self.model_version,
                    inference_ms=inference_ms,
                    num_detections=count,
                    analysis_index=analysis_index,
                    bucket_observation_state=(
                        None
                        if bucket_frame is None
                        else bucket_frame.bucket_observation_state.value
                    ),
                    diagnostic_flags=(
                        [] if bucket_frame is None else list(bucket_frame.diagnostic_flags)
                    ),
                )
            )
            frame_states = self._frame_states
            for detection in _detection_rows(
                frame,
                detections,
                run_id=self.run_id,
                view_id=self.view_id,
                model_version=self.model_version,
                analysis_index=analysis_index,
                assignments=self._tracked_rows,
                bucket_frame=bucket_frame,
                bed_frame=self._bed_frame,
                observation_states=(
                    None if frame_states is None else frame_states.detection_states
                ),
            ):
                tables.add_detection(detection)
            if frame_states is not None:
                for item in frame_states.track_states:
                    tables.add_track_state(track_state_record(item))
            analysis_index += 1
            due = _advance(due, timestamp, step)
            self._notify(timestamp)

    def _after_predict(
        self,
        frame: ReadFrame,
        detections: FrameDetections,
    ) -> FrameDetections:
        """Cross-class guard, then vehicle tracks, then both associators.

        Returned coordinates are the detector boxes. Same-class NMS runs
        only when ``same_class_nms_iou`` is set; the pilot value is null.
        ``truck_bed`` inside ``haul_truck`` remains its own row, and
        ``bucket`` keeps its class name. This is the only tracking entry.
        Active vehicles are this frame's confirmed boxes. A vehicle
        ByteTrack still holds in its lost buffer is not passed in and is
        not synthesized. The bed associator runs after the bucket associator
        on this same frame. Observation states run after that pair.
        """
        detections = suppress(
            detections,
            same_class_nms_iou=self.same_class_nms_iou,
        )
        source_time_s = frame.source_time_s
        if source_time_s is None:
            source_time_s = detections.timestamp_s
        stamp = float(source_time_s)
        self._tracked_rows = self.session.update(
            detections,
            video_id=frame.video_id,
            view_id=self.view_id,
            source_time_s=stamp,
        )
        # After the session update. ``last_positions`` is already this
        # frame only; the associator copies what it needs for the next one.
        self._bucket_frame = self.session.bucket.update(
            detections,
            bodies=self.session.positions_for(EXCAVATOR_BODY),
            source_time_s=stamp,
            run_id=self.run_id,
            video_id=frame.video_id,
            view_id=self.view_id,
        )
        self._bed_frame = self.session.bed.update(
            detections,
            trucks=self.session.positions_for(HAUL_TRUCK),
            source_time_s=stamp,
            run_id=self.run_id,
            video_id=frame.video_id,
            view_id=self.view_id,
        )
        self._frame_states = self.states.observe(
            detections,
            tracked_rows=self._tracked_rows,
            bucket_frame=self._bucket_frame,
            bed_frame=self._bed_frame,
            lost_vehicles=self.session.lost_buffer_vehicles(),
            run_id=self.run_id,
            video_id=frame.video_id,
            view_id=self.session.view_id,
            source_time_s=stamp,
            reset_count=self.session.reset_count,
        )
        return detections

    def _hold(self, image: NDArray[Any]) -> None:
        self._current_image = image
        self._held_frames = 1
        self.max_retained_frames = max(self.max_retained_frames, self._held_frames)

    def _release(self) -> None:
        self._current_image = None
        self._held_frames = 0

    def _notify(self, timestamp: float | None) -> None:
        if self.on_recorded is None or timestamp is None:
            return
        self.on_recorded(timestamp)


def validate_analysis_thresholds(
    display_threshold: float | None = None,
    bytetrack: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Check a form draft.

    Does not write parquet, does not open a detector, and does not start a job.
    A finished run stays as it was. The draft applies only on a new enqueue.
    """
    if display_threshold is None and bytetrack is None:
        return None
    body: dict[str, Any] = {}
    if display_threshold is not None:
        checked = _optional_display(display_threshold)
        if checked is None:
            raise LabError(
                "invalid_threshold",
                "Порог показа не задан. Завершённый прогон не изменён.",
            )
        body["display_threshold"] = checked
    if bytetrack is not None:
        checked_track = _optional_bytetrack(bytetrack)
        if checked_track is None:
            raise LabError(
                "invalid_threshold",
                "Пороги ByteTrack не заданы. Завершённый прогон не изменён.",
            )
        body["bytetrack"] = checked_track
    return body


def enqueue_analyze(
    workspace: Path,
    project_id: str,
    video_id: str,
    model_id: str,
    analysis_fps: float,
    *,
    start_s: float = 0.0,
    end_s: float | None = None,
    view_id: str | None = None,
    reset_reason: str | None = None,
    display_threshold: float | None = None,
    bytetrack: Mapping[str, Any] | None = None,
) -> AnalyzeEnqueue:
    """Queue one analyze job. Does not construct a detector.

    ``display_threshold`` and ``bytetrack`` are stored on this new request.
    They are not written into an older run's parquet.
    """
    fps = _require_fps(analysis_fps)
    require_half_open_interval(start_s, end_s)
    view = _optional_view(view_id)
    reason = _optional_reset_reason(reset_reason)
    overrides = validate_analysis_thresholds(display_threshold, bytetrack)
    project_sqlite(workspace, project_id)
    video = get_video(workspace, project_id, video_id)
    _require_media(video)
    origin, _bundle = resolve_model(workspace, project_id, model_id)
    duration = _duration_s(video)
    step = fps_step(fps, duration)
    source_fps = source_fps_of(video)

    def write_request(run_dir: Path) -> None:
        payload = {
            "video_id": video_id,
            "model_id": model_id,
            "model_origin": origin,
            "analysis_fps": fps,
            "start_s": float(start_s),
            "end_s": None if end_s is None else float(end_s),
            "view_id": view,
            "reset_reason": reason,
        }
        if overrides:
            payload.update(overrides)
        _atomic_json(run_dir / REQUEST_NAME, payload)

    job = start_job(
        workspace,
        project_id,
        kind="analyze",
        worker="analyze",
        video_duration_s=duration,
        step_s=step,
        on_run_dir=write_request,
    )
    return AnalyzeEnqueue(
        job_id=job.job_id,
        status=job.status.value,
        video_id=video_id,
        model_id=model_id,
        model_origin=origin,
        analysis_fps=fps,
        source_fps=source_fps,
        export_fps=None,
        start_s=float(start_s),
        end_s=None if end_s is None else float(end_s),
    )


def run_analyze_job(
    workspace: Path,
    job_id: str,
    *,
    detector: DetectorAdapter | None = None,
    reader_factory: Callable[[Path, str], _FrameSource] | None = None,
) -> int:
    """Worker entry. Opens the detector in this process when ``detector`` is omitted."""
    root = Path(workspace).resolve()
    project_id, run_dir, _record = worker_job(root, job_id)
    if not _wait_until_running(root, project_id, job_id):
        record = get_job(root, project_id, job_id)
        if record.status in {JobStatus.queued, JobStatus.running}:
            _fail(root, project_id, job_id, "worker_start_timeout")
        return 1
    opened: DetectorAdapter | None = None
    try:
        request = _read_request(run_dir)
        video = get_video(root, project_id, request["video_id"])
        _require_media(video)
        origin, bundle = resolve_model(root, project_id, request["model_id"])
        if detector is None:
            opened = open_analysis_detector(origin, bundle)
            detector = opened
        reader = _open_reader(video, reader_factory)
        model_version = _model_version(request["model_id"], origin)
        pipeline = AnalyzePipeline(
            reader,
            detector,
            run_id=job_id,
            output_dir=run_dir,
            analysis_fps=request["analysis_fps"],
            model_version=model_version,
            start_s=request["start_s"],
            end_s=request["end_s"],
            view_id=request["view_id"],
            reset_reason=request["reset_reason"],
            on_recorded=_progress(root, project_id, job_id, run_dir, _duration_s(video)),
            vehicles=_vehicles_for(request),
        )
        _apply_display(pipeline, request.get("display_threshold"))
        frame_path, detection_path, track_path = pipeline.run()
        if get_job(root, project_id, job_id).status != JobStatus.running:
            return 0
        _save_run_record(
            root,
            project_id,
            run_dir,
            run_id=job_id,
            job_id=job_id,
            video_id=request["video_id"],
            view_id=request["view_id"],
            model_id=request["model_id"],
            model_origin=origin,
            analysis_fps=request["analysis_fps"],
            source_fps=source_fps_of(video),
            start_s=request["start_s"],
            end_s=request["end_s"],
        )
        mark_completed(root, project_id, job_id, [frame_path, detection_path, track_path])
        return 0
    except _Cancelled:
        return 0
    except LabError as exc:
        _fail(root, project_id, job_id, exc.code)
        return 1
    except Exception as exc:  # noqa: BLE001 — the job row keeps the failure
        _fail(root, project_id, job_id, _exception_text(exc))
        return 1
    finally:
        if opened is not None:
            opened.close()


def resolve_model(
    workspace: Path,
    project_id: str,
    model_id: str,
) -> tuple[str, Path | None]:
    """Return ``(origin, bundle_dir)``. Does not construct a detector.

    ``baseline`` and ``baseline_coco`` select ``baseline_coco``. Any other id
    must already be a custom bundle. A missing bundle raises
    ``CheckpointMissingError`` and does not fall back to baseline.
    """
    if model_id in _BASELINE_IDS:
        return CheckpointOrigin.baseline_coco.value, None
    _require_model_id(model_id)
    directory = _bundle_directory(workspace, project_id, model_id)
    weights = directory / "weights.pth"
    if not weights.is_file():
        raise CheckpointMissingError(str(weights))
    return CheckpointOrigin.custom.value, directory


def open_analysis_detector(origin: str, bundle_dir: Path | None) -> DetectorAdapter:
    """Open the detector. Called from the worker, not from ``enqueue_analyze``."""
    if os.environ.get(FAKE_DETECTOR_ENV) == "1":
        return FakeDetector()
    if origin == CheckpointOrigin.baseline_coco.value:
        from excavation_lab.models.baseline import open_detector

        return open_detector(CheckpointOrigin.baseline_coco.value)
    if bundle_dir is None:
        raise CheckpointMissingError()
    from excavation_lab.models.bundle import load_bundle

    return load_bundle(bundle_dir)


def source_fps_of(video: VideoRecord) -> float | None:
    """Nominal FPS, otherwise average FPS. Never the analysis FPS, never 0."""
    nominal = _positive_fps(video["fps_nominal"])
    if nominal is not None:
        return nominal
    return _positive_fps(video["fps_average"])


def fps_step(analysis_fps: float, duration_s: float) -> float:
    """A positive step strictly below ``duration_s`` for the job row.

    The analyze worker does not sleep on this value. The run row keeps
    ``analysis_fps`` separately.
    """
    period = 1.0 / analysis_fps
    if period < duration_s:
        return period
    return duration_s / 2.0


def _configured_nms_iou(override: float | None) -> float | None:
    """Pilot threshold unless the caller passed one.

    ``configs/pilot.yaml`` stores null, so v0.1 keeps same-class duplicates.
    """
    if override is not None:
        return override
    return load_pilot().same_class_nms_iou


def format_analyze_summary(enqueued: AnalyzeEnqueue) -> str:
    fps = f"{enqueued.analysis_fps:g}"
    return (
        f"Анализ поставлен в очередь: {enqueued.job_id}. "
        f"Видео {enqueued.video_id}. Модель {enqueued.model_id}. "
        f"Частота анализа {fps} кадр/с."
    )


def _open_reader(
    video: VideoRecord,
    reader_factory: Callable[[Path, str], _FrameSource] | None,
) -> _FrameSource:
    path = Path(str(video["path"]))
    if reader_factory is not None:
        return reader_factory(path, video["video_id"])
    return FrameReader(path, video_id=video["video_id"])


def _model_version(model_id: str, origin: str) -> str:
    if origin == CheckpointOrigin.baseline_coco.value:
        return CheckpointOrigin.baseline_coco.value
    return model_id


def _progress(
    workspace: Path,
    project_id: str,
    job_id: str,
    run_dir: Path,
    video_duration_s: float,
) -> Callable[[float], None]:
    def on_recorded(source_time_s: float) -> None:
        # One timestamp, not the frames read so far.
        write_progress(
            run_dir,
            {
                "job_id": job_id,
                "video_duration_s": video_duration_s,
                "analyzed_end_s": source_time_s,
                "frames": [{"source_time_s": source_time_s}],
            },
        )
        current = get_job(workspace, project_id, job_id)
        if current.status in {JobStatus.cancel_requested, JobStatus.cancelled}:
            raise _Cancelled()

    return on_recorded


def _save_run_record(
    workspace: Path,
    project_id: str,
    run_dir: Path,
    *,
    run_id: str,
    job_id: str,
    video_id: str,
    view_id: str | None,
    model_id: str,
    model_origin: str,
    analysis_fps: float,
    source_fps: float | None,
    start_s: float,
    end_s: float | None,
) -> None:
    """Persist the run. Export FPS and decode/encode timings are null."""
    insert_analysis_run(
        workspace,
        project_id,
        run_id=run_id,
        job_id=job_id,
        video_id=video_id,
        view_id=view_id,
        model_id=model_id,
        model_origin=model_origin,
        analysis_fps=analysis_fps,
        source_fps=source_fps,
        export_fps=None,
        start_s=start_s,
        end_s=end_s,
        decode_ms=None,
        encode_ms=None,
    )
    _atomic_json(
        run_dir / RUN_RECORD_NAME,
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "project_id": project_id,
            "job_id": job_id,
            "video_id": video_id,
            "view_id": view_id,
            "model_id": model_id,
            "model_origin": model_origin,
            "analysis_fps": analysis_fps,
            "source_fps": source_fps,
            "export_fps": None,
            "start_s": start_s,
            "end_s": end_s,
            "decode_ms": None,
            "encode_ms": None,
        },
    )


def _frame_row(
    frame: ReadFrame,
    *,
    run_id: str,
    view_id: str | None,
    analyzed: bool,
    model_version: str,
    inference_ms: float | None,
    num_detections: int,
    analysis_index: int | None,
    bucket_observation_state: str | None = None,
    diagnostic_flags: list[str] | None = None,
) -> dict[str, Any]:
    row = {name: None for name in FRAME_STATUS_COLUMNS}
    row.update(
        {
            "run_id": run_id,
            "video_id": frame.video_id,
            "view_id": view_id,
            "source_frame_index": frame.source_frame_index,
            "source_time_s": frame.source_time_s,
            "decode_status": frame.decode_status.value,
            "analyzed": analyzed,
            "model_version": model_version,
            "inference_ms": inference_ms,
            "num_detections": num_detections,
            "bucket_observation_state": bucket_observation_state,
            "diagnostic_flags": [] if diagnostic_flags is None else list(diagnostic_flags),
            "source_pts": frame.source_pts,
            "time_base": frame.time_base,
            "analysis_index": analysis_index,
            "width": frame.width,
            "height": frame.height,
            "decode_ms": None,
            "encode_ms": None,
        }
    )
    return row


def _detection_rows(
    frame: ReadFrame,
    detections: FrameDetections,
    *,
    run_id: str,
    view_id: str | None,
    model_version: str,
    analysis_index: int,
    assignments: tuple[TrackedRow, ...] = (),
    bucket_frame: BucketFrame | None = None,
    bed_frame: BedFrame | None = None,
    observation_states: Mapping[int, ObservationState] | None = None,
) -> list[dict[str, Any]]:
    public_ids = {
        item.index: item.track_id
        for item in assignments
        if item.track_id is not None
    }
    bucket_links = _bucket_links(bucket_frame)
    bed_links = _bed_links(bed_frame)
    rows: list[dict[str, Any]] = []
    count = len(detections.class_name)
    for index in range(count):
        # Detector box. A tracked or predicted box is not written here.
        x1, y1, x2, y2 = (float(value) for value in detections.xyxy[index])
        track_id = public_ids.get(index)
        class_name = str(detections.class_name[index])
        link = _row_link(class_name, index, bucket_links, bed_links)
        if observation_states is not None and index in observation_states:
            observation_state = observation_states[index].value
        elif link is None:
            observation_state = None
        else:
            observation_state = link.association_state.value
        row = {name: None for name in DETECTIONS_COLUMNS}
        row.update(
            {
                "run_id": run_id,
                "observation_id": f"{run_id}:{analysis_index}:{index}",
                "video_id": frame.video_id,
                "view_id": view_id,
                "source_frame_index": frame.source_frame_index,
                "source_time_s": frame.source_time_s,
                "class_id": int(detections.class_id[index]),
                "class_name": str(detections.class_name[index]),
                "score": float(detections.score[index]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "frame_width": int(detections.frame_width),
                "frame_height": int(detections.frame_height),
                "track_id_nullable": track_id,
                "parent_track_id_nullable": None if link is None else link.parent_track_id,
                "observation_state": observation_state,
                "association_score_nullable": None if link is None else link.association_score,
                "model_version": model_version,
                "source_pts": frame.source_pts,
                "time_base": frame.time_base,
                "analysis_index": analysis_index,
                "width": frame.width,
                "height": frame.height,
            }
        )
        rows.append(row)
    return rows


def _bucket_links(frame: BucketFrame | None) -> dict[int, Any]:
    """Detector-row links only. A predicted dash has no index, so it is absent."""
    if frame is None:
        return {}
    return {
        item.index: item
        for item in frame.observations
        if item.index is not None and item.is_observed
    }


def _bed_links(frame: BedFrame | None) -> dict[int, Any]:
    """Detector-row bed links. The associator does not emit a synthetic bed."""
    if frame is None:
        return {}
    return {item.index: item for item in frame.links}


def _row_link(
    class_name: str,
    index: int,
    bucket_links: dict[int, Any],
    bed_links: dict[int, Any],
) -> Any:
    if class_name == BUCKET:
        return bucket_links.get(index)
    if class_name == TRUCK_BED:
        return bed_links.get(index)
    return None


def _first_tick(start_s: float, step: Fraction) -> Fraction:
    """First grid tick at or after ``start_s``. Ticks are ``n / analysis_fps`` from 0."""
    start = Fraction.from_float(float(start_s)).limit_denominator(1_000_000_000)
    whole = max(math.ceil((start / step) - _EDGE), 0)
    return Fraction(whole) * step


def _advance(due: Fraction, timestamp: float, step: Fraction) -> Fraction:
    """Next tick strictly after this frame and after the tick it satisfied.

    One source frame produces one observation. Missed ticks are not filled.
    """
    mark = Fraction.from_float(float(timestamp)).limit_denominator(1_000_000_000)
    nxt = due + step
    while nxt <= mark + _EDGE:
        nxt += step
    return nxt


def _tick_not_after(timestamp: float, step: Fraction) -> Fraction:
    """Latest grid tick at or before ``timestamp``.

    Ticks stay on ``n / analysis_fps`` from zero. The frame index is not a tick.
    """
    mark = Fraction.from_float(float(timestamp)).limit_denominator(1_000_000_000)
    if mark <= 0:
        return Fraction(0)
    return (mark // step) * step


def _bundle_directory(workspace: Path, project_id: str, model_id: str) -> Path:
    candidate = project_directory(workspace, project_id) / "models" / model_id
    if candidate.is_dir() and not candidate.is_symlink():
        return candidate
    recorded = _recorded_model_path(workspace, project_id, model_id)
    if recorded is not None and recorded.is_dir() and not recorded.is_symlink():
        return recorded
    raise CheckpointMissingError(model_id)


def _recorded_model_path(workspace: Path, project_id: str, model_id: str) -> Path | None:
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT path FROM model_versions
            WHERE project_id = ? AND model_id = ?
            """,
            (project_id, model_id),
        ).fetchone()
    if row is None or not isinstance(row["path"], str):
        return None
    return Path(row["path"])


def _read_request(run_dir: Path) -> dict[str, Any]:
    path = run_dir / REQUEST_NAME
    if not path.is_file():
        raise LabError("invalid_job", "В каталоге задания нет analyze_request.json.")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise LabError("invalid_job", "analyze_request.json должен быть объектом.")
    video_id = document.get("video_id")
    model_id = document.get("model_id")
    fps = document.get("analysis_fps")
    start_s = document.get("start_s")
    end_s = document.get("end_s")
    view_id = document.get("view_id")
    if not isinstance(video_id, str) or not isinstance(model_id, str):
        raise LabError("invalid_job", "analyze_request.json не содержит видео и модель.")
    if isinstance(start_s, bool) or not isinstance(start_s, (int, float)):
        raise LabError("invalid_job", "analyze_request.json не содержит начало интервала.")
    if end_s is not None and (isinstance(end_s, bool) or not isinstance(end_s, (int, float))):
        raise LabError("invalid_job", "analyze_request.json содержит некорректный конец интервала.")
    if view_id is not None and not isinstance(view_id, str):
        raise LabError("invalid_job", "analyze_request.json содержит некорректный view_id.")
    checked = _require_fps(fps)
    require_half_open_interval(float(start_s), None if end_s is None else float(end_s))
    return {
        "video_id": video_id,
        "model_id": model_id,
        "analysis_fps": checked,
        "start_s": float(start_s),
        "end_s": None if end_s is None else float(end_s),
        "view_id": _optional_view(view_id),
        "reset_reason": _optional_reset_reason(document.get("reset_reason")),
        "display_threshold": _optional_display(document.get("display_threshold")),
        "bytetrack": _optional_bytetrack(document.get("bytetrack")),
    }


def _require_media(video: VideoRecord) -> None:
    raw = video["path"]
    if not isinstance(raw, str) or raw == "" or not Path(raw).is_file():
        raise MediaNotFoundError(raw if isinstance(raw, str) else None)


def _duration_s(video: VideoRecord) -> float:
    duration = video["duration_s"]
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise LabError(
            "invalid_video",
            "У видео нет длительности. Повторите probe и запустите анализ снова.",
        )
    number = float(duration)
    if not math.isfinite(number) or number <= 0:
        raise LabError(
            "invalid_video",
            "У видео нет длительности. Повторите probe и запустите анализ снова.",
        )
    return number


def _positive_fps(value: float | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _require_fps(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError("invalid_analysis_fps", _BAD_FPS)
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise LabError("invalid_analysis_fps", _BAD_FPS)
    return number


def _optional_display(value: object) -> float | None:
    """A new-run display threshold. ``None`` leaves the pilot value."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError(
            "invalid_threshold",
            "Порог показа должен быть числом. Завершённый прогон не изменён.",
        )
    number = float(value)
    floor = load_pilot().detection_floor
    if not math.isfinite(number) or number < floor or number > 1.0:
        raise LabError(
            "invalid_threshold",
            "Порог показа не ниже порога детекции и не выше единицы. "
            "Завершённый прогон не изменён.",
        )
    return number


def _optional_bytetrack(value: object) -> dict[str, Any] | None:
    """A new-run ByteTrack draft. ``None`` leaves the pilot config."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise LabError(
            "invalid_threshold",
            "Пороги ByteTrack должны быть объектом. Завершённый прогон не изменён.",
        )
    try:
        config = ByteTrackConfig.model_validate(dict(value))
    except ValidationError as exc:
        raise LabError("invalid_threshold", _threshold_message(exc)) from exc
    return config.model_dump(mode="json")


def _threshold_message(exc: ValidationError) -> str:
    text = " ".join(str(error.get("msg", "")) for error in exc.errors())
    if "activation" in text:
        return (
            "Порог активации не ниже порога высокой уверенности. "
            "Завершённый прогон не изменён."
        )
    return "Пороги ByteTrack не приняты. Завершённый прогон не изменён."


def _vehicles_for(request: Mapping[str, Any]) -> VehicleTrackers | None:
    """Trackers for this new run. ``None`` keeps ``from_pilot``.

    Called from the worker after the request is read. Not from the UI.
    """
    raw = request.get("bytetrack")
    display = request.get("display_threshold")
    if raw is None and display is None:
        return None
    pilot = load_pilot()
    config = pilot.bytetrack if raw is None else ByteTrackConfig.model_validate(raw)
    threshold = pilot.display_threshold if display is None else float(display)
    return VehicleTrackers(
        detection_floor=pilot.detection_floor,
        display_threshold=threshold,
        bytetrack=config,
        analysis_fps=float(request["analysis_fps"]),
    )


def _apply_display(pipeline: AnalyzePipeline, display: float | None) -> None:
    """Copy a new-run display threshold onto the associators and the ledger.

    ``set_display_threshold`` does not write parquet. The new run writes its
    own tables when ``pipeline.run`` starts.
    """
    if display is None:
        return
    pipeline.session.set_display_threshold(float(display))
    number = pipeline.session.vehicles.display_threshold
    pipeline.session.bucket.display_threshold = number
    pipeline.session.bed.display_threshold = number
    pipeline.states.display_threshold = number


def _optional_reset_reason(reset_reason: str | None) -> str | None:
    if reset_reason is None:
        return None
    if (
        not isinstance(reset_reason, str)
        or not reset_reason.strip()
        or "\x00" in reset_reason
    ):
        raise LabError("invalid_reset_reason", "Недопустимая причина сброса треков.")
    return reset_reason.strip()


def _optional_view(view_id: str | None) -> str | None:
    if view_id is None:
        return None
    if (
        not isinstance(view_id, str)
        or not view_id.strip()
        or view_id in {".", ".."}
        or "/" in view_id
        or "\\" in view_id
        or "\x00" in view_id
    ):
        raise LabError("invalid_view", "Недопустимый идентификатор ракурса.")
    return view_id


def _require_model_id(model_id: str) -> None:
    if (
        not isinstance(model_id, str)
        or not model_id
        or model_id in {".", ".."}
        or "/" in model_id
        or "\\" in model_id
        or "\x00" in model_id
    ):
        raise CheckpointMissingError(model_id if isinstance(model_id, str) else None)


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
        return "analyze_failed"
    return text[:2000]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
