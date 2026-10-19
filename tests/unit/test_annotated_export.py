"""Annotated MP4 stays on the preview clock. Synthetic clips are not pit-video evidence."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
import signal
import sqlite3
import subprocess
import time
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import ImageDraw
from typer.testing import CliRunner

from excavation_lab import jobs
from excavation_lab.cli import app
from excavation_lab.datasets.catalog import (
    add_class,
    list_catalogs,
    require_project_catalog,
)
from excavation_lab.errors import LabError
from excavation_lab.jobs import (
    get_job,
    lab_database,
    request_cancel,
    run_directory,
    start_job,
)
from excavation_lab.media.encoder import (
    EXPORT_HOLD_ENV,
    GAP_SLATE_TEXT,
    render_gap_slate,
)
from excavation_lab.media.probe import add_video, ffprobe_executable
from excavation_lab.media.reader import FrameReader
from excavation_lab.models.adapter import (
    CheckpointOrigin,
    FakeDetector,
    ScriptedDetection,
)
from excavation_lab.schemas import JobStatus
from excavation_lab.storage import (
    connect,
    create_project,
    get_analysis_run,
    get_video,
    insert_analysis_run,
    project_directory,
    project_sqlite,
)
from excavation_lab.tracking import analyze as analyze_mod
from excavation_lab.tracking.analyze import AnalyzePipeline
from excavation_lab.tracking.export_tables import DETECTIONS_NAME, TRACK_STATES_NAME
from excavation_lab.visualization import export_video
from excavation_lab.visualization.annotate import (
    BASELINE_CAPTION,
    CLASS_COLORS,
    AnnotateToggles,
    DrawBox,
    TrailSegment,
    ZonePolygon,
    annotate_rgb,
    bucket_trail,
    label_text,
)
from excavation_lab.visualization.export_video import (
    DEFAULT_EXPORT_FPS,
    _titles,
    encode_annotated_video,
    enqueue_video_export,
    run_export_job,
)

_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SYNTHETIC_SPEC = importlib.util.spec_from_file_location(
    "excavation_lab_synthetic_mp4",
    _SYNTHETIC_PATH,
)
assert _SYNTHETIC_SPEC is not None and _SYNTHETIC_SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SYNTHETIC_SPEC)
_SYNTHETIC_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_mp4 = _SYNTHETIC.write_synthetic_mp4

_RUNNER = CliRunner()
_PROJECT = "pilot"
_VIDEO = "clip"
_RUN = "analysis-run"
_OUTPUT_FPS = 10.0
_MAP_COLUMNS = [
    "preview_time_s",
    "source_time_s",
    "source_frame_index",
    "repeated",
    "gap_slate",
]
_BODY = (2.0, 2.0, 30.0, 28.0)
_BUCKET_0 = (4.0, 30.0, 16.0, 44.0)
_BUCKET_1 = (16.0, 30.0, 28.0, 44.0)
_OLD_LAB_SQL = """
CREATE TABLE lab_jobs (
    job_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    worker TEXT NOT NULL,
    status TEXT NOT NULL,
    pid INTEGER,
    partial INTEGER NOT NULL DEFAULT 0,
    analyzed_end_s REAL,
    error_code TEXT,
    orphaned INTEGER NOT NULL DEFAULT 0,
    run_dir TEXT NOT NULL,
    video_duration_s REAL NOT NULL,
    step_s REAL NOT NULL,
    sleep_s REAL NOT NULL,
    created_at TEXT NOT NULL,
    CHECK (kind IN ('analyze', 'train'))
)
"""


class _DummyProcess:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.pid = os.getpid()

    def poll(self) -> None:
        return None


def _dummy_spawn(*_args: object, **_kwargs: object) -> _DummyProcess:
    return _DummyProcess()


class _Schedule(FakeDetector):
    """Body on every analyzed frame. Bucket only on the first two."""

    def predict(self, frame_rgb: np.ndarray, timestamp_s: float):  # type: ignore[no-untyped-def]
        if timestamp_s < 0.1:
            boxes = (_scripted(_BODY, 0, "excavator_body"), _scripted(_BUCKET_0, 1, "bucket"))
        elif timestamp_s < 0.3:
            boxes = (_scripted(_BODY, 0, "excavator_body"), _scripted(_BUCKET_1, 1, "bucket"))
        else:
            boxes = (_scripted(_BODY, 0, "excavator_body"),)
        self._scripted = boxes
        return super().predict(frame_rgb, timestamp_s)


def _scripted(
    xyxy: tuple[float, float, float, float],
    class_id: int,
    class_name: str,
) -> ScriptedDetection:
    return ScriptedDetection(xyxy, class_id, 0.95, class_name=class_name)


def _custom_detector() -> _Schedule:
    return _Schedule(
        class_names=("excavator_body", "bucket", "haul_truck", "truck_bed"),
        origin=CheckpointOrigin.custom,
    )


def _truck_detector() -> FakeDetector:
    return FakeDetector(
        class_names=("truck",),
        scripted=(_scripted((8.0, 8.0, 40.0, 36.0), 0, "truck"),),
        origin=CheckpointOrigin.baseline_coco,
    )


def _prepare(
    tmp_path: Path,
    detector: FakeDetector,
    *,
    analysis_fps: float,
    origin: str,
) -> tuple[Path, Path]:
    """Register one completed analysis. The export parent does not run it."""
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    create_project(_PROJECT, workspace, catalog_id="excavation")
    source = workspace / "clip.mp4"
    write_synthetic_mp4(source, 1.0, 30)
    add_video(workspace, _PROJECT, source, video_id=_VIDEO)
    tables = project_directory(workspace, _PROJECT) / "runs" / _RUN
    pipeline = AnalyzePipeline(
        FrameReader(source, video_id=_VIDEO),
        detector,
        run_id=_RUN,
        output_dir=tables,
        analysis_fps=analysis_fps,
        model_version="fake-1",
        start_s=0.0,
        end_s=1.0,
    )
    pipeline.run()
    with connect(project_sqlite(workspace, _PROJECT)) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                job_id, project_id, status, pid, partial,
                analyzed_end_s, error_code, created_at
            ) VALUES (?, ?, 'completed', NULL, 0, NULL, NULL, ?)
            """,
            (_RUN, _PROJECT, "2026-10-10T00:00:00+00:00"),
        )
    insert_analysis_run(
        workspace,
        _PROJECT,
        run_id=_RUN,
        job_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        model_id="fake-1",
        model_origin=origin,
        analysis_fps=analysis_fps,
        source_fps=30.0,
        export_fps=None,
        start_s=0.0,
        end_s=1.0,
        decode_ms=None,
        encode_ms=None,
    )
    return workspace, tables


def _forbid_detector(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"n": 0}
    real_init = FakeDetector.__init__

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parent constructed a detector")

    def counting(self: FakeDetector, *args: object, **kwargs: object) -> None:
        calls["n"] += 1
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(analyze_mod, "open_analysis_detector", boom)
    monkeypatch.setattr(FakeDetector, "__init__", counting)
    return calls


def _capture_text(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    real = ImageDraw.ImageDraw.text

    def spy(self: ImageDraw.ImageDraw, xy: object, text: object = None, *args: object, **kwargs: object):
        value = text if isinstance(text, str) else kwargs.get("text")
        if isinstance(value, str):
            seen.append(value)
        if text is None:
            return real(self, xy, *args, **kwargs)
        return real(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    return seen


def _read_map(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == _MAP_COLUMNS
        rows: list[dict[str, object]] = []
        for raw in reader:
            index_text = raw["source_frame_index"]
            rows.append(
                {
                    "preview_time_s": float(raw["preview_time_s"]),
                    "source_time_s": float(raw["source_time_s"]),
                    "source_frame_index": None if index_text == "" else int(index_text),
                    "repeated": raw["repeated"] == "true",
                    "gap_slate": raw["gap_slate"] == "true",
                }
            )
    return rows


def _assert_silent_cfr(path: Path, output_fps: float) -> float:
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
    assert completed.returncode == 0
    payload = json.loads(completed.stdout.decode("utf-8"))
    streams = payload["streams"]
    video = [item for item in streams if item["codec_type"] == "video"]
    audio = [item for item in streams if item["codec_type"] == "audio"]
    assert len(video) == 1
    assert audio == []
    rate = video[0]["avg_frame_rate"]
    numerator, denominator = rate.split("/", 1)
    assert float(numerator) / float(denominator) == pytest.approx(output_fps)
    return float(payload["format"]["duration"])


def _decoded_rgb(path: Path) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        assert not container.streams.audio
        for frame in container.decode(container.streams.video[0]):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def _mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.int16) - right.astype(np.int16))))


def _predicted_bucket_rows(tables: Path) -> list[dict[str, object]]:
    tracks = pq.read_table(tables / TRACK_STATES_NAME).to_pylist()
    return [
        row
        for row in tracks
        if row["class_name"] == "bucket"
        and row["state"] == "predicted"
        and row["is_observed"] is False
        and row["x1_nullable"] is not None
    ]


def _assert_predicted_absent(tables: Path) -> None:
    detections = pq.read_table(tables / DETECTIONS_NAME).to_pylist()
    assert detections
    assert all(row["observation_state"] != "predicted" for row in detections)
    predicted = _predicted_bucket_rows(tables)
    assert predicted
    for item in predicted:
        stamp = float(item["source_time_s"])
        same_time = [
            row
            for row in detections
            if row["class_name"] == "bucket"
            and abs(float(row["source_time_s"]) - stamp) <= 1e-3
        ]
        assert same_time == []
        box = (
            float(item["x1_nullable"]),
            float(item["y1_nullable"]),
            float(item["x2_nullable"]),
            float(item["y2_nullable"]),
        )
        for row in detections:
            other = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            if other == pytest.approx(box):
                assert abs(float(row["source_time_s"]) - stamp) > 1e-3


def test_labels_keep_checkpoint_names_and_dash_predicted_trails() -> None:
    """Baseline ``truck`` is not a project class. A long gap draws no line."""
    titles = {"truck": "Экскаватор: корпус", "bucket": "Ковш"}
    baseline = label_text(
        DrawBox("truck", (0.0, 0.0, 10.0, 10.0), score=0.8, track_id="t-1"),
        AnnotateToggles(),
        titles=titles,
        baseline=True,
    )
    assert baseline is not None
    assert baseline.startswith("truck")
    assert "excavator_body" not in baseline
    assert "Экскаватор" not in baseline
    custom = label_text(
        DrawBox("bucket", (0.0, 0.0, 10.0, 10.0), score=0.91, track_id="b-1"),
        AnnotateToggles(),
        titles=titles,
    )
    assert custom is not None
    assert custom.startswith("Ковш")
    assert "0.91" in custom
    omitted = label_text(
        DrawBox("bucket", (0.0, 0.0, 10.0, 10.0), score=None, predicted=True),
        AnnotateToggles(),
        titles=titles,
    )
    assert omitted == "Ковш"
    assert "0.00" not in (omitted or "")
    assert CLASS_COLORS["truck"] != CLASS_COLORS["excavator_body"]

    gap = bucket_trail(
        (
            ("b", 0.0, (0.0, 0.0, 10.0, 10.0)),
            ("b", 0.8, (20.0, 0.0, 30.0, 10.0)),
        ),
        source_time_s=0.8,
        trail_s=5.0,
        trail_gap_s=0.5,
        extrapolation_s=0.4,
    )
    assert gap == []
    predicted_box = (20.0, 0.0, 30.0, 10.0)
    segments = bucket_trail(
        (
            ("b", 0.0, (0.0, 0.0, 10.0, 10.0)),
            ("b", 0.2, (10.0, 0.0, 20.0, 10.0)),
        ),
        source_time_s=0.4,
        trail_s=5.0,
        trail_gap_s=0.5,
        predicted_boxes=(("b", predicted_box),),
        extrapolation_s=0.4,
    )
    solid = [item for item in segments if not item.predicted]
    dashed = [item for item in segments if item.predicted]
    assert len(solid) == 1
    assert len(dashed) == 1
    assert dashed[0].points[1] not in solid[0].points


def test_dashed_trail_and_empty_zones(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_text(monkeypatch)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    trail = (
        TrailSegment("bucket", ((4.0, 12.0), (60.0, 12.0)), predicted=True),
        TrailSegment("bucket", ((4.0, 30.0), (60.0, 30.0)), predicted=False),
    )
    toggles = AnnotateToggles(
        boxes=False,
        class_names=False,
        scores=False,
        track_ids=False,
        bucket_trail=True,
        zones=True,
    )
    untouched = annotate_rgb(image, [], toggles=toggles, trail=trail, zones=())
    assert int(image.sum()) == 0
    assert _has_gap(untouched, 12)
    assert not _has_gap(untouched, 30)
    zone = ZonePolygon("pit", ((2.0, 2.0), (40.0, 2.0), (20.0, 30.0)))
    off = AnnotateToggles(zones=False, bucket_trail=False, boxes=False, class_names=False)
    assert np.array_equal(annotate_rgb(image, [], toggles=off, zones=(zone,)), image)
    drawn = annotate_rgb(image, [], toggles=toggles, zones=(zone,))
    assert not np.array_equal(drawn, image)
    wide = np.zeros((64, 320, 3), dtype=np.uint8)
    burned = annotate_rgb(
        wide,
        [DrawBox("truck", (8.0, 20.0, 80.0, 50.0), score=0.8)],
        toggles=AnnotateToggles(),
        titles={"truck": "Экскаватор: корпус"},
        baseline=True,
        burn_baseline_caption=True,
    )
    assert not np.array_equal(burned, wide)
    assert BASELINE_CAPTION in seen
    assert any(item.startswith("truck") for item in seen)
    assert all("excavator_body" not in item for item in seen)
    assert all("Экскаватор" not in item for item in seen)
    with pytest.raises(LabError) as exc:
        ZonePolygon("pit", ((0.0, 0.0), (1.0, 1.0)))
    assert exc.value.code == "invalid_zone"


def _has_gap(image: np.ndarray, row: int) -> bool:
    band = image[row]
    lit = band[:, 0] > 180
    if int(lit.sum()) < 4:
        return False
    indexes = np.flatnonzero(lit)
    interior = band[int(indexes[0]) : int(indexes[-1]) + 1, 0]
    return bool(np.any(interior < 20))


def test_one_second_export_keeps_duration_and_skips_predicted_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, tables = _prepare(
        tmp_path,
        _custom_detector(),
        analysis_fps=5.0,
        origin="custom",
    )
    before = (tables / DETECTIONS_NAME).read_bytes()
    _assert_predicted_absent(tables)
    calls = _forbid_detector(monkeypatch)
    seen = _capture_text(monkeypatch)
    order: list[str] = []
    real_preview = export_video.write_preview
    real_probe = export_video.ffprobe_json
    real_done = export_video.mark_completed

    def preview(*args: object, **kwargs: object):
        order.append("encode")
        rows = real_preview(*args, **kwargs)
        order.append("closed")
        return rows

    def probe(path: Path):
        order.append("ffprobe")
        return real_probe(path)

    def done(*args: object, **kwargs: object):
        order.append("mark_completed")
        return real_done(*args, **kwargs)

    monkeypatch.setattr(export_video, "write_preview", preview)
    monkeypatch.setattr(export_video, "ffprobe_json", probe)
    monkeypatch.setattr(export_video, "mark_completed", done)
    monkeypatch.setattr(jobs, "_spawn", _dummy_spawn)

    enqueued = enqueue_video_export(workspace, _PROJECT, _RUN)
    assert calls["n"] == 0
    assert enqueued.export_fps == DEFAULT_EXPORT_FPS
    assert run_export_job(workspace, enqueued.job_id) == 0
    assert calls["n"] == 0
    assert order.index("closed") < order.index("ffprobe") < order.index("mark_completed")

    video = Path(enqueued.output)
    duration_s = _assert_silent_cfr(video, _OUTPUT_FPS)
    assert duration_s == pytest.approx(1.0, abs=1.0 / _OUTPUT_FPS)
    rows = _read_map(video.with_name("preview_time_map.csv"))
    assert rows[0]["preview_time_s"] == pytest.approx(0.0, abs=1e-6)
    observations = [
        row for row in rows if row["gap_slate"] is False and row["repeated"] is False
    ]
    assert len(observations) == 5
    assert any(row["repeated"] is True for row in rows)
    assert all(row["gap_slate"] is False for row in rows)
    assert all(0.0 <= float(row["source_time_s"]) < 1.0 for row in rows)
    assert (tables / DETECTIONS_NAME).read_bytes() == before
    _assert_predicted_absent(tables)
    stored = get_analysis_run(workspace, _PROJECT, _RUN)
    assert stored["export_fps"] == pytest.approx(_OUTPUT_FPS)
    assert stored["encode_ms"] is None or stored["encode_ms"] > 0
    assert stored["decode_ms"] is None
    sidecar = json.loads(video.with_name("annotated.json").read_text(encoding="utf-8"))
    assert sidecar["baseline"] is False
    assert sidecar["caption"] is None
    assert any(item.startswith("Ковш") for item in seen)
    assert any(item.startswith("Экскаватор: корпус") for item in seen)
    job = get_job(workspace, _PROJECT, enqueued.job_id)
    assert job.status == JobStatus.completed


def test_gap_slate_is_not_an_observation_and_truck_stays_truck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, tables = _prepare(
        tmp_path,
        _truck_detector(),
        analysis_fps=1.25,
        origin="baseline_coco",
    )
    before = (tables / DETECTIONS_NAME).read_bytes()
    detections = pq.read_table(tables / DETECTIONS_NAME).to_pylist()
    assert detections
    assert {row["class_name"] for row in detections} == {"truck"}
    seen = _capture_text(monkeypatch)
    media = get_video(workspace, _PROJECT, _VIDEO)
    source = Path(str(media["path"]))
    destination = tmp_path / "annotated.mp4"
    result = encode_annotated_video(
        video_path=source,
        video_id=_VIDEO,
        tables_dir=tables,
        output_path=destination,
        start_s=0.0,
        end_s=1.0,
        export_fps=_OUTPUT_FPS,
        baseline=True,
    )
    duration_s = _assert_silent_cfr(destination, _OUTPUT_FPS)
    assert duration_s == pytest.approx(1.0, abs=1.0 / _OUTPUT_FPS)
    rows = _read_map(result.map_path)
    slate = [row for row in rows if row["gap_slate"] is True]
    observations = [
        row for row in rows if row["gap_slate"] is False and row["repeated"] is False
    ]
    assert len(observations) == 2
    assert slate
    for row in slate:
        assert row["repeated"] is False
        assert row["source_frame_index"] is None
    decoded = _decoded_rgb(destination)
    assert len(decoded) == len(rows)
    reference = render_gap_slate(int(decoded[0].shape[1]), int(decoded[0].shape[0]))
    matched = [
        _mae(frame, reference) < 20
        for row, frame in zip(rows, decoded, strict=True)
        if row["gap_slate"] is True
    ]
    assert matched and all(matched)
    assert GAP_SLATE_TEXT == "нет наблюдения"
    assert (tables / DETECTIONS_NAME).read_bytes() == before
    sidecar = json.loads(result.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["caption"] == BASELINE_CAPTION
    assert BASELINE_CAPTION in seen
    assert any(item.startswith("truck") for item in seen)
    assert all("excavator_body" not in item for item in seen)
    stored = get_analysis_run(workspace, _PROJECT, _RUN)
    assert stored["export_fps"] is None
    assert stored["encode_ms"] is None


def test_failed_encode_does_not_mark_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, tables = _prepare(
        tmp_path,
        _custom_detector(),
        analysis_fps=5.0,
        origin="custom",
    )
    before = (tables / DETECTIONS_NAME).read_bytes()
    calls = _forbid_detector(monkeypatch)
    completed: list[str] = []

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("mid-encode")

    def done(*_args: object, **_kwargs: object) -> None:
        completed.append("mark_completed")

    monkeypatch.setattr(export_video, "write_preview", boom)
    monkeypatch.setattr(export_video, "mark_completed", done)
    monkeypatch.setattr(jobs, "_spawn", _dummy_spawn)
    enqueued = enqueue_video_export(workspace, _PROJECT, _RUN)
    assert run_export_job(workspace, enqueued.job_id) == 1
    job = get_job(workspace, _PROJECT, enqueued.job_id)
    assert job.status == JobStatus.failed
    assert job.status != JobStatus.completed
    assert completed == []
    assert calls["n"] == 0
    assert (tables / DETECTIONS_NAME).read_bytes() == before
    progress = run_directory(workspace, _PROJECT, enqueued.job_id) / "progress.json"
    assert progress.is_file()
    stored = get_analysis_run(workspace, _PROJECT, _RUN)
    assert stored["export_fps"] is None


def test_kill_during_encode_stays_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, tables = _prepare(
        tmp_path,
        _custom_detector(),
        analysis_fps=5.0,
        origin="custom",
    )
    before = (tables / DETECTIONS_NAME).read_bytes()
    monkeypatch.setenv(EXPORT_HOLD_ENV, "1")
    enqueued = enqueue_video_export(workspace, _PROJECT, _RUN)
    monkeypatch.delenv(EXPORT_HOLD_ENV, raising=False)
    run_dir = run_directory(workspace, _PROJECT, enqueued.job_id)
    progress = run_dir / "progress.json"
    pid = get_job(workspace, _PROJECT, enqueued.job_id).pid
    try:
        deadline = time.monotonic() + 20.0
        while not progress.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not progress.is_file():
            log = run_dir / "worker.log"
            detail = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
            raise AssertionError(detail)
        cancelled = request_cancel(workspace, _PROJECT, enqueued.job_id, kill_after_s=0.75)
        time.sleep(0.3)
        job = get_job(workspace, _PROJECT, enqueued.job_id)
        assert cancelled.status == JobStatus.cancelled
        assert job.status == JobStatus.cancelled
        assert job.partial is True
        assert job.analyzed_end_s is not None
        assert job.status != JobStatus.completed
    finally:
        if isinstance(pid, int) and pid > 0 and pid != os.getpid():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
    assert (tables / DETECTIONS_NAME).read_bytes() == before
    stored = get_analysis_run(workspace, _PROJECT, _RUN)
    assert stored["export_fps"] is None
    assert stored["encode_ms"] is None


def test_cli_video_enqueues_without_a_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _tables = _prepare(
        tmp_path,
        _truck_detector(),
        analysis_fps=1.25,
        origin="baseline_coco",
    )
    calls = _forbid_detector(monkeypatch)
    monkeypatch.setattr(jobs, "_spawn", _dummy_spawn)
    result = _RUNNER.invoke(
        app,
        [
            "export",
            "--project",
            _PROJECT,
            "--run",
            _RUN,
            "--format",
            "video",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    body = json.loads(result.stdout)
    assert body["ok"] is True
    assert body["data"]["export_fps"] == DEFAULT_EXPORT_FPS
    assert body["data"]["run_id"] == _RUN
    assert calls["n"] == 0
    requests = list(project_directory(workspace, _PROJECT).rglob("export_request.json"))
    assert len(requests) == 1
    payload = json.loads(requests[0].read_text(encoding="utf-8"))
    assert payload["zones"] == []
    assert payload["toggles"] == {
        "boxes": True,
        "class_names": True,
        "scores": True,
        "track_ids": True,
        "bucket_trail": True,
        "zones": True,
    }
    refused = _RUNNER.invoke(
        app,
        [
            "export",
            "--project",
            _PROJECT,
            "--run",
            _RUN,
            "--format",
            "events",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert refused.exit_code != 0
    assert json.loads(refused.stdout)["error"]["code"] == "unavailable"
    assert list(workspace.rglob("events.csv")) == []
    assert list(workspace.rglob("truck_visits.csv")) == []


def test_old_lab_jobs_accepts_export_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    create_project(_PROJECT, workspace, catalog_id="excavation")
    path = lab_database(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(_OLD_LAB_SQL)
    connection.commit()
    connection.close()
    monkeypatch.setattr(jobs, "_spawn", _dummy_spawn)
    record = start_job(
        workspace,
        _PROJECT,
        kind="export",
        worker="export",
        video_duration_s=10.0,
        step_s=1.0,
    )
    assert record.status == JobStatus.running
    with sqlite3.connect(path) as opened:
        sql = opened.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'lab_jobs'"
        ).fetchone()[0]
    assert "'export'" in sql


def test_drawer_does_not_call_bytetrack_or_write_tables() -> None:
    root = Path(__file__).resolve().parents[2]
    annotate_src = (root / "src/excavation_lab/visualization/annotate.py").read_text(
        encoding="utf-8"
    )
    export_src = (root / "src/excavation_lab/visualization/export_video.py").read_text(
        encoding="utf-8"
    )
    for node in ast.walk(ast.parse(annotate_src)):
        if isinstance(node, ast.Attribute) and node.attr == "ByteTrack":
            raise AssertionError(node.lineno)
    assert "shell=True" not in export_src
    assert "streamlit" not in export_src
    assert "to_parquet" not in export_src
    assert "add_detection" not in export_src
    assert "sv.ByteTrack" not in export_src


def test_custom_titles_come_from_the_project_catalog(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    create_project(_PROJECT, workspace, catalog_id="excavation")
    add_class(workspace, "excavation", "shovel", "Лопата", "Видимая лопата.")
    catalog = require_project_catalog(workspace, _PROJECT)
    titles = _titles(False, catalog)
    assert titles["shovel"] == "Лопата"
    assert titles["bucket"] == "Ковш"
    assert titles["excavator_body"] == "Экскаватор: корпус"
    seed = _titles(True, catalog)
    assert "shovel" not in seed
    assert seed["bucket"] == "Ковш"
    assert _titles(False) == {}
