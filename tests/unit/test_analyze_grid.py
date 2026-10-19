"""Analysis grid keeps reader PTS. Synthetic clips are not pit-video evidence."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from excavation_lab import jobs
from excavation_lab.cli import app
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.errors import CheckpointMissingError
from excavation_lab.media.reader import FrameReader, ReadFrame
from excavation_lab.media.timebase import TimeMode, in_half_open
from excavation_lab.models import baseline as baseline_mod
from excavation_lab.models import bundle as bundle_mod
from excavation_lab.models.adapter import (
    CheckpointOrigin,
    FakeDetector,
    FrameDetections,
    ScriptedDetection,
)
from excavation_lab.schemas import DecodeStatus, JobStatus
from excavation_lab.storage import connect, create_project, project_sqlite
from excavation_lab.tracking import analyze as analyze_mod
from excavation_lab.tracking.analyze import (
    FAKE_DETECTOR_ENV,
    AnalyzePipeline,
    enqueue_analyze,
    open_analysis_detector,
)
from excavation_lab.tracking.export_tables import (
    DETECTIONS_COLUMNS,
    FRAME_STATUS_COLUMNS,
)

_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SYNTHETIC_SPEC = importlib.util.spec_from_file_location(
    "excavation_lab_synthetic_mp4_grid",
    _SYNTHETIC_PATH,
)
assert _SYNTHETIC_SPEC is not None and _SYNTHETIC_SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SYNTHETIC_SPEC)
_SYNTHETIC_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_mp4 = _SYNTHETIC.write_synthetic_mp4

_RUNNER = CliRunner()
_FRAMES = 300
_SOURCE_FPS = 30


@pytest.fixture(scope="module")
def long_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("grid") / "long.mp4"
    write_synthetic_mp4(path, _FRAMES / _SOURCE_FPS, _SOURCE_FPS)
    return path


def test_intervals_keep_reader_pts_and_skip_the_rest(long_clip: Path, tmp_path: Path) -> None:
    intervals = ((0.0, 2.0), (4.0, 6.0), (9.0, 10.0))
    for start_s, end_s in intervals:
        out = tmp_path / f"{start_s:g}-{end_s:g}"
        rows = _analyze_file(long_clip, out, start_s, end_s)
        assert rows
        times = [float(row["source_time_s"]) for row in rows]
        assert times == sorted(times)
        assert all(times[index] < times[index + 1] for index in range(len(times) - 1))
        clock = _clock(long_clip, start_s, end_s)
        by_index = {index: stamp for index, stamp, status in clock if status is DecodeStatus.ok}
        for row in rows:
            assert row["decode_status"] == "ok"
            assert row["analyzed"] is True
            assert row["num_detections"] == 0
            assert row["inference_ms"] is not None
            assert row["decode_ms"] is None
            assert row["encode_ms"] is None
            assert in_half_open(float(row["source_time_s"]), start_s, end_s)
            index = int(row["source_frame_index"])
            assert float(row["source_time_s"]) == pytest.approx(by_index[index])
        if start_s > 0:
            assert rows[0]["analysis_index"] == 0
            assert float(rows[0]["source_time_s"]) != pytest.approx(0.0)
            assert float(rows[0]["source_time_s"]) != pytest.approx(
                rows[0]["analysis_index"] / 5
            )
        skipped = _skipped_ok_index(clock, {int(row["source_frame_index"]) for row in rows})
        assert skipped is not None
        assert skipped not in {int(row["source_frame_index"]) for row in rows}
        detections = _read(out / "detections.parquet")
        assert detections == []


def test_vfr_pts_is_not_replaced_by_analysis_index_over_fps(tmp_path: Path) -> None:
    path = tmp_path / "vfr.mp4"
    _write_vfr(path, [210, 160, 180])
    out = tmp_path / "out"
    rows = _analyze_file(path, out, 0.0, None)
    assert [int(row["analysis_index"]) for row in rows] == [0, 1]
    clock = _clock(path, 0.0, None)
    reader_time = next(
        stamp
        for index, stamp, _status in clock
        if index == int(rows[1]["source_frame_index"])
    )
    assert reader_time is not None
    assert float(rows[1]["source_time_s"]) == pytest.approx(reader_time)
    assert float(rows[1]["source_time_s"]) != pytest.approx(1 / 5)
    kept = {int(row["source_frame_index"]) for row in rows}
    skipped = [
        index
        for index, _stamp, status in clock
        if status is DecodeStatus.ok and index not in kept
    ]
    assert skipped


def test_empty_detector_still_records_the_frame_without_boxes(tmp_path: Path) -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[0, 0, 0] = 255
    frames = [
        _read_frame(0, 0.0, image),
        _read_frame(1, 0.05, image),
        _read_frame(2, 0.21, image),
    ]
    seen: list[np.ndarray] = []

    class _Empty:
        def predict(self, frame_rgb: np.ndarray, timestamp_s: float) -> FrameDetections:
            seen.append(frame_rgb)
            assert timestamp_s != 1  # analysis_index is never passed as seconds here
            return FrameDetections(
                timestamp_s=float(timestamp_s),
                frame_width=6,
                frame_height=4,
                origin=CheckpointOrigin.baseline_coco,
                xyxy=np.zeros((0, 4), dtype=np.float64),
                class_id=np.zeros((0,), dtype=np.int64),
                class_name=(),
                score=np.zeros((0,), dtype=np.float64),
            )

        def metadata(self):
            raise AssertionError("unused")

        def close(self) -> None:
            return None

    out = tmp_path / "empty"
    pipeline = AnalyzePipeline(
        _ListReader(frames),
        _Empty(),
        run_id="empty",
        output_dir=out,
        analysis_fps=5,
        model_version="baseline_coco",
    )
    pipeline.run()
    rows = _read(out / "frame_status.parquet")
    assert [float(row["source_time_s"]) for row in rows] == pytest.approx([0.0, 0.21])
    assert rows[1]["analysis_index"] == 1
    assert float(rows[1]["source_time_s"]) != pytest.approx(0.2)
    assert all(row["analyzed"] is True and row["num_detections"] == 0 for row in rows)
    assert _read(out / "detections.parquet") == []
    assert seen[0] is image
    assert seen[0][0, 0, 0] == 255
    assert seen[0][0, 0, 2] == 0
    assert 1 not in {int(row["source_frame_index"]) for row in rows}


def test_detector_boxes_keep_class_names_and_null_tracks(tmp_path: Path) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    detector = FakeDetector(
        class_names=("truck",),
        scripted=(ScriptedDetection((1.0, 2.0, 5.0, 6.0), 7, 0.42, "truck"),),
    )
    out = tmp_path / "boxes"
    pipeline = AnalyzePipeline(
        _ListReader([_read_frame(0, 0.21, image)]),
        detector,
        run_id="boxes",
        output_dir=out,
        analysis_fps=5,
        model_version="baseline_coco",
    )
    pipeline.run()
    frames = _read(out / "frame_status.parquet")
    assert frames[0]["analyzed"] is True
    assert frames[0]["num_detections"] == 1
    assert float(frames[0]["source_time_s"]) == pytest.approx(0.21)
    detections = _read(out / "detections.parquet")
    assert len(detections) == 1
    row = detections[0]
    assert row["class_name"] == "truck"
    assert row["class_id"] == 7
    assert float(row["score"]) == pytest.approx(0.42)
    assert [float(row[name]) for name in ("x1", "y1", "x2", "y2")] == pytest.approx(
        [1.0, 2.0, 5.0, 6.0]
    )
    assert row["track_id_nullable"] is None
    assert row["parent_track_id_nullable"] is None
    assert row["observation_state"] is None
    assert row["association_score_nullable"] is None
    assert set(FRAME_STATUS_COLUMNS) <= set(frames[0])
    assert set(DETECTIONS_COLUMNS) <= set(row)


def test_corrupt_span_is_not_downtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from excavation_lab.media import reader as reader_mod

    path = tmp_path / "corrupt.mp4"
    write_synthetic_mp4(path, 0.5, 10)
    original = reader_mod.decode_packet
    state = {"n": 0}

    def fail_once(packet: av.Packet):
        state["n"] += 1
        if state["n"] == 2:
            raise av.error.InvalidDataError(int(av.error.ErrorType.INVALIDDATA), "damaged")
        return original(packet)

    monkeypatch.setattr(reader_mod, "decode_packet", fail_once)
    out = tmp_path / "out"
    rows = _analyze_file(path, out, 0.0, None)
    corrupt = [row for row in rows if row["decode_status"] == "corrupt"]
    assert len(corrupt) == 1
    assert corrupt[0]["analyzed"] is False
    assert corrupt[0]["num_detections"] == 0
    assert corrupt[0]["inference_ms"] is None
    assert corrupt[0]["decode_ms"] is None
    assert corrupt[0]["encode_ms"] is None
    assert corrupt[0]["bucket_observation_state"] is None
    assert corrupt[0]["diagnostic_flags"] == []
    blob = json.dumps(corrupt[0])
    assert "downtime" not in blob
    assert "простой" not in blob
    ok = [row for row in rows if row["decode_status"] == "ok"]
    assert ok
    assert all(row["analyzed"] is True for row in ok)
    assert _read(out / "detections.parquet") == []


def test_unknown_model_does_not_fall_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _video_id = _project_with_clip(tmp_path, seconds=0.4, fps=10)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("baseline was constructed")

    monkeypatch.setattr(baseline_mod, "open_detector", boom)
    monkeypatch.setattr(baseline_mod, "_load_rfdetr_nano", boom)
    monkeypatch.setattr(bundle_mod, "load_bundle", boom)
    monkeypatch.setattr(analyze_mod, "open_analysis_detector", boom)
    with pytest.raises(CheckpointMissingError) as caught:
        enqueue_analyze(workspace, "pilot", "clip", "missing-model", 5)
    assert caught.value.code == "checkpoint_missing"
    assert "не подставляется" in caught.value.message


def test_custom_open_uses_the_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "custom-1"
    bundle.mkdir()
    (bundle / "weights.pth").write_bytes(b"not-a-coco-checkpoint")
    calls: list[Path] = []

    def load(path: Path, **_kwargs: object) -> FakeDetector:
        calls.append(Path(path))
        return FakeDetector()

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("baseline was constructed")

    monkeypatch.delenv(FAKE_DETECTOR_ENV, raising=False)
    monkeypatch.setattr(bundle_mod, "load_bundle", load)
    monkeypatch.setattr(baseline_mod, "open_detector", boom)
    detector = open_analysis_detector("custom", bundle)
    assert calls == [bundle]
    detector.close()


def test_parent_enqueue_does_not_construct_the_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _video_id = _project_with_clip(tmp_path, seconds=1.0, fps=30)
    calls = {"open": 0}

    def boom(*_args: object, **_kwargs: object) -> None:
        calls["open"] += 1
        raise AssertionError("parent constructed a detector")

    monkeypatch.setenv(FAKE_DETECTOR_ENV, "1")
    monkeypatch.setattr(analyze_mod, "open_analysis_detector", boom)
    monkeypatch.setattr(baseline_mod, "open_detector", boom)
    monkeypatch.setattr(bundle_mod, "load_bundle", boom)
    enqueued = enqueue_analyze(
        workspace,
        "pilot",
        "clip",
        "baseline",
        5,
        start_s=0.4,
        end_s=1.0,
    )
    assert calls["open"] == 0
    try:
        record = _wait_job(workspace, enqueued.job_id)
        assert record.status is JobStatus.completed
        assert record.error_code is None
        run_dir = jobs.run_directory(workspace, "pilot", enqueued.job_id)
        request = json.loads((run_dir / "analyze_request.json").read_text(encoding="utf-8"))
        assert request["model_origin"] == "baseline_coco"
        assert request["analysis_fps"] == 5
        assert request["start_s"] == pytest.approx(0.4)
        sidecar = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert sidecar["analysis_fps"] == 5
        assert sidecar["source_fps"] == pytest.approx(30)
        assert sidecar["export_fps"] is None
        assert sidecar["decode_ms"] is None
        assert sidecar["encode_ms"] is None
        assert sidecar["analysis_fps"] != sidecar["source_fps"]
        with connect(project_sqlite(workspace, "pilot")) as connection:
            row = connection.execute(
                "SELECT analysis_fps, source_fps, export_fps, decode_ms, encode_ms FROM runs"
            ).fetchone()
        assert row["analysis_fps"] == 5
        assert row["source_fps"] == pytest.approx(30)
        assert row["export_fps"] is None
        assert row["decode_ms"] is None
        assert row["encode_ms"] is None
        rows = _read(run_dir / "frame_status.parquet")
        assert rows
        assert rows[0]["analysis_index"] == 0
        assert float(rows[0]["source_time_s"]) == pytest.approx(0.4)
        assert float(rows[0]["source_time_s"]) != pytest.approx(0.0)
        times = [float(row["source_time_s"]) for row in rows]
        assert times == sorted(times)
        assert all(in_half_open(stamp, 0.4, 1.0) for stamp in times)
    finally:
        _stop(jobs.get_job(workspace, "pilot", enqueued.job_id).pid)


def test_cli_json_unknown_model_and_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _video_id = _project_with_clip(tmp_path, seconds=0.5, fps=10)
    denied = _RUNNER.invoke(
        app,
        [
            "analyze",
            "--project",
            "pilot",
            "--video",
            "clip",
            "--model",
            "missing-model",
            "--fps",
            "5",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert denied.exit_code == 1
    assert "Traceback" not in denied.output
    payload = json.loads(denied.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "checkpoint_missing"
    assert "не подставляется" in payload["error"]["message"]

    monkeypatch.setenv(FAKE_DETECTOR_ENV, "1")
    opened = {"n": 0}
    real_open = analyze_mod.open_analysis_detector

    def spy(*args: object, **kwargs: object):
        opened["n"] += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(analyze_mod, "open_analysis_detector", spy)
    accepted = _RUNNER.invoke(
        app,
        [
            "analyze",
            "--project",
            "pilot",
            "--video",
            "clip",
            "--model",
            "baseline",
            "--fps",
            "5",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert accepted.exit_code == 0
    assert "Traceback" not in accepted.output
    body = json.loads(accepted.stdout)
    assert body["ok"] is True
    assert body["data"]["model_origin"] == "baseline_coco"
    assert body["data"]["analysis_fps"] == 5
    assert body["data"]["export_fps"] is None
    assert body["data"]["source_fps"] == pytest.approx(10)
    assert opened["n"] == 0
    job_id = body["data"]["job_id"]
    try:
        record = _wait_job(workspace, job_id)
        assert record.status is JobStatus.completed
    finally:
        _stop(jobs.get_job(workspace, "pilot", job_id).pid)


def test_analyze_module_has_one_writer() -> None:
    source = Path(analyze_mod.__file__).read_text(encoding="utf-8")
    assert "write_analysis_tables" in source
    assert "to_parquet" not in source
    assert "write_table" not in source
    assert "streamlit" not in source
    assert "from trackers import" not in source
    assert "sv.ByteTrack" not in source


class _ListReader:
    def __init__(self, frames: list[ReadFrame]) -> None:
        self.frames = frames

    def iterate(self, start_s: float = 0.0, end_s: float | None = None):
        for frame in self.frames:
            stamp = frame.source_time_s
            if stamp is not None and not in_half_open(stamp, start_s, end_s):
                continue
            yield frame


def _analyze_file(
    path: Path,
    out: Path,
    start_s: float,
    end_s: float | None,
) -> list[dict[str, object]]:
    detector = FakeDetector()
    pipeline = AnalyzePipeline(
        FrameReader(path, video_id="clip"),
        detector,
        run_id="grid",
        output_dir=out,
        analysis_fps=5,
        model_version="baseline_coco",
        start_s=start_s,
        end_s=end_s,
    )
    pipeline.run()
    rows = _read(out / "frame_status.parquet")
    assert len(detector.boundary_frames) <= 1
    return rows


def _clock(
    path: Path,
    start_s: float,
    end_s: float | None,
) -> list[tuple[int | None, float | None, DecodeStatus]]:
    rows: list[tuple[int | None, float | None, DecodeStatus]] = []
    for frame in FrameReader(path, video_id="clip").iterate(start_s, end_s):
        rows.append((frame.source_frame_index, frame.source_time_s, frame.decode_status))
    return rows


def _skipped_ok_index(
    clock: list[tuple[int | None, float | None, DecodeStatus]],
    kept: set[int],
) -> int | None:
    for index, _stamp, status in clock:
        if status is DecodeStatus.ok and index is not None and index not in kept:
            return index
    return None


def _read(path: Path) -> list[dict[str, object]]:
    table = pq.read_table(path)
    return table.to_pylist()


def _read_frame(index: int, time_s: float, image: np.ndarray) -> ReadFrame:
    return ReadFrame(
        video_id="clip",
        source_frame_index=index,
        source_pts=int(time_s * 1000),
        time_base_numerator=1,
        time_base_denominator=1000,
        source_time_s=time_s,
        span_end_s=None,
        start_offset_s=0.0,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        coded_width=int(image.shape[1]),
        coded_height=int(image.shape[0]),
        rotation_deg=0,
        image=image,
        decode_status=DecodeStatus.ok,
        time_mode=TimeMode.pts,
    )


def _write_vfr(path: Path, durations_ms: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 32
    height = 24
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(0))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.options = {"preset": "ultrafast", "g": "1", "keyint": "1", "scenecut": "0"}
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        cursor = 0
        for duration_ms in durations_ms:
            frame = av.VideoFrame.from_ndarray(blank, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            frame.pts = cursor
            frame.time_base = Fraction(1, 1000)
            cursor += duration_ms
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def _project_with_clip(tmp_path: Path, *, seconds: float, fps: int) -> tuple[Path, str]:
    from excavation_lab.media.probe import add_video

    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    create_project("pilot", workspace, catalog_id="excavation")
    clip = workspace / "clip.mp4"
    write_synthetic_mp4(clip, seconds, fps)
    added = add_video(workspace, "pilot", clip, video_id="clip")
    assert added["video_id"] == "clip"
    return workspace, "clip"


def _wait_job(workspace: Path, job_id: str):
    deadline = time.monotonic() + 30
    record = jobs.get_job(workspace, "pilot", job_id)
    while time.monotonic() < deadline:
        record = jobs.get_job(workspace, "pilot", job_id)
        if record.status in {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}:
            if record.status is not JobStatus.completed:
                log = jobs.run_directory(workspace, "pilot", job_id) / "worker.log"
                text = log.read_text(encoding="utf-8") if log.is_file() else ""
                raise AssertionError(f"{record.status} {record.error_code}\n{text}")
            return record
        time.sleep(0.05)
    raise AssertionError(f"job still {record.status}")


def _stop(pid: int | None) -> None:
    if pid is None or pid == os.getpid():
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        return
