"""Track ids are namespaced. A seek resets both vehicle trackers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from trackers import ByteTrackTracker

from excavation_lab.errors import LabError
from excavation_lab.media.reader import ReadFrame
from excavation_lab.media.timebase import TimeMode, in_half_open
from excavation_lab.models.adapter import (
    CheckpointOrigin,
    FakeDetector,
    FrameDetections,
    ScriptedDetection,
)
from excavation_lab.schemas import DecodeStatus
from excavation_lab.tracking.analyze import REQUEST_NAME, AnalyzePipeline, _read_request
from excavation_lab.tracking.session import (
    RESET_ANALYSIS,
    RESET_SEEK,
    RESET_VIDEO_ID,
    RESET_VIEW_ID,
    UNSET_VIEW_ID,
    TrackingSession,
    public_track_id,
)
from excavation_lab.tracking.vehicles import VehicleTrackers

_TRUCK_BOX = (10.0, 10.0, 80.0, 70.0)
_BODY_BOX = (100.0, 10.0, 170.0, 70.0)


def test_public_track_id_differs_by_video_and_run() -> None:
    same_local = 7
    video_a = public_track_id("run-1", "video-a", None, "haul_truck", same_local)
    video_b = public_track_id("run-1", "video-b", None, "haul_truck", same_local)
    second_run = public_track_id("run-2", "video-a", None, "haul_truck", same_local)
    assert video_a == f"run-1:video-a:{UNSET_VIEW_ID}:haul_truck:7"
    assert video_a != video_b
    assert video_a != second_run
    assert video_b != second_run
    assert public_track_id("run-1", "video-a", None, "haul_truck", same_local) == video_a
    assert public_track_id("run-1", "video-a", "", "haul_truck", same_local) == video_a
    chosen = public_track_id("run-1", "video-a", "north", "excavator_body", same_local)
    assert chosen == "run-1:video-a:north:excavator_body:7"


def test_forward_pass_keeps_the_local_id_and_unset_view() -> None:
    session = _session()
    rows = [_haul(_step(session, stamp)) for stamp in (0.0, 0.05, 0.10)]
    assert session.reset_count == 0
    assert rows[0].local_track_id is None
    assert rows[1].local_track_id == 0
    assert rows[2].local_track_id == rows[1].local_track_id
    assert rows[2].track_id == rows[1].track_id
    assert rows[2].track_id == f"run-1:video-a:{UNSET_VIEW_ID}:haul_truck:0"
    assert rows[2].xyxy == pytest.approx(_TRUCK_BOX)
    body = _body(_step(session, 0.15))
    assert body.track_id == f"run-1:video-a:{UNSET_VIEW_ID}:excavator_body:0"


def test_seek_backward_resets_trackers_and_issues_a_new_local_id() -> None:
    session = _session()
    assert isinstance(session.vehicles.haul_truck, ByteTrackTracker)
    assert isinstance(session.vehicles.excavator_body, ByteTrackTracker)
    assert type(session.vehicles.haul_truck) is ByteTrackTracker
    observed: list[tuple[str, list[object], list[object], dict[str, object]]] = []
    original = session.reset

    def spy(reason: str) -> None:
        original(reason)
        observed.append(
            (
                reason,
                list(session.vehicles.haul_truck.tracks),
                list(session.vehicles.excavator_body.tracks),
                dict(session.last_positions),
            )
        )

    session.reset = spy  # type: ignore[method-assign]
    first = [_step(session, stamp) for stamp in (0.0, 0.05, 0.10)]
    haul_before = [_haul(frame) for frame in first]
    body_before = [_body(frame) for frame in first]
    assert haul_before[1].local_track_id == haul_before[2].local_track_id == 0
    assert body_before[1].local_track_id == body_before[2].local_track_id == 0
    assert haul_before[1].track_id == haul_before[2].track_id
    assert session.vehicles.haul_truck.tracks
    assert session.vehicles.excavator_body.tracks
    assert session.last_positions
    assert observed == []

    seek = _step(session, 0.02)
    assert len(observed) == 1
    reason, haul_tracks, body_tracks, positions = observed[0]
    assert reason == RESET_SEEK
    assert haul_tracks == []
    assert body_tracks == []
    assert positions == {}
    assert session.last_reset_reason == RESET_SEEK
    seek_haul = _haul(seek)
    assert seek_haul.local_track_id != haul_before[2].local_track_id
    assert seek_haul.xyxy == pytest.approx(_TRUCK_BOX)

    followed = _step(session, 0.07)
    new_haul = _haul(followed)
    new_body = _body(followed)
    assert new_haul.local_track_id == 1
    assert new_body.local_track_id == 1
    assert new_haul.local_track_id != haul_before[2].local_track_id
    assert new_body.local_track_id != body_before[2].local_track_id
    assert new_haul.track_id == f"run-1:video-a:{UNSET_VIEW_ID}:haul_truck:1"
    assert new_haul.track_id != haul_before[2].track_id
    assert new_body.track_id != body_before[2].track_id
    assert {haul_before[2].track_id, body_before[2].track_id}.isdisjoint(
        {new_haul.track_id, new_body.track_id}
    )
    assert len(observed) == 1


def test_video_and_view_changes_may_repeat_the_local_id() -> None:
    session = _session()
    _step(session, 0.0)
    established = _haul(_step(session, 0.05))
    assert established.local_track_id == 0
    _step(session, 0.10, video="video-b")
    repeated = _haul(_step(session, 0.15, video="video-b"))
    assert session.last_reset_reason == RESET_VIDEO_ID
    assert repeated.local_track_id == established.local_track_id
    assert repeated.track_id != established.track_id
    assert repeated.track_id == "run-1:video-b:unset:haul_truck:0"
    assert session.vehicles.haul_truck.tracks

    _step(session, 0.20, video="video-b", view="south")
    south = _haul(_step(session, 0.25, video="video-b", view="south"))
    assert session.last_reset_reason == RESET_VIEW_ID
    assert south.local_track_id == 0
    assert south.track_id == "run-1:video-b:south:haul_truck:0"
    assert south.track_id != repeated.track_id
    assert south.xyxy == pytest.approx(_TRUCK_BOX)


def test_analysis_restart_and_explicit_reason_clear_state() -> None:
    session = _session()
    _step(session, 0.0)
    continued = _haul(_step(session, 0.05))
    assert continued.local_track_id == 0
    assert session.last_positions
    session.restart()
    assert session.last_reset_reason == RESET_ANALYSIS
    assert session.vehicles.haul_truck.tracks == []
    assert session.vehicles.excavator_body.tracks == []
    assert session.last_positions == {}
    assert session.positions_for("haul_truck") == ()
    _step(session, 0.0)
    restarted = _haul(_step(session, 0.05))
    assert restarted.local_track_id == 1
    assert restarted.track_id != continued.track_id

    session.restart("camera_moved")
    assert session.last_reset_reason == "camera_moved"
    assert session.vehicles.haul_truck.tracks == []
    assert session.vehicles.excavator_body.tracks == []
    assert session.last_positions == {}


def test_display_threshold_edit_does_not_rewrite_parquet(tmp_path: Path) -> None:
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    out = tmp_path / "finished-run"
    pipeline = AnalyzePipeline(
        _ListReader([_read_frame(0, 0.0, image), _read_frame(1, 0.05, image)]),
        _detector(),
        run_id="finished-run",
        output_dir=out,
        analysis_fps=100,
        model_version="custom-test",
        vehicles=VehicleTrackers.from_pilot(analysis_fps=100),
    )
    pipeline.run()
    detection_path = out / "detections.parquet"
    frame_path = out / "frame_status.parquet"
    names_before = sorted(path.name for path in out.iterdir())
    detection_hash = _sha256(detection_path)
    frame_hash = _sha256(frame_path)
    reloaded = pq.read_table(detection_path).to_pylist()
    confirmed = [row for row in reloaded if row["track_id_nullable"]]
    assert confirmed
    assert confirmed[0]["track_id_nullable"].startswith("finished-run:clip:unset:")
    resets = pipeline.session.reset_count
    pipeline.session.set_display_threshold(0.8)
    with pytest.raises(ValueError):
        pipeline.session.set_display_threshold(1.2)
    assert pipeline.session.vehicles.display_threshold == pytest.approx(0.8)
    assert pipeline.session.run_id == "finished-run"
    assert pipeline.session.reset_count == resets
    assert sorted(path.name for path in out.iterdir()) == names_before
    assert _sha256(detection_path) == detection_hash
    assert _sha256(frame_path) == frame_hash
    assert pq.read_table(detection_path).to_pylist() == reloaded


def test_pipeline_seek_namespaces_ids_and_keeps_detector_boxes(tmp_path: Path) -> None:
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    frames = [
        _read_frame(100, 0.0, image),
        _read_frame(101, 0.05, image),
        _read_frame(102, 0.10, image),
        _read_frame(103, 0.02, image),
        _read_frame(104, 0.07, image),
    ]
    out = tmp_path / "seek"
    pipeline = AnalyzePipeline(
        _ListReader(frames),
        _detector(),
        run_id="run-seek",
        output_dir=out,
        analysis_fps=100,
        model_version="custom-test",
        vehicles=VehicleTrackers.from_pilot(analysis_fps=100),
    )
    original = np.array(_DETECTIONS.xyxy, copy=True)
    pipeline.run()
    assert np.array_equal(_DETECTIONS.xyxy, original)
    assert pipeline.session.last_reset_reason == RESET_SEEK
    assert pipeline.session.reset_count == 2
    rows = pq.read_table(out / "detections.parquet").to_pylist()
    assert len(rows) == 10
    continued = _at(rows, 0.10, "haul_truck")
    seek = _at(rows, 0.02, "haul_truck")
    new = _at(rows, 0.07, "haul_truck")
    body = _at(rows, 0.07, "excavator_body")
    assert continued["track_id_nullable"] == "run-seek:clip:unset:haul_truck:0"
    assert _at(rows, 0.05, "haul_truck")["track_id_nullable"] == continued["track_id_nullable"]
    assert seek["track_id_nullable"] is None
    assert new["track_id_nullable"] == "run-seek:clip:unset:haul_truck:1"
    assert new["track_id_nullable"] != continued["track_id_nullable"]
    assert body["track_id_nullable"] == "run-seek:clip:unset:excavator_body:1"
    assert seek["view_id"] is None
    assert seek["source_frame_index"] == 103
    assert float(seek["source_time_s"]) == pytest.approx(0.02)
    assert ":103" not in (new["track_id_nullable"] or "")
    for row, box in (
        (continued, _TRUCK_BOX),
        (seek, _TRUCK_BOX),
        (new, _TRUCK_BOX),
        (body, _BODY_BOX),
    ):
        assert [float(row[name]) for name in ("x1", "y1", "x2", "y2")] == pytest.approx(box)


def test_analyze_request_reset_reason_is_explicit(tmp_path: Path) -> None:
    document = {
        "video_id": "clip",
        "model_id": "baseline",
        "analysis_fps": 5,
        "start_s": 0,
        "end_s": None,
        "view_id": None,
        "reset_reason": "camera_moved",
    }
    (tmp_path / REQUEST_NAME).write_text(json.dumps(document), encoding="utf-8")
    assert _read_request(tmp_path)["reset_reason"] == "camera_moved"
    document.pop("reset_reason")
    (tmp_path / REQUEST_NAME).write_text(json.dumps(document), encoding="utf-8")
    assert _read_request(tmp_path)["reset_reason"] is None
    document["reset_reason"] = " "
    (tmp_path / REQUEST_NAME).write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(LabError) as raised:
        _read_request(tmp_path)
    assert raised.value.code == "invalid_reset_reason"

    image = np.zeros((120, 200, 3), dtype=np.uint8)
    pipeline = AnalyzePipeline(
        _ListReader([_read_frame(0, 0.0, image)]),
        _detector(),
        run_id="scene",
        output_dir=tmp_path / "scene",
        analysis_fps=5,
        model_version="custom-test",
        reset_reason="camera_moved",
        vehicles=VehicleTrackers.from_pilot(analysis_fps=5),
    )
    pipeline.run()
    assert pipeline.session.last_reset_reason == "camera_moved"
    assert pipeline.session.reset_count == 1


_DETECTIONS = FrameDetections(
    timestamp_s=0.0,
    frame_width=200,
    frame_height=120,
    origin=CheckpointOrigin.custom,
    xyxy=np.array([_TRUCK_BOX, _BODY_BOX], dtype=np.float64),
    class_id=np.array([0, 1], dtype=np.int64),
    class_name=("haul_truck", "excavator_body"),
    score=np.array([0.9, 0.9], dtype=np.float64),
)


def _session() -> TrackingSession:
    return TrackingSession(
        run_id="run-1",
        video_id="video-a",
        view_id=None,
        vehicles=VehicleTrackers.from_pilot(analysis_fps=5),
    )


def _step(
    session: TrackingSession,
    source_time_s: float,
    *,
    video: str = "video-a",
    view: str | None = None,
):
    before = np.array(_DETECTIONS.xyxy, copy=True)
    rows = session.update(
        _DETECTIONS,
        video_id=video,
        view_id=view,
        source_time_s=source_time_s,
    )
    assert np.array_equal(_DETECTIONS.xyxy, before)
    return rows


def _haul(rows):
    return next(row for row in rows if row.class_name == "haul_truck")


def _body(rows):
    return next(row for row in rows if row.class_name == "excavator_body")


def _detector() -> FakeDetector:
    return FakeDetector(
        class_names=("haul_truck", "excavator_body"),
        origin=CheckpointOrigin.custom,
        scripted=(
            ScriptedDetection(_TRUCK_BOX, 0, 0.9, "haul_truck"),
            ScriptedDetection(_BODY_BOX, 1, 0.9, "excavator_body"),
        ),
    )


def _at(rows: list[dict[str, object]], time_s: float, class_name: str) -> dict[str, object]:
    matches = [
        row
        for row in rows
        if row["class_name"] == class_name and abs(float(row["source_time_s"]) - time_s) < 1e-6
    ]
    assert len(matches) == 1
    return matches[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _ListReader:
    def __init__(self, frames: list[ReadFrame]) -> None:
        self.frames = frames

    def iterate(self, start_s: float = 0.0, end_s: float | None = None):
        for frame in self.frames:
            stamp = frame.source_time_s
            if stamp is not None and not in_half_open(stamp, start_s, end_s):
                continue
            yield frame


def _read_frame(index: int, time_s: float, image: np.ndarray) -> ReadFrame:
    return ReadFrame(
        video_id="clip",
        source_frame_index=index,
        source_pts=round(time_s * 1000),
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
