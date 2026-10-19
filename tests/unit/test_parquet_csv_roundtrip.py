"""Parquet and CSV round-trip. Empty stays empty. Partial does not pad."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from excavation_lab.cli import app
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.errors import LabError
from excavation_lab.models.adapter import CheckpointOrigin, FrameDetections
from excavation_lab.schemas import SCHEMA_VERSION
from excavation_lab.storage import (
    connect,
    create_project,
    insert_analysis_run,
    insert_review,
    insert_video,
    project_directory,
    project_sqlite,
)
from excavation_lab.tracking.bed import BedTruckAssociator
from excavation_lab.tracking.export_tables import (
    DETECTIONS_COLUMNS,
    DETECTIONS_NAME,
    FRAME_STATUS_COLUMNS,
    FRAME_STATUS_NAME,
    TABLES_SIDECAR,
    TRACK_STATE_COLUMNS,
    TRACK_STATES_NAME,
    AnalysisTableWriter,
    export_directory,
    track_state_record,
    write_analysis_tables,
)
from excavation_lab.tracking.session import public_track_id
from excavation_lab.tracking.states import ObservationLedger

_RUNNER = CliRunner()
_RUN = "run-1"
_VIDEO = "clip"
_PARENT = "run-1:clip:unset:haul_truck:1"


def test_parquet_and_csv_keep_nulls_and_predicted_boxes(tmp_path: Path) -> None:
    directory = tmp_path / "tables"
    frames = [_frame(0.0, 0), _frame(0.2, 1)]
    scored = _detection(
        "scored",
        0.0,
        score=0.25,
        parent=_PARENT,
        association_score=0.5,
        observation_state="observed",
    )
    missing = _detection(
        "missing-score",
        0.2,
        score=None,
        parent=None,
        association_score=None,
        observation_state="weak",
    )
    predicted = _track(
        "run-1:clip:unset:bucket:0",
        0.2,
        state="predicted",
        is_observed=False,
        xyxy=(5.0, 6.0, 7.0, 8.0),
        parent=None,
    )
    missed = _track(
        "run-1:clip:unset:haul_truck:2",
        0.2,
        state="missing",
        is_observed=False,
        xyxy=None,
        parent=None,
    )
    write_analysis_tables(directory, frames, [scored, missing], [predicted, missed])

    frame_path = directory / FRAME_STATUS_NAME
    detection_path = directory / DETECTIONS_NAME
    track_path = directory / TRACK_STATES_NAME
    assert pq.read_table(frame_path).column_names == list(FRAME_STATUS_COLUMNS)
    assert pq.read_table(detection_path).column_names == list(DETECTIONS_COLUMNS)
    assert pq.read_table(track_path).column_names == list(TRACK_STATE_COLUMNS)
    frame_row = pq.read_table(frame_path).to_pylist()[0]
    for name in ("source_pts", "time_base", "analysis_index", "width", "height"):
        assert name in frame_row
    assert frame_row["source_pts"] == 0
    assert frame_row["time_base"] == "1/30"
    assert frame_row["width"] == 64
    assert frame_row["height"] == 48

    detections = pq.read_table(detection_path).to_pylist()
    assert all(row["observation_state"] != "predicted" for row in detections)
    blank = next(row for row in detections if row["observation_id"] == "missing-score")
    assert blank["score"] is None
    assert blank["score"] != 0
    assert blank["parent_track_id_nullable"] is None
    assert blank["parent_track_id_nullable"] != 0
    assert blank["association_score_nullable"] is None
    assert blank["association_score_nullable"] != 0
    assert json.dumps({"score": blank["score"]}) == '{"score": null}'
    assert json.dumps({"parent": blank["parent_track_id_nullable"]}) == '{"parent": null}'
    kept = next(row for row in detections if row["observation_id"] == "scored")
    assert kept["score"] == pytest.approx(0.25)

    tracks = pq.read_table(track_path).to_pylist()
    dashed = next(row for row in tracks if row["state"] == "predicted")
    assert dashed["is_observed"] is False
    assert dashed["x1_nullable"] == pytest.approx(5.0)
    assert dashed["parent_track_id_nullable"] is None
    absent = next(row for row in tracks if row["state"] == "missing")
    assert absent["x1_nullable"] is None
    assert absent["y1_nullable"] is None
    assert absent["x2_nullable"] is None
    assert absent["y2_nullable"] is None
    assert absent["parent_track_id_nullable"] is None

    sidecar = json.loads((directory / TABLES_SIDECAR).read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == SCHEMA_VERSION
    assert sidecar["partial"] is False
    assert sidecar["analyzed_end_s"] is None

    destination = tmp_path / "csv"
    exported = export_directory(
        directory,
        destination,
        notes=[
            {
                "review_id": "rev-1",
                "video_id": None,
                "source_frame_index": None,
                "review_status": None,
                "note": "=cmd",
            }
        ],
        excel=False,
        run_id=_RUN,
        project_id="pilot",
    )
    assert exported["schema_version"] == SCHEMA_VERSION
    assert exported["partial"] is False
    assert exported["analyzed_end_s"] is None
    assert "events.csv" not in exported["files"]
    assert not (destination / "events.csv").exists()
    assert not (destination / "truck_visits.csv").exists()

    csv_detections = _read_csv(destination / "detections.csv")
    csv_blank = next(row for row in csv_detections if row["observation_id"] == "missing-score")
    assert csv_blank["score"] == ""
    assert csv_blank["score"] != "0"
    assert csv_blank["parent_track_id_nullable"] == ""
    assert csv_blank["parent_track_id_nullable"] != "0"
    assert csv_blank["association_score_nullable"] == ""
    csv_kept = next(row for row in csv_detections if row["observation_id"] == "scored")
    assert csv_kept["score"] == "0.25"
    assert "," not in csv_kept["score"]
    raw = (destination / "detections.csv").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"0.25" in raw
    assert b"0,25" not in raw

    csv_tracks = _read_csv(destination / "track_states.csv")
    csv_dashed = next(row for row in csv_tracks if row["state"] == "predicted")
    assert csv_dashed["is_observed"] == "false"
    assert csv_dashed["x1_nullable"] == "5"
    csv_absent = next(row for row in csv_tracks if row["state"] == "missing")
    assert csv_absent["x1_nullable"] == ""
    assert csv_absent["parent_track_id_nullable"] == ""
    notes = _read_csv(destination / "notes.csv")
    assert notes[0]["note"] == "'=cmd"
    assert notes[0]["source_frame_index"] == ""
    assert notes[0]["source_frame_index"] != "0"


def test_partial_run_stops_at_analyzed_end(tmp_path: Path) -> None:
    directory = tmp_path / "partial"
    tail = _frame(8.0, 3)
    tail["num_detections"] = 5
    frames = [_frame(0.0, 0), _frame(0.2, 1), tail]
    leaked = _detection(
        "tail",
        8.0,
        score=0.0,
        parent=None,
        association_score=None,
        observation_state="observed",
    )
    write_analysis_tables(
        directory,
        frames,
        [leaked],
        [],
        partial=True,
        analyzed_end_s=0.2,
    )
    stored = pq.read_table(directory / FRAME_STATUS_NAME).to_pylist()
    assert [float(row["source_time_s"]) for row in stored] == pytest.approx([0.0, 0.2])
    assert all(row["num_detections"] == 0 for row in stored)
    detections = pq.read_table(directory / DETECTIONS_NAME).to_pylist()
    assert detections == []
    assert all(row["source_time_s"] <= 0.2 for row in stored)
    sidecar = json.loads((directory / TABLES_SIDECAR).read_text(encoding="utf-8"))
    assert sidecar["schema_version"] == SCHEMA_VERSION
    assert sidecar["partial"] is True
    assert sidecar["analyzed_end_s"] == pytest.approx(0.2)
    assert sidecar["analyzed_end_s"] != 0
    assert sidecar["analyzed_end_s"] != 10


def test_csv_export_quotes_formula_notes_and_refuses_events(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    create_project("pilot", workspace, catalog_id="excavation")
    insert_video(workspace, "pilot", _VIDEO, name="clip")
    with connect(project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                job_id, project_id, status, pid, partial,
                analyzed_end_s, error_code, created_at
            ) VALUES (?, ?, 'completed', NULL, 0, NULL, NULL, ?)
            """,
            (_RUN, "pilot", "2026-10-10T00:00:00+00:00"),
        )
    insert_analysis_run(
        workspace,
        "pilot",
        run_id=_RUN,
        job_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        model_id="baseline",
        model_origin="baseline_coco",
        analysis_fps=5.0,
        source_fps=30.0,
        export_fps=None,
        start_s=0.0,
        end_s=None,
        decode_ms=None,
        encode_ms=None,
    )
    run_dir = project_directory(workspace, "pilot") / "runs" / _RUN
    write_analysis_tables(
        run_dir,
        [_frame(0.0, 0)],
        [
            _detection(
                "scored",
                0.0,
                score=0.25,
                parent=None,
                association_score=None,
                observation_state="observed",
            )
        ],
        [
            _track(
                "run-1:clip:unset:bucket:0",
                0.0,
                state="predicted",
                is_observed=False,
                xyxy=(5.0, 6.0, 7.0, 8.0),
                parent=None,
            )
        ],
    )
    insert_review(workspace, "pilot", review_id="rev-1", note="=cmd")
    assert _stored_note(workspace) == "=cmd"

    exported = _RUNNER.invoke(
        app,
        [
            "export",
            "--project",
            "pilot",
            "--run",
            _RUN,
            "--format",
            "csv",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert exported.exit_code == 0
    assert "Traceback" not in exported.stdout
    body = json.loads(exported.stdout)
    assert body["ok"] is True
    assert body["data"]["schema_version"] == SCHEMA_VERSION
    assert body["data"]["excel_bom"] is False
    assert body["data"]["analyzed_end_s"] is None
    export_dir = project_directory(workspace, "pilot") / "exports" / _RUN
    notes_raw = (export_dir / "notes.csv").read_bytes()
    assert not notes_raw.startswith(b"\xef\xbb\xbf")
    assert _read_csv(export_dir / "notes.csv")[0]["note"] == "'=cmd"
    assert _stored_note(workspace) == "=cmd"
    detections = pq.read_table(run_dir / DETECTIONS_NAME).to_pylist()
    assert all(row["observation_state"] != "predicted" for row in detections)
    assert detections[0]["parent_track_id_nullable"] is None
    tracks = pq.read_table(run_dir / TRACK_STATES_NAME).to_pylist()
    assert tracks[0]["state"] == "predicted"
    assert tracks[0]["is_observed"] is False

    excel = _RUNNER.invoke(
        app,
        [
            "export",
            "--project",
            "pilot",
            "--run",
            _RUN,
            "--format",
            "csv",
            "--excel",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert excel.exit_code == 0
    excel_body = json.loads(excel.stdout)
    assert excel_body["data"]["excel_bom"] is True
    bom = (export_dir / "notes.csv").read_bytes()
    assert bom.startswith(b"\xef\xbb\xbf")
    assert _read_csv(export_dir / "notes.csv")[0]["note"] == "'=cmd"
    assert (export_dir / "detections.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert b"0.25" in (export_dir / "detections.csv").read_bytes()
    assert _stored_note(workspace) == "=cmd"
    names = {path.name for path in export_dir.iterdir()}
    assert "events.csv" not in names
    assert "truck_visits.csv" not in names

    refused = _RUNNER.invoke(
        app,
        [
            "export",
            "--project",
            "pilot",
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
    assert "Traceback" not in refused.stdout
    error = json.loads(refused.stdout)
    assert error["ok"] is False
    assert error["error"]["code"] == "unavailable"
    assert "События" in error["error"]["message"]
    assert list(workspace.rglob("events.csv")) == []
    assert list(workspace.rglob("truck_visits.csv")) == []
    assert _stored_note(workspace) == "=cmd"


def test_export_events_refuses_before_creating_a_file(tmp_path: Path) -> None:
    workspace = tmp_path / "empty"
    refused = _RUNNER.invoke(
        app,
        [
            "export",
            "--project",
            "missing",
            "--run",
            "missing",
            "--format",
            "events",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert refused.exit_code != 0
    error = json.loads(refused.stdout)
    assert error["error"]["code"] == "unavailable"
    assert list(tmp_path.rglob("events.csv")) == []
    assert list(tmp_path.rglob("truck_visits.csv")) == []
    assert not workspace.exists()


def test_writer_rejects_a_predicted_detection(tmp_path: Path) -> None:
    opened = write_analysis_tables(tmp_path / "bad")
    assert isinstance(opened, AnalysisTableWriter)
    row = _detection(
        "dash",
        0.0,
        score=0.2,
        parent=None,
        association_score=None,
        observation_state="predicted",
    )
    with pytest.raises(LabError) as caught:
        opened.add_detection(row)
    assert caught.value.code == "predicted_detection"
    assert not (tmp_path / "bad" / DETECTIONS_NAME).exists()


def test_bed_track_id_in_parquet_ignores_detector_order(tmp_path: Path) -> None:
    bed = (10.0, 40.0, 50.0, 70.0)
    truck = (0.0, 0.0, 80.0, 80.0)
    associator = BedTruckAssociator.from_pilot()
    first = _boxes(("truck_bed",), (0.91,), (bed,), 0.0)
    second = _boxes(("haul_truck", "truck_bed"), (0.95, 0.91), (truck, bed), 0.2)
    frame_a = associator.update(
        first,
        trucks=(),
        source_time_s=0.0,
        run_id="run-bed",
        video_id=_VIDEO,
        view_id=None,
    )
    frame_b = associator.update(
        second,
        trucks=(),
        source_time_s=0.2,
        run_id="run-bed",
        video_id=_VIDEO,
        view_id=None,
    )
    link_a = frame_a.links[0]
    link_b = frame_b.links[0]
    assert link_a.index == 0
    assert link_b.index == 1
    assert link_a.local_id == link_b.local_id == 0
    ledger = ObservationLedger(detection_floor=0.10, display_threshold=0.35)
    ledger.begin(1)
    state_a = ledger.observe(
        first,
        tracked_rows=(),
        bucket_frame=None,
        bed_frame=frame_a,
        lost_vehicles=(),
        run_id="run-bed",
        video_id=_VIDEO,
        view_id=None,
        source_time_s=0.0,
        reset_count=1,
    )
    state_b = ledger.observe(
        second,
        tracked_rows=(),
        bucket_frame=None,
        bed_frame=frame_b,
        lost_vehicles=(),
        run_id="run-bed",
        video_id=_VIDEO,
        view_id=None,
        source_time_s=0.2,
        reset_count=1,
    )
    expected = public_track_id("run-bed", _VIDEO, None, "truck_bed", 0)
    assert state_a.track_states[0].track_id == expected
    assert state_b.track_states[0].track_id == expected
    assert state_b.track_states[0].xyxy == pytest.approx(list(bed))
    opened = write_analysis_tables(tmp_path / "bed")
    assert isinstance(opened, AnalysisTableWriter)
    for item in (*state_a.track_states, *state_b.track_states):
        opened.add_track_state(track_state_record(item))
    opened.close()
    rows = pq.read_table(tmp_path / "bed" / TRACK_STATES_NAME).to_pylist()
    assert [row["track_id"] for row in rows] == [expected, expected]
    assert all(row["parent_track_id_nullable"] is None for row in rows)
    assert all(row["parent_track_id_nullable"] != 0 for row in rows)
    assert public_track_id("run-bed", _VIDEO, None, "truck_bed", link_b.index) != expected


def _frame(stamp: float, index: int) -> dict[str, object]:
    row: dict[str, object] = {name: None for name in FRAME_STATUS_COLUMNS}
    row.update(
        {
            "run_id": _RUN,
            "video_id": _VIDEO,
            "view_id": None,
            "source_frame_index": index,
            "source_time_s": stamp,
            "decode_status": "ok",
            "analyzed": True,
            "model_version": "baseline_coco",
            "inference_ms": 1.5,
            "num_detections": 0,
            "bucket_observation_state": None,
            "diagnostic_flags": [],
            "source_pts": index,
            "time_base": "1/30",
            "analysis_index": index,
            "width": 64,
            "height": 48,
            "decode_ms": None,
            "encode_ms": None,
        }
    )
    return row


def _detection(
    observation_id: str,
    stamp: float,
    *,
    score: float | None,
    parent: str | None,
    association_score: float | None,
    observation_state: str,
) -> dict[str, object]:
    row: dict[str, object] = {name: None for name in DETECTIONS_COLUMNS}
    row.update(
        {
            "run_id": _RUN,
            "observation_id": observation_id,
            "video_id": _VIDEO,
            "source_frame_index": 0,
            "source_time_s": stamp,
            "class_id": 3,
            "class_name": "truck_bed",
            "score": score,
            "x1": 1.0,
            "y1": 2.0,
            "x2": 3.0,
            "y2": 4.0,
            "frame_width": 64,
            "frame_height": 48,
            "track_id_nullable": None,
            "parent_track_id_nullable": parent,
            "observation_state": observation_state,
            "association_score_nullable": association_score,
            "model_version": "baseline_coco",
            "source_pts": 0,
            "time_base": "1/30",
            "analysis_index": 0,
            "width": 64,
            "height": 48,
        }
    )
    return row


def _track(
    track_id: str,
    stamp: float,
    *,
    state: str,
    is_observed: bool,
    xyxy: tuple[float, float, float, float] | None,
    parent: str | None,
) -> dict[str, object]:
    row: dict[str, object] = {name: None for name in TRACK_STATE_COLUMNS}
    coords = (None, None, None, None) if xyxy is None else xyxy
    row.update(
        {
            "run_id": _RUN,
            "track_id": track_id,
            "class_name": "bucket" if state == "predicted" else "haul_truck",
            "source_time_s": stamp,
            "state": state,
            "last_observed_time_s": None,
            "is_observed": is_observed,
            "x1_nullable": coords[0],
            "y1_nullable": coords[1],
            "x2_nullable": coords[2],
            "y2_nullable": coords[3],
            "parent_track_id_nullable": parent,
            "identity_uncertain": False,
        }
    )
    return row


def _boxes(
    names: tuple[str, ...],
    scores: tuple[float, ...],
    boxes: tuple[tuple[float, float, float, float], ...],
    stamp: float,
) -> FrameDetections:
    mapping = {"excavator_body": 0, "bucket": 1, "haul_truck": 2, "truck_bed": 3}
    count = len(names)
    xyxy = np.zeros((count, 4), dtype=np.float64)
    xyxy[:] = np.asarray(boxes, dtype=np.float64)
    return FrameDetections(
        timestamp_s=stamp,
        frame_width=160,
        frame_height=120,
        origin=CheckpointOrigin.custom,
        xyxy=xyxy,
        class_id=np.asarray([mapping[name] for name in names], dtype=np.int64),
        class_name=names,
        score=np.asarray(scores, dtype=np.float64),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _stored_note(workspace: Path) -> str | None:
    with connect(project_sqlite(workspace, "pilot")) as connection:
        row = connection.execute("SELECT note FROM reviews").fetchone()
    assert row is not None
    note = row["note"]
    assert note is None or isinstance(note, str)
    return note
