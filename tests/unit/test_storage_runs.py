"""Analysis-run lookup for later action training. Synthetic rows, no video."""

from __future__ import annotations

from pathlib import Path

import pytest

from excavation_lab import jobs
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.errors import LabError
from excavation_lab.storage import (
    connect,
    create_project,
    insert_analysis_run,
    insert_video,
    list_analysis_runs,
    project_sqlite,
)
from excavation_lab.tracking.analyze import run_tables
from excavation_lab.tracking.export_tables import (
    DETECTIONS_NAME,
    FRAME_STATUS_NAME,
    TRACK_STATES_NAME,
)

_TABLES = (FRAME_STATUS_NAME, DETECTIONS_NAME, TRACK_STATES_NAME)


def test_list_analysis_runs_filters_by_origin_and_video(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    create_project("pilot", workspace, catalog_id="excavation")
    insert_video(workspace, "pilot", "v-one", name="one")
    insert_video(workspace, "pilot", "v-two", name="two")
    _insert_run(
        workspace,
        run_id="run-custom-b",
        job_id="job-custom-b",
        video_id="v-two",
        model_id="smoke-b",
        model_origin="custom",
        end_s=60.0,
        created_at="2026-10-19T00:00:03+00:00",
        lab_status=None,
    )
    _insert_run(
        workspace,
        run_id="run-base",
        job_id="job-base",
        video_id="v-one",
        model_id="baseline",
        model_origin="baseline_coco",
        end_s=None,
        created_at="2026-10-19T00:00:01+00:00",
        lab_status="running",
    )
    _insert_run(
        workspace,
        run_id="run-custom-a",
        job_id="job-custom-a",
        video_id="v-one",
        model_id="smoke-a",
        model_origin="custom",
        end_s=120.0,
        created_at="2026-10-19T00:00:02+00:00",
        lab_status="completed",
    )

    everything = list_analysis_runs(workspace, "pilot")
    assert [row["run_id"] for row in everything] == [
        "run-base",
        "run-custom-a",
        "run-custom-b",
    ]
    custom = list_analysis_runs(workspace, "pilot", origin="custom")
    assert [row["run_id"] for row in custom] == ["run-custom-a", "run-custom-b"]
    assert {row["model_origin"] for row in custom} == {"custom"}
    video = list_analysis_runs(workspace, "pilot", video_id="v-one")
    assert [row["run_id"] for row in video] == ["run-base", "run-custom-a"]
    both = list_analysis_runs(workspace, "pilot", video_id="v-one", origin="custom")
    assert [row["run_id"] for row in both] == ["run-custom-a"]

    by_id = {row["run_id"]: row for row in everything}
    full = by_id["run-base"]
    assert full["end_s"] is None
    assert full["analysis_fps"] == 5.0
    assert full["start_s"] == 0.0
    assert full["completed"] is False
    assert full["run_dir"] == str(jobs.run_directory(workspace, "pilot", "job-base"))
    assert by_id["run-custom-a"]["end_s"] == 120.0
    assert by_id["run-custom-a"]["completed"] is True
    assert by_id["run-custom-a"]["model_id"] == "smoke-a"
    assert by_id["run-custom-b"]["completed"] is False
    assert by_id["run-custom-b"]["video_id"] == "v-two"


@pytest.mark.parametrize("missing", _TABLES)
def test_run_tables_refuses_a_missing_parquet(tmp_path: Path, missing: str) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in _TABLES:
        if name != missing:
            (run_dir / name).write_bytes(b"parquet")
    with pytest.raises(LabError) as caught:
        run_tables(run_dir)
    assert caught.value.code == "run_incomplete"
    assert missing in caught.value.message

    for name in _TABLES:
        (run_dir / name).write_bytes(b"parquet")
    tables = run_tables(run_dir)
    assert tables.frame_status == run_dir / FRAME_STATUS_NAME
    assert tables.detections == run_dir / DETECTIONS_NAME
    assert tables.track_states == run_dir / TRACK_STATES_NAME


def _insert_run(
    workspace: Path,
    *,
    run_id: str,
    job_id: str,
    video_id: str,
    model_id: str,
    model_origin: str,
    end_s: float | None,
    created_at: str,
    lab_status: str | None,
) -> None:
    with connect(project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                job_id, project_id, status, pid, partial,
                analyzed_end_s, error_code, created_at
            ) VALUES (?, ?, 'completed', NULL, 0, NULL, NULL, ?)
            """,
            (job_id, "pilot", created_at),
        )
    insert_analysis_run(
        workspace,
        "pilot",
        run_id=run_id,
        job_id=job_id,
        video_id=video_id,
        view_id=None,
        model_id=model_id,
        model_origin=model_origin,
        analysis_fps=5.0,
        source_fps=None,
        export_fps=None,
        start_s=0.0,
        end_s=end_s,
        decode_ms=None,
        encode_ms=None,
    )
    with connect(project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            "UPDATE runs SET created_at = ? WHERE run_id = ?",
            (created_at, run_id),
        )
    if lab_status is None:
        return
    run_dir = jobs.run_directory(workspace, "pilot", job_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with jobs._lab(workspace) as connection:
        connection.execute(
            """
            INSERT INTO lab_jobs (
                job_id, project_id, kind, worker, status, pid, partial,
                analyzed_end_s, error_code, orphaned, run_dir,
                video_duration_s, step_s, sleep_s, created_at
            ) VALUES (
                ?, ?, 'analyze', 'analyze', ?, NULL, 0, NULL, NULL, 0, ?,
                10, 0.2, 0.05, ?
            )
            """,
            (job_id, "pilot", lab_status, str(run_dir), created_at),
        )
