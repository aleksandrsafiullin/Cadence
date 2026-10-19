"""Action-train job: test frames stay out of the window frame. Uses tmp_path."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from excavation_lab import jobs, storage
from excavation_lab.actions.catalog import add_action
from excavation_lab.actions.dataset import build_windows
from excavation_lab.actions.labels import add_interval
from excavation_lab.actions.train import (
    ActionTrainConfig,
    check_ready,
    enqueue_action_train,
    run_action_train_job,
)
from excavation_lab.cli import app
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.splits import freeze_split, write_blocks
from excavation_lab.errors import JobAlreadyRunningError, LabError
from excavation_lab.jobs import (
    get_job,
    lab_database,
    request_cancel,
    run_directory,
    start_job,
)
from excavation_lab.schemas import JobStatus
from excavation_lab.tracking.export_tables import (
    DETECTIONS_COLUMNS,
    DETECTIONS_NAME,
    FRAME_STATUS_COLUMNS,
    FRAME_STATUS_NAME,
    TRACK_STATE_COLUMNS,
    TRACK_STATES_NAME,
)

_RUNNER = CliRunner()
_PROJECT = "mvp"
_VIDEO = "vid"
_MODEL = "det_custom"
_FPS = 5.0
_BLOCKS = [
    RoleBlock(start_s=0, end_s=30, role="train"),
    RoleBlock(start_s=40, end_s=70, role="validation"),
    RoleBlock(start_s=80, end_s=100, role="test"),
]
_TEST_TIME = 90.0
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
    CHECK (kind IN ('analyze', 'export', 'train'))
)
"""


class _DummyProcess:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.pid = os.getpid()

    def poll(self) -> None:
        return None


def _dummy_spawn(*_args: object, **_kwargs: object) -> _DummyProcess:
    return _DummyProcess()


def _times(start_s: float, end_s: float, step_s: float = 0.2) -> list[float]:
    values: list[float] = []
    stamp = start_s
    while stamp < end_s:
        values.append(round(stamp, 6))
        stamp += step_s
    return values


def _write_tables(run_dir: Path, run_id: str, times: list[float]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, stamp in enumerate(times):
        row = {name: None for name in FRAME_STATUS_COLUMNS}
        row.update(
            {
                "run_id": run_id,
                "video_id": _VIDEO,
                "source_frame_index": index if stamp != _TEST_TIME else 9_000,
                "source_time_s": stamp,
                "decode_status": "ok",
                "analyzed": True,
                "width": 64,
                "height": 48,
                "num_detections": 0,
            }
        )
        rows.append(row)
    pd.DataFrame(rows, columns=list(FRAME_STATUS_COLUMNS)).to_parquet(run_dir / FRAME_STATUS_NAME)
    pd.DataFrame(columns=list(DETECTIONS_COLUMNS)).to_parquet(run_dir / DETECTIONS_NAME)
    pd.DataFrame(columns=list(TRACK_STATE_COLUMNS)).to_parquet(run_dir / TRACK_STATES_NAME)


def _finish_run(
    workspace: Path,
    *,
    run_id: str,
    start_s: float,
    end_s: float,
    times: list[float],
    model_id: str = _MODEL,
    origin: str = "custom",
    analysis_fps: float = _FPS,
) -> str:
    record = start_job(
        workspace,
        _PROJECT,
        kind="analyze",
        worker="sleep",
        video_duration_s=10.0,
        step_s=1.0,
    )
    run_dir = run_directory(workspace, _PROJECT, record.job_id)
    _write_tables(run_dir, run_id, times)
    jobs.mark_completed(workspace, _PROJECT, record.job_id)
    storage.insert_analysis_run(
        workspace,
        _PROJECT,
        run_id=run_id,
        job_id=record.job_id,
        video_id=_VIDEO,
        view_id=None,
        model_id=model_id,
        model_origin=origin,
        analysis_fps=analysis_fps,
        source_fps=None,
        export_fps=None,
        start_s=start_s,
        end_s=end_s,
        decode_ms=None,
        encode_ms=None,
    )
    return run_id


def _ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    validation_end_s: float = 70.0,
    model_b: str = _MODEL,
    origin_b: str = "custom",
    fps_b: float = _FPS,
) -> tuple[Path, str, str]:
    monkeypatch.setattr(jobs, "_spawn", _dummy_spawn)
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project(_PROJECT, workspace, catalog_id="excavation")
    storage.insert_video(workspace, _PROJECT, _VIDEO, name="clip")
    add_action(workspace, _PROJECT, "digging", "Копание")
    proposal = write_blocks(workspace, _PROJECT, _VIDEO, _BLOCKS)
    freeze_split(workspace, _PROJECT, proposal)
    add_interval(workspace, _PROJECT, _VIDEO, 1.0, 7.0, "digging")
    add_interval(workspace, _PROJECT, _VIDEO, 41.0, 47.0, "digging")
    train_times = _times(1.0, 7.0) + [_TEST_TIME]
    validation_times = _times(41.0, 47.0)
    run_a = _finish_run(
        workspace,
        run_id="run_a",
        start_s=0.0,
        end_s=30.0,
        times=train_times,
    )
    run_b = _finish_run(
        workspace,
        run_id="run_b",
        start_s=40.0,
        end_s=validation_end_s,
        times=validation_times,
        model_id=model_b,
        origin=origin_b,
        analysis_fps=fps_b,
    )
    return workspace, run_a, run_b


def _set_origin(workspace: Path, run_id: str, origin: str) -> None:
    with storage.connect(storage.project_sqlite(workspace, _PROJECT)) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "UPDATE runs SET model_origin = ? WHERE run_id = ?",
            (origin, run_id),
        )


def test_two_runs_keep_test_frames_out_of_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, run_a, run_b = _ready(tmp_path, monkeypatch)
    seen: list[pd.DataFrame] = []

    def spy(features_df: pd.DataFrame, *args: object, **kwargs: object) -> object:
        seen.append(features_df.copy())
        return build_windows(features_df, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("excavation_lab.actions.dataset.build_windows", spy)
    started = time.perf_counter()
    record = enqueue_action_train(
        workspace,
        _PROJECT,
        [run_a, run_b],
        ActionTrainConfig(epochs=2, patience=2),
    )
    assert run_action_train_job(workspace, record.job_id) == 0
    elapsed = time.perf_counter() - started
    assert elapsed < 30.0
    job = get_job(workspace, _PROJECT, record.job_id)
    assert job.status is JobStatus.completed
    models = storage.list_action_models(workspace, _PROJECT)
    assert len(models) == 1
    assert models[0]["source_run_ids"] == [run_a, run_b]
    assert models[0]["detector_model_id"] == _MODEL
    bundle = Path(models[0]["path"])
    for name in (
        "weights.pt",
        "standardizer.json",
        "action_bundle.json",
        "feature_spec.json",
        "metrics_history.csv",
        "validation_metrics.json",
        "kept_time_ranges.json",
        "checksums.json",
    ):
        assert (bundle / name).is_file()
    document = json.loads((bundle / "action_bundle.json").read_text(encoding="utf-8"))
    assert document["source_run_ids"] == [run_a, run_b]
    assert document["class_order"] == ["digging"]
    assert document["dataset_hash"]
    assert document["feature_spec"]["analysis_fps"] == _FPS
    assert document["window"]["window_s"] == 3.0
    assert document["selection_unstable"] is True
    assert document["test_read"] is False
    assert document["classes_scored"] == ["digging"]
    kept = json.loads((bundle / "kept_time_ranges.json").read_text(encoding="utf-8"))
    assert kept
    assert {row["role"] for row in kept} <= {"train", "validation"}
    assert all(row["role"] != "test" for row in kept)
    job_kept = json.loads(
        (run_directory(workspace, _PROJECT, record.job_id) / "kept_time_ranges.json").read_text(
            encoding="utf-8"
        )
    )
    assert job_kept == kept
    log = (run_directory(workspace, _PROJECT, record.job_id) / "action_train.log").read_text(
        encoding="utf-8"
    )
    assert "role=test" in log
    assert "device=cpu" in log
    assert seen
    for frame in seen:
        times = frame["source_time_s"].to_numpy()
        assert not bool(((times >= 80.0) & (times < 100.0)).any())
        for row in kept:
            inside = frame[
                (frame["run_id"] == row["run_id"])
                & (frame["source_time_s"] >= row["start_s"])
                & (frame["source_time_s"] < row["end_s"])
            ]
            assert not inside.empty
            assert bool(inside.iloc[0]["run_start"])
    analyze_job = storage.get_analysis_run(workspace, _PROJECT, run_a)["job_id"]
    source = pd.read_parquet(run_directory(workspace, _PROJECT, analyze_job) / FRAME_STATUS_NAME)
    assert _TEST_TIME in set(source["source_time_s"].tolist())
    history = (bundle / "metrics_history.csv").read_text(encoding="utf-8")
    assert "epoch" in history.splitlines()[0]
    assert "val_macro_f1" in history.splitlines()[0]
    assert len(history.splitlines()) >= 2
    progress = json.loads(
        (run_directory(workspace, _PROJECT, record.job_id) / "progress.json").read_text(
            encoding="utf-8"
        )
    )
    assert progress["status"] == "completed"
    assert progress["epoch"] >= 1
    assert isinstance(progress["val_macro_f1"], float)
    import torch

    state = torch.load(bundle / "weights.pt", map_location="cpu", weights_only=True)
    assert "conv1.weight" in state


def test_different_model_id_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, run_a, run_b = _ready(tmp_path, monkeypatch, model_b="other_det")
    readiness = check_ready(workspace, _PROJECT, [run_a, run_b])
    assert readiness.ok is False
    assert any(
        "model_id" in reason and "other_det" in reason and run_a in reason
        for reason in readiness.reasons
    )
    with pytest.raises(LabError) as refused:
        enqueue_action_train(workspace, _PROJECT, [run_a, run_b], ActionTrainConfig(epochs=1))
    assert refused.value.code == "action_not_ready"
    assert "other_det" in refused.value.message


def test_synthetic_origin_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, run_a, run_b = _ready(tmp_path, monkeypatch)
    _set_origin(workspace, run_b, "synthetic")
    readiness = check_ready(workspace, _PROJECT, [run_a, run_b])
    assert readiness.ok is False
    assert any("synthetic" in reason and run_b in reason for reason in readiness.reasons)


def test_baseline_origin_is_refused_by_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, run_a, run_b = _ready(tmp_path, monkeypatch, origin_b="baseline_coco")
    readiness = check_ready(workspace, _PROJECT, [run_a, run_b])
    assert readiness.ok is False
    assert any("baseline" in reason and run_b in reason for reason in readiness.reasons)


def test_uncovered_interval_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, run_a, run_b = _ready(tmp_path, monkeypatch, validation_end_s=45.0)
    readiness = check_ready(workspace, _PROJECT, [run_a, run_b])
    assert readiness.ok is False
    text = "\n".join(readiness.reasons)
    assert "41" in text
    assert "47" in text
    assert "digging" in text
    assert _VIDEO in text


def test_second_start_refused_while_lock_is_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, run_a, run_b = _ready(tmp_path, monkeypatch)
    start_job(
        workspace,
        _PROJECT,
        kind="analyze",
        worker="sleep",
        video_duration_s=10.0,
        step_s=1.0,
    )
    with pytest.raises(JobAlreadyRunningError):
        enqueue_action_train(
            workspace,
            _PROJECT,
            [run_a, run_b],
            ActionTrainConfig(epochs=1),
        )


def test_cancel_mid_training_writes_no_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, run_a, run_b = _ready(tmp_path, monkeypatch)
    started = threading.Event()

    def hold(*_args: object, should_stop: object = None, **_kwargs: object) -> None:
        started.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if callable(should_stop) and should_stop():
                return
            time.sleep(0.02)
        raise AssertionError("cancel did not land")

    monkeypatch.setattr("excavation_lab.actions.job.train_action_windows", hold)
    record = enqueue_action_train(
        workspace,
        _PROJECT,
        [run_a, run_b],
        ActionTrainConfig(epochs=2),
    )
    outcome: dict[str, int] = {}

    def _run() -> None:
        outcome["code"] = run_action_train_job(workspace, record.job_id)

    thread = threading.Thread(target=_run)
    thread.start()
    assert started.wait(5.0)
    request_cancel(workspace, _PROJECT, record.job_id)
    thread.join(10.0)
    assert not thread.is_alive()
    assert outcome["code"] == 0
    assert get_job(workspace, _PROJECT, record.job_id).status is JobStatus.cancelled
    assert storage.list_action_models(workspace, _PROJECT) == []


def test_cancel_requested_after_training_inserts_no_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, run_a, run_b = _ready(tmp_path, monkeypatch)
    record = enqueue_action_train(
        workspace,
        _PROJECT,
        [run_a, run_b],
        ActionTrainConfig(epochs=1, patience=1),
    )
    original = storage.insert_action_model

    def _flip_then_insert(*args: object, **kwargs: object) -> None:
        with sqlite3.connect(lab_database(workspace)) as connection:
            connection.execute(
                "UPDATE lab_jobs SET status = ? WHERE job_id = ?",
                (JobStatus.cancel_requested.value, record.job_id),
            )
        original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("excavation_lab.actions.job.insert_action_model", _flip_then_insert)
    assert run_action_train_job(workspace, record.job_id) == 0
    assert get_job(workspace, _PROJECT, record.job_id).status is JobStatus.cancel_requested
    assert storage.list_action_models(workspace, _PROJECT) == []


def test_ensure_kinds_keeps_old_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project(_PROJECT, workspace, catalog_id="excavation")
    path = lab_database(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(_OLD_LAB_SQL)
    connection.execute(
        """
        INSERT INTO lab_jobs (
            job_id, project_id, kind, worker, status, pid, partial,
            analyzed_end_s, error_code, orphaned, run_dir,
            video_duration_s, step_s, sleep_s, created_at
        ) VALUES (?, ?, ?, ?, ?, NULL, 0, NULL, NULL, 0, ?, 10, 1, 0.05, ?)
        """,
        (
            "old_job",
            _PROJECT,
            "analyze",
            "analyze",
            "completed",
            "/tmp/old-run",
            "2020-01-01T00:00:00+00:00",
        ),
    )
    connection.commit()
    connection.close()
    monkeypatch.setattr(jobs, "_spawn", _dummy_spawn)
    start_job(
        workspace,
        _PROJECT,
        kind="action_train",
        worker="sleep",
        video_duration_s=10.0,
        step_s=1.0,
    )
    with sqlite3.connect(path) as opened:
        sql = opened.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'lab_jobs'"
        ).fetchone()[0]
        rows = opened.execute("SELECT job_id, kind FROM lab_jobs ORDER BY job_id").fetchall()
    assert "'action_train'" in sql
    assert "'action_infer'" in sql
    assert "'export'" in sql
    assert ("old_job", "analyze") in rows
    assert len(rows) == 2


def test_cli_train_json_returns_job_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, run_a, run_b = _ready(tmp_path, monkeypatch)
    result = _RUNNER.invoke(
        app,
        [
            "actions",
            "train",
            "--project",
            _PROJECT,
            "--run",
            run_a,
            "--run",
            run_b,
            "--epochs",
            "2",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    job_id = payload["data"]["job_id"]
    request = json.loads(
        (run_directory(workspace, _PROJECT, job_id) / "action_train_request.json").read_text(
            encoding="utf-8"
        )
    )
    assert request["run_ids"] == [run_a, run_b]
    assert request["config"]["epochs"] == 2
    assert request["config"]["device"] == "cpu"
    refused = _RUNNER.invoke(
        app,
        [
            "actions",
            "train",
            "--project",
            _PROJECT,
            "--run",
            "missing_run",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert refused.exit_code != 0
    body = json.loads(refused.stdout)
    assert body["ok"] is False
    assert body["error"]["code"] == "action_not_ready"


def test_fps_mismatch_overlap_and_missing_run_are_named(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, run_a, run_b = _ready(tmp_path, monkeypatch, fps_b=2.0)
    rates = check_ready(workspace, _PROJECT, [run_a, run_b])
    assert rates.ok is False
    assert any("analysis_fps" in reason and "2" in reason for reason in rates.reasons)
    with storage.connect(storage.project_sqlite(workspace, _PROJECT)) as connection:
        connection.execute("UPDATE runs SET analysis_fps = 5, end_s = 50 WHERE run_id = ?", (run_a,))
    overlap = check_ready(workspace, _PROJECT, [run_a, run_b])
    assert overlap.ok is False
    assert any("пересекаются" in reason and run_a in reason and run_b in reason for reason in overlap.reasons)
    missing = check_ready(workspace, _PROJECT, ["missing_run"])
    assert any("missing_run" in reason and "не найден" in reason for reason in missing.reasons)
