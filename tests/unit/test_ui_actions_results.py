"""Actions page train, inference, and evaluation controls."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from excavation_lab import storage
from excavation_lab.actions.catalog import seed_actions
from excavation_lab.actions.labels import add_interval
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.splits import freeze_split, write_blocks
from excavation_lab.errors import LabError
from excavation_lab.jobs import lab_database, request_cancel, start_job
from excavation_lab.ui.copy import action_seed_titles, catalog
from excavation_lab.web.app import create_app

_PROJECT = "mvp"
_VIDEO = "vid"
_BLOCKS = [
    RoleBlock(start_s=0, end_s=80, role="train"),
    RoleBlock(start_s=90, end_s=160, role="validation"),
    RoleBlock(start_s=170, end_s=200, role="test"),
]


def test_readiness_reasons_render_verbatim(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    _project_job(workspace, "jobbase0001")
    storage.insert_analysis_run(
        workspace,
        _PROJECT,
        run_id="run_base",
        job_id="jobbase0001",
        video_id=_VIDEO,
        view_id=None,
        model_id="coco",
        model_origin="baseline_coco",
        analysis_fps=2.0,
        source_fps=None,
        export_fps=None,
        start_s=0.0,
        end_s=40.0,
        decode_ms=None,
        encode_ms=None,
    )
    with TestClient(create_app(workspace)) as client:
        ready = client.get("/actions/ready", params={"runs": "run_base"})
        page = client.get("/actions", params={"runs": "run_base"})
    assert ready.status_code == 200
    reasons = ready.json()["reasons"]
    quoted = next(item for item in reasons if "baseline" in item)
    assert quoted in page.text
    assert 'id="action-readiness"' in page.text
    assert 'id="action-train-start"' in page.text
    assert _disabled(page.text, "action-train-start")


def test_start_refused_while_a_sleep_job_is_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = _project(tmp_path)
    job = start_job(
        workspace,
        _PROJECT,
        kind="analyze",
        worker="sleep",
        video_duration_s=100,
        step_s=1,
        sleep_s=30,
    )

    def _refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("enqueue")

    monkeypatch.setattr("excavation_lab.actions.job.enqueue_action_train", _refuse)
    try:
        with TestClient(create_app(workspace)) as client:
            page = client.get("/actions")
            posted = client.post("/actions/train", json={"run_ids": ["run_a"]})
        assert "уже выполняется задание" in page.text
        assert _disabled(page.text, "action-train-start")
        assert "jobs/" not in page.text.split('id="action-job-poll"', 1)[-1] or (
            'data-job-id="' in page.text and "/jobs/" not in page.text
        )
        assert 'data-progress-url="jobs/' not in page.text
        assert posted.status_code == 409
        assert posted.json()["code"] == "job_already_running"
        assert "уже выполняется задание" in posted.json()["error"]
    finally:
        request_cancel(workspace, _PROJECT, job.job_id)


def test_action_job_history_and_traversal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = _project(tmp_path)

    class _Idle:
        pid = 2_147_483_647

    monkeypatch.setattr("excavation_lab.jobs._spawn", lambda *_args, **_kwargs: _Idle())
    job = start_job(
        workspace,
        _PROJECT,
        kind="action_train",
        worker="sleep",
        video_duration_s=100,
        step_s=1,
        sleep_s=30,
    )
    with TestClient(create_app(workspace)) as client:
        fresh = client.get("/actions/job", params={"id": job.job_id})
        refused = client.get("/actions/job", params={"id": "../secret"})
    from excavation_lab.jobs import run_directory

    directory = run_directory(workspace, _PROJECT, job.job_id)
    (directory / "metrics_history.csv").write_text(
        "epoch,train_loss,val_accuracy,val_macro_f1,seconds\n1,0.4,0.5,0.25,1\n",
        encoding="utf-8",
    )
    (directory / "progress.json").write_text(
        json.dumps({"status": "running", "epoch": 1, "epochs": 4, "val_macro_f1": 0.25}),
        encoding="utf-8",
    )
    with TestClient(create_app(workspace)) as client:
        filled = client.get("/actions/job", params={"id": job.job_id})
    assert fresh.status_code == 200
    assert fresh.json()["history"] == []
    assert fresh.json()["epoch"] is None
    assert fresh.json()["val_macro_f1"] is None
    assert fresh.json()["error_code"] is None
    assert refused.status_code == 400
    assert refused.json()["code"] == "invalid_job"
    body = filled.json()
    assert body["epoch"] == 1
    assert body["epochs"] == 4
    assert body["val_macro_f1"] == 0.25
    assert body["history"] == [{"epoch": 1, "train_loss": 0.4, "val_macro_f1": 0.25}]


def test_infer_select_disables_another_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _project(tmp_path)
    _clock(workspace)
    _completed_run(monkeypatch, workspace, "run_same", "det-a")
    _completed_run(monkeypatch, workspace, "run_other", "det-b")
    model_dir = tmp_path / "action-model"
    model_dir.mkdir()
    storage.insert_action_model(
        workspace,
        _PROJECT,
        action_model_id="action_a",
        detector_model_id="det-a",
        source_run_ids=["run_same"],
        path=model_dir,
    )
    with TestClient(create_app(workspace)) as client:
        page = client.get("/actions", params={"model": "action_a"})
    select = re.search(
        r'<select\b[^>]*id="action-infer-run"[^>]*>(.*?)</select>',
        page.text,
        re.DOTALL,
    )
    assert select is not None
    options = select.group(1)
    other = re.search(r"<option\b[^>]*value=\"run_other\"[^>]*>", options)
    same = re.search(r"<option\b[^>]*value=\"run_same\"[^>]*>", options)
    assert other is not None and same is not None
    assert "disabled" in other.group(0)
    assert "другой детектор" in options
    assert "disabled" not in same.group(0)


def test_comparison_hides_test_until_evaluation_file(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    _clock(workspace)
    path = write_blocks(workspace, _PROJECT, _VIDEO, _BLOCKS)
    freeze_split(workspace, _PROJECT, path)
    seed_actions(workspace, _PROJECT, action_seed_titles["ru"])
    add_interval(workspace, _PROJECT, _VIDEO, 10.0, 20.0, "digging")
    add_interval(workspace, _PROJECT, _VIDEO, 100.0, 120.0, "swing_loaded")
    add_interval(workspace, _PROJECT, _VIDEO, 175.0, 185.0, "dumping")
    model_dir = tmp_path / "action-model"
    model_dir.mkdir()
    storage.insert_action_model(
        workspace,
        _PROJECT,
        action_model_id="action_a",
        detector_model_id="det-a",
        source_run_ids=["run_same"],
        path=model_dir,
    )
    with TestClient(create_app(workspace)) as client:
        before = client.get("/actions", params={"model": "action_a"})
        (model_dir / "evaluation_test.json").write_text("{}\n", encoding="utf-8")
        after = client.get("/actions", params={"model": "action_a"})
    hidden = _comparison_json(before.text)
    shown = _comparison_json(after.text)
    assert {row["role"] for row in hidden} == {"train", "validation"}
    assert all(row["action_name"] != "dumping" for row in hidden)
    assert 'data-role="test"' not in _comparison_html(before.text)
    assert catalog("ru").action_test_hidden in _comparison_html(before.text)
    assert any(row["role"] == "test" and row["action_name"] == "dumping" for row in shown)
    assert 'data-role="test"' in _comparison_html(after.text)
    assert 'data-action="dumping"' in _comparison_html(after.text)


def test_older_inference_hides_test_when_only_newer_model_is_final(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    _clock(workspace)
    path = write_blocks(workspace, _PROJECT, _VIDEO, _BLOCKS)
    freeze_split(workspace, _PROJECT, path)
    seed_actions(workspace, _PROJECT, action_seed_titles["ru"])
    add_interval(workspace, _PROJECT, _VIDEO, 10.0, 20.0, "digging")
    add_interval(workspace, _PROJECT, _VIDEO, 175.0, 185.0, "dumping")
    old_dir = tmp_path / "action-old"
    new_dir = tmp_path / "action-new"
    old_dir.mkdir()
    new_dir.mkdir()
    (new_dir / "evaluation_test.json").write_text("{}\n", encoding="utf-8")
    storage.insert_action_model(
        workspace,
        _PROJECT,
        action_model_id="action_old",
        detector_model_id="det-a",
        source_run_ids=["run_old"],
        path=old_dir,
    )
    storage.insert_action_model(
        workspace,
        _PROJECT,
        action_model_id="action_new",
        detector_model_id="det-a",
        source_run_ids=["run_new"],
        path=new_dir,
    )
    storage.insert_action_inference(
        workspace,
        _PROJECT,
        job_id="infer_old",
        source_run_id="run_old",
        action_model_id="action_old",
        path=old_dir,
    )
    storage.insert_action_inference(
        workspace,
        _PROJECT,
        job_id="infer_new",
        source_run_id="run_new",
        action_model_id="action_new",
        path=new_dir,
    )
    with TestClient(create_app(workspace)) as client:
        older = client.get(
            "/actions",
            params={"model": "action_new", "inference": "infer_old"},
        )
        newer = client.get("/actions", params={"inference": "infer_new"})
    hidden = _comparison_json(older.text)
    shown = _comparison_json(newer.text)
    assert all(row["role"] != "test" for row in hidden)
    assert any(row["role"] == "test" for row in shown)


def test_predicted_table_reads_interval_csv(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    _clock(workspace)
    seed_actions(workspace, _PROJECT, action_seed_titles["ru"])
    infer_dir = workspace / "infer"
    infer_dir.mkdir()
    (infer_dir / "action_intervals.csv").write_text(
        "start_s,end_s,duration_s,action_name,mean_confidence,frames\n"
        "1.5,4.5,3,digging,0.8,3\n"
        "4.5,6,1.5,unknown,0.2,2\n",
        encoding="utf-8",
    )
    (infer_dir / "action_run.json").write_text(
        json.dumps(
            {
                "source_run_id": "run_same",
                "action_model_id": "action_a",
                "detector_model_id": "det-a",
                "min_confidence": 0.5,
                "min_duration_s": 1.0,
                "unknown_share": 0.25,
            }
        ),
        encoding="utf-8",
    )
    model_dir = workspace / "action-model"
    model_dir.mkdir()
    storage.insert_action_model(
        workspace,
        _PROJECT,
        action_model_id="action_a",
        detector_model_id="det-a",
        source_run_ids=["run_same"],
        path=model_dir,
    )
    storage.insert_action_inference(
        workspace,
        _PROJECT,
        job_id="infer_a",
        source_run_id="run_same",
        action_model_id="action_a",
        path=infer_dir,
    )
    with TestClient(create_app(workspace)) as client:
        page = client.get("/actions", params={"inference": "infer_a"})
    table = _table(page.text, "action-predicted-table")
    assert "1.5" in table
    assert "4.5" in table
    assert "Копание" in table
    assert "unknown" in table
    assert "0.8" in table
    share = page.text.split('id="action-unknown-share"', 1)[1].split("</p>", 1)[0]
    assert "0.25" in share
    assert "lab-action-span is-unknown" in page.text


def test_evaluation_nulls_and_unstable_caption(tmp_path: Path) -> None:
    workspace = _labelled(tmp_path)
    model_dir = tmp_path / "action-model"
    infer_dir = tmp_path / "infer"
    model_dir.mkdir()
    infer_dir.mkdir()
    (infer_dir / "action_run.json").write_text(
        json.dumps(
            {
                "source_run_id": "run_same",
                "action_model_id": "action_a",
                "min_confidence": 0.5,
                "min_duration_s": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "evaluation_validation_infer_a.json").write_text(
        json.dumps(
            {
                "accuracy": None,
                "macro_f1": None,
                "classes_scored": [],
                "evaluated_frames": 4,
                "unlabelled_frames": 1,
                "unknown_rate": None,
                "selection_unstable": True,
                "per_class": {
                    "digging": {
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "support_frames": 3,
                        "support_intervals": 1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    storage.insert_action_model(
        workspace,
        _PROJECT,
        action_model_id="action_a",
        detector_model_id="det-a",
        source_run_ids=["run_same"],
        path=model_dir,
    )
    storage.insert_action_inference(
        workspace,
        _PROJECT,
        job_id="infer_a",
        source_run_id="run_same",
        action_model_id="action_a",
        path=infer_dir,
    )
    book = catalog("ru")
    with TestClient(create_app(workspace)) as client:
        page = client.get("/actions", params={"inference": "infer_a"})
    table = _table(page.text, "action-eval-table")
    assert book.not_calculated in table
    assert "<td>3</td>" in table
    assert "<td>1</td>" in table
    assert 'id="action-selection-unstable"' in page.text
    assert 'data-selection-unstable="true"' in page.text
    assert book.action_selection_unstable in page.text
    assert f"{book.action_accuracy}: {book.not_calculated}" in page.text


def test_final_button_disabled_with_mismatch_reason(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    model_dir = workspace / "action-model"
    current = workspace / "infer_current"
    other = workspace / "infer_other"
    model_dir.mkdir()
    current.mkdir()
    other.mkdir()
    _run_file(current, "action_a", "run_same", 0.9, 2.0)
    _run_file(other, "action_a", "run_same", 0.5, 1.0)
    (model_dir / "evaluation_validation_infer_other.json").write_text(
        json.dumps({"infer_job_id": "infer_other", "action_model_id": "action_a"}),
        encoding="utf-8",
    )
    storage.insert_action_model(
        workspace,
        _PROJECT,
        action_model_id="action_a",
        detector_model_id="det-a",
        source_run_ids=["run_same"],
        path=model_dir,
    )
    storage.insert_action_inference(
        workspace,
        _PROJECT,
        job_id="infer_current",
        source_run_id="run_same",
        action_model_id="action_a",
        path=current,
    )
    storage.insert_action_inference(
        workspace,
        _PROJECT,
        job_id="infer_other",
        source_run_id="run_same",
        action_model_id="action_a",
        path=other,
    )
    with TestClient(create_app(workspace)) as client:
        page = client.get("/actions", params={"inference": "infer_current"})
    assert _disabled(page.text, "action-final-start")
    reason = page.text.split('id="action-final-reason"', 1)[1].split("</p>", 1)[0]
    assert "Пороги финальной оценки не совпадают" in reason


def test_export_refuses_traversal_job(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    with TestClient(create_app(workspace)) as client:
        refused = client.get("/actions/export", params={"job": "../secret"})
        missing = client.get("/actions/export", params={"job": "missingjob"})
    assert refused.status_code == 400
    assert refused.json()["code"] == "invalid_job"
    assert missing.status_code == 404
    assert missing.json()["code"] == "job_not_found"


def test_train_start_releases_adapter_before_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _project(tmp_path)
    application = create_app(workspace)

    class _Adapter:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    adapter = _Adapter()
    application.state.suggestion_adapter = adapter
    seen: dict[str, bool] = {}

    def _enqueue(*_args: object, **_kwargs: object) -> None:
        seen["closed"] = adapter.closed
        raise LabError("action_not_ready", "остановлено")

    monkeypatch.setattr("excavation_lab.actions.job.enqueue_action_train", _enqueue)
    with TestClient(application) as client:
        response = client.post("/actions/train", json={"run_ids": ["run_a"], "epochs": 1})
    assert seen["closed"] is True
    assert response.status_code == 400
    assert response.json()["code"] == "action_not_ready"
    assert response.json()["error"] == "остановлено"


def _project(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project(_PROJECT, workspace, catalog_id="excavation")
    storage.insert_video(workspace, _PROJECT, _VIDEO, name="clip.mp4")
    return workspace


def _project_job(workspace: Path, job_id: str) -> None:
    with storage.connect(storage.project_sqlite(workspace, _PROJECT)) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                job_id, project_id, status, pid, partial,
                analyzed_end_s, error_code, created_at
            ) VALUES (?, ?, 'completed', NULL, 0, NULL, NULL, ?)
            """,
            (job_id, _PROJECT, "2026-01-01T00:00:00+00:00"),
        )


def _clock(workspace: Path) -> None:
    with storage.connect(storage.project_sqlite(workspace, _PROJECT)) as connection:
        connection.execute(
            """
            UPDATE videos
            SET duration_s = ?, fps_nominal = ?, fps_average = ?,
                time_base = ?, start_offset_s = ?
            WHERE video_id = ?
            """,
            (200.0, 10.0, 10.0, "1/1000", 0.0, _VIDEO),
        )


def _labelled(tmp_path: Path) -> Path:
    workspace = _project(tmp_path)
    _clock(workspace)
    path = write_blocks(workspace, _PROJECT, _VIDEO, _BLOCKS)
    freeze_split(workspace, _PROJECT, path)
    seed_actions(workspace, _PROJECT, action_seed_titles["ru"])
    return workspace


def _completed_run(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    run_id: str,
    model_id: str,
) -> None:
    class _Idle:
        pid = 2_147_483_647

    monkeypatch.setattr("excavation_lab.jobs._spawn", lambda *_args, **_kwargs: _Idle())
    job = start_job(
        workspace,
        _PROJECT,
        kind="analyze",
        worker="sleep",
        video_duration_s=100,
        step_s=1,
        sleep_s=30,
    )
    with sqlite3.connect(lab_database(workspace)) as connection:
        connection.execute(
            "UPDATE lab_jobs SET status = 'completed', pid = NULL WHERE job_id = ?",
            (job.job_id,),
        )
        connection.execute("DELETE FROM workspace_lock")
    storage.insert_analysis_run(
        workspace,
        _PROJECT,
        run_id=run_id,
        job_id=job.job_id,
        video_id=_VIDEO,
        view_id=None,
        model_id=model_id,
        model_origin="custom",
        analysis_fps=2.0,
        source_fps=None,
        export_fps=None,
        start_s=0.0,
        end_s=40.0,
        decode_ms=None,
        encode_ms=None,
    )


def _run_file(
    directory: Path,
    action_model_id: str,
    source_run_id: str,
    min_confidence: float,
    min_duration_s: float,
) -> None:
    (directory / "action_run.json").write_text(
        json.dumps(
            {
                "action_model_id": action_model_id,
                "source_run_id": source_run_id,
                "min_confidence": min_confidence,
                "min_duration_s": min_duration_s,
            }
        ),
        encoding="utf-8",
    )


def _comparison_html(text: str) -> str:
    return text.split('id="action-comparison"', 1)[1].split('id="action-predicted"', 1)[0]


def _comparison_json(text: str) -> list[dict[str, object]]:
    raw = text.split('id="action-comparison-state">', 1)[1].split("</script>", 1)[0]
    loaded = json.loads(raw)
    assert isinstance(loaded, list)
    return loaded


def _table(text: str, element_id: str) -> str:
    match = re.search(rf'<table\b[^>]*id="{element_id}"[^>]*>(.*?)</table>', text, re.DOTALL)
    assert match is not None, element_id
    return match.group(1)


def _disabled(html: str, element_id: str) -> bool:
    match = re.search(rf'<button\b[^>]*id="{element_id}"[^>]*>', html)
    assert match is not None, element_id
    return re.search(r"\sdisabled(?:\s|=|>)", match.group(0)) is not None
