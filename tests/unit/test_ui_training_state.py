"""Training page view-model: no invented loss curve, no early ETA."""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
from starlette.testclient import TestClient

from excavation_lab import jobs as job_module
from excavation_lab.errors import JobAlreadyRunningError
from excavation_lab.jobs import PROGRESS_FILENAME, reopen
from excavation_lab.schemas import JobRecord, JobStatus
from excavation_lab.storage import connect, create_project, project_sqlite
from excavation_lab.ui import copy as ui_copy
from excavation_lab.ui import training_page
from excavation_lab.ui.charts import loss_chart
from excavation_lab.ui.video_page import job_status_line
from excavation_lab.web.app import create_app

_TWO = (
    "epoch,train_loss,epoch_duration_s\n"
    "1,1.2,10\n"
    "2,1.1,20\n"
)
_THREE = (
    "epoch,train_loss,epoch_duration_s\n"
    "1,1.2,10\n"
    "2,1.1,20\n"
    "3,1.0,30\n"
)


def test_missing_log_has_no_chart() -> None:
    view = training_page.training_log_view(None, planned_epochs=50)
    assert view.show_chart is False
    assert view.latest_epoch is None
    assert view.eta_s is None
    assert training_page.should_draw_chart(view, None) is False


def test_header_only_log_has_no_chart() -> None:
    view = training_page.training_log_view("epoch\n", planned_epochs=50)
    assert view.show_chart is False
    assert view.latest_epoch is None
    assert view.eta_s is None
    assert training_page.format_epoch(view.latest_epoch) == ui_copy.NOT_CALCULATED
    assert training_page.format_eta(view.eta_s) == ui_copy.NOT_CALCULATED


def test_two_epochs_have_no_eta() -> None:
    view = training_page.training_log_view(_TWO, planned_epochs=50)
    assert view.show_chart is True
    assert view.latest_epoch == 2
    assert view.measured_epochs == 2
    assert view.eta_s is None
    assert training_page.format_eta(view.eta_s) == ui_copy.NOT_CALCULATED
    assert training_page.should_draw_chart(view, None) is True


_COMPACT = (
    "epoch,train/loss_bbox_epoch,train/loss_epoch,epoch_duration_s\n"
    "1,0.4,1.2,10\n"
    "2,0.3,1.1,20\n"
)


def test_loss_epoch_beats_bbox_component() -> None:
    view = training_page.training_log_view(_COMPACT, planned_epochs=50)
    assert view.losses == (1.2, 1.1)
    assert view.eta_s is None
    assert training_page.format_eta(view.eta_s) == ui_copy.NOT_CALCULATED


def test_live_metrics_csv_is_read(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text(_TWO, encoding="utf-8")
    assert not (run_dir / "metrics_history.csv").exists()
    found = training_page._history_file(tmp_path, "pilot", "job1", run_dir, None)
    assert found == run_dir / "metrics.csv"
    view = training_page._view_for_run(tmp_path, "pilot", "job1", run_dir)
    assert view.show_chart is True
    assert view.eta_s is None
    assert view.losses == (1.2, 1.1)


def test_header_only_metrics_csv_does_not_chart(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text("epoch\n", encoding="utf-8")
    view = training_page._view_for_run(tmp_path, "pilot", "job1", run_dir)
    assert view.show_chart is False
    assert view.latest_epoch is None
    assert view.eta_s is None
    assert training_page.format_epoch(view.latest_epoch) == ui_copy.NOT_CALCULATED
    assert training_page.format_eta(view.eta_s) == ui_copy.NOT_CALCULATED


def test_three_measured_epochs_set_eta() -> None:
    view = training_page.training_log_view(_THREE, planned_epochs=10)
    assert view.measured_epochs == 3
    assert view.eta_s == pytest.approx(20.0 * 7)


def test_dataset_not_ready_hides_a_curve() -> None:
    view = training_page.training_log_view(_THREE, planned_epochs=10)
    assert view.show_chart is True
    assert training_page.should_draw_chart(view, "dataset_not_ready") is False


def test_active_job_disables_start() -> None:
    assert training_page.start_enabled([JobStatus.running]) is False
    assert training_page.start_enabled([JobStatus.queued]) is False
    assert training_page.start_enabled([JobStatus.cancel_requested]) is False
    assert training_page.start_enabled(
        [JobStatus.completed, JobStatus.failed, JobStatus.cancelled]
    )
    assert training_page.start_enabled([]) is True


def test_page_does_not_import_the_detector() -> None:
    source = Path(training_page.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules: list[str] = []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module.split(".", 1)[0])
            names.extend(alias.name for alias in node.names)
    assert "rfdetr" not in modules
    assert "starlette" not in modules
    assert "jinja2" not in modules
    assert "DetectorAdapter" not in names
    assert "RFDETR" not in source
    assert "shell=True" not in source
    assert "import streamlit" not in source
    assert "st.fragment" not in source
    assert "enqueue_train" in names
    assert "request_cancel" in names


def test_form_settings_keep_run_test_off() -> None:
    form = _form(profile="smoke", variant="nano", epochs=3)
    body = training_page.form_settings(form)
    assert body["run_test"] is False
    assert body["epochs"] == 3
    assert body["variant"] == "nano"
    assert body["resolution"] == 384


def test_active_status_does_not_enqueue(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("enqueued")

    monkeypatch.setattr(training_page, "enqueue_train", fail)
    outcome = training_page.request_training(
        Path("."),
        "pilot",
        _form(),
        [JobStatus.running],
    )
    assert outcome.job_id is None
    assert outcome.code == "job_already_running"


def test_missing_dataset_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("enqueued")

    monkeypatch.setattr(training_page, "enqueue_train", fail)
    outcome = training_page.request_training(Path("."), "pilot", _form(dataset_id=None), [])
    assert outcome.code == "dataset_not_ready"
    assert outcome.job_id is None


def test_variant_mismatch_does_not_start(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("start_job")

    monkeypatch.setattr("excavation_lab.jobs.start_job", fail)
    outcome = training_page.request_training(
        Path("."),
        "pilot",
        _form(variant="small", profile="smoke"),
        [],
    )
    assert outcome.code == "profile_variant"
    assert outcome.job_id is None


def test_running_job_disables_the_start_button(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    create_project("pilot", tmp_path)
    monkeypatch.setattr(
        training_page,
        "reopen",
        lambda _workspace: [JobRecord(job_id="abc123", status=JobStatus.running, pid=123)],
    )
    calls = {"n": 0}

    def fail(*_args: object, **_kwargs: object) -> None:
        calls["n"] += 1
        raise AssertionError("enqueued")

    monkeypatch.setattr(training_page, "enqueue_train", fail)
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/training")
    assert page.status_code == 200
    assert _disabled(page.text, "train-start")
    assert calls["n"] == 0
    assert not _chart_markup(page.text)


def test_profile_defaults_fill_advanced_fields(tmp_path: Path) -> None:
    create_project("pilot", tmp_path)
    book = ui_copy.catalog("ru")
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/training")
        assert page.status_code == 200
        text = page.text
        assert ui_copy.SMOKE_HELP in text
        assert ui_copy.NOT_CALCULATED in text
        assert "dataset_not_ready" in text
        assert _selected(text, "train-variant") == ("nano", "Nano")
        assert _selected(text, "train-profile") == ("smoke", ui_copy.SMOKE_HELP)
        assert _selected(text, "train-device") == ("auto", "auto")
        assert _input_value(text, "train-resolution") == "384"
        assert _input_value(text, "train-epochs") == "2"
        assert _input_value(text, "train-batch") == "1"
        assert _input_value(text, "train-accumulation") == "1"
        assert _input_value(text, "train-workers") == "0"
        assert _input_value(text, "train-seed") == "42"
        assert float(_input_value(text, "train-lr")) == pytest.approx(1e-4)
        assert float(_input_value(text, "train-encoder-lr")) == pytest.approx(1e-5)
        assert _node_text(text, "train-epoch") == ui_copy.NOT_CALCULATED
        assert _node_text(text, "train-checkpoint") == ui_copy.NOT_CALCULATED
        assert _node_text(text, "train-eta") == ui_copy.NOT_CALCULATED
        assert 'name="run_test"' not in text
        assert 'value="overfit"' not in text
        assert 'action="/training/start"' in text
        assert not _disabled(text, "train-start")
        assert not _chart_markup(text)
        switched = client.get("/training?profile=pilot-small")
    assert switched.status_code == 200
    body = switched.text
    assert _selected(body, "train-profile") == ("pilot-small", ui_copy.PILOT_HELP)
    assert _node_text(body, "train-profile-help") == ui_copy.PILOT_HELP
    assert _selected(body, "train-variant") == ("small", "Small")
    assert _input_value(body, "train-resolution") == "512"
    assert _input_value(body, "train-epochs") == "50"
    assert _input_value(body, "train-accumulation") == "4"
    assert book.pilot_help == ui_copy.PILOT_HELP
    assert not _chart_markup(body)


def test_not_ready_dataset_does_not_enqueue_twice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    create_project("pilot", tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _insert_dataset(tmp_path, "snapshot", snapshot)
    calls: list[str] = []
    real = training_page.enqueue_train

    def wrapped(*args: object, **kwargs: object) -> object:
        profile = kwargs.get("profile", args[3] if len(args) > 3 else "")
        calls.append(str(profile))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(training_page, "enqueue_train", wrapped)
    with TestClient(create_app(tmp_path)) as client:
        text = _post(client, "/training/start", _fields(dataset_id="snapshot"))
        assert "dataset_not_ready" in text
        assert not _chart_markup(text)
        assert _job_count(tmp_path) == 0
        assert calls == ["smoke"]
        again = client.get("/training")
        assert again.status_code == 200
        assert calls == ["smoke"]
        assert _job_count(tmp_path) == 0
        text = _post(client, "/training/start", _fields(dataset_id="snapshot"))
    assert "dataset_not_ready" in text
    assert calls == ["smoke", "smoke"]
    assert _job_count(tmp_path) == 0
    assert not _chart_markup(text)


def test_start_without_a_dataset_does_not_enqueue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    create_project("pilot", tmp_path)
    calls = {"n": 0}

    def fail(*_args: object, **_kwargs: object) -> None:
        calls["n"] += 1
        raise AssertionError("enqueued")

    monkeypatch.setattr(training_page, "enqueue_train", fail)
    with TestClient(create_app(tmp_path)) as client:
        text = _post(client, "/training/start", _fields())
        assert "dataset_not_ready" in text
        assert not _chart_markup(text)
        assert calls["n"] == 0
        assert _job_count(tmp_path) == 0
        again = client.get("/training")
    assert again.status_code == 200
    assert calls["n"] == 0
    assert _job_count(tmp_path) == 0


@pytest.mark.parametrize("status", ["queued", "running", "cancel_requested"])
def test_post_during_an_active_job_does_not_enqueue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
) -> None:
    create_project("pilot", tmp_path)
    pid = os.getpid() if status == "running" else None
    _insert_job(tmp_path, "activejob01", status=status, pid=pid)
    calls = {"n": 0}

    def fail(*_args: object, **_kwargs: object) -> None:
        calls["n"] += 1
        raise AssertionError("enqueued")

    monkeypatch.setattr(training_page, "enqueue_train", fail)
    before = _job_count(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/training")
        assert _disabled(page.text, "train-start")
        text = _post(client, "/training/start", _fields())
    assert JobAlreadyRunningError().message in text
    assert "job_already_running" in text
    assert calls["n"] == 0
    assert _job_count(tmp_path) == before


def test_variant_mismatch_post_does_not_enqueue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    create_project("pilot", tmp_path)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    _insert_dataset(tmp_path, "snapshot", snapshot)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("start_job")

    monkeypatch.setattr("excavation_lab.models.train.start_job", fail)
    data = _fields(dataset_id="snapshot")
    data["variant"] = "small"
    with TestClient(create_app(tmp_path)) as client:
        text = _post(client, "/training/start", data)
    assert "profile_variant" in text
    assert _job_count(tmp_path) == 0


def test_overfit_is_not_on_the_form(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    create_project("pilot", tmp_path)
    calls = {"n": 0}

    def fail(*_args: object, **_kwargs: object) -> None:
        calls["n"] += 1
        raise AssertionError("enqueued")

    monkeypatch.setattr(training_page, "enqueue_train", fail)
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/training")
        assert 'value="overfit"' not in page.text
        text = _post(client, "/training/start", {"profile": "overfit", "variant": "nano"})
    assert "unknown_profile" in text
    assert calls["n"] == 0
    assert _job_count(tmp_path) == 0


def test_cancel_uses_the_active_job_and_ignores_another_pid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    create_project("pilot", tmp_path)
    job_id = "activejob01"
    _insert_job(tmp_path, job_id, status="queued")
    seen: list[str] = []
    killed: list[int] = []
    real = training_page.request_training_cancel

    def wrapped(
        workspace: Path,
        project_id: str,
        posted_job_id: str,
        text: ui_copy.Catalog | None = None,
    ) -> training_page.TrainAttempt:
        seen.append(posted_job_id)
        return real(workspace, project_id, posted_job_id, text)

    def record_kill(pid: int, _sig: int) -> None:
        killed.append(pid)

    monkeypatch.setattr(training_page, "request_training_cancel", wrapped)
    monkeypatch.setattr(os, "kill", record_kill)
    with TestClient(create_app(tmp_path)) as client:
        _post(
            client,
            "/training/cancel",
            {"job_id": "ffffffffffffffffffffffffffffffff", "pid": "424242"},
        )
        assert seen == []
        assert killed == []
        assert reopen(tmp_path)[0].status is JobStatus.queued
        text = _post(client, "/training/cancel", {"job_id": job_id, "pid": "424242"})
    assert seen == [job_id]
    assert killed == []
    assert 424242 not in killed
    assert reopen(tmp_path)[0].status is JobStatus.cancelled
    assert ui_copy.catalog("ru").train_cancelled in text


def test_two_measured_epochs_on_the_page_have_no_eta(tmp_path: Path) -> None:
    create_project("pilot", tmp_path)
    job_id = "epochjob01"
    run_dir = _insert_job(tmp_path, job_id, status="completed")
    (run_dir / "train_request.json").write_text(
        '{"dataset_id":"snapshot","profile":"smoke"}\n',
        encoding="utf-8",
    )
    (run_dir / "metrics.csv").write_text(_TWO, encoding="utf-8")
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/training")
        progress = client.get(f"/jobs/{job_id}/progress")
    assert page.status_code == 200
    assert _node_text(page.text, "train-epoch") == "2"
    assert _node_text(page.text, "train-eta") == ui_copy.NOT_CALCULATED
    assert _node_text(page.text, "train-checkpoint") == ui_copy.NOT_CALCULATED
    assert ui_copy.TRAINING_EMPTY not in page.text
    assert not _chart_markup(page.text)
    assert progress.status_code == 200
    body = progress.json()
    assert body["chart"] == loss_chart((1, 2), (1.2, 1.1), "dark").to_dict()
    assert body["progress"] is None


def test_header_only_log_on_the_page_is_not_a_curve(tmp_path: Path) -> None:
    create_project("pilot", tmp_path)
    job_id = "headerjob1"
    run_dir = _insert_job(tmp_path, job_id, status="completed")
    (run_dir / "train_request.json").write_text(
        '{"profile":"smoke"}\n',
        encoding="utf-8",
    )
    (run_dir / "metrics.csv").write_text("epoch\n", encoding="utf-8")
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/training")
    assert page.status_code == 200
    assert _node_text(page.text, "train-epoch") == ui_copy.NOT_CALCULATED
    assert _node_text(page.text, "train-epoch") != "0"
    assert _node_text(page.text, "train-eta") == ui_copy.NOT_CALCULATED
    assert ui_copy.TRAINING_EMPTY in page.text
    assert not _chart_markup(page.text)
    assert training_page.should_draw_chart(
        training_page.training_log_view("epoch\n", planned_epochs=50),
        None,
    ) is False


def test_progress_payload_includes_orphaned_and_the_status_line(tmp_path: Path) -> None:
    create_project("pilot", tmp_path)
    job_id = "orphaned01"
    run_dir = _insert_job(
        tmp_path,
        job_id,
        status="failed",
        partial=True,
        analyzed_end_s=12.5,
        error_code="orphaned",
        orphaned=True,
    )
    (run_dir / "metrics.csv").write_text(_TWO, encoding="utf-8")
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / PROGRESS_FILENAME).write_text('{"epoch": 0}\n', encoding="utf-8")
    names_before = sorted(path.name for path in run_dir.iterdir())
    with TestClient(create_app(tmp_path)) as client:
        missing = client.get("/jobs/doesnotexist/progress")
        response = client.get(
            f"/jobs/{job_id}/progress",
            params={"run_dir": str(decoy)},
        )
        zero = _insert_job(
            tmp_path,
            "zeroend01",
            status="completed",
            analyzed_end_s=0,
        )
        zero_response = client.get("/jobs/zeroend01/progress")
    assert missing.status_code == 404
    assert response.status_code == 200
    match = next(job for job in reopen(tmp_path) if job.job_id == job_id)
    body = response.json()
    assert "orphaned" in body
    assert body["orphaned"] is True
    assert body["partial"] is True
    assert body["analyzed_end_s"] == 12.5
    assert body["error_code"] == "orphaned"
    assert body["status"] == "failed"
    assert body["job_id"] == job_id
    assert body["line"] == job_status_line(match)
    assert body["progress"] is None
    assert body["chart"] is None
    assert "epoch" not in response.text
    assert sorted(path.name for path in run_dir.iterdir()) == names_before
    assert not (run_dir / PROGRESS_FILENAME).exists()
    assert zero.is_dir()
    zero_body = zero_response.json()
    assert zero_body["analyzed_end_s"] is None
    assert zero_body["progress"] is None
    assert zero_body["chart"] is None
    written = {"step": 2, "loss": 1.1}
    (run_dir / PROGRESS_FILENAME).write_text(
        json.dumps(written),
        encoding="utf-8",
    )
    with TestClient(create_app(tmp_path)) as client:
        filled = client.get(f"/jobs/{job_id}/progress")
    assert filled.status_code == 200
    filled_body = filled.json()
    assert filled_body["progress"] == written
    assert filled_body["chart"] is None
    assert json.loads((run_dir / PROGRESS_FILENAME).read_text(encoding="utf-8")) == written


def test_poll_uses_a_relative_progress_url(tmp_path: Path) -> None:
    create_project("pilot", tmp_path)
    job_id = "queuedjob01"
    _insert_job(tmp_path, job_id, status="queued")
    script = (
        Path(__file__).resolve().parents[2] / "src" / "excavation_lab" / "web" / "static" / "lab.js"
    ).read_text(encoding="utf-8")
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/training")
    assert page.status_code == 200
    assert f'data-progress-url="jobs/{job_id}/progress"' in page.text
    assert 'data-status="queued"' in page.text
    assert "data-job-poll" in page.text
    assert "http://" not in page.text
    assert "https://" not in page.text
    match = reopen(tmp_path)[0]
    assert _node_text(page.text, "train-status") == job_status_line(match)
    assert "http://" not in script
    assert "https://" not in script
    for name in (
        "queued",
        "running",
        "cancel_requested",
        "completed",
        "failed",
        "cancelled",
    ):
        assert name in script
    assert "data-progress-url" in script
    assert "body.line" in script
    assert "data-job-line" in script
    assert "loss-chart" not in script
    assert "altair" not in script


def _form(
    *,
    dataset_id: str | None = "snapshot",
    variant: str = "nano",
    profile: str = "smoke",
    epochs: int = 2,
) -> training_page.TrainForm:
    return training_page.TrainForm(
        dataset_id=dataset_id,
        variant=variant,
        profile=profile,
        device="auto",
        resolution=384,
        epochs=epochs,
        batch_size=1,
        gradient_accumulation=1,
        learning_rate=1e-4,
        encoder_learning_rate=1e-5,
        workers=0,
        seed=42,
    )


def _fields(
    profile: str = "smoke",
    dataset_id: str | None = None,
) -> dict[str, str]:
    values = training_page.profile_field_values(profile)
    data = {
        "profile": profile,
        "variant": str(values["variant"]),
        "device": "auto",
        "resolution": str(values["resolution"]),
        "epochs": str(values["epochs"]),
        "batch_size": str(values["batch_size"]),
        "gradient_accumulation": str(values["gradient_accumulation"]),
        "learning_rate": str(values["learning_rate"]),
        "encoder_learning_rate": str(values["encoder_learning_rate"]),
        "workers": str(values["workers"]),
        "seed": str(values["seed"]),
    }
    if dataset_id:
        data["dataset_id"] = dataset_id
    return data


def _post(client: TestClient, path: str, data: dict[str, str]) -> str:
    response = client.post(path, data=data, follow_redirects=False)
    assert response.status_code == 303, response.text
    location = urlparse(response.headers["location"])
    assert location.path == "/training"
    assert location.query == ""
    page = client.get("/training")
    assert page.status_code == 200
    return page.text


def _insert_job(
    workspace: Path,
    job_id: str,
    *,
    status: str,
    pid: int | None = None,
    partial: bool = False,
    analyzed_end_s: float | None = None,
    error_code: str | None = None,
    orphaned: bool = False,
) -> Path:
    """Write one lab_jobs row. Does not spawn a worker."""
    run_dir = job_module.run_directory(workspace, "pilot", job_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with job_module._lab(workspace) as connection:
        connection.execute(
            """
            INSERT INTO lab_jobs (
                job_id, project_id, kind, worker, status, pid, partial,
                analyzed_end_s, error_code, orphaned, run_dir,
                video_duration_s, step_s, sleep_s, created_at
            ) VALUES (
                ?, ?, 'train', 'sleep', ?, ?, ?, ?, ?, ?, ?, 10, 1, 0.05, ?
            )
            """,
            (
                job_id,
                "pilot",
                status,
                pid,
                1 if partial else 0,
                analyzed_end_s,
                error_code,
                1 if orphaned else 0,
                str(run_dir),
                "2026-01-01T00:00:00+00:00",
            ),
        )
    return run_dir


def _insert_dataset(workspace: Path, dataset_id: str, path: Path) -> None:
    db_path = project_sqlite(workspace, "pilot")
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO dataset_versions (
                dataset_id, project_id, content_hash, path, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                "pilot",
                "hash",
                str(path),
                "2026-01-01T00:00:00+00:00",
            ),
        )


def _job_count(workspace: Path) -> int:
    total = _count(workspace / "lab.sqlite", "lab_jobs")
    total += _count(project_sqlite(workspace, "pilot"), "jobs")
    return total


def _count(path: Path, table: str) -> int:
    if not path.is_file():
        return 0
    with connect(path) as connection:
        row = connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
    assert row is not None
    return int(row["n"])


def _input_value(html: str, element_id: str) -> str:
    match = re.search(rf'<input\b[^>]*id="{element_id}"[^>]*>', html)
    assert match is not None, element_id
    found = re.search(r'value="([^"]*)"', match.group(0))
    assert found is not None, element_id
    return found.group(1)


def _select_body(html: str, element_id: str) -> str:
    match = re.search(
        rf'<select\b[^>]*id="{element_id}"[^>]*>(.*?)</select>',
        html,
        re.DOTALL,
    )
    assert match is not None, element_id
    return match.group(1)


def _selected(html: str, element_id: str) -> tuple[str, str]:
    body = _select_body(html, element_id)
    for attrs, label in re.findall(r"<option\b([^>]*)>([^<]*)</option>", body):
        if re.search(r"\sselected(?:\s|>|$)", attrs):
            value = re.search(r'value="([^"]*)"', attrs)
            assert value is not None, attrs
            return value.group(1), label
    raise AssertionError(element_id)


def _node_text(html: str, element_id: str) -> str:
    match = re.search(rf'id="{element_id}"[^>]*>([^<]*)</', html)
    assert match is not None, element_id
    return match.group(1)


def _disabled(html: str, element_id: str) -> bool:
    match = re.search(rf'<button\b[^>]*id="{element_id}"[^>]*>', html)
    assert match is not None, element_id
    return re.search(r"\sdisabled(?:\s|=|>)", match.group(0)) is not None


def _chart_markup(html: str) -> bool:
    lowered = html.lower()
    return (
        "lab-chart" in html
        or 'id="loss-chart"' in html
        or "<canvas" in lowered
        or "<svg" in lowered
    )
