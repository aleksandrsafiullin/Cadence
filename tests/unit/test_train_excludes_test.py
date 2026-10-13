"""Training opens train and valid, never the project's test split.

The trainer is a double. It records kwargs and the directories it opens.
It does not call ``RFDETR.train`` and it does not download weights.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from excavation_lab import jobs, storage
from excavation_lab.cli import app, train_command
from excavation_lab.models import train as train_mod
from excavation_lab.models.augment import EVAL_AUGMENTATION, MILD_PRESET
from excavation_lab.models.train import (
    STABLE_VALIDATION_IMAGES,
    build_validation_metrics,
)
from excavation_lab.models.train_config import resolve_train_profile
from excavation_lab.schemas import JobStatus

_RUNNER = CliRunner()
_RUNTIME_KEYS = {
    "dataset_dir",
    "output_dir",
    "device",
    "log_per_class_metrics",
    "tensorboard",
    "wandb",
    "mlflow",
    "clearml",
    "progress_bar",
}


class _DummyProcess:
    def __init__(self) -> None:
        self.pid = os.getpid()

    def poll(self) -> None:
        return None


def _dummy_spawn(*_args: object, **_kwargs: object) -> _DummyProcess:
    return _DummyProcess()


def _write_split(root: Path, name: str, count: int) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    images = []
    for index in range(count):
        file_name = f"{name}_{index}.jpg"
        (directory / file_name).write_bytes(b"\xff\xd8\xff\xd9")
        images.append(
            {
                "id": index + 1,
                "file_name": file_name,
                "width": 8,
                "height": 8,
            }
        )
    (directory / "_annotations.coco.json").write_text(
        json.dumps({"annotations": [], "categories": [], "images": images}),
        encoding="utf-8",
    )


def _dataset(root: Path, *, train: int, valid: int, test: int = 1) -> Path:
    dataset = root / "dataset"
    _write_split(dataset, "train", train)
    _write_split(dataset, "valid", valid)
    _write_split(dataset, "test", test)
    (dataset / "test" / "sentinel.txt").write_text("leave-me", encoding="utf-8")
    return dataset


def _ready_workspace(tmp_path: Path, *, train: int = 1, valid: int = 1) -> tuple[Path, str, Path]:
    workspace = tmp_path / "workspace"
    storage.create_project("pilot", workspace)
    dataset = _dataset(workspace, train=train, valid=valid)
    dataset_id = "ds1"
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            """
            INSERT INTO dataset_versions (
                dataset_id, project_id, content_hash, path, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                "pilot",
                "ab" * 32,
                str(dataset),
                datetime.now(UTC).isoformat(),
            ),
        )
    return workspace, dataset_id, dataset


def _invoke(workspace: Path, dataset_id: str, profile: str):
    return _RUNNER.invoke(
        app,
        [
            "train",
            "--project",
            "pilot",
            "--dataset",
            dataset_id,
            "--profile",
            profile,
            "--workspace",
            str(workspace),
            "--json",
        ],
    )


def _cancel_active(workspace: Path) -> None:
    if not jobs.lab_database(workspace).is_file():
        return
    for record in jobs.reopen(workspace):
        if record.status in {
            JobStatus.queued,
            JobStatus.running,
            JobStatus.cancel_requested,
        }:
            jobs.request_cancel(workspace, "pilot", record.job_id, kill_after_s=0.5)


def _wait_calls(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "trainer_calls.json"
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list) and loaded:
                return loaded
        time.sleep(0.02)
    log = ""
    worker_log = run_dir / "worker.log"
    if worker_log.is_file():
        log = worker_log.read_text(encoding="utf-8", errors="replace")
    train_log = run_dir / "train.log"
    if train_log.is_file():
        log = f"{log}\n{train_log.read_text(encoding='utf-8')}"
    raise AssertionError(f"trainer double wrote no call\n{log}")


def _wait_status(workspace: Path, job_id: str, statuses: set[JobStatus]):
    deadline = time.monotonic() + 45
    last = None
    run_dir = jobs.run_directory(workspace, "pilot", job_id)
    while time.monotonic() < deadline:
        last = jobs.get_job(workspace, "pilot", job_id)
        if last.status in statuses:
            return last
        time.sleep(0.02)
    log = ""
    worker_log = run_dir / "worker.log"
    if worker_log.is_file():
        log = worker_log.read_text(encoding="utf-8", errors="replace")
    raise AssertionError(f"status stayed {last}\n{log}")


def _round_trip(value: object) -> object:
    return json.loads(json.dumps(value))


def _assert_hides_test(call: dict[str, object], dataset: Path, profile: str) -> dict[str, object]:
    kwargs = call["kwargs"]
    opened = call["opened"]
    assert isinstance(kwargs, dict)
    assert isinstance(opened, list)
    assert call["model_constructed"] is False
    resolved = resolve_train_profile(profile)
    assert set(kwargs) == set(resolved.train_kwargs) | _RUNTIME_KEYS
    assert kwargs["run_test"] is False
    assert kwargs["augmentation_train"] == _round_trip(MILD_PRESET)
    assert kwargs["augmentation_eval"] == _round_trip(EVAL_AUGMENTATION)
    assert kwargs["aug_config"] == {}
    assert kwargs["scale_jitter"] is False
    assert kwargs["multi_scale"] is False
    assert type(kwargs["batch_size"]) is int
    assert kwargs["batch_size"] == resolved.train_kwargs["batch_size"]
    assert kwargs["batch_size"] != "auto"
    assert kwargs["device"] in {"cuda", "mps", "cpu"}
    assert "gradient_checkpointing" not in kwargs
    assert kwargs["log_per_class_metrics"] is True
    assert kwargs["tensorboard"] is False
    assert kwargs["best_model_metric"] == "map"
    assert kwargs["amp_dtype"] is None
    names = [Path(str(path)).name for path in opened]
    assert "train" in names
    assert "valid" in names
    assert "test" not in names
    for path in opened:
        assert "test" not in Path(str(path)).parts
    train_dir = (dataset / "train").resolve()
    valid_dir = (dataset / "valid").resolve()
    test_dir = (dataset / "test").resolve()
    assert any(Path(str(path)) == train_dir for path in opened)
    assert any(Path(str(path)) == valid_dir for path in opened)
    assert not any(Path(str(path)) == test_dir or test_dir in Path(str(path)).parents for path in opened)
    assert (dataset / "test" / "sentinel.txt").read_text(encoding="utf-8") == "leave-me"
    view = Path(str(kwargs["dataset_dir"]))
    assert (view / "train").is_dir()
    assert (view / "valid").is_dir()
    assert not (view / "test").exists()
    assert view.resolve() != dataset.resolve()
    return kwargs


def _start_held(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: str,
) -> tuple[Path, str, Path, str, Path]:
    monkeypatch.setenv(train_mod.FAKE_TRAINER_ENV, "hold")
    workspace, dataset_id, dataset = _ready_workspace(tmp_path)
    started = _invoke(workspace, dataset_id, profile)
    assert started.exit_code == 0, started.output
    body = json.loads(started.stdout)
    assert body["ok"] is True
    job_id = body["data"]["job_id"]
    assert body["data"]["status"] == "running"
    run_dir = jobs.run_directory(workspace, "pilot", job_id)
    assert run_dir == workspace / "projects" / "pilot" / "runs" / job_id
    return workspace, dataset_id, dataset, job_id, run_dir


def test_smoke_hides_test_and_a_second_train_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import rfdetr
    from rfdetr import variants

    resolve_train_profile("smoke")
    resolve_train_profile("pilot-small")
    constructed: list[str] = []

    class _Spy:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            constructed.append("model")
            raise AssertionError("parent constructed an RF-DETR model")

    monkeypatch.setattr(rfdetr, "RFDETRNano", _Spy)
    monkeypatch.setattr(rfdetr, "RFDETRSmall", _Spy)
    monkeypatch.setattr(variants, "RFDETRNano", _Spy)
    monkeypatch.setattr(variants, "RFDETRSmall", _Spy)
    workspace, dataset_id, dataset, job_id, run_dir = _start_held(monkeypatch, tmp_path, "smoke")
    try:
        calls = _wait_calls(run_dir)
        assert len(calls) == 1
        kwargs = _assert_hides_test(calls[0], dataset, "smoke")
        assert kwargs["early_stopping"] is False
        assert "early_stopping_patience" not in kwargs
        assert "early_stopping_min_delta" not in kwargs
        assert kwargs["epochs"] == 2
        log = (run_dir / "train.log").read_text(encoding="utf-8")
        assert log.count("device=") == 1
        assert "run_test=false" in log
        assert "batch_size=1" in log
        resolved_text = (run_dir / "resolved_train_config.json").read_text(encoding="utf-8")
        assert json.loads(resolved_text)["run_test"] is False
        assert str(dataset / "test") not in resolved_text
        assert jobs.get_job(workspace, "pilot", job_id).status == JobStatus.running
        second = _invoke(workspace, dataset_id, "pilot-small")
        assert second.exit_code != 0, second.output
        failure = json.loads(second.stdout)
        assert failure["ok"] is False
        assert failure["error"]["code"] == "job_already_running"
        assert len(json.loads((run_dir / "trainer_calls.json").read_text(encoding="utf-8"))) == 1
        assert constructed == []
    finally:
        _cancel_active(workspace)


def test_pilot_small_enqueues_and_a_second_train_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace, dataset_id, dataset, job_id, run_dir = _start_held(
        monkeypatch,
        tmp_path,
        "pilot-small",
    )
    try:
        calls = _wait_calls(run_dir)
        kwargs = _assert_hides_test(calls[0], dataset, "pilot-small")
        assert kwargs["early_stopping"] is True
        assert kwargs["early_stopping_patience"] == 12
        assert kwargs["early_stopping_min_delta"] == _round_trip(0.001)
        assert kwargs["epochs"] == 50
        assert kwargs["grad_accum_steps"] == 4
        assert jobs.get_job(workspace, "pilot", job_id).status == JobStatus.running
        second = _invoke(workspace, dataset_id, "smoke")
        assert second.exit_code != 0, second.output
        failure = json.loads(second.stdout)
        assert failure["error"]["code"] == "job_already_running"
    finally:
        _cancel_active(workspace)


@pytest.mark.parametrize(("train_count", "valid_count"), [(0, 1), (1, 0), (0, 0)])
def test_dataset_not_ready_exits_nonzero_and_writes_no_weights(
    tmp_path: Path,
    train_count: int,
    valid_count: int,
) -> None:
    workspace, dataset_id, dataset = _ready_workspace(
        tmp_path,
        train=train_count,
        valid=valid_count,
    )
    try:
        result = _invoke(workspace, dataset_id, "smoke")
        assert result.exit_code != 0, result.output
        body = json.loads(result.stdout)
        assert body["ok"] is False
        assert body["error"]["code"] == "dataset_not_ready"
        assert "weights.pth" in body["error"]["message"]
        assert any("а" <= char.lower() <= "я" for char in body["error"]["message"])
        runs = workspace / "projects" / "pilot" / "runs"
        assert list(runs.iterdir()) == []
        assert list(workspace.rglob("*.pth")) == []
        assert (dataset / "test" / "sentinel.txt").read_text(encoding="utf-8") == "leave-me"
    finally:
        _cancel_active(workspace)


def test_empty_output_is_empty_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(train_mod.FAKE_TRAINER_ENV, "empty")
    workspace, dataset_id, _dataset_dir = _ready_workspace(tmp_path)
    try:
        started = _invoke(workspace, dataset_id, "smoke")
        assert started.exit_code == 0, started.output
        job_id = json.loads(started.stdout)["data"]["job_id"]
        record = _wait_status(workspace, job_id, {JobStatus.failed})
        assert record.error_code == "empty_checkpoint"
        assert list(workspace.rglob("*.pth")) == []
        assert list(workspace.rglob("weights.pth")) == []
        metrics_path = jobs.run_directory(workspace, "pilot", job_id) / "validation_metrics.json"
        text = metrics_path.read_text(encoding="utf-8")
        metrics = json.loads(text)
        assert metrics["selection_metric"] == "val/mAP_50_95"
        assert metrics["box_map_50_95"] == 0.5
        assert metrics["bucket_ap"] == 0.25
        assert metrics["selection_unstable"] is True
        assert "0.125" not in text
        assert "0.75" not in text
        assert len(_wait_calls(jobs.run_directory(workspace, "pilot", job_id))) == 1
    finally:
        _cancel_active(workspace)


def test_failed_backend_is_stored_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(train_mod.FAKE_TRAINER_ENV, "fail")
    workspace, dataset_id, _dataset_dir = _ready_workspace(tmp_path)
    try:
        started = _invoke(workspace, dataset_id, "pilot-small")
        assert started.exit_code == 0, started.output
        job_id = json.loads(started.stdout)["data"]["job_id"]
        record = _wait_status(workspace, job_id, {JobStatus.failed})
        assert record.error_code is not None
        assert "RuntimeError" in record.error_code
        assert "mps backend failed: simulated" in record.error_code
        calls = _wait_calls(jobs.run_directory(workspace, "pilot", job_id))
        assert len(calls) == 1
        assert list(workspace.rglob("*.pth")) == []
        log = (jobs.run_directory(workspace, "pilot", job_id) / "train.log").read_text(encoding="utf-8")
        assert log.count("device=") == 1
    finally:
        _cancel_active(workspace)


def test_metrics_use_validation_map_and_mark_a_small_set_unstable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(train_mod.FAKE_TRAINER_ENV, "metrics")
    workspace, dataset_id, _dataset_dir = _ready_workspace(tmp_path)
    try:
        started = _invoke(workspace, dataset_id, "smoke")
        assert started.exit_code == 0, started.output
        job_id = json.loads(started.stdout)["data"]["job_id"]
        record = _wait_status(workspace, job_id, {JobStatus.completed, JobStatus.failed})
        assert record.status == JobStatus.completed, record.error_code
        weights = jobs.run_directory(workspace, "pilot", job_id) / "weights.pth"
        assert weights.is_file()
        assert weights.stat().st_size > 0
        text = (jobs.run_directory(workspace, "pilot", job_id) / "validation_metrics.json").read_text(
            encoding="utf-8"
        )
        metrics = json.loads(text)
        assert metrics["box_map_50_95"] == 0.5
        assert metrics["bucket_ap"] == 0.25
        assert metrics["selection_unstable"] is True
        assert metrics["validation_images"] == 1
        assert "0.125" not in text
        assert "0.75" not in text
    finally:
        _cancel_active(workspace)


def test_failed_mps_does_not_restart_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace, dataset_id, _dataset_dir = _ready_workspace(tmp_path)
    monkeypatch.setattr(jobs, "_spawn", _dummy_spawn)
    attempts: list[str] = []
    device_calls = {"n": 0}

    def choose_mps() -> str:
        device_calls["n"] += 1
        return "mps"

    def trainer(**kwargs: object) -> None:
        attempts.append(str(kwargs["device"]))
        assert type(kwargs["batch_size"]) is int
        assert kwargs["batch_size"] == 1
        assert kwargs["run_test"] is False
        raise RuntimeError("mps backend failed: simulated")

    monkeypatch.setattr(train_mod, "choose_train_device", choose_mps)
    first = train_mod.enqueue_train(workspace, "pilot", dataset_id, "smoke")
    assert train_mod.run_train_job(workspace, first.job_id, trainer=trainer) == 1
    assert attempts == ["mps"]
    assert device_calls["n"] == 1
    failed = jobs.get_job(workspace, "pilot", first.job_id)
    assert failed.status == JobStatus.failed
    assert failed.error_code is not None
    assert "RuntimeError" in failed.error_code
    assert "mps backend failed: simulated" in failed.error_code
    assert list(workspace.rglob("*.pth")) == []

    cpu_seen: list[dict[str, object]] = []

    def choose_cpu() -> str:
        return "cpu"

    def cpu_trainer(**kwargs: object) -> dict[str, float]:
        cpu_seen.append(dict(kwargs))
        Path(str(kwargs["output_dir"])).joinpath("weights.pth").write_bytes(b"ok")
        return {"val/mAP_50_95": 0.5, "val/AP/bucket": 0.25, "test/mAP_50_95": 0.125}

    monkeypatch.setattr(train_mod, "choose_train_device", choose_cpu)
    second = train_mod.enqueue_train(workspace, "pilot", dataset_id, "smoke")
    assert train_mod.run_train_job(workspace, second.job_id, trainer=cpu_trainer) == 0
    assert len(cpu_seen) == 1
    assert cpu_seen[0]["device"] == "cpu"
    assert type(cpu_seen[0]["batch_size"]) is int
    assert cpu_seen[0]["batch_size"] == resolve_train_profile("smoke").train_kwargs["batch_size"]
    assert cpu_seen[0]["augmentation_train"] == MILD_PRESET
    metrics_path = jobs.run_directory(workspace, "pilot", second.job_id) / "validation_metrics.json"
    text = metrics_path.read_text(encoding="utf-8")
    metrics = json.loads(text)
    assert metrics["box_map_50_95"] == 0.5
    assert metrics["bucket_ap"] == 0.25
    assert "0.125" not in text
    assert attempts == ["mps"]


def test_validation_metrics_do_not_borrow_test_and_flag_small_sets() -> None:
    reported = {
        "val/mAP_50_95": 0.5,
        "test/mAP_50_95": 0.125,
        "val/AP/bucket": 0.25,
        "test/AP/bucket": 0.75,
    }
    small = build_validation_metrics(reported, validation_images=STABLE_VALIDATION_IMAGES - 1)
    assert small["selection_unstable"] is True
    assert small["box_map_50_95"] == 0.5
    assert small["bucket_ap"] == 0.25
    stable = build_validation_metrics(reported, validation_images=STABLE_VALIDATION_IMAGES)
    assert stable["selection_unstable"] is False
    assert stable["box_map_50_95"] == 0.5
    test_only = build_validation_metrics(
        {"test/mAP_50_95": 0.125, "test/AP/bucket": 0.75},
        validation_images=STABLE_VALIDATION_IMAGES,
    )
    assert test_only["box_map_50_95"] is None
    assert test_only["bucket_ap"] is None
    assert "0.125" not in json.dumps(test_only)
    rows = build_validation_metrics(
        {
            "val/mAP_50_95": 0.5,
            "per_class": [
                {"name": "bucket", "ap": 0.75, "split": "test"},
                {"name": "bucket", "ap": 0.25, "split": "validation"},
            ],
        },
        validation_images=STABLE_VALIDATION_IMAGES,
    )
    assert rows["bucket_ap"] == 0.25
    assert rows["box_map_50_95"] == 0.5


def test_runner_calls_train_api_and_does_not_apply_or_fall_back() -> None:
    source = Path(train_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    train_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "train"
    ]
    apply_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "apply")
            or (isinstance(node.func, ast.Name) and node.func.id == "apply")
        )
    ]
    assert len(train_calls) == 1
    assert apply_calls == []
    assert "allow_cpu_fallback" not in source
    assert "allow-cpu-fallback" not in source
    assert "shell=True" not in source
    command = inspect.getsource(train_command)
    assert "RFDETRNano" not in command
    assert "RFDETRSmall" not in command
    assert "allow_cpu_fallback" not in command
    assert "allow-cpu-fallback" not in command
    assert "streamlit" not in source
