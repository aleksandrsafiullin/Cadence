"""Train-frame suggestions. A fake detector, never RF-DETR and never a download."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import signal
import sqlite3
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from excavation_lab import jobs, storage
from excavation_lab.datasets.coco_import import load_imported_frames
from excavation_lab.datasets.splits import require_frozen
from excavation_lab.errors import LabError
from excavation_lab.models.adapter import (
    CheckpointOrigin,
    FrameDetections,
    ModelMetadata,
)
from excavation_lab.schemas import SplitName
from excavation_lab.storage import connect, project_sqlite
from excavation_lab.ui import analysis_page, training_page
from excavation_lab.ui.copy import catalog
from excavation_lab.ui.data_page import ActionNote
from excavation_lab.ui.locale import write_locale_file
from excavation_lab.web import label_routes
from excavation_lab.web.app import create_app

_PROJECT = "pilot"
_VIDEO = "vid"
_MODEL = "pit-model"


def _helpers():
    path = Path(__file__).with_name("test_label_page.py")
    spec = importlib.util.spec_from_file_location("label_page_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("label page helpers are missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _button(html: str, element_id: str) -> str:
    match = re.search(rf'<button\b[^>]*id="{element_id}"[^>]*>', html)
    assert match is not None, element_id
    return match.group(0)


def _red_jpeg(width: int, height: int) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 2] = 220
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return encoded.tobytes()


def _detections(frame: np.ndarray, rows: tuple[tuple[object, ...], ...]) -> FrameDetections:
    count = len(rows)
    xyxy = (
        np.zeros((0, 4), dtype=np.float64)
        if count == 0
        else np.array([[row[2], row[3], row[4], row[5]] for row in rows], dtype=np.float64)
    )
    return FrameDetections(
        timestamp_s=0.0,
        frame_width=int(frame.shape[1]),
        frame_height=int(frame.shape[0]),
        origin=CheckpointOrigin.custom,
        xyxy=xyxy,
        class_id=np.zeros(count, dtype=np.int64),
        class_name=tuple(str(row[0]) for row in rows),
        score=np.array([float(row[1]) for row in rows], dtype=np.float64),
    )


class _FakeDetector:
    def __init__(
        self,
        rows: tuple[tuple[object, ...], ...],
        *,
        order: list[str] | None = None,
        entered: threading.Event | None = None,
        resume: threading.Event | None = None,
    ) -> None:
        self.rows = rows
        self.closed = False
        self.frames: list[tuple[np.ndarray, float]] = []
        self.order = order
        self.entered = entered
        self.resume = resume

    def predict(self, frame_rgb: np.ndarray, timestamp_s: float) -> FrameDetections:
        self.frames.append((frame_rgb, float(timestamp_s)))
        if self.order is not None:
            self.order.append("predict_enter")
        if self.entered is not None:
            self.entered.set()
        if self.resume is not None:
            assert self.resume.wait(3)
        if self.order is not None:
            self.order.append("predict_exit")
        return _detections(frame_rgb, self.rows)

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            origin=CheckpointOrigin.custom,
            class_names=("bucket", "haul_truck"),
            boxes_are_original=True,
            variant="fake",
            resolution=384,
        )

    def close(self) -> None:
        if self.order is not None:
            self.order.append("closed")
        self.closed = True


def _plant_named(workspace: Path, model_id: str, body: dict[str, object] | None) -> None:
    bundle = workspace / "weights" / model_id
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "weights.pth").write_bytes(b"not-loaded")
    if body is not None:
        (bundle / "model_config.json").write_text(
            json.dumps(body),
            encoding="utf-8",
        )
    with connect(project_sqlite(workspace, _PROJECT)) as connection:
        connection.execute(
            """
            INSERT INTO model_versions (
                model_id, project_id, dataset_id, variant, path, created_at
            ) VALUES (?, ?, NULL, 'nano', ?, ?)
            """,
            (model_id, _PROJECT, str(bundle.resolve()), "2026-01-02T00:00:00+00:00"),
        )


def _insert_dataset(workspace: Path) -> None:
    folder = workspace / "snap"
    folder.mkdir()
    with connect(project_sqlite(workspace, _PROJECT)) as connection:
        connection.execute(
            """
            INSERT INTO dataset_versions (
                dataset_id, project_id, content_hash, path, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("snap", _PROJECT, "hash", str(folder), "2026-01-01T00:00:00+00:00"),
        )


def _plant_finished_run(workspace: Path, run_id: str) -> None:
    run = storage.project_directory(workspace, _PROJECT) / "runs" / run_id
    run.mkdir(parents=True)
    (run / "detections.parquet").write_bytes(b"not-read")
    (run / "frame_status.parquet").write_bytes(b"not-read")
    (run / "run.json").write_text(
        json.dumps({"video_id": _VIDEO}),
        encoding="utf-8",
    )


def _job_rows(workspace: Path) -> int:
    path = jobs.lab_database(workspace)
    if not path.is_file():
        return 0
    connection = sqlite3.connect(path)
    try:
        row = connection.execute("SELECT COUNT(*) FROM lab_jobs").fetchone()
    finally:
        connection.close()
    assert row is not None
    return int(row[0])


def _stop(pid: int | None) -> None:
    if pid is None:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        return


def test_validation_and_test_frames_are_refused(tmp_path: Path) -> None:
    book = catalog("ru")
    helpers = _helpers()
    workspace = helpers._lab(tmp_path)
    helpers._plant_bundle(workspace)
    frozen = require_frozen(workspace, _PROJECT, _VIDEO)
    payload = helpers._jpeg(32, 24, 40)
    helpers._plant_sample(
        workspace,
        frozen.content_hash,
        [
            (10, 10.0, payload, SplitName.train),
            (11, 12.0, payload, SplitName.train),
            (80, 80.0, payload, SplitName.validation),
            (200, 110.0, payload, SplitName.test),
        ],
    )
    fake = _FakeDetector((("bucket", 0.9, 1, 1, 8, 6),))
    app = create_app(workspace)
    app.state.suggestion_loader = lambda _path: fake
    with TestClient(app) as client:
        validation = client.post(
            "/label/suggest",
            params={"video": _VIDEO, "frame": "80"},
            json={"model_id": _MODEL},
        )
        held = client.post(
            "/label/suggest",
            params={"video": _VIDEO, "frame": "200"},
            json={"model_id": _MODEL},
        )
        page = client.get("/label", params={"video": _VIDEO, "frame": "80"})
        train_page = client.get("/label", params={"video": _VIDEO, "frame": "10"})
        write_locale_file(workspace, "en")
        english = client.get("/label", params={"video": _VIDEO, "frame": "200"})
    assert validation.status_code == 400
    assert validation.json()["code"] == "not_train_frame"
    assert validation.json()["error"] == book.label_not_train_frame
    assert held.json()["code"] == "not_train_frame"
    assert fake.frames == []
    assert fake.closed is False
    assert book.label_not_train_frame in page.text
    assert "disabled" in _button(page.text, "label-suggest")
    assert book.label_suggest_button in train_page.text
    assert "disabled" not in _button(train_page.text, "label-suggest")
    assert f'value="{_MODEL}"' in train_page.text
    assert catalog("en").label_not_train_frame in english.text
    assert catalog("en").label_suggest_button in english.text
    assert book.label_not_train_frame not in english.text
    assert load_imported_frames(workspace, _PROJECT) == []


def test_fake_boxes_stay_unstored_until_a_normal_save(tmp_path: Path) -> None:
    helpers = _helpers()
    workspace = helpers._lab(tmp_path)
    helpers._plant_bundle(workspace)
    frozen = require_frozen(workspace, _PROJECT, _VIDEO)
    red = _red_jpeg(32, 24)
    plain = helpers._jpeg(32, 24, 40)
    helpers._plant_sample(
        workspace,
        frozen.content_hash,
        [
            (10, 10.0, red, SplitName.train),
            (11, 12.0, plain, SplitName.train),
            (80, 80.0, plain, SplitName.validation),
        ],
    )
    rows = (
        ("bucket", 0.91, 1.0, 2.0, 8.0, 6.0),
        ("haul_truck", 0.5, 3.0, 3.0, 12.0, 10.0),
    )
    fake = _FakeDetector(rows)
    app = create_app(workspace)
    app.state.suggestion_loader = lambda _path: fake
    with TestClient(app) as client:
        suggested = client.post(
            "/label/suggest",
            params={"video": _VIDEO, "frame": "10"},
            json={"model_id": _MODEL},
        )
        assert load_imported_frames(workspace, _PROJECT) == []
        saved = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "10"},
            json={"review_status": "unreviewed", "boxes": []},
        )
    assert suggested.status_code == 200
    body = suggested.json()
    assert body["dropped"] == 0
    assert len(body["boxes"]) == 2
    assert [item["suggested"] for item in body["boxes"]] == [True, True]
    assert [item["class_name"] for item in body["boxes"]] == ["bucket", "haul_truck"]
    assert body["boxes"][0]["score"] == pytest.approx(0.91)
    assert body["boxes"][0]["x1"] == pytest.approx(1.0)
    frame, stamp = fake.frames[0]
    assert stamp == pytest.approx(10.0)
    assert float(frame[:, :, 0].mean()) > float(frame[:, :, 2].mean())
    assert saved.status_code == 200
    assert saved.json()["boxes"] == []
    stored = load_imported_frames(workspace, _PROJECT)
    assert len(stored) == 1
    assert len(stored[0].record.boxes) == 0
    assert app.state.suggestion_adapter is fake
    assert fake.closed is False


def test_low_scores_and_unknown_names_are_not_labels(tmp_path: Path) -> None:
    helpers = _helpers()
    workspace = helpers._lab(tmp_path)
    helpers._plant_bundle(workspace)
    fake = _FakeDetector(
        (
            ("bucket", 0.35, 1, 1, 6, 6),
            ("person", 0.99, 1, 1, 6, 6),
            ("haul_truck", 0.34, 1, 1, 6, 6),
            ("bucket", 0.8, 2, 2, 7, 7),
        )
    )
    app = create_app(workspace)
    app.state.suggestion_loader = lambda _path: fake
    with TestClient(app) as client:
        response = client.post(
            "/label/suggest",
            params={"video": _VIDEO, "frame": "10"},
            json={"model_id": _MODEL},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["dropped"] == 1
    assert [item["class_name"] for item in body["boxes"]] == ["bucket", "bucket"]
    assert all(item["suggested"] is True for item in body["boxes"])
    assert load_imported_frames(workspace, _PROJECT) == []


def test_synthetic_and_baseline_are_refused(tmp_path: Path) -> None:
    book = catalog("ru")
    helpers = _helpers()
    workspace = helpers._lab(tmp_path)
    helpers._plant_bundle(workspace)
    _plant_named(workspace, "wire-model", {"synthetic": True, "metric_role": "synthetic_wiring"})
    role_only = workspace / "weights" / "role-only"
    _plant_named(workspace, "role-only", {"metric_role": "synthetic_wiring"})
    assert (role_only / "weights.pth").is_file()
    fake = _FakeDetector((("bucket", 0.9, 1, 1, 4, 4),))
    app = create_app(workspace)
    app.state.suggestion_loader = lambda _path: (_ for _ in ()).throw(AssertionError("loaded"))
    with TestClient(app) as client:
        baseline = client.post(
            "/label/suggest",
            params={"video": _VIDEO, "frame": "10"},
            json={"model_id": "baseline_coco"},
        )
        synthetic = client.post(
            "/label/suggest",
            params={"video": _VIDEO, "frame": "10"},
            json={"model_id": "wire-model"},
        )
        role = client.post(
            "/label/suggest",
            params={"video": _VIDEO, "frame": "10"},
            json={"model_id": "role-only"},
        )
        page = client.get("/label", params={"video": _VIDEO, "frame": "10"})
    assert baseline.json()["code"] == "bundle_missing"
    assert synthetic.json()["code"] == "bundle_missing"
    assert synthetic.json()["error"] == book.label_bundle_missing
    assert role.json()["code"] == "bundle_missing"
    assert "wire-model" not in page.text
    assert "role-only" not in page.text
    assert f'value="{_MODEL}"' in page.text
    assert fake.closed is False


def test_running_sleep_job_closes_the_adapter(tmp_path: Path) -> None:
    helpers = _helpers()
    workspace = helpers._lab(tmp_path)
    helpers._plant_bundle(workspace)
    fake = _FakeDetector((("bucket", 0.9, 1, 1, 4, 4),))
    app = create_app(workspace)
    app.state.suggestion_loader = lambda _path: fake
    job = None
    try:
        with TestClient(app) as client:
            warmed = client.post(
                "/label/suggest",
                params={"video": _VIDEO, "frame": "10"},
                json={"model_id": _MODEL},
            )
            assert warmed.status_code == 200
            assert fake.closed is False
            assert app.state.suggestion_adapter is fake
            job = jobs.start_job(
                workspace,
                _PROJECT,
                kind="analyze",
                worker="sleep",
                video_duration_s=100,
                step_s=1,
                sleep_s=30,
            )
            refused = client.post(
                "/label/suggest",
                params={"video": _VIDEO, "frame": "10"},
                json={"model_id": _MODEL},
            )
        assert refused.status_code == 400
        assert refused.json()["code"] == "job_running"
        assert fake.closed is True
        assert app.state.suggestion_adapter is None
        assert load_imported_frames(workspace, _PROJECT) == []
    finally:
        _stop(None if job is None else job.pid)


def test_queued_and_cancel_requested_also_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helpers = _helpers()
    workspace = helpers._lab(tmp_path)
    helpers._plant_bundle(workspace)
    for status in ("queued", "cancel_requested"):
        fake = _FakeDetector((("bucket", 0.9, 1, 1, 4, 4),))
        app = create_app(workspace)
        app.state.suggestion_adapter = fake
        monkeypatch.setattr(
            jobs,
            "reopen",
            lambda _workspace, status=status: [type("Job", (), {"status": status})()],
        )
        with TestClient(app) as client:
            refused = client.post(
                "/label/suggest",
                params={"video": _VIDEO, "frame": "10"},
                json={"model_id": _MODEL},
            )
        assert refused.json()["code"] == "job_running"
        assert fake.closed is True


def test_training_start_closes_the_adapter_before_the_job_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXCAVATION_LAB_FAKE_TRAINER", "fail")
    helpers = _helpers()
    workspace = helpers._lab(tmp_path)
    helpers._plant_bundle(workspace)
    _insert_dataset(workspace)
    fake = _FakeDetector((("bucket", 0.9, 1, 1, 4, 4),))
    seen: dict[str, object] = {}

    def spy(*_args: object, **_kwargs: object) -> None:
        seen["closed"] = fake.closed
        seen["rows"] = _job_rows(workspace)
        raise LabError("invalid_job", "stop")

    monkeypatch.setattr(training_page, "enqueue_train", spy)
    app = create_app(workspace)
    app.state.suggestion_adapter = fake
    with TestClient(app) as client:
        response = client.post(
            "/training/start",
            data={"profile": "smoke", "dataset_id": "snap"},
            follow_redirects=False,
        )
    assert response.status_code == 303
    assert seen == {"closed": True, "rows": 0}
    assert fake.closed is True
    assert _job_rows(workspace) == 0


def test_analysis_start_and_export_close_the_adapter_before_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXCAVATION_LAB_FAKE_TRAINER", "fail")
    helpers = _helpers()
    workspace = helpers._lab(tmp_path)
    helpers._plant_bundle(workspace)
    fake = _FakeDetector((("bucket", 0.9, 1, 1, 4, 4),))
    seen: dict[str, object] = {}

    def spy(*_args: object, **_kwargs: object) -> None:
        seen["closed"] = fake.closed
        seen["rows"] = _job_rows(workspace)
        raise LabError("invalid_job", "stop")

    monkeypatch.setattr(analysis_page, "enqueue_analyze", spy)
    export_calls = {"n": 0}

    def export_spy(*_args: object, **_kwargs: object) -> None:
        export_calls["n"] += 1
        assert fake.closed is True
        assert _job_rows(workspace) == 0
        raise LabError("invalid_job", "stop")

    monkeypatch.setattr(analysis_page, "enqueue_video_export", export_spy)
    app = create_app(workspace)
    app.state.suggestion_adapter = fake
    with TestClient(app) as client:
        started = client.post(
            "/analysis/start",
            data={
                "video_id": _VIDEO,
                "mode": "custom",
                "bundle": _MODEL,
                "analysis_fps": "5",
                "high_confidence": "0.35",
                "activation": "0.45",
                "lost_buffer_s": "2",
                "min_observations": "2",
                "display_threshold": "0.35",
                "start_s": "0",
                "end_s": "1",
            },
            follow_redirects=False,
        )
        assert started.status_code == 303
        assert seen == {"closed": True, "rows": 0}
        fake.closed = False
        app.state.suggestion_adapter = fake
        _plant_finished_run(workspace, "run-a")
        exported = client.post(
            "/analysis/export",
            data={"video_id": _VIDEO, "mode": "custom", "run_id": "run-a"},
            follow_redirects=False,
        )
    assert exported.status_code == 303
    assert export_calls["n"] == 1
    assert fake.closed is True
    assert _job_rows(workspace) == 0


def test_release_waits_until_predict_finishes(tmp_path: Path) -> None:
    helpers = _helpers()
    workspace = helpers._lab(tmp_path)
    helpers._plant_bundle(workspace)
    order: list[str] = []
    entered = threading.Event()
    resume = threading.Event()
    fake = _FakeDetector(
        (
            ("bucket", 0.9, 1, 1, 8, 6),
            ("haul_truck", 0.6, 2, 2, 9, 7),
        ),
        order=order,
        entered=entered,
        resume=resume,
    )
    app = create_app(workspace)
    holder: dict[str, object] = {}

    def run() -> None:
        holder["pair"] = label_routes.suggest_holding_lock(
            app,
            workspace,
            _PROJECT,
            _VIDEO,
            10,
            _MODEL,
            catalog("ru"),
            loader=lambda _path: fake,
        )

    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(3)
    released = threading.Event()

    def drop() -> None:
        order.append("release_enter")
        label_routes.release_suggestion_adapter(app)
        order.append("release_exit")
        released.set()

    releaser = threading.Thread(target=drop)
    releaser.start()
    deadline = time.monotonic() + 1
    while "release_enter" not in order and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "release_enter" in order
    time.sleep(0.05)
    assert not released.is_set()
    resume.set()
    worker.join(3)
    releaser.join(3)
    assert not worker.is_alive()
    assert not releaser.is_alive()
    assert order.index("predict_exit") < order.index("release_exit")
    pair = holder["pair"]
    assert isinstance(pair, tuple)
    result, dropped = pair
    assert not isinstance(result, ActionNote)
    assert dropped == 0
    assert len(result) == 2
    assert all(item["suggested"] is True for item in result)
    assert fake.closed is True
