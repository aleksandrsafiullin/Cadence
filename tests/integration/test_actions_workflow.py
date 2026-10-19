"""synthetic; mechanics only, not accuracy evidence"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from excavation_lab import jobs, storage
from excavation_lab.actions.catalog import seed_actions
from excavation_lab.actions.dataset import ActionWindows, build_windows
from excavation_lab.actions.evaluate import evaluate_actions
from excavation_lab.actions.infer import enqueue_action_infer, run_action_infer_job
from excavation_lab.actions.labels import add_interval
from excavation_lab.actions.model import Standardizer
from excavation_lab.actions.train import (
    ActionTrainConfig,
    enqueue_action_train,
    run_action_train_job,
)
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.splits import freeze_split, write_blocks
from excavation_lab.errors import LabError
from excavation_lab.jobs import get_job, run_directory
from excavation_lab.schemas import JobStatus
from excavation_lab.ui.copy import action_seed_titles

_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SPEC = importlib.util.spec_from_file_location(
    "excavation_lab_synthetic_actions",
    _SYNTHETIC_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_run = _SYNTHETIC.write_synthetic_run

_PROJECT = "pilot"
_VIDEO = "vid"
_DETECTOR = "det_custom"
_FPS = 5.0
_TEST = (80.0, 100.0)
_BLOCKS = (
    RoleBlock(start_s=0, end_s=60, role="train"),
    RoleBlock(start_s=60, end_s=80, role="validation"),
    RoleBlock(start_s=80, end_s=100, role="test"),
)
_TRAIN_INTERVALS = (
    (1.0, 10.0, "digging"),
    (12.0, 21.0, "swing_loaded"),
    (23.0, 32.0, "dumping"),
    (34.0, 43.0, "swing_empty"),
    (45.0, 54.0, "idle"),
)
_RUNS = (
    ("run_a", 0.0, 60.0),
    ("run_b", 60.0, 80.0),
    ("run_c", 80.0, 100.0),
)


class _DummyProcess:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.pid = os.getpid()

    def poll(self) -> None:
        return None


def _dummy_spawn(*_args: object, **_kwargs: object) -> _DummyProcess:
    return _DummyProcess()


def test_three_runs_train_infer_and_keep_test_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(jobs, "_spawn", _dummy_spawn)
    workspace = _project(tmp_path)
    run_a, run_b, run_c = _runs(workspace)
    seen: list[pd.DataFrame] = []
    built: list[ActionWindows] = []

    def spy(features_df: pd.DataFrame, *args: object, **kwargs: object) -> ActionWindows:
        seen.append(features_df.copy())
        windows = build_windows(features_df, *args, **kwargs)  # type: ignore[arg-type]
        built.append(windows)
        return windows

    monkeypatch.setattr("excavation_lab.actions.dataset.build_windows", spy)
    trained = enqueue_action_train(
        workspace,
        _PROJECT,
        [run_a, run_b],
        ActionTrainConfig(epochs=2, patience=2),
    )
    assert run_action_train_job(workspace, trained.job_id) == 0
    assert get_job(workspace, _PROJECT, trained.job_id).status is JobStatus.completed
    models = storage.list_action_models(workspace, _PROJECT)
    assert len(models) == 1
    assert models[0]["source_run_ids"] == [run_a, run_b]
    bundle = Path(models[0]["path"])
    document = json.loads((bundle / "action_bundle.json").read_text(encoding="utf-8"))
    assert document["test_read"] is False
    kept = json.loads((bundle / "kept_time_ranges.json").read_text(encoding="utf-8"))
    job_kept = json.loads(
        (run_directory(workspace, _PROJECT, trained.job_id) / "kept_time_ranges.json").read_text(
            encoding="utf-8"
        )
    )
    assert job_kept == kept
    assert kept
    assert {row["role"] for row in kept} <= {"train", "validation"}
    assert all(not _hits(_TEST, row["start_s"], row["end_s"]) for row in kept)
    assert seen
    for frame in seen:
        times = frame["source_time_s"].to_numpy()
        assert not bool(((times >= _TEST[0]) & (times < _TEST[1])).any())
        for row in kept:
            inside = frame[
                (frame["run_id"] == row["run_id"])
                & (frame["source_time_s"] >= row["start_s"])
                & (frame["source_time_s"] < row["end_s"])
            ]
            assert not inside.empty
            assert bool(inside.iloc[0]["run_start"])
    train_windows, val_windows = _roles(built)
    _assert_windows_stay_inside(seen[0], train_windows, 0.0, 60.0)
    _assert_windows_stay_inside(seen[0], val_windows, 60.0, 80.0)
    assert train_windows.dropped["crosses_run"] > 0
    assert val_windows.dropped["crosses_run"] > 0
    expected = Standardizer.fit(train_windows.x)
    stored = json.loads((bundle / "standardizer.json").read_text(encoding="utf-8"))
    assert stored["mean"] == pytest.approx(expected.to_json()["mean"])
    assert stored["std"] == pytest.approx(expected.to_json()["std"])
    leaked = Standardizer.fit(np.concatenate([train_windows.x, val_windows.x], axis=0))
    assert not np.allclose(leaked.mean, expected.mean)

    infer_b = enqueue_action_infer(workspace, _PROJECT, run_b, models[0]["action_model_id"])
    assert run_action_infer_job(workspace, infer_b.job_id) == 0
    infer_c = enqueue_action_infer(workspace, _PROJECT, run_c, models[0]["action_model_id"])
    assert run_action_infer_job(workspace, infer_c.job_id) == 0
    for record in (infer_b, infer_c):
        run_dir = run_directory(workspace, _PROJECT, record.job_id)
        assert (run_dir / "actions.parquet").is_file()
        assert (run_dir / "action_intervals.csv").is_file()
        assert get_job(workspace, _PROJECT, record.job_id).status is JobStatus.completed

    validation = evaluate_actions(workspace, _PROJECT, infer_b.job_id)
    assert validation.evaluated_frames > 0
    assert validation.per_class["digging"].support_frames > 0
    validation_name = f"evaluation_validation_{infer_b.job_id}.json"
    assert (bundle / validation_name).is_file()
    with pytest.raises(LabError) as without_final:
        evaluate_actions(workspace, _PROJECT, infer_c.job_id, split="test")
    assert without_final.value.code == "not_final"
    assert not (bundle / "evaluation_test.json").exists()
    with pytest.raises(LabError) as wrong_run:
        evaluate_actions(workspace, _PROJECT, infer_b.job_id, split="test", final=True)
    assert wrong_run.value.code == "no_frames_for_split"
    assert not (bundle / "evaluation_test.json").exists()
    final = evaluate_actions(workspace, _PROJECT, infer_c.job_id, split="test", final=True)
    destination = bundle / "evaluation_test.json"
    body = json.loads(destination.read_text(encoding="utf-8"))
    assert final.evaluated_frames > 0
    assert body["validation_evaluation_file"] == validation_name
    assert body["final"] is True
    before = destination.read_bytes()
    with pytest.raises(LabError) as second:
        evaluate_actions(workspace, _PROJECT, infer_c.job_id, split="test", final=True)
    assert second.value.code == "final_exists"
    assert destination.read_bytes() == before


def _project(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project(_PROJECT, workspace, catalog_id="excavation")
    storage.insert_video(workspace, _PROJECT, _VIDEO, name="clip")
    freeze_split(
        workspace,
        _PROJECT,
        write_blocks(workspace, _PROJECT, _VIDEO, list(_BLOCKS)),
    )
    seed_actions(workspace, _PROJECT, action_seed_titles["ru"])
    for start_s, end_s, name in _TRAIN_INTERVALS:
        add_interval(workspace, _PROJECT, _VIDEO, start_s, end_s, name)
    add_interval(workspace, _PROJECT, _VIDEO, 62.0, 74.0, "digging")
    add_interval(workspace, _PROJECT, _VIDEO, 82.0, 94.0, "digging")
    return workspace


def _runs(workspace: Path) -> tuple[str, str, str]:
    ids: list[str] = []
    for run_id, start_s, end_s in _RUNS:
        record = jobs.start_job(
            workspace,
            _PROJECT,
            kind="analyze",
            worker="sleep",
            video_duration_s=10.0,
            step_s=1.0,
        )
        run_dir = run_directory(workspace, _PROJECT, record.job_id)
        write_synthetic_run(
            run_dir,
            run_id=run_id,
            video_id=_VIDEO,
            start_s=start_s,
            end_s=end_s,
            fps=_FPS,
        )
        jobs.mark_completed(workspace, _PROJECT, record.job_id)
        storage.insert_analysis_run(
            workspace,
            _PROJECT,
            run_id=run_id,
            job_id=record.job_id,
            video_id=_VIDEO,
            view_id=None,
            model_id=_DETECTOR,
            model_origin="custom",
            analysis_fps=_FPS,
            source_fps=None,
            export_fps=None,
            start_s=start_s,
            end_s=end_s,
            decode_ms=None,
            encode_ms=None,
        )
        ids.append(run_id)
    assert len(ids) == 3
    return ids[0], ids[1], ids[2]


def _hits(span: tuple[float, float], start_s: float, end_s: float) -> bool:
    return start_s < span[1] and end_s > span[0]


def _roles(built: list[ActionWindows]) -> tuple[ActionWindows, ActionWindows]:
    by_role = {item.role: item for item in built}
    assert set(by_role) == {"train", "validation"}
    return by_role["train"], by_role["validation"]


def _assert_windows_stay_inside(
    frame: pd.DataFrame,
    windows: ActionWindows,
    start_s: float,
    end_s: float,
) -> None:
    times = frame["source_time_s"].to_numpy()
    run_ids = frame["run_id"].to_numpy()
    centers = windows.center_time_s
    width = windows.window_frames
    half = width // 2
    assert centers.size > 0
    assert bool(((centers >= _TEST[0]) & (centers < _TEST[1])).any()) is False
    assert bool(((centers >= start_s) & (centers < end_s)).all())
    for center in centers:
        indexes = np.flatnonzero(np.isclose(times, float(center), atol=1e-6))
        assert indexes.size == 1
        start = int(indexes[0]) - half
        chunk = times[start : start + width]
        owners = run_ids[start : start + width]
        assert chunk.size == width
        assert float(chunk.min()) >= start_s
        assert float(chunk.max()) < end_s
        assert len(set(owners.tolist())) == 1
        assert not bool(((chunk >= _TEST[0]) & (chunk < _TEST[1])).any())
        assert not (float(chunk.min()) < 60.0 <= float(chunk.max()))
        assert not (float(chunk.min()) < 80.0 <= float(chunk.max()))
