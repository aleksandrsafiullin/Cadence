"""Action evaluation on synthetic parquet. Metrics are computed, not invented."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from excavation_lab import storage
from excavation_lab.actions.catalog import seed_actions
from excavation_lab.actions.evaluate import evaluate_actions, format_action_evaluation
from excavation_lab.actions.labels import add_interval
from excavation_lab.cli import app
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.splits import freeze_split, write_blocks
from excavation_lab.errors import LabError
from excavation_lab.ui.copy import catalog

_RUNNER = CliRunner()
_PROJECT = "mvp"
_VIDEO = "vid"
_MODEL = "act1"
_TITLES = {
    "digging": "Dig",
    "swing_loaded": "Swing loaded",
    "dumping": "Dump",
    "swing_empty": "Swing empty",
    "idle": "Idle",
}
_BLOCKS = [
    RoleBlock(start_s=0, end_s=30, role="validation"),
    RoleBlock(start_s=40, end_s=70, role="test"),
]
_VAL_TIMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 33]
_VAL_PREDS = [
    "digging",
    "digging",
    "swing_loaded",
    "unknown",
    "swing_loaded",
    "swing_loaded",
    "digging",
    "unknown",
    "digging",
    "swing_loaded",
    "digging",
    "digging",
    "digging",
]


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "workspace"
    list_catalogs(root)
    storage.create_project(_PROJECT, root, catalog_id="excavation")
    storage.insert_video(root, _PROJECT, _VIDEO, name="clip")
    seed_actions(root, _PROJECT, _TITLES)
    proposal = write_blocks(root, _PROJECT, _VIDEO, _BLOCKS)
    freeze_split(root, _PROJECT, proposal)
    add_interval(root, _PROJECT, _VIDEO, 0, 4, "digging")
    add_interval(root, _PROJECT, _VIDEO, 4, 8, "swing_loaded")
    add_interval(root, _PROJECT, _VIDEO, 10, 12, "digging")
    add_interval(root, _PROJECT, _VIDEO, 40, 48, "digging")
    with storage.connect(storage.project_sqlite(root, _PROJECT)) as connection:
        connection.execute(
            """
            INSERT INTO action_labels (
                label_id, project_id, video_id, start_s, end_s,
                action_name, annotator, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "gap1",
                _PROJECT,
                _VIDEO,
                32.0,
                34.0,
                "digging",
                "operator",
                "2026-01-01T00:00:00+00:00",
            ),
        )
    bundle = tmp_path / "act1"
    bundle.mkdir()
    (bundle / "action_bundle.json").write_text(
        json.dumps({"action_model_id": _MODEL, "class_order": ["digging", "swing_loaded", "idle"]})
        + "\n",
        encoding="utf-8",
    )
    storage.insert_action_model(
        root,
        _PROJECT,
        action_model_id=_MODEL,
        detector_model_id="det",
        source_run_ids=["run_val"],
        path=bundle,
    )
    return root, bundle


def _inference(
    root: Path,
    directory: Path,
    *,
    job_id: str,
    run_id: str,
    start_s: float,
    end_s: float,
    times: list[float],
    preds: list[str],
    min_confidence: float = 0.5,
    min_duration_s: float = 1.0,
    action_model_id: str = _MODEL,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "source_time_s": times,
            "action": preds,
            "action_name": preds,
            "unknown": [item == "unknown" for item in preds],
        }
    )
    frame.to_parquet(directory / "actions.parquet", index=False)
    payload = {
        "action_model_id": action_model_id,
        "detector_model_id": "det",
        "min_confidence": min_confidence,
        "min_duration_s": min_duration_s,
        "source_run_id": run_id,
        "unknown_share": None,
    }
    (directory / "action_run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    analysis_job_id = f"analysis_{job_id}"
    with storage.connect(storage.project_sqlite(root, _PROJECT)) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                job_id, project_id, status, pid, partial,
                analyzed_end_s, error_code, created_at
            ) VALUES (?, ?, 'completed', NULL, 0, NULL, NULL, ?)
            """,
            (analysis_job_id, _PROJECT, "2026-01-01T00:00:00+00:00"),
        )
    storage.insert_analysis_run(
        root,
        _PROJECT,
        run_id=run_id,
        job_id=analysis_job_id,
        video_id=_VIDEO,
        view_id=None,
        model_id="det",
        model_origin="custom",
        analysis_fps=1.0,
        source_fps=None,
        export_fps=None,
        start_s=start_s,
        end_s=end_s,
        decode_ms=None,
        encode_ms=None,
    )
    storage.insert_action_inference(
        root,
        _PROJECT,
        job_id=job_id,
        source_run_id=run_id,
        action_model_id=action_model_id,
        path=directory,
    )


def _validation_case(tmp_path: Path) -> tuple[Path, Path]:
    root, bundle = _workspace(tmp_path)
    _inference(
        root,
        tmp_path / "infer_val",
        job_id="infer_val",
        run_id="run_val",
        start_s=0.0,
        end_s=36.0,
        times=_VAL_TIMES,
        preds=_VAL_PREDS,
    )
    return root, bundle


def test_validation_report_excludes_unlabelled_and_nulls_zero_support(tmp_path: Path) -> None:
    root, bundle = _validation_case(tmp_path)
    report = evaluate_actions(root, _PROJECT, "infer_val")
    digging = report.per_class["digging"]
    swing = report.per_class["swing_loaded"]
    idle = report.per_class["idle"]
    assert report.evaluated_frames == 10
    assert report.unlabelled_frames == 3
    assert report.relabelled_outside_role == 1
    assert report.unknown_rate == pytest.approx(0.2)
    assert report.accuracy == pytest.approx(0.6)
    assert digging.support_frames == 6
    assert digging.support_intervals == 2
    assert digging.precision == pytest.approx(0.8)
    assert digging.recall == pytest.approx(2 / 3)
    assert digging.f1 == pytest.approx(8 / 11)
    assert swing.support_frames == 4
    assert swing.support_intervals == 1
    assert swing.precision == pytest.approx(2 / 3)
    assert swing.recall == pytest.approx(0.5)
    assert swing.f1 == pytest.approx(4 / 7)
    assert idle.precision is None
    assert idle.recall is None
    assert idle.f1 is None
    assert idle.support_frames == 0
    assert idle.support_intervals == 0
    assert report.classes_scored == ("digging", "swing_loaded")
    assert report.macro_f1 == pytest.approx(50 / 77)
    assert report.selection_unstable is True
    assert report.confusion["columns"] == ["digging", "swing_loaded", "idle", "unknown"]
    assert report.confusion["counts"] == [
        [4, 1, 0, 1],
        [1, 2, 0, 1],
        [0, 0, 0, 0],
    ]
    written = json.loads((bundle / "evaluation_validation_infer_val.json").read_text(encoding="utf-8"))
    assert written["per_class"]["idle"]["precision"] is None
    assert written["per_class"]["idle"]["f1"] is None
    text = format_action_evaluation(report, catalog("ru"))
    assert "idle: precision не рассчитано, recall не рассчитано, f1 не рассчитано" in text
    assert "Мало кадров для выбора: да" in text
    english = format_action_evaluation(report, catalog("en"))
    assert "idle: precision not calculated, recall not calculated, f1 not calculated" in english


def test_test_without_final_is_refused(tmp_path: Path) -> None:
    root, bundle = _validation_case(tmp_path)
    _inference(
        root,
        tmp_path / "infer_test",
        job_id="infer_test",
        run_id="run_test",
        start_s=40.0,
        end_s=70.0,
        times=[40, 41, 42, 43, 44, 45, 46, 47],
        preds=["digging"] * 8,
    )
    with pytest.raises(LabError) as refused:
        evaluate_actions(root, _PROJECT, "infer_test", split="test")
    assert refused.value.code == "not_final"
    assert not (bundle / "evaluation_test.json").exists()


def test_final_without_validation_is_refused(tmp_path: Path) -> None:
    root, bundle = _workspace(tmp_path)
    _inference(
        root,
        tmp_path / "infer_test",
        job_id="infer_test",
        run_id="run_test",
        start_s=40.0,
        end_s=70.0,
        times=[40, 41, 42, 43],
        preds=["digging"] * 4,
    )
    with pytest.raises(LabError) as refused:
        evaluate_actions(root, _PROJECT, "infer_test", split="test", final=True)
    assert refused.value.code == "final_thresholds_mismatch"
    assert not (bundle / "evaluation_test.json").exists()


def test_final_refuses_a_different_min_confidence(tmp_path: Path) -> None:
    root, bundle = _validation_case(tmp_path)
    evaluate_actions(root, _PROJECT, "infer_val")
    _inference(
        root,
        tmp_path / "infer_hi",
        job_id="infer_hi",
        run_id="run_hi",
        start_s=40.0,
        end_s=70.0,
        times=[40, 41, 42, 43],
        preds=["digging"] * 4,
        min_confidence=0.8,
    )
    with pytest.raises(LabError) as refused:
        evaluate_actions(root, _PROJECT, "infer_hi", split="test", final=True)
    assert refused.value.code == "final_thresholds_mismatch"
    assert "0.5" in refused.value.message
    assert "0.8" in refused.value.message
    assert not (bundle / "evaluation_test.json").exists()


def test_no_frames_for_split_writes_nothing(tmp_path: Path) -> None:
    root, bundle = _validation_case(tmp_path)
    evaluate_actions(root, _PROJECT, "infer_val")
    with pytest.raises(LabError) as test_split:
        evaluate_actions(root, _PROJECT, "infer_val", split="test", final=True)
    assert test_split.value.code == "no_frames_for_split"
    assert "test" in test_split.value.message
    assert "[0, 36)" in test_split.value.message
    assert not (bundle / "evaluation_test.json").exists()
    _inference(
        root,
        tmp_path / "infer_only_test",
        job_id="infer_only_test",
        run_id="run_only_test",
        start_s=40.0,
        end_s=70.0,
        times=[40, 41, 42, 43],
        preds=["digging"] * 4,
    )
    with pytest.raises(LabError) as validation_split:
        evaluate_actions(root, _PROJECT, "infer_only_test", split="validation")
    assert validation_split.value.code == "no_frames_for_split"
    assert "validation" in validation_split.value.message
    assert "[40, 70)" in validation_split.value.message
    assert not (bundle / "evaluation_validation_infer_only_test.json").exists()


def test_final_succeeds_once_when_the_run_covers_test(tmp_path: Path) -> None:
    root, bundle = _validation_case(tmp_path)
    evaluate_actions(root, _PROJECT, "infer_val")
    _inference(
        root,
        tmp_path / "infer_test",
        job_id="infer_test",
        run_id="run_test",
        start_s=40.0,
        end_s=70.0,
        times=[40, 41, 42, 43, 44, 45, 46, 47],
        preds=["digging"] * 8,
    )
    report = evaluate_actions(root, _PROJECT, "infer_test", split="test", final=True)
    path = bundle / "evaluation_test.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert report.evaluated_frames == 8
    assert report.accuracy == pytest.approx(1.0)
    assert report.classes_scored == ("digging",)
    assert report.per_class["swing_loaded"].f1 is None
    assert body["action_model_id"] == _MODEL
    assert body["min_confidence"] == 0.5
    assert body["min_duration_s"] == 1.0
    assert body["validation_evaluation_file"] == "evaluation_validation_infer_val.json"
    assert body["infer_job_id"] == "infer_test"
    assert body["final"] is True
    before = path.read_bytes()
    with pytest.raises(LabError) as second:
        evaluate_actions(root, _PROJECT, "infer_test", split="test", final=True)
    assert second.value.code == "final_exists"
    assert path.read_bytes() == before


def test_cli_evaluate_json_and_final(tmp_path: Path) -> None:
    root, bundle = _validation_case(tmp_path)
    listed = _RUNNER.invoke(
        app,
        [
            "actions",
            "evaluate",
            "--project",
            _PROJECT,
            "--job",
            "infer_val",
            "--workspace",
            str(root),
            "--json",
        ],
    )
    assert listed.exit_code == 0
    payload = json.loads(listed.stdout)
    assert payload["ok"] is True
    assert payload["data"]["per_class"]["idle"]["precision"] is None
    assert payload["data"]["evaluated_frames"] == 10
    refused = _RUNNER.invoke(
        app,
        [
            "actions",
            "evaluate",
            "--project",
            _PROJECT,
            "--job",
            "infer_val",
            "--split",
            "test",
            "--workspace",
            str(root),
            "--json",
        ],
    )
    assert refused.exit_code == 1
    assert json.loads(refused.stdout)["error"]["code"] == "not_final"
    _inference(
        root,
        tmp_path / "infer_test",
        job_id="infer_test",
        run_id="run_test",
        start_s=40.0,
        end_s=70.0,
        times=[40, 41, 42, 43],
        preds=["digging"] * 4,
    )
    final = _RUNNER.invoke(
        app,
        [
            "actions",
            "evaluate",
            "--project",
            _PROJECT,
            "--job",
            "infer_test",
            "--split",
            "test",
            "--final",
            "--workspace",
            str(root),
            "--json",
        ],
    )
    assert final.exit_code == 0
    body = json.loads(final.stdout)
    assert body["data"]["infer_job_id"] == "infer_test"
    assert (bundle / "evaluation_test.json").is_file()
