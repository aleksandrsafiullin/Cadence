"""Quality report, run comparison, and the train-only labeling queue."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from excavation_lab import storage
from excavation_lab.datasets.episodes import EpisodeSpec
from excavation_lab.datasets.splits import (
    freeze_split,
    frozen_manifest_path,
    propose_split,
    write_proposal,
)
from excavation_lab.errors import DatasetHashMismatchError as ErrorsHashError
from excavation_lab.errors import LabError
from excavation_lab.models.bundle import CLASS_NAMES
from excavation_lab.models.evaluate import NOT_EVALUATED as EVALUATE_NOT_EVALUATED
from excavation_lab.models.quality import (
    NOT_CALCULATED,
    NOT_EVALUATED,
    DatasetHashMismatchError,
    compare,
    load_quality_report,
    metric_text,
    send_to_labeling,
)
from excavation_lab.ui import copy as ui_copy

_HASH_A = "ab" * 32
_HASH_B = "cd" * 32
_CLASS_KEYS = (
    "ap50",
    "ap50_95",
    "precision",
    "recall",
    "recall_clear",
    "recall_partial",
    "gt_count",
)
_REPORT_KEYS = (
    "reviewed_frame_count",
    "evaluated_frame_count",
    "gt_object_count",
    "excluded_label_fraction",
    "recall_clear",
    "recall_partial",
    "decode_ms",
    "inference_ms",
    "encode_ms",
    "score_floor",
    "operating_threshold",
    "iou",
    "max_detections",
)
_EPISODES = (
    EpisodeSpec("train-ep", "vid", "g-train", 0, 60, "погрузка"),
    EpisodeSpec("val-ep", "vid", "g-val", 72, 92, "погрузка"),
    EpisodeSpec("test-ep", "vid", "g-test", 104, 124, "погрузка"),
)


def _numbers(value: object) -> list[float]:
    found: list[float] = []
    if isinstance(value, bool):
        return found
    if isinstance(value, (int, float)):
        found.append(float(value))
        return found
    if isinstance(value, dict):
        for item in value.values():
            found.extend(_numbers(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_numbers(item))
    return found


def _project(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    storage.create_project("pilot", workspace)
    return workspace


def _write_json(workspace: Path, eval_id: str, name: str, document: object) -> None:
    directory = storage.project_directory(workspace, "pilot") / "runs" / eval_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        json.dumps(document, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _freeze(workspace: Path) -> Path:
    storage.insert_video(workspace, "pilot", "vid", name="pilot")
    proposal = propose_split("vid", _EPISODES)
    manifest = write_proposal(workspace, "pilot", proposal)
    frozen, path = freeze_split(workspace, "pilot", manifest)
    assert [episode.split.value for episode in frozen.episodes] == [
        "train",
        "validation",
        "test",
    ]
    return path


def _write_sample(workspace: Path) -> None:
    """Sample split column lies. The frozen episode is the authority."""
    directory = storage.project_directory(workspace, "pilot") / "samples" / "vid"
    directory.mkdir(parents=True)
    document = {
        "rows": [
            {
                "video_id": "vid",
                "source_frame_index": 1,
                "source_time_s": 10.0,
                "episode_id": "train-ep",
                "split": "test",
            },
            {
                "video_id": "vid",
                "source_frame_index": 2,
                "source_time_s": 80.0,
                "episode_id": "val-ep",
                "split": "train",
            },
            {
                "video_id": "vid",
                "source_frame_index": 3,
                "source_time_s": 110.0,
                "episode_id": "test-ep",
                "split": "train",
            },
            {
                "video_id": "vid",
                "source_frame_index": 8,
                "source_time_s": 124.0,
                "episode_id": "train-ep",
                "split": "test",
            },
            {
                "video_id": "vid",
                "source_frame_index": 9,
                "source_time_s": 200.0,
                "episode_id": "train-ep",
                "split": "test",
            },
        ]
    }
    (directory / "sample_manifest.json").write_text(
        json.dumps(document) + "\n",
        encoding="utf-8",
    )


def _batches(workspace: Path) -> list[object]:
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        return list(
            connection.execute(
                """
                SELECT batch_id, video_id, status, path
                FROM annotation_batches
                ORDER BY batch_id
                """
            ).fetchall()
        )


def test_quality_does_not_import_streamlit_or_analytics() -> None:
    source = Path(load_quality_report.__code__.co_filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert "streamlit" not in modules
    assert not any(name == "excavation_lab.analytics" or name.startswith("excavation_lab.analytics.") for name in modules)
    assert "shell=True" not in source
    assert DatasetHashMismatchError is ErrorsHashError
    assert NOT_EVALUATED == EVALUATE_NOT_EVALUATED == "not_evaluated"
    assert NOT_CALCULATED == ui_copy.NOT_CALCULATED == "не рассчитано"


def test_missing_metrics_json_is_not_computed_not_zero(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    report = load_quality_report(workspace, "pilot", "absent")
    empty = load_quality_report(workspace, "pilot", None)

    assert report["status"] == "not_computed"
    assert empty["status"] == "not_computed"
    assert report["errors"] == []
    assert report["errors_status"] == "unavailable"
    assert report["errors_message"] == ui_copy.ERRORS_UNAVAILABLE
    assert report["idf1"] == "not_evaluated"
    assert report["hota"] == "not_evaluated"
    assert report["target_caption"] == ui_copy.TARGET_CAPTION
    for key in _REPORT_KEYS:
        assert report[key] == NOT_CALCULATED
        assert report[key] != 0
        assert empty[key] == NOT_CALCULATED
    for name in CLASS_NAMES:
        for key in _CLASS_KEYS:
            assert report["classes"][name][key] == NOT_CALCULATED
            assert report["classes"][name][key] != 0
    assert report["classes"]["bucket"]["ap50"] != 0
    truck = report["targets"]["haul_truck"]["precision"]
    bucket = report["targets"]["bucket"]["recall"]
    assert truck["status"] == "target"
    assert truck["value"] == NOT_CALCULATED
    assert truck["target"] == NOT_CALCULATED
    assert bucket["status"] != "achieved"
    assert _numbers(report) == [1.0]
    assert metric_text(None) == NOT_CALCULATED
    assert metric_text(report["classes"]["bucket"]["ap50"]) == NOT_CALCULATED


def test_present_metrics_keep_measured_values_and_null_timings(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    _write_json(
        workspace,
        "run-measured",
        "metrics.json",
        {
            "dataset_hash": _HASH_A,
            "split": "validation",
            "evaluated_frame_count": 4,
            "gt_object_count": 8,
            "excluded_label_fraction": 0.1,
            "recall_clear": 0.5,
            "recall_partial": 0.25,
            "decode_ms": None,
            "inference_ms": 3.5,
            "idf1": 0.99,
            "hota": 0.4,
            "throughput": {"encode_ms": 8.0},
            "classes": {
                "bucket": {
                    "ap50": 0.4,
                    "ap50_95": 0.2,
                    "precision": 0.4,
                    "recall": 0.4,
                    "recall_clear": 0.5,
                    "recall_partial": 0.25,
                    "gt_count": 3,
                },
                "haul_truck": {
                    "ap50": 0.4,
                    "ap50_95": 0.2,
                    "precision": 0.4,
                    "recall": 0.4,
                    "gt_count": 2,
                },
                "truck_bed": {
                    "ap50": 0,
                    "ap50_95": 0.0,
                    "precision": 0,
                    "recall": 0,
                    "gt_count": 0,
                },
            },
            "targets": {
                "haul_truck": {
                    "precision": {"target": 0.90, "value": 0.4, "status": "achieved"},
                    "recall": {"target": 0.90, "value": 0.4, "status": "target"},
                },
                "bucket": {
                    "precision": {"target": 0.85, "value": 0.4, "status": "target"},
                    "recall": {"target": 0.85, "value": 0.90, "status": "target"},
                },
            },
        },
    )
    report = load_quality_report(workspace, "pilot", "run-measured")
    bucket = report["classes"]["bucket"]
    assert report["status"] == "computed"
    assert report["reviewed_frame_count"] == 4
    assert report["evaluated_frame_count"] == 4
    assert report["gt_object_count"] == 8
    assert report["recall_clear"] == pytest.approx(0.5)
    assert report["recall_partial"] == pytest.approx(0.25)
    assert bucket["ap50"] == pytest.approx(0.4)
    assert bucket["ap50_95"] == pytest.approx(0.2)
    assert bucket["precision"] == pytest.approx(0.4)
    assert bucket["recall"] == pytest.approx(0.4)
    assert bucket["recall_clear"] == pytest.approx(0.5)
    assert bucket["recall_partial"] == pytest.approx(0.25)
    assert bucket["gt_count"] == 3
    assert bucket["ap50"] != 0.85
    assert report["classes"]["excavator_body"]["ap50"] == NOT_CALCULATED
    assert report["classes"]["truck_bed"]["ap50"] == 0
    assert report["classes"]["truck_bed"]["recall_clear"] == NOT_CALCULATED
    assert report["decode_ms"] == NOT_CALCULATED
    assert report["decode_ms"] != 0
    assert report["inference_ms"] == pytest.approx(3.5)
    assert report["encode_ms"] == pytest.approx(8.0)
    assert metric_text(None) == NOT_CALCULATED
    assert metric_text(report["decode_ms"]) == NOT_CALCULATED
    assert report["idf1"] == "not_evaluated"
    assert report["hota"] == "not_evaluated"
    assert report["idf1"] != 0.99
    haul = report["targets"]["haul_truck"]["precision"]
    assert haul["status"] == "target"
    assert haul["target"] == pytest.approx(0.90)
    assert haul["value"] == pytest.approx(0.4)
    assert report["classes"]["haul_truck"]["precision"] == pytest.approx(0.4)
    gated = report["targets"]["bucket"]["recall"]
    assert gated["status"] == "target"
    assert gated["value"] == pytest.approx(0.90)
    assert gated["target"] == pytest.approx(0.85)
    assert report["errors"] == []
    assert report["errors_status"] == "unavailable"


def test_compare_rejects_hash_mismatch_and_returns_deltas(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workspace = _project(tmp_path)
    row_low = {"ap50": 0.25, "ap50_95": 0.25, "precision": 0.25, "recall": 0.25}
    row_high = {"ap50": 0.5, "ap50_95": 0.5, "precision": 0.5, "recall": 0.5}
    _write_json(
        workspace,
        "run-a",
        "metrics.json",
        {"dataset_hash": _HASH_A, "classes": {name: dict(row_low) for name in CLASS_NAMES}},
    )
    _write_json(
        workspace,
        "run-b",
        "metrics.json",
        {"dataset_hash": _HASH_B, "classes": {name: dict(row_high) for name in CLASS_NAMES}},
    )
    _write_json(
        workspace,
        "run-c",
        "metrics.json",
        {"dataset_hash": _HASH_A, "classes": {name: dict(row_high) for name in CLASS_NAMES}},
    )

    with pytest.raises(DatasetHashMismatchError) as caught:
        compare(workspace, "pilot", "run-a", "run-b")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert caught.value is not None
    assert "delta" not in vars(caught.value)
    assert caught.value.code == "dataset_hash_mismatch"
    assert not hasattr(caught.value, "delta")

    matched = compare(workspace, "pilot", "run-a", "run-c")
    assert matched["dataset_hash"] == _HASH_A
    assert set(matched["delta"]) == set(CLASS_NAMES)
    for name in CLASS_NAMES:
        for key in ("ap50", "ap50_95", "precision", "recall"):
            assert matched["delta"][name][key] == pytest.approx(0.25)


def test_send_to_labeling_refuses_validation_and_test(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    manifest = _freeze(workspace)
    _write_sample(workspace)
    before = manifest.read_bytes()

    with pytest.raises(LabError) as test_split:
        send_to_labeling(workspace, "pilot", "vid:3")
    assert test_split.value.code == "not_train_frame"
    assert "test" in test_split.value.message

    with pytest.raises(LabError) as validation_split:
        send_to_labeling(workspace, "pilot", "vid:2")
    assert validation_split.value.code == "not_train_frame"
    assert "validation" in validation_split.value.message

    for frame_id in ("vid:8", "vid:9"):
        with pytest.raises(LabError) as outside:
            send_to_labeling(workspace, "pilot", frame_id)
        assert outside.value.code == "not_train_frame"
    assert _batches(workspace) == []
    batch_root = storage.project_directory(workspace, "pilot") / "annotation_batches"
    assert list(batch_root.iterdir()) == []

    queued = send_to_labeling(workspace, "pilot", "vid:1")
    assert queued["split"] == "train"
    assert queued["status"] == "queued"
    assert queued["frame_id"] == "vid:1"
    body = json.loads(Path(queued["path"]).read_text(encoding="utf-8"))
    assert body["split"] == "train"
    assert body["kind"] == "annotation_batch"
    assert body["predictions_included"] is False
    assert body["ground_truth"] is False
    rows = _batches(workspace)
    assert len(rows) == 1
    assert rows[0]["status"] == "queued"
    assert rows[0]["video_id"] == "vid"

    with pytest.raises(LabError) as again:
        send_to_labeling(workspace, "pilot", "vid:3")
    assert again.value.code == "not_train_frame"
    assert len(_batches(workspace)) == 1
    assert manifest.read_bytes() == before
    assert frozen_manifest_path(workspace, "pilot", "vid").read_bytes() == before


def test_errors_come_from_the_match_file(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    _freeze(workspace)
    _write_json(
        workspace,
        "run-errors",
        "matches.json",
        {
            "video_id": "vid",
            "false_positives": [
                {
                    "class_name": "bucket",
                    "source_frame_index": 3,
                    "source_time_s": 110.0,
                    "split": "train",
                    "xyxy": [1, 2, 3, 4],
                }
            ],
            "false_negatives": [
                {
                    "class_name": "haul_truck",
                    "source_frame_index": 1,
                    "source_time_s": 10.0,
                    "xyxy": [0, 0, 8, 8],
                }
            ],
            "rows": [
                {
                    "kind": "true_positive",
                    "source_frame_index": 9,
                    "source_time_s": 1.0,
                    "xyxy": [9, 9, 9, 9],
                }
            ],
        },
    )
    missing = load_quality_report(workspace, "pilot", "no-such-run")
    assert missing["status"] == "not_computed"
    assert missing["errors"] == []
    assert missing["errors_status"] == "unavailable"
    assert missing["classes"]["bucket"]["ap50"] == NOT_CALCULATED

    report = load_quality_report(workspace, "pilot", "run-errors")
    assert report["status"] == "not_computed"
    assert report["errors_status"] == "available"
    assert report["errors_message"] == ui_copy.ERRORS_FROM_MATCHES
    assert [row["kind"] for row in report["errors"]] == ["false_negative", "false_positive"]
    missed, false = report["errors"]
    assert missed["source_frame_index"] == 1
    assert missed["source_time_s"] == pytest.approx(10.0)
    assert missed["split"] == "train"
    assert missed["class_name"] == "haul_truck"
    assert missed["xyxy"] == [0.0, 0.0, 8.0, 8.0]
    assert false["source_frame_index"] == 3
    assert false["source_time_s"] == pytest.approx(110.0)
    assert false["split"] == "test"
    assert false["xyxy"] == [1.0, 2.0, 3.0, 4.0]
    assert all(row["xyxy"] != [9.0, 9.0, 9.0, 9.0] for row in report["errors"])
    assert len(report["errors"]) == 2
