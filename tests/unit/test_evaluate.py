"""Detection metrics from synthetic boxes. Does not load RF-DETR weights."""

from __future__ import annotations

import errno
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from excavation_lab.cli import app
from excavation_lab.datasets.catalog import catalog_from_seed, list_catalogs
from excavation_lab.datasets.coco_import import build_class_mapping
from excavation_lab.errors import (
    DatasetHashMismatchError,
    DiskFullError,
    LabError,
)
from excavation_lab.models import evaluate as evaluate_module
from excavation_lab.models.bundle import (
    NOT_CALCULATED,
    ModelBundleSpec,
    verify_bundle_checksums,
    write_model_bundle,
)
from excavation_lab.models.evaluate import (
    NOT_EVALUATED,
    DetectorBox,
    EvalFrame,
    GroundTruthBox,
    _frames_from_coco,
    compare_evaluations,
    compute_detection_metrics,
    evaluate_project,
)
from excavation_lab.schemas import pilot_path
from excavation_lab.storage import connect, create_project, project_sqlite

_RUNNER = CliRunner()
_HASH = "ab" * 32
_OTHER_HASH = "cd" * 32
_FROZEN_HASH = "ef" * 32
_SENTINEL = "SENTINEL-VAL-METRICS"


def _mapping() -> dict[str, object]:
    return build_class_mapping(catalog_from_seed())


def _compute(
    frames: list[EvalFrame],
    *,
    operating_threshold: float = 0.10,
    score_floor: float = 0.001,
    max_detections: int = 100,
    independence: str | None = None,
    dataset_hash: str = _HASH,
    track_gt_path: Path | None = None,
) -> dict[str, object]:
    return compute_detection_metrics(
        frames,
        class_mapping=_mapping(),
        score_floor=score_floor,
        operating_threshold=operating_threshold,
        iou=0.5,
        max_detections=max_detections,
        dataset_hash=dataset_hash,
        split="validation",
        independence=independence,
        track_gt_path=track_gt_path,
    )


def _box(
    score: float,
    xyxy: tuple[float, float, float, float],
    class_name: str = "bucket",
    class_id: int | None = None,
    source: str = "detector",
    track_id: str | None = None,
) -> DetectorBox:
    return DetectorBox(
        score=score,
        xyxy=xyxy,
        class_name=class_name,
        class_id=class_id,
        source=source,
        track_id=track_id,
    )


def _gt(
    xyxy: tuple[float, float, float, float],
    class_name: str = "bucket",
    visibility: str | None = "clear",
) -> GroundTruthBox:
    return GroundTruthBox(class_name, xyxy, visibility)


def _frame(
    image_id: str,
    ground_truth: tuple[GroundTruthBox, ...] = (),
    detections: tuple[DetectorBox, ...] = (),
    *,
    excluded: bool = False,
) -> EvalFrame:
    return EvalFrame(
        image_id,
        64,
        64,
        ground_truth=ground_truth,
        detections=detections,
        excluded=excluded,
    )


def test_one_to_one_match_matches_the_hand_case() -> None:
    # One GT. Two identical detections. The higher score takes the only GT.
    # TP=1, FP=1, precision=1/2, recall=1/1.
    report = _compute(
        [
            _frame(
                "one",
                (_gt((0, 0, 10, 10)),),
                (_box(0.9, (0, 0, 10, 10)), _box(0.8, (0, 0, 10, 10))),
            )
        ]
    )
    row = report["classes"]["bucket"]
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(1.0)
    assert row["pr_curve"] == [
        {"score": 0.9, "precision": 1.0, "recall": 1.0, "tp": 1, "fp": 0},
        {"score": 0.8, "precision": 0.5, "recall": 1.0, "tp": 1, "fp": 1},
    ]
    # IoU([0,0,10,10], [0,0,10,20]) = 100/200 = 0.5, which matches.
    touching = _compute(
        [_frame("edge", (_gt((0, 0, 10, 10)),), (_box(0.95, (0, 0, 10, 20)),))]
    )
    assert touching["classes"]["bucket"]["recall"] == pytest.approx(1.0)
    assert touching["classes"]["bucket"]["precision"] == pytest.approx(1.0)
    # IoU([0,0,10,10], [5,0,15,10]) = 50/150 < 0.5, so the detection is an FP.
    missed = _compute(
        [_frame("miss", (_gt((0, 0, 10, 10)),), (_box(0.95, (5, 0, 15, 10)),))]
    )
    assert missed["classes"]["bucket"]["precision"] == pytest.approx(0.0)
    assert missed["classes"]["bucket"]["recall"] == pytest.approx(0.0)
    assert report["classes"]["bucket"]["recall_partial"] == NOT_CALCULATED


def test_low_score_box_stays_on_the_pr_curve() -> None:
    kept = _frame(
        "low",
        (_gt((0, 0, 10, 10)),),
        (
            _box(0.5, (40, 40, 50, 50)),
            _box(0.01, (0, 0, 10, 10)),
        ),
    )
    dropped = _frame(
        "low",
        (_gt((0, 0, 10, 10)),),
        (_box(0.5, (40, 40, 50, 50)),),
    )
    with_low = _compute([kept])
    without = _compute([dropped])
    scores = [point["score"] for point in with_low["classes"]["bucket"]["pr_curve"]]
    assert 0.01 in scores
    assert with_low["classes"]["bucket"]["ap50"] > without["classes"]["bucket"]["ap50"]
    assert without["classes"]["bucket"]["ap50"] == pytest.approx(0.0)
    assert with_low["classes"]["bucket"]["ap50"] == pytest.approx(_reference_ap50(kept))


def test_score_below_detection_floor_is_not_dropped() -> None:
    report = _compute(
        [
            _frame(
                "floor",
                (_gt((0, 0, 10, 10), "haul_truck"),),
                (_box(0.05, (0, 0, 10, 10), "haul_truck"),),
            )
        ]
    )
    row = report["classes"]["haul_truck"]
    assert [point["score"] for point in row["pr_curve"]] == [0.05]
    assert row["ap50"] == pytest.approx(1.0)
    assert row["ap50_95"] == pytest.approx(1.0)
    assert report["operating_threshold"] == pytest.approx(0.10)
    assert report["score_floor"] == pytest.approx(0.001)
    assert row["recall"] == pytest.approx(0.0)
    assert row["precision"] == NOT_CALCULATED


def test_perfect_overlap_matches_pycocotools() -> None:
    frame = _frame(
        "perfect",
        (_gt((0, 0, 10, 10)),),
        (_box(0.9, (0, 0, 10, 10)),),
    )
    report = _compute([frame])
    assert report["classes"]["bucket"]["ap50"] == pytest.approx(1.0)
    assert report["classes"]["bucket"]["ap50_95"] == pytest.approx(1.0)
    assert report["classes"]["bucket"]["ap50"] == pytest.approx(_reference_ap50(frame))
    assert report["max_detections"] == 100
    assert report["operating_threshold"] == pytest.approx(0.10)


def test_max_detections_is_separate_from_the_operating_threshold() -> None:
    detections = tuple(
        _box(0.2 + index * 0.001, (40, 40, 50, 50)) for index in range(101)
    )
    report = _compute([_frame("cap", (_gt((0, 0, 10, 10)),), detections)])
    scores = [point["score"] for point in report["classes"]["bucket"]["pr_curve"]]
    assert report["max_detections"] == 100
    assert report["operating_threshold"] == pytest.approx(0.10)
    assert len(scores) == 100
    assert min(scores) == pytest.approx(0.201)
    assert 0.2 not in scores


def test_unlabelable_frame_is_excluded_not_a_false_positive() -> None:
    report = _compute(
        [
            _frame(
                "hidden",
                (_gt((0, 0, 10, 10), visibility="unlabelable"),),
                (_box(0.9, (0, 0, 10, 10)),),
            )
        ]
    )
    assert report["gt_object_count"] == 0
    assert report["excluded_label_fraction"] == pytest.approx(1.0)
    assert report["classes"]["bucket"]["precision"] == NOT_CALCULATED
    assert report["classes"]["bucket"]["ap50"] == NOT_CALCULATED
    tracked = _frame(
        "tracked",
        (_gt((0, 0, 10, 10), visibility="unlabelable"),),
        (_box(0.9, (0, 0, 10, 10), source="tracked"),),
    )
    with pytest.raises(LabError) as caught:
        _compute([tracked])
    assert caught.value.code == "not_detector_boxes"


def test_max_detections_cap_is_per_class() -> None:
    trucks = tuple(_box(0.5, (40, 40, 50, 50), "haul_truck") for _index in range(100))
    report = _compute(
        [
            _frame(
                "mixed",
                (_gt((0, 0, 10, 10), "bucket"),),
                (*trucks, _box(0.05, (0, 0, 10, 10), "bucket")),
            )
        ]
    )
    scores = [point["score"] for point in report["classes"]["bucket"]["pr_curve"]]
    assert scores == [0.05]


def test_unknown_coco_visibility_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "ds"
    _write_visibility_coco(dataset, {"visibility": "occluded"})
    with pytest.raises(LabError) as caught:
        _frames_from_coco(dataset, "validation", _mapping())
    assert caught.value.code == "invalid_visibility"
    _write_visibility_coco(dataset, None)
    frames = _frames_from_coco(dataset, "validation", _mapping())
    assert frames is not None
    assert frames[0].ground_truth[0].visibility is None


def test_report_counts_strata_and_full_domain_precision() -> None:
    kept = _frame(
        "kept",
        (_gt((0, 0, 10, 10), visibility="clear"), _gt((20, 0, 30, 10), visibility="partial")),
        (
            _box(0.9, (0, 0, 10, 10)),
            _box(0.4, (50, 50, 60, 60)),
        ),
    )
    excluded = _frame(
        "drop",
        (_gt((0, 0, 10, 10)),),
        (_box(0.99, (0, 0, 10, 10)),),
        excluded=True,
    )
    report = _compute([kept, excluded])
    assert report["gt_object_count"] == 2
    assert report["evaluated_frame_count"] == 1
    assert report["excluded_label_fraction"] == pytest.approx(0.5)
    assert report["precision_domain"] == "evaluated"
    row = report["classes"]["bucket"]
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["recall_clear"] == pytest.approx(1.0)
    assert row["recall_partial"] == pytest.approx(0.0)
    assert report["recall_clear"] == pytest.approx(1.0)
    assert report["recall_partial"] == pytest.approx(0.0)


def test_class_id_is_matched_by_name() -> None:
    report = _compute(
        [
            _frame(
                "name",
                (_gt((0, 0, 10, 10)),),
                (_box(0.9, (0, 0, 10, 10), class_id=99),),
            )
        ]
    )
    assert report["classes"]["bucket"]["recall"] == pytest.approx(1.0)
    assert report["classes"]["bucket"]["ap50"] == pytest.approx(1.0)


def test_tracked_boxes_are_refused() -> None:
    frame = _frame(
        "track",
        (_gt((0, 0, 10, 10)),),
        (_box(0.9, (0, 0, 10, 10), source="tracked"),),
    )
    with pytest.raises(LabError) as caught:
        _compute([frame])
    assert caught.value.code == "not_detector_boxes"
    smoothed = _frame(
        "smooth",
        (_gt((0, 0, 10, 10)),),
        (_box(0.9, (0, 0, 10, 10), track_id="t1"),),
    )
    with pytest.raises(LabError) as smoothed_caught:
        _compute([smoothed])
    assert smoothed_caught.value.code == "not_detector_boxes"


def test_targets_stay_targets_until_the_value_meets_them() -> None:
    short = _compute(
        [
            _frame(
                "short",
                (_gt((0, 0, 10, 10)),),
                (_box(0.9, (0, 0, 10, 10)), _box(0.8, (0, 0, 10, 10))),
            )
        ]
    )
    encoded = json.dumps(short["targets"])
    assert "achieved" not in encoded
    assert short["targets"]["bucket"]["recall"]["status"] == "target"
    assert short["targets"]["bucket"]["recall"]["target"] == pytest.approx(0.85)
    assert short["targets"]["haul_truck"]["precision"]["target"] == pytest.approx(0.90)
    met = _compute(
        [
            _frame(
                "met",
                (
                    _gt((0, 0, 10, 10), "bucket"),
                    _gt((20, 0, 30, 10), "haul_truck"),
                ),
                (
                    _box(0.9, (0, 0, 10, 10), "bucket"),
                    _box(0.9, (20, 0, 30, 10), "haul_truck"),
                ),
            )
        ]
    )
    statuses = [
        met["targets"]["haul_truck"]["precision"]["status"],
        met["targets"]["haul_truck"]["recall"]["status"],
        met["targets"]["bucket"]["precision"]["status"],
        met["targets"]["bucket"]["recall"]["status"],
    ]
    assert statuses == ["achieved", "achieved", "achieved", "achieved"]


def test_same_scene_independence_is_kept() -> None:
    report = _compute(
        [_frame("scene", (_gt((0, 0, 10, 10)),), (_box(0.9, (0, 0, 10, 10)),))],
        independence="same_scene",
    )
    assert report["independence"] == "same_scene"


def test_hash_mismatch_raises_without_a_delta() -> None:
    left = _compute([_frame("a", (_gt((0, 0, 10, 10)),), (_box(0.9, (0, 0, 10, 10)),))])
    right = dict(left)
    right["dataset_hash"] = _OTHER_HASH
    with pytest.raises(DatasetHashMismatchError) as caught:
        compare_evaluations(left, right)
    assert caught.value.code == "dataset_hash_mismatch"
    assert "delta" not in caught.value.__dict__
    other = _compute(
        [_frame("b", (_gt((0, 0, 10, 10)),), (_box(0.2, (40, 40, 50, 50)),))]
    )
    compared = compare_evaluations(left, other)
    assert compared["dataset_hash"] == _HASH
    assert "delta" in compared


def test_idf1_and_hota_are_not_estimated(tmp_path: Path) -> None:
    track = tmp_path / "tracks.json"
    track.write_text(json.dumps({"idf1": 0.99, "hota": 0.88}), encoding="utf-8")
    report = _compute(
        [_frame("sparse", (_gt((0, 0, 10, 10)),), (_box(0.9, (0, 0, 10, 10)),))],
        track_gt_path=track,
    )
    assert report["idf1"] == NOT_EVALUATED
    assert report["hota"] == NOT_EVALUATED
    assert isinstance(report["idf1"], str)
    assert isinstance(report["hota"], str)
    blob = json.dumps(report)
    assert '"idf1": 0.99' not in blob
    assert '"hota": 0.88' not in blob


def test_evaluate_module_does_not_call_train() -> None:
    source = Path(evaluate_module.__file__).read_text(encoding="utf-8")
    assert "enqueue_train" not in source
    assert "run_train_job" not in source


def test_metric_write_maps_disk_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, _model_dir = _install_project(tmp_path)
    _write_detections(tmp_path / "boxes.json")

    def _no_space(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("excavation_lab.models.evaluate.os.replace", _no_space)
    with pytest.raises(DiskFullError) as caught:
        evaluate_project(
            workspace,
            "pilot",
            "m1",
            "validation",
            detections_path=tmp_path / "boxes.json",
        )
    assert caught.value.code == "disk_full"


def test_validation_command_writes_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, model_dir = _install_project(tmp_path, manifest=True)
    spies = _spy_train_and_weights(monkeypatch)
    detections = tmp_path / "boxes.json"
    _write_detections(detections)
    before_pilot = pilot_path().read_bytes()
    before_validation = (model_dir / "validation_metrics.json").read_bytes()
    result = _RUNNER.invoke(
        app,
        [
            "evaluate",
            "--project",
            "pilot",
            "--model",
            "m1",
            "--split",
            "validation",
            "--workspace",
            str(workspace),
            "--detections",
            str(detections),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Traceback" not in result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    data = payload["data"]
    assert data["max_detections"] == 100
    assert data["operating_threshold"] == pytest.approx(0.10)
    assert data["score_floor"] == pytest.approx(0.001)
    assert data["independence"] == "same_scene"
    assert data["classes"]["bucket"]["ap50"] == pytest.approx(1.0)
    assert [point["score"] for point in data["classes"]["bucket"]["pr_curve"]] == [0.01]
    metrics = _metrics_file(workspace, data["eval_id"])
    assert metrics.is_file()
    stored = json.loads(metrics.read_text(encoding="utf-8"))
    assert stored["max_detections"] == 100
    assert not (model_dir / "test_metrics.json").exists()
    assert (model_dir / "validation_metrics.json").read_bytes() == before_validation
    assert pilot_path().read_bytes() == before_pilot
    assert spies["train"] == 0
    assert spies["weights"] == 0


def test_final_refuses_to_train_when_requirements_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    create_project("pilot", workspace, catalog_id="excavation")
    spies = _spy_train_and_weights(monkeypatch)
    before = pilot_path().read_bytes()
    result = _RUNNER.invoke(
        app,
        [
            "evaluate",
            "--project",
            "pilot",
            "--model",
            "m1",
            "--split",
            "test",
            "--final",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "not_final"
    assert payload["error"]["code"].isascii()
    assert "delta" not in result.stdout
    assert _cyrillic(payload["error"]["message"])
    assert spies["train"] == 0
    assert spies["weights"] == 0
    assert pilot_path().read_bytes() == before
    assert "detection_floor: 0.10" in pilot_path().read_text(encoding="utf-8")


def test_final_missing_manifest_does_not_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _model_dir = _install_project(tmp_path, manifest=False)
    spies = _spy_train_and_weights(monkeypatch)
    before = pilot_path().read_bytes()
    result = _RUNNER.invoke(
        app,
        [
            "evaluate",
            "--project",
            "pilot",
            "--model",
            "m1",
            "--split",
            "test",
            "--final",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "not_final"
    assert "манифест" in payload["error"]["message"]
    assert "delta" not in result.stdout
    assert spies["train"] == 0
    assert pilot_path().read_bytes() == before


def test_final_writes_test_metrics_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, model_dir = _install_project(tmp_path, manifest=True)
    spies = _spy_train_and_weights(monkeypatch)
    detections = tmp_path / "boxes.json"
    _write_detections(detections)
    before_pilot = pilot_path().read_bytes()
    before_validation = (model_dir / "validation_metrics.json").read_bytes()
    result = _RUNNER.invoke(
        app,
        [
            "evaluate",
            "--project",
            "pilot",
            "--model",
            "m1",
            "--split",
            "test",
            "--final",
            "--workspace",
            str(workspace),
            "--detections",
            str(detections),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["final"] is True
    assert payload["data"]["split"] == "test"
    assert payload["data"]["independence"] == "same_scene"
    assert spies["train"] == 0
    assert spies["weights"] == 0
    assert pilot_path().read_bytes() == before_pilot
    assert (model_dir / "validation_metrics.json").read_bytes() == before_validation
    test_metrics = model_dir / "test_metrics.json"
    assert test_metrics.is_file()
    stored = json.loads(test_metrics.read_text(encoding="utf-8"))
    assert stored["split"] == "test"
    assert _SENTINEL not in test_metrics.read_text(encoding="utf-8")
    verify_bundle_checksums(model_dir)


def test_missing_test_labels_do_not_borrow_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, model_dir = _install_project(tmp_path, manifest=True)
    dataset = _dataset_dir(workspace)
    _write_coco(dataset / "valid", images=True)
    _write_coco(dataset / "test", images=False)
    spies = _spy_train_and_weights(monkeypatch)
    before_validation = (model_dir / "validation_metrics.json").read_bytes()
    result = _RUNNER.invoke(
        app,
        [
            "evaluate",
            "--project",
            "pilot",
            "--model",
            "m1",
            "--split",
            "test",
            "--final",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "test_not_labeled"
    assert _cyrillic(payload["error"]["message"])
    data = payload["data"]
    assert data["gt_object_count"] == NOT_CALCULATED
    assert data["evaluated_frame_count"] == NOT_CALCULATED
    assert data["classes"]["bucket"]["ap50"] == NOT_CALCULATED
    assert data["classes"]["haul_truck"]["precision"] == NOT_CALCULATED
    assert data["idf1"] == NOT_EVALUATED
    assert _SENTINEL not in result.stdout
    assert "0.777" not in result.stdout
    assert (model_dir / "validation_metrics.json").read_bytes() == before_validation
    assert not (model_dir / "test_metrics.json").exists()
    assert spies["train"] == 0
    assert spies["weights"] == 0
    text = _RUNNER.invoke(
        app,
        [
            "evaluate",
            "--project",
            "pilot",
            "--model",
            "m1",
            "--split",
            "test",
            "--final",
            "--workspace",
            str(workspace),
        ],
    )
    assert text.exit_code == 1
    assert "Traceback" not in text.output
    assert "test_not_labeled" in text.stderr
    assert NOT_CALCULATED in text.stderr


def test_compare_mismatch_prints_no_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _model_dir = _install_project(tmp_path, manifest=True)
    _spy_train_and_weights(monkeypatch)
    detections = tmp_path / "boxes.json"
    _write_detections(detections)
    other = _dataset_dir(workspace).parent.parent / "runs" / "other"
    other.mkdir(parents=True)
    (other / "metrics.json").write_text(
        json.dumps({"dataset_hash": _OTHER_HASH, "classes": {}}),
        encoding="utf-8",
    )
    result = _RUNNER.invoke(
        app,
        [
            "evaluate",
            "--project",
            "pilot",
            "--model",
            "m1",
            "--split",
            "validation",
            "--workspace",
            str(workspace),
            "--detections",
            str(detections),
            "--compare",
            "other",
            "--json",
        ],
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "dataset_hash_mismatch"
    assert "delta" not in result.stdout


def _reference_ap50(frame: EvalFrame) -> float:
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    mapping = build_class_mapping(catalog_from_seed())["by_name"]
    category_id = int(mapping["bucket"]["internal_id"])
    dataset = {
        "images": [{"id": 1, "file_name": frame.image_id, "width": 64, "height": 64}],
        "annotations": [
            {
                "id": index + 1,
                "image_id": 1,
                "category_id": category_id,
                "bbox": [
                    box.xyxy[0],
                    box.xyxy[1],
                    box.xyxy[2] - box.xyxy[0],
                    box.xyxy[3] - box.xyxy[1],
                ],
                "area": (box.xyxy[2] - box.xyxy[0]) * (box.xyxy[3] - box.xyxy[1]),
                "iscrowd": 0,
                "ignore": 0,
            }
            for index, box in enumerate(frame.ground_truth)
        ],
        "categories": [{"id": category_id, "name": "bucket"}],
    }
    detections = [
        {
            "image_id": 1,
            "category_id": category_id,
            "bbox": [
                box.xyxy[0],
                box.xyxy[1],
                box.xyxy[2] - box.xyxy[0],
                box.xyxy[3] - box.xyxy[1],
            ],
            "score": box.score,
        }
        for box in frame.detections
    ]
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        coco_gt = COCO()
        coco_gt.dataset = dataset
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"]
    values = precision[0, :, 0, 0, -1]
    valid = values[values > -1]
    return float(valid.mean())


def _install_project(tmp_path: Path, *, manifest: bool = True) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    record = create_project("pilot", workspace, catalog_id="excavation")
    project_dir = Path(record["project_dir"])
    model_dir = write_model_bundle(project_dir / "models" / "m1", _spec())
    with connect(project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            """
            INSERT INTO model_versions (
                model_id, project_id, dataset_id, variant, path, created_at
            ) VALUES ('m1', 'pilot', NULL, 'nano', ?, ?)
            """,
            (str(model_dir.resolve()), datetime.now(UTC).isoformat()),
        )
    dataset = project_dir / "dataset_versions" / "ds1"
    dataset.mkdir()
    if manifest:
        (dataset / "split_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "video_id": "v1",
                    "frozen": True,
                    "independence": "same_scene",
                    "content_hash": _FROZEN_HASH,
                }
            ),
            encoding="utf-8",
        )
    return workspace, model_dir


def _spec() -> ModelBundleSpec:
    return ModelBundleSpec(
        variant="nano",
        resolution=384,
        dataset_hash=_HASH,
        library_versions={"rfdetr": "1.11.2", "torch": None, "torchvision": None},
        resolved_train_config={
            "schema_version": 1,
            "profile": "smoke",
            "run_test": False,
        },
        dataset_manifest={
            "schema_version": 1,
            "dataset_id": "ds1",
            "project_id": "pilot",
            "dataset_hash": _HASH,
        },
        environment_report={"schema_version": 1, "inference_device": "cpu"},
        validation_metrics={"schema_version": 1, "note": _SENTINEL},
        weights=b"not-a-real-checkpoint",
        split_independence="same_scene",
    )


def _dataset_dir(workspace: Path) -> Path:
    return workspace / "projects" / "pilot" / "dataset_versions" / "ds1"


def _write_visibility_coco(dataset: Path, attributes: dict[str, str] | None) -> None:
    valid = dataset / "valid"
    valid.mkdir(parents=True, exist_ok=True)
    annotation: dict[str, object] = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [0, 0, 10, 10],
        "area": 100,
        "iscrowd": 0,
    }
    if attributes is not None:
        annotation["attributes"] = attributes
    document = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 32, "height": 32}],
        "annotations": [annotation],
        "categories": [{"id": 1, "name": "bucket"}],
    }
    (valid / "_annotations.coco.json").write_text(json.dumps(document), encoding="utf-8")


def _write_coco(directory: Path, *, images: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if images:
        body: dict[str, object] = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 32, "height": 32}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "bucket"}],
        }
    else:
        body = {"images": [], "annotations": [], "categories": []}
    (directory / "_annotations.coco.json").write_text(
        json.dumps(body),
        encoding="utf-8",
    )


def _write_detections(path: Path) -> None:
    document = {
        "images": [
            {
                "image_id": "frame-1",
                "width": 64,
                "height": 64,
                "excluded": False,
                "ground_truth": [
                    {
                        "class_name": "bucket",
                        "xyxy": [0, 0, 10, 10],
                        "visibility": "clear",
                    }
                ],
                "detections": [
                    {
                        "class_name": "bucket",
                        "class_id": 99,
                        "score": 0.01,
                        "xyxy": [0, 0, 10, 10],
                        "source": "detector",
                    }
                ],
            }
        ]
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def _metrics_file(workspace: Path, eval_id: str) -> Path:
    return workspace / "projects" / "pilot" / "runs" / eval_id / "metrics.json"


def _spy_train_and_weights(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    state = {"train": 0, "weights": 0}

    def _train(*_args: object, **_kwargs: object) -> None:
        state["train"] += 1
        raise AssertionError("train was called")

    def _weights(*_args: object, **_kwargs: object) -> None:
        state["weights"] += 1
        raise AssertionError("RF-DETR weights were loaded")

    monkeypatch.setattr("excavation_lab.models.train.enqueue_train", _train)
    monkeypatch.setattr("excavation_lab.models.train.run_train_job", _train)
    monkeypatch.setattr("excavation_lab.cli.enqueue_train", _train)
    monkeypatch.setattr("excavation_lab.models.bundle.load_bundle", _weights)
    return state


def _cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)
