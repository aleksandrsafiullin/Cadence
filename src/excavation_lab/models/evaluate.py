"""Detection metrics on raw boxes from a frozen split.

AP uses detector scores at or above ``eval.score_floor`` (0.001 in the
pilot file). The operating threshold is ``detection_floor`` (0.10) and
applies only to the precision/recall point. It does not clip the PR
curve. ``eval.max_detections`` is stored beside that threshold.

Class ids are resolved through ``class_mapping.json`` by name before
scores are compared. Tracked and smoothed boxes are refused.

``--final`` checks a frozen manifest, the bundle checksum, and a hash
of ``resolved_train_config.json``. It does not train and it does not
write ``configs/pilot.yaml``. A test split with no labels returns
``test_not_labeled`` and «не рассчитано». Validation metrics are not
substituted.

IDF1 and HOTA stay ``not_evaluated``. They are not estimated from
sparse detection labels, even when a track file path is passed.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from excavation_lab.datasets.geometry import xyxy_to_xywh
from excavation_lab.errors import (
    DatasetHashMismatchError,
    LabError,
    NotFinalError,
    TestNotLabeledError,
    UnknownCategoryError,
    UnsupportedCrowdError,
    map_os_error,
)
from excavation_lab.models.bundle import (
    CHECKSUMS_NAME,
    CLASS_NAMES,
    NOT_CALCULATED,
    BundleChecksumError,
    verify_bundle_checksums,
)
from excavation_lab.schemas import SCHEMA_VERSION, load_pilot
from excavation_lab.storage import connect, project_directory, project_sqlite

NOT_EVALUATED = "not_evaluated"
PRECISION_DOMAIN = "evaluated"
_SPLITS = frozenset({"validation", "test"})
_SPLIT_DIR = {"validation": "valid", "test": "test"}
_INDEPENDENCE = frozenset({"episode_split", "same_scene"})
_VISIBILITY = frozenset({"clear", "partial", "unlabelable"})
_REJECTED_SOURCES = frozenset({"tracked", "smoothed", "predicted"})
_HEX = re.compile(r"^[0-9a-f]{64}$")
# Pilot orientation, not a guarantee. Printed as target unless the value meets it.
_TRUCK_TARGET = 0.90
_BUCKET_TARGET = 0.85
_METRIC_NAMES = (
    "ap50",
    "ap50_95",
    "precision",
    "recall",
    "recall_clear",
    "recall_partial",
)


@dataclass(frozen=True, slots=True)
class DetectorBox:
    """One raw detector box. ``class_name`` wins over ``class_id``."""

    score: float
    xyxy: tuple[float, float, float, float]
    class_name: str | None = None
    class_id: int | None = None
    source: str = "detector"
    track_id: str | None = None


@dataclass(frozen=True, slots=True)
class GroundTruthBox:
    """One label box. ``unlabelable`` excludes the whole frame."""

    class_name: str
    xyxy: tuple[float, float, float, float]
    visibility: str | None = None


@dataclass(frozen=True, slots=True)
class EvalFrame:
    """One frame. Excluded frames stay out of precision, recall, and AP."""

    image_id: str
    width: int
    height: int
    ground_truth: tuple[GroundTruthBox, ...] = ()
    detections: tuple[DetectorBox, ...] = ()
    excluded: bool = False


def compute_detection_metrics(
    frames: Sequence[EvalFrame],
    *,
    class_mapping: Mapping[str, Any],
    score_floor: float,
    operating_threshold: float,
    iou: float,
    max_detections: int,
    dataset_hash: str,
    split: str,
    independence: str | None = None,
    labelability_known: bool = True,
    track_gt_path: Path | None = None,
    project_id: str | None = None,
    model_id: str | None = None,
    eval_id: str | None = None,
    final: bool = False,
    resolved_config_hash: str | None = None,
) -> dict[str, Any]:
    """Per-class AP and operating-point precision/recall.

    Boxes below ``score_floor`` are dropped. Boxes below
    ``operating_threshold`` stay on the PR curve and out of the
    precision/recall point. Matching is one-to-one at ``iou``.
    """
    _require_split(split)
    _require_hash(dataset_hash)
    names = _mapping_names(class_mapping)
    prepared = [_prepare_frame(frame, names) for frame in frames]
    evaluated = [frame for frame in prepared if not frame.excluded]
    by_class = _metrics_by_class(
        evaluated,
        names=names,
        score_floor=score_floor,
        operating_threshold=operating_threshold,
        iou=iou,
        max_detections=max_detections,
    )
    gt_count = sum(row["gt_count"] for row in by_class.values())
    clear_hit, clear_gt, partial_hit, partial_gt = _pool_visibility(by_class)
    independence_value = _shown_independence(independence)
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_id": eval_id if eval_id else uuid.uuid4().hex,
        "project_id": project_id,
        "model_id": model_id,
        "split": split,
        "final": final,
        "dataset_hash": dataset_hash,
        "independence": independence_value,
        "inputs": "detector_boxes",
        "score_floor": score_floor,
        "operating_threshold": operating_threshold,
        "iou": iou,
        "max_detections": max_detections,
        "gt_object_count": gt_count,
        "evaluated_frame_count": len(evaluated),
        "excluded_label_fraction": _excluded_fraction(prepared, labelability_known),
        "precision_domain": PRECISION_DOMAIN,
        "recall_clear": _ratio(clear_hit, clear_gt),
        "recall_partial": _ratio(partial_hit, partial_gt),
        "idf1": NOT_EVALUATED,
        "hota": NOT_EVALUATED,
        "track_gt_path": _track_gt_note(track_gt_path),
        "resolved_config_hash": _shown_hash(resolved_config_hash),
        "classes": {name: _public_class(by_class[name]) for name in CLASS_NAMES},
        "targets": _targets(by_class),
    }


def compare_evaluations(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    """Delta of two reports. Mismatched hashes raise and return nothing."""
    left_hash = left.get("dataset_hash")
    right_hash = right.get("dataset_hash")
    if (
        not isinstance(left_hash, str)
        or not isinstance(right_hash, str)
        or left_hash != right_hash
    ):
        raise DatasetHashMismatchError()
    delta: dict[str, dict[str, Any]] = {}
    left_classes = _class_block(left)
    right_classes = _class_block(right)
    for name in CLASS_NAMES:
        row: dict[str, Any] = {}
        for key in ("ap50", "ap50_95", "precision", "recall"):
            row[key] = _numeric_delta(
                _class_value(left_classes, name, key),
                _class_value(right_classes, name, key),
            )
        delta[name] = row
    return {"dataset_hash": left_hash, "delta": delta}


def evaluate_project(
    workspace: Path,
    project_id: str,
    model_id: str,
    split: str,
    *,
    final: bool = False,
    detections_path: Path | None = None,
    compare_to: str | None = None,
    pilot_file: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one frozen split and write ``runs/<eval_id>/metrics.json``.

    Does not train. Does not write the pilot file. A test-split run
    writes ``test_metrics.json`` next to the bundle and does not open
    the bundle's validation metrics as the result.
    """
    _require_split(split)
    if final and split != "test":
        raise NotFinalError("флаг final допустим только вместе со split test")
    pilot = load_pilot(pilot_file)
    project_dir = project_directory(workspace, project_id)
    _require_project_db(workspace, project_id)
    model_dir = _model_directory(workspace, project_id, model_id, final=final)
    dataset_dir, dataset_hash = _dataset_dir(model_dir, project_dir, final=final)
    config_hash = _final_or_hash(model_dir, dataset_dir, final=final)
    independence = _read_independence(dataset_dir / "split_manifest.json")
    mapping = _read_mapping(model_dir / "class_mapping.json")
    eval_id = uuid.uuid4().hex
    if detections_path is not None:
        frames = _frames_from_json(detections_path)
        known = True
    else:
        loaded = _frames_from_coco(dataset_dir, split, mapping)
        if loaded is None:
            report = _unlabeled_test_report(
                eval_id=eval_id,
                project_id=project_id,
                model_id=model_id,
                dataset_hash=dataset_hash,
                independence=independence,
                score_floor=pilot.eval.score_floor,
                operating_threshold=pilot.detection_floor,
                iou=pilot.eval.iou,
                max_detections=pilot.eval.max_detections,
                final=final,
                resolved_config_hash=config_hash,
            )
            _write_json(_run_metrics_path(project_dir, eval_id), report)
            raise TestNotLabeledError(report)
        frames = _attach_predictions(dataset_dir, split, loaded, model_dir)
        known = False
    report = compute_detection_metrics(
        frames,
        class_mapping=mapping,
        score_floor=pilot.eval.score_floor,
        operating_threshold=pilot.detection_floor,
        iou=pilot.eval.iou,
        max_detections=pilot.eval.max_detections,
        dataset_hash=dataset_hash,
        split=split,
        independence=independence,
        labelability_known=known,
        project_id=project_id,
        model_id=model_id,
        eval_id=eval_id,
        final=final,
        resolved_config_hash=config_hash,
    )
    _write_json(_run_metrics_path(project_dir, eval_id), report)
    if split == "test":
        _write_test_metrics(model_dir, report)
    if compare_to:
        other = _read_metrics(project_dir, compare_to)
        report = dict(report)
        report["comparison"] = compare_evaluations(other, report)
    return report


def format_evaluation_summary(report: Mapping[str, Any]) -> str:
    """Russian operator text. Targets stay targets unless they are met."""
    lines = [
        f"Прогон {report.get('eval_id')}",
        f"split: {report.get('split')}",
        f"dataset_hash: {report.get('dataset_hash')}",
        f"independence: {report.get('independence')}",
        f"max_detections: {report.get('max_detections')}",
        f"operating_threshold: {report.get('operating_threshold')}",
        f"score_floor: {report.get('score_floor')}",
        f"GT: {report.get('gt_object_count')}",
        f"Оценённых кадров: {report.get('evaluated_frame_count')}",
        f"Доля исключённых по labelability: {report.get('excluded_label_fraction')}",
        f"recall clear: {report.get('recall_clear')}",
        f"recall partial: {report.get('recall_partial')}",
        f"IDF1: {report.get('idf1')}",
        f"HOTA: {report.get('hota')}",
        "Ориентиры пилота не являются приёмкой.",
    ]
    classes = report.get("classes")
    if isinstance(classes, Mapping):
        for name in CLASS_NAMES:
            row = classes.get(name)
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"{name}: AP50 {row.get('ap50')}, AP50:95 {row.get('ap50_95')}, "
                f"precision {row.get('precision')}, recall {row.get('recall')}"
            )
    targets = report.get("targets")
    if isinstance(targets, Mapping):
        for name, metrics in targets.items():
            if not isinstance(metrics, Mapping):
                continue
            for metric, spec in metrics.items():
                if not isinstance(spec, Mapping):
                    continue
                lines.append(
                    f"{name} {metric}: target {spec.get('target')}, "
                    f"value {spec.get('value')}, {spec.get('status')}"
                )
    comparison = report.get("comparison")
    if isinstance(comparison, Mapping) and "delta" in comparison:
        lines.append(f"dataset_hash сравнения: {comparison.get('dataset_hash')}")
        lines.append("delta записана для совпадающего dataset_hash.")
    return "\n".join(lines)


def _metrics_by_class(
    frames: Sequence[EvalFrame],
    *,
    names: dict[str, int],
    score_floor: float,
    operating_threshold: float,
    iou: float,
    max_detections: int,
) -> dict[str, dict[str, Any]]:
    capped = {
        frame.image_id: _cap_detections(frame.detections, score_floor, max_detections)
        for frame in frames
    }
    ap = _average_precision(frames, capped, names, max_detections)
    rows: dict[str, dict[str, Any]] = {}
    for name in CLASS_NAMES:
        gt = _gt_for_class(frames, name)
        dets = _dets_for_class(capped, name)
        curve = _pr_curve(gt, dets, iou)
        operating = [item for item in dets if item.box.score >= operating_threshold]
        operating.sort(key=lambda item: item.box.score, reverse=True)
        matched = _greedy_matches(gt, operating, iou)
        tp = sum(1 for key in matched if key is not None)
        fp = len(matched) - tp
        clear_gt, clear_hit, partial_gt, partial_hit = _visibility_hits(gt, matched)
        rows[name] = {
            "ap50": ap[name][0],
            "ap50_95": ap[name][1],
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, len(gt)),
            "recall_clear": _ratio(clear_hit, clear_gt),
            "recall_partial": _ratio(partial_hit, partial_gt),
            "gt_count": len(gt),
            "clear_gt": clear_gt,
            "clear_hit": clear_hit,
            "partial_gt": partial_gt,
            "partial_hit": partial_hit,
            "pr_curve": curve,
        }
    return rows


def _cap_detections(
    detections: Sequence[DetectorBox],
    score_floor: float,
    max_detections: int,
) -> tuple[DetectorBox, ...]:
    """Top scores per class at or above the eval floor.

    The cap is not frame-wide: haul-truck boxes do not evict a bucket.
    The operating floor is not applied here.
    """
    kept = [item for item in detections if item.score >= score_floor]
    by_class: dict[str, list[DetectorBox]] = {}
    for item in kept:
        name = item.class_name if isinstance(item.class_name, str) else ""
        by_class.setdefault(name, []).append(item)
    chosen: list[DetectorBox] = []
    for boxes in by_class.values():
        boxes.sort(key=lambda item: item.score, reverse=True)
        chosen.extend(boxes[:max_detections])
    return tuple(chosen)


def _gt_for_class(frames: Sequence[EvalFrame], class_name: str) -> list[_IndexedGt]:
    found: list[_IndexedGt] = []
    for frame in frames:
        for index, box in enumerate(frame.ground_truth):
            if box.class_name != class_name:
                continue
            found.append(_IndexedGt(frame.image_id, index, box))
    return found


def _dets_for_class(
    capped: Mapping[str, Sequence[DetectorBox]],
    class_name: str,
) -> list[_IndexedDet]:
    found: list[_IndexedDet] = []
    for image_id, boxes in capped.items():
        for box in boxes:
            if box.class_name == class_name:
                found.append(_IndexedDet(image_id, box))
    found.sort(key=lambda item: item.box.score, reverse=True)
    return found


@dataclass(frozen=True, slots=True)
class _IndexedGt:
    image_id: str
    index: int
    box: GroundTruthBox


@dataclass(frozen=True, slots=True)
class _IndexedDet:
    image_id: str
    box: DetectorBox


def _pr_curve(
    gt: Sequence[_IndexedGt],
    dets: Sequence[_IndexedDet],
    iou: float,
) -> list[dict[str, Any]]:
    matched = _greedy_matches(gt, dets, iou)
    points: list[dict[str, Any]] = []
    tp = 0
    fp = 0
    total = len(gt)
    for item, key in zip(dets, matched, strict=True):
        if key is not None:
            tp += 1
        else:
            fp += 1
        points.append(
            {
                "score": item.box.score,
                "precision": _ratio(tp, tp + fp),
                "recall": _ratio(tp, total),
                "tp": tp,
                "fp": fp,
            }
        )
    return points


def _greedy_matches(
    gt: Sequence[_IndexedGt],
    dets: Sequence[_IndexedDet],
    iou_threshold: float,
) -> list[tuple[str, int] | None]:
    """Score-desc one-to-one matches. IoU equal to the threshold counts."""
    pools: dict[str, list[_IndexedGt]] = {}
    for item in gt:
        pools.setdefault(item.image_id, []).append(item)
    taken: set[tuple[str, int]] = set()
    matched: list[tuple[str, int] | None] = []
    for det in dets:
        best_iou = iou_threshold
        best_key: tuple[str, int] | None = None
        for candidate in pools.get(det.image_id, []):
            key = (candidate.image_id, candidate.index)
            if key in taken:
                continue
            value = _box_iou(det.box.xyxy, candidate.box.xyxy)
            if value < iou_threshold:
                continue
            if best_key is None or value > best_iou:
                best_iou = value
                best_key = key
        if best_key is None:
            matched.append(None)
            continue
        taken.add(best_key)
        matched.append(best_key)
    return matched


def _visibility_hits(
    gt: Sequence[_IndexedGt],
    matched: Sequence[tuple[str, int] | None],
) -> tuple[int, int, int, int]:
    """Count clear and partial hits from the full-GT matching."""
    hit_keys = {key for key in matched if key is not None}
    clear_gt = sum(1 for item in gt if item.box.visibility == "clear")
    partial_gt = sum(1 for item in gt if item.box.visibility == "partial")
    clear_hit = sum(
        1
        for item in gt
        if item.box.visibility == "clear" and (item.image_id, item.index) in hit_keys
    )
    partial_hit = sum(
        1
        for item in gt
        if item.box.visibility == "partial" and (item.image_id, item.index) in hit_keys
    )
    return clear_gt, clear_hit, partial_gt, partial_hit


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    width = ix2 - ix1
    height = iy2 - iy1
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _average_precision(
    frames: Sequence[EvalFrame],
    capped: Mapping[str, Sequence[DetectorBox]],
    names: Mapping[str, int],
    max_detections: int,
) -> dict[str, tuple[Any, Any]]:
    empty = {name: (NOT_CALCULATED, NOT_CALCULATED) for name in CLASS_NAMES}
    if not frames:
        return empty
    images, annotations, detections = _coco_rows(frames, capped, names)
    if not annotations and not detections:
        return empty
    if not detections:
        found = {annotation["category_id"] for annotation in annotations}
        result = dict(empty)
        for name, class_id in names.items():
            if class_id in found:
                result[name] = (0.0, 0.0)
        return result
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dataset = {
        "info": {"description": "Excavation Video Lab detection eval"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": class_id, "name": name, "supercategory": "none"}
            for name, class_id in sorted(names.items(), key=lambda item: item[1])
        ],
    }
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        coco_gt = COCO()
        coco_gt.dataset = dataset
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        if max_detections == 100:
            evaluator.params.maxDets = [1, 10, 100]
        else:
            evaluator.params.maxDets = [max_detections]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"]
    cat_ids = [int(cat_id) for cat_id in evaluator.params.catIds]
    iou_index = int(np.where(np.isclose(evaluator.params.iouThrs, 0.5))[0][0])
    by_id = {class_id: name for name, class_id in names.items()}
    result = dict(empty)
    for index, cat_id in enumerate(cat_ids):
        name = by_id.get(cat_id)
        if name is None:
            continue
        result[name] = (
            _mean_precision(precision[iou_index, :, index, 0, -1]),
            _mean_precision(precision[:, :, index, 0, -1]),
        )
    return result


def _coco_rows(
    frames: Sequence[EvalFrame],
    capped: Mapping[str, Sequence[DetectorBox]],
    names: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    image_ids = {frame.image_id: index for index, frame in enumerate(frames, start=1)}
    for frame in frames:
        images.append(
            {
                "id": image_ids[frame.image_id],
                "file_name": frame.image_id,
                "width": frame.width,
                "height": frame.height,
            }
        )
        for box in frame.ground_truth:
            x, y, width, height = xyxy_to_xywh(box.xyxy)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_ids[frame.image_id],
                    "category_id": names[box.class_name],
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "ignore": 0,
                }
            )
        for box in capped.get(frame.image_id, ()):
            if box.class_name is None:
                continue
            x, y, width, height = xyxy_to_xywh(box.xyxy)
            detections.append(
                {
                    "image_id": image_ids[frame.image_id],
                    "category_id": names[box.class_name],
                    "bbox": [x, y, width, height],
                    "score": box.score,
                }
            )
    return images, annotations, detections


def _mean_precision(values: Any) -> Any:
    array = np.asarray(values, dtype=np.float64)
    valid = array[array > -1]
    if valid.size == 0:
        return NOT_CALCULATED
    # 101-point AP is a binary mean. 12 digits drops 1-1e-16 noise.
    return round(float(valid.mean()), 12)


def _prepare_frame(frame: EvalFrame, names: Mapping[str, int]) -> EvalFrame:
    if not isinstance(frame.image_id, str) or not frame.image_id:
        raise LabError("invalid_frame", "У кадра нет image_id.")
    if frame.width <= 0 or frame.height <= 0:
        raise LabError("invalid_frame", f"Кадр {frame.image_id}: размер должен быть положительным.")
    gt: list[GroundTruthBox] = []
    unlabelable = False
    for box in frame.ground_truth:
        name = _require_known_name(box.class_name, names, frame.image_id)
        if box.visibility is not None and box.visibility not in _VISIBILITY:
            raise LabError(
                "invalid_visibility",
                f"Кадр {frame.image_id}: неизвестная visibility «{box.visibility}».",
            )
        _require_box(box.xyxy, frame.image_id)
        # Annotation rules exclude the frame. The box is not a false-positive GT.
        if box.visibility == "unlabelable":
            unlabelable = True
            continue
        gt.append(GroundTruthBox(name, box.xyxy, box.visibility))
    detections: list[DetectorBox] = []
    for box in frame.detections:
        _reject_non_detector(box, frame.image_id)
        name = _resolve_detection_name(box, names, frame.image_id)
        score = _require_score(box.score, frame.image_id)
        _require_box(box.xyxy, frame.image_id)
        detections.append(
            DetectorBox(
                score=score,
                xyxy=box.xyxy,
                class_name=name,
                class_id=names[name],
                source="detector",
            )
        )
    if frame.excluded or unlabelable:
        return EvalFrame(frame.image_id, frame.width, frame.height, excluded=True)
    return EvalFrame(
        frame.image_id,
        frame.width,
        frame.height,
        ground_truth=tuple(gt),
        detections=tuple(detections),
        excluded=False,
    )


def _reject_non_detector(box: DetectorBox, image_id: str) -> None:
    source = box.source if isinstance(box.source, str) else "detector"
    if source in _REJECTED_SOURCES or box.track_id is not None:
        raise LabError(
            "not_detector_boxes",
            f"Кадр {image_id}: оценка принимает рамки детектора. "
            "Треки и сглаженные рамки не используются.",
        )


def _resolve_detection_name(
    box: DetectorBox,
    names: Mapping[str, int],
    image_id: str,
) -> str:
    """Name in the mapping wins. The numeric id is not the match key."""
    if isinstance(box.class_name, str) and box.class_name.strip():
        return _require_known_name(box.class_name, names, image_id)
    if isinstance(box.class_id, bool) or not isinstance(box.class_id, int):
        raise UnknownCategoryError(str(box.class_name), image_id)
    matches = [name for name, class_id in names.items() if class_id == box.class_id]
    if len(matches) != 1:
        raise UnknownCategoryError(str(box.class_id), image_id)
    return matches[0]


def _require_known_name(value: object, names: Mapping[str, int], image_id: str) -> str:
    if not isinstance(value, str):
        raise UnknownCategoryError(str(value), image_id)
    name = value.strip()
    if name not in names:
        raise UnknownCategoryError(name, image_id)
    return name


def _require_score(value: object, image_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError("invalid_score", f"Кадр {image_id}: score должен быть числом.")
    score = float(value)
    if not np.isfinite(score) or score < 0 or score > 1:
        raise LabError("invalid_score", f"Кадр {image_id}: score вне диапазона 0..1.")
    return score


def _require_box(xyxy: object, image_id: str) -> None:
    try:
        xyxy_to_xywh(xyxy)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise LabError(
            "invalid_bbox",
            f"Кадр {image_id}: рамка должна иметь положительный размер.",
        ) from exc


def _mapping_names(mapping: Mapping[str, Any]) -> dict[str, int]:
    by_name = mapping.get("by_name") if isinstance(mapping, Mapping) else None
    if isinstance(by_name, (str, bytes)) or not isinstance(by_name, Mapping):
        raise LabError("class_order", "В class_mapping.json нет by_name.")
    found: dict[str, int] = {}
    for name in CLASS_NAMES:
        row = by_name.get(name)
        if isinstance(row, (str, bytes)) or not isinstance(row, Mapping):
            raise LabError("class_order", f"В class_mapping.json нет класса «{name}».")
        class_id = row.get("internal_id")
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise LabError("class_order", f"У класса «{name}» нет internal id.")
        found[name] = class_id
    if set(by_name) != set(CLASS_NAMES):
        raise LabError(
            "class_order",
            "class_mapping.json должен содержать ровно четыре класса.",
        )
    return found


def _public_class(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in (*_METRIC_NAMES, "gt_count", "pr_curve")}


def _pool_visibility(rows: Mapping[str, Mapping[str, Any]]) -> tuple[int, int, int, int]:
    clear_hit = sum(int(row["clear_hit"]) for row in rows.values())
    clear_gt = sum(int(row["clear_gt"]) for row in rows.values())
    partial_hit = sum(int(row["partial_hit"]) for row in rows.values())
    partial_gt = sum(int(row["partial_gt"]) for row in rows.values())
    return clear_hit, clear_gt, partial_hit, partial_gt


def _targets(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    truck = rows["haul_truck"]
    bucket = rows["bucket"]
    return {
        "haul_truck": {
            "precision": _target_row(truck["precision"], _TRUCK_TARGET),
            "recall": _target_row(truck["recall"], _TRUCK_TARGET),
        },
        "bucket": {
            "precision": _target_row(bucket["precision"], _BUCKET_TARGET),
            "recall": _target_row(
                bucket["recall"],
                _BUCKET_TARGET,
                gate=bucket["precision"],
                gate_at=_BUCKET_TARGET,
            ),
        },
    }


def _target_row(
    value: Any,
    threshold: float,
    *,
    gate: Any = None,
    gate_at: float | None = None,
) -> dict[str, Any]:
    """``achieved`` only when the computed value meets the pilot target."""
    status = "target"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        gate_ok = True
        if gate_at is not None:
            gate_ok = (
                isinstance(gate, (int, float))
                and not isinstance(gate, bool)
                and float(gate) >= gate_at
            )
        if gate_ok and float(value) >= threshold:
            status = "achieved"
    return {"target": threshold, "value": value, "status": status}


def _ratio(numerator: int, denominator: int) -> Any:
    if denominator <= 0:
        return NOT_CALCULATED
    return numerator / denominator


def _excluded_fraction(frames: Sequence[EvalFrame], known: bool) -> Any:
    if not known or not frames:
        return NOT_CALCULATED
    excluded = sum(1 for frame in frames if frame.excluded)
    return excluded / len(frames)


def _shown_independence(value: str | None) -> str:
    if value in _INDEPENDENCE:
        return str(value)
    return NOT_CALCULATED


def _shown_hash(value: str | None) -> str:
    if isinstance(value, str) and _HEX.fullmatch(value):
        return value
    return NOT_CALCULATED


def _track_gt_note(path: Path | None) -> str | None:
    """Record a path without scoring it. Detection boxes are not track GT."""
    if path is None:
        return None
    return str(path)


def _unlabeled_test_report(
    *,
    eval_id: str,
    project_id: str,
    model_id: str,
    dataset_hash: str,
    independence: str | None,
    score_floor: float,
    operating_threshold: float,
    iou: float,
    max_detections: int,
    final: bool,
    resolved_config_hash: str | None,
) -> dict[str, Any]:
    blank = {
        "ap50": NOT_CALCULATED,
        "ap50_95": NOT_CALCULATED,
        "precision": NOT_CALCULATED,
        "recall": NOT_CALCULATED,
        "recall_clear": NOT_CALCULATED,
        "recall_partial": NOT_CALCULATED,
        "gt_count": NOT_CALCULATED,
        "pr_curve": [],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_id": eval_id,
        "project_id": project_id,
        "model_id": model_id,
        "split": "test",
        "final": final,
        "status": "test_not_labeled",
        "dataset_hash": dataset_hash,
        "independence": _shown_independence(independence),
        "inputs": "detector_boxes",
        "score_floor": score_floor,
        "operating_threshold": operating_threshold,
        "iou": iou,
        "max_detections": max_detections,
        "gt_object_count": NOT_CALCULATED,
        "evaluated_frame_count": NOT_CALCULATED,
        "excluded_label_fraction": NOT_CALCULATED,
        "precision_domain": PRECISION_DOMAIN,
        "recall_clear": NOT_CALCULATED,
        "recall_partial": NOT_CALCULATED,
        "idf1": NOT_EVALUATED,
        "hota": NOT_EVALUATED,
        "track_gt_path": None,
        "resolved_config_hash": _shown_hash(resolved_config_hash),
        "classes": {name: dict(blank) for name in CLASS_NAMES},
        "targets": {
            "haul_truck": {
                "precision": _target_row(NOT_CALCULATED, _TRUCK_TARGET),
                "recall": _target_row(NOT_CALCULATED, _TRUCK_TARGET),
            },
            "bucket": {
                "precision": _target_row(NOT_CALCULATED, _BUCKET_TARGET),
                "recall": _target_row(NOT_CALCULATED, _BUCKET_TARGET),
            },
        },
    }


def _require_split(split: str) -> None:
    if split not in _SPLITS:
        raise LabError(
            "invalid_split",
            "split должен быть validation или test.",
        )


def _require_hash(value: object) -> None:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise LabError("dataset_hash", "dataset_hash должен быть SHA-256.")


def _class_block(report: Mapping[str, Any]) -> Mapping[str, Any]:
    classes = report.get("classes")
    if isinstance(classes, Mapping):
        return classes
    return {}


def _class_value(classes: Mapping[str, Any], name: str, key: str) -> Any:
    row = classes.get(name)
    if not isinstance(row, Mapping):
        return NOT_CALCULATED
    return row.get(key, NOT_CALCULATED)


def _numeric_delta(left: Any, right: Any) -> Any:
    if isinstance(left, bool) or isinstance(right, bool):
        return NOT_CALCULATED
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(right) - float(left)
    return NOT_CALCULATED


def _require_project_db(workspace: Path, project_id: str) -> None:
    project_sqlite(workspace, project_id)


def _model_directory(
    workspace: Path,
    project_id: str,
    model_id: str,
    *,
    final: bool,
) -> Path:
    if not isinstance(model_id, str) or not model_id or model_id in {".", ".."}:
        return _missing_model(final)
    if "/" in model_id or "\\" in model_id:
        return _missing_model(final)
    candidate = project_directory(workspace, project_id) / "models" / model_id
    if candidate.is_dir() and not candidate.is_symlink():
        return candidate
    recorded = _recorded_model_path(workspace, project_id, model_id)
    if recorded is not None and recorded.is_dir() and not recorded.is_symlink():
        return recorded
    return _missing_model(final)


def _missing_model(final: bool) -> Path:
    if final:
        raise NotFinalError(
            "нет проверенной контрольной суммы bundle; "
            "нет хеша resolved config; "
            "нет зафиксированного манифеста"
        )
    raise LabError(
        "model_not_found",
        "Каталог модели не найден. Базовая COCO-модель не подставляется.",
    )


def _recorded_model_path(workspace: Path, project_id: str, model_id: str) -> Path | None:
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT path FROM model_versions
            WHERE project_id = ? AND model_id = ?
            """,
            (project_id, model_id),
        ).fetchone()
    if row is None or not isinstance(row["path"], str):
        return None
    return Path(row["path"])


def _dataset_dir(model_dir: Path, project_dir: Path, *, final: bool) -> tuple[Path, str]:
    manifest_path = model_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        if final:
            raise NotFinalError(
                "нет проверенной контрольной суммы bundle; "
                "нет хеша resolved config; "
                "нет зафиксированного манифеста"
            )
        raise LabError("dataset_missing", "В bundle нет dataset_manifest.json.")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError("invalid_snapshot", "dataset_manifest.json повреждён.") from exc
    if not isinstance(document, dict):
        raise LabError("invalid_snapshot", "dataset_manifest.json должен быть объектом.")
    dataset_hash = document.get("dataset_hash")
    dataset_id = document.get("dataset_id")
    _require_hash(dataset_hash)
    if not isinstance(dataset_id, str) or not dataset_id or "/" in dataset_id:
        raise LabError("dataset_missing", "В dataset_manifest.json нет dataset_id.")
    directory = project_dir / "dataset_versions" / dataset_id
    if not directory.is_dir():
        if final:
            raise NotFinalError("нет зафиксированного манифеста")
        raise LabError(
            "dataset_missing",
            f"Снимок «{dataset_id}» не найден. Метрики не подменяются.",
        )
    assert isinstance(dataset_hash, str)
    return directory, dataset_hash


def _final_or_hash(model_dir: Path, dataset_dir: Path, *, final: bool) -> str | None:
    if not final:
        digest = _config_hash(model_dir / "resolved_train_config.json")
        return digest
    missing: list[str] = []
    try:
        verify_bundle_checksums(model_dir)
    except BundleChecksumError:
        missing.append("нет проверенной контрольной суммы bundle")
    digest = _config_hash(model_dir / "resolved_train_config.json")
    if digest is None:
        missing.append("нет хеша resolved config")
    if not _manifest_is_frozen(dataset_dir / "split_manifest.json"):
        missing.append("нет зафиксированного манифеста")
    if missing:
        raise NotFinalError("; ".join(missing))
    return digest


def _config_hash(path: Path) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict) or not document:
        return None
    return _sha256(path)


def _manifest_is_frozen(path: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return _frozen_document(document)


def _frozen_document(document: object) -> bool:
    if not isinstance(document, dict):
        return False
    videos = document.get("videos")
    if isinstance(videos, dict) and videos:
        return all(_one_frozen(item) for item in videos.values())
    return _one_frozen(document)


def _one_frozen(document: object) -> bool:
    if not isinstance(document, dict):
        return False
    if document.get("frozen") is False:
        return False
    digest = document.get("content_hash")
    return isinstance(digest, str) and _HEX.fullmatch(digest) is not None


def _read_independence(path: Path) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    direct = document.get("independence")
    if direct in _INDEPENDENCE:
        return str(direct)
    videos = document.get("videos")
    if not isinstance(videos, dict) or not videos:
        return None
    found: set[str] = set()
    for item in videos.values():
        if not isinstance(item, dict):
            return None
        value = item.get("independence")
        if value not in _INDEPENDENCE:
            return None
        found.add(str(value))
    if len(found) != 1:
        return None
    return next(iter(found))


def _read_mapping(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LabError("class_order", "В bundle нет class_mapping.json.")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError("class_order", "class_mapping.json повреждён.") from exc
    if not isinstance(document, dict):
        raise LabError("class_order", "class_mapping.json должен быть объектом.")
    _mapping_names(document)
    return document


def _frames_from_json(path: Path) -> list[EvalFrame]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError("invalid_detections", "Файл детекций не читается как JSON.") from exc
    images = document.get("images") if isinstance(document, dict) else None
    if not isinstance(images, list) or not images:
        raise LabError("invalid_detections", "В файле детекций нет images.")
    frames: list[EvalFrame] = []
    for image in images:
        if not isinstance(image, dict):
            raise LabError("invalid_detections", "Элемент images должен быть объектом.")
        image_id = image.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise LabError("invalid_detections", "У кадра нет image_id.")
        frames.append(
            EvalFrame(
                image_id=image_id,
                width=_positive_int(image.get("width"), "width"),
                height=_positive_int(image.get("height"), "height"),
                ground_truth=tuple(_gt_from_json(item, image_id) for item in _list(image, "ground_truth")),
                detections=tuple(_det_from_json(item, image_id) for item in _list(image, "detections")),
                excluded=image.get("excluded") is True,
            )
        )
    return frames


def _list(image: Mapping[str, Any], key: str) -> list[Any]:
    value = image.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise LabError("invalid_detections", f"Поле {key} должно быть списком.")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LabError("invalid_detections", f"Поле {name} должно быть положительным целым.")
    return value


def _gt_from_json(item: object, image_id: str) -> GroundTruthBox:
    if not isinstance(item, dict):
        raise LabError("invalid_detections", f"Кадр {image_id}: GT должен быть объектом.")
    visibility = item.get("visibility")
    if visibility is not None and not isinstance(visibility, str):
        raise LabError("invalid_visibility", f"Кадр {image_id}: visibility должна быть строкой.")
    return GroundTruthBox(
        class_name=str(item.get("class_name")),
        xyxy=_xyxy(item.get("xyxy"), image_id),
        visibility=visibility,
    )


def _det_from_json(item: object, image_id: str) -> DetectorBox:
    if not isinstance(item, dict):
        raise LabError("invalid_detections", f"Кадр {image_id}: детекция должна быть объектом.")
    class_name = item.get("class_name")
    class_id = item.get("class_id")
    if class_id is not None and (isinstance(class_id, bool) or not isinstance(class_id, int)):
        raise LabError("invalid_detections", f"Кадр {image_id}: class_id должен быть целым.")
    source = item.get("source", "detector")
    if not isinstance(source, str):
        raise LabError("not_detector_boxes", f"Кадр {image_id}: source должен быть строкой.")
    track_id = item.get("track_id")
    if track_id is not None and not isinstance(track_id, str):
        raise LabError("not_detector_boxes", f"Кадр {image_id}: track_id должен быть строкой.")
    score = item.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise LabError("invalid_score", f"Кадр {image_id}: score должен быть числом.")
    return DetectorBox(
        score=float(score),
        xyxy=_xyxy(item.get("xyxy"), image_id),
        class_name=class_name if isinstance(class_name, str) else None,
        class_id=class_id,
        source=source,
        track_id=track_id,
    )


def _xyxy(value: object, image_id: str) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)) or len(value) != 4:
        raise LabError("invalid_bbox", f"Кадр {image_id}: xyxy должен содержать четыре числа.")
    coords: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise LabError("invalid_bbox", f"Кадр {image_id}: xyxy должен содержать четыре числа.")
        coords.append(float(item))
    return (coords[0], coords[1], coords[2], coords[3])


def _frames_from_coco(
    dataset_dir: Path,
    split: str,
    mapping: Mapping[str, Any],
) -> list[EvalFrame] | None:
    """Load one split. A missing test file is unlabeled, not the other split."""
    path = dataset_dir / _SPLIT_DIR[split] / "_annotations.coco.json"
    if split == "test" and (path.is_symlink() or not path.is_file()):
        return None
    if path.is_symlink() or not path.is_file():
        raise LabError(
            "split_not_labeled",
            "В validation нет разметки. Метрики не подставляются из другого split.",
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError("invalid_snapshot", f"{path.name} повреждён.") from exc
    if not isinstance(document, dict):
        raise LabError("invalid_snapshot", f"{path.name} должен быть объектом.")
    images = document.get("images")
    if not isinstance(images, list) or not images:
        if split == "test":
            return None
        raise LabError(
            "split_not_labeled",
            "В validation нет кадров. Метрики не подставляются из другого split.",
        )
    names = _category_names(document.get("categories"))
    by_image: dict[int, list[GroundTruthBox]] = {}
    annotations = document.get("annotations")
    if annotations is None:
        annotations = []
    if not isinstance(annotations, list):
        raise LabError("invalid_snapshot", "annotations должен быть списком.")
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise LabError("invalid_snapshot", "Аннотация должна быть объектом.")
        _reject_crowd_annotation(annotation)
        image_key = annotation.get("image_id")
        if isinstance(image_key, bool) or not isinstance(image_key, int):
            raise LabError("invalid_snapshot", "У аннотации нет image_id.")
        category_id = annotation.get("category_id")
        class_name = names.get(category_id) if not isinstance(category_id, bool) else None
        if class_name is None:
            raise UnknownCategoryError(str(category_id))
        _require_known_name(class_name, _mapping_names(mapping), str(image_key))
        xyxy = _coco_xyxy(annotation.get("bbox"))
        by_image.setdefault(image_key, []).append(
            GroundTruthBox(class_name, xyxy, _visibility(annotation, str(image_key)))
        )
    frames: list[EvalFrame] = []
    for image in images:
        if not isinstance(image, dict):
            raise LabError("invalid_snapshot", "Изображение COCO должно быть объектом.")
        image_key = image.get("id")
        file_name = image.get("file_name")
        if isinstance(image_key, bool) or not isinstance(image_key, int):
            raise LabError("invalid_snapshot", "У изображения нет id.")
        if not isinstance(file_name, str) or not file_name or Path(file_name).name != file_name:
            raise LabError("invalid_snapshot", "У изображения нет file_name.")
        frames.append(
            EvalFrame(
                image_id=file_name,
                width=_positive_int(image.get("width"), "width"),
                height=_positive_int(image.get("height"), "height"),
                ground_truth=tuple(by_image.get(image_key, [])),
            )
        )
    return frames


def _category_names(categories: object) -> dict[object, str]:
    if not isinstance(categories, list):
        raise LabError("invalid_snapshot", "В COCO нет categories.")
    names: dict[object, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            raise LabError("invalid_snapshot", "Категория COCO должна быть объектом.")
        name = category.get("name")
        if not isinstance(name, str) or not name:
            raise LabError("invalid_snapshot", "У категории нет имени.")
        names[category.get("id")] = name
    return names


def _reject_crowd_annotation(annotation: Mapping[str, Any]) -> None:
    iscrowd = annotation.get("iscrowd", 0)
    if iscrowd == 0 and not isinstance(iscrowd, bool):
        return
    shown = iscrowd if isinstance(iscrowd, int) and not isinstance(iscrowd, bool) else None
    raise UnsupportedCrowdError(None, shown)


def _coco_xyxy(bbox: object) -> tuple[float, float, float, float]:
    from excavation_lab.datasets.geometry import xywh_to_xyxy

    try:
        return xywh_to_xyxy(bbox)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise LabError("invalid_bbox", "Рамка COCO не читается.") from exc


def _visibility(annotation: Mapping[str, Any], image_id: str) -> str | None:
    """Missing visibility stays unset. A present unknown value is refused."""
    attributes = annotation.get("attributes")
    if not isinstance(attributes, dict) or "visibility" not in attributes:
        return None
    value = attributes.get("visibility")
    if isinstance(value, str) and value in _VISIBILITY:
        return value
    raise LabError(
        "invalid_visibility",
        f"Кадр {image_id}: неизвестная visibility «{value}».",
    )


def _attach_predictions(
    dataset_dir: Path,
    split: str,
    frames: Sequence[EvalFrame],
    model_dir: Path,
) -> list[EvalFrame]:
    """Run the bundle detector. The operating floor is not passed in."""
    import cv2

    from excavation_lab.models.bundle import load_bundle
    from excavation_lab.models.preprocess import bgr_to_rgb, model_rgb_frame

    folder = dataset_dir / _SPLIT_DIR[split]
    detector = load_bundle(model_dir)
    updated: list[EvalFrame] = []
    try:
        for frame in frames:
            image_path = folder / frame.image_id
            if image_path.is_symlink() or not image_path.is_file():
                raise LabError(
                    "image_missing",
                    f"Нет кадра {frame.image_id}. Рамки не подставляются.",
                )
            bgr = cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise LabError(
                    "image_missing",
                    f"Кадр {frame.image_id} не читается. Рамки не подставляются.",
                )
            rgb = model_rgb_frame(bgr_to_rgb(bgr))
            predicted = detector.predict(rgb, 0.0)
            detections = tuple(
                DetectorBox(
                    score=float(predicted.score[index]),
                    xyxy=(
                        float(predicted.xyxy[index, 0]),
                        float(predicted.xyxy[index, 1]),
                        float(predicted.xyxy[index, 2]),
                        float(predicted.xyxy[index, 3]),
                    ),
                    class_name=predicted.class_name[index],
                    class_id=int(predicted.class_id[index]),
                    source="detector",
                )
                for index in range(len(predicted.class_name))
            )
            updated.append(
                EvalFrame(
                    frame.image_id,
                    frame.width,
                    frame.height,
                    ground_truth=frame.ground_truth,
                    detections=detections,
                    excluded=frame.excluded,
                )
            )
    finally:
        detector.close()
    return updated


def _run_metrics_path(project_dir: Path, eval_id: str) -> Path:
    return project_dir / "runs" / eval_id / "metrics.json"


def _write_test_metrics(model_dir: Path, report: Mapping[str, Any]) -> None:
    """Create ``test_metrics.json``. Leave ``validation_metrics.json`` untouched."""
    destination = model_dir / "test_metrics.json"
    payload = _json_bytes(report)
    try:
        _write_exclusive(destination, payload)
    except FileExistsError as exc:
        raise LabError(
            "metrics_exist",
            "test_metrics.json уже есть. validation_metrics.json не перезаписывается.",
        ) from exc
    _rewrite_checksums(model_dir)


def _rewrite_checksums(directory: Path) -> None:
    files: dict[str, str] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name == CHECKSUMS_NAME:
            continue
        if path.is_symlink() or not path.is_file():
            raise BundleChecksumError(
                f"«{path.name}» не является файлом bundle. Соберите каталог заново.",
            )
        files[path.name] = _sha256(path)
    _write_json(
        directory / CHECKSUMS_NAME,
        {
            "schema_version": SCHEMA_VERSION,
            "algorithm": "sha256",
            "files": files,
        },
    )


def _read_metrics(project_dir: Path, eval_id: str) -> dict[str, Any]:
    path = _run_metrics_path(project_dir, eval_id)
    if not isinstance(eval_id, str) or not eval_id or "/" in eval_id or "\\" in eval_id:
        raise LabError("eval_not_found", "Прогон оценки не найден.")
    if path.is_symlink() or not path.is_file():
        raise LabError("eval_not_found", f"Прогон «{eval_id}» не найден.")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError("invalid_snapshot", "metrics.json повреждён.") from exc
    if not isinstance(document, dict):
        raise LabError("invalid_snapshot", "metrics.json должен быть объектом.")
    return document


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    _write_replace(path, _json_bytes(document))


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_replace(path: Path, payload: bytes) -> None:
    tmp_path: Path | None = None
    try:
        with map_os_error(str(path)):
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=".eval-", suffix=".tmp", dir=path.parent)
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            tmp_path = None
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _write_exclusive(path: Path, payload: bytes) -> None:
    with map_os_error(str(path)):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
