"""Per-class detection quality and a train-only labeling queue.

Reads ``runs/<eval_id>/metrics.json`` from evaluation. A missing file is
``status: not_computed`` and «не рассчитано» for every detection metric.
It does not turn a missing AP into 0 and does not copy pilot targets into
the measured fields.

False positives and false negatives come from ``matches.json`` beside that
report. No match file means the error list is unavailable. Boxes are not
invented. IDF1 and HOTA stay ``not_evaluated`` in v0.1.

``send_to_labeling`` writes an annotation batch only when the frozen split
of that frame is train. Validation and test raise ``not_train_frame``.

A worker does not follow the browser locale.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from excavation_lab.datasets.splits import require_frozen
from excavation_lab.errors import (
    DatasetHashMismatchError,
    LabError,
    SplitNotFrozenError,
    map_os_error,
)
from excavation_lab.models.bundle import CLASS_NAMES
from excavation_lab.models.evaluate import compare_evaluations
from excavation_lab.schemas import SCHEMA_VERSION
from excavation_lab.storage import connect, project_directory, project_sqlite
from excavation_lab.ui import copy as ui_copy

NOT_CALCULATED = ui_copy.catalog("ru").not_calculated
NOT_EVALUATED = ui_copy.NOT_EVALUATED
MATCHES_NAME = "matches.json"
__all__ = [
    "CLASS_METRIC_KEYS",
    "MATCHES_NAME",
    "NOT_CALCULATED",
    "NOT_EVALUATED",
    "DatasetHashMismatchError",
    "compare",
    "load_quality_report",
    "metric_text",
    "send_to_labeling",
]
CLASS_METRIC_KEYS = (
    "ap50",
    "ap50_95",
    "precision",
    "recall",
    "recall_clear",
    "recall_partial",
    "gt_count",
)
_TIMING_KEYS = ("decode_ms", "inference_ms", "encode_ms")
_SETTING_KEYS = ("score_floor", "operating_threshold", "iou", "max_detections")
_ERROR_KINDS = frozenset({"false_positive", "false_negative"})
_KNOWN_SPLITS = frozenset({"train", "validation", "test"})
_TIME_EPS = 1e-6
_IGNORED_FROZEN = frozenset(
    {"video_not_found", "invalid_video_id", "project_not_found", "invalid_project_name"}
)


def load_quality_report(
    workspace: Path,
    project_id: str,
    eval_id: str | None = None,
) -> dict[str, Any]:
    """Read one evaluation. Missing ``metrics.json`` is not a zero score."""
    project_sqlite(workspace, project_id)
    project_dir = project_directory(workspace, project_id)
    document: dict[str, Any] | None = None
    if eval_id is not None:
        _require_safe_eval_id(eval_id)
        document = _read_metrics(project_dir / "runs" / eval_id / "metrics.json")
    errors, errors_status, errors_message = _errors(
        workspace,
        project_id,
        project_dir,
        eval_id,
    )
    return _report(
        project_id,
        eval_id,
        document,
        errors,
        errors_status,
        errors_message,
    )


def compare(
    workspace: Path,
    project_id: str,
    run_a: str,
    run_b: str,
) -> dict[str, Any]:
    """Class deltas when ``dataset_hash`` matches.

    A mismatch raises ``DatasetHashMismatchError`` and does not return a delta.
    """
    project_sqlite(workspace, project_id)
    project_dir = project_directory(workspace, project_id)
    left = _require_metrics(project_dir, run_a)
    right = _require_metrics(project_dir, run_b)
    return compare_evaluations(left, right)


def send_to_labeling(
    workspace: Path,
    project_id: str,
    frame_id: str,
    *,
    eval_id: str | None = None,
) -> dict[str, Any]:
    """Queue one train frame. Validation and test raise ``not_train_frame``.

    The frozen manifest decides the split. A split column on the sample or
    the match file is ignored. The manifest and detections are not rewritten.
    """
    project_sqlite(workspace, project_id)
    video_id, index = _parse_frame_id(frame_id)
    frozen = require_frozen(workspace, project_id, video_id)
    source_time_s, episode_id = _frame_clock(
        project_directory(workspace, project_id),
        video_id,
        index,
        frame_id,
        eval_id,
    )
    split_name = _shown_split(frozen, episode_id, source_time_s)
    if split_name != "train":
        raise LabError("not_train_frame", ui_copy.not_train_frame_message(frame_id, split_name))
    return _create_batch(
        workspace,
        project_id,
        video_id,
        frame_id,
        index,
        source_time_s,
    )


def metric_text(value: object) -> str:
    """Label text. Null is «не рассчитано», not ``0``."""
    if value is None or isinstance(value, bool):
        return NOT_CALCULATED
    if isinstance(value, float):
        if not math.isfinite(value):
            return NOT_CALCULATED
        return format(value, "g")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    return NOT_CALCULATED


def _report(
    project_id: str,
    eval_id: str | None,
    document: Mapping[str, Any] | None,
    errors: list[dict[str, Any]],
    errors_status: str,
    errors_message: str,
) -> dict[str, Any]:
    reviewed = _reviewed(document)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": _status(document),
        "project_id": project_id,
        "eval_id": eval_id,
        "dataset_hash": _text_field(document, "dataset_hash"),
        "split": _split_field(document),
        "independence": _text_field(document, "independence"),
        "reviewed_frame_count": reviewed,
        "evaluated_frame_count": _evaluated(document, reviewed),
        "gt_object_count": _optional_number(document, "gt_object_count"),
        "excluded_label_fraction": _optional_number(document, "excluded_label_fraction"),
        "recall_clear": _optional_number(document, "recall_clear"),
        "recall_partial": _optional_number(document, "recall_partial"),
        "idf1": NOT_EVALUATED,
        "hota": NOT_EVALUATED,
        "classes": _classes(document),
        "targets": _targets(document),
        "target_caption": ui_copy.TARGET_CAPTION,
        "errors": errors,
        "errors_status": errors_status,
        "errors_message": errors_message,
    }
    for key in _TIMING_KEYS:
        report[key] = _timing(document, key)
    for key in _SETTING_KEYS:
        report[key] = _optional_number(document, key)
    return report


def _status(document: Mapping[str, Any] | None) -> str:
    if document is None:
        return "not_computed"
    raw = document.get("status")
    if raw in {"test_not_labeled", "not_computed"}:
        return str(raw)
    return "computed"


def _reviewed(document: Mapping[str, Any] | None) -> object:
    if document is None:
        return NOT_CALCULATED
    if "reviewed_frame_count" in document:
        return _shown_number(document["reviewed_frame_count"])
    if "evaluated_frame_count" in document:
        return _shown_number(document["evaluated_frame_count"])
    return NOT_CALCULATED


def _evaluated(document: Mapping[str, Any] | None, reviewed: object) -> object:
    if document is not None and "evaluated_frame_count" in document:
        return _shown_number(document["evaluated_frame_count"])
    return reviewed


def _classes(document: Mapping[str, Any] | None) -> dict[str, dict[str, object]]:
    raw = document.get("classes") if document is not None else None
    if not isinstance(raw, Mapping):
        raw = {}
    built: dict[str, dict[str, object]] = {}
    for name in CLASS_NAMES:
        row = raw.get(name)
        if not isinstance(row, Mapping):
            row = {}
        built[name] = {key: _shown_number(row.get(key)) for key in CLASS_METRIC_KEYS}
    return built


def _targets(document: Mapping[str, Any] | None) -> dict[str, dict[str, dict[str, object]]]:
    raw = document.get("targets") if document is not None else None
    if not isinstance(raw, Mapping):
        raw = {}
    names = list(dict.fromkeys(("haul_truck", "bucket", *raw.keys())))
    result: dict[str, dict[str, dict[str, object]]] = {}
    for name in names:
        block = raw.get(name)
        if not isinstance(block, Mapping):
            block = {}
        result[str(name)] = {
            "precision": _caption(block.get("precision")),
            "recall": _caption(block.get("recall")),
        }
    return result


def _caption(spec: object) -> dict[str, object]:
    """Pilot target stays a caption unless the stored value meets it.

    A stored ``target`` status is kept when a gate failed, even if the
    value is at least the printed threshold.
    """
    if not isinstance(spec, Mapping):
        return {
            "target": NOT_CALCULATED,
            "value": NOT_CALCULATED,
            "status": "target",
        }
    target = _shown_number(spec.get("target"))
    value = _shown_number(spec.get("value"))
    meets = _is_number(value) and _is_number(target) and float(value) >= float(target)
    if not meets or spec.get("status") == "target":
        status = "target"
    else:
        status = "achieved"
    return {"target": target, "value": value, "status": status}


def _timing(document: Mapping[str, Any] | None, key: str) -> object:
    if document is None:
        return NOT_CALCULATED
    if key in document:
        return _shown_number(document[key])
    nested = document.get("throughput")
    if isinstance(nested, Mapping) and key in nested:
        return _shown_number(nested[key])
    return NOT_CALCULATED


def _optional_number(document: Mapping[str, Any] | None, key: str) -> object:
    if document is None or key not in document:
        return NOT_CALCULATED
    return _shown_number(document[key])


def _text_field(document: Mapping[str, Any] | None, key: str) -> object:
    if document is None:
        return NOT_CALCULATED
    value = document.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return NOT_CALCULATED


def _split_field(document: Mapping[str, Any] | None) -> object:
    if document is None:
        return NOT_CALCULATED
    value = document.get("split")
    if value in _KNOWN_SPLITS:
        return value
    return NOT_CALCULATED


def _shown_number(value: object) -> object:
    if value is None or isinstance(value, bool):
        return NOT_CALCULATED
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return NOT_CALCULATED
        return value
    if value == NOT_CALCULATED:
        return NOT_CALCULATED
    return NOT_CALCULATED


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _errors(
    workspace: Path,
    project_id: str,
    project_dir: Path,
    eval_id: str | None,
) -> tuple[list[dict[str, Any]], str, str]:
    if eval_id is None:
        return [], "unavailable", ui_copy.ERRORS_UNAVAILABLE
    path = project_dir / "runs" / eval_id / MATCHES_NAME
    if path.is_symlink() or not path.is_file():
        return [], "unavailable", ui_copy.ERRORS_UNAVAILABLE
    document = _read_json_value(path, ui_copy.MATCH_BROKEN)
    default_video = document.get("video_id") if isinstance(document, Mapping) else None
    cache: dict[str, object] = {}
    rows = [
        _public_error(workspace, project_id, kind, row, default_video, cache)
        for kind, row in _error_rows(document)
    ]
    rows.sort(key=_error_order)
    return rows, "available", ui_copy.ERRORS_FROM_MATCHES


def _error_order(item: Mapping[str, Any]) -> tuple[float, int, str]:
    return (float(item["source_time_s"]), int(item["source_frame_index"]), str(item["kind"]))


def _error_rows(document: object) -> list[tuple[str, Mapping[str, Any]]]:
    if isinstance(document, list):
        return _kinds_from_rows(document)
    if not isinstance(document, Mapping):
        raise LabError("invalid_match", ui_copy.MATCH_BROKEN)
    if "false_positives" in document or "false_negatives" in document:
        return [
            *_tagged(document.get("false_positives"), "false_positive"),
            *_tagged(document.get("false_negatives"), "false_negative"),
        ]
    for key in ("errors", "matches", "rows"):
        if key not in document:
            continue
        value = document[key]
        if not isinstance(value, list):
            raise LabError("invalid_match", ui_copy.MATCH_BROKEN)
        return _kinds_from_rows(value)
    return []


def _tagged(value: object, kind: str) -> list[tuple[str, Mapping[str, Any]]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise LabError("invalid_match", ui_copy.MATCH_BROKEN)
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise LabError("invalid_match", ui_copy.MATCH_BROKEN)
        rows.append((kind, item))
    return rows


def _kinds_from_rows(items: list[object]) -> list[tuple[str, Mapping[str, Any]]]:
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise LabError("invalid_match", ui_copy.MATCH_BROKEN)
        kind = item.get("kind")
        if kind not in _ERROR_KINDS:
            continue
        rows.append((str(kind), item))
    return rows


def _public_error(
    workspace: Path,
    project_id: str,
    kind: str,
    row: Mapping[str, Any],
    default_video: object,
    cache: dict[str, object],
) -> dict[str, Any]:
    index = _require_index(row)
    stamp = _require_time(row)
    video_id = _video_id(row.get("video_id", default_video))
    frame_id = row.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id:
        frame_id = f"{video_id}:{index}" if video_id else None
    public: dict[str, Any] = {
        "kind": kind,
        "frame_id": frame_id,
        "video_id": video_id,
        "source_frame_index": index,
        "source_time_s": stamp,
        "split": _lookup_split(workspace, project_id, video_id, stamp, cache),
    }
    class_name = row.get("class_name")
    if isinstance(class_name, str) and class_name:
        public["class_name"] = class_name
    if "xyxy" in row and row["xyxy"] is not None:
        public["xyxy"] = _copy_xyxy(row["xyxy"])
    return public


def _lookup_split(
    workspace: Path,
    project_id: str,
    video_id: str | None,
    source_time_s: float,
    cache: dict[str, object],
) -> str | None:
    if video_id is None:
        return None
    if video_id not in cache:
        cache[video_id] = _load_frozen_or_none(workspace, project_id, video_id)
    frozen = cache[video_id]
    if frozen is None:
        return None
    role = _assignment_at(frozen, source_time_s)
    if isinstance(role, str):
        return role
    return None


def _load_frozen_or_none(workspace: Path, project_id: str, video_id: str) -> object:
    try:
        return require_frozen(workspace, project_id, video_id)
    except SplitNotFrozenError:
        return None
    except LabError as exc:
        if exc.code in _IGNORED_FROZEN:
            return None
        raise


def _frame_clock(
    project_dir: Path,
    video_id: str,
    index: int,
    frame_id: str,
    eval_id: str | None,
) -> tuple[float, str | None]:
    sample = _clock_from_sample(project_dir, video_id, index, frame_id)
    if sample is not None:
        return sample
    matched = _clock_from_matches(project_dir, video_id, index, frame_id, eval_id)
    if matched is not None:
        return matched
    raise LabError("frame_not_found", ui_copy.frame_not_found_message(frame_id))


def _clock_from_sample(
    project_dir: Path,
    video_id: str,
    index: int,
    frame_id: str,
) -> tuple[float, str | None] | None:
    path = project_dir / "samples" / video_id / "sample_manifest.json"
    if path.is_symlink() or not path.is_file():
        return None
    document = _read_json_value(path, ui_copy.SAMPLE_BROKEN)
    if not isinstance(document, Mapping):
        raise LabError("sample_missing", ui_copy.SAMPLE_BROKEN)
    rows = document.get("rows", [])
    if not isinstance(rows, list):
        raise LabError("sample_missing", ui_copy.SAMPLE_BROKEN)
    return _unique_clock(_clocks(rows, video_id, index), frame_id)


def _clock_from_matches(
    project_dir: Path,
    video_id: str,
    index: int,
    frame_id: str,
    eval_id: str | None,
) -> tuple[float, str | None] | None:
    found: list[tuple[float, str | None]] = []
    for path in _match_paths(project_dir, eval_id):
        document = _read_json_value(path, ui_copy.MATCH_BROKEN)
        default_video = document.get("video_id") if isinstance(document, Mapping) else None
        found.extend(_clocks(_match_items(document), video_id, index, default_video))
    return _unique_clock(found, frame_id)


def _match_paths(project_dir: Path, eval_id: str | None) -> Iterator[Path]:
    runs = project_dir / "runs"
    if eval_id is not None:
        _require_safe_eval_id(eval_id)
        path = runs / eval_id / MATCHES_NAME
        if _is_regular_file(path):
            yield path
        return
    if runs.is_symlink() or not runs.is_dir():
        return
    for child in sorted(runs.iterdir(), key=lambda item: item.name):
        if child.is_symlink() or not child.is_dir():
            continue
        path = child / MATCHES_NAME
        if _is_regular_file(path):
            yield path


def _match_items(document: object) -> list[object]:
    if isinstance(document, list):
        return document
    if not isinstance(document, Mapping):
        raise LabError("invalid_match", ui_copy.MATCH_BROKEN)
    items: list[object] = []
    for key in ("false_positives", "false_negatives", "errors", "matches", "rows"):
        value = document.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            raise LabError("invalid_match", ui_copy.MATCH_BROKEN)
        items.extend(value)
    return items


def _clocks(
    items: list[object],
    video_id: str,
    index: int,
    default_video: object = None,
) -> list[tuple[float, str | None]]:
    found: list[tuple[float, str | None]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        fallback = video_id if default_video is None else default_video
        row_video = _video_id(item.get("video_id", fallback))
        row_index = item.get("source_frame_index")
        if row_video != video_id or isinstance(row_index, bool) or row_index != index:
            continue
        if "source_time_s" not in item:
            continue
        episode = item.get("episode_id")
        if not isinstance(episode, str) or not episode.strip():
            episode = None
        found.append((_require_time(item), episode))
    return found


def _unique_clock(
    found: list[tuple[float, str | None]],
    frame_id: str,
) -> tuple[float, str | None] | None:
    if not found:
        return None
    stamp = found[0][0]
    if any(abs(item[0] - stamp) > _TIME_EPS for item in found[1:]):
        raise LabError("frame_not_found", ui_copy.frame_ambiguous_message(frame_id))
    episodes = {episode for _, episode in found if episode}
    if len(episodes) > 1:
        raise LabError("frame_not_found", ui_copy.frame_ambiguous_message(frame_id))
    episode = next(iter(episodes)) if episodes else None
    return stamp, episode


def _assignment_at(frozen: object, time_s: float) -> str | None:
    """Split at ``time_s`` from the frozen episodes, guards, and blocks."""
    splits = {
        str(episode.split.value)
        for episode in frozen.episodes
        if episode.start_s <= time_s < episode.end_s
    }
    if len(splits) > 1:
        raise LabError("episode_overlap", ui_copy.EPISODE_OVERLAP)
    if len(splits) == 1:
        return next(iter(splits))
    for guard in frozen.guards:
        if guard.start_s <= time_s < guard.end_s:
            return str(guard.role)
    for block in frozen.blocks:
        if block.start_s <= time_s < block.end_s:
            return str(block.role)
    return None


def _shown_split(frozen: object, episode_id: str | None, source_time_s: float) -> str:
    """Train only when the frozen time says train and the episode does not contradict it.

    A missing time role stays unknown. ``episode_id`` does not fill that gap,
    and a named episode from another split blocks train.
    """
    role = _assignment_at(frozen, source_time_s)
    episode_role = _episode_role(frozen, episode_id)
    if role == "train" and episode_role in {None, "train"}:
        return "train"
    if isinstance(role, str) and role != "train":
        return role
    if role == "train":
        return episode_role if episode_role not in {None, "train"} else "unknown"
    if episode_role not in {None, "train"}:
        return episode_role
    return "unknown"


def _episode_role(frozen: object, episode_id: str | None) -> str | None:
    if not episode_id:
        return None
    matches = [
        episode for episode in frozen.episodes if episode.episode_id == episode_id
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise LabError("episode_not_found", ui_copy.episode_ambiguous_message(episode_id))
    return str(matches[0].split.value)


def _create_batch(
    workspace: Path,
    project_id: str,
    video_id: str,
    frame_id: str,
    index: int,
    source_time_s: float,
) -> dict[str, Any]:
    batch_id = f"quality__{video_id}__{index}"
    project_dir = project_directory(workspace, project_id)
    directory = project_dir / "annotation_batches" / batch_id
    resolved_dir = directory.resolve()
    if not resolved_dir.is_relative_to(project_dir.resolve()):
        raise LabError("invalid_batch_id", ui_copy.INVALID_BATCH)
    document = {
        "schema_version": SCHEMA_VERSION,
        "kind": "annotation_batch",
        "purpose": "quality_relabel",
        "batch_id": batch_id,
        "project_id": project_id,
        "video_id": video_id,
        "frame_id": frame_id,
        "source_frame_index": index,
        "source_time_s": source_time_s,
        "split": "train",
        "predictions_included": False,
        "ground_truth": False,
    }
    destination = directory / "batch.json"
    _write_json(destination, document)
    stored = destination.resolve()
    _record_batch(workspace, project_id, batch_id, video_id, stored)
    return {
        "batch_id": batch_id,
        "project_id": project_id,
        "video_id": video_id,
        "frame_id": frame_id,
        "source_frame_index": index,
        "source_time_s": source_time_s,
        "split": "train",
        "status": "queued",
        "path": str(stored),
    }


def _record_batch(
    workspace: Path,
    project_id: str,
    batch_id: str,
    video_id: str,
    destination: Path,
) -> None:
    created_at = datetime.now(UTC).isoformat()
    try:
        with connect(project_sqlite(workspace, project_id)) as connection:
            connection.execute(
                """
                INSERT INTO annotation_batches (
                    batch_id, project_id, video_id, status, path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    video_id = excluded.video_id,
                    status = excluded.status,
                    path = excluded.path
                """,
                (batch_id, project_id, video_id, "queued", str(destination), created_at),
            )
    except sqlite3.IntegrityError as exc:
        destination.unlink(missing_ok=True)
        raise LabError("video_not_found", "Видео не найдено в этом проекте.") from exc


def _require_metrics(project_dir: Path, eval_id: str) -> dict[str, Any]:
    _require_safe_eval_id(eval_id)
    document = _read_metrics(project_dir / "runs" / eval_id / "metrics.json")
    if document is None:
        raise LabError("eval_not_found", ui_copy.eval_not_found_message(eval_id))
    return document


def _read_metrics(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError("invalid_snapshot", ui_copy.METRICS_BROKEN) from exc
    if not isinstance(document, dict):
        raise LabError("invalid_snapshot", ui_copy.METRICS_OBJECT)
    return document


def _read_json_value(path: Path, broken: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError("invalid_match", broken) from exc


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    tmp_path: Path | None = None
    try:
        with map_os_error(str(path)):
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=".quality-", suffix=".tmp", dir=path.parent)
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _require_safe_eval_id(eval_id: object) -> None:
    if (
        isinstance(eval_id, str)
        and eval_id
        and eval_id not in {".", ".."}
        and "/" not in eval_id
        and "\\" not in eval_id
        and "\x00" not in eval_id
    ):
        return
    shown = eval_id if isinstance(eval_id, str) else ""
    raise LabError("eval_not_found", ui_copy.eval_not_found_message(shown))


def _parse_frame_id(frame_id: str) -> tuple[str, int]:
    if not isinstance(frame_id, str) or ":" not in frame_id:
        raise LabError("frame_not_found", ui_copy.INVALID_FRAME)
    video_id, _, index_text = frame_id.rpartition(":")
    if not _video_id(video_id) or not index_text.isdigit():
        raise LabError("frame_not_found", ui_copy.INVALID_FRAME)
    return video_id, int(index_text)


def _video_id(value: object) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        return None
    return value


def _require_index(row: Mapping[str, Any]) -> int:
    value = row.get("source_frame_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LabError("invalid_match", ui_copy.MATCH_BROKEN)
    return value


def _require_time(row: Mapping[str, Any]) -> float:
    value = row.get("source_time_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError("invalid_match", ui_copy.MATCH_BROKEN)
    stamp = float(value)
    if not math.isfinite(stamp) or stamp < 0:
        raise LabError("invalid_match", ui_copy.MATCH_BROKEN)
    return stamp


def _copy_xyxy(value: object) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise LabError("invalid_match", ui_copy.MATCH_BROKEN)
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise LabError("invalid_match", ui_copy.MATCH_BROKEN)
        number = float(item)
        if not math.isfinite(number):
            raise LabError("invalid_match", ui_copy.MATCH_BROKEN)
        numbers.append(number)
    return numbers


def _is_regular_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()
