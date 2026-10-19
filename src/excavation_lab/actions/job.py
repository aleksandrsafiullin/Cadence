"""Queue and run action training on one or more custom analysis runs.

Test-block frames are dropped inside ``build_features`` before a window
exists. The standardizer is fit later, on train windows only. The action
net stays on CPU. A cancel after an epoch publishes no bundle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from excavation_lab.actions.catalog import ActionCatalog, list_actions
from excavation_lab.actions.dataset import class_index, dataset_hash
from excavation_lab.actions.features import FeatureSpec, build_features, concat_runs
from excavation_lab.actions.labels import ActionInterval, list_intervals, role_of
from excavation_lab.actions.train import ActionTrainConfig, train_action_windows
from excavation_lab.datasets.blocks import RoleBlock, validate_role_blocks
from excavation_lab.datasets.catalog import require_project_catalog
from excavation_lab.datasets.splits import (
    FrozenSplit,
    canonical_json_bytes,
    require_frozen,
)
from excavation_lab.errors import LabError, SplitNotFrozenError
from excavation_lab.jobs import (
    get_job,
    mark_completed,
    mark_failed,
    start_job,
    worker_job,
    write_progress,
)
from excavation_lab.schemas import SCHEMA_VERSION, JobStatus
from excavation_lab.storage import (
    connect,
    get_video,
    insert_action_model,
    list_analysis_runs,
    list_videos,
    project_directory,
    project_sqlite,
)
from excavation_lab.tracking.analyze import run_tables

REQUEST_NAME = "action_train_request.json"
KEPT_RANGES_NAME = "kept_time_ranges.json"
BUNDLE_NAME = "action_bundle.json"
METRICS_NAME = "metrics_history.csv"
LOG_NAME = "action_train.log"
_KEPT_ROLES = frozenset({"train", "validation"})
_BUNDLE_FILES = (
    "weights.pt",
    "standardizer.json",
    "feature_spec.json",
    BUNDLE_NAME,
    METRICS_NAME,
    "validation_metrics.json",
    KEPT_RANGES_NAME,
    "checksums.json",
)
_NO_RUNS = "Не переданы прогоны анализа."
_EMPTY_CATALOG = "Каталог действий пуст."
_NO_TRAIN = "Нет размеченного интервала роли train."
_NO_VALIDATION = "Нет размеченного интервала роли validation."
_BLOCKS_OVERLAP = (
    "Блоки разных видео пересекаются по времени. "
    "Окна действий строятся на одной шкале."
)


@dataclass(slots=True)
class Readiness:
    """Whether the given runs may train an action model. Reasons are verbatim."""

    ok: bool
    reasons: list[str]
    detector_model_id: str | None
    analysis_fps: float | None


@dataclass(frozen=True, slots=True)
class _Span:
    run_id: str
    video_id: str
    model_id: str
    analysis_fps: float
    start_s: float
    end_s: float
    run_dir: Path


def check_ready(
    workspace: Path,
    project_id: str,
    run_ids: Sequence[str],
) -> Readiness:
    """Gate for ``enqueue_action_train``. Every reason is safe to print as-is."""
    reasons: list[str] = []
    ids = _run_ids(run_ids, reasons)
    if not ids and reasons:
        return Readiness(False, reasons, None, None)
    listed = {row["run_id"]: row for row in list_analysis_runs(workspace, project_id)}
    present: list[Mapping[str, Any]] = []
    usable: list[_Span] = []
    seen: set[str] = set()
    for run_id in ids:
        if run_id in seen:
            reasons.append(f"Прогон «{run_id}» указан повторно.")
            continue
        seen.add(run_id)
        row = listed.get(run_id)
        if row is None:
            reasons.append(f"Прогон «{run_id}» не найден.")
            continue
        present.append(row)
        if row["model_origin"] != "custom":
            reasons.append(_origin_reason(run_id, str(row["model_origin"])))
            continue
        if not row["completed"]:
            reasons.append(f"Прогон «{run_id}» не завершён.")
            continue
        try:
            run_tables(Path(row["run_dir"]))
        except LabError:
            reasons.append(f"Прогон «{run_id}» неполный: нет таблиц прогона.")
            continue
        span = _span(workspace, project_id, row, reasons)
        if span is not None:
            usable.append(span)
    _disagree(present, "model_id", reasons)
    _disagree(present, "analysis_fps", reasons)
    _overlaps(usable, reasons)
    frozen = _frozen_by_video(workspace, project_id, _videos(workspace, project_id, usable), reasons)
    _blocks_one_clock(frozen, reasons)
    classes = list_actions(workspace, project_id)
    if not classes:
        reasons.append(_EMPTY_CATALOG)
    train_rows, validation_rows = _role_intervals(workspace, project_id, frozen)
    if not train_rows:
        reasons.append(_NO_TRAIN)
    if not validation_rows:
        reasons.append(_NO_VALIDATION)
    _uncovered(train_rows + validation_rows, usable, reasons)
    model_id, fps = _shared(usable)
    return Readiness(not reasons, reasons, model_id, fps)


def enqueue_action_train(
    workspace: Path,
    project_id: str,
    run_ids: list[str],
    config: ActionTrainConfig,
) -> Any:
    """Queue ``action_train``. The worker loads no detector. CPU only."""
    if not isinstance(config, ActionTrainConfig):
        raise LabError("invalid_action_train", "Нужна конфигурация обучения действий.")
    settings = config.model_copy(update={"device": "cpu"})
    readiness = check_ready(workspace, project_id, run_ids)
    if not readiness.ok:
        raise LabError("action_not_ready", "\n".join(readiness.reasons))
    ids = list(run_ids)

    def write_request(run_dir: Path) -> None:
        _atomic_json(
            run_dir / REQUEST_NAME,
            {"run_ids": ids, "config": settings.model_dump(mode="json")},
        )

    return start_job(
        workspace,
        project_id,
        kind="action_train",
        worker="action_train",
        on_run_dir=write_request,
    )


def run_action_train_job(workspace: Path, job_id: str) -> int:
    """Child entry. ``completed`` is set only after the bundle files exist."""
    root = Path(workspace).resolve()
    project_id, run_dir, record = worker_job(root, job_id)
    if record.status in {JobStatus.cancelled, JobStatus.cancel_requested}:
        return 0
    try:
        return _execute(root, project_id, job_id, run_dir)
    except LabError as exc:
        _fail(root, project_id, job_id, run_dir, exc.code)
        return 1
    except Exception as exc:  # noqa: BLE001 — the job row keeps the failure
        _fail(root, project_id, job_id, run_dir, _exception_text(exc))
        return 1


def _execute(workspace: Path, project_id: str, job_id: str, run_dir: Path) -> int:
    if _cancelled(workspace, project_id, job_id):
        return 0
    run_ids, config = _read_request(run_dir)
    readiness = check_ready(workspace, project_id, run_ids)
    if not readiness.ok or readiness.detector_model_id is None or readiness.analysis_fps is None:
        raise LabError("action_not_ready", "\n".join(readiness.reasons) or _NO_RUNS)
    by_id = {row["run_id"]: row for row in list_analysis_runs(workspace, project_id)}
    spans = [_require_span(workspace, project_id, by_id[run_id]) for run_id in run_ids]
    frozen = {
        video_id: require_frozen(workspace, project_id, video_id)
        for video_id in {span.video_id for span in spans}
    }
    spec = _feature_spec(workspace, project_id, readiness.analysis_fps)
    frames = []
    kept: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        split = frozen[span.video_id]
        keep = _keep_ranges(split.blocks)
        frame = build_features(
            run_tables(span.run_dir),
            run_id=span.run_id,
            run_index=index,
            keep_ranges=keep,
            spec=spec,
        )
        frames.append(frame)
        kept.extend(_intersected(span, split.blocks))
        _log_dropped(run_dir, span, split.blocks)
    _atomic_json(run_dir / KEPT_RANGES_NAME, kept)
    stacked, spec = concat_runs(frames, [spec] * len(frames))
    _log(run_dir, f"device=cpu frames={len(stacked)}")
    catalog = ActionCatalog(classes=list_actions(workspace, project_id))
    names = _class_order(catalog)
    blocks = _blocks(frozen)
    intervals = _train_validation_intervals(workspace, project_id, frozen)
    blocks_hash = _blocks_hash(frozen)
    digest = dataset_hash(spec, intervals, blocks_hash, config.window, catalog)
    from excavation_lab.actions import dataset as windows

    built = [
        windows.build_windows(
            stacked,
            spec,
            intervals,
            blocks,
            role,
            config.window,
            catalog=catalog,
        )
        for role in ("train", "validation")
    ]
    if _cancelled(workspace, project_id, job_id):
        return 0
    write_progress(
        run_dir,
        {
            "job_id": job_id,
            "status": JobStatus.running.value,
            "epoch": 0,
            "epochs": config.epochs,
            "val_macro_f1": None,
        },
    )

    def on_epoch(row: Mapping[str, Any]) -> None:
        write_progress(
            run_dir,
            {
                "job_id": job_id,
                "status": JobStatus.running.value,
                "epoch": row["epoch"],
                "epochs": config.epochs,
                "val_macro_f1": row["val_macro_f1"],
            },
        )
        _append_metrics(run_dir / METRICS_NAME, row)

    def should_stop() -> bool:
        return _cancelled(workspace, project_id, job_id)

    outcome = train_action_windows(
        built,
        names,
        config,
        on_epoch=on_epoch,
        should_stop=should_stop,
        feature_spec=spec,
    )
    if should_stop():
        return 0
    bundle = project_directory(workspace, project_id) / "action_models" / job_id
    bundle.mkdir(parents=True, exist_ok=False)
    artifacts = _publish_bundle(
        bundle,
        run_dir,
        outcome=outcome,
        spec=spec,
        config=config,
        catalog=catalog,
        names=names,
        run_ids=run_ids,
        detector_model_id=readiness.detector_model_id,
        analysis_fps=readiness.analysis_fps,
        dataset_hash_value=digest,
        window_frames=int(built[0].window_frames),
        action_model_id=job_id,
    )
    if _cancelled(workspace, project_id, job_id):
        return 0
    insert_action_model(
        workspace,
        project_id,
        action_model_id=job_id,
        detector_model_id=readiness.detector_model_id,
        source_run_ids=run_ids,
        path=bundle,
    )
    if _cancelled(workspace, project_id, job_id):
        _drop_action_model(workspace, project_id, job_id)
        return 0
    try:
        mark_completed(workspace, project_id, job_id, artifacts)
    except LabError:
        if not _cancelled(workspace, project_id, job_id):
            raise
        _drop_action_model(workspace, project_id, job_id)
        return 0
    last = outcome.metrics_history[-1]
    write_progress(
        run_dir,
        {
            "job_id": job_id,
            "status": JobStatus.completed.value,
            "epoch": last.epoch,
            "epochs": config.epochs,
            "val_macro_f1": last.val_macro_f1,
        },
    )
    return 0


def _publish_bundle(
    bundle: Path,
    run_dir: Path,
    *,
    outcome: Any,
    spec: FeatureSpec,
    config: ActionTrainConfig,
    catalog: ActionCatalog,
    names: Sequence[str],
    run_ids: Sequence[str],
    detector_model_id: str,
    analysis_fps: float,
    dataset_hash_value: str,
    window_frames: int,
    action_model_id: str,
) -> list[Path]:
    import torch

    _write_weights(bundle / "weights.pt", outcome.state_dict)
    _atomic_json(bundle / "standardizer.json", outcome.standardizer.to_json())
    spec_document = spec.model_dump(mode="json")
    _atomic_json(bundle / "feature_spec.json", spec_document)
    metrics = outcome.validation_metrics
    history_path = bundle / METRICS_NAME
    if (run_dir / METRICS_NAME).is_file():
        shutil.copyfile(run_dir / METRICS_NAME, history_path)
    else:
        _write_history(history_path, outcome.metrics_history)
    _atomic_json(bundle / "validation_metrics.json", metrics.to_dict())
    shutil.copyfile(run_dir / KEPT_RANGES_NAME, bundle / KEPT_RANGES_NAME)
    window = config.window.model_dump(mode="json")
    window["window_frames"] = window_frames
    document = {
        "action_model_id": action_model_id,
        "detector_model_id": detector_model_id,
        "source_run_ids": list(run_ids),
        "analysis_fps": analysis_fps,
        "window_frames": window_frames,
        "window_s": config.window.window_s,
        "action_catalog": [
            {"id": item.id, "name": item.name, "title": item.title}
            for item in sorted(catalog.classes, key=lambda item: item.id)
        ],
        "class_order": list(names),
        "classes_scored": list(metrics.classes_scored),
        "feature_spec": spec_document,
        "feature_spec_hash": hashlib.sha256(canonical_json_bytes(spec_document)).hexdigest(),
        "dataset_hash": dataset_hash_value,
        "support_train": dict(metrics.support_train),
        "support_validation": dict(metrics.support_validation),
        "selection_unstable": metrics.selection_unstable,
        "window": window,
        "config": config.model_dump(mode="json"),
        "test_read": False,
        "library_versions": {"torch": torch.__version__},
        "created_at": datetime.now(UTC).isoformat(),
    }
    _atomic_json(bundle / BUNDLE_NAME, document)
    checksums = _write_checksums(bundle)
    paths = [bundle / name for name in _BUNDLE_FILES]
    if checksums not in paths:
        paths.append(checksums)
    return paths


def _run_ids(run_ids: Sequence[str], reasons: list[str]) -> list[str]:
    if isinstance(run_ids, (str, bytes)) or not isinstance(run_ids, Sequence) or not run_ids:
        reasons.append(_NO_RUNS)
        return []
    ids: list[str] = []
    for item in run_ids:
        if not isinstance(item, str) or not item or "\x00" in item:
            reasons.append("Недопустимый идентификатор прогона.")
            continue
        ids.append(item)
    if not ids and not reasons:
        reasons.append(_NO_RUNS)
    return ids


def _origin_reason(run_id: str, origin: str) -> str:
    if "baseline" in origin:
        return f"Прогон «{run_id}» использует baseline («{origin}»), а не custom."
    return f"Прогон «{run_id}» имеет источник «{origin}», а не custom."


def _span(
    workspace: Path,
    project_id: str,
    row: Mapping[str, Any],
    reasons: list[str],
) -> _Span | None:
    end_s = row["end_s"]
    if end_s is None:
        duration = get_video(workspace, project_id, str(row["video_id"]))["duration_s"]
        if duration is None:
            reasons.append(f"У прогона «{row['run_id']}» нет конца интервала.")
            return None
        end_s = float(duration)
    start_s = float(row["start_s"])
    end = float(end_s)
    if not end > start_s:
        reasons.append(f"У прогона «{row['run_id']}» нет конца интервала.")
        return None
    return _Span(
        run_id=str(row["run_id"]),
        video_id=str(row["video_id"]),
        model_id=str(row["model_id"]),
        analysis_fps=float(row["analysis_fps"]),
        start_s=start_s,
        end_s=end,
        run_dir=Path(row["run_dir"]),
    )


def _require_span(workspace: Path, project_id: str, row: Mapping[str, Any]) -> _Span:
    reasons: list[str] = []
    span = _span(workspace, project_id, row, reasons)
    if span is None:
        raise LabError("action_not_ready", reasons[0] if reasons else _NO_RUNS)
    return span


def _disagree(rows: Sequence[Mapping[str, Any]], field: str, reasons: list[str]) -> None:
    if len(rows) < 2:
        return
    values = [row[field] for row in rows]
    if field == "analysis_fps":
        numbers = [float(value) for value in values]
        if max(numbers) - min(numbers) <= 1e-9:
            return
        listed = ", ".join(
            f"«{row['run_id']}» — {_num(float(row[field]))}" for row in rows
        )
        reasons.append(f"Разный analysis_fps: {listed}.")
        return
    if len(set(values)) <= 1:
        return
    listed = ", ".join(f"«{row['run_id']}» — {row[field]}" for row in rows)
    reasons.append(f"Разный {field}: {listed}.")


def _overlaps(spans: Sequence[_Span], reasons: list[str]) -> None:
    for index, earlier in enumerate(spans):
        for later in spans[index + 1 :]:
            if earlier.video_id != later.video_id:
                continue
            if earlier.start_s < later.end_s and later.start_s < earlier.end_s:
                reasons.append(
                    f"Прогоны «{earlier.run_id}» и «{later.run_id}» пересекаются на видео «{earlier.video_id}»: "
                    f"[{_num(earlier.start_s)}, {_num(earlier.end_s)}) и [{_num(later.start_s)}, {_num(later.end_s)})."
                )


def _videos(workspace: Path, project_id: str, spans: Sequence[_Span]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for span in spans:
        if span.video_id not in seen:
            seen.add(span.video_id)
            ordered.append(span.video_id)
    for video in list_videos(workspace, project_id):
        video_id = str(video["video_id"])
        if video_id in seen:
            continue
        if list_intervals(workspace, project_id, video_id):
            seen.add(video_id)
            ordered.append(video_id)
    return ordered


def _frozen_by_video(
    workspace: Path,
    project_id: str,
    video_ids: Sequence[str],
    reasons: list[str],
) -> dict[str, FrozenSplit]:
    frozen: dict[str, FrozenSplit] = {}
    for video_id in video_ids:
        try:
            frozen[video_id] = require_frozen(workspace, project_id, video_id)
        except SplitNotFrozenError:
            reasons.append(f"Разбиение видео «{video_id}» не зафиксировано.")
        except LabError as exc:
            reasons.append(exc.message)
    return frozen


def _blocks_one_clock(frozen: Mapping[str, FrozenSplit], reasons: list[str]) -> None:
    if len(frozen) < 2:
        return
    blocks = [block for split in frozen.values() for block in split.blocks]
    try:
        validate_role_blocks(blocks)
    except LabError:
        reasons.append(_BLOCKS_OVERLAP)


def _role_intervals(
    workspace: Path,
    project_id: str,
    frozen: Mapping[str, FrozenSplit],
) -> tuple[list[ActionInterval], list[ActionInterval]]:
    train_rows: list[ActionInterval] = []
    validation_rows: list[ActionInterval] = []
    for video_id, split in frozen.items():
        for interval in list_intervals(workspace, project_id, video_id):
            role = role_of(interval, split.blocks)
            if role == "train":
                train_rows.append(interval)
            elif role == "validation":
                validation_rows.append(interval)
    return train_rows, validation_rows


def _uncovered(
    intervals: Sequence[ActionInterval],
    spans: Sequence[_Span],
    reasons: list[str],
) -> None:
    by_video: dict[str, list[tuple[float, float]]] = {}
    for span in spans:
        by_video.setdefault(span.video_id, []).append((span.start_s, span.end_s))
    merged = {video_id: _merge(ranges) for video_id, ranges in by_video.items()}
    for interval in intervals:
        if _covers(interval.start_s, interval.end_s, merged.get(interval.video_id, [])):
            continue
        reasons.append(
            f"Интервал [{_num(interval.start_s)}, {_num(interval.end_s)}) «{interval.action_name}» на видео «{interval.video_id}» не покрыт прогонами."
        )


def _covers(start_s: float, end_s: float, spans: Sequence[tuple[float, float]]) -> bool:
    cursor = start_s
    for span_start, span_end in spans:
        if span_end <= cursor + 1e-9:
            continue
        if span_start > cursor + 1e-6:
            return False
        cursor = max(cursor, span_end)
        if cursor + 1e-9 >= end_s:
            return True
    return cursor + 1e-9 >= end_s


def _merge(spans: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start_s, end_s in sorted(spans):
        if not merged or start_s > merged[-1][1]:
            merged.append([start_s, end_s])
        else:
            merged[-1][1] = max(merged[-1][1], end_s)
    return [(item[0], item[1]) for item in merged]


def _shared(spans: Sequence[_Span]) -> tuple[str | None, float | None]:
    if not spans:
        return None, None
    model_ids = {span.model_id for span in spans}
    rates = {span.analysis_fps for span in spans}
    if len(model_ids) != 1 or max(rates) - min(rates) > 1e-9:
        return None, None
    return spans[0].model_id, spans[0].analysis_fps


def _feature_spec(workspace: Path, project_id: str, analysis_fps: float) -> FeatureSpec:
    catalog = require_project_catalog(workspace, project_id)
    names = tuple(item.name for item in sorted(catalog.classes, key=lambda item: item.id))
    return FeatureSpec(class_names=names, analysis_fps=analysis_fps)


def _keep_ranges(blocks: Sequence[RoleBlock]) -> list[tuple[float, float]]:
    """Train and validation blocks only. A test block is not a kept range."""
    return [
        (block.start_s, block.end_s)
        for block in blocks
        if block.role in _KEPT_ROLES
    ]


def _intersected(span: _Span, blocks: Sequence[RoleBlock]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in blocks:
        if block.role not in _KEPT_ROLES:
            continue
        start_s = max(block.start_s, span.start_s)
        end_s = min(block.end_s, span.end_s)
        if end_s <= start_s:
            continue
        rows.append(
            {
                "run_id": span.run_id,
                "role": block.role,
                "start_s": start_s,
                "end_s": end_s,
            }
        )
    return rows


def _log_dropped(run_dir: Path, span: _Span, blocks: Sequence[RoleBlock]) -> None:
    for block in blocks:
        if block.role == "test":
            _log(
                run_dir,
                f"dropped range run_id={span.run_id} role=test start_s={_num(block.start_s)} end_s={_num(block.end_s)}",
            )
    cursor = span.start_s
    for block in sorted(blocks, key=lambda item: item.start_s):
        if block.role not in _KEPT_ROLES:
            continue
        start_s = max(block.start_s, span.start_s)
        end_s = min(block.end_s, span.end_s)
        if end_s <= start_s:
            continue
        if start_s > cursor + 1e-9:
            _log(
                run_dir,
                f"dropped range run_id={span.run_id} role=gap start_s={_num(cursor)} end_s={_num(start_s)}",
            )
        cursor = max(cursor, end_s)
    if span.end_s > cursor + 1e-9:
        _log(
            run_dir,
            f"dropped range run_id={span.run_id} role=gap start_s={_num(cursor)} end_s={_num(span.end_s)}",
        )


def _class_order(catalog: ActionCatalog) -> tuple[str, ...]:
    order = class_index(catalog)
    return tuple(name for name, _index in sorted(order.items(), key=lambda item: item[1]))


def _blocks(frozen: Mapping[str, FrozenSplit]) -> list[RoleBlock]:
    return [block for split in frozen.values() for block in split.blocks]


def _train_validation_intervals(
    workspace: Path,
    project_id: str,
    frozen: Mapping[str, FrozenSplit],
) -> list[ActionInterval]:
    train_rows, validation_rows = _role_intervals(workspace, project_id, frozen)
    return train_rows + validation_rows


def _blocks_hash(frozen: Mapping[str, FrozenSplit]) -> str:
    if len(frozen) == 1:
        return next(iter(frozen.values())).content_hash
    lines = [f"{video_id}:{split.content_hash}" for video_id, split in sorted(frozen.items())]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _read_request(run_dir: Path) -> tuple[list[str], ActionTrainConfig]:
    path = run_dir / REQUEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError("invalid_action_train", "Не читается action_train_request.json.") from exc
    if not isinstance(payload, dict):
        raise LabError("invalid_action_train", "Не читается action_train_request.json.")
    run_ids = payload.get("run_ids")
    if not isinstance(run_ids, list) or any(not isinstance(item, str) for item in run_ids):
        raise LabError("invalid_action_train", "В запросе нет списка прогонов.")
    try:
        config = ActionTrainConfig.model_validate(payload.get("config"))
    except ValidationError as exc:
        raise LabError(
            "invalid_action_train",
            "Некорректная конфигурация обучения действий.",
        ) from exc
    return list(run_ids), config.model_copy(update={"device": "cpu"})


def _cancelled(workspace: Path, project_id: str, job_id: str) -> bool:
    status = get_job(workspace, project_id, job_id).status
    return status in {JobStatus.cancel_requested, JobStatus.cancelled}


def _append_metrics(path: Path, row: Mapping[str, Any]) -> None:
    new = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["epoch", "train_loss", "val_accuracy", "val_macro_f1", "seconds"])
        writer.writerow(
            [
                row["epoch"],
                row["train_loss"],
                row["val_accuracy"],
                row["val_macro_f1"],
                row["seconds"],
            ]
        )


def _write_history(path: Path, rows: Sequence[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_accuracy", "val_macro_f1", "seconds"])
        for row in rows:
            writer.writerow(
                [row.epoch, row.train_loss, row.val_accuracy, row.val_macro_f1, row.seconds]
            )


def _write_weights(path: Path, state: Mapping[str, Any]) -> None:
    import torch

    tmp = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(dict(state), tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_checksums(directory: Path) -> Path:
    files: dict[str, str] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name == "checksums.json" or not path.is_file():
            continue
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    destination = directory / "checksums.json"
    _atomic_json(
        destination,
        {"schema_version": SCHEMA_VERSION, "algorithm": "sha256", "files": files},
    )
    return destination


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _log(run_dir: Path, line: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / LOG_NAME).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _num(value: float) -> str:
    return format(float(value), ".6f").rstrip("0").rstrip(".") or "0"


def _drop_action_model(workspace: Path, project_id: str, action_model_id: str) -> None:
    """Remove the row a cancel landed on after ``insert_action_model``."""
    with connect(project_sqlite(workspace, project_id)) as connection:
        connection.execute(
            """
            DELETE FROM action_models
            WHERE project_id = ? AND action_model_id = ?
            """,
            (project_id, action_model_id),
        )


def _fail(workspace: Path, project_id: str, job_id: str, run_dir: Path, code: str) -> None:
    _log(run_dir, f"failed {code}")
    record = mark_failed(workspace, project_id, job_id, code)
    if record.status is not JobStatus.failed:
        return
    write_progress(
        run_dir,
        {"job_id": job_id, "status": JobStatus.failed.value, "error": code},
    )


def _exception_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text[:500] if text else "action_train_failed"
