"""Run one action bundle over one analysis run.

``keep_ranges`` is ``None``: inference may cover a test block. The bundle
standardizer is applied and not refit. A frame the window does not cover,
a masked frame, a low confidence, or a short span is ``unknown``. A gap
with no frame stays absent. ``unknown`` is not a trained class.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from excavation_lab.actions.features import FeatureSpec, build_features
from excavation_lab.actions.model import ActionNet, Standardizer, load_action_state
from excavation_lab.datasets.splits import canonical_json_bytes
from excavation_lab.errors import LabError, map_os_error
from excavation_lab.jobs import (
    get_job,
    mark_completed,
    mark_failed,
    start_job,
    worker_job,
    write_progress,
)
from excavation_lab.schemas import JobStatus
from excavation_lab.storage import (
    insert_action_inference,
    list_action_models,
    list_analysis_runs,
)
from excavation_lab.tracking.analyze import run_tables

REQUEST_NAME = "action_infer_request.json"
ACTIONS_NAME = "actions.parquet"
INTERVALS_NAME = "action_intervals.csv"
RUN_NAME = "action_run.json"
INTERVAL_COLUMNS: tuple[str, ...] = (
    "start_s",
    "end_s",
    "duration_s",
    "action_name",
    "mean_confidence",
    "frames",
)
_REQUIRED_BUNDLE = (
    "weights.pt",
    "standardizer.json",
    "feature_spec.json",
    "action_bundle.json",
)
_RUN_JSON = (
    "source_run_id",
    "action_model_id",
    "detector_model_id",
    "min_confidence",
    "min_duration_s",
    "unknown_share",
)
_UNKNOWN = "unknown"

_ONE_RUN = "Инференс действий выполняется на одном прогоне."
_BAD_TIME = "Время кадров должно строго расти."
_BAD_THRESHOLD = "Порог уверенности или длительности действия задан неверно."
_DETECTOR = "Детектор прогона не совпадает с моделью действий."
_FPS = "Частота analysis_fps прогона не совпадает с моделью действий."
_NOT_CUSTOM = "Прогон «{run_id}» имеет источник «{origin}», а не custom."
_NOT_DONE = "Прогон «{run_id}» не завершён."
_NO_MODEL = "Модель действий не найдена."
_NO_RUN = "Прогон не найден в этом проекте."
_SPEC = "Спецификация признаков не совпадает с bundle."
_ORDER = "Порядок классов не совпадает с bundle."
_CHECKSUM = "Контрольная сумма bundle действий не сходится."
_UNREADABLE = "Не читается {name}."
_WINDOW = "Окно классификатора действий не совпадает со спецификацией."


@dataclass(frozen=True, slots=True)
class ActionSpan:
    """One half-open interval on the source clock. ``[start_s, end_s)``."""

    start_s: float
    end_s: float
    duration_s: float
    action_name: str
    mean_confidence: float | None
    frames: int


@dataclass(frozen=True, slots=True)
class ActionBundle:
    """A checked action bundle. The standardizer is the one training fit."""

    directory: Path
    action_model_id: str
    detector_model_id: str
    analysis_fps: float
    window_frames: int
    window_s: float
    class_order: tuple[str, ...]
    feature_spec: FeatureSpec
    feature_spec_hash: str
    standardizer: Standardizer
    state_dict: dict[str, Any]
    min_confidence: float
    min_duration_s: float


@dataclass(slots=True)
class _Piece:
    start_row: int
    stop_row: int
    label: str
    start_s: float
    end_s: float


def load_action_bundle(path: Path) -> ActionBundle:
    """Verify checksums, the feature-spec hash, and class order.

    ``standardizer.json`` is loaded as stored. It is not refit.
    """
    directory = Path(path)
    _verify_checksums(directory)
    document = _read_json(directory / "action_bundle.json")
    spec_document = _read_json(directory / "feature_spec.json")
    standardizer = Standardizer.from_json(_read_json(directory / "standardizer.json"))
    spec = _spec(spec_document)
    digest = hashlib.sha256(canonical_json_bytes(spec_document)).hexdigest()
    stored_hash = document.get("feature_spec_hash")
    if not isinstance(stored_hash, str) or stored_hash != digest:
        raise LabError("feature_spec_mismatch", _SPEC)
    embedded = document.get("feature_spec")
    if embedded is not None:
        if not isinstance(embedded, dict):
            raise LabError("feature_spec_mismatch", _SPEC)
        embedded_hash = hashlib.sha256(canonical_json_bytes(embedded)).hexdigest()
        if embedded_hash != digest:
            raise LabError("feature_spec_mismatch", _SPEC)
    _same_standardizer(document.get("standardizer"), standardizer)
    names = _class_order(document)
    state = load_action_state(directory / "weights.pt")
    n_features, n_classes, _hidden = _architecture(state)
    if n_classes != len(names):
        raise LabError("class_order", _ORDER)
    if n_features != len(spec.columns) or standardizer.mean.shape[0] != n_features:
        raise LabError("feature_spec_mismatch", _SPEC)
    fps = _fps(document.get("analysis_fps"))
    if not math.isclose(fps, spec.analysis_fps, rel_tol=0.0, abs_tol=1e-9):
        raise LabError("feature_spec_mismatch", _SPEC)
    window_frames, window_s = _window(document, fps)
    confidence, duration = _bundle_thresholds(document)
    model_id = _token(document.get("action_model_id"), "invalid_action_infer")
    detector_id = _token(document.get("detector_model_id"), "invalid_action_infer")
    return ActionBundle(
        directory=directory,
        action_model_id=model_id,
        detector_model_id=detector_id,
        analysis_fps=fps,
        window_frames=window_frames,
        window_s=window_s,
        class_order=names,
        feature_spec=spec,
        feature_spec_hash=digest,
        standardizer=standardizer,
        state_dict=state,
        min_confidence=confidence,
        min_duration_s=duration,
    )


def predict_frames(bundle: ActionBundle, features_df: pd.DataFrame) -> pd.DataFrame:
    """Class probabilities for every frame of one run.

    The window is centred: its middle frame is ``start + window_frames // 2``
    (the later middle when the length is even). A frame without that full
    window, or with ``frame_mask`` 0, is ``unknown`` and its confidence is NaN.
    """
    if not isinstance(bundle, ActionBundle):
        raise LabError("invalid_action_infer", "Нужен проверенный bundle действий.")
    frame = _one_run(features_df, bundle.feature_spec)
    count = len(frame)
    names = bundle.class_order
    probabilities = np.full((count, len(names)), np.nan, dtype=np.float64)
    confidence = np.full(count, np.nan, dtype=np.float64)
    labels = np.array([_UNKNOWN] * count, dtype=object)
    if count:
        _fill_predictions(bundle, frame, probabilities, confidence, labels)
    result = pd.DataFrame(
        {
            "source_frame_index": frame["source_frame_index"].to_numpy() if count else [],
            "source_time_s": frame["source_time_s"].to_numpy(dtype=np.float64) if count else [],
            **{f"p_{name}": probabilities[:, index] for index, name in enumerate(names)},
            "action_name": labels,
            "confidence": confidence,
            "frame_mask": _mask_column(frame) if count else [],
        }
    )
    result.attrs["analysis_fps"] = bundle.analysis_fps
    return result


def merge_intervals(
    frame_df: pd.DataFrame,
    *,
    min_confidence: float,
    min_duration_s: float,
    analysis_fps: float | None = None,
) -> list[ActionSpan]:
    """Run-length intervals. Short or unsure spans become ``unknown``.

    Interior ends sit on the next frame time. The last span of a contiguous
    stretch ends at its last frame plus ``1/analysis_fps``. A hole larger
    than ``2/analysis_fps`` is not filled.
    """
    confidence = _unit_interval(min_confidence)
    duration = _positive_duration(min_duration_s)
    if "action_name" not in frame_df.columns or "confidence" not in frame_df.columns:
        raise LabError("invalid_action_infer", "В кадрах нет action_name и confidence.")
    fps = _resolve_fps(frame_df, analysis_fps)
    period = 1.0 / fps
    limit = 2.0 / fps
    if frame_df is None or len(frame_df) == 0:
        return []
    frame = _sorted_frames(frame_df)
    times = frame["source_time_s"].to_numpy(dtype=np.float64)
    confidences = frame["confidence"].to_numpy(dtype=np.float64)
    raw = frame["action_name"].tolist()
    labels = [_provisional(name, value, confidence) for name, value in zip(raw, confidences, strict=True)]
    pieces: list[_Piece] = []
    for segment in _segments(times, limit):
        parts = _runs(segment, labels, times, period)
        for part in parts:
            if part.label != _UNKNOWN and (part.end_s - part.start_s) < duration:
                part.label = _UNKNOWN
        pieces.extend(_collapse_unknown(parts))
    return [_span(part, confidences) for part in pieces]


def write_action_results(
    directory: Path,
    predicted: pd.DataFrame,
    spans: Sequence[ActionSpan],
    *,
    source_run_id: str,
    action_model_id: str,
    detector_model_id: str,
    min_confidence: float,
    min_duration_s: float,
) -> list[Path]:
    """Write parquet, CSV, and ``action_run.json``. Return them after a reread."""
    directory = Path(directory)
    published = _publish(predicted, spans)
    share = _unknown_share(spans)
    payload = {
        "source_run_id": source_run_id,
        "action_model_id": action_model_id,
        "detector_model_id": detector_model_id,
        "min_confidence": min_confidence,
        "min_duration_s": min_duration_s,
        "unknown_share": share,
    }
    parquet_path = directory / ACTIONS_NAME
    csv_path = directory / INTERVALS_NAME
    json_path = directory / RUN_NAME
    _write_parquet(parquet_path, published)
    _write_intervals(csv_path, spans)
    _write_json(json_path, payload)
    _reread(parquet_path, csv_path, json_path)
    return [parquet_path, csv_path, json_path]


def enqueue_action_infer(
    workspace: Path,
    project_id: str,
    run_id: str,
    action_model_id: str,
    *,
    min_confidence: float | None = None,
    min_duration_s: float | None = None,
) -> Any:
    """Queue ``action_infer`` for exactly one run. The worker loads no detector."""
    if isinstance(run_id, (list, tuple)) or not isinstance(run_id, str):
        raise LabError("invalid_action_infer", _ONE_RUN)
    bundle, run = _require_compatible(workspace, project_id, run_id, action_model_id)
    confidence = bundle.min_confidence if min_confidence is None else _unit_interval(min_confidence)
    duration = bundle.min_duration_s if min_duration_s is None else _positive_duration(min_duration_s)
    request = {
        "run_id": run["run_id"],
        "action_model_id": action_model_id,
        "min_confidence": confidence,
        "min_duration_s": duration,
    }

    def write_request(run_dir: Path) -> None:
        _write_json(run_dir / REQUEST_NAME, request)

    return start_job(
        workspace,
        project_id,
        kind="action_infer",
        worker="action_infer",
        on_run_dir=write_request,
    )


def run_action_infer_job(workspace: Path, job_id: str) -> int:
    """Child entry. ``completed`` follows a readable parquet and CSV."""
    root = Path(workspace).resolve()
    project_id, run_dir, record = worker_job(root, job_id)
    if record.status in {JobStatus.cancelled, JobStatus.cancel_requested}:
        return 0
    try:
        return _execute(root, project_id, job_id, run_dir)
    except LabError as exc:
        _fail(root, project_id, job_id, exc.code)
        return 1
    except Exception as exc:  # noqa: BLE001 — the job row keeps the failure
        _fail(root, project_id, job_id, _exception_text(exc))
        return 1


def export_action_intervals(
    workspace: Path,
    job_id: str,
    destination: Path,
    *,
    excel: bool = False,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Copy ``action_intervals.csv``. ``excel`` adds a UTF-8 BOM."""
    project, run_dir, record = worker_job(Path(workspace).resolve(), job_id)
    if project_id is not None and project_id != project:
        raise LabError("job_not_found", "Задание не найдено в этом проекте.")
    if record.status is not JobStatus.completed:
        raise LabError("missing_artifact", "Интервалы действий ещё не записаны.")
    source = Path(run_dir) / INTERVALS_NAME
    if not source.is_file():
        raise LabError("missing_artifact", "В задании нет action_intervals.csv.")
    payload = source.read_bytes()
    if excel:
        payload = b"\xef\xbb\xbf" + payload
    target = Path(destination)
    _write_bytes(target, payload)
    return {
        "job_id": job_id,
        "project_id": project,
        "csv": str(target.resolve()),
        "excel_bom": bool(excel),
    }


def _execute(workspace: Path, project_id: str, job_id: str, run_dir: Path) -> int:
    if _cancelled(workspace, project_id, job_id):
        return 0
    request = _read_request(run_dir)
    bundle, run = _require_compatible(
        workspace,
        project_id,
        request["run_id"],
        request["action_model_id"],
    )
    confidence = _unit_interval(request["min_confidence"])
    duration = _positive_duration(request["min_duration_s"])
    write_progress(
        run_dir,
        {"job_id": job_id, "status": JobStatus.running.value},
    )
    features = build_features(
        run_tables(Path(run["run_dir"])),
        run_id=request["run_id"],
        run_index=0,
        keep_ranges=None,
        spec=bundle.feature_spec,
    )
    if _cancelled(workspace, project_id, job_id):
        return 0
    predicted = predict_frames(bundle, features)
    spans = merge_intervals(
        predicted,
        min_confidence=confidence,
        min_duration_s=duration,
        analysis_fps=bundle.analysis_fps,
    )
    if _cancelled(workspace, project_id, job_id):
        return 0
    artifacts = write_action_results(
        run_dir,
        predicted,
        spans,
        source_run_id=request["run_id"],
        action_model_id=request["action_model_id"],
        detector_model_id=bundle.detector_model_id,
        min_confidence=confidence,
        min_duration_s=duration,
    )
    if _cancelled(workspace, project_id, job_id):
        return 0
    insert_action_inference(
        workspace,
        project_id,
        job_id=job_id,
        source_run_id=request["run_id"],
        action_model_id=request["action_model_id"],
        path=run_dir,
    )
    mark_completed(workspace, project_id, job_id, artifacts)
    write_progress(
        run_dir,
        {"job_id": job_id, "status": JobStatus.completed.value},
    )
    return 0


def _fill_predictions(
    bundle: ActionBundle,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    confidence: np.ndarray,
    labels: np.ndarray,
) -> None:
    width = bundle.window_frames
    # Centre matches training: start + width // 2, the later middle if even.
    half = width // 2
    limit = 2.0 / bundle.analysis_fps
    times = frame["source_time_s"].to_numpy(dtype=np.float64)
    run_start = _run_start(frame)
    run_ids = np.array([str(item) for item in frame["run_id"].tolist()], dtype=object)
    values = frame.loc[:, list(bundle.feature_spec.columns)].to_numpy(dtype=np.float64, copy=True)
    masked = _masked_rows(frame)
    centres: list[int] = []
    windows: list[np.ndarray] = []
    count = len(frame)
    for index in range(count):
        if masked[index]:
            continue
        start = index - half
        if not _full_window(times, run_start, run_ids, start, width, limit):
            continue
        centres.append(index)
        windows.append(values[start : start + width])
    if not windows:
        return
    probs = _forward(bundle, np.stack(windows, axis=0))
    names = bundle.class_order
    for row, centre in enumerate(centres):
        probabilities[centre] = probs[row]
        choice = int(np.argmax(probs[row]))
        confidence[centre] = float(probs[row, choice])
        labels[centre] = names[choice]


def _forward(bundle: ActionBundle, windows: np.ndarray) -> np.ndarray:
    import torch

    scaled = bundle.standardizer.apply(windows)
    if not np.isfinite(scaled).all():
        raise LabError("invalid_window", "После стандартизации в окне бесконечное значение.")
    _n_features, n_classes, hidden = _architecture(bundle.state_dict)
    module = ActionNet(int(windows.shape[-1]), n_classes, hidden)
    module.load_state_dict(bundle.state_dict)
    module.eval()
    tensor = torch.from_numpy(np.ascontiguousarray(scaled, dtype=np.float32))
    with torch.no_grad():
        logits = module(tensor)
        return torch.softmax(logits, dim=-1).cpu().numpy()


def _full_window(
    times: np.ndarray,
    run_start: np.ndarray,
    run_ids: np.ndarray,
    start: int,
    width: int,
    limit: float,
) -> bool:
    stop = start + width
    count = int(times.shape[0])
    if start < 0 or stop > count:
        return False
    if bool(run_start[start + 1 : stop].any()):
        return False
    if len(set(run_ids[start:stop].tolist())) > 1:
        return False
    deltas = np.diff(times[start:stop])
    return not (deltas.size and bool(np.any(deltas <= 0.0) or np.any(deltas > limit)))


def _segments(times: np.ndarray, limit: float) -> list[range]:
    """Contiguous frames. A larger step is a gap and stays unlabeled."""
    count = int(times.shape[0])
    if count == 0:
        return []
    groups: list[range] = []
    start = 0
    for index in range(1, count):
        if float(times[index] - times[index - 1]) > limit:
            groups.append(range(start, index))
            start = index
    groups.append(range(start, count))
    return groups


def _runs(
    segment: range,
    labels: Sequence[str],
    times: np.ndarray,
    period: float,
) -> list[_Piece]:
    pieces: list[_Piece] = []
    start = segment.start
    label = labels[start]
    for index in range(segment.start + 1, segment.stop):
        if labels[index] != label:
            pieces.append(_piece(start, index, label, times, period, last=False, nxt=index))
            start = index
            label = labels[index]
    pieces.append(_piece(start, segment.stop, label, times, period, last=True, nxt=None))
    return pieces


def _piece(
    start: int,
    stop: int,
    label: str,
    times: np.ndarray,
    period: float,
    *,
    last: bool,
    nxt: int | None,
) -> _Piece:
    start_s = float(times[start])
    if last or nxt is None:
        end_s = float(times[stop - 1]) + period
    else:
        end_s = float(times[nxt])
    return _Piece(start, stop, label, start_s, end_s)


def _collapse_unknown(pieces: Sequence[_Piece]) -> list[_Piece]:
    merged: list[_Piece] = []
    for piece in pieces:
        previous = merged[-1] if merged else None
        abut = previous is not None and math.isclose(previous.end_s, piece.start_s, rel_tol=0.0, abs_tol=1e-9)
        if previous is not None and previous.label == _UNKNOWN and piece.label == _UNKNOWN and abut:
            previous.stop_row = piece.stop_row
            previous.end_s = piece.end_s
            continue
        merged.append(piece)
    return merged


def _span(piece: _Piece, confidences: np.ndarray) -> ActionSpan:
    window = confidences[piece.start_row : piece.stop_row]
    finite = window[np.isfinite(window)]
    mean = float(finite.mean()) if finite.size else None
    start_s = piece.start_s
    end_s = piece.end_s
    return ActionSpan(
        start_s=start_s,
        end_s=end_s,
        duration_s=end_s - start_s,
        action_name=piece.label,
        mean_confidence=mean,
        frames=piece.stop_row - piece.start_row,
    )


def _provisional(name: object, confidence: float, minimum: float) -> str:
    if not isinstance(name, str) or name == _UNKNOWN:
        return _UNKNOWN
    if not math.isfinite(confidence) or confidence < minimum:
        return _UNKNOWN
    return name


def _publish(predicted: pd.DataFrame, spans: Sequence[ActionSpan]) -> pd.DataFrame:
    count = len(predicted)
    labels = np.array([_UNKNOWN] * count, dtype=object)
    if count:
        times = predicted["source_time_s"].to_numpy(dtype=np.float64)
        matched = np.zeros(count, dtype=bool)
        for span in spans:
            hit = (times >= span.start_s) & (times < span.end_s)
            labels[hit] = span.action_name
            matched |= hit
        if not bool(matched.all()):
            raise LabError(
                "invalid_action_infer",
                "Кадр остался вне интервалов. Пропуск не заполняется классом.",
            )
    published = predicted.copy()
    published["action_name"] = labels
    published["action"] = labels
    published["unknown"] = published["action"].eq(_UNKNOWN)
    columns = ["source_frame_index", "source_time_s"]
    columns.extend(name for name in published.columns if name.startswith("p_"))
    columns.extend(["action_name", "action", "confidence", "frame_mask", "unknown"])
    return published.loc[:, columns]


def _unknown_share(spans: Sequence[ActionSpan]) -> float | None:
    total = sum(span.duration_s for span in spans)
    if total <= 0.0:
        return None
    unknown = sum(span.duration_s for span in spans if span.action_name == _UNKNOWN)
    return unknown / total


def _require_compatible(
    workspace: Path,
    project_id: str,
    run_id: str,
    action_model_id: str,
) -> tuple[ActionBundle, Mapping[str, Any]]:
    if not isinstance(run_id, str) or not run_id:
        raise LabError("invalid_action_infer", _ONE_RUN)
    model = _model_row(workspace, project_id, action_model_id)
    bundle = load_action_bundle(Path(model["path"]))
    if bundle.action_model_id != action_model_id:
        raise LabError("invalid_action_infer", "Идентификатор модели в bundle не совпадает.")
    run = _run_row(workspace, project_id, run_id)
    if not run["completed"]:
        raise LabError("action_not_ready", _NOT_DONE.format(run_id=run_id))
    if run["model_origin"] != "custom":
        raise LabError(
            "action_not_custom",
            _NOT_CUSTOM.format(run_id=run_id, origin=run["model_origin"]),
        )
    if run["model_id"] != bundle.detector_model_id:
        raise LabError("action_detector_mismatch", _DETECTOR)
    if not math.isclose(float(run["analysis_fps"]), bundle.analysis_fps, rel_tol=0.0, abs_tol=1e-9):
        raise LabError("action_fps_mismatch", _FPS)
    run_tables(Path(run["run_dir"]))
    return bundle, run


def _model_row(workspace: Path, project_id: str, action_model_id: str) -> Mapping[str, Any]:
    if not isinstance(action_model_id, str) or not action_model_id:
        raise LabError("invalid_action_infer", "Нужен один идентификатор модели действий.")
    for row in list_action_models(workspace, project_id):
        if row["action_model_id"] == action_model_id:
            return row
    raise LabError("action_model_not_found", _NO_MODEL)


def _run_row(workspace: Path, project_id: str, run_id: str) -> Mapping[str, Any]:
    for row in list_analysis_runs(workspace, project_id):
        if row["run_id"] == run_id:
            return row
    raise LabError("run_not_found", _NO_RUN)


def _one_run(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise LabError("feature_spec_mismatch", _SPEC)
    required = (
        "run_id",
        "run_index",
        "run_start",
        "source_frame_index",
        "source_time_s",
        "frame_mask",
        *spec.columns,
    )
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise LabError("feature_spec_mismatch", _SPEC)
    if len(frame) == 0:
        return frame
    ordered = _sorted_frames(frame)
    run_ids = {str(item) for item in ordered["run_id"].tolist()}
    indexes = ordered["run_index"].to_numpy()
    if len(run_ids) != 1 or bool(np.any(indexes != 0)):
        raise LabError("action_infer_one_run", _ONE_RUN)
    return ordered


def _sorted_frames(frame: pd.DataFrame) -> pd.DataFrame:
    if "source_time_s" not in frame.columns:
        raise LabError("invalid_action_infer", _BAD_TIME)
    times = pd.to_numeric(frame["source_time_s"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(times).all():
        raise LabError("invalid_action_infer", _BAD_TIME)
    order = np.argsort(times, kind="mergesort")
    ordered = frame.iloc[order].reset_index(drop=True)
    sorted_times = times[order]
    if sorted_times.size and bool(np.any(np.diff(sorted_times) <= 0.0)):
        raise LabError("invalid_action_infer", _BAD_TIME)
    return ordered


def _resolve_fps(frame_df: pd.DataFrame, analysis_fps: float | None) -> float:
    stored = frame_df.attrs.get("analysis_fps") if isinstance(frame_df, pd.DataFrame) else None
    if analysis_fps is None:
        if stored is None:
            raise LabError("action_fps_mismatch", _FPS)
        return _fps(stored)
    fps = _fps(analysis_fps)
    if stored is not None and not math.isclose(fps, _fps(stored), rel_tol=0.0, abs_tol=1e-9):
        raise LabError("action_fps_mismatch", _FPS)
    return fps


def _verify_checksums(directory: Path) -> None:
    checksum_path = directory / "checksums.json"
    if not checksum_path.is_file():
        raise LabError("action_bundle_checksum", _CHECKSUM)
    document = _read_json(checksum_path)
    if document.get("algorithm") != "sha256":
        raise LabError("action_bundle_checksum", _CHECKSUM)
    listed = document.get("files")
    if not isinstance(listed, dict):
        raise LabError("action_bundle_checksum", _CHECKSUM)
    actual: dict[str, str] = {}
    for path in directory.iterdir():
        if not path.is_file() or path.name == "checksums.json":
            continue
        actual[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if set(actual) != set(listed):
        raise LabError("action_bundle_checksum", _CHECKSUM)
    for name, digest in actual.items():
        expected = listed.get(name)
        if not isinstance(expected, str) or expected != digest:
            raise LabError("action_bundle_checksum", _CHECKSUM)
    missing = [name for name in _REQUIRED_BUNDLE if name not in actual]
    if missing:
        raise LabError("action_bundle_checksum", _CHECKSUM)


def _class_order(document: Mapping[str, Any]) -> tuple[str, ...]:
    raw = document.get("class_order")
    if not isinstance(raw, list) or not raw or any(not isinstance(item, str) for item in raw):
        raise LabError("class_order", _ORDER)
    names = tuple(raw)
    if len(set(names)) != len(names) or _UNKNOWN in names:
        raise LabError("class_order", _ORDER)
    catalog = document.get("action_catalog")
    if catalog is not None and tuple(_catalog_names(catalog)) != names:
        raise LabError("class_order", _ORDER)
    return names


def _catalog_names(catalog: object) -> list[str]:
    if not isinstance(catalog, list):
        raise LabError("class_order", _ORDER)
    rows: list[tuple[int, str]] = []
    for item in catalog:
        if not isinstance(item, dict):
            raise LabError("class_order", _ORDER)
        name = item.get("name")
        identifier = item.get("id")
        if name == _UNKNOWN:
            continue
        if not isinstance(name, str) or isinstance(identifier, bool) or not isinstance(identifier, int):
            raise LabError("class_order", _ORDER)
        rows.append((identifier, name))
    rows.sort(key=lambda item: item[0])
    return [name for _identifier, name in rows]


def _same_standardizer(embedded: object, standardizer: Standardizer) -> None:
    if embedded is None:
        return
    other = Standardizer.from_json(embedded)
    if other.mean.shape != standardizer.mean.shape:
        raise LabError("feature_spec_mismatch", _SPEC)
    if not np.allclose(other.mean, standardizer.mean) or not np.allclose(other.std, standardizer.std):
        raise LabError("feature_spec_mismatch", _SPEC)


def _architecture(state: Mapping[str, Any]) -> tuple[int, int, int]:
    try:
        conv1 = state["conv1.weight"]
        conv2 = state["conv2.weight"]
        fc = state["fc.weight"]
    except KeyError as exc:
        raise LabError("action_weights_missing", "В весах нет слоёв ActionNet.") from exc
    if getattr(conv1, "ndim", None) != 3 or getattr(fc, "ndim", None) != 2:
        raise LabError("action_weights_missing", "В весах нет слоёв ActionNet.")
    hidden, n_features, kernel = (int(conv1.shape[0]), int(conv1.shape[1]), int(conv1.shape[2]))
    n_classes = int(fc.shape[0])
    if kernel != 3 or int(fc.shape[1]) != hidden or tuple(int(item) for item in conv2.shape) != (
        hidden,
        hidden,
        3,
    ):
        raise LabError("action_weights_missing", "В весах нет слоёв ActionNet.")
    if hidden < 1 or n_features < 1 or n_classes < 1:
        raise LabError("action_weights_missing", "В весах нет слоёв ActionNet.")
    return n_features, n_classes, hidden


def _window(document: Mapping[str, Any], fps: float) -> tuple[int, float]:
    raw = document.get("window_frames")
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 3:
        raise LabError("feature_spec_mismatch", _WINDOW)
    window_s = document.get("window_s")
    if isinstance(window_s, bool) or not isinstance(window_s, (int, float)) or not math.isfinite(float(window_s)):
        raise LabError("feature_spec_mismatch", _WINDOW)
    seconds = float(window_s)
    if seconds <= 0.0 or max(3, round(seconds * fps)) != raw:
        raise LabError("feature_spec_mismatch", _WINDOW)
    return raw, seconds


def _bundle_thresholds(document: Mapping[str, Any]) -> tuple[float, float]:
    config = document.get("config")
    if not isinstance(config, dict):
        config = {}
    confidence = config.get("min_confidence", 0.5)
    duration = config.get("min_duration_s", 1.0)
    return _unit_interval(confidence), _positive_duration(duration)


def _spec(document: Mapping[str, Any]) -> FeatureSpec:
    try:
        return FeatureSpec.model_validate(document)
    except ValidationError as exc:
        raise LabError("feature_spec_mismatch", _SPEC) from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LabError("action_bundle_unreadable", _UNREADABLE.format(name=path.name)) from exc
    if not isinstance(loaded, dict):
        raise LabError("action_bundle_unreadable", _UNREADABLE.format(name=path.name))
    return loaded


def _read_request(run_dir: Path) -> dict[str, Any]:
    path = run_dir / REQUEST_NAME
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LabError("invalid_action_infer", "Не читается action_infer_request.json.") from exc
    if not isinstance(loaded, dict):
        raise LabError("invalid_action_infer", "Не читается action_infer_request.json.")
    run_id = loaded.get("run_id")
    model_id = loaded.get("action_model_id")
    if not isinstance(run_id, str) or not isinstance(model_id, str):
        raise LabError("invalid_action_infer", _ONE_RUN)
    return loaded


def _mask_column(frame: pd.DataFrame) -> list[int]:
    return [0 if flagged else 1 for flagged in _masked_rows(frame)]


def _masked_rows(frame: pd.DataFrame) -> np.ndarray:
    return np.array([_is_masked(value) for value in frame["frame_mask"].tolist()], dtype=bool)


def _is_masked(value: object) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        return True
    if isinstance(value, (bool, np.bool_)):
        return not bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) == 0.0
    return True


def _run_start(frame: pd.DataFrame) -> np.ndarray:
    return np.array([bool(item) for item in frame["run_start"].tolist()], dtype=bool)


def _fps(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError("action_fps_mismatch", _FPS)
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0.0:
        raise LabError("action_fps_mismatch", _FPS)
    return fps


def _unit_interval(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError("invalid_action_infer", _BAD_THRESHOLD)
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise LabError("invalid_action_infer", _BAD_THRESHOLD)
    return number


def _positive_duration(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError("invalid_action_infer", _BAD_THRESHOLD)
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise LabError("invalid_action_infer", _BAD_THRESHOLD)
    return number


def _token(value: object, code: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "/" in value or "\\" in value:
        raise LabError(code, "Недопустимый идентификатор в bundle действий.")
    return value


def _reread(parquet_path: Path, csv_path: Path, json_path: Path) -> None:
    try:
        frame = pd.read_parquet(parquet_path)
        table = pd.read_csv(csv_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise LabError("missing_artifact", "Результат инференса действий не читается.") from exc
    required = {"source_time_s", "action", "confidence", "unknown"}
    if required - set(frame.columns) or list(table.columns) != list(INTERVAL_COLUMNS):
        raise LabError("missing_artifact", "Результат инференса действий не читается.")
    if not isinstance(payload, dict) or any(key not in payload for key in _RUN_JSON):
        raise LabError("missing_artifact", "Результат инференса действий не читается.")


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with map_os_error(str(path)):
            frame.to_parquet(tmp, index=False)
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_intervals(path: Path, spans: Sequence[ActionSpan]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(INTERVAL_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for span in spans:
        writer.writerow(
            {
                "start_s": _number(span.start_s),
                "end_s": _number(span.end_s),
                "duration_s": _number(span.duration_s),
                "action_name": span.action_name,
                "mean_confidence": _number(span.mean_confidence),
                "frames": str(span.frames),
            }
        )
    _write_bytes(path, buffer.getvalue().encode("utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_bytes(path, text.encode("utf-8"))


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with map_os_error(str(path)):
            tmp.write_bytes(payload)
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return format(float(value), ".15g")


def _cancelled(workspace: Path, project_id: str, job_id: str) -> bool:
    status = get_job(workspace, project_id, job_id).status
    return status in {JobStatus.cancel_requested, JobStatus.cancelled}


def _fail(workspace: Path, project_id: str, job_id: str, code: str) -> None:
    record = mark_failed(workspace, project_id, job_id, code)
    if record.status is not JobStatus.failed:
        return
    _project, run_dir, _stored = worker_job(workspace, job_id)
    del _project, _stored, project_id
    write_progress(
        Path(run_dir),
        {"job_id": job_id, "status": JobStatus.failed.value, "error": code},
    )


def _exception_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text[:500] if text else "action_infer_failed"
