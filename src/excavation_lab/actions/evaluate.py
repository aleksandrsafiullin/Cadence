"""Frame-level action metrics against interval labels of one frozen role.

Validation is the default. ``split="test"`` is only ``final=True``: that
command evaluates, and it does not train or retune thresholds. A test
report is written once, and only when a validation report in the same
action-model directory came from an inference with the same
``action_model_id``, ``min_confidence``, and ``min_duration_s``.

Roles are read from the frozen manifest. An interval that no longer sits
inside one block is ``relabelled_outside_role`` and is not a label.
Predicted ``unknown`` on a labelled frame is a miss, not a class.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from excavation_lab.actions.labels import list_intervals, role_of
from excavation_lab.datasets.splits import require_frozen
from excavation_lab.errors import LabError, map_os_error
from excavation_lab.jobs import run_directory
from excavation_lab.storage import (
    list_action_inferences,
    list_action_models,
    list_analysis_runs,
)

_UNKNOWN = "unknown"
_SELECTION_FRAMES = 300
_VALIDATION_PREFIX = "evaluation_validation_"
_TEST_NAME = "evaluation_test.json"
_ACTIONS = "actions.parquet"
_RUN = "action_run.json"
_BUNDLE = "action_bundle.json"

_NOT_FINAL = (
    "Оценка split test выполняется только с --final. "
    "Команда не обучает и не меняет пороги."
)
_FINAL_SPLIT = (
    "Флаг --final допустим только вместе со split test. "
    "Обучение не запускается и пороги не меняются."
)
_BAD_SPLIT = "split должен быть validation или test."
_NO_JOB = "Задание инференса действий не найдено."
_NO_PARQUET = "В задании инференса нет actions.parquet."
_NO_RUN_FILE = "В задании инференса нет action_run.json."
_BAD_RUN = "action_run.json задания инференса не читается."
_NO_MODEL = "Модель действий не найдена."
_NO_ORDER = "В action_bundle.json нет порядка классов."
_NO_SOURCE = "Прогон инференса действий не найден в этом проекте."
_FINAL_EXISTS = (
    "Финальная оценка test для этой модели уже записана. Повтор не выполняется."
)
_NO_VALIDATION = (
    "Нет оценки validation для этой модели. Финальная оценка test не записана."
)


@dataclass(frozen=True, slots=True)
class ClassScore:
    """One class. Null precision, recall, and F1 mean the class had no frames."""

    precision: float | None
    recall: float | None
    f1: float | None
    support_frames: int
    support_intervals: int


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    """One written report. ``run_end_s`` stays null when the analysis run has none."""

    split: str
    final: bool
    infer_job_id: str
    action_model_id: str
    min_confidence: float
    min_duration_s: float
    accuracy: float
    macro_f1: float
    classes_scored: tuple[str, ...]
    per_class: Mapping[str, ClassScore]
    confusion: Mapping[str, Any]
    evaluated_frames: int
    unlabelled_frames: int
    unknown_rate: float
    relabelled_outside_role: int
    selection_unstable: bool
    run_start_s: float
    run_end_s: float | None
    validation_evaluation_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON object. Missing class metrics stay ``null``, not zero."""
        per_class = {
            name: {
                "precision": score.precision,
                "recall": score.recall,
                "f1": score.f1,
                "support_frames": score.support_frames,
                "support_intervals": score.support_intervals,
            }
            for name, score in self.per_class.items()
        }
        payload: dict[str, Any] = {
            "accuracy": self.accuracy,
            "action_model_id": self.action_model_id,
            "classes_scored": list(self.classes_scored),
            "confusion": {
                "labels": list(self.confusion["labels"]),
                "columns": list(self.confusion["columns"]),
                "counts": [list(row) for row in self.confusion["counts"]],
            },
            "evaluated_frames": self.evaluated_frames,
            "final": self.final,
            "infer_job_id": self.infer_job_id,
            "macro_f1": self.macro_f1,
            "min_confidence": self.min_confidence,
            "min_duration_s": self.min_duration_s,
            "per_class": per_class,
            "relabelled_outside_role": self.relabelled_outside_role,
            "run_end_s": self.run_end_s,
            "run_start_s": self.run_start_s,
            "selection_unstable": self.selection_unstable,
            "split": self.split,
            "unknown_rate": self.unknown_rate,
            "unlabelled_frames": self.unlabelled_frames,
        }
        if self.validation_evaluation_file is not None:
            payload["validation_evaluation_file"] = self.validation_evaluation_file
        return payload


def evaluate_actions(
    workspace: Path,
    project_id: str,
    infer_job_id: str,
    *,
    split: str = "validation",
    final: bool = False,
) -> ActionEvaluation:
    """Score one inference against frozen intervals of ``split``.

    ``evaluated_frames == 0`` raises ``no_frames_for_split`` before any
    file is written. Test writes ``evaluation_test.json`` once, in the
    action-model directory. Validation writes
    ``evaluation_validation_<infer_job_id>.json`` there.
    """
    role = _split(split)
    if role == "test" and not final:
        raise LabError("not_final", _NOT_FINAL)
    if final and role != "test":
        raise LabError("not_final", _FINAL_SPLIT)
    report = _measure(workspace, project_id, infer_job_id, role, final=final)
    bundle = _model_directory(workspace, project_id, report.action_model_id)
    if role == "validation":
        _write_json(bundle / f"{_VALIDATION_PREFIX}{infer_job_id}.json", report.to_dict())
        return report
    matched, mismatch = _match_validation(
        workspace,
        project_id,
        bundle,
        report.action_model_id,
        report.min_confidence,
        report.min_duration_s,
    )
    if matched is None:
        raise LabError("final_thresholds_mismatch", mismatch)
    destination = bundle / _TEST_NAME
    if destination.exists() or destination.is_symlink():
        raise LabError("final_exists", _FINAL_EXISTS)
    final_report = ActionEvaluation(
        split=report.split,
        final=True,
        infer_job_id=report.infer_job_id,
        action_model_id=report.action_model_id,
        min_confidence=report.min_confidence,
        min_duration_s=report.min_duration_s,
        accuracy=report.accuracy,
        macro_f1=report.macro_f1,
        classes_scored=report.classes_scored,
        per_class=report.per_class,
        confusion=report.confusion,
        evaluated_frames=report.evaluated_frames,
        unlabelled_frames=report.unlabelled_frames,
        unknown_rate=report.unknown_rate,
        relabelled_outside_role=report.relabelled_outside_role,
        selection_unstable=report.selection_unstable,
        run_start_s=report.run_start_s,
        run_end_s=report.run_end_s,
        validation_evaluation_file=matched.name,
    )
    _write_json(destination, final_report.to_dict())
    return final_report


def format_action_evaluation(
    report: ActionEvaluation | Mapping[str, Any],
    book: object,
) -> str:
    """Operator text. A null class metric uses ``book.not_calculated``."""
    data = report.to_dict() if isinstance(report, ActionEvaluation) else dict(report)
    missing = getattr(book, "not_calculated", None)
    if not isinstance(missing, str) or not missing:
        missing = "не рассчитано"
    english = missing == "not calculated"
    unstable = bool(data.get("selection_unstable"))
    if english:
        unstable_text = "yes" if unstable else "no"
        lines = [
            f"Action evaluation, split: {data.get('split')}",
            f"Job: {data.get('infer_job_id')}",
            f"Run: {_range_text(data.get('run_start_s'), data.get('run_end_s'))}",
            f"Evaluated frames: {data.get('evaluated_frames')}",
            f"Unlabelled frames: {data.get('unlabelled_frames')}",
            f"Intervals outside a role: {data.get('relabelled_outside_role')}",
            f"unknown share: {_metric(data.get('unknown_rate'), missing)}",
            f"Accuracy: {_metric(data.get('accuracy'), missing)}",
            f"macro-F1: {_metric(data.get('macro_f1'), missing)}",
            f"Classes in macro-F1: {_scored_text(data.get('classes_scored'))}",
            f"Too few frames to select on: {unstable_text}",
        ]
        frame_word, interval_word = "frames", "intervals"
    else:
        unstable_text = "да" if unstable else "нет"
        lines = [
            f"Оценка действий, split: {data.get('split')}",
            f"Задание: {data.get('infer_job_id')}",
            f"Прогон: {_range_text(data.get('run_start_s'), data.get('run_end_s'))}",
            f"Кадров оценено: {data.get('evaluated_frames')}",
            f"Кадров без метки: {data.get('unlabelled_frames')}",
            f"Интервалов вне роли: {data.get('relabelled_outside_role')}",
            f"Доля unknown: {_metric(data.get('unknown_rate'), missing)}",
            f"Точность: {_metric(data.get('accuracy'), missing)}",
            f"macro-F1: {_metric(data.get('macro_f1'), missing)}",
            f"Классы в macro-F1: {_scored_text(data.get('classes_scored'))}",
            f"Мало кадров для выбора: {unstable_text}",
        ]
        frame_word, interval_word = "кадры", "интервалы"
    per_class = data.get("per_class")
    if isinstance(per_class, Mapping):
        for name, row in per_class.items():
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"{name}: precision {_metric(row.get('precision'), missing)}, "
                f"recall {_metric(row.get('recall'), missing)}, "
                f"f1 {_metric(row.get('f1'), missing)}, "
                f"{frame_word} {row.get('support_frames')}, "
                f"{interval_word} {row.get('support_intervals')}"
            )
    return "\n".join(lines)


def _measure(
    workspace: Path,
    project_id: str,
    infer_job_id: str,
    split: str,
    *,
    final: bool,
) -> ActionEvaluation:
    directory = _inference_directory(workspace, project_id, infer_job_id)
    action_run = _read_action_run(directory / _RUN)
    frame = _read_frames(directory / _ACTIONS)
    source = _source_run(workspace, project_id, action_run["source_run_id"])
    start_s = float(source["start_s"])
    end_s = source["end_s"]
    blocks = list(require_frozen(workspace, project_id, source["video_id"]).blocks)
    intervals = list_intervals(workspace, project_id, source["video_id"])
    names = list(_class_order(_model_directory(workspace, project_id, action_run["action_model_id"])))
    resolved: list[tuple[Any, str]] = []
    relabelled = 0
    for interval in intervals:
        role = role_of(interval, blocks)
        if role is None:
            relabelled += 1
            continue
        resolved.append((interval, role))
    times = pd.to_numeric(frame["source_time_s"], errors="coerce").to_numpy(dtype=np.float64)
    predictions = _predictions(frame)
    pairs: list[tuple[str, str]] = []
    unlabelled = 0
    for index, stamp in enumerate(times):
        if not math.isfinite(float(stamp)):
            unlabelled += 1
            continue
        interval = _interval_at(float(stamp), resolved, split)
        if interval is None or interval.action_name == _UNKNOWN:
            unlabelled += 1
            continue
        pairs.append((interval.action_name, predictions[index]))
    evaluated = len(pairs)
    if evaluated == 0:
        raise LabError(
            "no_frames_for_split",
            (
                f"Для split {split} нет размеченных кадров в прогоне "
                f"{_range_text(start_s, end_s)}. Оценка не записана."
            ),
        )
    for true, _pred in pairs:
        if true not in names:
            names.append(true)
    interval_support = {name: 0 for name in names}
    for interval, role in resolved:
        if role != split or interval.action_name == _UNKNOWN:
            continue
        if interval.action_name not in interval_support:
            names.append(interval.action_name)
            interval_support[interval.action_name] = 0
        if _overlaps(interval.start_s, interval.end_s, start_s, end_s):
            interval_support[interval.action_name] += 1
    per_class, accuracy, macro, scored, confusion = _scores(
        pairs,
        names,
        interval_support,
    )
    return ActionEvaluation(
        split=split,
        final=final,
        infer_job_id=infer_job_id,
        action_model_id=str(action_run["action_model_id"]),
        min_confidence=float(action_run["min_confidence"]),
        min_duration_s=float(action_run["min_duration_s"]),
        accuracy=accuracy,
        macro_f1=macro,
        classes_scored=scored,
        per_class=per_class,
        confusion=confusion,
        evaluated_frames=evaluated,
        unlabelled_frames=unlabelled,
        unknown_rate=sum(pred == _UNKNOWN for _true, pred in pairs) / evaluated,
        relabelled_outside_role=relabelled,
        selection_unstable=evaluated < _SELECTION_FRAMES,
        run_start_s=start_s,
        run_end_s=None if end_s is None else float(end_s),
    )


def _scores(
    pairs: Sequence[tuple[str, str]],
    names: Sequence[str],
    interval_support: Mapping[str, int],
) -> tuple[dict[str, ClassScore], float, float, tuple[str, ...], dict[str, Any]]:
    labels = list(names)
    columns = [*labels, _UNKNOWN]
    matrix = [[0 for _column in columns] for _label in labels]
    index = {name: row for row, name in enumerate(labels)}
    correct = 0
    for true, pred in pairs:
        if pred == true:
            correct += 1
        row = index[true]
        column = index[pred] if pred in index else len(labels)
        matrix[row][column] += 1
    per_class: dict[str, ClassScore] = {}
    scored: list[str] = []
    f1s: list[float] = []
    for row, name in enumerate(labels):
        support = int(sum(matrix[row]))
        intervals = int(interval_support.get(name, 0))
        if support == 0:
            per_class[name] = ClassScore(None, None, None, 0, intervals)
            continue
        tp = matrix[row][row]
        predicted = sum(matrix[other][row] for other in range(len(labels)))
        false_pos = predicted - tp
        precision = tp / (tp + false_pos) if tp + false_pos else 0.0
        recall = tp / support
        f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)
        per_class[name] = ClassScore(float(precision), float(recall), float(f1), support, intervals)
        scored.append(name)
        f1s.append(float(f1))
    accuracy = correct / len(pairs)
    macro = float(np.mean(np.asarray(f1s, dtype=np.float64)))
    confusion = {"labels": labels, "columns": columns, "counts": matrix}
    return per_class, float(accuracy), macro, tuple(scored), confusion


def _match_validation(
    workspace: Path,
    project_id: str,
    bundle: Path,
    action_model_id: str,
    min_confidence: float,
    min_duration_s: float,
) -> tuple[Path | None, str]:
    """A validation report whose inference ``action_run.json`` matches."""
    current = {
        "action_model_id": action_model_id,
        "min_confidence": min_confidence,
        "min_duration_s": min_duration_s,
    }
    paths = sorted(
        path
        for path in bundle.glob(f"{_VALIDATION_PREFIX}*.json")
        if path.is_file() and not path.is_symlink()
    )
    if not paths:
        return None, _NO_VALIDATION
    notes: list[str] = []
    for path in paths:
        document = _load_object(path)
        if document is None:
            notes.append(f"{path.name}: файл не читается")
            continue
        job_id = _validation_job_id(path, document)
        if job_id is None:
            notes.append(f"{path.name}: нет infer_job_id")
            continue
        try:
            other_dir = _inference_directory(workspace, project_id, job_id)
            other = _read_action_run(other_dir / _RUN)
        except LabError as exc:
            notes.append(f"{path.name}: {exc.message}")
            continue
        diffs = _threshold_diffs(other, current)
        if not diffs:
            return path, ""
        notes.append(f"{path.name}: " + "; ".join(diffs))
    detail = " ".join(notes)
    return None, (
        "Пороги финальной оценки не совпадают с проверкой validation. "
        f"{detail} Файл не записан."
    )


def _threshold_diffs(validation: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    diffs: list[str] = []
    if validation["action_model_id"] != current["action_model_id"]:
        diffs.append(
            "action_model_id: проверка "
            f"{validation['action_model_id']}, эта оценка {current['action_model_id']}"
        )
    if not _close(validation["min_confidence"], current["min_confidence"]):
        diffs.append(
            "min_confidence: проверка "
            f"{_num(validation['min_confidence'])}, эта оценка {_num(current['min_confidence'])}"
        )
    if not _close(validation["min_duration_s"], current["min_duration_s"]):
        diffs.append(
            "min_duration_s: проверка "
            f"{_num(validation['min_duration_s'])}, эта оценка {_num(current['min_duration_s'])}"
        )
    return diffs


def _validation_job_id(path: Path, document: Mapping[str, Any]) -> str | None:
    stored = document.get("infer_job_id")
    if isinstance(stored, str) and stored and "/" not in stored and "\\" not in stored:
        return stored
    name = path.name
    if not name.startswith(_VALIDATION_PREFIX) or not name.endswith(".json"):
        return None
    job_id = name[len(_VALIDATION_PREFIX) : -len(".json")]
    if not job_id or "/" in job_id or "\\" in job_id:
        return None
    return job_id


def _interval_at(
    stamp: float,
    resolved: Sequence[tuple[Any, str]],
    split: str,
) -> Any | None:
    for interval, role in resolved:
        if role == split and interval.start_s <= stamp < interval.end_s:
            return interval
    return None


def _overlaps(
    start_s: float,
    end_s: float,
    run_start: float,
    run_end: float | None,
) -> bool:
    if run_end is None:
        return end_s > run_start
    return start_s < run_end and end_s > run_start


def _inference_directory(workspace: Path, project_id: str, job_id: str) -> Path:
    for row in list_action_inferences(workspace, project_id):
        if row["job_id"] == job_id:
            return Path(row["path"])
    directory = run_directory(workspace, project_id, job_id)
    if (directory / _ACTIONS).is_file() and (directory / _RUN).is_file():
        return directory
    raise LabError("job_not_found", _NO_JOB)


def _model_directory(workspace: Path, project_id: str, action_model_id: str) -> Path:
    for row in list_action_models(workspace, project_id):
        if row["action_model_id"] == action_model_id:
            return Path(row["path"])
    raise LabError("action_model_not_found", _NO_MODEL)


def _source_run(workspace: Path, project_id: str, run_id: object) -> Mapping[str, Any]:
    if not isinstance(run_id, str) or not run_id:
        raise LabError("run_not_found", _NO_SOURCE)
    for row in list_analysis_runs(workspace, project_id):
        if row["run_id"] == run_id:
            return row
    raise LabError("run_not_found", _NO_SOURCE)


def _class_order(bundle: Path) -> tuple[str, ...]:
    document = _load_object(bundle / _BUNDLE)
    raw = None if document is None else document.get("class_order")
    if not isinstance(raw, list) or not raw:
        raise LabError("class_order", _NO_ORDER)
    names: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item or item == _UNKNOWN:
            raise LabError("class_order", _NO_ORDER)
        names.append(item)
    if len(set(names)) != len(names):
        raise LabError("class_order", _NO_ORDER)
    return tuple(names)


def _read_frames(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise LabError("missing_artifact", _NO_PARQUET)
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise LabError("missing_artifact", _NO_PARQUET) from exc
    if "source_time_s" not in frame.columns:
        raise LabError("missing_artifact", _NO_PARQUET)
    if "action" not in frame.columns and "action_name" not in frame.columns:
        raise LabError("missing_artifact", _NO_PARQUET)
    return frame


def _predictions(frame: pd.DataFrame) -> list[str]:
    column = "action" if "action" in frame.columns else "action_name"
    names = frame[column].tolist()
    if "unknown" in frame.columns:
        flags = frame["unknown"].tolist()
    else:
        flags = [False] * len(names)
    return [_prediction(name, flag) for name, flag in zip(names, flags, strict=True)]


def _prediction(name: object, flag: object) -> str:
    if _flag(flag):
        return _UNKNOWN
    if name is None:
        return _UNKNOWN
    if isinstance(name, float) and math.isnan(name):
        return _UNKNOWN
    text = str(name).strip()
    if not text or text == _UNKNOWN or text == "nan":
        return _UNKNOWN
    return text


def _flag(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value) == 1
    return False


def _read_action_run(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LabError("missing_artifact", _NO_RUN_FILE)
    document = _load_object(path)
    if document is None:
        raise LabError("missing_artifact", _BAD_RUN)
    model_id = document.get("action_model_id")
    source = document.get("source_run_id")
    if not isinstance(model_id, str) or not model_id or not isinstance(source, str) or not source:
        raise LabError("missing_artifact", _BAD_RUN)
    confidence = _unit(document.get("min_confidence"))
    duration = _duration(document.get("min_duration_s"))
    if confidence is None or duration is None:
        raise LabError("missing_artifact", _BAD_RUN)
    return {
        "action_model_id": model_id,
        "source_run_id": source,
        "min_confidence": confidence,
        "min_duration_s": duration,
    }


def _load_object(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with map_os_error(str(path)):
            tmp.write_text(text + "\n", encoding="utf-8")
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _split(value: object) -> str:
    if value == "validation" or value == "test":
        return value
    raise LabError("invalid_split", _BAD_SPLIT)


def _unit(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        return None
    return number


def _duration(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _close(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


def _range_text(start_s: object, end_s: object) -> str:
    end = "конец не задан" if end_s is None else _num(end_s)
    return f"[{_num(start_s)}, {end})"


def _num(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    number = float(value)
    if not math.isfinite(number):
        return ""
    return format(number, ".15g")


def _metric(value: object, missing: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return missing
    if not math.isfinite(float(value)):
        return missing
    return format(float(value), ".6g")


def _scored_text(value: object) -> str:
    if not isinstance(value, list) or not value:
        return ""
    return ", ".join(str(item) for item in value)
