"""Train ``ActionNet`` on train windows and pick a checkpoint on validation.

The standardizer is fit on ``role="train"`` windows only. Windows with
``role="test"`` are dropped before they are concatenated into ``x``.
Macro-F1 averages classes that have validation support. A class with none
stores null precision, recall, and F1, and is listed out of ``classes_scored``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from excavation_lab.actions.dataset import ActionWindows, WindowConfig
from excavation_lab.actions.features import FeatureSpec
from excavation_lab.actions.model import ActionNet, Standardizer, set_seed
from excavation_lab.errors import LabError, map_os_error
from excavation_lab.schemas import StrictModel

WEIGHTS_NAME = "weights.pt"
STANDARDIZER_NAME = "standardizer.json"
_SELECTION_WINDOWS = 300

_BAD_CONFIG = "Нужна конфигурация обучения действий."
_BAD_NAMES = "Нужен непустой порядок классов без повторов."
_BAD_ARRAY = "Окна действий должны быть массивом формы (n, время, признаки)."
_BAD_LABELS = "Метки окон не совпадают с порядком классов."
_BAD_FEATURES = "Число признаков train и validation не совпадает."
_BAD_WINDOWS = "Нужны окна действий с ролью train, validation или test."
_BAD_ROLE = "Роль окон должна быть train, validation или test."
_BAD_SPEC = "Нужна FeatureSpec с порядком классов и analysis_fps."
_BAD_DIR = "Каталог весов классификатора действий должен быть папкой."
_VAL_EMPTY = "Нет окон валидации. Чекпоинт выбирать не на чем, обучение не запускается."
_INFINITE = "После стандартизации в окнах осталось бесконечное значение."
_WEIGHTS_SHAPE = "В весах нет слоёв ActionNet."
_KEPT_ROLES = frozenset({"train", "validation"})


class ActionTrainConfig(StrictModel):
    """Loop settings. Thresholds are stored for the bundle; training ignores them."""

    epochs: int = Field(default=40, ge=1)
    lr: float = Field(default=1e-3, gt=0, allow_inf_nan=False)
    batch: int = Field(default=64, ge=1)
    patience: int = Field(default=8, ge=1)
    min_confidence: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    min_duration_s: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    window: WindowConfig = Field(default_factory=WindowConfig)
    device: str = "cpu"
    seed: int = Field(default=42, ge=0)


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """One class on one validation pass. Null scores mean the class was absent."""

    precision: float | None
    recall: float | None
    f1: float | None
    support: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


@dataclass(frozen=True, slots=True)
class HistoryRow:
    """One finished epoch. ``seconds`` is wall time from the start of the loop."""

    epoch: int
    train_loss: float
    val_accuracy: float
    val_macro_f1: float
    seconds: float
    per_class: dict[str, ClassMetrics]

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_accuracy": self.val_accuracy,
            "val_macro_f1": self.val_macro_f1,
            "seconds": self.seconds,
            "per_class": {name: item.to_dict() for name, item in self.per_class.items()},
        }


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    """Scores of the selected checkpoint. ``classes_scored`` had support > 0."""

    classes_scored: tuple[str, ...]
    per_class: dict[str, ClassMetrics]
    selection_unstable: bool
    support_train: dict[str, int]
    support_validation: dict[str, int]
    val_accuracy: float
    val_macro_f1: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "classes_scored": list(self.classes_scored),
            "per_class": {name: item.to_dict() for name, item in self.per_class.items()},
            "selection_unstable": self.selection_unstable,
            "support_train": dict(self.support_train),
            "support_validation": dict(self.support_validation),
            "val_accuracy": self.val_accuracy,
            "val_macro_f1": self.val_macro_f1,
        }


@dataclass(frozen=True, slots=True)
class TrainOutcome:
    """Best weights, the train-only standardizer, and the metadata for a bundle."""

    validation_metrics: ValidationMetrics
    metrics_history: tuple[HistoryRow, ...]
    standardizer: Standardizer
    state_dict: dict[str, Any]
    class_order: tuple[str, ...]
    best_epoch: int
    weights_path: Path | None
    metadata: dict[str, Any]


def split_action_windows(
    windows: Sequence[ActionWindows],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack train and validation. Role ``test`` is not copied into either array."""
    rows = _require_windows(windows)
    grouped: dict[str, list[ActionWindows]] = {"train": [], "validation": []}
    for item in rows:
        if item.role == "test":
            continue
        if item.role not in _KEPT_ROLES:
            raise LabError("invalid_role", _BAD_ROLE)
        grouped[item.role].append(item)
    template = _template(grouped["train"]) or _template(grouped["validation"])
    x_train, y_train = _concat(grouped["train"], template)
    x_val, y_val = _concat(grouped["validation"], template)
    return x_train, y_train, x_val, y_val


def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> tuple[dict[str, ClassMetrics], float, float, tuple[str, ...]]:
    """Accuracy and macro-F1. Classes with no true validation windows stay null.

    Macro-F1 is the unweighted mean of F1 over ``classes_scored`` only. An
    absent class does not contribute a zero.
    """
    names = _class_names(class_names)
    true = _as_labels(y_true, len(names))
    pred = _as_labels(y_pred, len(names), n=int(true.shape[0]))
    if true.shape[0] == 0:
        raise LabError("action_validation_empty", _VAL_EMPTY)
    n_classes = len(names)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (true, pred), 1)
    per_class: dict[str, ClassMetrics] = {}
    scored: list[str] = []
    f1s: list[float] = []
    for index, name in enumerate(names):
        support = int(confusion[index].sum())
        if support == 0:
            per_class[name] = ClassMetrics(None, None, None, 0)
            continue
        tp = int(confusion[index, index])
        false_pos = int(confusion[:, index].sum()) - tp
        precision = tp / (tp + false_pos) if tp + false_pos else 0.0
        recall = tp / support
        f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall)
        per_class[name] = ClassMetrics(float(precision), float(recall), float(f1), support)
        scored.append(name)
        f1s.append(float(f1))
    if not f1s:
        raise LabError("action_validation_empty", _VAL_EMPTY)
    accuracy = float(np.mean(true == pred))
    macro = float(np.mean(np.asarray(f1s, dtype=np.float64)))
    return per_class, accuracy, macro, tuple(scored)


def train_action_windows(
    windows: Sequence[ActionWindows],
    class_names: Sequence[str],
    config: ActionTrainConfig,
    *,
    on_epoch: Callable[[Mapping[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    output_dir: Path | None = None,
    feature_spec: FeatureSpec | None = None,
) -> TrainOutcome:
    """Train on ``role="train"`` only. Test windows never enter ``x``."""
    x_train, y_train, x_val, y_val = split_action_windows(windows)
    return train_action_model(
        x_train,
        y_train,
        x_val,
        y_val,
        class_names,
        config,
        on_epoch=on_epoch,
        should_stop=should_stop,
        output_dir=output_dir,
        feature_spec=feature_spec,
    )


def train_action_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_names: Sequence[str],
    config: ActionTrainConfig,
    *,
    on_epoch: Callable[[Mapping[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    output_dir: Path | None = None,
    feature_spec: FeatureSpec | None = None,
) -> TrainOutcome:
    """Fit the standardizer on ``x_train``, then select by validation macro-F1.

    Ties in macro-F1 go to the higher validation accuracy. Early stop waits
    ``patience`` epochs without that improvement. ``should_stop`` is read
    after each finished epoch.
    """
    settings = _config(config)
    names = _class_names(class_names)
    train_x, train_y = _as_xy(x_train, y_train, len(names), None)
    val_x, val_y = _as_xy(x_val, y_val, len(names), int(train_x.shape[2]))
    support_train = _count(train_y, names)
    missing = [name for name in names if support_train[name] == 0]
    if missing:
        raise LabError("action_support_missing", _missing_message(missing))
    if val_y.shape[0] == 0:
        raise LabError("action_validation_empty", _VAL_EMPTY)
    support_validation = _count(val_y, names)
    standardizer = Standardizer.fit(train_x)
    scaled_train = _finite(standardizer.apply(train_x))
    scaled_val = _finite(standardizer.apply(val_x))
    set_seed(settings.seed)
    model = ActionNet(int(train_x.shape[2]), len(names))
    state, history, metrics, best_epoch = _loop(
        model,
        scaled_train,
        train_y,
        scaled_val,
        val_y,
        names,
        support_train,
        support_validation,
        settings,
        on_epoch,
        should_stop,
    )
    weights_path = None
    if output_dir is not None:
        weights_path = _write_training(Path(output_dir), state, standardizer)
    metadata = _metadata(
        names,
        settings,
        int(train_x.shape[1]),
        feature_spec,
        standardizer,
        metrics,
    )
    return TrainOutcome(
        validation_metrics=metrics,
        metrics_history=tuple(history),
        standardizer=standardizer,
        state_dict=state,
        class_order=names,
        best_epoch=best_epoch,
        weights_path=weights_path,
        metadata=metadata,
    )


def predict_actions(
    state: Mapping[str, Any],
    x: np.ndarray,
    *,
    standardizer: Standardizer,
    device: str = "cpu",
) -> np.ndarray:
    """Class index per window. ``x`` is raw features; the standardizer is applied."""
    import torch

    if not isinstance(standardizer, Standardizer):
        raise LabError("invalid_window", _BAD_ARRAY)
    weight = state.get("conv1.weight")
    head = state.get("fc.weight")
    if weight is None or head is None:
        raise LabError("action_weights_missing", _WEIGHTS_SHAPE)
    n_features = int(weight.shape[1])
    hidden = int(weight.shape[0])
    n_classes = int(head.shape[0])
    module = ActionNet(n_features, n_classes, hidden)
    module.load_state_dict(dict(state))
    module.to(device)
    module.eval()
    scaled = _finite(standardizer.apply(x))
    tensor = torch.tensor(np.ascontiguousarray(scaled), dtype=torch.float32)
    preds: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, int(tensor.shape[0]), 256):
            logits = module(tensor[start : start + 256].to(device))
            preds.append(logits.argmax(dim=1).detach().cpu().numpy())
    if not preds:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(preds)


def _loop(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    names: Sequence[str],
    support_train: dict[str, int],
    support_validation: dict[str, int],
    config: ActionTrainConfig,
    on_epoch: Callable[[Mapping[str, Any]], None] | None,
    should_stop: Callable[[], bool] | None,
) -> tuple[dict[str, Any], list[HistoryRow], ValidationMetrics, int]:
    import torch

    device = config.device
    module = model.to(device)
    counts = np.array([support_train[name] for name in names], dtype=np.float64)
    weights = torch.tensor(counts.sum() / (len(names) * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(module.parameters(), lr=config.lr)
    train_x = torch.tensor(np.ascontiguousarray(x_train), dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.long)
    val_x = torch.tensor(np.ascontiguousarray(x_val), dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    best_state: dict[str, Any] | None = None
    best_metrics: ValidationMetrics | None = None
    best_key: tuple[float, float] | None = None
    best_epoch = 0
    wait = 0
    history: list[HistoryRow] = []
    started = time.perf_counter()
    n_val = int(y_val.shape[0])
    for epoch in range(1, config.epochs + 1):
        loss = _train_epoch(module, train_x, train_y, criterion, optimizer, config.batch, generator, device)
        predicted = _predict_tensor(module, val_x, config.batch, device)
        per_class, accuracy, macro, scored = score_predictions(y_val, predicted, names)
        metrics = ValidationMetrics(
            classes_scored=scored,
            per_class=per_class,
            selection_unstable=n_val < _SELECTION_WINDOWS,
            support_train=support_train,
            support_validation=support_validation,
            val_accuracy=accuracy,
            val_macro_f1=macro,
        )
        row = HistoryRow(
            epoch=epoch,
            train_loss=loss,
            val_accuracy=accuracy,
            val_macro_f1=macro,
            seconds=time.perf_counter() - started,
            per_class=per_class,
        )
        history.append(row)
        if on_epoch is not None:
            on_epoch(row.as_dict())
        key = (macro, accuracy)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = metrics
            best_state = _cpu_state(module)
            wait = 0
        else:
            wait += 1
        if wait >= config.patience:
            break
        if should_stop is not None and should_stop():
            break
    if best_state is None or best_metrics is None:
        raise LabError("action_validation_empty", _VAL_EMPTY)
    return best_state, history, best_metrics, best_epoch


def _train_epoch(
    module: Any,
    x: Any,
    y: Any,
    criterion: Any,
    optimizer: Any,
    batch: int,
    generator: Any,
    device: str,
) -> float:
    import torch

    module.train()
    count = int(y.shape[0])
    order = torch.randperm(count, generator=generator).tolist()
    total = 0.0
    seen = 0
    for start in range(0, count, batch):
        index = order[start : start + batch]
        xb = x[index].to(device)
        yb = y[index].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(module(xb), yb)
        loss.backward()
        optimizer.step()
        size = len(index)
        total += float(loss.detach().cpu()) * size
        seen += size
    return total / seen


def _predict_tensor(module: Any, x: Any, batch: int, device: str) -> np.ndarray:
    import torch

    module.eval()
    preds: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, int(x.shape[0]), batch):
            logits = module(x[start : start + batch].to(device))
            preds.append(logits.argmax(dim=1).detach().cpu().numpy())
    return np.concatenate(preds).astype(np.int64, copy=False)


def _cpu_state(module: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _metadata(
    names: Sequence[str],
    config: ActionTrainConfig,
    window_frames: int,
    feature_spec: FeatureSpec | None,
    standardizer: Standardizer,
    metrics: ValidationMetrics,
) -> dict[str, Any]:
    window = config.window.model_dump(mode="json")
    window["window_frames"] = window_frames
    return {
        "class_order": list(names),
        "window": window,
        "feature_spec": _feature_document(feature_spec),
        "standardizer": standardizer.to_json(),
        "classes_scored": list(metrics.classes_scored),
        "selection_unstable": metrics.selection_unstable,
    }


def _feature_document(feature_spec: FeatureSpec | None) -> dict[str, Any] | None:
    if feature_spec is None:
        return None
    if not isinstance(feature_spec, FeatureSpec):
        raise LabError("feature_spec_mismatch", _BAD_SPEC)
    return feature_spec.model_dump(mode="json")


def _write_training(output_dir: Path, state: Mapping[str, Any], standardizer: Standardizer) -> Path:
    import torch

    if output_dir.exists() and not output_dir.is_dir():
        raise LabError("invalid_action_train", _BAD_DIR)
    with map_os_error(str(output_dir)):
        output_dir.mkdir(parents=True, exist_ok=True)
    document = json.dumps(
        standardizer.to_json(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    _atomic_text(output_dir / STANDARDIZER_NAME, document + "\n")
    weights = output_dir / WEIGHTS_NAME
    tmp = weights.with_name(f".{weights.name}.tmp")
    try:
        with map_os_error(str(weights)):
            torch.save(dict(state), tmp)
            os.replace(tmp, weights)
    finally:
        tmp.unlink(missing_ok=True)
    return weights


def _atomic_text(path: Path, payload: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with map_os_error(str(path)):
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _config(config: ActionTrainConfig) -> ActionTrainConfig:
    if not isinstance(config, ActionTrainConfig):
        raise LabError("invalid_action_train", _BAD_CONFIG)
    if not isinstance(config.device, str) or not config.device:
        raise LabError("invalid_action_train", _BAD_CONFIG)
    return config


def _class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(class_names, (str, bytes)) or not isinstance(class_names, Sequence):
        raise LabError("invalid_action", _BAD_NAMES)
    names = tuple(class_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise LabError("invalid_action", _BAD_NAMES)
    if len(set(names)) != len(names):
        raise LabError("invalid_action", _BAD_NAMES)
    return names


def _as_xy(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    n_features: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        values = np.array(x, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise LabError("invalid_window", _BAD_ARRAY) from exc
    if values.ndim != 3 or values.shape[1] < 1 or values.shape[2] < 1:
        raise LabError("invalid_window", _BAD_ARRAY)
    if n_features is not None and int(values.shape[2]) != n_features:
        raise LabError("feature_spec_mismatch", _BAD_FEATURES)
    labels = _as_labels(y, n_classes, n=int(values.shape[0]))
    return values, labels


def _as_labels(y: np.ndarray, n_classes: int, *, n: int | None = None) -> np.ndarray:
    try:
        labels = np.array(y, copy=True)
    except (TypeError, ValueError) as exc:
        raise LabError("invalid_action_label", _BAD_LABELS) from exc
    if labels.ndim != 1:
        raise LabError("invalid_action_label", _BAD_LABELS)
    if n is not None and int(labels.shape[0]) != n:
        raise LabError("invalid_action_label", _BAD_LABELS)
    if labels.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if labels.dtype.kind not in {"i", "u"}:
        if labels.dtype.kind != "f" or not bool(np.isfinite(labels).all()):
            raise LabError("invalid_action_label", _BAD_LABELS)
        if not bool(np.all(labels == np.floor(labels))):
            raise LabError("invalid_action_label", _BAD_LABELS)
    labels = labels.astype(np.int64, copy=False)
    if int(labels.min()) < 0 or int(labels.max()) >= n_classes:
        raise LabError("invalid_action_label", _BAD_LABELS)
    return labels


def _count(labels: np.ndarray, names: Sequence[str]) -> dict[str, int]:
    if labels.size == 0:
        counts = np.zeros(len(names), dtype=np.int64)
    else:
        counts = np.bincount(labels, minlength=len(names))
    return {name: int(counts[index]) for index, name in enumerate(names)}


def _finite(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        raise LabError("invalid_window", _INFINITE)
    return values


def _missing_message(names: Sequence[str]) -> str:
    quoted = ", ".join(f"«{name}»" for name in names)
    if len(names) == 1:
        return f"В обучающих окнах нет класса {quoted}. Обучение действий не запускается."
    return f"В обучающих окнах нет классов {quoted}. Обучение действий не запускается."


def _require_windows(windows: Sequence[ActionWindows]) -> list[ActionWindows]:
    if isinstance(windows, (str, bytes)) or not isinstance(windows, Sequence):
        raise LabError("invalid_window", _BAD_WINDOWS)
    rows: list[ActionWindows] = []
    for item in windows:
        if not isinstance(item, ActionWindows):
            raise LabError("invalid_window", _BAD_WINDOWS)
        if not isinstance(item.role, str) or item.role not in {"train", "validation", "test"}:
            raise LabError("invalid_role", _BAD_ROLE)
        rows.append(item)
    return rows


def _template(parts: Sequence[ActionWindows]) -> tuple[int, int] | None:
    for part in parts:
        array = np.asarray(part.x)
        if array.ndim == 3 and array.shape[1] >= 1 and array.shape[2] >= 1:
            return int(array.shape[1]), int(array.shape[2])
    return None


def _concat(
    parts: Sequence[ActionWindows],
    template: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not parts:
        time_steps, features = template or (1, 1)
        return (
            np.zeros((0, time_steps, features), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    values = [np.array(part.x, dtype=np.float64, copy=True) for part in parts]
    labels = [np.array(part.y, dtype=np.int64, copy=True) for part in parts]
    return np.concatenate(values, axis=0), np.concatenate(labels, axis=0)


def check_ready(
    workspace: Path,
    project_id: str,
    run_ids: Sequence[str],
) -> Any:
    """Readiness of custom analysis runs. Reasons are printed verbatim."""
    from excavation_lab.actions.job import check_ready as _check_ready

    return _check_ready(workspace, project_id, run_ids)


def enqueue_action_train(
    workspace: Path,
    project_id: str,
    run_ids: list[str],
    config: ActionTrainConfig,
) -> Any:
    """Queue one ``action_train`` job. Does not build windows in this process."""
    from excavation_lab.actions.job import enqueue_action_train as _enqueue

    return _enqueue(workspace, project_id, run_ids, config)


def run_action_train_job(workspace: Path, job_id: str) -> int:
    """Worker entry used by ``python -m excavation_lab.jobs``."""
    from excavation_lab.actions.job import run_action_train_job as _run

    return _run(workspace, job_id)
