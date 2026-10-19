"""ActionNet on synthetic windows. No detector, no video, no invented mAP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from excavation_lab.actions.dataset import ActionWindows
from excavation_lab.actions.features import FeatureSpec
from excavation_lab.actions.model import Standardizer, load_action_state
from excavation_lab.actions.train import (
    ActionTrainConfig,
    TrainOutcome,
    predict_actions,
    score_predictions,
    split_action_windows,
    train_action_model,
    train_action_windows,
)
from excavation_lab.errors import LabError


def test_standardizer_ignores_nan_and_zeroes_nan_on_apply() -> None:
    raw = np.array(
        [
            [[1.0, np.nan], [3.0, 10.0], [np.nan, np.nan]],
        ],
        dtype=np.float64,
    )
    fitted = Standardizer.fit(raw)
    assert fitted.mean[0] == pytest.approx(2.0)
    assert fitted.std[0] == pytest.approx(1.0)
    assert fitted.mean[1] == pytest.approx(10.0)
    assert fitted.std[1] == pytest.approx(1e-6)
    applied = fitted.apply(np.array([[[np.nan, np.nan]]], dtype=np.float64))
    assert applied.tolist() == [[[0.0, 0.0]]]
    scaled = fitted.apply(np.array([[[1.0, 10.0]]], dtype=np.float64))
    assert scaled[0, 0, 0] == pytest.approx(-1.0)
    assert scaled[0, 0, 1] == pytest.approx(0.0)
    restored = Standardizer.from_json(fitted.to_json())
    assert restored.mean.tolist() == pytest.approx(fitted.mean.tolist())
    assert restored.std.tolist() == pytest.approx(fitted.std.tolist())


def test_macro_f1_skips_a_class_with_no_validation_support() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    per_class, accuracy, macro, scored = score_predictions(y_true, y_pred, ("digging", "swing", "idle"))
    assert scored == ("digging", "swing")
    assert per_class["idle"].precision is None
    assert per_class["idle"].recall is None
    assert per_class["idle"].f1 is None
    assert per_class["idle"].support == 0
    assert per_class["digging"].f1 == pytest.approx(2.0 / 3.0)
    assert per_class["swing"].f1 == pytest.approx(0.8)
    assert accuracy == pytest.approx(0.75)
    expected = ((2.0 / 3.0) + 0.8) / 2.0
    assert macro == pytest.approx(expected)
    assert macro != pytest.approx((expected * 2.0) / 3.0)


def test_separable_set_reaches_high_train_accuracy() -> None:
    x_train, y_train = _separable(48, seed=0)
    x_val, y_val = _separable(12, seed=1)
    outcome = train_action_model(
        x_train,
        y_train,
        x_val,
        y_val,
        ("digging", "idle"),
        ActionTrainConfig(epochs=15, patience=15, lr=1e-2, batch=32),
    )
    predicted = predict_actions(outcome.state_dict, x_train, standardizer=outcome.standardizer)
    accuracy = float(np.mean(predicted == y_train))
    # Class 1 is class 0 plus 5 on feature 0, noise 0.01. The set is separable
    # by construction, so this accuracy is not a quality claim and is not mAP.
    assert accuracy >= 0.95
    assert outcome.validation_metrics.selection_unstable is True
    assert outcome.best_epoch <= 15


def test_absent_validation_class_is_null_on_the_outcome() -> None:
    names = ("digging", "swing", "idle")
    x_train, y_train = _blocks(3, 4, seed=2)
    x_val, y_val = _blocks(3, 4, seed=3, skip={2})
    outcome = train_action_model(
        x_train,
        y_train,
        x_val,
        y_val,
        names,
        ActionTrainConfig(epochs=1, patience=1),
    )
    idle = outcome.validation_metrics.per_class["idle"]
    assert idle.support == 0
    assert idle.precision is None
    assert idle.recall is None
    assert idle.f1 is None
    assert outcome.validation_metrics.classes_scored == ("digging", "swing")
    assert outcome.validation_metrics.support_train["idle"] == 4
    assert outcome.validation_metrics.support_validation["idle"] == 0
    scored = [
        outcome.validation_metrics.per_class[name].f1
        for name in outcome.validation_metrics.classes_scored
    ]
    assert all(value is not None for value in scored)
    assert outcome.validation_metrics.val_macro_f1 == pytest.approx(sum(scored) / len(scored))


def test_selection_is_stable_once_validation_has_300_windows() -> None:
    x_train, y_train = _blocks(2, 4, seed=4)
    x_val, y_val = _blocks(2, 150, seed=5)
    assert y_val.shape[0] == 300
    outcome = train_action_model(
        x_train,
        y_train,
        x_val,
        y_val,
        ("digging", "idle"),
        ActionTrainConfig(epochs=1, patience=1),
    )
    assert outcome.validation_metrics.selection_unstable is False


def test_should_stop_after_epoch_2_keeps_two_rows() -> None:
    x_train, y_train = _blocks(2, 4, seed=6)
    x_val, y_val = _blocks(2, 4, seed=7)
    seen: list[int] = []

    def on_epoch(row: dict[str, object]) -> None:
        seen.append(int(row["epoch"]))

    def stop() -> bool:
        return len(seen) >= 2

    outcome = train_action_model(
        x_train,
        y_train,
        x_val,
        y_val,
        ("digging", "idle"),
        ActionTrainConfig(epochs=10, patience=10),
        on_epoch=on_epoch,
        should_stop=stop,
    )
    assert [row.epoch for row in outcome.metrics_history] == [1, 2]
    assert seen == [1, 2]
    assert set(outcome.metrics_history[0].as_dict()) >= {
        "epoch",
        "train_loss",
        "val_accuracy",
        "val_macro_f1",
        "seconds",
    }


def test_same_seed_replays_the_metric_history() -> None:
    x_train, y_train = _blocks(2, 8, seed=8)
    x_val, y_val = _blocks(2, 8, seed=9)
    config = ActionTrainConfig(epochs=3, patience=3, seed=42)
    first = train_action_model(x_train, y_train, x_val, y_val, ("digging", "idle"), config)
    second = train_action_model(x_train, y_train, x_val, y_val, ("digging", "idle"), config)
    # seconds is wall time, so the seed check is the metric columns.
    assert _metric_body(first) == _metric_body(second)
    assert len(first.metrics_history) == len(second.metrics_history) == 3


def test_missing_train_class_is_named() -> None:
    x_train, y_train = _blocks(2, 4, seed=1, skip={1})
    x_val, y_val = _blocks(2, 2, seed=2)
    with pytest.raises(LabError) as caught:
        train_action_model(
            x_train,
            y_train,
            x_val,
            y_val,
            ("digging", "idle"),
            ActionTrainConfig(epochs=1),
        )
    assert caught.value.code == "action_support_missing"
    assert "idle" in caught.value.message


def test_empty_validation_is_refused() -> None:
    x_train, y_train = _blocks(2, 4, seed=1)
    x_val = np.zeros((0, x_train.shape[1], x_train.shape[2]), dtype=np.float64)
    y_val = np.zeros((0,), dtype=np.int64)
    with pytest.raises(LabError) as caught:
        train_action_model(
            x_train,
            y_train,
            x_val,
            y_val,
            ("digging", "idle"),
            ActionTrainConfig(epochs=1),
        )
    assert caught.value.code == "action_validation_empty"


def test_test_labelled_windows_never_enter_x() -> None:
    train = _role_windows("train", 0.0, np.array([0, 1, 0, 1]))
    held_out = _role_windows("test", 1000.0, np.array([0, 0, 0, 0]))
    validation = _role_windows("validation", -3.0, np.array([0, 1]))
    x_train, y_train, x_val, y_val = split_action_windows([train, held_out, validation])
    assert x_train.shape[0] == 4
    assert x_val.shape[0] == 2
    assert np.all(x_train == 0.0)
    assert np.all(x_val == -3.0)
    assert y_train.tolist() == [0, 1, 0, 1]
    assert y_val.tolist() == [0, 1]
    assert not np.shares_memory(x_train, held_out.x)
    assert not np.shares_memory(x_val, held_out.x)
    assert not np.any(x_train == 1000.0)
    assert not np.any(x_val == 1000.0)
    outcome = train_action_windows(
        [train, held_out, validation],
        ("digging", "idle"),
        ActionTrainConfig(epochs=1, patience=1),
    )
    assert outcome.validation_metrics.support_train == {"digging": 2, "idle": 2}
    assert outcome.standardizer.mean[0] == pytest.approx(0.0)


def test_writes_weights_and_standardizer(tmp_path: Path) -> None:
    spec = FeatureSpec(class_names=("bucket",), analysis_fps=5)
    x_train, y_train = _blocks(2, 4, seed=10)
    x_val, y_val = _blocks(2, 4, seed=11)
    outcome = train_action_model(
        x_train,
        y_train,
        x_val,
        y_val,
        ("digging", "idle"),
        ActionTrainConfig(epochs=1, patience=1),
        output_dir=tmp_path,
        feature_spec=spec,
    )
    weights = tmp_path / "weights.pt"
    standardizer_path = tmp_path / "standardizer.json"
    assert weights.is_file()
    assert standardizer_path.is_file()
    assert outcome.weights_path == weights
    saved = json.loads(standardizer_path.read_text(encoding="utf-8"))
    assert saved == outcome.standardizer.to_json()
    assert saved == outcome.metadata["standardizer"]
    assert outcome.metadata["class_order"] == ["digging", "idle"]
    assert outcome.metadata["classes_scored"] == ["digging", "idle"]
    assert outcome.metadata["selection_unstable"] is True
    assert outcome.metadata["feature_spec"]["class_names"] == ["bucket"]
    assert outcome.metadata["window"]["window_s"] == 3.0
    assert outcome.metadata["window"]["window_frames"] == x_train.shape[1]
    json.dumps(outcome.metadata, allow_nan=False)
    reloaded = load_action_state(weights)
    assert set(reloaded) == set(outcome.state_dict)


def _separable(n_per_class: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    windows = n_per_class * 2
    values = rng.normal(0.0, 0.01, size=(windows, 6, 3)).astype(np.float64)
    labels = np.zeros(windows, dtype=np.int64)
    labels[n_per_class:] = 1
    values[n_per_class:, :, 0] += 5.0
    return values, labels


def _blocks(
    n_classes: int,
    per_class: int,
    *,
    seed: int,
    skip: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    values: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    omitted = skip or set()
    for index in range(n_classes):
        if index in omitted:
            continue
        values.append(rng.normal(float(index), 0.05, size=(per_class, 4, 2)))
        labels.append(np.full(per_class, index, dtype=np.int64))
    return np.concatenate(values).astype(np.float64), np.concatenate(labels)


def _role_windows(role: str, feature: float, labels: np.ndarray) -> ActionWindows:
    values = np.full((labels.shape[0], 3, 1), feature, dtype=np.float64)
    return ActionWindows(
        x=values,
        y=labels.astype(np.int64, copy=True),
        center_time_s=np.zeros(labels.shape[0], dtype=np.float64),
        run_id=np.array(["run"] * labels.shape[0], dtype=object),
        support={},
        dropped={},
        role=role,
        window_frames=3,
    )


def _metric_body(outcome: TrainOutcome) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for row in outcome.metrics_history:
        per = tuple(
            (name, item.precision, item.recall, item.f1, item.support)
            for name, item in row.per_class.items()
        )
        rows.append((row.epoch, row.train_loss, row.val_accuracy, row.val_macro_f1, per))
    return rows
