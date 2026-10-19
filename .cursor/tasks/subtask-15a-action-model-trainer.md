# Subtask 15a: `ActionNet`, standardizer, trainer loop (pure compute)

## Goal
A tiny PyTorch temporal classifier and a trainer that selects the checkpoint on validation macro-F1 over classes that actually have validation support, with honest nulls elsewhere.

## Context
Spec §8 roles (validation selects checkpoints), §12.2 analogy (seed 42, early stopping on validation only), §18 (no invented numbers). Inputs are the windows from 14; nothing here reads SQLite or parquet.

## Acceptance Criteria
- [ ] `actions/model.py`: `ActionNet(n_features, n_classes, hidden=64)` = Conv1d(k=3, padding=1) → ReLU → Conv1d(k=3, padding=1) → ReLU → mean-pool over time → Linear; `Standardizer.fit(x_train)` (per-feature mean/std ignoring NaN, std floor 1e-6) — **fitted on train windows only**; `Standardizer.apply(x)` standardises then replaces NaN with 0 (mask columns carry absence); `to_json()/from_json()`; `torch.manual_seed(42)` + numpy seed in `set_seed()`.
- [ ] `actions/train.py`: `ActionTrainConfig(epochs=40, lr=1e-3, batch=64, patience=8, min_confidence=0.5, min_duration_s=1.0, window=WindowConfig())`; `train_action_model(x_train, y_train, x_val, y_val, class_names, config, *, on_epoch=None, should_stop=None) -> TrainOutcome` with: class-weighted cross-entropy from train support; per epoch `train_loss`, `val_accuracy`, `val_macro_f1`, per-class `precision/recall/f1/support`; best state by `val_macro_f1` (ties → `val_accuracy`); early stop after `patience` epochs without improvement; `metrics_history` rows `epoch, train_loss, val_accuracy, val_macro_f1, seconds`.
- [ ] Macro-F1 is averaged **only over classes with `support_validation > 0`**; `TrainOutcome.validation_metrics` stores `classes_scored: list[str]`, per-class metrics `null` for classes with zero validation support, `selection_unstable = n_val_windows < 300`, `support_train`, `support_validation`.
- [ ] Refuses (`action_support_missing`, naming the classes) when any catalog class has 0 train windows; refuses (`action_validation_empty`) when `x_val` is empty.
- [ ] `should_stop()` is polled between epochs (cancel hook for 15b).
- [ ] Tests `tests/unit/test_action_model.py`: overfits a 2-class synthetic set to ≥0.95 train accuracy in <40 epochs on CPU; standardizer ignores NaN in fit and zeroes NaN in apply; a class with validation support 0 gets `null` metrics and is excluded from macro-F1 (`classes_scored` lists the others); `selection_unstable` true below 300 windows; `should_stop` returning True after epoch 2 yields an outcome with 2 history rows; same seed → identical history.

## Implementation Notes
- Files: `src/excavation_lab/actions/{model,train}.py`, `tests/unit/test_action_model.py`.
- CPU by default (model is tiny); accept `device: str = "cpu"` for a later experiment but do not probe MPS here.
- Pure numpy for the metrics; no sklearn.
- No job, SQLite, parquet, or CLI code in this subtask.

## Dependencies
14.

## Verification
```bash
uv run pytest tests/unit/test_action_model.py -q
uv run pytest tests -q --ignore=tests/manual && uv run ruff check src tests
```
