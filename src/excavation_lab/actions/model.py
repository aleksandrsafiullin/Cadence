"""Tiny temporal classifier for action windows.

``ActionNet`` is a 1D conv over a window of feature vectors. It is not a
detector and does not construct RF-DETR. Torch is imported when a net is
built, a seed is set, or a weight file is read.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import numpy as np

from excavation_lab.errors import LabError

if TYPE_CHECKING:
    from pathlib import Path

    import torch
    from torch import nn

_STD_FLOOR = 1e-6
_BAD_ARCH = "Размеры ActionNet должны быть положительными целыми."
_BAD_FIT = "Стандартизатору нужен непустой массив признаков."
_BAD_SCALE = "Число признаков не совпадает со стандартизатором."
_BAD_STD = "В файле стандартизатора нет конечных mean и std."
_INFINITE = "В признаках окна бесконечное значение. Стандартизатор не строится."
_WEIGHTS = "Нет файла весов классификатора действий."
_SEED = "Зерно обучения должно быть целым числом."

_NET: type | None = None


def set_seed(seed: int = 42) -> None:
    """Seed torch and numpy. The trainer calls this before building the net."""
    import torch

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise LabError("invalid_action_train", _SEED)
    torch.manual_seed(seed)
    np.random.seed(seed)


class ActionNet:
    """Conv1d → ReLU → Conv1d → ReLU → mean over time → linear logits.

    Input layout is ``(batch, time, features)``, the same layout as
    ``ActionWindows.x``. Constructing the net imports torch.
    """

    def __new__(cls, n_features: int, n_classes: int, hidden: int = 64) -> nn.Module:
        _check_arch(n_features, n_classes, hidden)
        return _module_type()(n_features, n_classes, hidden)


class Standardizer:
    """Per-feature mean and std fit on train windows only.

    NaN is ignored while fitting. ``apply`` standardises and then writes 0
    where the input was NaN, so a mask column can mark a missing box.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        center = np.asarray(mean, dtype=np.float64).reshape(-1)
        scale = np.asarray(std, dtype=np.float64).reshape(-1)
        if center.shape != scale.shape or center.size == 0:
            raise LabError("invalid_window", _BAD_STD)
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise LabError("invalid_window", _BAD_STD)
        if np.any(scale < _STD_FLOOR):
            raise LabError("invalid_window", _BAD_STD)
        self.mean = center
        self.std = scale

    @classmethod
    def fit(cls, x_train: np.ndarray) -> Standardizer:
        """Mean and population std over every train element of each feature."""
        values = np.asarray(x_train, dtype=np.float64)
        if values.ndim < 1 or values.shape[-1] < 1 or values.size == 0:
            raise LabError("invalid_window", _BAD_FIT)
        flat = values.reshape(-1, values.shape[-1])
        if flat.shape[0] == 0:
            raise LabError("invalid_window", _BAD_FIT)
        mean = np.zeros(flat.shape[1], dtype=np.float64)
        std = np.ones(flat.shape[1], dtype=np.float64)
        for index in range(flat.shape[1]):
            column = flat[:, index]
            observed = column[~np.isnan(column)]
            if observed.size == 0:
                continue
            center = float(observed.mean())
            spread = float(observed.std(ddof=0))
            if not np.isfinite(center) or not np.isfinite(spread):
                raise LabError("invalid_window", _INFINITE)
            mean[index] = center
            std[index] = max(spread, _STD_FLOOR)
        return cls(mean, std)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Standardise ``x`` and replace NaN with 0. Does not mutate ``x``."""
        values = np.array(x, dtype=np.float64, copy=True)
        if values.shape[-1] != self.mean.shape[0]:
            raise LabError("feature_spec_mismatch", _BAD_SCALE)
        scaled = (values - self.mean) / self.std
        scaled[np.isnan(scaled)] = 0.0
        return scaled

    def to_json(self) -> dict[str, list[float]]:
        return {
            "mean": [float(value) for value in self.mean],
            "std": [float(value) for value in self.std],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Standardizer:
        if not isinstance(payload, dict):
            raise LabError("invalid_window", _BAD_STD)
        mean = payload.get("mean")
        std = payload.get("std")
        if not isinstance(mean, list) or not isinstance(std, list):
            raise LabError("invalid_window", _BAD_STD)
        try:
            return cls(np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64))
        except (TypeError, ValueError) as exc:
            raise LabError("invalid_window", _BAD_STD) from exc


def load_action_state(path: Path) -> dict[str, torch.Tensor]:
    """Load an ``ActionNet`` ``state_dict`` saved with ``torch.save``."""
    import torch

    file = path
    if not file.is_file():
        raise LabError("action_weights_missing", f"{_WEIGHTS} {file}")
    try:
        state = torch.load(file, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as exc:
        raise LabError("action_weights_missing", f"{_WEIGHTS} {file}") from exc
    if not isinstance(state, dict):
        raise LabError("action_weights_missing", f"{_WEIGHTS} {file}")
    return state


def _check_arch(n_features: int, n_classes: int, hidden: int) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (n_features, n_classes, hidden)):
        raise LabError("invalid_action_train", _BAD_ARCH)
    if n_features < 1 or n_classes < 1 or hidden < 1:
        raise LabError("invalid_action_train", _BAD_ARCH)


def _module_type() -> type:
    global _NET
    if _NET is None:
        _NET = _build_module()
    return _NET


def _build_module() -> type:
    import torch
    from torch import nn

    class _Net(nn.Module):
        def __init__(self, n_features: int, n_classes: int, hidden: int) -> None:
            super().__init__()
            self.n_features = n_features
            self.n_classes = n_classes
            self.hidden = hidden
            self.conv1 = nn.Conv1d(n_features, hidden, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
            self.fc = nn.Linear(hidden, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = x.transpose(1, 2)
            hidden = torch.relu(self.conv1(hidden))
            hidden = torch.relu(self.conv2(hidden))
            return self.fc(hidden.mean(dim=-1))

    _Net.__name__ = "ActionNet"
    _Net.__qualname__ = "ActionNet"
    return _Net
