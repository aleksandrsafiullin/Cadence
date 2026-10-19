"""Windows of action features inside one role block and one run.

The label is the action interval that contains the centre frame. Coverage is
the share of that window's frames that also lie in the interval. The test
role is refused: evaluation compares per-frame predictions with intervals
and does not build windows. The standardizer is fit later, on train windows
only. The caller passes blocks from ``require_frozen``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import Field

from excavation_lab.actions.catalog import ActionCatalog, ActionClass
from excavation_lab.actions.features import KEY_COLUMNS, FeatureSpec
from excavation_lab.actions.labels import ActionInterval, role_of
from excavation_lab.datasets.blocks import RoleBlock, validate_role_blocks
from excavation_lab.datasets.splits import canonical_json_bytes
from excavation_lab.errors import LabError
from excavation_lab.schemas import StrictModel

_DROP_REASONS: tuple[str, ...] = (
    "crosses_block",
    "crosses_run",
    "outside_role",
    "unlabelled",
    "frame_gap",
)
_TRAIN_ROLES = frozenset({"train", "validation"})
_TEST_WINDOWS = (
    "Окна для роли test не строятся. "
    "Оценка сравнивает покадровые предсказания с интервалами."
)
_BAD_ROLE = "Роль окон должна быть train или validation."
_BAD_CATALOG = "Нужен каталог действий, чтобы пронумеровать классы."
_BAD_SPEC = "Нужна FeatureSpec с порядком классов и analysis_fps."
_BAD_CONFIG = "Нужна конфигурация окна."
_BAD_INTERVALS = "Некорректный список интервалов действия."
_BAD_BLOCKS = "Нужны замороженные блоки ролей."
_BAD_FRAME = "Таблица признаков не совпадает со спецификацией."
_BAD_TIME = "У кадра нет конечного source_time_s."
_BAD_RUN = "У кадра нет run_id."
_UNKNOWN_ACTION = "Действия «{name}» нет в каталоге."
_OVERLAP = "Интервалы действий пересекаются. Одна секунда не относится к двум действиям."
_BAD_HASH = "Нужен blocks_hash замороженного разбиения."


class WindowConfig(StrictModel):
    """Length, stride, and how much of a window must sit in its centre interval.

    ``window_frames`` is ``max(3, round(window_s * analysis_fps))`` at build
    time. It is not stored here because the fps belongs to the feature spec.
    """

    window_s: float = Field(default=3.0, gt=0, allow_inf_nan=False)
    stride_frames: int = Field(default=1, ge=1)
    min_label_coverage: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)


@dataclass(frozen=True, slots=True)
class ActionWindows:
    """One role's windows. ``x`` is ``[n, window_frames, n_features]``."""

    x: np.ndarray
    y: np.ndarray
    center_time_s: np.ndarray
    run_id: np.ndarray
    support: dict[str, int]
    dropped: dict[str, int]
    role: str
    window_frames: int


def class_index(catalog: ActionCatalog) -> dict[str, int]:
    """Map action name to ``0 .. n-1`` in catalog id order.

    ``unknown`` is not inserted. Inference uses that token for a frame that
    is not a trained class.
    """
    if not isinstance(catalog, ActionCatalog):
        raise LabError("invalid_action", _BAD_CATALOG)
    ordered = [item for item in _by_id(catalog) if item.name != "unknown"]
    return {item.name: index for index, item in enumerate(ordered)}


def dataset_hash(
    features_spec: FeatureSpec,
    intervals: Sequence[ActionInterval],
    blocks_hash: str,
    config: WindowConfig,
    catalog: ActionCatalog,
) -> str:
    """SHA-256 of the canonical JSON of the inputs that define the windows."""
    if not isinstance(features_spec, FeatureSpec):
        raise LabError("feature_spec_mismatch", _BAD_SPEC)
    if not isinstance(config, WindowConfig):
        raise LabError("invalid_window", _BAD_CONFIG)
    if not isinstance(catalog, ActionCatalog):
        raise LabError("invalid_action", _BAD_CATALOG)
    if not isinstance(blocks_hash, str) or not blocks_hash:
        raise LabError("invalid_blocks_hash", _BAD_HASH)
    rows = _require_intervals(intervals)
    payload = {
        "blocks_hash": blocks_hash,
        "catalog": [item.model_dump(mode="json") for item in _by_id(catalog)],
        "config": config.model_dump(mode="json"),
        "features_spec": features_spec.model_dump(mode="json"),
        "intervals": [_interval_payload(item) for item in _by_time(rows)],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_windows(
    features_df: pd.DataFrame,
    spec: FeatureSpec,
    intervals: Sequence[ActionInterval],
    frozen_blocks: Sequence[RoleBlock],
    role: str,
    config: WindowConfig,
    *,
    catalog: ActionCatalog,
) -> ActionWindows:
    """Windows whose every frame is in one block of ``role`` and one run.

    ``role="test"`` raises ``test_windows_refused`` before the table is read.
    There is no flag that turns that off. A window whose centre lies in
    another role is ``outside_role`` even when those rows were passed in.
    """
    if role == "test":
        raise LabError("test_windows_refused", _TEST_WINDOWS)
    if role not in _TRAIN_ROLES:
        raise LabError("invalid_role", _BAD_ROLE)
    if not isinstance(catalog, ActionCatalog):
        raise LabError("invalid_action", _BAD_CATALOG)
    if not isinstance(spec, FeatureSpec):
        raise LabError("feature_spec_mismatch", _BAD_SPEC)
    if not isinstance(config, WindowConfig):
        raise LabError("invalid_window", _BAD_CONFIG)
    names = class_index(catalog)
    blocks = _require_blocks(frozen_blocks)
    usable = _usable_intervals(intervals, blocks, role, names)
    frame = _require_frame(features_df, spec)
    return _windows(frame, spec, usable, blocks, role, config, names)


def _windows(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    intervals: Sequence[ActionInterval],
    blocks: Sequence[RoleBlock],
    role: str,
    config: WindowConfig,
    names: dict[str, int],
) -> ActionWindows:
    width = _window_frames(config.window_s, spec.analysis_fps)
    limit = 2.0 / spec.analysis_fps
    times = _times(frame)
    run_start = _run_start(frame)
    run_ids = _run_ids(frame)
    values = frame.loc[:, list(spec.columns)].to_numpy(dtype=np.float64, copy=True)
    support = {name: 0 for name in names}
    dropped = {reason: 0 for reason in _DROP_REASONS}
    kept_x: list[np.ndarray] = []
    kept_y: list[int] = []
    kept_time: list[float] = []
    kept_run: list[str] = []
    count = int(times.shape[0])
    if count >= width:
        starts = range(0, count - width + 1, config.stride_frames)
    else:
        starts = range(0)
    for start in starts:
        reason, label = _classify(
            times,
            run_start,
            run_ids,
            start,
            width,
            limit,
            blocks,
            role,
            intervals,
            config.min_label_coverage,
        )
        if reason is not None:
            dropped[reason] += 1
            continue
        assert label is not None
        centre = start + width // 2
        kept_x.append(values[start : start + width])
        kept_y.append(names[label.action_name])
        kept_time.append(float(times[centre]))
        kept_run.append(str(run_ids[centre]))
        support[label.action_name] += 1
    return _pack(kept_x, kept_y, kept_time, kept_run, support, dropped, role, width, values.shape[1])


def _classify(
    times: np.ndarray,
    run_start: np.ndarray,
    run_ids: np.ndarray,
    start: int,
    width: int,
    limit: float,
    blocks: Sequence[RoleBlock],
    role: str,
    intervals: Sequence[ActionInterval],
    min_coverage: float,
) -> tuple[str | None, ActionInterval | None]:
    """One drop reason, or the centre interval when the window is an example.

    The centre frame is ``width // 2`` (the later middle when the length is
    even). Coverage counts frames inside that interval, not the other labels
    the window happens to touch.
    """
    stop = start + width
    # A run_start after the first frame, or two run ids, spans two runs.
    if bool(run_start[start + 1 : stop].any()):
        return "crosses_run", None
    if len(set(run_ids[start:stop].tolist())) > 1:
        return "crosses_run", None
    deltas = np.diff(times[start:stop])
    if bool(np.any(deltas <= 0.0) or np.any(deltas > limit)):
        return "frame_gap", None
    centre_time = float(times[start + width // 2])
    block = _block_at(blocks, centre_time)
    if block is None or block.role != role:
        return "outside_role", None
    window = times[start:stop]
    if bool(np.any(window < block.start_s) or np.any(window >= block.end_s)):
        return "crosses_block", None
    label = _interval_at(intervals, centre_time)
    if label is None:
        return "unlabelled", None
    inside = int(np.count_nonzero((window >= label.start_s) & (window < label.end_s)))
    if inside / width < min_coverage:
        return "unlabelled", None
    return None, label


def _pack(
    kept_x: list[np.ndarray],
    kept_y: list[int],
    kept_time: list[float],
    kept_run: list[str],
    support: dict[str, int],
    dropped: dict[str, int],
    role: str,
    width: int,
    n_features: int,
) -> ActionWindows:
    if kept_x:
        features = np.stack(kept_x, axis=0)
    else:
        features = np.zeros((0, width, n_features), dtype=np.float64)
    return ActionWindows(
        x=features,
        y=np.asarray(kept_y, dtype=np.int64),
        center_time_s=np.asarray(kept_time, dtype=np.float64),
        run_id=np.asarray(kept_run, dtype=object),
        support=support,
        dropped=dropped,
        role=role,
        window_frames=width,
    )


def _window_frames(window_s: float, analysis_fps: float) -> int:
    return max(3, round(window_s * analysis_fps))


def _usable_intervals(
    intervals: Sequence[ActionInterval],
    blocks: Sequence[RoleBlock],
    role: str,
    names: dict[str, int],
) -> list[ActionInterval]:
    """Intervals fully inside one block of ``role``. Others are not labels."""
    rows = _require_intervals(intervals)
    usable: list[ActionInterval] = []
    for item in rows:
        if item.action_name not in names or item.action_name == "unknown":
            raise LabError("unknown_action", _UNKNOWN_ACTION.format(name=item.action_name))
        if role_of(item, blocks) != role:
            continue
        usable.append(item)
    _require_disjoint(usable)
    return usable


def _require_intervals(intervals: Sequence[ActionInterval]) -> list[ActionInterval]:
    if isinstance(intervals, (str, bytes)) or not isinstance(intervals, Sequence):
        raise LabError("invalid_action_label", _BAD_INTERVALS)
    rows: list[ActionInterval] = []
    for item in intervals:
        if not isinstance(item, ActionInterval):
            raise LabError("invalid_action_label", _BAD_INTERVALS)
        rows.append(item)
    return rows


def _require_blocks(frozen_blocks: Sequence[RoleBlock]) -> list[RoleBlock]:
    if isinstance(frozen_blocks, (str, bytes)) or not isinstance(frozen_blocks, Sequence):
        raise LabError("invalid_blocks", _BAD_BLOCKS)
    try:
        return validate_role_blocks(frozen_blocks)
    except (TypeError, ValueError) as exc:
        raise LabError("invalid_blocks", _BAD_BLOCKS) from exc


def _require_frame(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise LabError("feature_spec_mismatch", _BAD_FRAME)
    expected = list(KEY_COLUMNS) + list(spec.columns)
    if list(frame.columns) != expected:
        raise LabError("feature_spec_mismatch", _BAD_FRAME)
    return frame


def _times(frame: pd.DataFrame) -> np.ndarray:
    values = frame["source_time_s"].to_numpy(dtype=np.float64, copy=True)
    if values.size and not bool(np.isfinite(values).all()):
        raise LabError("feature_time_missing", _BAD_TIME)
    return values


def _run_start(frame: pd.DataFrame) -> np.ndarray:
    flags: list[bool] = []
    for item in frame["run_start"].tolist():
        if item is None or item is pd.NA:
            raise LabError("feature_spec_mismatch", _BAD_FRAME)
        flags.append(bool(item))
    return np.asarray(flags, dtype=bool)


def _run_ids(frame: pd.DataFrame) -> np.ndarray:
    ids: list[str] = []
    for item in frame["run_id"].tolist():
        if not isinstance(item, str) or not item:
            raise LabError("invalid_run_id", _BAD_RUN)
        ids.append(item)
    return np.asarray(ids, dtype=object)


def _block_at(blocks: Sequence[RoleBlock], time_s: float) -> RoleBlock | None:
    found = [block for block in blocks if block.start_s <= time_s < block.end_s]
    if len(found) != 1:
        return None
    return found[0]


def _interval_at(intervals: Sequence[ActionInterval], time_s: float) -> ActionInterval | None:
    found = [item for item in intervals if item.start_s <= time_s < item.end_s]
    if len(found) > 1:
        raise LabError("action_overlap", _OVERLAP)
    if not found:
        return None
    return found[0]


def _require_disjoint(items: Sequence[ActionInterval]) -> None:
    ordered = sorted(items, key=lambda item: (item.start_s, item.end_s, item.label_id))
    for index, earlier in enumerate(ordered):
        later_items = ordered[index + 1 :]
        for later in later_items:
            if later.start_s >= earlier.end_s:
                break
            if earlier.start_s < later.end_s and later.start_s < earlier.end_s:
                raise LabError("action_overlap", _OVERLAP)


def _by_id(catalog: ActionCatalog) -> list[ActionClass]:
    return sorted(catalog.classes, key=lambda item: item.id)


def _by_time(intervals: Sequence[ActionInterval]) -> list[ActionInterval]:
    return sorted(intervals, key=lambda item: (item.start_s, item.end_s, item.label_id))


def _interval_payload(item: ActionInterval) -> dict[str, object]:
    return {
        "action_name": item.action_name,
        "annotator": item.annotator,
        "created_at": item.created_at,
        "end_s": item.end_s,
        "label_id": item.label_id,
        "start_s": item.start_s,
        "video_id": item.video_id,
    }
