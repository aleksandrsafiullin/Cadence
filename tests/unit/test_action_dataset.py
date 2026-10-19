"""Action windows from synthetic feature rows. No workspace writes."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from excavation_lab.actions.catalog import ActionCatalog, ActionClass
from excavation_lab.actions.dataset import (
    WindowConfig,
    build_windows,
    class_index,
    dataset_hash,
)
from excavation_lab.actions.features import (
    KEY_COLUMNS,
    FeatureSpec,
    concat_runs,
    read_features,
    write_features,
)
from excavation_lab.actions.labels import ActionInterval
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.errors import LabError

_ADDED = datetime(2026, 1, 1, tzinfo=UTC)
_BLOCKS = [
    RoleBlock(start_s=0, end_s=60, role="train"),
    RoleBlock(start_s=60, end_s=80, role="validation"),
    RoleBlock(start_s=80, end_s=100, role="test"),
]


def test_windows_stay_inside_their_role_block(tmp_path: Path) -> None:
    spec = _spec()
    frame = _load(tmp_path, spec, _times(0, 100), "run-a", name="grid")
    catalog = _catalog(("digging", 0), ("idle", 1), ("dumping", 2))
    intervals = [
        _interval("digging", 10, 40, "dig"),
        _interval("idle", 62, 75, "idle"),
        _interval("dumping", 85, 95, "dump"),
    ]
    config = WindowConfig()
    assert (config.window_s, config.stride_frames, config.min_label_coverage) == (3.0, 1, 1.0)
    train = build_windows(frame, spec, intervals, _BLOCKS, "train", config, catalog=catalog)
    column = spec.columns.index("bucket_cx")
    assert train.role == "train"
    assert train.window_frames == 15
    assert train.x.shape[1:] == (15, len(spec.columns))
    assert train.x.shape[0] == train.y.shape[0] == train.center_time_s.shape[0]
    assert float(train.x[:, :, column].max()) < 60
    assert not np.any(np.isclose(train.center_time_s, 59.8))
    assert train.dropped == {
        "crosses_block": 7,
        "crosses_run": 0,
        "outside_role": 193,
        "unlabelled": 150,
        "frame_gap": 0,
    }
    assert train.support == {"digging": 136, "idle": 0, "dumping": 0}
    assert set(train.y.tolist()) == {0}

    validation = build_windows(
        frame, spec, intervals, _BLOCKS, "validation", config, catalog=catalog
    )
    times = validation.x[:, :, column]
    assert validation.role == "validation"
    assert float(times.min()) >= 60
    assert float(times.max()) < 80
    assert validation.dropped["crosses_block"] == 14
    assert validation.dropped["outside_role"] == 386
    assert validation.support == {"digging": 0, "idle": 51, "dumping": 0}
    assert set(validation.y.tolist()) == {1}


def test_window_centred_at_59_9_crossing_60_is_dropped(tmp_path: Path) -> None:
    spec = _spec()
    width = 15
    offset = width // 2
    times = [59.9 + (index - offset) * 0.2 for index in range(width)]
    times[offset] = 59.9
    frame = _load(tmp_path, spec, times, "run-a", name="edge")
    catalog = _catalog(("digging", 0))
    windows = build_windows(
        frame,
        spec,
        [_interval("digging", 58, 60, "edge")],
        _BLOCKS,
        "train",
        WindowConfig(),
        catalog=catalog,
    )
    assert frame["source_time_s"].iloc[offset] == pytest.approx(59.9)
    assert windows.x.shape[0] == 0
    assert windows.dropped["crosses_block"] == 1
    assert windows.dropped["unlabelled"] == 0
    assert windows.dropped["outside_role"] == 0


def test_windows_do_not_cross_runs(tmp_path: Path) -> None:
    spec = _spec()
    first = _table(spec, _times(0, 30), "run-a", run_index=0)
    second = _table(spec, _times(30, 60), "run-b", run_index=1)
    stacked, stacked_spec = concat_runs([first, second], [spec, spec])
    frame = _load_frame(tmp_path, stacked_spec, stacked, name="stacked")
    catalog = _catalog(("digging", 0))
    windows = build_windows(
        frame,
        stacked_spec,
        [_interval("digging", 0, 60, "all")],
        _BLOCKS,
        "train",
        WindowConfig(),
        catalog=catalog,
    )
    column = stacked_spec.columns.index("bucket_cx")
    times = windows.x[:, :, column]
    assert windows.dropped["crosses_run"] == 14
    assert windows.support["digging"] == 272
    mixed = (times.min(axis=1) < 30) & (times.max(axis=1) >= 30)
    assert not np.any(mixed)
    assert set(windows.run_id.tolist()) == {"run-a", "run-b"}


def test_windows_refused(tmp_path: Path) -> None:
    spec = _spec()
    frame = _load(tmp_path, spec, _times(80, 90), "run-c", name="test-block")
    catalog = _catalog(("dumping", 0))
    intervals = [_interval("dumping", 82, 88, "dump")]
    with pytest.raises(LabError) as refused:
        build_windows(
            frame,
            spec,
            intervals,
            _BLOCKS,
            "test",
            WindowConfig(window_s=0.6),
            catalog=catalog,
        )
    assert refused.value.code == "test_windows_refused"
    with pytest.raises(LabError) as again:
        build_windows(None, None, (), (), "test", None, catalog=None)  # type: ignore[arg-type]
    assert again.value.code == "test_windows_refused"
    assert "allow_test" not in inspect.signature(build_windows).parameters
    with pytest.raises(ValidationError):
        WindowConfig(allow_test=True)  # type: ignore[call-arg]


def test_centre_in_test_is_refused_even_if_rows_were_passed(tmp_path: Path) -> None:
    spec = _spec()
    frame = _load(tmp_path, spec, _times(82, 86), "run-c", name="passed-in")
    catalog = _catalog(("digging", 0), ("dumping", 1))
    windows = build_windows(
        frame,
        spec,
        [_interval("dumping", 82, 86, "dump")],
        _BLOCKS,
        "train",
        WindowConfig(window_s=0.6),
        catalog=catalog,
    )
    assert windows.x.shape[0] == 0
    assert windows.center_time_s.size == 0
    assert windows.role == "train"
    assert windows.dropped["outside_role"] == 18
    assert windows.support == {"digging": 0, "dumping": 0}
    assert windows.y.dtype == np.int64


def test_coverage_below_the_centre_interval_is_unlabelled(tmp_path: Path) -> None:
    spec = _spec()
    # 4 of 5 frames lie in [0.2, 1.0). The centre frame is 0.4.
    frame = _load(tmp_path, spec, [0.0, 0.2, 0.4, 0.6, 0.8], "run-a", name="cover")
    catalog = _catalog(("idle", 1), ("digging", 4))
    intervals = [_interval("digging", 0.2, 1.0, "dig")]
    strict = build_windows(
        frame,
        spec,
        intervals,
        _BLOCKS,
        "train",
        WindowConfig(window_s=1.0, min_label_coverage=1.0),
        catalog=catalog,
    )
    assert strict.x.shape[0] == 0
    assert strict.dropped["unlabelled"] == 1
    assert strict.support == {"idle": 0, "digging": 0}
    for minimum in (0.8, 0.75):
        kept = build_windows(
            frame,
            spec,
            intervals,
            _BLOCKS,
            "train",
            WindowConfig(window_s=1.0, min_label_coverage=minimum),
            catalog=catalog,
        )
        assert kept.x.shape[0] == 1
        assert kept.y.tolist() == [class_index(catalog)["digging"]]
        assert kept.y.tolist() == [1]
        assert kept.center_time_s.tolist() == pytest.approx([0.4])
        assert kept.dropped["unlabelled"] == 0
        assert kept.support == {"idle": 0, "digging": 1}


def test_support_matches_intervals(tmp_path: Path) -> None:
    spec = _spec()
    frame = _load(tmp_path, spec, _times(0, 6), "run-a", name="support")
    catalog = _catalog(("digging", 0), ("idle", 2))
    windows = build_windows(
        frame,
        spec,
        [
            _interval("digging", 1.0, 3.0, "dig"),
            _interval("idle", 4.0, 5.0, "idle"),
        ],
        _BLOCKS,
        "train",
        WindowConfig(window_s=0.6),
        catalog=catalog,
    )
    assert windows.support == {"digging": 8, "idle": 3}
    assert windows.dropped["unlabelled"] == 17
    digging = windows.center_time_s[windows.y == 0]
    idle = windows.center_time_s[windows.y == 1]
    assert digging.tolist() == pytest.approx([round(1.2 + 0.2 * i, 10) for i in range(8)])
    assert idle.tolist() == pytest.approx([4.2, 4.4, 4.6])


def test_stride_steps_the_centre(tmp_path: Path) -> None:
    spec = _spec()
    frame = _load(tmp_path, spec, _times(0, 6), "run-a", name="stride")
    catalog = _catalog(("digging", 0))
    windows = build_windows(
        frame,
        spec,
        [_interval("digging", 0, 6, "dig")],
        _BLOCKS,
        "train",
        WindowConfig(window_s=0.6, stride_frames=2),
        catalog=catalog,
    )
    assert windows.center_time_s.tolist() == pytest.approx([0.2 + 0.4 * i for i in range(14)])
    assert 0.4 not in set(np.round(windows.center_time_s, 10))


def test_dataset_hash_changes_when_an_interval_moves() -> None:
    spec = _spec()
    catalog = _catalog(("digging", 0), ("idle", 1))
    config = WindowConfig()
    blocks_hash = "ab" * 32
    intervals = [
        _interval("idle", 4, 5, "idle"),
        _interval("digging", 1, 3, "dig"),
    ]
    moved = [
        _interval("idle", 4, 5, "idle"),
        _interval("digging", 1, 3.5, "dig"),
    ]
    original = dataset_hash(spec, intervals, blocks_hash, config, catalog)
    assert original == dataset_hash(spec, list(reversed(intervals)), blocks_hash, config, catalog)
    assert original != dataset_hash(spec, moved, blocks_hash, config, catalog)
    assert original != dataset_hash(spec, intervals, "cd" * 32, config, catalog)
    assert len(original) == 64
    assert original == original.lower()


def test_class_index_follows_catalog_id_order_and_omits_unknown() -> None:
    catalog = _catalog(("idle", 2), ("unknown", 1), ("digging", 0))
    assert class_index(catalog) == {"digging": 0, "idle": 1}
    assert "unknown" not in class_index(catalog)
    with pytest.raises(LabError) as refused:
        build_windows(
            _empty_frame(_spec()),
            _spec(),
            [_interval("unknown", 0, 1, "u")],
            _BLOCKS,
            "train",
            WindowConfig(window_s=0.6),
            catalog=catalog,
        )
    assert refused.value.code == "unknown_action"


def test_frame_gap_drops_the_window(tmp_path: Path) -> None:
    spec = _spec()
    gapped = _load(tmp_path, spec, [0.0, 0.2, 0.4, 1.2, 1.4], "run-a", name="gap")
    catalog = _catalog(("digging", 0))
    windows = build_windows(
        gapped,
        spec,
        [_interval("digging", 0, 2, "dig")],
        _BLOCKS,
        "train",
        WindowConfig(window_s=0.6),
        catalog=catalog,
    )
    assert windows.dropped["frame_gap"] == 2
    assert windows.support["digging"] == 1
    limit = np.float64(2.0) / np.float64(spec.analysis_fps)
    exact = _load(
        tmp_path,
        spec,
        [0.0, 0.25, float(np.float64(0.25) + limit)],
        "run-a",
        name="exact-gap",
    )
    kept = build_windows(
        exact,
        spec,
        [_interval("digging", 0, 2, "dig")],
        _BLOCKS,
        "train",
        WindowConfig(window_s=0.6),
        catalog=catalog,
    )
    assert kept.dropped["frame_gap"] == 0
    assert kept.support["digging"] == 1


def test_window_crossing_two_blocks_of_one_role_is_dropped(tmp_path: Path) -> None:
    spec = _spec()
    blocks = [
        RoleBlock(start_s=0, end_s=10, role="train"),
        RoleBlock(start_s=10, end_s=20, role="train"),
    ]
    frame = _load(tmp_path, spec, [9.6, 9.8, 10.0], "run-a", name="blocks")
    windows = build_windows(
        frame,
        spec,
        [_interval("digging", 0, 10, "dig")],
        blocks,
        "train",
        WindowConfig(window_s=0.6),
        catalog=_catalog(("digging", 0)),
    )
    assert windows.x.shape[0] == 0
    assert windows.dropped["crosses_block"] == 1


def test_interval_that_crosses_a_block_does_not_label(tmp_path: Path) -> None:
    spec = _spec()
    frame = _load(tmp_path, spec, [58.6, 58.8, 59.0], "run-a", name="cross-label")
    windows = build_windows(
        frame,
        spec,
        [_interval("digging", 58, 62, "cross")],
        _BLOCKS,
        "train",
        WindowConfig(window_s=0.6),
        catalog=_catalog(("digging", 0)),
    )
    assert windows.x.shape[0] == 0
    assert windows.dropped["unlabelled"] == 1
    assert windows.dropped["crosses_block"] == 0


def test_short_window_still_has_three_frames(tmp_path: Path) -> None:
    spec = _spec()
    frame = _load(tmp_path, spec, [0.0, 0.2, 0.4], "run-a", name="short")
    windows = build_windows(
        frame,
        spec,
        [_interval("digging", 0, 1, "dig")],
        _BLOCKS,
        "train",
        WindowConfig(window_s=0.1),
        catalog=_catalog(("digging", 0)),
    )
    assert windows.window_frames == 3
    assert windows.x.shape == (1, 3, len(spec.columns))


def _spec() -> FeatureSpec:
    return FeatureSpec(class_names=("bucket",), analysis_fps=5)


def _times(start: float, end: float, fps: float = 5.0) -> list[float]:
    count = round((end - start) * fps)
    return [round(start + index / fps, 10) for index in range(count)]


def _catalog(*pairs: tuple[str, int]) -> ActionCatalog:
    classes = [
        ActionClass(id=class_id, name=name, title=name, added_at=_ADDED)
        for name, class_id in pairs
    ]
    return ActionCatalog(classes=classes)


def _interval(name: str, start: float, end: float, label_id: str) -> ActionInterval:
    return ActionInterval(
        label_id=label_id,
        video_id="vid",
        start_s=start,
        end_s=end,
        action_name=name,
        annotator="operator",
        created_at="2026-01-01T00:00:00+00:00",
    )


def _table(
    spec: FeatureSpec,
    times: list[float],
    run_id: str,
    run_index: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, time_s in enumerate(times):
        row: dict[str, object] = {}
        for column in spec.columns:
            if column.endswith(("_present", "_mask")):
                row[column] = 0
            elif column == "bucket_cx":
                row[column] = time_s
            else:
                row[column] = 0.0
        row["run_id"] = run_id
        row["run_index"] = run_index
        row["run_start"] = index == 0
        row["source_frame_index"] = index
        row["source_time_s"] = time_s
        rows.append(row)
    columns = list(KEY_COLUMNS) + list(spec.columns)
    return pd.DataFrame(rows, columns=columns)


def _load(
    tmp_path: Path,
    spec: FeatureSpec,
    times: list[float],
    run_id: str,
    *,
    name: str,
    run_index: int = 0,
) -> pd.DataFrame:
    return _load_frame(tmp_path, spec, _table(spec, times, run_id, run_index), name=name)


def _load_frame(
    tmp_path: Path,
    spec: FeatureSpec,
    frame: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    directory = tmp_path / name
    write_features(frame, spec, directory)
    loaded, loaded_spec = read_features(directory)
    assert loaded_spec == spec
    return loaded


def _empty_frame(spec: FeatureSpec) -> pd.DataFrame:
    columns = list(KEY_COLUMNS) + list(spec.columns)
    return pd.DataFrame(columns=columns)
