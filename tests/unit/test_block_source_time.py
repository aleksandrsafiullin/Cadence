"""Block edges are source seconds.

A synthetic probe row is enough. These numbers are not pit-video accuracy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from excavation_lab.datasets.blocks import (
    RoleBlock,
    assignment_at,
    validate_role_blocks,
)
from excavation_lab.errors import LabError, TimebaseError
from excavation_lab.media.timebase import (
    TimeBase,
    cfr_fallback_time_s,
    seconds_to_ticks,
    source_time_s,
)
from excavation_lab.ui.data_page import source_interval_from_player

_MAP_HEADER = "preview_time_s,source_time_s,source_frame_index,repeated,gap_slate\n"


class _Raw:
    def __init__(self, start_s: object, end_s: object, role: object) -> None:
        self.start_s = start_s
        self.end_s = end_s
        self.role = role


def _row(
    offset_s: float | None,
    *,
    time_base: str | None = "1/1000",
) -> dict[str, object]:
    return {"video_id": "clip", "start_offset_s": offset_s, "time_base": time_base}


def _write_map(path: Path, pairs: list[tuple[float, float]]) -> None:
    lines = [_MAP_HEADER]
    for index, (preview_s, source_s) in enumerate(pairs):
        lines.append(f"{preview_s:.6f},{source_s:.6f},{index},false,false\n")
    path.write_text("".join(lines), encoding="utf-8")


def test_first_ten_seconds_keep_the_probed_offset() -> None:
    video = _row(300.0)
    base = TimeBase.parse("1/1000")
    expected = (
        source_time_s(seconds_to_ticks(0.0, base), base, 300.0),
        source_time_s(seconds_to_ticks(10.0, base), base, 300.0),
    )
    assert source_interval_from_player(video, 0.0, 10.0) == expected
    assert expected == (300.0, 310.0)
    assert expected != (0.0, 10.0)
    assert expected != (
        cfr_fallback_time_s(0, 25.0),
        cfr_fallback_time_s(10, 25.0),
    )
    later = source_interval_from_player(_row(120.0), 0.0, 10.0)
    assert later == (120.0, 130.0)
    assert later != expected


def test_half_open_on_the_offset_clock() -> None:
    start_s, end_s = source_interval_from_player(_row(300.0), 0.0, 10.0)
    blocks = validate_role_blocks(
        [
            RoleBlock(start_s=start_s, end_s=end_s, role="train"),
            RoleBlock(start_s=end_s, end_s=end_s + 10.0, role="test"),
        ]
    )
    assert [(block.start_s, block.end_s) for block in blocks] == [
        (300.0, 310.0),
        (310.0, 320.0),
    ]
    assert assignment_at(blocks, 300.0) == "train"
    assert assignment_at(blocks, 310.0) == "test"
    alone = validate_role_blocks([RoleBlock(start_s=300.0, end_s=310.0, role="train")])
    assert assignment_at(alone, 310.0) is None
    with pytest.raises(LabError) as overlap:
        validate_role_blocks(
            [
                RoleBlock(start_s=300.0, end_s=310.0, role="train"),
                RoleBlock(start_s=305.0, end_s=315.0, role="validation"),
            ]
        )
    assert overlap.value.code == "block_overlap"


def test_player_edge_follows_source_time_s() -> None:
    base = TimeBase.parse("1/2")
    expected = (
        source_time_s(seconds_to_ticks(0.2, base), base, 300.0),
        source_time_s(seconds_to_ticks(0.6, base), base, 300.0),
    )
    assert source_interval_from_player(_row(300.0, time_base="1/2"), 0.2, 0.6) == expected
    assert expected != (300.2, 300.6)
    assert expected != (0.2, 0.6)


def test_preview_map_is_applied_before_save(tmp_path: Path) -> None:
    path = tmp_path / "preview_time_map.csv"
    _write_map(path, [(0.0, 300.0), (10.0, 400.0)])
    stored = source_interval_from_player(_row(300.0), 0.0, 10.0, preview_map=path)
    assert stored == (300.0, 400.0)
    assert stored != (0.0, 10.0)
    assert stored != (300.0, 310.0)
    between = source_interval_from_player(_row(300.0), 0.0, 2.5, preview_map=path)
    assert between == (300.0, 325.0)
    assert between != (0.0, 2.5)


def test_missing_preview_map_does_not_assume_source_zero(tmp_path: Path) -> None:
    missing = tmp_path / "preview_time_map.csv"
    with pytest.raises(LabError) as caught:
        source_interval_from_player(_row(300.0), 0.0, 10.0, preview_map=missing)
    assert caught.value.code == "source_clock_unknown"
    assert "не рассчитано" in caught.value.message
    empty = tmp_path / "empty.csv"
    empty.write_text(_MAP_HEADER, encoding="utf-8")
    with pytest.raises(LabError) as header_only:
        source_interval_from_player(_row(300.0), 0.0, 10.0, preview_map=empty)
    assert header_only.value.code == "source_clock_unknown"
    assert "не рассчитано" in header_only.value.message


def test_preview_time_outside_the_map_is_not_source_zero(tmp_path: Path) -> None:
    path = tmp_path / "preview_time_map.csv"
    _write_map(path, [(1.0, 300.0), (2.0, 301.0)])
    with pytest.raises(LabError) as caught:
        source_interval_from_player(_row(300.0), 0.0, 1.5, preview_map=path)
    assert caught.value.code == "source_clock_unknown"
    assert "не рассчитано" in caught.value.message


def test_corrupt_preview_map_is_a_timebase_error(tmp_path: Path) -> None:
    path = tmp_path / "preview_time_map.csv"
    path.write_text(_MAP_HEADER + "abc,0,0,false,false\n", encoding="utf-8")
    with pytest.raises(TimebaseError) as caught:
        source_interval_from_player(_row(300.0), 0.0, 10.0, preview_map=path)
    assert caught.value.code == "timebase"


def test_decreasing_preview_map_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "preview_time_map.csv"
    _write_map(path, [(0.0, 310.0), (10.0, 300.0)])
    with pytest.raises(TimebaseError) as caught:
        source_interval_from_player(_row(300.0), 0.0, 10.0, preview_map=path)
    assert "убыва" in caught.value.message


def test_missing_offset_or_time_base_is_not_zero() -> None:
    for video in (_row(None), _row(300.0, time_base=None), _row(float("nan"))):
        with pytest.raises(LabError) as caught:
            source_interval_from_player(video, 0.0, 10.0)
        assert caught.value.code == "source_clock_unknown"
        assert "не рассчитано" in caught.value.message


def test_known_zero_offset_stays_zero() -> None:
    assert source_interval_from_player(_row(0.0), 0.0, 10.0) == (0.0, 10.0)


def test_decreasing_timestamp_and_negative_duration() -> None:
    with pytest.raises(TimebaseError) as negative:
        source_interval_from_player(_row(300.0), 10.0, 0.0)
    assert "Отрицательная длительность" in negative.value.message

    with pytest.raises(TimebaseError) as decreasing:
        source_interval_from_player(_row(300.0), -1.0, 4.0)
    assert "убыва" in decreasing.value.message

    with pytest.raises(LabError) as empty:
        source_interval_from_player(_row(300.0), 4.0, 4.0)
    assert empty.value.code == "invalid_interval"

    with pytest.raises(TimebaseError) as negative_start:
        validate_role_blocks([_Raw(-1.0, 5.0, "train")])
    assert "неотрицательн" in negative_start.value.message


def test_clock_modules_do_not_import_starlette() -> None:
    root = Path(validate_role_blocks.__code__.co_filename).resolve().parents[1]
    for relative in ("datasets/blocks.py", "media/timebase.py", "media/encoder.py"):
        text = (root / relative).read_text(encoding="utf-8").casefold()
        assert "starlette" not in text
