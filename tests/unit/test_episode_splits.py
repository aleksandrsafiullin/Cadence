"""Episode groups, guard gaps, and the unfrozen split proposal."""

from __future__ import annotations

import ast
import json
import math
from itertools import pairwise
from pathlib import Path

import pytest
from typer.testing import CliRunner

from excavation_lab import storage
from excavation_lab.cli import app
from excavation_lab.datasets import episodes as episode_mod
from excavation_lab.datasets import splits as split_mod
from excavation_lab.datasets.episodes import EpisodeSpec, add_episode, validate_episodes
from excavation_lab.datasets.splits import SAME_SCENE_MESSAGE, propose_split
from excavation_lab.errors import LabError
from excavation_lab.schemas import (
    SplitIndependence,
    SplitLimitation,
    SplitName,
    load_pilot,
)

_RUNNER = CliRunner()
_TOO_FEW = (
    "Мало эпизодов для независимого разбиения. "
    "Это проверка той же сцены, не переносимость."
)
_SPLIT_NAMES = ("train", "validation", "test")


def test_gap_of_at_least_ten_seconds_is_entirely_excluded_guard() -> None:
    """Episodes already ≥10 s apart may sit in different splits.

    Every time between them is excluded_guard. No train frame is closer
    than 10 s to a test frame.
    """
    pilot = load_pilot()
    assert pilot.guard_gap_s == 10
    specs = [
        EpisodeSpec("train-ep", "vid", "g-train", 0, 60, "погрузка"),
        EpisodeSpec("val-ep", "vid", "g-val", 72, 92, "погрузка"),
        EpisodeSpec("test-ep", "vid", "g-test", 104, 124, "погрузка"),
    ]
    proposal = propose_split("vid", specs)
    assert proposal.guard_gap_s == 10
    assert proposal.independence is SplitIndependence.episode_split
    assert proposal.limitation is None
    assert proposal.frozen is False

    assert proposal.split_for("train-ep") is SplitName.train
    assert proposal.split_for("val-ep") is SplitName.validation
    assert proposal.split_for("test-ep") is SplitName.test
    _assert_span_is_guard(proposal, 60, 72)
    _assert_span_is_guard(proposal, 92, 104)
    _assert_split_frames_at_least(proposal, 10, pilot.analysis_fps)
    for left, right in pairwise(specs):
        assert right.start_s - left.end_s >= 10


def test_episodes_five_seconds_apart_are_not_split_by_a_fake_guard() -> None:
    """A 5 s gap is not a guard.

    Those episodes stay in one split. The test fails if they remain in
    different splits, including when the 5 s between them is labeled
    excluded_guard. The boundary moves to a later gap that is at least 10 s.
    """
    near_a = EpisodeSpec("near-a", "vid", "g-a", 0, 60, "погрузка")
    near_b = EpisodeSpec("near-b", "vid", "g-b", 65, 85, "погрузка")
    far_c = EpisodeSpec("far-c", "vid", "g-c", 105, 125, "погрузка")
    assert near_a.split_group_id != near_b.split_group_id
    assert near_b.start_s - near_a.end_s == 5
    assert far_c.start_s - near_b.end_s == 20

    proposal = propose_split("vid", [far_c, near_b, near_a])
    assert proposal.guard_gap_s == 10
    assert [episode.episode_id for episode in proposal.episodes] == [
        "near-a",
        "near-b",
        "far-c",
    ]
    assert propose_split("vid", [near_a, near_b, far_c]).model_dump() == proposal.model_dump()

    split_a = proposal.split_for("near-a")
    split_b = proposal.split_for("near-b")
    assert split_a == split_b
    assert split_b is not proposal.split_for("far-c")
    for time_s in _times(60, 65):
        role = proposal.assignment_at(time_s)
        assert role != "excluded_guard"
        assert role not in _SPLIT_NAMES
    assert not any(_intersects(guard.start_s, guard.end_s, 60, 65) for guard in proposal.guards)
    assert all(guard.end_s - guard.start_s >= 10 for guard in proposal.guards)
    _assert_span_is_guard(proposal, near_b.end_s, far_c.start_s)
    _assert_split_frames_at_least(proposal, 10, load_pilot().analysis_fps)


def test_straddling_episode_is_not_sliced() -> None:
    """The episode whose duration span contains the 60% cut stays whole."""
    specs = [
        EpisodeSpec("a", "vid", "group-a", 0, 50, "заезд"),
        EpisodeSpec("h", "vid", "group-h", 60, 100, "заезд"),
        EpisodeSpec("i", "vid", "group-i", 110, 120, "заезд"),
    ]
    total = sum(spec.end_s - spec.start_s for spec in specs)
    h = specs[1]
    axis_start = sum(
        spec.end_s - spec.start_s for spec in specs if spec.start_s < h.start_s
    )
    axis_end = axis_start + (h.end_s - h.start_s)
    cut = 0.6 * total
    assert axis_start < cut < axis_end

    proposal = propose_split("vid", specs)
    rows = [episode for episode in proposal.episodes if episode.episode_id == "h"]
    assert len(rows) == 1
    assert rows[0].split is SplitName.validation
    assert rows[0].label == "заезд"
    roles = {proposal.assignment_at(time_s) for time_s in _times(h.start_s, h.end_s, step=1)}
    assert roles == {"validation"}


def test_split_group_stays_in_one_split() -> None:
    """Two episodes of one group are not sent to different splits.

    A per-episode walk would put g2 in validation. The group is train.
    """
    specs = [
        EpisodeSpec("g1", "vid", "group-g", 0, 55, "погрузка"),
        EpisodeSpec("g2", "vid", "group-g", 65, 90, "погрузка"),
        EpisodeSpec("h", "vid", "group-h", 100, 110, "ожидание"),
        EpisodeSpec("i", "vid", "group-i", 120, 130, "ожидание"),
    ]
    naive = _naive_episode_splits(specs)
    assert naive["g1"] != naive["g2"]

    proposal = propose_split("vid", specs)
    assert proposal.split_for("g1") is proposal.split_for("g2")
    assert proposal.split_for("g1") is SplitName.train
    for time_s in _times(55, 65):
        assert proposal.assignment_at(time_s) != "excluded_guard"


def test_too_few_episodes_are_same_scene_not_transfer() -> None:
    assert SAME_SCENE_MESSAGE == _TOO_FEW
    two = [
        EpisodeSpec("a", "vid", "g-a", 0, 60, "погрузка"),
        EpisodeSpec("b", "vid", "g-b", 80, 100, "погрузка"),
    ]
    two_proposal = propose_split("vid", two)
    assert two_proposal.independence is SplitIndependence.same_scene
    assert two_proposal.independence is not SplitIndependence.episode_split
    assert two_proposal.limitation is SplitLimitation.too_few_episodes
    assert two_proposal.message == _TOO_FEW

    shared = [
        EpisodeSpec("a", "vid", "g-a", 0, 30, "погрузка"),
        EpisodeSpec("b", "vid", "g-a", 40, 70, "погрузка"),
        EpisodeSpec("c", "vid", "g-c", 90, 120, "погрузка"),
    ]
    shared_proposal = propose_split("vid", shared)
    assert len({episode.split_group_id for episode in shared_proposal.episodes}) == 2
    assert shared_proposal.independence is SplitIndependence.same_scene
    assert shared_proposal.limitation is SplitLimitation.too_few_episodes
    assert shared_proposal.message == _TOO_FEW


def test_no_episodes_build_three_sequential_blocks() -> None:
    pilot = load_pilot()
    proposal = propose_split("vid", [], duration_s=100)
    assert proposal.independence is SplitIndependence.same_scene
    assert proposal.independence is not SplitIndependence.episode_split
    assert proposal.limitation is SplitLimitation.too_few_episodes
    assert proposal.message == _TOO_FEW
    assert proposal.frozen is False
    assert proposal.episodes == []
    assert proposal.guard_gap_s == pilot.guard_gap_s == 10
    assert proposal.train_fraction == pilot.split_fractions.train == 0.6
    assert proposal.validation_fraction == 0.2
    assert proposal.test_fraction == 0.2
    assert [block.role for block in proposal.blocks] == [
        "train",
        "excluded_guard",
        "validation",
        "excluded_guard",
        "test",
    ]
    assert proposal.blocks[0].start_s == pytest.approx(0)
    assert proposal.blocks[-1].end_s == pytest.approx(100)
    for prev, nxt in pairwise(proposal.blocks):
        assert prev.end_s == pytest.approx(nxt.start_s)
    assignable = 100 - 2 * proposal.guard_gap_s
    assert proposal.blocks[0].end_s - proposal.blocks[0].start_s == pytest.approx(
        0.6 * assignable
    )
    assert proposal.blocks[2].end_s - proposal.blocks[2].start_s == pytest.approx(
        0.2 * assignable
    )
    assert proposal.blocks[4].end_s - proposal.blocks[4].start_s == pytest.approx(
        0.2 * assignable
    )
    assert proposal.blocks[1].end_s - proposal.blocks[1].start_s == pytest.approx(10)
    assert proposal.blocks[3].end_s - proposal.blocks[3].start_s == pytest.approx(10)
    assert proposal.assignment_at(0) == "train"
    assert proposal.assignment_at(proposal.blocks[1].start_s) == "excluded_guard"
    assert proposal.assignment_at(proposal.blocks[2].start_s) == "validation"
    assert proposal.assignment_at(proposal.blocks[4].start_s) == "test"
    assert proposal.assignment_at(100) is None

    with pytest.raises(LabError) as missing:
        propose_split("vid", [], duration_s=None)
    assert missing.value.code == "duration_missing"


def test_end_not_after_start_is_rejected() -> None:
    for end_s in (5.0, 4.0):
        with pytest.raises(LabError) as caught:
            validate_episodes([EpisodeSpec("a", "vid", "g", 5.0, end_s, "заезд")])
        assert caught.value.code == "invalid_interval"


def test_overlap_without_concurrent_mark_is_rejected() -> None:
    same_group = [
        EpisodeSpec("a", "vid", "g", 0, 10, "заезд"),
        EpisodeSpec("b", "vid", "g", 4, 12, "заезд"),
    ]
    with pytest.raises(LabError) as unmarked:
        validate_episodes(same_group)
    assert unmarked.value.code == "episode_overlap"
    assert _cyrillic(unmarked.value.message)

    other_group = [
        EpisodeSpec("a", "vid", "g", 0, 10, "заезд", concurrent=True),
        EpisodeSpec("b", "vid", "other", 4, 12, "заезд", concurrent=True),
    ]
    with pytest.raises(LabError) as different:
        validate_episodes(other_group)
    assert different.value.code == "episode_overlap"

    mixed = [
        EpisodeSpec("a", "vid", "g", 0, 10, "заезд", concurrent=True),
        EpisodeSpec("b", "vid", "g", 4, 12, "заезд", concurrent=False),
    ]
    with pytest.raises(LabError) as inconsistent:
        validate_episodes(mixed)
    assert inconsistent.value.code == "inconsistent_concurrency"

    touching = [
        EpisodeSpec("a", "vid", "g-a", 0, 10, "заезд"),
        EpisodeSpec("b", "vid", "g-b", 10, 20, "заезд"),
    ]
    assert len(validate_episodes(touching)) == 2


def test_concurrent_group_may_overlap_and_shares_a_split(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    storage.create_project("pilot", workspace)
    storage.insert_video(workspace, "pilot", "vid-1")
    add_episode(
        workspace,
        "pilot",
        EpisodeSpec("a", "vid-1", "g", 0, 40, "заезд", concurrent=True),
    )
    add_episode(
        workspace,
        "pilot",
        EpisodeSpec("b", "vid-1", "g", 10, 30, "заезд", concurrent=True),
    )
    loaded = episode_mod.load_episode_specs(workspace, "pilot", "vid-1")
    assert [spec.concurrent for spec in loaded] == [True, True]
    proposal = propose_split("vid-1", loaded)
    assert proposal.split_for("a") is proposal.split_for("b")
    assert proposal.assignment_at(15) == proposal.split_for("a").value


def test_propose_rejects_another_video() -> None:
    specs = [
        EpisodeSpec("a", "vid-a", "g", 0, 10, "заезд"),
        EpisodeSpec("b", "vid-b", "h", 20, 30, "заезд"),
    ]
    with pytest.raises(LabError) as caught:
        propose_split("vid-a", specs)
    assert caught.value.code == "video_mismatch"


def test_guard_below_ten_seconds_is_rejected() -> None:
    with pytest.raises(LabError) as caught:
        propose_split("vid", [], duration_s=100, guard_gap_s=5)
    assert caught.value.code == "invalid_guard_gap"


def test_split_code_does_not_permute_frames() -> None:
    for module in (episode_mod, split_mod):
        source = Path(module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != "random" for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.module != "random"
        assert "shuffle" not in source
        assert "streamlit" not in source
        assert "urlretrieve" not in source


def test_split_propose_json_returns_limitation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    storage.create_project("pilot", workspace)
    storage.insert_video(workspace, "pilot", "vid-1", name="pilot")
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            "UPDATE videos SET duration_s = ? WHERE video_id = ?",
            (100.0, "vid-1"),
        )
    args = [
        "dataset",
        "split-propose",
        "--project",
        "pilot",
        "--video",
        "vid-1",
        "--workspace",
        str(workspace),
    ]
    text = _RUNNER.invoke(app, args)
    assert text.exit_code == 0
    assert _TOO_FEW in text.stdout
    assert "Разбиение не зафиксировано." in text.stdout

    result = _RUNNER.invoke(app, [*args, "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    data = payload["data"]
    assert data["limitation"] == "too_few_episodes"
    assert data["message"] == _TOO_FEW
    assert data["independence"] == "same_scene"
    assert data["frozen"] is False
    assert data["episodes"] == []
    path = Path(data["path"])
    assert path.is_file()
    assert path.name != "split_manifest.json"
    assert not (storage.project_directory(workspace, "pilot") / "split_manifest.json").exists()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["frozen"] is False
    assert written["limitation"] == "too_few_episodes"
    assert written["independence"] == "same_scene"
    assert [block["role"] for block in written["blocks"]] == [
        "train",
        "excluded_guard",
        "validation",
        "excluded_guard",
        "test",
    ]
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        count = connection.execute("SELECT COUNT(*) AS n FROM split_manifests").fetchone()
    assert count is not None
    assert count["n"] == 0


def _assert_span_is_guard(proposal, start: float, end: float) -> None:
    matches = [
        guard
        for guard in proposal.guards
        if math.isclose(guard.start_s, start) and math.isclose(guard.end_s, end)
    ]
    assert len(matches) == 1
    assert matches[0].role == "excluded_guard"
    assert end - start >= proposal.guard_gap_s
    for time_s in _times(start, end):
        assert proposal.assignment_at(time_s) == "excluded_guard"


def _assert_split_frames_at_least(proposal, guard: float, fps: float) -> None:
    frames = {name: [] for name in _SPLIT_NAMES}
    for episode in proposal.episodes:
        frames[episode.split.value].extend(_frames(episode.start_s, episode.end_s, fps))
    assert frames["train"]
    assert frames["test"]
    train_test = min(
        abs(left - right) for left in frames["train"] for right in frames["test"]
    )
    assert train_test >= guard
    present = [name for name in _SPLIT_NAMES if frames[name]]
    for left_index, left_name in enumerate(present):
        for right_name in present[left_index + 1 :]:
            distance = min(
                abs(left - right) for left in frames[left_name] for right in frames[right_name]
            )
            assert distance >= guard
    for left in proposal.episodes:
        for right in proposal.episodes:
            if left.split == right.split or left.episode_id >= right.episode_id:
                continue
            assert _separation(left.start_s, left.end_s, right.start_s, right.end_s) >= guard


def _naive_episode_splits(specs: list[EpisodeSpec]) -> dict[str, str]:
    ordered = sorted(specs, key=lambda spec: (spec.start_s, spec.episode_id))
    total = sum(spec.end_s - spec.start_s for spec in ordered)
    train_cut = 0.6 * total
    val_cut = 0.8 * total
    assigned: dict[str, str] = {}
    cursor = 0.0
    for spec in ordered:
        start = cursor
        end = cursor + (spec.end_s - spec.start_s)
        cursor = end
        lengths = {
            "train": _overlap(start, end, 0.0, train_cut),
            "validation": _overlap(start, end, train_cut, val_cut),
            "test": _overlap(start, end, val_cut, math.inf),
        }
        best = "train"
        best_length = -1.0
        for name in _SPLIT_NAMES:
            if lengths[name] > best_length:
                best = name
                best_length = lengths[name]
        assigned[spec.episode_id] = best
    return assigned


def _frames(start: float, end: float, fps: float) -> list[float]:
    step = 1.0 / fps
    frames: list[float] = []
    index = 0
    while index < 100_000:
        time_s = index * step
        if time_s >= end:
            break
        if time_s >= start:
            frames.append(time_s)
        index += 1
    return frames


def _times(start: float, end: float, step: float = 0.5) -> list[float]:
    times: list[float] = []
    cursor = start
    while cursor < end:
        times.append(cursor)
        cursor += step
    tail = math.nextafter(end, start)
    if start <= tail < end and (not times or tail != times[-1]):
        times.append(tail)
    return times


def _intersects(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return start_a < end_b and start_b < end_a


def _overlap(start: float, end: float, band_start: float, band_end: float) -> float:
    return max(0.0, min(end, band_end) - max(start, band_start))


def _separation(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    if end_a <= start_b:
        return start_b - end_a
    if end_b <= start_a:
        return start_a - end_b
    return 0.0


def _cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)
