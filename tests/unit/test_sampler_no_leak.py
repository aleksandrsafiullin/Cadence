"""Sampler stays inside frozen role blocks. A training block does not enter validation or test."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from itertools import pairwise
from pathlib import Path

import av
import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from excavation_lab import storage
from excavation_lab.cli import app
from excavation_lab.datasets import sampler as sampler_mod
from excavation_lab.datasets import splits as split_mod
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.episodes import EpisodeSpec, add_episode
from excavation_lab.datasets.sampler import (
    CloserPick,
    FrameTick,
    SelectedTick,
    plan_counts,
    require_sample_spacing,
    sample_frames,
    select_ticks,
)
from excavation_lab.datasets.splits import (
    SplitProposal,
    freeze_split,
    frozen_manifest_path,
    require_frozen,
    write_proposal,
)
from excavation_lab.errors import LabError
from excavation_lab.media.reader import FrameReader
from excavation_lab.schemas import SplitName, load_pilot

_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SYNTHETIC_SPEC = importlib.util.spec_from_file_location(
    "excavation_lab_synthetic_mp4",
    _SYNTHETIC_PATH,
)
assert _SYNTHETIC_SPEC is not None and _SYNTHETIC_SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SYNTHETIC_SPEC)
_SYNTHETIC_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_mp4 = _SYNTHETIC.write_synthetic_mp4

_RUNNER = CliRunner()
_LONG_BLOCKS = (
    RoleBlock(start_s=0, end_s=60, role="train"),
    RoleBlock(start_s=72, end_s=92, role="validation"),
    RoleBlock(start_s=104, end_s=124, role="test"),
)
_SHORT_BLOCKS = (RoleBlock(start_s=0, end_s=4, role="train"),)
_LEFTOVER_TEST_EPISODE = {
    "episode_id": "test-ep",
    "split": "test",
    "start_s": 0,
    "end_s": 60,
}


def _cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)


def _forbid_propose(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("propose_split")


def _arm_propose_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not hasattr(sampler_mod, "propose_split")
    monkeypatch.setattr(split_mod, "propose_split", _forbid_propose)


def _write_identical_mp4(path: Path, frames: int, fps: int) -> None:
    """Eight identical pictures. This clip is synthetic and is not accuracy evidence."""
    width, height = 64, 48
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (32, 64, 96)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "0", "g": "1"}
        for _ in range(frames):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            yuv = frame.reformat(format="yuv420p")
            for packet in stream.encode(yuv):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


_LONG_FPS = 2


def _tick(
    index: int,
    time_s: float,
    episode_id: str,
    split: SplitName = SplitName.train,
) -> FrameTick:
    return FrameTick("vid", index, time_s, episode_id, split)


@pytest.fixture(scope="module")
def long_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("long") / "long.mp4"
    write_synthetic_mp4(path, 126, _LONG_FPS)
    return path


@pytest.fixture(scope="module")
def twin_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("twin") / "twin.mp4"
    _write_identical_mp4(path, frames=8, fps=2)
    return path


def _bind(
    tmp_path: Path,
    clip: Path,
    blocks: tuple[RoleBlock, ...],
    episodes: list[dict[str, object]] | None = None,
) -> Path:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    storage.replace_extra_roots(workspace, "pilot", [clip.parent])
    storage.insert_video(workspace, "pilot", "vid", path=clip, name=clip.name)
    for item in episodes or []:
        episode_id = item["episode_id"]
        start_s = item["start_s"]
        end_s = item["end_s"]
        assert isinstance(episode_id, str)
        assert isinstance(start_s, (int, float)) and not isinstance(start_s, bool)
        assert isinstance(end_s, (int, float)) and not isinstance(end_s, bool)
        add_episode(
            workspace,
            "pilot",
            EpisodeSpec(episode_id, "vid", "g-leftover", float(start_s), float(end_s), "погрузка"),
        )
    proposal = SplitProposal(
        video_id="vid",
        blocks=list(blocks),
        episodes=list(episodes or []),
    )
    proposal_path = write_proposal(workspace, "pilot", proposal)
    freeze_split(workspace, "pilot", proposal_path)
    return workspace


def _sample_args(workspace: Path, *extra: str) -> list[str]:
    return [
        "dataset",
        "sample",
        "--project",
        "pilot",
        "--workspace",
        str(workspace),
        "--json",
        *extra,
    ]


def _invoke(monkeypatch: pytest.MonkeyPatch, workspace: Path, *extra: str):
    _arm_propose_guard(monkeypatch)
    result = _RUNNER.invoke(app, _sample_args(workspace, *extra))
    assert result.exit_code == 0, (result.stdout, result.exception)
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    return payload["data"]


def _spy_intervals(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[float, float | None]],
) -> None:
    original = FrameReader.iterate

    def wrapped(
        self: FrameReader,
        start_s: float = 0.0,
        end_s: float | None = None,
    ):
        calls.append((start_s, end_s))
        return original(self, start_s, end_s)

    monkeypatch.setattr(FrameReader, "iterate", wrapped)


def _overlaps(
    start: float,
    end: float | None,
    left: float,
    right: float,
) -> bool:
    if end is None:
        return start < right
    return start < right and left < end


def _indexes_between(clip: Path, start_s: float, end_s: float) -> set[int]:
    found: set[int] = set()
    for frame in FrameReader(clip, video_id="vid").iterate(start_s, end_s):
        if frame.source_frame_index is not None:
            found.add(frame.source_frame_index)
    return found


def test_sampler_source_calls_require_frozen_and_not_propose_split() -> None:
    source = Path(sampler_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            called.add(func.id)
        elif isinstance(func, ast.Attribute):
            called.add(func.attr)
    assert "require_frozen" in called
    assert "propose_split" not in called
    assert "propose_for_project" not in called
    assert "excluded_guard" not in source
    assert "image" not in FrameTick.__dataclass_fields__
    assert "image" not in SelectedTick.__dataclass_fields__


def test_sample_refuses_when_split_is_not_frozen(
    tmp_path: Path,
    twin_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    storage.replace_extra_roots(workspace, "pilot", [twin_clip.parent])
    storage.insert_video(workspace, "pilot", "vid", path=twin_clip, name="twin.mp4")
    _arm_propose_guard(monkeypatch)
    result = _RUNNER.invoke(
        app,
        _sample_args(workspace, "--split", "train", "--count", "60"),
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "split_not_frozen"
    assert payload["error"]["code"].isascii()
    assert _cyrillic(payload["error"]["message"])
    project_dir = storage.project_directory(workspace, "pilot")
    assert not (project_dir / "samples").exists()


def test_rejected_strategies_stop_before_a_project(tmp_path: Path) -> None:
    for name in ("active", "easiest", "hardest"):
        with pytest.raises(LabError) as caught:
            sample_frames(
                tmp_path,
                "pilot",
                split="train",
                count=1,
                strategy=name,
            )
        assert caught.value.code == "unknown_strategy"
        assert caught.value.code.isascii()
        assert _cyrillic(caught.value.message)
    with pytest.raises(LabError) as mixed:
        sample_frames(
            tmp_path,
            "pilot",
            plan="diagnostic",
            split="train",
            count=60,
        )
    assert mixed.value.code == "sample_args"


def test_spacing_rejects_a_closer_pick_without_a_reason() -> None:
    with pytest.raises(ValidationError):
        CloserPick(source_frame_index=1, selection_reason="   ")
    close = [
        SelectedTick("vid", 0, 0.0, "train-ep", SplitName.train, "", "spaced"),
        SelectedTick("vid", 1, 0.4, "train-ep", SplitName.train, "", "spaced"),
    ]
    with pytest.raises(LabError) as caught:
        require_sample_spacing(close, 2.0)
    assert caught.value.code == "sample_spacing"
    assert caught.value.code.isascii()
    assert _cyrillic(caught.value.message)
    allowed = [
        close[0],
        SelectedTick("vid", 1, 0.4, "train-ep", SplitName.train, "rare_pose", "spaced"),
    ]
    require_sample_spacing(allowed, 2.0)
    marked_between = [
        SelectedTick("vid", 0, 0.0, "train-ep", SplitName.train, "", "spaced"),
        SelectedTick("vid", 1, 0.5, "train-ep", SplitName.train, "rare_pose", "spaced"),
        SelectedTick("vid", 2, 2.0, "train-ep", SplitName.train, "", "spaced"),
    ]
    require_sample_spacing(marked_between, 2.0)
    other_episode = [
        close[0],
        SelectedTick("vid", 1, 0.2, "other-ep", SplitName.train, "", "spaced"),
    ]
    require_sample_spacing(other_episode, 2.0)

    ticks = [_tick(index, index * 0.5, "train-ep") for index in range(5)]
    kept = select_ticks(
        ticks,
        count=4,
        spacing_s=2.0,
        strategy="spaced",
        split="train",
        extras=(CloserPick(source_frame_index=1, selection_reason="rare_pose"),),
    )
    assert [item.source_frame_index for item in kept] == [0, 1, 4]
    assert kept[1].selection_reason == "rare_pose"
    require_sample_spacing(kept, 2.0)
    with pytest.raises(LabError) as quota:
        select_ticks(
            ticks,
            count=2,
            spacing_s=2.0,
            strategy="spaced",
            split="train",
            extras=(CloserPick(source_frame_index=1, selection_reason="rare_pose"),),
        )
    assert quota.value.code == "sample_quota"
    with pytest.raises(LabError) as foreign:
        select_ticks(
            ticks,
            count=4,
            spacing_s=2.0,
            strategy="spaced",
            split="train",
            extras=(CloserPick(source_frame_index=99, selection_reason="rare_pose"),),
        )
    assert foreign.value.code == "frame_not_in_split"


def test_spaced_does_not_duplicate_frames_to_fill_a_quota() -> None:
    ticks = [_tick(index, float(index), "train-ep") for index in range(5)]
    short = select_ticks(
        ticks,
        count=10,
        spacing_s=2.0,
        strategy="spaced",
        split="train",
    )
    assert [item.source_frame_index for item in short] == [0, 2, 4]
    assert len(short) == len({item.source_frame_index for item in short})
    wide = [_tick(index, float(index * 2), "train-ep") for index in range(10)]
    trimmed = select_ticks(
        wide,
        count=3,
        spacing_s=2.0,
        strategy="spaced",
        split="train",
    )
    assert [item.source_frame_index for item in trimmed] == [0, 5, 9]
    assert all(item.selection_reason == "" for item in trimmed)


def test_representative_stride_covers_episodes_not_easy_or_hard_frames() -> None:
    ticks: list[FrameTick] = []
    for episode_id, start in (("val-a", 0), ("val-b", 10)):
        for offset in range(10):
            ticks.append(
                _tick(
                    start + offset,
                    float(start + offset),
                    episode_id,
                    SplitName.validation,
                )
            )
    selected = select_ticks(
        ticks,
        count=4,
        spacing_s=2.0,
        strategy="representative",
        split="validation",
    )
    assert [item.source_frame_index for item in selected] == [0, 9, 10, 19]
    assert {item.episode_id for item in selected} == {"val-a", "val-b"}
    assert {item.strategy for item in selected} == {"representative"}
    assert {item.selection_reason for item in selected} == {"representative"}
    flipped = select_ticks(
        list(reversed(ticks)),
        count=4,
        spacing_s=2.0,
        strategy="representative",
        split="validation",
    )
    assert [item.source_frame_index for item in flipped] == [0, 9, 10, 19]
    middle = select_ticks(
        ticks[:10],
        count=1,
        spacing_s=2.0,
        strategy="representative",
        split="validation",
    )
    assert [item.source_frame_index for item in middle] == [5]
    close = [_tick(index, index * 0.5, "val-a", SplitName.validation) for index in range(8)]
    dense = select_ticks(
        close,
        count=8,
        spacing_s=2.0,
        strategy="representative",
        split="validation",
    )
    assert len(dense) == 8
    require_sample_spacing(dense, 2.0)
    spaced = select_ticks(
        close,
        count=8,
        spacing_s=2.0,
        strategy="spaced",
        split="validation",
    )
    assert len(spaced) < 8
    assert all(item.selection_reason == "" for item in spaced)


def test_train_block_with_test_episode_stays_out_of_other_splits(
    tmp_path: Path,
    long_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _bind(
        tmp_path,
        long_clip,
        _LONG_BLOCKS,
        episodes=[dict(_LEFTOVER_TEST_EPISODE)],
    )
    manifest_path = frozen_manifest_path(workspace, "pilot", "vid")
    manifest_before = manifest_path.read_bytes()
    frozen = require_frozen(workspace, "pilot", "vid")
    edges_before = [(block.start_s, block.end_s, block.role) for block in frozen.blocks]
    assert frozen.frozen is True
    assert frozen.guards == []
    assert "excluded_guard" not in manifest_before.decode("utf-8")
    assert any(
        item["episode_id"] == "test-ep" and item["split"] == "test"
        for item in frozen.episodes
    )
    calls: list[tuple[float, float | None]] = []
    frozen_calls = {"n": 0}
    real_require = sampler_mod.require_frozen

    def wrapped_require(*args: object, **kwargs: object):
        frozen_calls["n"] += 1
        return real_require(*args, **kwargs)

    _spy_intervals(monkeypatch, calls)
    monkeypatch.setattr(sampler_mod, "require_frozen", wrapped_require)
    data = _invoke(monkeypatch, workspace, "--split", "train", "--count", "60")
    assert frozen_calls["n"] >= 1
    pilot = load_pilot()
    assert pilot.sample_spacing_s == 2.0
    assert data["spacing_s"] == 2.0
    assert data["plan"] is None
    assert data["requested"] == {"train": 60, "validation": 0, "test": 0}
    assert data["written"]["validation"] == 0
    assert data["written"]["test"] == 0
    assert data["shortfall"]["test"] == 0
    assert data["shortfall"]["train"] == 60 - data["written"]["train"]
    assert data["written"]["train"] < 60
    assert data["shortfall"]["train"] > 0
    rows = data["rows"]
    assert rows
    assert len(rows) == data["written"]["train"]
    assert len(rows) == len({row["source_frame_index"] for row in rows})
    train_block = next(block for block in frozen.blocks if block.role == "train")
    test_block = next(block for block in frozen.blocks if block.role == "test")
    during_sample = list(calls)
    test_indexes = _indexes_between(long_clip, test_block.start_s, test_block.end_s)
    gap_indexes = _indexes_between(long_clip, train_block.end_s, 72.0)
    assert test_indexes
    assert gap_indexes
    sampled = {row["source_frame_index"] for row in rows}
    assert sampled.isdisjoint(test_indexes)
    assert sampled.isdisjoint(gap_indexes)
    for row in rows:
        assert row["split"] == "train"
        assert row["video_id"] == "vid"
        assert row["image_file"] == f"vid_{row['source_frame_index']}.jpg"
        assert row["selection_reason"] == ""
        assert row["strategy"] == "spaced"
        assert train_block.start_s <= row["source_time_s"] < train_block.end_s
        assert (
            _LEFTOVER_TEST_EPISODE["start_s"]
            <= row["source_time_s"]
            < _LEFTOVER_TEST_EPISODE["end_s"]
        )
    ordered = sorted(rows, key=lambda row: row["source_time_s"])
    for earlier, later in pairwise(ordered):
        assert later["source_time_s"] - earlier["source_time_s"] + 1e-3 >= 2.0
    assert during_sample
    for start_s, end_s in during_sample:
        assert end_s is not None
        assert train_block.start_s <= start_s and end_s <= train_block.end_s
        assert not _overlaps(start_s, end_s, test_block.start_s, test_block.end_s)
    images = Path(data["images_dir"])
    files = sorted(path.name for path in images.glob("*.jpg"))
    assert files == sorted(row["image_file"] for row in rows)
    for row in rows:
        blob = (images / row["image_file"]).read_bytes()
        assert blob.startswith(b"\xff\xd8")
        assert hashlib.sha256(blob).hexdigest() == row["image_sha256"]
    manifest = json.loads(Path(data["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["rows"][0]["image_sha256"] == rows[0]["image_sha256"]
    project_dir = storage.project_directory(workspace, "pilot")
    assert Path(data["manifest_path"]).is_relative_to(project_dir)
    assert not list(Path(data["manifest_path"]).parent.rglob("*annotations*"))
    stolen = next(iter(test_indexes))
    with pytest.raises(LabError) as caught:
        sample_frames(
            workspace,
            "pilot",
            split="train",
            count=60,
            extras=(CloserPick(source_frame_index=stolen, selection_reason="rare_pose"),),
        )
    assert caught.value.code == "frame_not_in_split"
    assert not (images / f"vid_{stolen}.jpg").exists()

    held = sample_frames(workspace, "pilot", plan="content")
    by_split = {"train": [], "validation": [], "test": []}
    for row in held.rows:
        by_split[row.split.value].append(row)
    assert by_split["train"] and by_split["validation"] and by_split["test"]
    train_ids = {row.source_frame_index for row in by_split["train"]}
    val_ids = {row.source_frame_index for row in by_split["validation"]}
    test_ids = {row.source_frame_index for row in by_split["test"]}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    for row in by_split["train"]:
        assert train_block.start_s <= row.source_time_s < train_block.end_s
    validation_block = next(block for block in frozen.blocks if block.role == "validation")
    for row in by_split["validation"]:
        assert validation_block.start_s <= row.source_time_s < validation_block.end_s
    for row in by_split["test"]:
        assert test_block.start_s <= row.source_time_s < test_block.end_s
    outside = gap_indexes | _indexes_between(long_clip, 92.0, test_block.start_s)
    assert outside.isdisjoint(train_ids | val_ids | test_ids)
    assert manifest_path.read_bytes() == manifest_before
    edges_after = [
        (block.start_s, block.end_s, block.role)
        for block in require_frozen(workspace, "pilot", "vid").blocks
    ]
    assert edges_after == edges_before


def test_second_train_piece_is_train_and_gaps_are_unsampled(
    tmp_path: Path,
    long_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocks = (
        RoleBlock(start_s=0, end_s=10, role="train"),
        RoleBlock(start_s=30, end_s=40, role="train"),
        RoleBlock(start_s=50, end_s=60, role="validation"),
        RoleBlock(start_s=80, end_s=90, role="test"),
    )
    workspace = _bind(tmp_path, long_clip, blocks)
    manifest_path = frozen_manifest_path(workspace, "pilot", "vid")
    manifest_before = manifest_path.read_bytes()
    _arm_propose_guard(monkeypatch)
    train = sample_frames(workspace, "pilot", split="train", count=40)
    validation = sample_frames(workspace, "pilot", split="validation", count=20)
    final = sample_frames(workspace, "pilot", split="test", count=20)
    train_times = [row.source_time_s for row in train.rows]
    assert any(0 <= time_s < 10 for time_s in train_times)
    assert any(30 <= time_s < 40 for time_s in train_times)
    assert all(row.split is SplitName.train for row in train.rows)
    assert all(
        (0 <= row.source_time_s < 10) or (30 <= row.source_time_s < 40) for row in train.rows
    )
    train_ids = {row.source_frame_index for row in train.rows}
    assert train_ids.isdisjoint(row.source_frame_index for row in validation.rows)
    assert train_ids.isdisjoint(row.source_frame_index for row in final.rows)
    assert all(50 <= row.source_time_s < 60 for row in validation.rows)
    assert all(80 <= row.source_time_s < 90 for row in final.rows)
    gap = _indexes_between(long_clip, 10.0, 30.0)
    assert gap
    sampled = train_ids | {row.source_frame_index for row in validation.rows}
    sampled |= {row.source_frame_index for row in final.rows}
    assert gap.isdisjoint(sampled)
    assert manifest_path.read_bytes() == manifest_before


def test_windows_follow_blocks_not_leftover_episodes() -> None:
    frozen = split_mod.FrozenSplit(
        video_id="vid",
        blocks=[
            RoleBlock(start_s=0, end_s=10, role="train"),
            RoleBlock(start_s=20, end_s=30, role="train"),
            RoleBlock(start_s=40, end_s=50, role="validation"),
            RoleBlock(start_s=60, end_s=70, role="test"),
        ],
        episodes=[dict(_LEFTOVER_TEST_EPISODE)],
        content_hash="ab" * 32,
    )
    windows = sampler_mod._windows(frozen, SplitName.train)
    assert [(item.start_s, item.end_s) for item in windows] == [(0.0, 10.0), (20.0, 30.0)]
    assert sampler_mod._assignment_at(frozen, 25.0) == "train"
    assert sampler_mod._assignment_at(frozen, 15.0) is None
    assert sampler_mod._assignment_at(frozen, 45.0) == "validation"
    assert sampler_mod._in_guard(frozen, 25.0) is False


def test_default_spacing_comes_from_config_and_closer_reason_keeps_the_frame(
    tmp_path: Path,
    twin_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _bind(tmp_path, twin_clip, _SHORT_BLOCKS)
    _arm_propose_guard(monkeypatch)
    spaced = sample_frames(workspace, "pilot", split="train", count=4)
    assert spaced.spacing_s == load_pilot().sample_spacing_s == 2.0
    assert len(spaced.rows) == 2
    assert all(row.selection_reason == "" for row in spaced.rows)
    assert all(row.strategy == "spaced" for row in spaced.rows)
    representative = sample_frames(
        workspace,
        "pilot",
        split="train",
        count=4,
        strategy="representative",
    )
    assert len(representative.rows) == 4
    assert {row.strategy for row in representative.rows} == {"representative"}
    times = sorted(row.source_time_s for row in representative.rows)
    assert times[0] == pytest.approx(0.0, abs=0.05)
    assert times[-1] == pytest.approx(3.5, abs=0.15)

    def spacing_8(path: Path | None = None):
        return load_pilot(path).model_copy(update={"sample_spacing_s": 8.0})

    monkeypatch.setattr(sampler_mod, "load_pilot", spacing_8)
    wide = sample_frames(workspace, "pilot", split="train", count=4)
    assert wide.spacing_s == 8.0
    assert len(wide.rows) == 1

    monkeypatch.setattr(sampler_mod, "load_pilot", load_pilot)
    half = _indexes_between(twin_clip, 0.4, 0.6)
    assert len(half) == 1
    index = next(iter(half))
    closer = sample_frames(
        workspace,
        "pilot",
        split="train",
        count=3,
        extras=(CloserPick(source_frame_index=index, selection_reason="rare_pose"),),
    )
    assert len(closer.rows) == 3
    matched = [row for row in closer.rows if row.source_frame_index == index]
    assert len(matched) == 1
    assert matched[0].selection_reason == "rare_pose"
    assert (Path(closer.images_dir) / matched[0].image_file).is_file()
    kept_times = sorted(row.source_time_s for row in closer.rows)
    assert kept_times[0] == pytest.approx(0.0, abs=0.05)
    assert any(abs(time_s - 0.5) < 0.05 for time_s in kept_times)
    assert any(abs(time_s - 2.0) < 0.05 for time_s in kept_times)


def test_near_duplicate_identical_frames_both_remain(
    tmp_path: Path,
    twin_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _bind(tmp_path, twin_clip, _SHORT_BLOCKS)
    data = _invoke(monkeypatch, workspace, "--split", "train", "--count", "2")
    rows = sorted(data["rows"], key=lambda row: row["source_time_s"])
    assert len(rows) == 2
    assert data["shortfall"]["train"] == 0
    images = Path(data["images_dir"])
    digests: list[str] = []
    for row in rows:
        blob = (images / row["image_file"]).read_bytes()
        assert blob.startswith(b"\xff\xd8")
        digest = hashlib.sha256(blob).hexdigest()
        assert digest == row["image_sha256"]
        digests.append(digest)
    assert digests[0] == digests[1]
    assert rows[0]["near_duplicate_warning"] is None
    assert rows[1]["near_duplicate_warning"]
    assert rows[1]["near_duplicate_of"] == rows[0]["source_frame_index"]
    assert "оставлен" in rows[1]["near_duplicate_warning"]


def test_diagnostic_plan_does_not_create_test_images(
    tmp_path: Path,
    long_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _bind(tmp_path, long_clip, _LONG_BLOCKS)
    frozen = require_frozen(workspace, "pilot", "vid")
    calls: list[tuple[float, float | None]] = []
    _spy_intervals(monkeypatch, calls)
    data = _invoke(
        monkeypatch,
        workspace,
        "--plan",
        "diagnostic",
        "--strategy",
        "spaced",
    )
    pilot = load_pilot()
    assert data["plan"] == "diagnostic"
    assert data["requested"]["train"] == pilot.sample_plans.diagnostic.train == 60
    assert data["requested"]["validation"] == pilot.sample_plans.diagnostic.validation == 20
    assert data["requested"]["test"] == pilot.sample_plans.diagnostic.test == 0
    assert data["written"]["test"] == 0
    assert data["shortfall"]["test"] == 0
    assert data["written"]["validation"] == 20
    assert data["shortfall"]["validation"] == 0
    assert data["written"]["train"] < 60
    assert data["shortfall"]["train"] == 60 - data["written"]["train"]
    rows = data["rows"]
    assert len(rows) == data["written"]["train"] + data["written"]["validation"]
    assert len({row["source_frame_index"] for row in rows}) == len(rows)
    assert all(row["split"] != "test" for row in rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    assert train_rows and val_rows
    assert {row["episode_id"] for row in train_rows} == {""}
    assert {row["episode_id"] for row in val_rows} == {""}
    assert {row["strategy"] for row in train_rows} == {"spaced"}
    assert {row["strategy"] for row in val_rows} == {"representative"}
    assert {row["selection_reason"] for row in val_rows} == {"representative"}
    train_block = next(block for block in frozen.blocks if block.role == "train")
    val_block = next(block for block in frozen.blocks if block.role == "validation")
    test_block = next(block for block in frozen.blocks if block.role == "test")
    during_sample = list(calls)
    test_indexes = _indexes_between(long_clip, test_block.start_s, test_block.end_s)
    val_indexes = sorted(_indexes_between(long_clip, val_block.start_s, val_block.end_s))
    assert test_indexes
    assert len(val_indexes) > 20
    sampled_val = [row["source_frame_index"] for row in val_rows]
    assert set(sampled_val).isdisjoint(test_indexes)
    assert sampled_val[0] == val_indexes[0]
    assert sampled_val[-1] == val_indexes[-1]
    assert any(right - left > 1 for left, right in pairwise(sampled_val))
    assert sampled_val != val_indexes[:20]
    opened = {(start_s, end_s) for start_s, end_s in during_sample}
    block_windows = {
        (train_block.start_s, train_block.end_s),
        (val_block.start_s, val_block.end_s),
    }
    # The tick pass reads whole blocks; the writer seeks one display span per tick.
    frame_period_s = 1.0 / _LONG_FPS
    assert block_windows <= opened
    for start_s, end_s in opened - block_windows:
        assert end_s is not None
        assert any(left <= start_s and end_s <= right for left, right in block_windows)
        assert end_s - start_s <= frame_period_s + 1e-6
    assert not any(
        _overlaps(start_s, end_s, test_block.start_s, test_block.end_s)
        for start_s, end_s in opened
    )
    images = Path(data["images_dir"])
    assert sorted(path.name for path in images.glob("*.jpg")) == sorted(
        row["image_file"] for row in rows
    )
    assert not any(row["split"] == "test" for row in rows)
    for row in train_rows:
        assert train_block.start_s <= row["source_time_s"] < train_block.end_s
    for row in val_rows:
        assert val_block.start_s <= row["source_time_s"] < val_block.end_s


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        ("diagnostic", (60, 20, 0)),
        ("content", (200, 60, 60)),
        ("expanded", (360, 120, 120)),
    ],
)
def test_plans_use_pilot_counts_and_do_not_invent_frames(
    plan: str,
    expected: tuple[int, int, int],
    tmp_path: Path,
    twin_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _bind(tmp_path, twin_clip, _SHORT_BLOCKS)
    counts = plan_counts(plan)
    assert (counts.train, counts.validation, counts.test) == expected
    data = _invoke(monkeypatch, workspace, "--plan", plan)
    assert data["requested"] == {
        "train": expected[0],
        "validation": expected[1],
        "test": expected[2],
    }
    assert data["requested"]["train"] == counts.train
    assert data["written"]["validation"] == 0
    assert data["written"]["test"] == 0
    assert data["shortfall"]["validation"] == expected[1]
    assert data["shortfall"]["test"] == expected[2]
    assert data["written"]["train"] < expected[0]
    assert data["shortfall"]["train"] == expected[0] - data["written"]["train"]
    rows = data["rows"]
    assert rows
    assert len(rows) == len({row["source_frame_index"] for row in rows})
    assert len(rows) == data["written"]["train"]
    assert {row["split"] for row in rows} == {"train"}
    images = list(Path(data["images_dir"]).glob("*.jpg"))
    assert len(images) == len(rows)
    assert not list(Path(data["manifest_path"]).parent.rglob("*annotations*"))


def test_plan_counts_follow_a_replaced_pilot_config(
    tmp_path: Path,
    twin_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _bind(tmp_path, twin_clip, _SHORT_BLOCKS)

    def replaced(path: Path | None = None):
        pilot = load_pilot(path)
        content = pilot.sample_plans.content.model_copy(
            update={"train": 7, "validation": 5, "test": 1}
        )
        plans = pilot.sample_plans.model_copy(update={"content": content})
        return pilot.model_copy(update={"sample_plans": plans})

    _arm_propose_guard(monkeypatch)
    monkeypatch.setattr(sampler_mod, "load_pilot", replaced)
    result = sample_frames(workspace, "pilot", plan="content")
    assert result.requested.train == 7
    assert result.requested.validation == 5
    assert result.requested.test == 1
    assert result.written.validation == 0
    assert result.written.test == 0
    assert result.shortfall.test == 1
    assert result.written.train <= 7
    assert all(row.split is SplitName.train for row in result.rows)
