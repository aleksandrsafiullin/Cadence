"""Sampler stays inside a frozen split. Test frames and guards do not enter train."""

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
    freeze_split,
    propose_split,
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
_LONG_EPISODES = (
    EpisodeSpec("train-ep", "vid", "g-train", 0, 60, "погрузка"),
    EpisodeSpec("val-ep", "vid", "g-val", 72, 92, "погрузка"),
    EpisodeSpec("test-ep", "vid", "g-test", 104, 124, "погрузка"),
)
_SHORT_EPISODES = (EpisodeSpec("train-ep", "vid", "g-train", 0, 4, "погрузка"),)


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
    write_synthetic_mp4(path, 126, 2)
    return path


@pytest.fixture(scope="module")
def twin_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("twin") / "twin.mp4"
    _write_identical_mp4(path, frames=8, fps=2)
    return path


def _bind(tmp_path: Path, clip: Path, episodes: tuple[EpisodeSpec, ...]) -> Path:
    workspace = tmp_path / "workspace"
    storage.create_project("pilot", workspace)
    storage.replace_extra_roots(workspace, "pilot", [clip.parent])
    storage.insert_video(workspace, "pilot", "vid", path=clip, name=clip.name)
    for episode in episodes:
        add_episode(workspace, "pilot", episode)
    proposal = propose_split("vid", list(episodes))
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
    assert "image" not in FrameTick.__dataclass_fields__
    assert "image" not in SelectedTick.__dataclass_fields__


def test_sample_refuses_when_split_is_not_frozen(
    tmp_path: Path,
    twin_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    storage.create_project("pilot", workspace)
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


def test_train_sample_excludes_test_episode_and_guards(
    tmp_path: Path,
    long_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _bind(tmp_path, long_clip, _LONG_EPISODES)
    frozen = require_frozen(workspace, "pilot", "vid")
    assert frozen.frozen is True
    assert any(
        episode.episode_id == "test-ep" and episode.split is SplitName.test
        for episode in frozen.episodes
    )
    assert frozen.guards
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
    test_episode = next(episode for episode in frozen.episodes if episode.episode_id == "test-ep")
    during_sample = list(calls)
    test_indexes = _indexes_between(long_clip, test_episode.start_s, test_episode.end_s)
    assert test_indexes
    sampled = {row["source_frame_index"] for row in rows}
    assert sampled.isdisjoint(test_indexes)
    for row in rows:
        assert row["split"] == "train"
        assert row["episode_id"] == "train-ep"
        assert row["video_id"] == "vid"
        assert row["image_file"] == f"vid_{row['source_frame_index']}.jpg"
        assert row["selection_reason"] == ""
        assert row["strategy"] == "spaced"
        assert 0 <= row["source_time_s"] < 60
        for guard in frozen.guards:
            assert not (guard.start_s <= row["source_time_s"] < guard.end_s)
    ordered = sorted(rows, key=lambda row: row["source_time_s"])
    for earlier, later in pairwise(ordered):
        assert later["source_time_s"] - earlier["source_time_s"] + 1e-3 >= 2.0
    assert during_sample
    for start_s, end_s in during_sample:
        assert end_s is not None
        assert not _overlaps(start_s, end_s, test_episode.start_s, test_episode.end_s)
        for guard in frozen.guards:
            assert not _overlaps(start_s, end_s, guard.start_s, guard.end_s)
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


def test_default_spacing_comes_from_config_and_closer_reason_keeps_the_frame(
    tmp_path: Path,
    twin_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _bind(tmp_path, twin_clip, _SHORT_EPISODES)
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
    workspace = _bind(tmp_path, twin_clip, _SHORT_EPISODES)
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
    workspace = _bind(tmp_path, long_clip, _LONG_EPISODES)
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
    assert {row["episode_id"] for row in train_rows} == {"train-ep"}
    assert {row["episode_id"] for row in val_rows} == {"val-ep"}
    assert {row["strategy"] for row in train_rows} == {"spaced"}
    assert {row["strategy"] for row in val_rows} == {"representative"}
    assert {row["selection_reason"] for row in val_rows} == {"representative"}
    test_episode = next(episode for episode in frozen.episodes if episode.split is SplitName.test)
    val_episode = next(
        episode for episode in frozen.episodes if episode.split is SplitName.validation
    )
    during_sample = list(calls)
    test_indexes = _indexes_between(long_clip, test_episode.start_s, test_episode.end_s)
    val_indexes = sorted(_indexes_between(long_clip, val_episode.start_s, val_episode.end_s))
    assert test_indexes
    assert len(val_indexes) > 20
    sampled_val = [row["source_frame_index"] for row in val_rows]
    assert set(sampled_val).isdisjoint(test_indexes)
    assert sampled_val[0] == val_indexes[0]
    assert sampled_val[-1] == val_indexes[-1]
    assert any(right - left > 1 for left, right in pairwise(sampled_val))
    assert sampled_val != val_indexes[:20]
    for start_s, end_s in during_sample:
        assert end_s is not None
        assert not _overlaps(start_s, end_s, test_episode.start_s, test_episode.end_s)
        for guard in frozen.guards:
            assert not _overlaps(start_s, end_s, guard.start_s, guard.end_s)
    images = Path(data["images_dir"])
    assert sorted(path.name for path in images.glob("*.jpg")) == sorted(
        row["image_file"] for row in rows
    )
    assert not any(row["split"] == "test" for row in rows)
    for row in rows:
        time_s = row["source_time_s"]
        for guard in frozen.guards:
            assert not (guard.start_s <= time_s < guard.end_s)


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
    workspace = _bind(tmp_path, twin_clip, _SHORT_EPISODES)
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
    workspace = _bind(tmp_path, twin_clip, _SHORT_EPISODES)

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
