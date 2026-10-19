"""Operator role blocks are the split. A stored episode id is not."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from excavation_lab import storage
from excavation_lab.cli import app
from excavation_lab.datasets import active_learning as active_mod
from excavation_lab.datasets import coco_import as coco_mod
from excavation_lab.datasets import cvat_export as cvat_mod
from excavation_lab.datasets import sampler as sampler_mod
from excavation_lab.datasets import snapshot as snapshot_mod
from excavation_lab.datasets import splits as split_mod
from excavation_lab.datasets.blocks import (
    RoleBlock,
    assignment_at,
    validate_role_blocks,
)
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.episodes import EpisodeSpec, add_episode
from excavation_lab.datasets.splits import (
    SAME_SCENE_MESSAGE,
    FrozenSplit,
    freeze_split,
    frozen_manifest_path,
    propose_for_project,
    propose_split,
    write_blocks,
)
from excavation_lab.errors import LabError
from excavation_lab.models import quality as quality_mod
from excavation_lab.models import train as train_mod
from excavation_lab.schemas import SplitName

_RUNNER = CliRunner()
_HASH = "ab" * 32


class _Raw:
    def __init__(self, start_s: object, end_s: object, role: object) -> None:
        self.start_s = start_s
        self.end_s = end_s
        self.role = role


def _blocks() -> list[RoleBlock]:
    return [
        RoleBlock(start_s=0, end_s=60, role="train"),
        RoleBlock(start_s=72, end_s=92, role="validation"),
        RoleBlock(start_s=104, end_s=124, role="test"),
    ]


def _touching() -> list[RoleBlock]:
    return [
        RoleBlock(start_s=0, end_s=10, role="train"),
        RoleBlock(start_s=10, end_s=20, role="test"),
    ]


def _project(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    storage.insert_video(workspace, "pilot", "vid", name="pilot")
    return workspace


def _frozen_with_leftover_episode() -> FrozenSplit:
    return FrozenSplit(
        video_id="vid",
        blocks=_touching(),
        content_hash=_HASH,
        episodes=[
            {
                "episode_id": "test-ep",
                "split": "test",
                "start_s": 0,
                "end_s": 60,
            }
        ],
    )


def test_module_sources_do_not_read_a_guard() -> None:
    blocks = Path(split_mod.__file__).with_name("blocks.py").read_text(encoding="utf-8")
    splits = Path(split_mod.__file__).read_text(encoding="utf-8")
    assert "guard_gap_s" not in blocks
    assert "guard_gap_s" not in splits
    assert "excluded_guard" not in blocks
    assert "excluded_guard" not in splits
    assert "episode_split" not in splits
    assert "переносим" not in SAME_SCENE_MESSAGE
    assert "transfer" not in SAME_SCENE_MESSAGE.casefold()


def test_half_open_interval_is_rejected() -> None:
    for start_s, end_s in ((5, 5), (8, 3), (True, 2)):
        with pytest.raises(LabError) as caught:
            validate_role_blocks([_Raw(start_s, end_s, "train")])
        assert caught.value.code == "invalid_interval"


def test_one_role_may_have_several_disjoint_pieces() -> None:
    blocks = validate_role_blocks(
        [
            RoleBlock(start_s=20, end_s=30, role="train"),
            RoleBlock(start_s=0, end_s=10, role="train"),
        ]
    )
    assert [(block.start_s, block.end_s) for block in blocks] == [(0, 10), (20, 30)]
    assert assignment_at(blocks, 5) == "train"
    assert assignment_at(blocks, 25) == "train"
    assert assignment_at(blocks, 15) is None


def test_overlap_is_rejected_and_touching_endpoints_are_allowed() -> None:
    with pytest.raises(LabError) as different:
        validate_role_blocks(
            [
                RoleBlock(start_s=0, end_s=10, role="train"),
                RoleBlock(start_s=9, end_s=12, role="validation"),
            ]
        )
    assert different.value.code == "block_overlap"
    with pytest.raises(LabError) as same:
        validate_role_blocks(
            [
                RoleBlock(start_s=0, end_s=10, role="train"),
                RoleBlock(start_s=5, end_s=15, role="train"),
            ]
        )
    assert same.value.code == "block_overlap"
    with pytest.raises(LabError) as role:
        validate_role_blocks([_Raw(0, 10, "excluded_guard")])
    assert role.value.code == "invalid_role"

    blocks = validate_role_blocks(_touching())
    assert assignment_at(blocks, 0) == "train"
    assert assignment_at(blocks, 9.999) == "train"
    assert assignment_at(blocks, 10) == "test"
    assert assignment_at(blocks, 20) is None


def test_a_second_outside_every_block_has_no_role() -> None:
    blocks = validate_role_blocks(_blocks())
    assert assignment_at(blocks, 65) is None
    assert assignment_at(blocks, 124) is None
    assert assignment_at(blocks, 200) is None
    assert "excluded_guard" not in {assignment_at(blocks, time) for time in (65, 124, 200)}


def test_propose_does_not_invent_a_split(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    add_episode(
        workspace,
        "pilot",
        EpisodeSpec("test-ep", "vid", "g-test", 0, 60, "погрузка"),
    )
    with pytest.raises(LabError) as service:
        propose_for_project(workspace, "pilot", "vid")
    assert service.value.code == "split_propose_refused"
    with pytest.raises(LabError) as direct:
        propose_split("vid", [])
    assert direct.value.code == "split_propose_refused"
    result = _RUNNER.invoke(
        app,
        [
            "dataset",
            "split-propose",
            "--project",
            "pilot",
            "--video",
            "vid",
            "--workspace",
            str(workspace),
        ],
    )
    assert result.exit_code != 0
    assert "split_propose_refused" in f"{result.stdout}{result.stderr}"
    project = storage.project_directory(workspace, "pilot")
    assert not (project / "split_proposals").exists()
    assert not (project / "split_manifests").exists()
    assert len(storage.list_episodes(workspace, "pilot", "vid")) == 1


def test_blocks_replace_before_freeze_and_not_after(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    path = write_blocks(
        workspace,
        "pilot",
        "vid",
        [RoleBlock(start_s=0, end_s=10, role="train")],
    )
    assert storage.list_episodes(workspace, "pilot", "vid") == []
    write_blocks(
        workspace,
        "pilot",
        "vid",
        [RoleBlock(start_s=0, end_s=12, role="validation")],
    )
    proposal = split_mod.SplitProposal.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )
    assert proposal.frozen is False
    assert proposal.independence.value == "same_scene"
    assert proposal.message == SAME_SCENE_MESSAGE
    assert proposal.blocks[0].end_s == 12
    assert proposal.blocks[0].role == "validation"
    assert proposal.episodes == []
    assert proposal.guards == []
    text = path.read_text(encoding="utf-8")
    assert "excluded_guard" not in text
    assert "guard_gap_s" not in text
    assert "episode_split" not in text

    frozen, manifest = freeze_split(workspace, "pilot", path)
    assert frozen.independence.value == "same_scene"
    assert frozen.assignment_at(6) == "validation"
    assert manifest.read_text(encoding="utf-8").find("excluded_guard") == -1
    before = manifest.read_bytes()
    proposal_before = path.read_bytes()
    with pytest.raises(LabError) as again:
        write_blocks(
            workspace,
            "pilot",
            "vid",
            [RoleBlock(start_s=0, end_s=15, role="train")],
        )
    assert again.value.code == "split_frozen"
    with pytest.raises(LabError) as refreeze:
        freeze_split(workspace, "pilot", path)
    assert refreeze.value.code == "split_frozen"
    assert manifest.read_bytes() == before
    assert path.read_bytes() == proposal_before
    assert frozen_manifest_path(workspace, "pilot", "vid").read_bytes() == before


def test_episode_id_does_not_choose_the_role() -> None:
    frozen = _frozen_with_leftover_episode()
    assert frozen.episodes
    assert frozen.assignment_at(4) == "train"
    assert frozen.assignment_at(10) == "test"
    assert frozen.assignment_at(25) is None
    assert sampler_mod._assignment_at(frozen, 4) == "train"
    assert sampler_mod._in_guard(frozen, 4) is False
    assert sampler_mod._in_guard(frozen, 65) is False
    windows = sampler_mod._windows(frozen, SplitName.train)
    assert [(window.episode_id, window.start_s, window.end_s) for window in windows] == [
        ("", 0.0, 10.0)
    ]
    assert quality_mod._shown_split(frozen, "test-ep", 4.0) == "train"
    assert quality_mod._shown_split(frozen, "test-ep", 65.0) == "unknown"
    assert quality_mod._assignment_at(frozen, 4.0) == "train"

    frame = SimpleNamespace(
        video_id="vid",
        source_frame_index=1,
        source_time_s=4.0,
        episode_id="test-ep",
        split=SplitName.train,
        split_group_id="g-test",
    )
    snapshot_mod._assert_frozen_split({"vid": frozen}, frame)
    refused = SimpleNamespace(
        video_id="vid",
        source_frame_index=1,
        source_time_s=4.0,
        episode_id="test-ep",
        split=SplitName.test,
        split_group_id="g-test",
    )
    with pytest.raises(LabError) as mismatch:
        snapshot_mod._assert_frozen_split({"vid": frozen}, refused)
    assert mismatch.value.code == "split_mismatch"
    outside = SimpleNamespace(
        video_id="vid",
        source_frame_index=2,
        source_time_s=25.0,
        episode_id="test-ep",
        split=SplitName.train,
        split_group_id="g-test",
    )
    with pytest.raises(LabError) as unassigned:
        snapshot_mod._assert_frozen_split({"vid": frozen}, outside)
    assert unassigned.value.code == "frame_unassigned"

    row = SimpleNamespace(
        frame_id="vid:1",
        source_time_s=4.0,
        episode_id="test-ep",
        split=SplitName.train,
        split_group_id="g-test",
    )
    coco_mod._match_frozen(frozen, row)
    assert coco_mod._expected_assignment(frozen, row) == "train"
    assert train_mod._frozen_role(Path("/tmp"), "pilot", frame, {"vid": frozen}) == "train"
    fields = cvat_mod._frozen_fields(
        frozen,
        SimpleNamespace(source_time_s=4.0, episode_id="test-ep"),
    )
    assert fields is not None
    assert fields[0] is SplitName.train
    assert fields[1] == ""

    held = active_mod._Frame(video_id="vid", index=1, time_s=4.0, episodes={"test-ep"})
    assert active_mod._is_train(frozen, held) is True
    gap = active_mod._Frame(video_id="vid", index=2, time_s=65.0, episodes={"train-ep"})
    assert active_mod._is_train(frozen, gap) is False
