"""Action intervals, role blocks, and stale replacement. Uses tmp_path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from excavation_lab import storage
from excavation_lab.actions.catalog import seed_actions
from excavation_lab.actions.labels import (
    add_interval,
    delete_interval,
    list_intervals,
    replace_intervals,
    revision,
    role_of,
    update_interval,
)
from excavation_lab.cli import app
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.splits import current_blocks, freeze_split, write_blocks
from excavation_lab.errors import LabError

_RUNNER = CliRunner()
_TITLES = {
    "digging": "Dig",
    "swing_loaded": "Swing loaded",
    "dumping": "Dump",
    "swing_empty": "Swing empty",
    "idle": "Idle",
}
_BLOCKS = [
    RoleBlock(start_s=0, end_s=30, role="train"),
    RoleBlock(start_s=40, end_s=70, role="validation"),
    RoleBlock(start_s=80, end_s=100, role="test"),
]


def _ready(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("mvp", workspace, catalog_id="excavation")
    storage.insert_video(workspace, "mvp", "vid", name="clip")
    seed_actions(workspace, "mvp", _TITLES)
    write_blocks(workspace, "mvp", "vid", _BLOCKS)
    return workspace


def test_role_of_train_validation_and_test(tmp_path: Path) -> None:
    workspace = _ready(tmp_path)
    blocks = current_blocks(workspace, "mvp", "vid")
    assert role_of(SimpleNamespace(start_s=0, end_s=30), blocks) == "train"
    assert role_of(SimpleNamespace(start_s=40, end_s=50), blocks) == "validation"
    assert role_of(SimpleNamespace(start_s=80, end_s=90), blocks) == "test"
    assert role_of(SimpleNamespace(start_s=32, end_s=38), blocks) is None
    assert role_of(SimpleNamespace(start_s=28, end_s=42), blocks) is None


def test_end_past_probed_duration_is_refused(tmp_path: Path) -> None:
    workspace = _ready(tmp_path)
    with storage.connect(storage.project_sqlite(workspace, "mvp")) as connection:
        connection.execute(
            "UPDATE videos SET duration_s = ? WHERE video_id = ?",
            (10.0, "vid"),
        )
    with pytest.raises(LabError) as refused:
        add_interval(workspace, "mvp", "vid", 12, 20, "digging")
    assert refused.value.code == "interval_past_end"
    assert "10" in refused.value.message
    assert list_intervals(workspace, "mvp", "vid") == []
    stored = add_interval(workspace, "mvp", "vid", 2, 8, "digging")
    assert (stored.start_s, stored.end_s) == (2, 8)


def test_interval_in_a_draft_block_is_stored(tmp_path: Path) -> None:
    workspace = _ready(tmp_path)
    stored = add_interval(workspace, "mvp", "vid", 12.0, 18.5, "digging")
    assert stored.end_s == 18.5
    assert role_of(stored, current_blocks(workspace, "mvp", "vid")) == "train"
    touching = add_interval(workspace, "mvp", "vid", 18.5, 24, "swing_loaded")
    assert touching.start_s == 18.5
    assert [item.action_name for item in list_intervals(workspace, "mvp", "vid")] == [
        "digging",
        "swing_loaded",
    ]


def test_overlap_boundary_and_gap_are_refused_without_clipping(tmp_path: Path) -> None:
    workspace = _ready(tmp_path)
    add_interval(workspace, "mvp", "vid", 0, 10, "digging")
    with pytest.raises(LabError) as overlap:
        add_interval(workspace, "mvp", "vid", 9, 12, "idle")
    assert overlap.value.code == "action_overlap"
    with pytest.raises(LabError) as crosses:
        add_interval(workspace, "mvp", "vid", 28, 42, "dumping")
    assert crosses.value.code == "action_crosses_block"
    assert "30" in crosses.value.message
    assert "не обрезается" in crosses.value.message
    with pytest.raises(LabError) as outside:
        add_interval(workspace, "mvp", "vid", 32, 38, "idle")
    assert outside.value.code == "action_outside_block"
    assert "не обрезается" in outside.value.message
    listed = list_intervals(workspace, "mvp", "vid")
    assert [(item.start_s, item.end_s, item.action_name) for item in listed] == [
        (0, 10, "digging"),
    ]


def test_frozen_blocks_win_over_a_tampered_proposal(tmp_path: Path) -> None:
    workspace = _ready(tmp_path)
    path = write_blocks(workspace, "mvp", "vid", _BLOCKS)
    freeze_split(workspace, "mvp", path)
    path.write_text("{}\n", encoding="utf-8")
    stored = add_interval(workspace, "mvp", "vid", 10, 20, "digging")
    assert role_of(stored, current_blocks(workspace, "mvp", "vid")) == "train"
    validation = add_interval(workspace, "mvp", "vid", 40, 55, "dumping")
    test = add_interval(workspace, "mvp", "vid", 80, 90, "idle")
    assert role_of(validation, current_blocks(workspace, "mvp", "vid")) == "validation"
    assert role_of(test, current_blocks(workspace, "mvp", "vid")) == "test"


def test_update_refuses_a_gap_and_delete_removes_the_row(tmp_path: Path) -> None:
    workspace = _ready(tmp_path)
    stored = add_interval(workspace, "mvp", "vid", 1, 5, "digging")
    with pytest.raises(LabError) as moved:
        update_interval(workspace, "mvp", stored.label_id, start_s=32, end_s=38)
    assert moved.value.code == "action_outside_block"
    assert list_intervals(workspace, "mvp", "vid")[0].start_s == 1
    changed = update_interval(
        workspace,
        "mvp",
        stored.label_id,
        action_name="idle",
        end_s=8,
    )
    assert changed.action_name == "idle"
    assert changed.end_s == 8
    assert changed.label_id == stored.label_id
    delete_interval(workspace, "mvp", stored.label_id)
    assert list_intervals(workspace, "mvp", "vid") == []


def test_stale_replace_is_refused_and_a_current_one_writes(tmp_path: Path) -> None:
    workspace = _ready(tmp_path)
    add_interval(workspace, "mvp", "vid", 0, 5, "digging")
    stale = revision(list_intervals(workspace, "mvp", "vid"))
    assert stale.count == 1
    assert len(stale.digest) == 64
    add_interval(workspace, "mvp", "vid", 5, 9, "idle")
    with pytest.raises(LabError) as refused:
        replace_intervals(
            workspace,
            "mvp",
            "vid",
            [{"start_s": 1, "end_s": 2, "action_name": "dumping"}],
            expected_revision=stale,
        )
    assert refused.value.code == "stale_labels"
    assert len(list_intervals(workspace, "mvp", "vid")) == 2
    current = revision(list_intervals(workspace, "mvp", "vid"))
    assert current.count == 2
    assert current.digest != stale.digest
    replaced = replace_intervals(
        workspace,
        "mvp",
        "vid",
        [
            {"start_s": 40, "end_s": 50, "action_name": "dumping"},
            {"start_s": 50, "end_s": 60, "action_name": "swing_empty"},
        ],
        expected_revision=current,
    )
    assert [item.action_name for item in replaced] == ["dumping", "swing_empty"]
    assert revision(replaced) == revision(list_intervals(workspace, "mvp", "vid"))


def test_cli_label_add_list_and_delete(tmp_path: Path) -> None:
    workspace = _ready(tmp_path)
    added = _RUNNER.invoke(
        app,
        [
            "actions",
            "label",
            "add",
            "--project",
            "mvp",
            "--video",
            "vid",
            "--start",
            "12.0",
            "--end",
            "18.5",
            "--action",
            "digging",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert added.exit_code == 0
    body = json.loads(added.stdout)["data"]
    assert body["action_name"] == "digging"
    assert body["start_s"] == 12.0
    assert body["end_s"] == 18.5
    listed = _RUNNER.invoke(
        app,
        [
            "actions",
            "label",
            "list",
            "--project",
            "mvp",
            "--video",
            "vid",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert listed.exit_code == 0
    assert json.loads(listed.stdout)["data"][0]["label_id"] == body["label_id"]
    removed = _RUNNER.invoke(
        app,
        [
            "actions",
            "label",
            "delete",
            "--project",
            "mvp",
            "--label",
            body["label_id"],
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert removed.exit_code == 0
    assert list_intervals(workspace, "mvp", "vid") == []
    refused = _RUNNER.invoke(
        app,
        [
            "actions",
            "label",
            "add",
            "--project",
            "mvp",
            "--video",
            "vid",
            "--start",
            "12",
            "--end",
            "12",
            "--action",
            "digging",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert refused.exit_code != 0
    assert json.loads(refused.stdout)["ok"] is False
