"""Frozen split_manifest.json: canonical hash, tamper, and no second freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from excavation_lab import storage
from excavation_lab.cli import app
from excavation_lab.datasets import splits as split_mod
from excavation_lab.datasets.blocks import GUARD_GAP_S, RoleBlock, guard_gap_warnings
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.splits import (
    SAME_SCENE_MESSAGE,
    SplitProposal,
    load_frozen,
    require_frozen,
    write_blocks,
)
from excavation_lab.errors import LabError, SplitHashMismatchError, SplitNotFrozenError
from excavation_lab.schemas import SCHEMA_VERSION

_RUNNER = CliRunner()


def _canonical(payload: dict) -> bytes:
    """The hash contract: UTF-8, sorted keys, compact separators, no newline."""
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert not text.endswith("\n")
    return text.encode("utf-8")


def _cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)


def _forbid_propose(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("propose_split")


def test_guard_gap_warnings_list_close_role_boundaries_only() -> None:
    blocks = [
        RoleBlock(start_s=0, end_s=60, role="train"),
        RoleBlock(start_s=60, end_s=70, role="train"),
        RoleBlock(start_s=75, end_s=90, role="validation"),
        RoleBlock(start_s=100, end_s=110, role="test"),
        RoleBlock(start_s=119.999999, end_s=130, role="train"),
        RoleBlock(start_s=138, end_s=150, role="validation"),
    ]
    found = guard_gap_warnings(blocks)
    pairs = [(item.earlier.end_s, item.later.start_s, item.gap_s) for item in found]
    # train→train touching: not a boundary. 90→100: exactly the guard. 110→119.999999:
    # within tolerance of the guard. 130→138: under.
    assert pairs == [(70.0, 75.0, 5.0), (130.0, 138.0, 8.0)]
    assert found[0].earlier.role == "train" and found[0].later.role == "validation"
    assert guard_gap_warnings([]) == []
    assert guard_gap_warnings(blocks[:1]) == []
    assert GUARD_GAP_S == 10.0


def test_cli_freeze_prints_guard_gap_warnings_and_still_freezes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    storage.insert_video(workspace, "pilot", "vid", name="pilot")
    proposal_path = write_blocks(
        workspace,
        "pilot",
        "vid",
        [
            RoleBlock(start_s=0, end_s=60, role="train"),
            RoleBlock(start_s=63, end_s=90, role="validation"),
        ],
    )
    monkeypatch.setattr(split_mod, "propose_split", _forbid_propose)
    base = ["dataset", "split-freeze", "--project", "pilot", "--manifest", str(proposal_path)]
    base += ["--workspace", str(workspace)]
    text = _RUNNER.invoke(app, base)
    assert text.exit_code == 0, text.stdout
    assert "Разбиение зафиксировано" in text.stdout
    assert "меньше 10 с" in text.stdout
    assert "train (60 с)" in text.stdout and "validation (63 с)" in text.stdout
    assert require_frozen(workspace, "pilot", "vid").blocks[1].start_s == 63.0

    again = _RUNNER.invoke(app, [*base, "--json"])
    assert again.exit_code != 0
    assert json.loads(again.stdout)["error"]["code"] == "split_frozen"

    storage.insert_video(workspace, "pilot", "vid2", name="second")
    second = write_blocks(
        workspace,
        "pilot",
        "vid2",
        [
            RoleBlock(start_s=0, end_s=60, role="train"),
            RoleBlock(start_s=63, end_s=90, role="validation"),
            RoleBlock(start_s=100, end_s=120, role="test"),
        ],
    )
    as_json = _RUNNER.invoke(
        app,
        ["dataset", "split-freeze", "--project", "pilot", "--manifest", str(second)]
        + ["--workspace", str(workspace), "--json"],
    )
    assert as_json.exit_code == 0, as_json.stdout
    data = json.loads(as_json.stdout)["data"]
    assert data["frozen"] is True
    assert data["guard_gap_warnings"] == [
        {
            "earlier": {"start_s": 0.0, "end_s": 60.0, "role": "train"},
            "later": {"start_s": 63.0, "end_s": 90.0, "role": "validation"},
            "gap_s": 3.0,
        }
    ]
    assert "guard_gap_warnings" not in json.loads(
        Path(data["path"]).read_text(encoding="utf-8")
    )


def test_freeze_hashes_proposal_bytes_and_rejects_tamper_and_refreeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    storage.insert_video(workspace, "pilot", "vid", name="pilot")
    blocks = [
        RoleBlock(start_s=0, end_s=60, role="train"),
        RoleBlock(start_s=72, end_s=92, role="validation"),
        RoleBlock(start_s=104, end_s=124, role="test"),
    ]
    proposal_path = write_blocks(workspace, "pilot", "vid", blocks)
    proposal = SplitProposal.model_validate(json.loads(proposal_path.read_text(encoding="utf-8")))
    assert proposal.frozen is False
    assert proposal.independence.value == "same_scene"
    reloaded = json.loads(proposal_path.read_text(encoding="utf-8"))
    assert reloaded["frozen"] is False

    with pytest.raises(SplitNotFrozenError) as missing:
        require_frozen(workspace, "pilot", "vid")
    assert isinstance(missing.value, LabError)
    assert missing.value.code == "split_not_frozen"
    assert missing.value.code.isascii()
    assert _cyrillic(missing.value.message)

    help_text = _RUNNER.invoke(app, ["dataset", "split-freeze", "--help"])
    assert help_text.exit_code == 0
    assert "--manifest" in help_text.stdout
    assert "--force" not in help_text.stdout

    monkeypatch.setattr(split_mod, "propose_split", _forbid_propose)
    args = [
        "dataset",
        "split-freeze",
        "--project",
        "pilot",
        "--manifest",
        str(proposal_path),
        "--workspace",
        str(workspace),
        "--json",
    ]
    first = _RUNNER.invoke(app, args)
    assert first.exit_code == 0, first.stdout
    payload = json.loads(first.stdout)
    assert payload["ok"] is True
    data = payload["data"]
    assert data["frozen"] is True
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["video_id"] == "vid"
    assert data["independence"] == "same_scene"
    assert "guard_gap_s" not in data
    assert data["message"] == SAME_SCENE_MESSAGE
    manifest_path = Path(data["path"])
    project_dir = storage.project_directory(workspace, "pilot")
    assert manifest_path.name == "split_manifest.json"
    assert manifest_path.is_relative_to(project_dir)
    assert manifest_path != proposal_path

    body = proposal.model_dump(mode="json")
    assert body["frozen"] is False
    body["frozen"] = True
    assert "content_hash" not in body
    expected_body = _canonical(body)
    digest = hashlib.sha256(expected_body).hexdigest()
    assert hashlib.sha256(expected_body + b"\n").hexdigest() != digest
    full = dict(body)
    full["content_hash"] = digest
    expected_file = _canonical(full)
    assert not expected_file.endswith(b"\n")
    assert manifest_path.read_bytes() == expected_file

    document = json.loads(expected_file)
    assert document["content_hash"] == digest
    for key in (
        "schema_version",
        "video_id",
        "independence",
        "limitation",
        "message",
        "blocks",
        "episodes",
        "guards",
        "content_hash",
    ):
        assert key in document
    assert "guard_gap_s" not in document
    assert "excluded_guard" not in expected_file.decode("utf-8")
    assert "episode_split" not in expected_file.decode("utf-8")
    assert document["episodes"] == []
    assert document["guards"] == []
    assert document["message"] == SAME_SCENE_MESSAGE
    assert "переносим" not in document["message"]
    assert "transfer" not in document["message"].casefold()
    stored_hash = document.pop("content_hash")
    assert _canonical(document) == expected_body
    assert stored_hash == digest
    assert [block["role"] for block in document["blocks"]] == [
        "train",
        "validation",
        "test",
    ]

    loaded = require_frozen(workspace, "pilot", "vid")
    assert loaded.content_hash == digest
    assert loaded.frozen is True
    assert [(block.role, block.start_s, block.end_s) for block in loaded.blocks] == [
        (block.role, block.start_s, block.end_s) for block in proposal.blocks
    ]
    assert loaded.assignment_at(10) == "train"
    assert loaded.assignment_at(65) is None
    assert json.loads(proposal_path.read_text(encoding="utf-8"))["frozen"] is False

    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        rows = connection.execute(
            "SELECT content_hash, frozen, path FROM split_manifests"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["content_hash"] == digest
    assert rows[0]["frozen"] == 1
    assert str(rows[0]["path"]).endswith("split_manifest.json")

    second = _RUNNER.invoke(app, args)
    assert second.exit_code != 0
    failure = json.loads(second.stdout)
    assert failure["ok"] is False
    assert failure["error"]["code"] == "split_frozen"
    assert failure["error"]["code"].isascii()
    assert _cyrillic(failure["error"]["message"])
    assert manifest_path.read_bytes() == expected_file
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        count = connection.execute("SELECT COUNT(*) AS n FROM split_manifests").fetchone()
    assert count is not None
    assert count["n"] == 1

    forced = _RUNNER.invoke(app, [*args, "--force"])
    assert forced.exit_code != 0
    assert manifest_path.read_bytes() == expected_file

    manifest_path.write_bytes(expected_file + b"\n")
    assert load_frozen(workspace, "pilot", "vid").content_hash == digest
    manifest_path.write_bytes(expected_file)

    tampered = json.loads(manifest_path.read_bytes())
    assert tampered["blocks"][0]["role"] == "train"
    tampered["blocks"][0]["end_s"] = 70
    assert tampered["content_hash"] == digest
    manifest_path.write_bytes(json.dumps(tampered, ensure_ascii=False).encode("utf-8"))
    with pytest.raises(SplitHashMismatchError) as mismatch:
        load_frozen(workspace, "pilot", "vid")
    assert isinstance(mismatch.value, LabError)
    assert mismatch.value.code == "split_hash_mismatch"
    assert mismatch.value.code.isascii()
    assert _cyrillic(mismatch.value.message)
    with pytest.raises(SplitHashMismatchError):
        require_frozen(workspace, "pilot", "vid")
