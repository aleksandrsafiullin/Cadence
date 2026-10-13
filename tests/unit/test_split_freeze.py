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
from excavation_lab.datasets.episodes import EpisodeSpec
from excavation_lab.datasets.splits import (
    load_frozen,
    propose_split,
    require_frozen,
    write_proposal,
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


def test_freeze_hashes_proposal_bytes_and_rejects_tamper_and_refreeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    storage.create_project("pilot", workspace)
    storage.insert_video(workspace, "pilot", "vid", name="pilot")
    specs = [
        EpisodeSpec("train-ep", "vid", "g-train", 0, 60, "погрузка"),
        EpisodeSpec("val-ep", "vid", "g-val", 72, 92, "погрузка"),
        EpisodeSpec("test-ep", "vid", "g-test", 104, 124, "погрузка"),
    ]
    proposal = propose_split("vid", specs)
    assert proposal.guard_gap_s == 10
    assert proposal.frozen is False
    proposal_path = write_proposal(workspace, "pilot", proposal)
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
    assert data["independence"] == "episode_split"
    assert data["guard_gap_s"] == 10
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
        "guard_gap_s",
        "episodes",
        "guards",
        "content_hash",
    ):
        assert key in document
    assert document["guards"]
    assert all(guard["role"] == "excluded_guard" for guard in document["guards"])
    stored_hash = document.pop("content_hash")
    assert _canonical(document) == expected_body
    assert stored_hash == digest
    assert [episode["split"] for episode in document["episodes"]] == [
        "train",
        "validation",
        "test",
    ]

    loaded = require_frozen(workspace, "pilot", "vid")
    assert loaded.content_hash == digest
    assert loaded.frozen is True
    assert [(episode.episode_id, episode.split.value) for episode in loaded.episodes] == [
        (episode.episode_id, episode.split.value) for episode in proposal.episodes
    ]
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
    assert tampered["episodes"][0]["episode_id"] == "train-ep"
    assert tampered["episodes"][0]["split"] == "train"
    tampered["episodes"][0]["split"] = "test"
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
