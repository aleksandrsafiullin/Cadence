"""Display-name rename leaves the project directory in place."""

from __future__ import annotations

from pathlib import Path

import pytest

from excavation_lab import storage
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.errors import LabError


def _project(workspace: Path, name: str) -> storage.ProjectRecord:
    list_catalogs(workspace)
    return storage.create_project(name, workspace, catalog_id="excavation")


def test_rename_changes_the_display_name_and_keeps_the_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    record = _project(workspace, "pilot")
    directory = Path(record["project_dir"])
    renamed = storage.rename_project("pilot", "  Пилот карьера  ", workspace)
    assert renamed["project_id"] == "pilot"
    assert renamed["name"] == "Пилот карьера"
    assert renamed["project_dir"] == str(directory)
    assert directory.is_dir()
    assert [item.name for item in (workspace / "projects").iterdir()] == ["pilot"]
    listed = storage.list_projects(workspace)
    assert [(item["project_id"], item["name"]) for item in listed] == [
        ("pilot", "Пилот карьера")
    ]
    with storage.connect(directory / "project.sqlite") as connection:
        row = connection.execute("SELECT project_id, name FROM projects").fetchone()
    assert row is not None
    assert row["project_id"] == "pilot"
    assert row["name"] == "Пилот карьера"


def test_rename_rejects_a_duplicate_a_blank_name_and_an_unknown_id(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    record = _project(workspace, "pilot")
    _project(workspace, "mvp")
    directory = Path(record["project_dir"])
    with pytest.raises(LabError) as duplicate:
        storage.rename_project("pilot", "mvp", workspace)
    assert duplicate.value.code == "duplicate_project_name"
    assert duplicate.value.message == "Проект «mvp» уже существует."
    for blank in ("", " ", "\t\n"):
        with pytest.raises(LabError) as invalid:
            storage.rename_project("pilot", blank, workspace)
        assert invalid.value.code == "invalid_project_name"
    with pytest.raises(LabError) as missing:
        storage.rename_project("missing", "Other", workspace)
    assert missing.value.code == "project_not_found"
    assert "«missing»" in missing.value.message
    assert directory.is_dir()
    assert [item.name for item in sorted((workspace / "projects").iterdir())] == [
        "mvp",
        "pilot",
    ]
    listed = {item["project_id"]: item["name"] for item in storage.list_projects(workspace)}
    assert listed == {"mvp": "mvp", "pilot": "pilot"}
