"""Action catalog, bundle class_order, and action-model rows. Uses tmp_path."""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from excavation_lab import storage
from excavation_lab.actions.catalog import (
    SEED_ACTIONS,
    add_action,
    delete_action,
    edit_title,
    list_actions,
    rename_action,
    seed_actions,
)
from excavation_lab.actions.labels import add_interval
from excavation_lab.cli import app
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.splits import write_blocks
from excavation_lab.errors import LabError
from excavation_lab.ui.copy import action_seed_titles

_RUNNER = CliRunner()
_REPO = Path(__file__).resolve().parents[2]
_TITLES = {
    "digging": "Dig",
    "swing_loaded": "Swing loaded custom",
    "dumping": "Dump",
    "swing_empty": "Swing empty custom",
    "idle": "Idle custom",
}


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("mvp", workspace, catalog_id="excavation")
    storage.insert_video(workspace, "mvp", "vid", name="clip")
    return workspace


def test_actions_package_does_not_import_ui() -> None:
    root = _REPO / "src" / "excavation_lab" / "actions"
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.Import):
                module = " ".join(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            if module == "ui" or module.startswith("excavation_lab.ui"):
                offenders.append(path.name)
    assert offenders == []


def test_seed_stores_titles_it_is_given_and_is_idempotent(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    seeded = seed_actions(workspace, "mvp", _TITLES)
    assert [item.name for item in seeded] == list(SEED_ACTIONS)
    assert [item.id for item in seeded] == [0, 1, 2, 3, 4]
    assert seeded[0].title == "Dig"
    assert seeded[0].added_at.tzinfo is not None
    again = seed_actions(workspace, "mvp", action_seed_titles["ru"])
    assert [item.title for item in again] == [item.title for item in seeded]
    assert [item.id for item in again] == [item.id for item in seeded]


def test_seventeenth_class_and_duplicate_name_are_refused(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    seed_actions(workspace, "mvp", _TITLES)
    for index in range(11):
        add_action(workspace, "mvp", f"extra_{index:02d}", f"Extra {index}")
    assert len(list_actions(workspace, "mvp")) == 16
    with pytest.raises(LabError) as full:
        add_action(workspace, "mvp", "extra_11", "Nope")
    assert full.value.code == "action_catalog_full"
    assert len(list_actions(workspace, "mvp")) == 16
    with pytest.raises(LabError) as duplicate:
        add_action(workspace, "mvp", "digging", "Other")
    assert duplicate.value.code == "duplicate_action_name"


def test_edit_title_and_rename_keep_id(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    seeded = seed_actions(workspace, "mvp", _TITLES)
    idle_id = next(item.id for item in seeded if item.name == "idle")
    edited = edit_title(workspace, "mvp", "idle", "Ожидание")
    assert edited.id == idle_id
    assert edited.name == "idle"
    assert edited.title == "Ожидание"
    assert edited.added_at == next(item.added_at for item in seeded if item.name == "idle")
    renamed = rename_action(workspace, "mvp", "swing_empty", "return_swing")
    empty_id = next(item.id for item in seeded if item.name == "swing_empty")
    assert renamed.id == empty_id
    assert renamed.name == "return_swing"
    names = {item.name: item.id for item in list_actions(workspace, "mvp")}
    assert names["digging"] == 0
    assert "swing_empty" not in names
    assert names["return_swing"] == empty_id


def test_rename_and_delete_refused_when_a_label_uses_the_name(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    seed_actions(workspace, "mvp", _TITLES)
    write_blocks(
        workspace,
        "mvp",
        "vid",
        [RoleBlock(start_s=0, end_s=30, role="train")],
    )
    add_interval(workspace, "mvp", "vid", 1, 5, "digging")
    with pytest.raises(LabError) as renamed:
        rename_action(workspace, "mvp", "digging", "excavate")
    assert renamed.value.code == "action_in_use"
    with pytest.raises(LabError) as deleted:
        delete_action(workspace, "mvp", "digging")
    assert deleted.value.code == "action_in_use"
    assert any(item.name == "digging" for item in list_actions(workspace, "mvp"))


def test_delete_refused_when_bundle_class_order_names_it(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    seed_actions(workspace, "mvp", _TITLES)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "action_bundle.json").write_text(
        json.dumps({"class_order": ["digging", "idle"]}),
        encoding="utf-8",
    )
    storage.insert_action_model(
        workspace,
        "mvp",
        action_model_id="am1",
        detector_model_id="det1",
        source_run_ids=["run-b", "run-a"],
        path=bundle,
    )
    listed = storage.list_action_models(workspace, "mvp")
    assert listed[0]["source_run_ids"] == ["run-b", "run-a"]
    db = storage.project_sqlite(workspace, "mvp")
    with sqlite3.connect(db) as connection:
        stored = connection.execute("SELECT source_run_ids FROM action_models").fetchone()
    assert stored is not None
    assert stored[0] == '["run-b","run-a"]'
    with pytest.raises(LabError) as refused:
        delete_action(workspace, "mvp", "digging")
    assert refused.value.code == "action_in_use"
    with pytest.raises(LabError) as renamed:
        rename_action(workspace, "mvp", "idle", "waiting")
    assert renamed.value.code == "action_in_use"
    delete_action(workspace, "mvp", "swing_loaded")
    assert "swing_loaded" not in {item.name for item in list_actions(workspace, "mvp")}


def test_old_schema_gains_action_tables_without_dropping_rows(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    db = storage.project_sqlite(workspace, "mvp")
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT name FROM videos").fetchone()[0] == "clip"
        connection.execute("DROP TABLE action_classes")
        connection.execute("DROP TABLE action_labels")
        connection.execute("DROP TABLE action_models")
        connection.execute("DROP TABLE IF EXISTS action_inferences")
    with sqlite3.connect(db) as connection:
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "action_classes" not in names
    assert list_actions(workspace, "mvp") == []
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT name FROM videos").fetchone()[0] == "clip"
        assert connection.execute("SELECT catalog_id FROM projects").fetchone()[0] == "excavation"
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"action_classes", "action_labels", "action_models", "action_inferences"} <= names
    seeded = seed_actions(workspace, "mvp", _TITLES)
    assert len(seeded) == 5
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 1


def test_delete_project_removes_action_rows_with_the_directory(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    seed_actions(workspace, "mvp", _TITLES)
    project_dir = storage.project_directory(workspace, "mvp")
    storage.delete_project("mvp", workspace)
    assert not project_dir.exists()


def test_cli_seed_uses_locale_titles_and_errors_exit_nonzero(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    english = _RUNNER.invoke(
        app,
        [
            "actions",
            "catalog",
            "seed",
            "--project",
            "mvp",
            "--workspace",
            str(workspace),
            "--locale",
            "en",
            "--json",
        ],
    )
    assert english.exit_code == 0
    payload = json.loads(english.stdout)
    assert payload["ok"] is True
    assert payload["data"][0]["name"] == "digging"
    assert payload["data"][0]["title"] == action_seed_titles["en"]["digging"]
    russian = _RUNNER.invoke(
        app,
        [
            "actions",
            "catalog",
            "seed",
            "--project",
            "mvp",
            "--workspace",
            str(workspace),
            "--locale",
            "ru",
            "--json",
        ],
    )
    assert russian.exit_code == 0
    assert json.loads(russian.stdout)["data"][0]["title"] == action_seed_titles["en"]["digging"]
    missing = _RUNNER.invoke(
        app,
        [
            "actions",
            "catalog",
            "list",
            "--project",
            "missing",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert missing.exit_code != 0
    assert json.loads(missing.stdout)["error"]["code"] == "project_not_found"
    assert "workspace/projects/mvp" not in str(workspace)
