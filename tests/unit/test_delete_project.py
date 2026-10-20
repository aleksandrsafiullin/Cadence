"""Deleting a project leaves the catalog and shared weights."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from excavation_lab import storage
from excavation_lab.datasets.catalog import (
    add_class,
    delete_class,
    edit_class,
    list_catalogs,
    read_catalog,
    rename_class,
    require_project_catalog,
)
from excavation_lab.datasets.class_supervision import class_supervision
from excavation_lab.datasets.labels import validate_frame
from excavation_lab.errors import CheckpointMissingError, LabError, PathNotAllowedError
from excavation_lab.tracking.analyze import resolve_model


def _project(workspace: Path, name: str) -> storage.ProjectRecord:
    list_catalogs(workspace)
    return storage.create_project(name, workspace, catalog_id="excavation")


def _remember(connection_path: Path, model_id: str, weights: Path) -> None:
    with storage.connect(connection_path) as connection:
        connection.execute(
            """
            INSERT INTO model_versions (
                model_id, project_id, dataset_id, variant, path, created_at
            ) VALUES (?, ?, NULL, 'nano', ?, ?)
            """,
            (
                model_id,
                connection_path.parent.name,
                str(weights.resolve()),
                datetime.now(UTC).isoformat(),
            ),
        )


def test_delete_removes_the_project_tree_and_keeps_catalog_and_weights(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    first = _project(workspace, "one")
    second = _project(workspace, "two")
    catalog = workspace / "catalogs" / "excavation" / "catalog.yaml"
    catalog_bytes = catalog.read_bytes()
    project_dir = Path(first["project_dir"])
    (project_dir / "media" / "clip.mp4").write_bytes(b"video")
    snapshot = project_dir / "dataset_versions" / "ds1"
    snapshot.mkdir()
    (snapshot / "split_manifest.json").write_text("{}", encoding="utf-8")
    run = project_dir / "runs" / "job-1"
    run.mkdir()
    (run / "run.json").write_text("{}", encoding="utf-8")

    shared = workspace / "weights" / "shared-model"
    shared.mkdir(parents=True)
    (shared / "weights.pth").write_bytes(b"shared-weights")
    (shared / "class_mapping.json").write_text("{}\n", encoding="utf-8")
    only = workspace / "weights" / "only-one"
    only.mkdir()
    (only / "weights.pth").write_bytes(b"only-weights")
    _remember(Path(first["sqlite_path"]), "shared-model", shared)
    _remember(Path(first["sqlite_path"]), "only-one", only)
    _remember(Path(second["sqlite_path"]), "shared-model", shared)

    storage.delete_project("one", workspace)
    assert not project_dir.exists()
    assert not (workspace / "projects" / "one").exists()
    assert Path(second["project_dir"]).is_dir()
    assert catalog.read_bytes() == catalog_bytes
    assert (shared / "weights.pth").read_bytes() == b"shared-weights"
    assert (only / "weights.pth").read_bytes() == b"only-weights"
    with storage.connect(Path(second["sqlite_path"])) as connection:
        row = connection.execute(
            "SELECT path FROM model_versions WHERE model_id = ?",
            ("shared-model",),
        ).fetchone()
    assert row is not None
    assert row["path"] == str(shared.resolve())

    with pytest.raises(LabError) as gone:
        storage.delete_project("one", workspace)
    assert gone.value.code == "project_not_found"
    assert "уже удалён" in gone.value.message
    assert catalog.read_bytes() == catalog_bytes
    assert (shared / "weights.pth").is_file()


def test_delete_refuses_a_symlink_outside_projects(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _project(workspace, "one")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("stay", encoding="utf-8")
    link = workspace / "projects" / "linked"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathNotAllowedError):
        storage.delete_project("linked", workspace)
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "stay"
    assert link.is_symlink()
    assert (workspace / "catalogs" / "excavation" / "catalog.yaml").is_file()


def test_delete_refuses_a_queued_or_running_lock_for_that_project(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _project(workspace, "one")
    _project(workspace, "two")
    database = workspace / "lab.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            """
            CREATE TABLE workspace_lock (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                job_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO workspace_lock (singleton, job_id, project_id, status)
            VALUES (1, 'job-1', 'two', 'running')
            """
        )
        connection.commit()
    finally:
        connection.close()
    storage.delete_project("one", workspace)
    assert not (workspace / "projects" / "one").exists()

    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE workspace_lock SET project_id = 'two', status = 'queued'"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(LabError) as queued:
        storage.delete_project("two", workspace)
    assert queued.value.code == "project_job_active"
    assert "«two»" in queued.value.message
    assert "в очереди" in queued.value.message
    assert (workspace / "projects" / "two").is_dir()

    connection = sqlite3.connect(database)
    try:
        connection.execute("UPDATE workspace_lock SET status = 'running'")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(LabError) as running:
        storage.delete_project("two", workspace)
    assert running.value.code == "project_job_active"
    assert "выполняется" in running.value.message
    assert (workspace / "projects" / "two").is_dir()
    connection = sqlite3.connect(database)
    try:
        row = connection.execute(
            "SELECT project_id, status FROM workspace_lock"
        ).fetchone()
    finally:
        connection.close()
    assert row == ("two", "running")


def test_delete_refuses_while_cancel_is_requested(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    record = _project(workspace, "one")
    directory = Path(record["project_dir"])
    database = workspace / "lab.sqlite"
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            """
            CREATE TABLE workspace_lock (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                job_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO workspace_lock (singleton, job_id, project_id, status)
            VALUES (1, 'job-1', 'one', 'cancel_requested')
            """
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(LabError) as caught:
        storage.delete_project("one", workspace)
    assert caught.value.code == "project_job_active"
    assert caught.value.message == "Проект «one» нельзя удалить: задание отменяется."
    assert directory.is_dir()
    connection = sqlite3.connect(database)
    try:
        row = connection.execute(
            "SELECT project_id, status FROM workspace_lock"
        ).fetchone()
    finally:
        connection.close()
    assert row == ("one", "cancel_requested")


def test_delete_refuses_when_lab_sqlite_is_a_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    record = _project(workspace, "one")
    directory = Path(record["project_dir"])
    outside = tmp_path / "outside"
    outside.mkdir()
    decoy = outside / "lab.sqlite"
    connection = sqlite3.connect(decoy)
    try:
        connection.execute("CREATE TABLE unused (id INTEGER)")
        connection.commit()
    finally:
        connection.close()
    link = workspace / "lab.sqlite"
    link.symlink_to(decoy)
    with pytest.raises(LabError) as caught:
        storage.delete_project("one", workspace)
    assert caught.value.code == "project_job_active"
    assert "символическая ссылка" in caught.value.message
    assert directory.is_dir()
    assert link.is_symlink()
    assert decoy.is_file()


def test_delete_refuses_when_projects_itself_is_a_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    catalog = workspace / "catalogs" / "excavation" / "catalog.yaml"
    before = catalog.read_bytes()
    projects = workspace / "projects"
    projects.symlink_to(workspace / "catalogs", target_is_directory=True)
    with pytest.raises(PathNotAllowedError):
        storage.delete_project("excavation", workspace)
    assert projects.is_symlink()
    assert catalog.read_bytes() == before


def test_stored_weights_path_wins_over_a_project_local_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    record = _project(workspace, "pilot")
    local = Path(record["project_dir"]) / "models" / "shared"
    local.mkdir()
    (local / "weights.pth").write_bytes(b"local")
    with pytest.raises(CheckpointMissingError):
        resolve_model(workspace, "pilot", "shared")
    shared = workspace / "weights" / "shared"
    shared.mkdir(parents=True)
    (shared / "weights.pth").write_bytes(b"shared")
    _remember(Path(record["sqlite_path"]), "shared", shared)
    _origin, directory = resolve_model(workspace, "pilot", "shared")
    assert directory is not None
    assert directory.resolve() == shared.resolve()
    assert (directory / "weights.pth").read_bytes() == b"shared"
    assert (local / "weights.pth").read_bytes() == b"local"


def test_class_name_in_weights_mapping_blocks_rename_and_delete(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    mapping = workspace / "weights" / "m1" / "class_mapping.json"
    mapping.parent.mkdir(parents=True)
    mapping.write_text(
        json.dumps(
            {
                "by_name": {"bucket": {"internal_id": 1, "coco_category_id": 1}},
                "class_names": ["bucket"],
            }
        ),
        encoding="utf-8",
    )
    catalog = workspace / "catalogs" / "excavation" / "catalog.yaml"
    before = catalog.read_bytes()
    mapping_bytes = mapping.read_bytes()
    with pytest.raises(LabError) as renamed:
        rename_class(workspace, "excavation", "bucket", "scoop")
    assert renamed.value.code == "class_in_use"
    assert catalog.read_bytes() == before
    edited = edit_class(workspace, "excavation", "bucket", title="Ковш на смене")
    assert edited.classes[1].name == "bucket"
    assert edited.classes[1].title == "Ковш на смене"
    added = add_class(workspace, "excavation", "shovel", "Лопата", "Видимая лопата.")
    assert added.classes[-1].name == "shovel"
    with pytest.raises(LabError) as removed:
        delete_class(workspace, "excavation", "bucket")
    assert removed.value.code == "class_in_use"
    kept = read_catalog(workspace, "excavation")
    assert any(item.name == "bucket" for item in kept.classes)
    assert any(item.name == "shovel" for item in kept.classes)
    assert mapping.read_bytes() == mapping_bytes


def test_second_project_keeps_shared_weights_when_shovel_follows_a_review(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    first = _project(workspace, "one")
    second = _project(workspace, "two")
    shared = workspace / "weights" / "shared-model"
    shared.mkdir(parents=True)
    weights = shared / "weights.pth"
    weights.write_bytes(b"shared-weights")
    (shared / "class_mapping.json").write_text("{}\n", encoding="utf-8")
    _remember(Path(first["sqlite_path"]), "shared-model", shared)
    _remember(Path(second["sqlite_path"]), "shared-model", shared)

    before = read_catalog(workspace, "excavation")
    reviewed_at = max(item.added_at for item in before.classes)
    record = validate_frame(
        {
            "video_id": "pit",
            "source_frame_index": 4,
            "review_status": "reviewed",
            "reviewed_at": reviewed_at,
            "boxes": [
                {
                    "category_name": "bucket",
                    "x1": 1.0,
                    "y1": 2.0,
                    "x2": 30.0,
                    "y2": 40.0,
                    "iscrowd": 0,
                }
            ],
        },
        before,
    )
    batch = Path(first["project_dir"]) / "annotation_batches" / "batch-1"
    batch.mkdir(parents=True)
    labels = batch / "imported_labels.json"
    labels.write_text(
        json.dumps(
            {"frames": [{"record": record.model_dump(mode="json")}]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    label_bytes = labels.read_bytes()
    weight_bytes = weights.read_bytes()

    added = add_class(workspace, "excavation", "shovel", "Лопата", "Видимая лопата.")
    shovel = added.classes[-1]
    assert shovel.name == "shovel"
    assert shovel.added_at > reviewed_at
    assert [item.name for item in added.classes[:4]] == [
        "excavator_body",
        "bucket",
        "haul_truck",
        "truck_bed",
    ]
    assert labels.read_bytes() == label_bytes
    assert weights.read_bytes() == weight_bytes
    assert Path(second["project_dir"]).is_dir()
    assert require_project_catalog(workspace, "two").classes[-1].name == "shovel"
    with storage.connect(Path(second["sqlite_path"])) as connection:
        row = connection.execute(
            "SELECT path FROM model_versions WHERE model_id = ?",
            ("shared-model",),
        ).fetchone()
    assert row is not None
    assert row["path"] == str(shared.resolve())
    _origin, directory = resolve_model(workspace, "two", "shared-model")
    assert directory is not None
    assert (directory / "weights.pth").read_bytes() == weight_bytes

    result = class_supervision(record, added)
    assert [box.class_name for box in record.boxes] == ["bucket"]
    assert "shovel" in result.unsupervised
    assert "shovel" not in result.negatives
    assert "bucket" in result.positives
