"""Project SQLite layout, allowed paths, and ``project create``."""

from __future__ import annotations

import inspect
import json
import math
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from excavation_lab import cli, schemas, storage
from excavation_lab.cli import app
from excavation_lab.datasets.catalog import list_catalogs, require_project_catalog
from excavation_lab.errors import LabError, PathNotAllowedError
from excavation_lab.schemas import JobStatus

_RUNNER = CliRunner()

_TABLES = {
    "projects",
    "videos",
    "episodes",
    "split_manifests",
    "annotation_batches",
    "dataset_versions",
    "model_versions",
    "jobs",
    "reviews",
    "views",
    "runs",
    "action_classes",
    "action_labels",
    "action_models",
    "action_inferences",
}

_VIDEO_COLUMNS = {
    "sha256",
    "size_bytes",
    "duration_s",
    "width",
    "height",
    "rotation_deg",
    "sar",
    "dar",
    "codec",
    "fps_nominal",
    "fps_average",
    "time_base",
    "start_offset_s",
    "source_url",
    "path",
}

_JOB_COLUMNS = {"status", "pid", "partial", "analyzed_end_s", "error_code"}

_SUBDIRS = (
    "media",
    "annotation_batches",
    "dataset_versions",
    "models",
    "runs",
    "exports",
)


def _cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return {row["name"] for row in rows}


def test_path_not_allowed_is_a_lab_error() -> None:
    error = PathNotAllowedError()
    assert isinstance(error, LabError)
    assert error.code == "path_not_allowed"
    assert error.code.isascii()
    assert _cyrillic(error.message)


def test_create_project_layout(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    record = storage.create_project("pilot", workspace, catalog_id="excavation")
    root = workspace.resolve()
    project_dir = root / "projects" / "pilot"
    sqlite_path = project_dir / "project.sqlite"

    assert record["project_id"] == "pilot"
    assert record["name"] == "pilot"
    assert record["workspace"] == str(root)
    assert record["project_dir"] == str(project_dir)
    assert record["sqlite_path"] == str(sqlite_path)
    assert sqlite_path.is_file()
    assert storage.PROJECT_SUBDIRS == _SUBDIRS
    for name in _SUBDIRS:
        assert (project_dir / name).is_dir()

    with storage.connect(sqlite_path) as connection:
        pragma = connection.execute("PRAGMA foreign_keys").fetchone()
        assert pragma is not None
        assert pragma[0] == 1
        row = connection.execute(
            "SELECT project_id, name, schema_version, extra_roots_json, catalog_id "
            "FROM projects"
        ).fetchone()
        assert row is not None
        assert row["project_id"] == "pilot"
        assert row["name"] == "pilot"
        assert row["schema_version"] == schemas.SCHEMA_VERSION
        assert row["extra_roots_json"] == "[]"
        assert row["catalog_id"] == "excavation"
    stored_names = {path.name for path in project_dir.rglob("*") if path.is_file()}
    assert "catalog.yaml" not in stored_names
    assert "labels.yaml" not in stored_names


def test_schema_tables_video_and_job_columns(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    record = storage.create_project("pilot", workspace, catalog_id="excavation")
    with storage.connect(Path(record["sqlite_path"])) as connection:
        tables = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
        assert {row["name"] for row in tables} == _TABLES
        assert _VIDEO_COLUMNS <= _columns(connection, "videos")
        assert _JOB_COLUMNS <= _columns(connection, "jobs")
        episode_columns = _columns(connection, "episodes")
        assert {"start_s", "end_s"} <= episode_columns

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO episodes (
                    episode_id, project_id, video_id, split_group_id,
                    start_s, end_s, label
                ) VALUES ('ep-missing', 'pilot', 'missing-video', 'group-1', 0, 1, 'load')
                """
            )


def test_job_status_and_partial_constraints(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    record = storage.create_project("pilot", workspace, catalog_id="excavation")
    with storage.connect(Path(record["sqlite_path"])) as connection:
        for status in JobStatus:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, project_id, status, pid, partial, analyzed_end_s, error_code,
                    created_at
                ) VALUES (?, 'pilot', ?, NULL, 0, NULL, NULL, '2026-01-01T00:00:00+00:00')
                """,
                (f"job-{status.value}", status.value),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, project_id, status, partial, created_at
                ) VALUES ('job-bad', 'pilot', 'nope', 0, '2026-01-01T00:00:00+00:00')
                """
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, project_id, status, partial, analyzed_end_s, created_at
                ) VALUES (
                    'job-partial', 'pilot', 'cancelled', 1, NULL,
                    '2026-01-01T00:00:00+00:00'
                )
                """
            )


def test_symlink_escape_is_rejected_after_resolve(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    list_catalogs(workspace)
    record = storage.create_project("pilot", workspace, catalog_id="excavation")
    link = Path(record["project_dir"]) / "media" / "linked"
    link.symlink_to(outside, target_is_directory=True)
    escaped = link / "clip.mp4"

    assert ".." not in escaped.parts
    assert escaped.is_relative_to(Path(record["workspace"]))
    assert not escaped.resolve().is_relative_to(Path(record["workspace"]))
    assert escaped.resolve().is_relative_to(outside.resolve())

    with pytest.raises(PathNotAllowedError) as caught:
        storage.resolve_project_path(workspace, "pilot", escaped)
    assert caught.value.code == "path_not_allowed"
    assert _cyrillic(caught.value.message)

    with pytest.raises(PathNotAllowedError):
        storage.insert_video(workspace, "pilot", "vid-escape", path=escaped)
    with storage.connect(Path(record["sqlite_path"])) as connection:
        count = connection.execute("SELECT COUNT(*) AS n FROM videos").fetchone()
        assert count is not None
        assert count["n"] == 0


def test_lexical_parent_path_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    lexical = workspace / ".." / "outside"
    assert ".." in lexical.parts
    assert "outside" in lexical.parts
    with pytest.raises(PathNotAllowedError) as caught:
        storage.resolve_project_path(workspace, "pilot", lexical)
    assert caught.value.code == "path_not_allowed"
    assert not lexical.resolve().is_relative_to(workspace.resolve())


def test_dotdot_that_stays_inside_is_allowed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    record = storage.create_project("pilot", workspace, catalog_id="excavation")
    media = Path(record["project_dir"]) / "media"
    folded = media / ".." / "media"
    assert ".." in folded.parts
    resolved = storage.resolve_project_path(workspace, "pilot", folded)
    assert resolved == media.resolve()


def test_extra_root_outside_workspace_is_allowed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    library = tmp_path / "library"
    library.mkdir()
    clip = library / "pilot.mp4"
    clip.write_bytes(b"synthetic")
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(PathNotAllowedError):
        storage.resolve_project_path(workspace, "pilot", other / "nope.mp4")
    storage.replace_extra_roots(workspace, "pilot", [library])
    assert storage.resolve_project_path(workspace, "pilot", clip) == clip.resolve()


def test_half_open_interval_is_enforced(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    record = storage.create_project("pilot", workspace, catalog_id="excavation")
    storage.insert_video(workspace, "pilot", "vid-1", name="pilot")
    kwargs = {
        "episode_id": "ep-1",
        "video_id": "vid-1",
        "split_group_id": "group-1",
        "label": "погрузка",
    }
    with pytest.raises(LabError) as equal:
        storage.insert_episode(workspace, "pilot", start_s=5.0, end_s=5.0, **kwargs)
    assert equal.value.code == "invalid_interval"
    assert _cyrillic(equal.value.message)
    with pytest.raises(LabError) as backwards:
        storage.insert_episode(workspace, "pilot", start_s=5.0, end_s=4.0, **kwargs)
    assert backwards.value.code == "invalid_interval"
    with pytest.raises(LabError) as nan:
        storage.insert_episode(
            workspace,
            "pilot",
            start_s=0.0,
            end_s=math.nan,
            **kwargs,
        )
    assert nan.value.code == "invalid_interval"

    storage.insert_episode(workspace, "pilot", start_s=5.0, end_s=8.0, **kwargs)
    with storage.connect(Path(record["sqlite_path"])) as connection:
        row = connection.execute(
            "SELECT start_s, end_s, label FROM episodes WHERE episode_id = 'ep-1'"
        ).fetchone()
        assert row is not None
        assert row["start_s"] == 5.0
        assert row["end_s"] == 8.0
        assert row["label"] == "погрузка"
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO episodes (
                    episode_id, project_id, video_id, split_group_id,
                    start_s, end_s, label
                ) VALUES ('ep-closed', 'pilot', 'vid-1', 'group-1', 1, 1, 'closed')
                """
            )


def test_duplicate_and_invalid_names(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    record = storage.create_project("pilot", workspace, catalog_id="excavation")
    sqlite_path = Path(record["sqlite_path"])
    with pytest.raises(LabError) as duplicate:
        list_catalogs(workspace)
        storage.create_project("pilot", workspace, catalog_id="excavation")
    assert duplicate.value.code == "duplicate_project_name"
    assert duplicate.value.code.isascii()
    assert _cyrillic(duplicate.value.message)
    assert sqlite_path.is_file()

    for name in ("", " ", "../outside", "pilot/x", "..", ".", "pilot name", "a" * 65):
        with pytest.raises(LabError) as invalid:
            list_catalogs(workspace)
            storage.create_project(name, workspace, catalog_id="excavation")
        assert invalid.value.code == "invalid_project_name"
        assert _cyrillic(invalid.value.message)
    assert sqlite_path.is_file()
    assert list((workspace / "projects").iterdir()) == [workspace / "projects" / "pilot"]


@pytest.mark.parametrize(
    "name",
    ["", "../outside", "pilot/evil", ".."],
)
def test_cli_rejects_invalid_name_as_json(tmp_path: Path, name: str) -> None:
    result = _RUNNER.invoke(
        app,
        [
            "project",
            "create",
            "--name",
            name,
            "--catalog",
            "excavation",
            "--workspace",
            str(tmp_path / "workspace"),
            "--json",
        ],
    )
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_project_name"
    assert payload["error"]["code"].isascii()
    assert _cyrillic(payload["error"]["message"])
    assert not (tmp_path / "workspace" / "projects").exists()


def test_cli_json_create_and_duplicate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    created = _RUNNER.invoke(
        app,
        [
            "project",
            "create",
            "--name",
            "pilot",
            "--catalog",
            "excavation",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert created.exit_code == 0
    payload = json.loads(created.stdout)
    assert list(payload) == ["ok", "data"]
    assert payload["ok"] is True
    data = payload["data"]
    assert data["project_id"] == "pilot"
    assert data["name"] == "pilot"
    sqlite_path = Path(data["sqlite_path"])
    assert sqlite_path == workspace.resolve() / "projects" / "pilot" / "project.sqlite"
    assert sqlite_path.is_file()
    for name in _SUBDIRS:
        assert (sqlite_path.parent / name).is_dir()

    duplicate = _RUNNER.invoke(
        app,
        [
            "project",
            "create",
            "--name",
            "pilot",
            "--catalog",
            "excavation",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert duplicate.exit_code == 1
    error_payload = json.loads(duplicate.stdout)
    assert error_payload["ok"] is False
    assert error_payload["error"]["code"] == "duplicate_project_name"
    assert _cyrillic(error_payload["error"]["message"])
    assert sqlite_path.is_file()


def test_cli_text_duplicate_exits_nonzero(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    args = [
        "project",
        "create",
        "--name",
        "pilot",
        "--catalog",
        "excavation",
        "--workspace",
        str(workspace),
    ]
    created = _RUNNER.invoke(app, args)
    assert created.exit_code == 0
    assert "pilot" in created.stdout
    duplicate = _RUNNER.invoke(app, args)
    assert duplicate.exit_code == 1
    assert "duplicate_project_name" in duplicate.stderr
    assert _cyrillic(duplicate.stderr)


def test_cli_registers_project_doctor_train_video_and_dataset() -> None:
    assert [group.name for group in app.registered_groups] == [
        "project",
        "video",
        "dataset",
        "actions",
        "events",
    ]
    assert [command.name for command in app.registered_commands] == [
        "doctor",
        "ui",
        "train",
        "evaluate",
        "analyze",
        "export",
    ]
    project = app.registered_groups[0].typer_instance
    assert project is not None
    assert [command.name for command in project.registered_commands] == ["create"]
    video = app.registered_groups[1].typer_instance
    assert video is not None
    assert [command.name for command in video.registered_commands] == ["add", "probe"]
    dataset = app.registered_groups[2].typer_instance
    assert dataset is not None
    assert [command.name for command in dataset.registered_commands] == [
        "split-propose",
        "split-freeze",
        "sample",
        "export-cvat",
        "import-coco",
        "validate",
        "snapshot",
    ]
    actions = app.registered_groups[3].typer_instance
    assert actions is not None
    assert [group.name for group in actions.registered_groups] == ["catalog", "label"]
    catalog = actions.registered_groups[0].typer_instance
    assert catalog is not None
    assert [command.name for command in catalog.registered_commands] == ["list", "add", "seed"]
    label = actions.registered_groups[1].typer_instance
    assert label is not None
    assert [command.name for command in label.registered_commands] == ["add", "list", "delete"]
    events = app.registered_groups[4].typer_instance
    assert events is not None
    assert [command.name for command in events.registered_commands] == [
        "propose",
        "export",
    ]


def test_create_project_refuses_a_missing_catalog(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    with pytest.raises(LabError) as caught:
        storage.create_project("pilot", workspace, catalog_id="excavation")
    assert caught.value.code == "catalog_missing"
    assert not (workspace / "projects").exists()
    assert not (workspace / "catalogs").exists()


def test_two_projects_share_one_catalog_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    first = storage.create_project("one", workspace, catalog_id="excavation")
    second = storage.create_project("two", workspace, catalog_id="excavation")
    assert list((workspace / "catalogs").rglob("catalog.yaml")) == [
        workspace / "catalogs" / "excavation" / "catalog.yaml"
    ]
    for record in (first, second):
        project_dir = Path(record["project_dir"])
        assert list(project_dir.rglob("catalog.yaml")) == []
        assert list(project_dir.rglob("labels.yaml")) == []
        with storage.connect(Path(record["sqlite_path"])) as connection:
            row = connection.execute("SELECT catalog_id FROM projects").fetchone()
        assert row is not None
        assert row["catalog_id"] == "excavation"


def test_require_project_catalog_rejects_null_and_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    record = storage.create_project("pilot", workspace, catalog_id="excavation")
    seed = workspace / "catalogs" / "excavation" / "catalog.yaml"

    def boom() -> None:
        raise AssertionError("load_labels")

    monkeypatch.setattr("excavation_lab.datasets.catalog.load_labels", boom)
    monkeypatch.setattr("excavation_lab.schemas.load_labels", boom)
    assert "load_labels" not in inspect.getsource(require_project_catalog)
    loaded = require_project_catalog(workspace, "pilot")
    assert loaded.classes[0].name == "excavator_body"

    seed.unlink()
    with pytest.raises(LabError) as missing_file:
        require_project_catalog(workspace, "pilot")
    assert missing_file.value.code == "catalog_missing"
    assert not seed.exists()

    with storage.connect(Path(record["sqlite_path"])) as connection:
        connection.execute("UPDATE projects SET catalog_id = ?", ("missing",))
    with pytest.raises(LabError) as unknown:
        require_project_catalog(workspace, "pilot")
    assert unknown.value.code == "catalog_missing"
    assert not (workspace / "catalogs" / "missing").exists()

    null_dir = workspace / "projects" / "blank"
    null_dir.mkdir()
    with storage.connect(null_dir / "project.sqlite") as connection:
        connection.execute("CREATE TABLE projects (project_id TEXT, catalog_id TEXT)")
        connection.execute(
            "INSERT INTO projects (project_id, catalog_id) VALUES ('blank', NULL)"
        )
    with pytest.raises(LabError) as null_case:
        require_project_catalog(workspace, "blank")
    assert null_case.value.code == "catalog_missing"


def test_cli_create_requires_a_catalog(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    missing = _RUNNER.invoke(
        app,
        ["project", "create", "--name", "demo", "--workspace", str(workspace)],
    )
    assert missing.exit_code != 0
    list_catalogs(workspace)
    created = _RUNNER.invoke(
        app,
        [
            "project",
            "create",
            "--name",
            "demo",
            "--catalog",
            "excavation",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert created.exit_code == 0
    unknown = _RUNNER.invoke(
        app,
        [
            "project",
            "create",
            "--name",
            "other",
            "--catalog",
            "nope",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert unknown.exit_code == 1
    payload = json.loads(unknown.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "catalog_missing"


def test_default_workspace_is_repo_local() -> None:
    assert storage.default_workspace_root() == schemas.repo_root() / "workspace"
    source = Path(storage.__file__).read_text(encoding="utf-8")
    cli_source = Path(cli.__file__).read_text(encoding="utf-8")
    for text in (source, cli_source):
        assert "Path.home" not in text
        assert "expanduser" not in text
