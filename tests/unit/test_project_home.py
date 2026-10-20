"""Project summary rows. Synthetic sqlite, no pilot video."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from excavation_lab import storage
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.errors import PathNotAllowedError
from excavation_lab.ui.copy import catalog
from excavation_lab.ui.project_home import display_rows

_STORAGE = Path("src/excavation_lab/storage.py")
_HOME = Path("src/excavation_lab/ui/project_home.py")
_SUMMARY_FUNCTIONS = ("list_project_summaries", "_summary_from_database")


def test_project_home_stays_off_the_page_server() -> None:
    source = _HOME.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(
        "starlette" in name or "jinja" in name or "streamlit" in name or "pyarrow" in name
        for name in imported
    )
    assert "mAP" not in source
    assert "loss" not in source
    assert "eta" not in source.casefold()
    calls = _call_names(_STORAGE, _SUMMARY_FUNCTIONS)
    assert "project_sqlite" in calls
    assert "connect" in calls
    assert "sqlite3.connect" not in calls
    body = "\n".join(_function_source(name) for name in _SUMMARY_FUNCTIONS)
    assert "parquet" not in body
    assert "reopen" not in body
    assert "lab.sqlite" not in body


def test_summary_keeps_zero_counts_and_the_stored_timestamp(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    storage.insert_video(workspace, "pilot", "clip", name="clip.mp4")
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        stored = connection.execute(
            "SELECT name, catalog_id, created_at FROM projects"
        ).fetchone()
    assert stored is not None
    rows = storage.list_project_summaries(workspace)
    assert len(rows) == 1
    row = rows[0]
    assert row["project_id"] == "pilot"
    assert row["name"] == stored["name"] == "pilot"
    assert row["catalog_id"] == stored["catalog_id"] == "excavation"
    assert row["created_at"] == stored["created_at"]
    assert isinstance(row["created_at"], str)
    assert row["video_count"] == 1
    assert row["episode_count"] == 0
    assert row["dataset_count"] == 0
    assert row["model_count"] == 0
    listed = storage.list_projects(workspace)
    assert set(listed[0]) == {
        "project_id",
        "name",
        "workspace",
        "project_dir",
        "sqlite_path",
    }
    assert set(storage.ProjectRecord.__annotations__) == set(listed[0])


def test_empty_videos_table_counts_zero(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    row = storage.list_project_summaries(workspace)[0]
    assert row["video_count"] == 0
    assert row["episode_count"] == 0
    assert row["dataset_count"] == 0
    assert row["model_count"] == 0


def test_missing_videos_table_nulls_only_that_count(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    storage.insert_video(workspace, "pilot", "clip", name="clip.mp4")
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("DROP TABLE videos")
    row = storage.list_project_summaries(workspace)[0]
    assert row["video_count"] is None
    assert row["name"] == "pilot"
    assert row["catalog_id"] == "excavation"
    assert isinstance(row["created_at"], str) and row["created_at"]
    assert row["episode_count"] == 0
    assert row["dataset_count"] == 0
    assert row["model_count"] == 0


def test_blank_catalog_and_created_time_are_unknown(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            "UPDATE projects SET catalog_id = '', created_at = '   '"
        )
    row = storage.list_project_summaries(workspace)[0]
    assert row["catalog_id"] is None
    assert row["created_at"] is None
    assert row["name"] == "pilot"
    assert row["video_count"] == 0


def test_created_at_is_not_rewritten(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    stored = "2020-01-02T03:04:05+00:00"
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            "UPDATE projects SET created_at = ?",
            (stored,),
        )
    assert storage.list_project_summaries(workspace)[0]["created_at"] == stored


def test_skip_hidden_invalid_and_sqlite_less_directories(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    projects = workspace / "projects"
    hidden = projects / ".secret"
    hidden.mkdir()
    (hidden / "project.sqlite").write_bytes(b"nope")
    invalid = projects / "1bad"
    invalid.mkdir()
    (invalid / "project.sqlite").write_bytes(b"nope")
    bare = projects / "empty"
    bare.mkdir()
    rows = storage.list_project_summaries(workspace)
    assert [item["project_id"] for item in rows] == ["pilot"]


def test_inside_sqlite_symlink_is_read(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    project = workspace / "projects" / "pilot"
    original = project / "project.sqlite"
    kept = project / "kept.sqlite"
    original.rename(kept)
    original.symlink_to(kept.name)
    row = storage.list_project_summaries(workspace)[0]
    assert row["project_id"] == "pilot"
    assert row["catalog_id"] == "excavation"
    assert row["video_count"] == 0


def test_escaping_sqlite_symlink_is_a_null_row(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    project = workspace / "projects" / "pilot"
    original = project / "project.sqlite"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "project.sqlite"
    original.rename(target)
    original.symlink_to(target)
    with pytest.raises(PathNotAllowedError):
        storage.project_sqlite(workspace, "pilot")
    with pytest.raises(PathNotAllowedError):
        storage.list_projects(workspace)
    rows = storage.list_project_summaries(workspace)
    assert rows == [
        {
            "project_id": "pilot",
            "name": None,
            "catalog_id": None,
            "created_at": None,
            "video_count": None,
            "episode_count": None,
            "dataset_count": None,
            "model_count": None,
        }
    ]


def test_summary_project_id_stays_the_directory_name(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        connection.execute("UPDATE projects SET project_id = 'stored'")
    row = storage.list_project_summaries(workspace)[0]
    assert row["project_id"] == "pilot"
    assert display_rows(workspace, catalog("ru"))[0]["project_id"] == "pilot"


def test_display_rows_use_not_calculated_for_unknown_counts(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    storage.insert_video(workspace, "pilot", "clip", name="clip.mp4")
    russian = catalog("ru")
    english = catalog("en")
    shown = display_rows(workspace, russian)[0]
    assert set(shown) == {
        "project_id",
        "name",
        "catalog_id",
        "created_at",
        "video_count",
        "episode_count",
        "dataset_count",
        "model_count",
    }
    assert shown["project_id"] == "pilot"
    assert shown["name"] == "pilot"
    assert shown["video_count"] == "1"
    assert shown["episode_count"] == "0"
    assert shown["dataset_count"] == "0"
    assert shown["model_count"] == "0"
    assert russian.not_calculated not in shown.values()
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("DROP TABLE videos")
    unknown = display_rows(workspace, english)[0]
    assert unknown["video_count"] == english.not_calculated == "not calculated"
    assert unknown["episode_count"] == "0"
    assert unknown["project_id"] == "pilot"
    assert unknown["catalog_id"] == "excavation"


def _project(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    return workspace


def _call_names(path: Path, functions: tuple[str, ...]) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in functions:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                owner = func.value
                if isinstance(owner, ast.Name):
                    names.add(f"{owner.id}.{func.attr}")
                else:
                    names.add(func.attr)
    return names


def _function_source(name: str) -> str:
    source = _STORAGE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(name)
