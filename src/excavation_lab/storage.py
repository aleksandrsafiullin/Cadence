"""SQLite project metadata and allowed-path checks.

Video bytes stay on disk. This module stores rows only.
``connect`` is the only SQLite opener and turns foreign keys on. No ORM.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import sqlite3
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from excavation_lab.errors import LabError, PathNotAllowedError
from excavation_lab.schemas import SCHEMA_VERSION, JobStatus, repo_root

_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    view_id TEXT,
    site_id TEXT,
    camera_id TEXT,
    equipment_family TEXT,
    model_id TEXT NOT NULL,
    model_origin TEXT NOT NULL,
    analysis_fps REAL NOT NULL,
    source_fps REAL,
    export_fps REAL,
    start_s REAL NOT NULL,
    end_s REAL,
    decode_ms REAL,
    encode_ms REAL,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    CHECK (model_origin IN ('baseline_coco', 'custom')),
    CHECK (analysis_fps > 0),
    CHECK (source_fps IS NULL OR source_fps > 0),
    CHECK (export_fps IS NULL OR export_fps > 0),
    CHECK (start_s >= 0),
    CHECK (end_s IS NULL OR end_s > start_s),
    CHECK (decode_ms IS NULL OR decode_ms >= 0),
    CHECK (encode_ms IS NULL OR encode_ms >= 0)
);
"""

_ACTION_CLASSES_SQL = """
CREATE TABLE IF NOT EXISTS action_classes (
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    id INTEGER NOT NULL,
    name TEXT NOT NULL,
    title TEXT NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY (project_id, id),
    UNIQUE (project_id, name),
    CHECK (id >= 0)
);
"""

_ACTION_LABELS_SQL = """
CREATE TABLE IF NOT EXISTS action_labels (
    label_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    action_name TEXT NOT NULL,
    annotator TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK (end_s > start_s)
);
"""

_ACTION_MODELS_SQL = """
CREATE TABLE IF NOT EXISTS action_models (
    action_model_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    detector_model_id TEXT NOT NULL,
    source_run_ids TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_ACTION_INFERENCES_SQL = """
CREATE TABLE IF NOT EXISTS action_inferences (
    job_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    source_run_id TEXT NOT NULL,
    action_model_id TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

PROJECT_SUBDIRS: tuple[str, ...] = (
    "media",
    "annotation_batches",
    "dataset_versions",
    "models",
    "runs",
    "exports",
)

_PROJECT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_INVALID_NAME = (
    "Недопустимое имя проекта. "
    "Используйте латинские буквы, цифры, «_» или «-» и начните с буквы."
)
_INVALID_INTERVAL = "Интервал должен быть полуоткрытым: end_s больше start_s."
_VIDEO_IDENTITY_COLUMNS = ("site_id", "camera_id", "view_id", "equipment_family")
_RUN_IDENTITY_COLUMNS = ("site_id", "camera_id", "equipment_family")
_IDENTITY_TABLES = frozenset({"videos", "runs"})
_IDENTITY_COLUMN = re.compile(r"^[a-z_]+$")

# Status literals come from JobStatus, not from user input.
_JOB_STATUS_SQL = ", ".join(f"'{item.value}'" for item in JobStatus)

SCHEMA_SQL = f"""
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    workspace_root TEXT NOT NULL,
    extra_roots_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    catalog_id TEXT NOT NULL
);

CREATE TABLE videos (
    video_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    name TEXT,
    sha256 TEXT,
    size_bytes INTEGER,
    duration_s REAL,
    width INTEGER,
    height INTEGER,
    rotation_deg REAL,
    sar TEXT,
    dar TEXT,
    codec TEXT,
    fps_nominal REAL,
    fps_average REAL,
    time_base TEXT,
    start_offset_s REAL,
    source_url TEXT,
    path TEXT,
    site_id TEXT,
    camera_id TEXT,
    view_id TEXT,
    equipment_family TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE episodes (
    episode_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    split_group_id TEXT NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    label TEXT NOT NULL,
    concurrent INTEGER NOT NULL DEFAULT 0,
    CHECK (end_s > start_s),
    CHECK (concurrent IN (0, 1))
);

CREATE TABLE split_manifests (
    manifest_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    video_id TEXT NOT NULL REFERENCES videos(video_id),
    content_hash TEXT,
    path TEXT,
    frozen INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE annotation_batches (
    batch_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    video_id TEXT REFERENCES videos(video_id),
    status TEXT,
    path TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE dataset_versions (
    dataset_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    content_hash TEXT,
    path TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE model_versions (
    model_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    dataset_id TEXT REFERENCES dataset_versions(dataset_id),
    variant TEXT,
    path TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    status TEXT NOT NULL,
    pid INTEGER,
    partial INTEGER NOT NULL DEFAULT 0,
    analyzed_end_s REAL,
    error_code TEXT,
    created_at TEXT NOT NULL,
    CHECK (status IN ({_JOB_STATUS_SQL})),
    CHECK (partial IN (0, 1)),
    CHECK (pid IS NULL OR pid > 0),
    CHECK (analyzed_end_s IS NULL OR analyzed_end_s >= 0),
    CHECK (partial = 0 OR analyzed_end_s IS NOT NULL)
);

CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    video_id TEXT REFERENCES videos(video_id),
    source_frame_index INTEGER,
    review_status TEXT,
    note TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE views (
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    view_id TEXT NOT NULL,
    site_id TEXT,
    camera_id TEXT,
    equipment_family TEXT,
    measured INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (project_id, view_id)
);

{_RUNS_SQL}
{_ACTION_CLASSES_SQL}
{_ACTION_LABELS_SQL}
{_ACTION_MODELS_SQL}
"""


class ProjectRecord(TypedDict):
    project_id: str
    name: str
    workspace: str
    project_dir: str
    sqlite_path: str


class EpisodeRow(TypedDict):
    episode_id: str
    project_id: str
    video_id: str
    split_group_id: str
    start_s: float
    end_s: float
    label: str
    concurrent: bool


class AnalysisRunRecord(TypedDict):
    run_id: str
    job_id: str
    video_id: str
    model_id: str
    model_origin: str
    analysis_fps: float
    start_s: float
    end_s: float | None
    created_at: str
    run_dir: str
    completed: bool


class ActionModelRecord(TypedDict):
    action_model_id: str
    project_id: str
    detector_model_id: str
    source_run_ids: list[str]
    path: str
    created_at: str


class ActionInferenceRecord(TypedDict):
    job_id: str
    project_id: str
    source_run_id: str
    action_model_id: str
    path: str
    created_at: str


class VideoRecord(TypedDict):
    video_id: str
    project_id: str
    name: str | None
    sha256: str | None
    size_bytes: int | None
    duration_s: float | None
    width: int | None
    height: int | None
    rotation_deg: float | None
    sar: str | None
    dar: str | None
    codec: str | None
    fps_nominal: float | None
    fps_average: float | None
    time_base: str | None
    start_offset_s: float | None
    source_url: str | None
    path: str | None
    site_id: str | None
    camera_id: str | None
    view_id: str | None
    equipment_family: str | None
    created_at: str


def default_workspace_root() -> Path:
    """Repo ``workspace/`` directory. Callers pass ``--workspace`` to override."""
    return repo_root() / "workspace"


def validate_project_name(name: str) -> str:
    if _PROJECT_NAME.fullmatch(name) is None:
        raise LabError("invalid_project_name", _INVALID_NAME)
    return name


def require_half_open_interval(start_s: float, end_s: float) -> None:
    """Reject a closed or backwards interval. Stored rows are ``[start_s, end_s)``."""
    if (
        isinstance(start_s, bool)
        or isinstance(end_s, bool)
        or not isinstance(start_s, (int, float))
        or not isinstance(end_s, (int, float))
        or not math.isfinite(start_s)
        or not math.isfinite(end_s)
        or end_s <= start_s
    ):
        raise LabError("invalid_interval", _INVALID_INTERVAL)


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open one SQLite connection with foreign keys on. Commit on success."""
    connection = sqlite3.connect(db_path)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        _ensure_action_tables(connection)
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def resolve_allowed_path(path: Path, roots: Sequence[Path]) -> Path:
    """Return ``path`` resolved, if that location is inside an allowed root.

    ``Path.resolve()`` runs before the root check. A symlink that lives inside
    a root and points outside is rejected, and so is a lexical ``../`` that
    leaves the root. A ``../`` that still resolves inside a root is allowed.
    """
    if not roots:
        raise PathNotAllowedError()
    resolved = Path(path).resolve()
    for root in roots:
        base = Path(root).resolve()
        if resolved.is_relative_to(base):
            return resolved
    raise PathNotAllowedError(str(resolved))


def create_project(
    name: str,
    workspace: Path | None = None,
    *,
    catalog_id: str,
) -> ProjectRecord:
    """Create ``<workspace>/projects/<project_id>/`` and its SQLite file.

    ``catalog_id`` must already be a file under ``workspace/catalogs/``.
    This does not create that file and does not copy it into the project.
    ``project_id`` is the validated name. ``workspace`` defaults to the repo
    ``workspace/`` directory and is never taken from a home-directory path.
    """
    project_id = validate_project_name(name)
    root = _workspace_root(workspace)
    from excavation_lab.datasets.catalog import require_catalog_file

    require_catalog_file(root, catalog_id)
    projects = (root / "projects").resolve()
    if not projects.is_relative_to(root):
        raise PathNotAllowedError(str(projects))
    leaf = projects / project_id
    if leaf.exists() or leaf.is_symlink():
        resolved_existing = leaf.resolve()
        if not resolved_existing.is_relative_to(root):
            raise PathNotAllowedError(str(resolved_existing))
        raise LabError(
            "duplicate_project_name",
            f"Проект «{name}» уже существует.",
        )
    leaf.mkdir(parents=True, exist_ok=False)
    resolved_leaf = leaf.resolve()
    try:
        if not resolved_leaf.is_relative_to(root):
            raise PathNotAllowedError(str(resolved_leaf))
        for sub in PROJECT_SUBDIRS:
            (resolved_leaf / sub).mkdir()
        db_path = resolved_leaf / "project.sqlite"
        _init_database(
            db_path,
            project_id=project_id,
            name=name,
            workspace_root=root,
            catalog_id=catalog_id,
        )
    except Exception:
        _discard_new_project(resolved_leaf, root=root)
        raise
    return ProjectRecord(
        project_id=project_id,
        name=name,
        workspace=str(root),
        project_dir=str(resolved_leaf),
        sqlite_path=str(resolved_leaf / "project.sqlite"),
    )


def delete_project(name: str, workspace: Path | None = None) -> None:
    """Remove ``workspace/projects/<project_id>/`` and nothing above it.

    Does not delete ``workspace/catalogs`` or ``workspace/weights``.
    A path stored in ``model_versions`` is not followed.
    Action tables in ``project.sqlite`` go with the project directory.
    """
    project_id = validate_project_name(name)
    root = _workspace_root(workspace)
    projects_path = root / "projects"
    if projects_path.is_symlink():
        raise PathNotAllowedError(str(projects_path))
    projects = projects_path.resolve()
    if not projects.is_relative_to(root):
        raise PathNotAllowedError(str(projects))
    leaf = projects / project_id
    if leaf.is_symlink():
        raise PathNotAllowedError(str(leaf))
    if not leaf.exists():
        raise LabError(
            "project_not_found",
            f"Проект «{project_id}» уже удалён.",
        )
    if not leaf.is_dir():
        raise PathNotAllowedError(str(leaf))
    resolved = leaf.resolve()
    if (
        resolved == projects
        or resolved == root
        or resolved.parent != projects
        or not resolved.is_relative_to(projects)
    ):
        raise PathNotAllowedError(str(resolved))
    shutil.rmtree(resolved)


def project_directory(workspace: Path, project_id: str) -> Path:
    validate_project_name(project_id)
    root = Path(workspace).resolve()
    projects = (root / "projects").resolve()
    if not projects.is_relative_to(root):
        raise PathNotAllowedError(str(projects))
    directory = projects / project_id
    if directory.exists() or directory.is_symlink():
        directory = directory.resolve()
    if not directory.is_relative_to(root):
        raise PathNotAllowedError(str(directory))
    return directory


def published_weights_directory(workspace: Path, model_id: str) -> Path:
    """``workspace/weights/<model_id>``. Not the official COCO cache."""
    if (
        not isinstance(model_id, str)
        or not model_id
        or model_id in {".", ".."}
        or "/" in model_id
        or "\\" in model_id
        or "\x00" in model_id
    ):
        raise LabError("invalid_model_version", "Недопустимое имя версии модели.")
    root = Path(workspace).resolve()
    weights = (root / "weights").resolve()
    if not weights.is_relative_to(root):
        raise PathNotAllowedError(str(weights))
    destination = (weights / model_id).resolve()
    if destination.parent != weights or not destination.is_relative_to(weights):
        raise PathNotAllowedError(str(destination))
    return destination


def recorded_model_directory(
    workspace: Path,
    project_id: str,
    model_id: str,
) -> Path | None:
    """Directory stored in ``model_versions.path`` for this project and id.

    A project-local ``models/`` directory is not consulted.
    """
    db_path = project_sqlite(workspace, project_id)
    if not db_path.is_file() or db_path.is_symlink():
        return None
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT path FROM model_versions
            WHERE project_id = ? AND model_id = ?
            """,
            (project_id, model_id),
        ).fetchone()
    if row is None or not isinstance(row["path"], str) or not row["path"].strip():
        return None
    return Path(row["path"])


def project_sqlite(workspace: Path, project_id: str) -> Path:
    directory = project_directory(workspace, project_id)
    db_path = directory / "project.sqlite"
    if not db_path.is_file():
        raise LabError(
            "project_not_found",
            f"Проект «{project_id}» не найден.",
        )
    resolved = db_path.resolve()
    root = Path(workspace).resolve()
    if not resolved.is_relative_to(root):
        raise PathNotAllowedError(str(resolved))
    return resolved


def list_projects(workspace: Path | None = None) -> list[ProjectRecord]:
    """Projects under ``<workspace>/projects`` that have a SQLite file."""
    root = _workspace_root(workspace)
    projects_dir = root / "projects"
    if not projects_dir.exists():
        return []
    resolved_dir = projects_dir.resolve()
    if not resolved_dir.is_relative_to(root):
        raise PathNotAllowedError(str(resolved_dir))
    if not resolved_dir.is_dir():
        return []
    records: list[ProjectRecord] = []
    for child in sorted(resolved_dir.iterdir(), key=lambda item: item.name):
        if child.name.startswith("."):
            continue
        try:
            project_id = validate_project_name(child.name)
        except LabError:
            continue
        database = child / "project.sqlite"
        if not database.is_file() and not database.is_symlink():
            continue
        try:
            records.append(_project_record(root, project_id))
        except LabError as exc:
            if exc.code == "path_not_allowed":
                raise
            continue
    return records


def _project_record(workspace: Path, project_id: str) -> ProjectRecord:
    directory = project_directory(workspace, project_id)
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT project_id, name
            FROM projects
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
    if row is None:
        raise LabError(
            "project_not_found",
            f"Проект «{project_id}» не найден.",
        )
    return ProjectRecord(
        project_id=row["project_id"],
        name=row["name"],
        workspace=str(workspace),
        project_dir=str(directory),
        sqlite_path=str(db_path),
    )


def allowed_media_roots(workspace: Path, project_id: str) -> list[Path]:
    """Workspace root plus extra directories stored on the project row."""
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT workspace_root, extra_roots_json
            FROM projects
            WHERE project_id = ?
            """,
            (project_id,),
        ).fetchone()
    if row is None:
        raise LabError(
            "project_not_found",
            f"Проект «{project_id}» не найден.",
        )
    roots = [Path(row["workspace_root"])]
    extra = _parse_extra_roots(row["extra_roots_json"])
    roots.extend(Path(item) for item in extra)
    return roots


def replace_extra_roots(
    workspace: Path,
    project_id: str,
    roots: Sequence[Path],
) -> None:
    """Replace extra media roots. Each one must be an existing directory."""
    stored: list[str] = []
    for root in roots:
        resolved = Path(root).resolve()
        if not resolved.is_dir():
            raise LabError(
                "invalid_roots",
                f"Дополнительный каталог не найден: {root}.",
            )
        text = str(resolved)
        if text not in stored:
            stored.append(text)
    db_path = project_sqlite(workspace, project_id)
    payload = json.dumps(stored)
    with connect(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE projects
            SET extra_roots_json = ?
            WHERE project_id = ?
            """,
            (payload, project_id),
        )
        if cursor.rowcount != 1:
            raise LabError(
                "project_not_found",
                f"Проект «{project_id}» не найден.",
            )


def resolve_project_path(workspace: Path, project_id: str, path: Path) -> Path:
    """Resolve ``path`` against the project's allowed media roots."""
    return resolve_allowed_path(path, allowed_media_roots(workspace, project_id))


def insert_video(
    workspace: Path,
    project_id: str,
    video_id: str,
    *,
    path: Path | None = None,
    name: str | None = None,
    site_id: str | None = None,
    camera_id: str | None = None,
    view_id: str | None = None,
    equipment_family: str | None = None,
) -> None:
    """Insert a video row. Probe columns stay null until media import.

    Camera identity is null when unknown. There is no default quarry.
    """
    project_id = validate_project_name(project_id)
    _require_token(video_id, "invalid_video_id", "Недопустимый идентификатор видео.")
    identity = _camera_identity(
        site_id=site_id,
        camera_id=camera_id,
        view_id=view_id,
        equipment_family=equipment_family,
        code="invalid_video",
    )
    stored_path: str | None = None
    if path is not None:
        stored_path = str(resolve_project_path(workspace, project_id, path))
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        _ensure_text_columns(connection, "videos", _VIDEO_IDENTITY_COLUMNS)
        connection.execute(
            """
            INSERT INTO videos (
                video_id, project_id, name, path,
                site_id, camera_id, view_id, equipment_family,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (video_id, project_id, name, stored_path, *identity, _utc_now()),
        )


def insert_analysis_run(
    workspace: Path,
    project_id: str,
    *,
    run_id: str,
    job_id: str,
    video_id: str,
    view_id: str | None,
    model_id: str,
    model_origin: str,
    analysis_fps: float,
    source_fps: float | None,
    export_fps: float | None,
    start_s: float,
    end_s: float | None,
    decode_ms: float | None,
    encode_ms: float | None,
    site_id: str | None = None,
    camera_id: str | None = None,
    equipment_family: str | None = None,
) -> None:
    """Insert one analysis run.

    ``analysis_fps``, ``source_fps``, and ``export_fps`` are separate columns.
    A null measurement stays null. Camera identity null means unknown.
    """
    project_id = validate_project_name(project_id)
    site, camera, view, family = _camera_identity(
        site_id=site_id,
        camera_id=camera_id,
        view_id=view_id,
        equipment_family=equipment_family,
        code="invalid_run",
    )
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        connection.execute(_RUNS_SQL)
        _ensure_text_columns(connection, "runs", _RUN_IDENTITY_COLUMNS)
        connection.execute(
            """
            INSERT INTO runs (
                run_id, project_id, job_id, video_id, view_id,
                site_id, camera_id, equipment_family,
                model_id, model_origin, analysis_fps, source_fps, export_fps,
                start_s, end_s, decode_ms, encode_ms, schema_version, created_at
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?
            )
            """,
            (
                run_id,
                project_id,
                job_id,
                video_id,
                view,
                site,
                camera,
                family,
                model_id,
                model_origin,
                analysis_fps,
                source_fps,
                export_fps,
                start_s,
                end_s,
                decode_ms,
                encode_ms,
                SCHEMA_VERSION,
                _utc_now(),
            ),
        )


def get_analysis_run(workspace: Path, project_id: str, run_id: str) -> dict[str, Any]:
    """Return one analysis run. Missing measurements stay null."""
    project_id = validate_project_name(project_id)
    _require_token(run_id, "invalid_run", "Недопустимый идентификатор прогона.")
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        connection.execute(_RUNS_SQL)
        _ensure_text_columns(connection, "runs", _RUN_IDENTITY_COLUMNS)
        row = connection.execute(
            """
            SELECT
                run_id, project_id, job_id, video_id, view_id,
                site_id, camera_id, equipment_family,
                model_id, model_origin, analysis_fps, source_fps, export_fps,
                start_s, end_s, decode_ms, encode_ms, schema_version
            FROM runs
            WHERE run_id = ? AND project_id = ?
            """,
            (run_id, project_id),
        ).fetchone()
    if row is None:
        raise LabError("run_not_found", "Прогон не найден в этом проекте.")
    return dict(row)


def list_analysis_runs(
    workspace: Path,
    project_id: str,
    *,
    video_id: str | None = None,
    origin: str | None = None,
) -> list[AnalysisRunRecord]:
    """Runs for one project, oldest ``created_at`` first.

    ``end_s`` stays ``None`` when the row is a full-video run. ``run_dir`` is
    ``jobs.run_directory``. ``completed`` is true only when ``lab.sqlite``
    says that job is ``completed``.
    """
    project_id = validate_project_name(project_id)
    if video_id is not None:
        _require_token(video_id, "invalid_video_id", "Недопустимый идентификатор видео.")
    if origin is not None:
        _require_token(origin, "invalid_run", "Недопустимый источник модели.")
    clauses = ["project_id = ?"]
    params: list[Any] = [project_id]
    if video_id is not None:
        clauses.append("video_id = ?")
        params.append(video_id)
    if origin is not None:
        clauses.append("model_origin = ?")
        params.append(origin)
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        connection.execute(_RUNS_SQL)
        rows = connection.execute(
            f"""
            SELECT
                run_id, job_id, video_id, model_id, model_origin,
                analysis_fps, start_s, end_s, created_at
            FROM runs
            WHERE {" AND ".join(clauses)}
            ORDER BY created_at, run_id
            """,
            params,
        ).fetchall()
    completed = _completed_jobs(workspace, project_id, [row["job_id"] for row in rows])
    from excavation_lab.jobs import run_directory

    listed: list[AnalysisRunRecord] = []
    for row in rows:
        end_s = row["end_s"]
        listed.append(
            AnalysisRunRecord(
                run_id=row["run_id"],
                job_id=row["job_id"],
                video_id=row["video_id"],
                model_id=row["model_id"],
                model_origin=row["model_origin"],
                analysis_fps=float(row["analysis_fps"]),
                start_s=float(row["start_s"]),
                end_s=None if end_s is None else float(end_s),
                created_at=row["created_at"],
                run_dir=str(run_directory(workspace, project_id, row["job_id"])),
                completed=row["job_id"] in completed,
            )
        )
    return listed


def record_analysis_export(
    workspace: Path,
    project_id: str,
    run_id: str,
    *,
    export_fps: float,
    encode_ms: float | None,
) -> None:
    """Store the annotated-export rate. A missing encode time stays null.

    ``export_fps`` is the preview rate, not the analysis rate and not the
    source rate. Zero is not a stand-in for an unmeasured ``encode_ms``.
    """
    project_id = validate_project_name(project_id)
    _require_token(run_id, "invalid_run", "Недопустимый идентификатор прогона.")
    fps = _positive_export_fps(export_fps)
    measured = _positive_optional_ms(encode_ms)
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        connection.execute(_RUNS_SQL)
        cursor = connection.execute(
            """
            UPDATE runs
            SET export_fps = ?, encode_ms = ?
            WHERE run_id = ? AND project_id = ?
            """,
            (fps, measured, run_id, project_id),
        )
        if cursor.rowcount != 1:
            raise LabError("run_not_found", "Прогон не найден в этом проекте.")


def insert_review(
    workspace: Path,
    project_id: str,
    *,
    review_id: str,
    video_id: str | None = None,
    source_frame_index: int | None = None,
    review_status: str | None = None,
    note: str | None = None,
) -> None:
    """Store a review note unchanged. Export quoting does not happen here."""
    project_id = validate_project_name(project_id)
    _require_token(review_id, "invalid_review", "Недопустимый идентификатор проверки.")
    if note is not None and not isinstance(note, str):
        raise LabError("invalid_review", "Заметка должна быть текстом.")
    if video_id is not None:
        _require_token(video_id, "invalid_video_id", "Недопустимый идентификатор видео.")
    if source_frame_index is not None and (
        isinstance(source_frame_index, bool)
        or not isinstance(source_frame_index, int)
        or source_frame_index < 0
    ):
        raise LabError("invalid_review", "Номер кадра должен быть неотрицательным целым.")
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO reviews (
                review_id, project_id, video_id, source_frame_index,
                review_status, note, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review_id,
                project_id,
                video_id,
                source_frame_index,
                review_status,
                note,
                _utc_now(),
            ),
        )


def list_reviews(workspace: Path, project_id: str) -> list[dict[str, Any]]:
    """Review rows for CSV. Null frame indexes and notes stay null."""
    project_id = validate_project_name(project_id)
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT review_id, video_id, source_frame_index, review_status, note
            FROM reviews
            WHERE project_id = ?
            ORDER BY created_at, review_id
            """,
            (project_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def insert_action_model(
    workspace: Path,
    project_id: str,
    *,
    action_model_id: str,
    detector_model_id: str,
    source_run_ids: Sequence[str],
    path: str | Path,
) -> None:
    """Store one action model. ``source_run_ids`` is a canonical JSON array.

    List order is preserved. The path is the bundle directory that will hold
    ``action_bundle.json``. The directory does not have to exist yet.
    """
    project_id = validate_project_name(project_id)
    _require_token(
        action_model_id,
        "invalid_action_model",
        "Недопустимый идентификатор модели действий.",
    )
    _require_token(
        detector_model_id,
        "invalid_action_model",
        "Недопустимый идентификатор детектора.",
    )
    payload = _canonical_source_run_ids(source_run_ids)
    stored_path = _action_model_path(path)
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        try:
            connection.execute(
                """
                INSERT INTO action_models (
                    action_model_id, project_id, detector_model_id,
                    source_run_ids, path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    action_model_id,
                    project_id,
                    detector_model_id,
                    payload,
                    stored_path,
                    _utc_now(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise LabError(
                "duplicate_action_model",
                "Модель действий с таким идентификатором уже есть.",
            ) from exc


def list_action_models(workspace: Path, project_id: str) -> list[ActionModelRecord]:
    """Action models for one project, oldest first. ``source_run_ids`` is a list."""
    project_id = validate_project_name(project_id)
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                action_model_id, project_id, detector_model_id,
                source_run_ids, path, created_at
            FROM action_models
            WHERE project_id = ?
            ORDER BY created_at, action_model_id
            """,
            (project_id,),
        ).fetchall()
    listed: list[ActionModelRecord] = []
    for row in rows:
        listed.append(
            ActionModelRecord(
                action_model_id=row["action_model_id"],
                project_id=row["project_id"],
                detector_model_id=row["detector_model_id"],
                source_run_ids=_parse_source_run_ids(row["source_run_ids"]),
                path=row["path"],
                created_at=row["created_at"],
            )
        )
    return listed


def insert_action_inference(
    workspace: Path,
    project_id: str,
    *,
    job_id: str,
    source_run_id: str,
    action_model_id: str,
    path: str | Path,
) -> None:
    """Store one finished action-inference job. ``path`` is its run directory."""
    project_id = validate_project_name(project_id)
    _require_token(job_id, "invalid_job", "Недопустимый идентификатор задания.")
    _require_token(source_run_id, "invalid_run", "Недопустимый идентификатор прогона.")
    _require_token(
        action_model_id,
        "invalid_action_model",
        "Недопустимый идентификатор модели действий.",
    )
    stored_path = _action_model_path(path)
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        try:
            connection.execute(
                """
                INSERT INTO action_inferences (
                    job_id, project_id, source_run_id, action_model_id, path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    project_id,
                    source_run_id,
                    action_model_id,
                    stored_path,
                    _utc_now(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise LabError(
                "duplicate_action_inference",
                "Инференс действий с таким заданием уже записан.",
            ) from exc


def list_action_inferences(workspace: Path, project_id: str) -> list[ActionInferenceRecord]:
    """Action inferences for one project, oldest first."""
    project_id = validate_project_name(project_id)
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT job_id, project_id, source_run_id, action_model_id, path, created_at
            FROM action_inferences
            WHERE project_id = ?
            ORDER BY created_at, job_id
            """,
            (project_id,),
        ).fetchall()
    return [
        ActionInferenceRecord(
            job_id=row["job_id"],
            project_id=row["project_id"],
            source_run_id=row["source_run_id"],
            action_model_id=row["action_model_id"],
            path=row["path"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


def get_video(workspace: Path, project_id: str, video_id: str) -> VideoRecord:
    """Return one video row. Probe columns may still be null."""
    project_id = validate_project_name(project_id)
    _require_token(video_id, "invalid_video_id", "Недопустимый идентификатор видео.")
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        _ensure_text_columns(connection, "videos", _VIDEO_IDENTITY_COLUMNS)
        row = connection.execute(
            """
            SELECT
                video_id, project_id, name, sha256, size_bytes, duration_s,
                width, height, rotation_deg, sar, dar, codec,
                fps_nominal, fps_average, time_base, start_offset_s,
                source_url, path, site_id, camera_id, view_id, equipment_family,
                created_at
            FROM videos
            WHERE video_id = ? AND project_id = ?
            """,
            (video_id, project_id),
        ).fetchone()
    if row is None:
        raise LabError("video_not_found", "Видео не найдено в этом проекте.")
    return _video_record(row)


def list_videos(workspace: Path, project_id: str) -> list[VideoRecord]:
    """Videos in one project, oldest first. Probe columns may still be null."""
    project_id = validate_project_name(project_id)
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        _ensure_text_columns(connection, "videos", _VIDEO_IDENTITY_COLUMNS)
        rows = connection.execute(
            """
            SELECT
                video_id, project_id, name, sha256, size_bytes, duration_s,
                width, height, rotation_deg, sar, dar, codec,
                fps_nominal, fps_average, time_base, start_offset_s,
                source_url, path, site_id, camera_id, view_id, equipment_family,
                created_at
            FROM videos
            WHERE project_id = ?
            ORDER BY created_at, video_id
            """,
            (project_id,),
        ).fetchall()
    return [_video_record(row) for row in rows]


def insert_probed_video(
    workspace: Path,
    project_id: str,
    metadata: Mapping[str, Any],
) -> None:
    """Insert a probed video. ``path`` is resolved against allowed roots first."""
    project_id = validate_project_name(project_id)
    video_id = _metadata_token(metadata, "video_id")
    stored_path = str(_metadata_path(workspace, project_id, metadata))
    values = _probe_columns(metadata, path=stored_path)
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        _reject_duplicate_video(connection, project_id, video_id, stored_path)
        try:
            connection.execute(
                """
                INSERT INTO videos (
                    video_id, project_id, name, sha256, size_bytes, duration_s,
                    width, height, rotation_deg, sar, dar, codec,
                    fps_nominal, fps_average, time_base, start_offset_s,
                    source_url, path, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?
                )
                """,
                (
                    video_id,
                    project_id,
                    values["name"],
                    values["sha256"],
                    values["size_bytes"],
                    values["duration_s"],
                    values["width"],
                    values["height"],
                    values["rotation_deg"],
                    values["sar"],
                    values["dar"],
                    values["codec"],
                    values["fps_nominal"],
                    values["fps_average"],
                    values["time_base"],
                    values["start_offset_s"],
                    values["source_url"],
                    stored_path,
                    _utc_now(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise LabError(
                "duplicate_video",
                "Видео с таким идентификатором или путём уже зарегистрировано.",
            ) from exc


def update_probed_video(
    workspace: Path,
    project_id: str,
    metadata: Mapping[str, Any],
) -> None:
    """Rewrite probe columns for an existing video. ``created_at`` stays."""
    project_id = validate_project_name(project_id)
    video_id = _metadata_token(metadata, "video_id")
    stored_path = str(_metadata_path(workspace, project_id, metadata))
    values = _probe_columns(metadata, path=stored_path)
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        other = connection.execute(
            """
            SELECT video_id FROM videos
            WHERE project_id = ? AND path = ? AND video_id != ?
            """,
            (project_id, stored_path, video_id),
        ).fetchone()
        if other is not None:
            raise LabError(
                "duplicate_video",
                f"Этот путь уже зарегистрирован как «{other['video_id']}».",
            )
        cursor = connection.execute(
            """
            UPDATE videos SET
                name = ?,
                sha256 = ?,
                size_bytes = ?,
                duration_s = ?,
                width = ?,
                height = ?,
                rotation_deg = ?,
                sar = ?,
                dar = ?,
                codec = ?,
                fps_nominal = ?,
                fps_average = ?,
                time_base = ?,
                start_offset_s = ?,
                source_url = ?,
                path = ?
            WHERE video_id = ? AND project_id = ?
            """,
            (
                values["name"],
                values["sha256"],
                values["size_bytes"],
                values["duration_s"],
                values["width"],
                values["height"],
                values["rotation_deg"],
                values["sar"],
                values["dar"],
                values["codec"],
                values["fps_nominal"],
                values["fps_average"],
                values["time_base"],
                values["start_offset_s"],
                values["source_url"],
                stored_path,
                video_id,
                project_id,
            ),
        )
        if cursor.rowcount != 1:
            raise LabError("video_not_found", "Видео не найдено в этом проекте.")


def insert_episode(
    workspace: Path,
    project_id: str,
    *,
    episode_id: str,
    video_id: str,
    split_group_id: str,
    start_s: float,
    end_s: float,
    label: str,
    concurrent: bool = False,
) -> None:
    """Insert one episode. ``end_s <= start_s`` is rejected before SQL."""
    project_id = validate_project_name(project_id)
    row = _episode_insert_row(
        project_id,
        episode_id=episode_id,
        video_id=video_id,
        split_group_id=split_group_id,
        start_s=start_s,
        end_s=end_s,
        label=label,
        concurrent=concurrent,
    )
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        _require_video_row(connection, project_id, video_id)
        connection.execute(_EPISODE_INSERT, row)


def replace_video_episodes(
    workspace: Path,
    project_id: str,
    video_id: str,
    episodes: Sequence[Mapping[str, Any]],
) -> None:
    """Delete one video's episodes and insert ``episodes`` in one transaction.

    Rows for every other video stay. A failed insert rolls the delete back.
    """
    project_id = validate_project_name(project_id)
    _require_token(video_id, "invalid_video_id", "Недопустимый идентификатор видео.")
    rows = [
        _episode_insert_row(
            project_id,
            episode_id=str(episode["episode_id"]),
            video_id=video_id,
            split_group_id=str(episode["split_group_id"]),
            start_s=episode["start_s"],
            end_s=episode["end_s"],
            label=episode["label"],
            concurrent=episode["concurrent"],
        )
        for episode in episodes
    ]
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        _require_video_row(connection, project_id, video_id)
        try:
            connection.execute(
                """
                DELETE FROM episodes
                WHERE project_id = ? AND video_id = ?
                """,
                (project_id, video_id),
            )
            if rows:
                connection.executemany(_EPISODE_INSERT, rows)
        except sqlite3.IntegrityError as exc:
            raise LabError(
                "duplicate_episode",
                "Идентификатор эпизода повторяется.",
            ) from exc


def other_episode_ids(workspace: Path, project_id: str, video_id: str) -> list[str]:
    """Episode ids stored on every video except ``video_id``.

    ``episode_id`` is the primary key for the whole project. A new id on
    this video must skip these. This video's own ids are not reserved:
    replacing its rows frees them.
    """
    project_id = validate_project_name(project_id)
    _require_token(video_id, "invalid_video_id", "Недопустимый идентификатор видео.")
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT episode_id
            FROM episodes
            WHERE project_id = ? AND video_id != ?
            ORDER BY episode_id
            """,
            (project_id, video_id),
        ).fetchall()
    return [str(row["episode_id"]) for row in rows]


def list_episodes(workspace: Path, project_id: str, video_id: str) -> list[EpisodeRow]:
    """Episodes of one video, ordered by ``start_s`` then ``episode_id``."""
    get_video(workspace, project_id, video_id)
    project_id = validate_project_name(project_id)
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                episode_id, project_id, video_id, split_group_id,
                start_s, end_s, label, concurrent
            FROM episodes
            WHERE project_id = ? AND video_id = ?
            ORDER BY start_s, episode_id
            """,
            (project_id, video_id),
        ).fetchall()
    return [
        EpisodeRow(
            episode_id=row["episode_id"],
            project_id=row["project_id"],
            video_id=row["video_id"],
            split_group_id=row["split_group_id"],
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            label=row["label"],
            concurrent=bool(row["concurrent"]),
        )
        for row in rows
    ]


def get_split_manifest(
    workspace: Path,
    project_id: str,
    video_id: str,
) -> dict[str, Any] | None:
    """Return the split-manifest row for one video, if freeze stored one."""
    project_id = validate_project_name(project_id)
    _require_token(video_id, "invalid_video_id", "Недопустимый идентификатор видео.")
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT
                manifest_id, project_id, video_id, content_hash,
                path, frozen, created_at
            FROM split_manifests
            WHERE project_id = ? AND video_id = ?
            """,
            (project_id, video_id),
        ).fetchone()
    if row is None:
        return None
    return {
        "manifest_id": row["manifest_id"],
        "project_id": row["project_id"],
        "video_id": row["video_id"],
        "content_hash": row["content_hash"],
        "path": row["path"],
        "frozen": row["frozen"],
        "created_at": row["created_at"],
    }


def insert_split_manifest(
    workspace: Path,
    project_id: str,
    *,
    video_id: str,
    content_hash: str,
    path: Path,
) -> str:
    """Insert one frozen split row. The caller refuses a second freeze."""
    project_id = validate_project_name(project_id)
    _require_token(video_id, "invalid_video_id", "Недопустимый идентификатор видео.")
    if not isinstance(content_hash, str) or not content_hash:
        raise LabError("invalid_manifest", "У зафиксированного разбиения нет хеша.")
    manifest_id = uuid.uuid4().hex
    stored_path = str(Path(path).resolve())
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        video = connection.execute(
            "SELECT project_id FROM videos WHERE video_id = ?",
            (video_id,),
        ).fetchone()
        if video is None or video["project_id"] != project_id:
            raise LabError("video_not_found", "Видео не найдено в этом проекте.")
        try:
            connection.execute(
                """
                INSERT INTO split_manifests (
                    manifest_id, project_id, video_id, content_hash,
                    path, frozen, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest_id,
                    project_id,
                    video_id,
                    content_hash,
                    stored_path,
                    1,
                    _utc_now(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise LabError(
                "video_not_found",
                "Видео не найдено в этом проекте.",
            ) from exc
    return manifest_id


def _video_record(row: sqlite3.Row) -> VideoRecord:
    return VideoRecord(
        video_id=row["video_id"],
        project_id=row["project_id"],
        name=row["name"],
        sha256=row["sha256"],
        size_bytes=row["size_bytes"],
        duration_s=row["duration_s"],
        width=row["width"],
        height=row["height"],
        rotation_deg=row["rotation_deg"],
        sar=row["sar"],
        dar=row["dar"],
        codec=row["codec"],
        fps_nominal=row["fps_nominal"],
        fps_average=row["fps_average"],
        time_base=row["time_base"],
        start_offset_s=row["start_offset_s"],
        source_url=row["source_url"],
        path=row["path"],
        site_id=row["site_id"],
        camera_id=row["camera_id"],
        view_id=row["view_id"],
        equipment_family=row["equipment_family"],
        created_at=row["created_at"],
    )


def _metadata_token(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str):
        raise LabError("invalid_video", "Некорректные метаданные видео.")
    _require_token(value, "invalid_video_id", "Недопустимый идентификатор видео.")
    return value


def _metadata_path(workspace: Path, project_id: str, metadata: Mapping[str, Any]) -> Path:
    raw = metadata.get("path")
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise LabError("invalid_video", "Некорректные метаданные видео.")
    return resolve_project_path(workspace, project_id, Path(raw))


def _probe_columns(metadata: Mapping[str, Any], *, path: str) -> dict[str, Any]:
    sha256 = _required_sha256(metadata.get("sha256"))
    return {
        "name": _optional_name(metadata.get("name")),
        "sha256": sha256,
        "size_bytes": _required_positive_int(metadata.get("size_bytes"), "size_bytes"),
        "duration_s": _required_positive_float(metadata.get("duration_s")),
        "width": _required_positive_int(metadata.get("width"), "width"),
        "height": _required_positive_int(metadata.get("height"), "height"),
        "rotation_deg": _optional_float(metadata.get("rotation_deg")),
        "sar": _optional_ratio(metadata.get("sar")),
        "dar": _optional_ratio(metadata.get("dar")),
        "codec": _required_text(metadata.get("codec")),
        "fps_nominal": _optional_positive_float(metadata.get("fps_nominal")),
        "fps_average": _optional_positive_float(metadata.get("fps_average")),
        "time_base": _required_text(metadata.get("time_base")),
        "start_offset_s": _optional_float(metadata.get("start_offset_s")),
        "source_url": _optional_text(metadata.get("source_url")),
        "path": path,
    }


def _reject_duplicate_video(
    connection: sqlite3.Connection,
    project_id: str,
    video_id: str,
    path: str,
) -> None:
    existing_id = connection.execute(
        "SELECT video_id FROM videos WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    if existing_id is not None:
        raise LabError(
            "duplicate_video",
            "Видео с таким идентификатором уже зарегистрировано.",
        )
    existing_path = connection.execute(
        """
        SELECT video_id FROM videos
        WHERE project_id = ? AND path = ?
        """,
        (project_id, path),
    ).fetchone()
    if existing_path is not None:
        raise LabError(
            "duplicate_video",
            f"Этот путь уже зарегистрирован как «{existing_path['video_id']}».",
        )


def _required_sha256(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise LabError("invalid_video", "Некорректные метаданные видео.")
    if any(char not in "0123456789abcdef" for char in value):
        raise LabError("invalid_video", "Некорректные метаданные видео.")
    return value


def _required_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise LabError("invalid_video", "Некорректные метаданные видео.")
    return value


def _optional_name(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise LabError("invalid_video", "Некорректные метаданные видео.")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or "\x00" in value:
        raise LabError("invalid_video", "Некорректные метаданные видео.")
    text = value.strip()
    if not text:
        return None
    return text


def _optional_ratio(value: object) -> str | None:
    return _optional_text(value)


def _required_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LabError("invalid_video", f"Некорректное поле видео: {label}.")
    return value


def _required_positive_float(value: object) -> float:
    number = _finite_number(value)
    if number is None or number <= 0:
        raise LabError("invalid_video", "Некорректная длительность видео.")
    return number


def _optional_positive_float(value: object) -> float | None:
    if value is None:
        return None
    number = _finite_number(value)
    if number is None or number <= 0:
        raise LabError("invalid_video", "Некорректные метаданные видео.")
    return number


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    number = _finite_number(value)
    if number is None:
        raise LabError("invalid_video", "Некорректные метаданные видео.")
    return number


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def _workspace_root(workspace: Path | None) -> Path:
    if workspace is None:
        return default_workspace_root().resolve()
    return Path(workspace).resolve()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _completed_jobs(workspace: Path, project_id: str, job_ids: Sequence[str]) -> set[str]:
    """Job ids whose ``lab.sqlite`` status is ``completed``. Missing rows are not."""
    if not job_ids:
        return set()
    from excavation_lab.jobs import lab_database

    path = lab_database(workspace)
    if path.is_symlink() or not path.is_file():
        return set()
    unique = list(dict.fromkeys(job_ids))
    placeholders = ", ".join("?" * len(unique))
    with connect(path) as connection:
        present = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'lab_jobs'"
        ).fetchone()
        if present is None:
            return set()
        rows = connection.execute(
            f"""
            SELECT job_id FROM lab_jobs
            WHERE project_id = ? AND status = ? AND job_id IN ({placeholders})
            """,
            (project_id, JobStatus.completed.value, *unique),
        ).fetchall()
    return {row["job_id"] for row in rows}


def _positive_export_fps(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError("invalid_export", "Частота ролика должна быть положительной.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise LabError("invalid_export", "Частота ролика должна быть положительной.")
    return number


def _positive_optional_ms(value: float | None) -> float | None:
    """Null when the measurement is missing or not a positive duration."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def _camera_identity(
    *,
    site_id: str | None,
    camera_id: str | None,
    view_id: str | None,
    equipment_family: str | None,
    code: str,
) -> tuple[str | None, str | None, str | None, str | None]:
    """Null means unknown. An empty string is rejected, not replaced with a site name."""
    return (
        _optional_identity(site_id, code),
        _optional_identity(camera_id, code),
        _optional_identity(view_id, code),
        _optional_identity(equipment_family, code),
    )


def _optional_identity(value: object, code: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or "\x00" in value or "/" in value or "\\" in value:
        raise LabError(code, "Идентификатор задаётся текстом или null.")
    text = value.strip()
    if not text or text in {".", ".."}:
        raise LabError(code, "Неизвестное значение задаётся как null, а не пустой строкой.")
    return text


def _canonical_source_run_ids(source_run_ids: Sequence[str]) -> str:
    """JSON array. Order is the caller's order; keys are not sorted."""
    if isinstance(source_run_ids, (str, bytes)) or not isinstance(source_run_ids, Sequence):
        raise LabError(
            "invalid_action_model",
            "source_run_ids должен быть списком идентификаторов прогонов.",
        )
    ids = list(source_run_ids)
    if not ids or any(not isinstance(item, str) or not item or "\x00" in item for item in ids):
        raise LabError(
            "invalid_action_model",
            "source_run_ids должен быть непустым списком идентификаторов прогонов.",
        )
    return json.dumps(ids, separators=(",", ":"))


def _parse_source_run_ids(payload: object) -> list[str]:
    if not isinstance(payload, str):
        raise LabError(
            "invalid_action_model",
            "Список прогонов модели действий повреждён.",
        )
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LabError(
            "invalid_action_model",
            "Список прогонов модели действий повреждён.",
        ) from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise LabError(
            "invalid_action_model",
            "Список прогонов модели действий повреждён.",
        )
    return list(parsed)


def _action_model_path(path: str | Path) -> str:
    if isinstance(path, Path):
        text = str(path)
    elif isinstance(path, str):
        text = path
    else:
        text = ""
    if not text or "\x00" in text:
        raise LabError("invalid_action_model", "Нужен путь к каталогу модели действий.")
    return text


def _ensure_action_tables(connection: sqlite3.Connection) -> None:
    """Create action tables on a project database that predates them.

    ``CREATE TABLE IF NOT EXISTS`` does not rebuild a table that is already
    there, so existing rows stay. A CHECK change, as with ``lab_jobs`` kinds,
    would copy rows into a new table and only then drop the old one. These
    tables are new: a missing name is created and an existing name is left
    untouched. ``lab_jobs`` kinds are not changed here.
    """
    names = {
        str(row["name"])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if "projects" not in names or "videos" not in names:
        return
    for statement in (
        _ACTION_CLASSES_SQL,
        _ACTION_LABELS_SQL,
        _ACTION_MODELS_SQL,
        _ACTION_INFERENCES_SQL,
    ):
        connection.execute(statement)


def _ensure_text_columns(
    connection: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
) -> None:
    if table not in _IDENTITY_TABLES:
        raise LabError("invalid_schema", "Неизвестная таблица проекта.")
    existing = {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name in columns:
        if _IDENTITY_COLUMN.fullmatch(name) is None:
            raise LabError("invalid_schema", "Неизвестная колонка проекта.")
        if name not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} TEXT")


_EPISODE_INSERT = """
INSERT INTO episodes (
    episode_id, project_id, video_id, split_group_id,
    start_s, end_s, label, concurrent
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


def _episode_insert_row(
    project_id: str,
    *,
    episode_id: str,
    video_id: str,
    split_group_id: str,
    start_s: object,
    end_s: object,
    label: object,
    concurrent: object,
) -> tuple[object, ...]:
    if (
        isinstance(start_s, bool)
        or isinstance(end_s, bool)
        or not isinstance(start_s, (int, float))
        or not isinstance(end_s, (int, float))
    ):
        raise LabError(
            "invalid_interval",
            "Интервал должен быть полуоткрытым: end_s больше start_s.",
        )
    require_half_open_interval(float(start_s), float(end_s))
    _require_token(episode_id, "invalid_episode_id", "Недопустимый идентификатор эпизода.")
    _require_token(video_id, "invalid_video_id", "Недопустимый идентификатор видео.")
    _require_token(
        split_group_id,
        "invalid_split_group_id",
        "Недопустимый идентификатор группы разбиения.",
    )
    if concurrent is not True and concurrent is not False:
        raise LabError(
            "invalid_episode",
            "Признак одновременности должен быть логическим.",
        )
    if not isinstance(label, str) or not label.strip():
        raise LabError("invalid_label", "У эпизода должна быть непустая метка.")
    return (
        episode_id,
        project_id,
        video_id,
        split_group_id,
        float(start_s),
        float(end_s),
        label.strip(),
        1 if concurrent else 0,
    )


def _require_video_row(
    connection: sqlite3.Connection,
    project_id: str,
    video_id: str,
) -> None:
    video = connection.execute(
        """
        SELECT project_id FROM videos WHERE video_id = ?
        """,
        (video_id,),
    ).fetchone()
    if video is None or video["project_id"] != project_id:
        raise LabError("video_not_found", "Видео не найдено в этом проекте.")


def _require_token(value: str, code: str, message: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise LabError(code, message)


def _parse_extra_roots(payload: str) -> list[str]:
    try:
        extra = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LabError(
            "invalid_roots",
            "Список дополнительных каталогов проекта повреждён.",
        ) from exc
    if not isinstance(extra, list) or any(not isinstance(item, str) for item in extra):
        raise LabError(
            "invalid_roots",
            "Список дополнительных каталогов проекта повреждён.",
        )
    return extra


def _init_database(
    db_path: Path,
    *,
    project_id: str,
    name: str,
    workspace_root: Path,
    catalog_id: str,
) -> None:
    with connect(db_path) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.execute(
            """
            INSERT INTO projects (
                project_id, name, workspace_root, extra_roots_json,
                schema_version, created_at, catalog_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                name,
                str(workspace_root),
                "[]",
                SCHEMA_VERSION,
                _utc_now(),
                catalog_id,
            ),
        )


def _discard_new_project(path: Path, *, root: Path) -> None:
    target = path.resolve()
    if target == root or root.is_relative_to(target):
        return
    if target.is_symlink():
        target.unlink()
        return
    if target.is_dir():
        shutil.rmtree(target)
