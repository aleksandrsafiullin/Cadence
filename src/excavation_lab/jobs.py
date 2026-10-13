"""One train or analyze job per workspace.

The lock is ``<workspace>/lab.sqlite``, shared by every project under that
root. A second workspace has its own file. Progress is published with
``os.replace`` so a crashed writer never leaves a partial JSON object.
The parent does not load a model. The child is a new process:

``[sys.executable, "-m", "excavation_lab.jobs", "run", job_id]``

The ``rfdetr`` and ``analyze`` workers load the model in that child.
The ``export`` worker does not. The parent does not.
"""

from __future__ import annotations

import json
import math
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from excavation_lab.errors import JobAlreadyRunningError, LabError
from excavation_lab.schemas import JobRecord, JobStatus
from excavation_lab.storage import (
    connect,
    project_directory,
    project_sqlite,
    validate_project_name,
)

PROGRESS_FILENAME = "progress.json"
DEFAULT_KILL_AFTER_S = 0.75
_WORKSPACE_ENV = "EXCAVATION_LAB_WORKSPACE"
_KINDS = frozenset({"train", "analyze", "export"})
_SLEEP_WORKERS = frozenset({"sleep", "sleep_ignore_term"})
_ACTIVE = frozenset(
    {
        JobStatus.queued.value,
        JobStatus.running.value,
        JobStatus.cancel_requested.value,
    }
)
_TERMINAL = frozenset(
    {
        JobStatus.completed.value,
        JobStatus.failed.value,
        JobStatus.cancelled.value,
    }
)
JobKind = Literal["train", "analyze", "export"]
SleepWorker = Literal["sleep", "sleep_ignore_term"]
WorkerName = Literal["sleep", "sleep_ignore_term", "rfdetr", "analyze", "export"]
_TRAIN_WORKER = "rfdetr"
_ANALYZE_WORKER = "analyze"
_EXPORT_WORKER = "export"
_WORKERS = _SLEEP_WORKERS | {_TRAIN_WORKER, _ANALYZE_WORKER, _EXPORT_WORKER}

_STATUS_SQL = ", ".join(f"'{item.value}'" for item in JobStatus)
_KIND_SQL = ", ".join(f"'{item}'" for item in sorted(_KINDS))
_ACTIVE_SQL = ", ".join(
    f"'{item}'"
    for item in (
        JobStatus.queued.value,
        JobStatus.running.value,
        JobStatus.cancel_requested.value,
    )
)

_LAB_JOBS_SQL = f"""
CREATE TABLE IF NOT EXISTS lab_jobs (
    job_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    worker TEXT NOT NULL,
    status TEXT NOT NULL,
    pid INTEGER,
    partial INTEGER NOT NULL DEFAULT 0,
    analyzed_end_s REAL,
    error_code TEXT,
    orphaned INTEGER NOT NULL DEFAULT 0,
    run_dir TEXT NOT NULL,
    video_duration_s REAL NOT NULL,
    step_s REAL NOT NULL,
    sleep_s REAL NOT NULL,
    created_at TEXT NOT NULL,
    CHECK (status IN ({_STATUS_SQL})),
    CHECK (kind IN ({_KIND_SQL})),
    CHECK (partial IN (0, 1)),
    CHECK (orphaned IN (0, 1)),
    CHECK (pid IS NULL OR pid > 0),
    CHECK (analyzed_end_s IS NULL OR analyzed_end_s >= 0),
    CHECK (partial = 0 OR analyzed_end_s IS NOT NULL),
    CHECK (orphaned = 0 OR status = 'failed')
)
"""

_LOCK_SQL = f"""
CREATE TABLE IF NOT EXISTS workspace_lock (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    job_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    status TEXT NOT NULL,
    CHECK (status IN ({_ACTIVE_SQL}))
)
"""


@dataclass(frozen=True, slots=True)
class _StoredJob:
    job_id: str
    project_id: str
    kind: str
    worker: str
    status: str
    pid: int | None
    partial: bool
    analyzed_end_s: float | None
    error_code: str | None
    orphaned: bool
    run_dir: str
    video_duration_s: float
    step_s: float
    sleep_s: float
    created_at: str


def lab_database(workspace: Path) -> Path:
    """Workspace-level job database. Not a project's ``project.sqlite``."""
    return Path(workspace).resolve() / "lab.sqlite"


def run_directory(workspace: Path, project_id: str, job_id: str) -> Path:
    project_id = validate_project_name(project_id)
    _require_job_id(job_id)
    return project_directory(Path(workspace).resolve(), project_id) / "runs" / job_id


def write_progress(run_dir: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically publish ``progress.json`` in ``run_dir``.

    The destination is replaced only after a temp file in the same directory
    holds one complete JSON object. A kill before ``os.replace`` leaves the
    previous file untouched.
    """
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if not isinstance(payload, Mapping):
        raise LabError("invalid_progress", "Прогресс должен быть одним JSON-объектом.")
    try:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        parsed = json.loads(body)
    except (TypeError, ValueError) as exc:
        raise LabError(
            "invalid_progress",
            "Прогресс должен быть одним JSON-объектом.",
        ) from exc
    if not isinstance(parsed, dict):
        raise LabError("invalid_progress", "Прогресс должен быть одним JSON-объектом.")
    target = directory / PROGRESS_FILENAME
    fd, tmp_name = tempfile.mkstemp(prefix=".progress-", suffix=".tmp", dir=directory)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return target


def start_job(
    workspace: Path,
    project_id: str,
    *,
    kind: JobKind,
    worker: WorkerName = "sleep",
    video_duration_s: float = 100.0,
    step_s: float = 2.5,
    sleep_s: float = 0.05,
    on_run_dir: Callable[[Path], None] | None = None,
) -> JobRecord:
    """Insert a job and take the workspace lock, then spawn the child.

    A queued, running, or cancel-requested job in any project of this
    workspace raises ``JobAlreadyRunningError``. ``on_run_dir`` runs after
    the run directory exists and before the child starts.
    """
    root = _root(workspace)
    project_id = validate_project_name(project_id)
    project_sqlite(root, project_id)
    _require_kind(kind)
    _require_worker(worker)
    duration = _positive_seconds(video_duration_s, "Некорректная длительность видео.")
    step = _positive_seconds(step_s, "Некорректный шаг времени кадра.")
    pause = _positive_seconds(sleep_s, "Некорректная пауза задания.")
    if step >= duration:
        raise LabError(
            "invalid_job",
            "Шаг времени кадра должен быть меньше длительности видео.",
        )
    _reconcile(root)
    job_id = uuid.uuid4().hex
    run_dir = run_directory(root, project_id, job_id)
    job = _StoredJob(
        job_id=job_id,
        project_id=project_id,
        kind=kind,
        worker=worker,
        status=JobStatus.queued.value,
        pid=None,
        partial=False,
        analyzed_end_s=None,
        error_code=None,
        orphaned=False,
        run_dir=str(run_dir),
        video_duration_s=duration,
        step_s=step,
        sleep_s=pause,
        created_at=_utc_now(),
    )
    _acquire(root, job)
    process: subprocess.Popen[bytes] | None = None
    try:
        run_dir.mkdir(exist_ok=False)
        if on_run_dir is not None:
            on_run_dir(run_dir)
        _sync_project(root, job)
        process = _spawn(root, job.job_id, run_dir)
        job = _save(
            root,
            replace(job, status=JobStatus.running.value, pid=process.pid),
        )
    except Exception as exc:
        _abandon_start(root, job, process)
        if isinstance(exc, LabError):
            raise
        raise LabError(
            "spawn_failed",
            "Не удалось запустить фоновое задание.",
        ) from exc
    return _to_record(job)


def request_cancel(
    workspace: Path,
    project_id: str,
    job_id: str,
    *,
    kill_after_s: float = DEFAULT_KILL_AFTER_S,
) -> JobRecord:
    """Mark ``cancel_requested``, signal that job's PID, then ``cancelled``.

    SIGTERM is sent only when the process command line is this job's worker.
    SIGKILL follows after ``kill_after_s`` if that same process is still alive.
    Frames already written set ``partial`` and ``analyzed_end_s`` from the last
    stored timestamp. A cancelled partial run is not ``completed``.
    """
    root = _root(workspace)
    project_id = validate_project_name(project_id)
    _require_job_id(job_id)
    timeout = _positive_seconds(kill_after_s, "Некорректный таймаут отмены.")
    job = _require_job(root, project_id, job_id)
    if job.status in _TERMINAL:
        return _to_record(job)
    job = _save(root, replace(job, status=JobStatus.cancel_requested.value))
    try:
        _signal_worker(job.pid, job.job_id, timeout)
    except OSError:
        pass
    partial, end_s = _outcome_from_progress(Path(job.run_dir))
    job = _save(
        root,
        replace(
            job,
            status=JobStatus.cancelled.value,
            partial=partial,
            analyzed_end_s=end_s,
            error_code=None,
            orphaned=False,
        ),
    )
    return _to_record(job)


def mark_completed(
    workspace: Path,
    project_id: str,
    job_id: str,
    artifact_paths: Sequence[Path] | None = None,
) -> JobRecord:
    """Set ``completed`` only when every required artifact path exists.

    Omitted or missing paths fail. A cancelled job stays ``cancelled``.
    """
    root = _root(workspace)
    project_id = validate_project_name(project_id)
    _require_job_id(job_id)
    job = _require_job(root, project_id, job_id)
    if job.status == JobStatus.cancelled.value:
        raise LabError(
            "job_not_completable",
            "Отменённое задание нельзя пометить как completed. "
            "Частичный прогон остаётся со статусом cancelled.",
        )
    if job.status not in {JobStatus.queued.value, JobStatus.running.value, JobStatus.completed.value}:
        raise LabError(
            "job_not_completable",
            "Это задание нельзя пометить как completed.",
        )
    if job.kind == "analyze":
        from excavation_lab.tracking.export_tables import (
            DETECTIONS_NAME,
            FRAME_STATUS_NAME,
            TRACK_STATES_NAME,
        )

        produced = [
            Path(job.run_dir) / FRAME_STATUS_NAME,
            Path(job.run_dir) / DETECTIONS_NAME,
            Path(job.run_dir) / TRACK_STATES_NAME,
        ]
        passed = list(artifact_paths) if artifact_paths is not None else []
        _require_artifacts([*produced, *passed])
    else:
        _require_artifacts(artifact_paths)
    if job.status == JobStatus.completed.value:
        return _to_record(job)
    if job.pid is not None and job.pid != os.getpid():
        try:
            _signal_worker(job.pid, job.job_id, DEFAULT_KILL_AFTER_S)
        except OSError:
            pass
    job = _save(
        root,
        replace(
            job,
            status=JobStatus.completed.value,
            partial=False,
            error_code=None,
            orphaned=False,
        ),
    )
    return _to_record(job)


def mark_failed(
    workspace: Path,
    project_id: str,
    job_id: str,
    error_code: str,
) -> JobRecord:
    """Set ``failed`` and store ``error_code`` on the job row.

    A cancelled or cancel-requested job is left as it is. The lock is
    released only when this call is the one that records ``failed``.
    """
    root = _root(workspace)
    project_id = validate_project_name(project_id)
    _require_job_id(job_id)
    if not isinstance(error_code, str) or not error_code.strip():
        raise LabError("invalid_job", "Код ошибки задания пуст.")
    job = _require_job(root, project_id, job_id)
    if job.status in _TERMINAL or job.status == JobStatus.cancel_requested.value:
        return _to_record(job)
    return _to_record(
        _save(
            root,
            replace(
                job,
                status=JobStatus.failed.value,
                partial=False,
                analyzed_end_s=None,
                error_code=error_code,
                orphaned=False,
            ),
        )
    )


def worker_job(workspace: Path, job_id: str) -> tuple[str, Path, JobRecord]:
    """Project id, run directory, and record for the worker child."""
    root = _root(workspace)
    _require_job_id(job_id)
    job = _load_job(root, job_id)
    if job is None:
        raise LabError("job_not_found", "Задание не найдено в этом проекте.")
    return job.project_id, Path(job.run_dir), _to_record(job)


def reopen(workspace: Path) -> list[JobRecord]:
    """Re-read jobs. A dead PID left ``running`` becomes failed and orphaned.

    Does not start a replacement process.
    """
    root = _root(workspace)
    _reconcile(root)
    return [_to_record(job) for job in _load_all(root)]


def get_job(workspace: Path, project_id: str, job_id: str) -> JobRecord:
    root = _root(workspace)
    project_id = validate_project_name(project_id)
    _require_job_id(job_id)
    return _to_record(_require_job(root, project_id, job_id))


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "run":
        print("usage: python -m excavation_lab.jobs run JOB_ID", file=sys.stderr)
        return 2
    try:
        _require_job_id(args[1])
    except LabError:
        print("usage: python -m excavation_lab.jobs run JOB_ID", file=sys.stderr)
        return 2
    return _child_main(args[1])


def _child_main(job_id: str) -> int:
    raw = os.environ.get(_WORKSPACE_ENV)
    if not raw or "\x00" in raw:
        print("EXCAVATION_LAB_WORKSPACE is not set", file=sys.stderr)
        return 2
    job = _load_job(Path(raw).resolve(), job_id)
    if job is None:
        print("job not found", file=sys.stderr)
        return 2
    if job.worker == _TRAIN_WORKER:
        from excavation_lab.models.train import run_train_job

        return run_train_job(Path(raw).resolve(), job_id)
    if job.worker == _ANALYZE_WORKER:
        from excavation_lab.tracking.analyze import run_analyze_job

        return run_analyze_job(Path(raw).resolve(), job_id)
    if job.worker == _EXPORT_WORKER:
        from excavation_lab.visualization.export_video import run_export_job

        return run_export_job(Path(raw).resolve(), job_id)
    if job.worker not in _SLEEP_WORKERS:
        print("unsupported worker", file=sys.stderr)
        return 2
    _sleep_worker(job)
    return 0


def _sleep_worker(job: _StoredJob) -> None:
    """Fake worker. Timestamps start at ``step_s`` and stop before the duration."""
    stop = False

    def _on_term(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    if job.worker == "sleep_ignore_term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, _on_term)

    frames: list[dict[str, float]] = []
    stamp = job.step_s
    run_dir = Path(job.run_dir)
    while stamp < job.video_duration_s and not stop:
        frames.append({"source_time_s": stamp})
        write_progress(
            run_dir,
            {
                "job_id": job.job_id,
                "video_duration_s": job.video_duration_s,
                "analyzed_end_s": stamp,
                "frames": list(frames),
            },
        )
        if stop:
            break
        time.sleep(job.sleep_s)
        stamp += job.step_s


def _spawn(workspace: Path, job_id: str, run_dir: Path) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env[_WORKSPACE_ENV] = str(workspace)
    log_handle = (run_dir / "worker.log").open("ab")
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "excavation_lab.jobs", "run", job_id],
            cwd=str(workspace),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()


def _signal_worker(pid: int | None, job_id: str, kill_after_s: float) -> None:
    """SIGTERM, then SIGKILL, and only for this job's worker PID."""
    if pid is None:
        return
    if not _is_our_worker(pid, job_id):
        _reap(pid)
        return
    _kill(pid, signal.SIGTERM)
    if not _wait_until_gone(pid, job_id, kill_after_s) and _is_our_worker(pid, job_id):
        _kill(pid, signal.SIGKILL)
        _wait_until_gone(pid, job_id, kill_after_s)
    # A SIGTERM can land between the liveness check and ``ps``. The PID is
    # then a zombie whose command line is ``<defunct>`` until this parent reaps it.
    _reap(pid)


def _wait_until_gone(pid: int, job_id: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if not _process_alive(pid):
            return True
        if not _is_our_worker(pid, job_id):
            _reap(pid)
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def _is_our_worker(pid: int, job_id: str) -> bool:
    command = _process_command(pid)
    if command is None:
        return False
    needle = f"-m excavation_lab.jobs run {job_id}"
    return needle in command and command.endswith(job_id)


def _process_command(pid: int) -> str | None:
    completed = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip()
    return text or None


def _kill(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return


def _reap(pid: int) -> bool:
    """Collect a dead child. True when this call reaped ``pid``."""
    try:
        waited, _status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False
    return waited == pid


def _process_alive(pid: int | None) -> bool:
    if pid is None or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    if _reap(pid):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reconcile(workspace: Path) -> None:
    """Fail a ``running`` job whose PID is gone. Never spawns a process."""
    if not lab_database(workspace).is_file():
        return
    for job in _load_all(workspace):
        if job.status != JobStatus.running.value:
            continue
        if _process_alive(job.pid):
            continue
        partial, end_s = _outcome_from_progress(Path(job.run_dir))
        _save(
            workspace,
            replace(
                job,
                status=JobStatus.failed.value,
                partial=partial,
                analyzed_end_s=end_s,
                error_code="orphaned",
                orphaned=True,
            ),
        )


def _acquire(workspace: Path, job: _StoredJob) -> None:
    with _lab(workspace) as connection, _immediate(connection):
        current = connection.execute(
            "SELECT job_id, status FROM workspace_lock WHERE singleton = 1"
        ).fetchone()
        if current is not None and current["status"] in _ACTIVE:
            holder = connection.execute(
                "SELECT status FROM lab_jobs WHERE job_id = ?",
                (current["job_id"],),
            ).fetchone()
            if holder is not None and holder["status"] in _ACTIVE:
                raise JobAlreadyRunningError()
            connection.execute("DELETE FROM workspace_lock WHERE singleton = 1")
        elif current is not None:
            connection.execute("DELETE FROM workspace_lock WHERE singleton = 1")
        _insert_lab(connection, job)
        connection.execute(
            """
            INSERT INTO workspace_lock (singleton, job_id, project_id, status)
            VALUES (1, ?, ?, ?)
            """,
            (job.job_id, job.project_id, job.status),
        )


def _abandon_start(
    workspace: Path,
    job: _StoredJob,
    process: subprocess.Popen[bytes] | None,
) -> None:
    if process is not None and process.poll() is None:
        _kill(process.pid, signal.SIGKILL)
        _process_alive(process.pid)
    current = _load_job(workspace, job.job_id)
    if current is None or current.status in _TERMINAL:
        return
    try:
        _save(
            workspace,
            replace(
                current,
                status=JobStatus.failed.value,
                pid=process.pid if process is not None else current.pid,
                partial=False,
                analyzed_end_s=None,
                error_code="spawn_failed",
                orphaned=False,
            ),
        )
    except (LabError, sqlite3.Error, OSError):
        return


def _save(workspace: Path, job: _StoredJob) -> _StoredJob:
    _to_record(job)
    with _lab(workspace) as connection, _immediate(connection):
        cursor = connection.execute(
            """
            UPDATE lab_jobs
            SET status = ?, pid = ?, partial = ?, analyzed_end_s = ?,
                error_code = ?, orphaned = ?
            WHERE job_id = ?
            """,
            (
                job.status,
                job.pid,
                1 if job.partial else 0,
                job.analyzed_end_s,
                job.error_code,
                1 if job.orphaned else 0,
                job.job_id,
            ),
        )
        if cursor.rowcount != 1:
            raise LabError("job_not_found", "Задание не найдено в этом проекте.")
        if job.status in _ACTIVE:
            connection.execute(
                """
                INSERT INTO workspace_lock (singleton, job_id, project_id, status)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    job_id = excluded.job_id,
                    project_id = excluded.project_id,
                    status = excluded.status
                WHERE workspace_lock.job_id = excluded.job_id
                """,
                (job.job_id, job.project_id, job.status),
            )
        else:
            connection.execute(
                "DELETE FROM workspace_lock WHERE singleton = 1 AND job_id = ?",
                (job.job_id,),
            )
    _sync_project(workspace, job)
    return job


def _insert_lab(connection: sqlite3.Connection, job: _StoredJob) -> None:
    connection.execute(
        """
        INSERT INTO lab_jobs (
            job_id, project_id, kind, worker, status, pid, partial,
            analyzed_end_s, error_code, orphaned, run_dir,
            video_duration_s, step_s, sleep_s, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            job.job_id,
            job.project_id,
            job.kind,
            job.worker,
            job.status,
            job.pid,
            1 if job.partial else 0,
            job.analyzed_end_s,
            job.error_code,
            1 if job.orphaned else 0,
            job.run_dir,
            job.video_duration_s,
            job.step_s,
            job.sleep_s,
            job.created_at,
        ),
    )


def _sync_project(workspace: Path, job: _StoredJob) -> None:
    db_path = project_sqlite(workspace, job.project_id)
    partial = 1 if job.partial else 0
    with connect(db_path) as connection:
        existing = connection.execute(
            "SELECT job_id FROM jobs WHERE job_id = ?",
            (job.job_id,),
        ).fetchone()
        if existing is None:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, project_id, status, pid, partial,
                    analyzed_end_s, error_code, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.project_id,
                    job.status,
                    job.pid,
                    partial,
                    job.analyzed_end_s,
                    job.error_code,
                    job.created_at,
                ),
            )
            return
        cursor = connection.execute(
            """
            UPDATE jobs
            SET status = ?, pid = ?, partial = ?, analyzed_end_s = ?, error_code = ?
            WHERE job_id = ? AND project_id = ?
            """,
            (
                job.status,
                job.pid,
                partial,
                job.analyzed_end_s,
                job.error_code,
                job.job_id,
                job.project_id,
            ),
        )
        if cursor.rowcount != 1:
            raise LabError("job_not_found", "Задание не найдено в этом проекте.")


def _require_job(workspace: Path, project_id: str, job_id: str) -> _StoredJob:
    job = _load_job(workspace, job_id)
    if job is None or job.project_id != project_id:
        raise LabError("job_not_found", "Задание не найдено в этом проекте.")
    return job


def _load_job(workspace: Path, job_id: str) -> _StoredJob | None:
    if not lab_database(workspace).is_file():
        return None
    with _lab(workspace) as connection:
        row = connection.execute(
            "SELECT * FROM lab_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    return _stored(row)


def _load_all(workspace: Path) -> list[_StoredJob]:
    if not lab_database(workspace).is_file():
        return []
    with _lab(workspace) as connection:
        rows = connection.execute(
            "SELECT * FROM lab_jobs ORDER BY created_at, job_id"
        ).fetchall()
    return [_stored(row) for row in rows]


def _stored(row: sqlite3.Row) -> _StoredJob:
    pid = row["pid"]
    end_s = row["analyzed_end_s"]
    error_code = row["error_code"]
    return _StoredJob(
        job_id=row["job_id"],
        project_id=row["project_id"],
        kind=row["kind"],
        worker=row["worker"],
        status=row["status"],
        pid=int(pid) if pid is not None else None,
        partial=bool(row["partial"]),
        analyzed_end_s=float(end_s) if end_s is not None else None,
        error_code=error_code if error_code else None,
        orphaned=bool(row["orphaned"]),
        run_dir=row["run_dir"],
        video_duration_s=float(row["video_duration_s"]),
        step_s=float(row["step_s"]),
        sleep_s=float(row["sleep_s"]),
        created_at=row["created_at"],
    )


def _to_record(job: _StoredJob) -> JobRecord:
    try:
        return JobRecord(
            job_id=job.job_id,
            status=JobStatus(job.status),
            pid=job.pid,
            partial=job.partial,
            analyzed_end_s=job.analyzed_end_s,
            error_code=job.error_code,
            orphaned=job.orphaned,
        )
    except ValidationError as exc:
        raise LabError("invalid_job", "Некорректная запись задания.") from exc


def _outcome_from_progress(run_dir: Path) -> tuple[bool, float | None]:
    try:
        return _partial_from_progress(run_dir)
    except LabError:
        return False, None


def _partial_from_progress(run_dir: Path) -> tuple[bool, float | None]:
    payload = _read_progress(run_dir)
    if payload is None:
        return False, None
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        return False, None
    last = frames[-1]
    if not isinstance(last, dict):
        return False, None
    stamp = last.get("source_time_s")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        return False, None
    value = float(stamp)
    if not math.isfinite(value) or value < 0:
        return False, None
    return True, value


def _read_progress(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / PROGRESS_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError(
            "corrupt_progress",
            "Файл прогресса повреждён. Неполный JSON не читается как результат.",
        ) from exc
    if not isinstance(data, dict):
        raise LabError(
            "corrupt_progress",
            "Файл прогресса повреждён. Неполный JSON не читается как результат.",
        )
    return data


def _require_artifacts(paths: Sequence[Path] | None) -> None:
    if not paths:
        raise LabError(
            "missing_artifact",
            "Завершение отклонено: не переданы обязательные файлы результата.",
        )
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        listed = ", ".join(missing)
        raise LabError(
            "missing_artifact",
            f"Завершение отклонено: нет обязательных файлов результата: {listed}.",
        )


def _require_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise LabError(
            "invalid_job",
            "Недопустимый тип задания. Допустимы train, analyze и export.",
        )


def _require_worker(worker: str) -> None:
    if worker not in _WORKERS:
        raise LabError(
            "unsupported_job",
            "Этот обработчик задания ещё не подключён.",
        )


def _require_job_id(job_id: str) -> None:
    if (
        not isinstance(job_id, str)
        or not job_id
        or job_id in {".", ".."}
        or "/" in job_id
        or "\\" in job_id
        or "\x00" in job_id
    ):
        raise LabError("invalid_job", "Недопустимый идентификатор задания.")


def _positive_seconds(value: float, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError("invalid_job", message)
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise LabError("invalid_job", message)
    return number


def _root(workspace: Path) -> Path:
    return Path(workspace).resolve()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_export_kind(connection: sqlite3.Connection) -> None:
    """Rebuild ``lab_jobs`` when an older file rejects kind ``export``.

    ``CREATE TABLE IF NOT EXISTS`` does not update a CHECK constraint.
    Fresh databases already list ``export`` and skip this.
    """
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'lab_jobs'"
    ).fetchone()
    if row is None or not row["sql"] or "'export'" in row["sql"]:
        return
    columns = (
        "job_id, project_id, kind, worker, status, pid, partial, "
        "analyzed_end_s, error_code, orphaned, run_dir, "
        "video_duration_s, step_s, sleep_s, created_at"
    )
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute("ALTER TABLE lab_jobs RENAME TO lab_jobs_old")
        connection.execute(_LAB_JOBS_SQL)
        connection.execute(
            f"INSERT INTO lab_jobs ({columns}) SELECT {columns} FROM lab_jobs_old"
        )
        connection.execute("DROP TABLE lab_jobs_old")
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise


@contextmanager
def _lab(workspace: Path) -> Iterator[sqlite3.Connection]:
    path = lab_database(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5.0)
    try:
        connection.isolation_level = None
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(_LAB_JOBS_SQL)
        connection.execute(_LOCK_SQL)
        _ensure_export_kind(connection)
        yield connection
    finally:
        connection.close()


@contextmanager
def _immediate(connection: sqlite3.Connection) -> Iterator[None]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
