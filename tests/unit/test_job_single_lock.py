"""Workspace job lock. The worker sleeps; it is not RF-DETR."""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from excavation_lab import jobs, storage
from excavation_lab.errors import JobAlreadyRunningError, LabError
from excavation_lab.schemas import JobStatus

_WRITER = """
import sys
from pathlib import Path
from excavation_lab.jobs import write_progress

run_dir = Path(sys.argv[1])
index = 0
while True:
    write_progress(
        run_dir,
        {
            "index": index,
            "analyzed_end_s": index + 1.5,
            "blob": "x" * 120000,
        },
    )
    index += 1
"""


def _cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)


def _stop(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        return


@pytest.fixture
def stop_pids() -> Iterator[list[int]]:
    pids: list[int] = []
    try:
        yield pids
    finally:
        for pid in pids:
            _stop(pid)


def _track(stop_pids: list[int], pid: int | None) -> int:
    assert pid is not None
    stop_pids.append(pid)
    return pid


def _workspace(tmp_path: Path, *names: str) -> Path:
    root = tmp_path / "workspace"
    for name in names:
        storage.create_project(name, root)
    return root


def _wait_for_frames(run_dir: Path) -> dict[str, object]:
    path = run_dir / jobs.PROGRESS_FILENAME
    log_path = run_dir / "worker.log"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            frames = data.get("frames")
            if isinstance(frames, list) and frames:
                return data
        time.sleep(0.02)
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    raise AssertionError(f"worker wrote no frames\n{log}")


def _force_status(workspace: Path, job_id: str, status: str) -> None:
    connection = sqlite3.connect(jobs.lab_database(workspace))
    try:
        connection.execute(
            "UPDATE lab_jobs SET status = ? WHERE job_id = ?",
            (status, job_id),
        )
        connection.execute(
            "UPDATE workspace_lock SET status = ? WHERE job_id = ?",
            (status, job_id),
        )
        connection.commit()
    finally:
        connection.close()


def _project_status(workspace: Path, project_id: str, job_id: str) -> sqlite3.Row:
    with storage.connect(storage.project_sqlite(workspace, project_id)) as connection:
        row = connection.execute(
            """
            SELECT status, partial, analyzed_end_s, error_code
            FROM jobs WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
    assert row is not None
    return row


def test_same_project_second_start_raises(
    tmp_path: Path,
    stop_pids: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []
    real_popen = jobs.subprocess.Popen

    def spy(command: list[str], **kwargs: object) -> subprocess.Popen[bytes]:
        seen.append(list(command))
        assert "shell" not in kwargs
        return real_popen(command, **kwargs)

    monkeypatch.setattr(jobs.subprocess, "Popen", spy)
    workspace = _workspace(tmp_path, "alpha")
    first = jobs.start_job(workspace, "alpha", kind="analyze")
    pid = _track(stop_pids, first.pid)
    with pytest.raises(JobAlreadyRunningError) as caught:
        jobs.start_job(workspace, "alpha", kind="analyze")

    assert isinstance(caught.value, LabError)
    assert caught.value.code == "job_already_running"
    assert caught.value.code.isascii()
    assert _cyrillic(caught.value.message)
    assert seen == [[sys.executable, "-m", "excavation_lab.jobs", "run", first.job_id]]
    command = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert command.endswith(f"-m excavation_lab.jobs run {first.job_id}")
    assert jobs.lab_database(workspace) == workspace.resolve() / "lab.sqlite"
    assert jobs.lab_database(workspace) != Path(storage.project_sqlite(workspace, "alpha"))


def test_cross_project_lock_is_per_workspace(tmp_path: Path, stop_pids: list[int]) -> None:
    workspace = _workspace(tmp_path, "alpha", "beta")
    first = jobs.start_job(workspace, "alpha", kind="analyze")
    _track(stop_pids, first.pid)
    with pytest.raises(JobAlreadyRunningError) as caught:
        jobs.start_job(workspace, "beta", kind="train")
    assert caught.value.code == "job_already_running"
    assert _cyrillic(caught.value.message)

    with storage.connect(storage.project_sqlite(workspace, "beta")) as connection:
        count = connection.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()
        assert count is not None
        assert count["n"] == 0
    assert _project_status(workspace, "alpha", first.job_id)["status"] == "running"
    assert jobs.lab_database(workspace).parent == workspace.resolve()

    other = _workspace(tmp_path / "elsewhere", "alpha")
    second = jobs.start_job(other, "alpha", kind="train")
    _track(stop_pids, second.pid)
    assert second.job_id != first.job_id
    assert second.status is JobStatus.running
    assert jobs.lab_database(other) != jobs.lab_database(workspace)


def test_queued_running_and_cancel_requested_share_the_lock(
    tmp_path: Path,
    stop_pids: list[int],
) -> None:
    workspace = _workspace(tmp_path, "alpha", "beta")
    first = jobs.start_job(workspace, "alpha", kind="train")
    _track(stop_pids, first.pid)
    for status in ("queued", "running", "cancel_requested"):
        _force_status(workspace, first.job_id, status)
        with pytest.raises(JobAlreadyRunningError):
            jobs.start_job(workspace, "beta", kind="analyze")
    with storage.connect(storage.project_sqlite(workspace, "beta")) as connection:
        count = connection.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()
        assert count is not None
        assert count["n"] == 0


def test_cancel_keeps_last_timestamp_and_spares_other_processes(
    tmp_path: Path,
    stop_pids: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duration = 80.0
    workspace = _workspace(tmp_path, "alpha")
    job = jobs.start_job(
        workspace,
        "alpha",
        kind="analyze",
        video_duration_s=duration,
        step_s=2.5,
        sleep_s=0.05,
    )
    job_pid = _track(stop_pids, job.pid)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    _track(stop_pids, unrelated.pid)
    run_dir = jobs.run_directory(workspace, "alpha", job.job_id)
    _wait_for_frames(run_dir)

    sent: list[tuple[int, int]] = []
    real_kill = jobs.os.kill

    def spy(pid: int, sig: int) -> None:
        if sig != 0:
            sent.append((pid, sig))
            if sig == signal.SIGTERM:
                mid = jobs.get_job(workspace, "alpha", job.job_id)
                assert mid.status is JobStatus.cancel_requested
        real_kill(pid, sig)

    monkeypatch.setattr(jobs.os, "kill", spy)
    done = jobs.request_cancel(workspace, "alpha", job.job_id)
    progress = json.loads((run_dir / jobs.PROGRESS_FILENAME).read_text(encoding="utf-8"))
    frames = progress["frames"]
    assert isinstance(frames, list)
    last = frames[-1]["source_time_s"]

    assert done.status is JobStatus.cancelled
    assert done.partial is True
    assert done.analyzed_end_s == last
    assert last > 0
    assert last < duration
    assert done.analyzed_end_s != duration
    signaled = {pid for pid, sig in sent if sig in {signal.SIGTERM, signal.SIGKILL}}
    assert signaled == {job_pid}
    assert any(sig == signal.SIGTERM and pid == job_pid for pid, sig in sent)
    assert unrelated.poll() is None

    stored = _project_status(workspace, "alpha", job.job_id)
    assert stored["status"] == "cancelled"
    assert stored["partial"] == 1
    assert stored["analyzed_end_s"] == last

    artifact = tmp_path / "frame_status.parquet"
    artifact.write_bytes(b"partial")
    with pytest.raises(LabError) as caught:
        jobs.mark_completed(workspace, "alpha", job.job_id, [artifact])
    assert caught.value.code == "job_not_completable"
    again = jobs.get_job(workspace, "alpha", job.job_id)
    assert again.status is JobStatus.cancelled
    assert again.partial is True
    assert again.analyzed_end_s == last

    follow = jobs.start_job(workspace, "alpha", kind="train")
    _track(stop_pids, follow.pid)
    assert follow.status is JobStatus.running
    assert follow.job_id != job.job_id


def test_cancel_does_not_signal_a_foreign_pid(
    tmp_path: Path,
    stop_pids: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path, "alpha")
    job = jobs.start_job(workspace, "alpha", kind="analyze")
    real_pid = _track(stop_pids, job.pid)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    _track(stop_pids, unrelated.pid)
    connection = sqlite3.connect(jobs.lab_database(workspace))
    try:
        connection.execute(
            "UPDATE lab_jobs SET pid = ? WHERE job_id = ?",
            (unrelated.pid, job.job_id),
        )
        connection.commit()
    finally:
        connection.close()

    sent: list[tuple[int, int]] = []
    real_kill = jobs.os.kill

    def spy(pid: int, sig: int) -> None:
        if sig in {signal.SIGTERM, signal.SIGKILL}:
            sent.append((pid, sig))
        real_kill(pid, sig)

    monkeypatch.setattr(jobs.os, "kill", spy)
    jobs.request_cancel(workspace, "alpha", job.job_id)
    assert sent == []
    assert unrelated.poll() is None
    os.kill(real_pid, 0)


def test_sigkill_follows_ignored_sigterm(
    tmp_path: Path,
    stop_pids: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path, "alpha")
    job = jobs.start_job(
        workspace,
        "alpha",
        kind="analyze",
        worker="sleep_ignore_term",
        video_duration_s=80.0,
        step_s=2.5,
        sleep_s=0.05,
    )
    _track(stop_pids, job.pid)
    run_dir = jobs.run_directory(workspace, "alpha", job.job_id)
    _wait_for_frames(run_dir)
    sent: list[tuple[int, int]] = []
    real_kill = jobs.os.kill

    def spy(pid: int, sig: int) -> None:
        if sig in {signal.SIGTERM, signal.SIGKILL}:
            sent.append((pid, sig))
        real_kill(pid, sig)

    monkeypatch.setattr(jobs.os, "kill", spy)
    done = jobs.request_cancel(workspace, "alpha", job.job_id, kill_after_s=0.3)
    sigs = [sig for pid, sig in sent if pid == job.pid]
    assert sigs[0] == signal.SIGTERM
    assert signal.SIGKILL in sigs
    assert {pid for pid, _sig in sent} == {job.pid}
    assert done.status is JobStatus.cancelled
    assert done.partial is True
    assert done.analyzed_end_s is not None
    assert done.analyzed_end_s > 0
    assert done.analyzed_end_s < 80.0


def test_reopen_marks_dead_running_job_orphaned(
    tmp_path: Path,
    stop_pids: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path, "alpha")
    job = jobs.start_job(workspace, "alpha", kind="analyze")
    pid = _track(stop_pids, job.pid)
    live = jobs.reopen(workspace)
    assert len(live) == 1
    assert live[0].job_id == job.job_id
    assert live[0].status is JobStatus.running
    assert live[0].pid == pid
    assert live[0].orphaned is False

    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("reopen started a replacement")

    monkeypatch.setattr(jobs.subprocess, "Popen", boom)
    failed = jobs.reopen(workspace)
    assert len(failed) == 1
    assert failed[0].job_id == job.job_id
    assert failed[0].status is JobStatus.failed
    assert failed[0].orphaned is True
    assert failed[0].error_code == "orphaned"
    assert failed[0].pid == pid
    if failed[0].partial:
        assert failed[0].analyzed_end_s is not None
        assert failed[0].analyzed_end_s > 0
    else:
        assert failed[0].analyzed_end_s is None
    stored = _project_status(workspace, "alpha", job.job_id)
    assert stored["status"] == "failed"
    assert stored["error_code"] == "orphaned"
    again = jobs.reopen(workspace)
    assert [item.job_id for item in again] == [job.job_id]
    assert again[0].status is JobStatus.failed
    assert again[0].orphaned is True


def test_mark_completed_requires_existing_artifacts(
    tmp_path: Path,
    stop_pids: list[int],
) -> None:
    workspace = _workspace(tmp_path, "alpha")
    job = jobs.start_job(workspace, "alpha", kind="train")
    pid = _track(stop_pids, job.pid)
    with pytest.raises(LabError) as missing:
        jobs.mark_completed(workspace, "alpha", job.job_id)
    assert missing.value.code == "missing_artifact"
    assert _cyrillic(missing.value.message)
    absent = tmp_path / "missing.parquet"
    with pytest.raises(LabError) as still_missing:
        jobs.mark_completed(workspace, "alpha", job.job_id, [absent])
    assert still_missing.value.code == "missing_artifact"
    assert jobs.get_job(workspace, "alpha", job.job_id).status is JobStatus.running
    with pytest.raises(JobAlreadyRunningError):
        jobs.start_job(workspace, "alpha", kind="analyze")

    artifact = tmp_path / "frame_status.parquet"
    artifact.write_bytes(b"parquet")
    done = jobs.mark_completed(workspace, "alpha", job.job_id, [artifact])
    assert done.status is JobStatus.completed
    assert done.partial is False
    assert done.orphaned is False
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert _project_status(workspace, "alpha", job.job_id)["status"] == "completed"
    follow = jobs.start_job(workspace, "alpha", kind="analyze")
    _track(stop_pids, follow.pid)
    assert follow.status is JobStatus.running


def test_failed_replace_keeps_the_previous_progress_object(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    jobs.write_progress(run_dir, {"index": 1, "analyzed_end_s": 1.5})

    def boom(_src: str | Path, _dst: str | Path) -> None:
        raise RuntimeError("interrupted before replace")

    original = jobs.os.replace
    jobs.os.replace = boom
    try:
        with pytest.raises(RuntimeError):
            jobs.write_progress(run_dir, {"index": 2, "analyzed_end_s": 3.0})
    finally:
        jobs.os.replace = original
    assert json.loads((run_dir / jobs.PROGRESS_FILENAME).read_text(encoding="utf-8")) == {
        "index": 1,
        "analyzed_end_s": 1.5,
    }


def test_interrupted_writer_never_leaves_partial_json(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / jobs.PROGRESS_FILENAME
    process = subprocess.Popen([sys.executable, "-c", _WRITER, str(run_dir)])
    try:
        deadline = time.monotonic() + 10
        while not path.is_file():
            if time.monotonic() >= deadline or process.poll() is not None:
                log = path.read_text(encoding="utf-8") if path.is_file() else ""
                raise AssertionError(f"writer produced no progress\n{log}")
            time.sleep(0.01)
        read_until = time.monotonic() + 0.25
        reads = 0
        while time.monotonic() < read_until:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            assert isinstance(parsed, dict)
            assert "index" in parsed
            reads += 1
        assert reads > 0
        process.kill()
        process.wait(timeout=5)
        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict)
        assert "analyzed_end_s" in parsed
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
