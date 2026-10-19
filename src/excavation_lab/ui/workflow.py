"""Cross-page workflow. No Starlette.

Five stages, in order: video, data, training, analysis, actions.
``current`` is the first stage that is not done. A project with every stage
done has no ``current`` stage. A missing project leaves every stage
``waiting`` with the page's create-project sentence.

Captions are the sentences the destination page already shows. Counts come
from SQLite and directory checks. This module does not open parquet and does
not call ``jobs.reopen`` (that call can rewrite a dead job).
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from excavation_lab import storage
from excavation_lab.actions.catalog import list_actions
from excavation_lab.actions.labels import list_intervals, role_of
from excavation_lab.datasets.splits import current_blocks
from excavation_lab.errors import LabError
from excavation_lab.ui import copy as ui_copy
from excavation_lab.ui.analysis_page import list_bundles
from excavation_lab.ui.data_page import (
    contact_jpegs,
    first_unfrozen_video_id,
    snapshot_gate,
    snapshot_written,
)
from excavation_lab.ui.training_page import format_error, missing_snapshot_message

_STAGE_SPEC = (
    ("video", "page_video", "/"),
    ("data", "page_data", "/data"),
    ("training", "page_training", "/training"),
    ("analysis", "page_analysis", "/analysis"),
    ("actions", "page_actions", "/actions"),
)
_ACTIVE = frozenset({"queued", "running", "cancel_requested"})


@dataclass(frozen=True, slots=True)
class Stage:
    """One stepper cell. ``state`` is ``done``, ``current``, or ``waiting``."""

    stage_id: str
    title: str
    path: str
    state: str
    caption: str
    status: str


def project_stages(
    workspace: Path,
    project_id: str | None,
    book: ui_copy.Catalog,
) -> tuple[Stage, ...]:
    """Stages for one project. ``project_id`` empty means there is no project."""
    if not isinstance(project_id, str) or not project_id.strip():
        return _waiting(book, book.need_project)
    try:
        videos = storage.list_videos(workspace, project_id)
    except LabError as exc:
        return _waiting(book, exc.message)
    probed = _has_probed(videos)
    snapshot = snapshot_written(workspace, project_id)
    bundles = bool(list_bundles(workspace, project_id))
    custom = _custom_completed(workspace, project_id)
    inferred = _inference_exists(workspace, project_id)
    jobs = _job_rows(workspace, project_id)
    done = (probed, snapshot, bundles, custom, inferred)
    captions = (
        book.add_local_video,
        _data_caption(workspace, project_id, videos, book),
        _training_caption(workspace, project_id, book, snapshot, jobs),
        _analysis_caption(book, probed, bundles, jobs),
        _actions_caption(workspace, project_id, book, custom),
    )
    return _assemble(book, done, captions)


def _waiting(book: ui_copy.Catalog, caption: str) -> tuple[Stage, ...]:
    return tuple(
        Stage(
            stage_id=stage_id,
            title=getattr(book, title_attr),
            path=path,
            state="waiting",
            caption=caption,
            status=book.step_wait,
        )
        for stage_id, title_attr, path in _STAGE_SPEC
    )


def _assemble(
    book: ui_copy.Catalog,
    done: Sequence[bool],
    captions: Sequence[str],
) -> tuple[Stage, ...]:
    words = {
        "done": book.step_done,
        "current": book.step_now,
        "waiting": book.step_wait,
    }
    states = _states(done)
    stages: list[Stage] = []
    for (stage_id, title_attr, path), state, caption, finished in zip(
        _STAGE_SPEC,
        states,
        captions,
        done,
        strict=True,
    ):
        stages.append(
            Stage(
                stage_id=stage_id,
                title=getattr(book, title_attr),
                path=path,
                state=state,
                caption="" if finished else caption,
                status=words[state],
            )
        )
    return tuple(stages)


def _states(done: Sequence[bool]) -> list[str]:
    states: list[str] = []
    opened = False
    for finished in done:
        if finished:
            states.append("done")
            continue
        if not opened:
            states.append("current")
            opened = True
            continue
        states.append("waiting")
    return states


def _has_probed(videos: Sequence[Mapping[str, object]]) -> bool:
    return any(_probed(video) for video in videos)


def _probed(video: Mapping[str, object]) -> bool:
    duration = video.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        return False
    return math.isfinite(duration) and duration > 0


def _data_caption(
    workspace: Path,
    project_id: str,
    videos: Sequence[Mapping[str, object]],
    book: ui_copy.Catalog,
) -> str:
    """First missing piece, using the data page's own sentences."""
    if not videos or first_unfrozen_video_id(workspace, project_id) is not None:
        return snapshot_gate(workspace, project_id, book).reason
    sampled = _sample_caption(workspace, project_id, videos, book)
    if sampled is not None:
        return sampled
    gate = snapshot_gate(workspace, project_id, book)
    if not gate.enabled:
        return gate.reason
    return book.snapshot_not_built


def _sample_caption(
    workspace: Path,
    project_id: str,
    videos: Sequence[Mapping[str, object]],
    book: ui_copy.Catalog,
) -> str | None:
    """``None`` when a sampled JPEG exists. Otherwise the contact-sheet sentence."""
    broken = ""
    for video in videos:
        video_id = video.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            continue
        paths, error = contact_jpegs(workspace, project_id, video_id, book)
        if paths:
            return None
        if error and not broken:
            broken = error
    if broken:
        return broken
    return book.contact_empty


def _training_caption(
    workspace: Path,
    project_id: str,
    book: ui_copy.Catalog,
    snapshot: bool,
    jobs: Sequence[tuple[str, str]],
) -> str:
    if not snapshot:
        return format_error(
            "dataset_not_ready",
            missing_snapshot_message(workspace, project_id, book),
        )
    if any(status in _ACTIVE for _kind, status in jobs):
        return book.job_already_running
    if not any(kind == "train" for kind, _status in jobs):
        return book.train_jobs_empty
    return book.training_empty


def _analysis_caption(
    book: ui_copy.Catalog,
    probed: bool,
    bundles: bool,
    jobs: Sequence[tuple[str, str]],
) -> str:
    if not probed:
        return book.analysis_no_video
    if not bundles:
        return book.custom_needs_bundle
    if any(status in _ACTIVE for _kind, status in jobs):
        return book.job_already_running
    return book.analysis_empty


def _actions_caption(
    workspace: Path,
    project_id: str,
    book: ui_copy.Catalog,
    custom: bool,
) -> str:
    try:
        classes = list_actions(workspace, project_id)
    except LabError:
        classes = []
    if not classes:
        return book.action_step_need_catalog
    if not _intervals_ready(workspace, project_id):
        return book.action_coverage_hint
    if not custom:
        return book.action_no_runs
    if not _action_model_exists(workspace, project_id):
        return book.action_no_model
    return book.action_no_inference


def _intervals_ready(workspace: Path, project_id: str) -> bool:
    try:
        videos = storage.list_videos(workspace, project_id)
    except LabError:
        return False
    train = 0
    validation = 0
    for video in videos:
        video_id = video.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            continue
        try:
            blocks = current_blocks(workspace, project_id, video_id)
            intervals = list_intervals(workspace, project_id, video_id)
        except LabError:
            continue
        for interval in intervals:
            role = role_of(interval, blocks)
            if role == "train":
                train += 1
            elif role == "validation":
                validation += 1
            if train and validation:
                return True
    return False


def _custom_completed(workspace: Path, project_id: str) -> bool:
    try:
        rows = storage.list_analysis_runs(workspace, project_id, origin="custom")
    except LabError:
        return False
    return any(row["completed"] for row in rows)


def _action_model_exists(workspace: Path, project_id: str) -> bool:
    try:
        return bool(storage.list_action_models(workspace, project_id))
    except LabError:
        return False


def _inference_exists(workspace: Path, project_id: str) -> bool:
    try:
        return bool(storage.list_action_inferences(workspace, project_id))
    except LabError:
        return False


def _job_rows(workspace: Path, project_id: str) -> list[tuple[str, str]]:
    """``(kind, status)`` from ``lab.sqlite``. Missing file is an empty list.

    Opened read-only so a stepper render does not create the database.
    """
    from excavation_lab.jobs import lab_database

    path = lab_database(workspace)
    try:
        if path.is_symlink() or not path.is_file():
            return []
    except OSError:
        return []
    uri = path.resolve().as_uri() + "?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return []
    try:
        connection.row_factory = sqlite3.Row
        present = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'lab_jobs'"
        ).fetchone()
        if present is None:
            return []
        rows = connection.execute(
            "SELECT kind, status FROM lab_jobs WHERE project_id = ?",
            (project_id,),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()
    found: list[tuple[str, str]] = []
    for row in rows:
        kind = row["kind"]
        status = row["status"]
        if isinstance(kind, str) and isinstance(status, str):
            found.append((kind, status))
    return found
