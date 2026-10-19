"""Action catalog and interval timeline. No Starlette.

Intervals on screen use the player clock. ``save_intervals`` converts each
edge with ``source_interval_from_player`` before ``replace_intervals``.
Frame-step rounding stays in the browser. Stored seconds are the source
clock the server computed.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from excavation_lab import storage
from excavation_lab.actions.catalog import (
    ActionClass,
    add_action,
    edit_title,
    list_actions,
    seed_actions,
)
from excavation_lab.actions.labels import (
    ActionInterval,
    LabelRevision,
    list_intervals,
    replace_intervals,
    revision,
    role_of,
)
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.splits import FrozenSplit
from excavation_lab.errors import LabError
from excavation_lab.ui.copy import Catalog
from excavation_lab.ui.data_page import (
    ActionNote,
    load_split_view,
    source_interval_from_player,
    split_label,
    timeline_pieces,
)


@dataclass(frozen=True, slots=True)
class ActionStep:
    """One step on «Действия». Without a video, train and results stay waiting."""

    anchor: str
    state: str
    title: str
    instruction: str
    status: str
    waiting_reason: str


@dataclass(frozen=True, slots=True)
class ActionWrite:
    """A refused write keeps ``note.code``. A success carries ``payload``."""

    note: ActionNote | None = None
    payload: dict[str, object] | None = None


def catalog_rows(workspace: Path, project_id: str) -> list[dict[str, object]]:
    """Catalog classes in id order. ``title`` is the stored operator text."""
    return [_class_row(item) for item in list_actions(workspace, project_id)]


def workflow_steps(
    *,
    catalog_count: int,
    has_video: bool,
    text: Catalog,
) -> tuple[ActionStep, ...]:
    """Four steps. A video opens train and results; otherwise they stay waiting."""
    has_catalog = catalog_count > 0
    titles = (
        text.action_step_catalog,
        text.action_step_labels,
        text.action_step_train,
        text.action_step_results,
    )
    if not has_video:
        if has_catalog:
            states = ("done", "current", "waiting", "waiting")
        else:
            states = ("current", "waiting", "waiting", "waiting")
        instructions = (
            text.action_step_catalog_text,
            text.action_no_video,
            text.action_step_train_wait,
            text.action_step_results_wait,
        )
        if not has_catalog:
            label_reason = text.action_step_need_catalog
        else:
            label_reason = text.action_no_video
        reasons = (
            "",
            label_reason,
            text.action_step_train_wait,
            text.action_step_results_wait,
        )
    else:
        states = (
            "done" if has_catalog else "current",
            "current" if has_catalog else "waiting",
            "current",
            "current",
        )
        instructions = (
            text.action_step_catalog_text,
            text.action_step_labels_text if has_catalog else text.action_step_need_catalog,
            text.action_step_train_text,
            text.action_step_results_text,
        )
        reasons = (
            "",
            "" if has_catalog else text.action_step_need_catalog,
            "",
            "",
        )
    status_words = {
        "current": text.step_now,
        "done": text.step_done,
        "waiting": text.step_wait,
    }
    anchors = ("step-catalog", "step-labels", "step-train", "step-results")
    return tuple(
        ActionStep(
            anchor=anchor,
            state=state,
            title=title,
            instruction=instruction,
            status=status_words[state],
            waiting_reason=reason,
        )
        for anchor, state, title, instruction, reason in zip(
            anchors,
            states,
            titles,
            instructions,
            reasons,
            strict=True,
        )
    )


def timeline_payload(
    video: Mapping[str, object],
    blocks: Sequence[RoleBlock],
    intervals: Sequence[ActionInterval],
    book: Catalog,
    *,
    frozen: bool = False,
    classes: Sequence[ActionClass] | None = None,
) -> dict[str, object]:
    """Player seconds for the track. Source seconds are what a save stores.

    ``offset_s`` is null when the probe has no start offset. The page does
    not treat that as zero. ``frame_step_s`` is cosmetic and is not a clock.
    """
    offset_s = _offset(video.get("start_offset_s"))
    duration_s = _positive(video.get("duration_s"))
    fps = _positive(video.get("fps_average"))
    if fps is None:
        fps = _positive(video.get("fps_nominal"))
    frame_step_s = None if fps is None else 1.0 / fps
    titles = {item.name: item.title for item in classes or ()}
    role_blocks = [
        _block_row(block, offset_s, book) for block in blocks
    ]
    rows = [
        _interval_row(item, blocks, offset_s, titles, book, draft=not frozen)
        for item in intervals
    ]
    return {
        "video_id": str(video.get("video_id") or ""),
        "duration_s": duration_s,
        "offset_s": offset_s,
        "fps": fps,
        "frame_step_s": frame_step_s,
        "frozen": frozen,
        "revision": revision_payload(intervals),
        "blocks": role_blocks,
        "intervals": rows,
        "classes": [
            {"name": item.name, "title": item.title} for item in classes or ()
        ],
        "strings": _client_strings(book),
    }


def coverage_rows(
    classes: Sequence[ActionClass],
    intervals: Sequence[ActionInterval],
    blocks: Sequence[RoleBlock],
) -> list[dict[str, object]]:
    """Train and validation counts. A zero on either side is missing.

    Saving is not refused here. Test-block intervals are not coverage.
    """
    counts = {item.name: {"train": 0, "validation": 0} for item in classes}
    for interval in intervals:
        role = role_of(interval, blocks)
        bucket = counts.get(interval.action_name)
        if bucket is None or role not in {"train", "validation"}:
            continue
        bucket[role] += 1
    rows: list[dict[str, object]] = []
    for item in classes:
        train = counts[item.name]["train"]
        validation = counts[item.name]["validation"]
        rows.append(
            {
                "name": item.name,
                "title": item.title,
                "train": train,
                "validation": validation,
                "missing": train == 0 or validation == 0,
            }
        )
    return rows


def revision_payload(intervals: Sequence[ActionInterval]) -> dict[str, object]:
    """``count`` plus the SHA-256 of the canonical interval list."""
    found = revision(intervals)
    return {"count": found.count, "sha256": found.digest}


def save_class(workspace: Path, project_id: str, body: object) -> ActionWrite:
    """Add a class, or change ``title`` when ``edit`` is true."""
    if not isinstance(body, Mapping):
        return _invalid()
    name = body.get("name")
    title = body.get("title")
    try:
        if body.get("edit") is True:
            edit_title(workspace, project_id, name, title)  # type: ignore[arg-type]
        else:
            add_action(workspace, project_id, name, title)  # type: ignore[arg-type]
    except LabError as exc:
        return ActionWrite(note=ActionNote(text=exc.message, code=exc.code))
    return ActionWrite(payload={"classes": catalog_rows(workspace, project_id)})


def seed_catalog(
    workspace: Path,
    project_id: str,
    titles: Mapping[str, str],
) -> ActionWrite:
    """Insert missing seed classes. Titles already stored stay as entered."""
    try:
        seed_actions(workspace, project_id, titles)
    except LabError as exc:
        return ActionWrite(note=ActionNote(text=exc.message, code=exc.code))
    return ActionWrite(payload={"classes": catalog_rows(workspace, project_id)})


def save_intervals(
    workspace: Path,
    project_id: str,
    video_id: str,
    body: object,
    book: Catalog,
) -> ActionWrite:
    """Replace one video when ``revision`` is the list the client loaded.

    Each edge is a player time. The stored pair is the source clock from
    ``source_interval_from_player``. A stale or missing revision writes nothing.
    """
    if not isinstance(body, Mapping):
        return ActionWrite(
            note=ActionNote(text=book.action_invalid_interval, code="invalid_action_label")
        )
    expected = _expected_revision(body.get("revision"))
    if expected is None:
        return ActionWrite(note=ActionNote(text=book.action_stale, code="action_labels_stale"))
    raw_items = body.get("intervals")
    if not isinstance(raw_items, list):
        return ActionWrite(
            note=ActionNote(text=book.action_invalid_interval, code="invalid_action_label")
        )
    try:
        video = storage.get_video(workspace, project_id, video_id)
        converted = [_player_item(video, item) for item in raw_items]
        stored = replace_intervals(
            workspace,
            project_id,
            video_id,
            converted,
            expected_revision=expected,
        )
    except LabError as exc:
        code = "action_labels_stale" if exc.code == "stale_labels" else exc.code
        return ActionWrite(note=ActionNote(text=exc.message, code=code))
    blocks = _blocks(workspace, project_id, video_id)
    classes = list_actions(workspace, project_id)
    frozen = _is_frozen(workspace, project_id, video_id, book)
    return ActionWrite(
        payload={
            "revision": revision_payload(stored),
            "intervals": [
                _interval_row(
                    item,
                    blocks,
                    _offset(video.get("start_offset_s")),
                    {item.name: item.title for item in classes},
                    book,
                    draft=not frozen,
                )
                for item in stored
            ],
            "coverage": coverage_rows(classes, stored, blocks),
        }
    )


def intervals_document(
    workspace: Path,
    project_id: str,
    video_id: str,
    book: Catalog,
) -> dict[str, object]:
    """GET body: source intervals plus the revision of that list."""
    video = storage.get_video(workspace, project_id, video_id)
    stored = list_intervals(workspace, project_id, video_id)
    blocks = _blocks(workspace, project_id, video_id)
    classes = list_actions(workspace, project_id)
    frozen = _is_frozen(workspace, project_id, video_id, book)
    titles = {item.name: item.title for item in classes}
    offset_s = _offset(video.get("start_offset_s"))
    return {
        "video_id": video_id,
        "frozen": frozen,
        "revision": revision_payload(stored),
        "intervals": [
            _interval_row(item, blocks, offset_s, titles, book, draft=not frozen)
            for item in stored
        ],
    }


def http_status(code: str) -> int:
    """409 when the echoed revision is stale or a job already holds the lock."""
    if code in {"action_labels_stale", "job_already_running"}:
        return 409
    if code == "unsupported_media_type":
        return 415
    if code in {"video_not_found", "job_not_found"}:
        return 404
    return 400


def _blocks(workspace: Path, project_id: str, video_id: str) -> tuple[RoleBlock, ...]:
    document, _error = load_split_view(workspace, project_id, video_id)
    if document is None:
        return ()
    return tuple(document.blocks)


def _is_frozen(workspace: Path, project_id: str, video_id: str, book: Catalog) -> bool:
    document, _error = load_split_view(workspace, project_id, video_id, book)
    return isinstance(document, FrozenSplit)


def _player_item(video: Mapping[str, object], item: object) -> dict[str, object]:
    if not isinstance(item, Mapping):
        raise LabError("invalid_action_label", "Некорректный интервал действия.")
    start = _number(item, "player_start_s", "start_s")
    end = _number(item, "player_end_s", "end_s")
    if start is None or end is None:
        raise LabError("invalid_action_label", "Некорректный интервал действия.")
    source_start, source_end = source_interval_from_player(video, start, end)
    draft: dict[str, object] = {
        "start_s": source_start,
        "end_s": source_end,
        "action_name": item.get("action_name"),
    }
    for key in ("label_id", "annotator", "created_at"):
        value = item.get(key)
        if isinstance(value, str) and value:
            draft[key] = value
    return draft


def _number(item: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        if key not in item:
            continue
        value = item[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return number
    return None


def _expected_revision(value: object) -> LabelRevision | None:
    if not isinstance(value, Mapping):
        return None
    count = value.get("count")
    digest = value.get("sha256")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    if not isinstance(digest, str) or len(digest) != 64:
        return None
    if any(char not in "0123456789abcdef" for char in digest):
        return None
    return LabelRevision(count=count, digest=digest)


def _class_row(item: ActionClass) -> dict[str, object]:
    return {"id": item.id, "name": item.name, "title": item.title}


def _block_row(block: RoleBlock, offset_s: float | None, book: Catalog) -> dict[str, object]:
    return {
        "start_s": block.start_s,
        "end_s": block.end_s,
        "player_start_s": _player(block.start_s, offset_s),
        "player_end_s": _player(block.end_s, offset_s),
        "role": block.role,
        "label": split_label(block.role, book),
    }


def _interval_row(
    item: ActionInterval,
    blocks: Sequence[RoleBlock],
    offset_s: float | None,
    titles: Mapping[str, str],
    book: Catalog,
    *,
    draft: bool,
) -> dict[str, object]:
    role = role_of(item, blocks)
    return {
        "label_id": item.label_id,
        "start_s": item.start_s,
        "end_s": item.end_s,
        "player_start_s": _player(item.start_s, offset_s),
        "player_end_s": _player(item.end_s, offset_s),
        "action_name": item.action_name,
        "title": titles.get(item.action_name, item.action_name),
        "annotator": item.annotator,
        "created_at": item.created_at,
        "role": role,
        "role_label": _role_text(role, book, draft=draft),
        "duration_s": item.end_s - item.start_s,
    }


def _role_text(role: str | None, book: Catalog, *, draft: bool) -> str:
    label = book.not_calculated if role is None else split_label(role, book)
    if draft:
        return f"{label} · {book.action_draft}"
    return label


def _client_strings(book: Catalog) -> dict[str, str]:
    return {
        "overlap": book.action_overlap,
        "crosses": book.action_crosses_block,
        "outside": book.action_outside_block,
        "invalid": book.action_invalid_interval,
        "save_failed": book.action_save_failed,
        "missing": book.action_coverage_missing,
        "draft": book.action_draft,
        "not_calculated": book.not_calculated,
        "train": book.split_name_train,
        "validation": book.split_name_validation,
        "reload": book.action_reload,
    }


def _player(source_s: float, offset_s: float | None) -> float | None:
    if offset_s is None:
        return None
    return float(source_s) - offset_s


def _positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _offset(value: object) -> float | None:
    """A stored 0 is a real offset. Missing or non-finite stays null."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _invalid() -> ActionWrite:
    return ActionWrite(
        note=ActionNote(text="Некорректный интервал действия.", code="invalid_action_label")
    )


def role_strip(
    blocks: Sequence[RoleBlock],
    video: Mapping[str, object],
    book: Catalog,
) -> list[dict[str, object]]:
    """Read-only role pieces in the same shape as the data-page strip."""
    offset_s = _offset(video.get("start_offset_s"))
    duration_s = _positive(video.get("duration_s"))
    return timeline_pieces(blocks, offset_s, duration_s, book)


_ACTIVE = frozenset({"queued", "running", "cancel_requested"})
_DEFAULT_EPOCHS = 40
_DEFAULT_WINDOW_S = 3.0
_DEFAULT_CONFIDENCE = 0.5
_DEFAULT_DURATION_S = 1.0
_HISTORY_NAME = "metrics_history.csv"
_PROGRESS_NAME = "progress.json"
_BUNDLE_NAME = "action_bundle.json"
_INTERVALS_NAME = "action_intervals.csv"
_RUN_NAME = "action_run.json"
_TEST_EVAL = "evaluation_test.json"
_VALIDATION_PREFIX = "evaluation_validation_"
_UNKNOWN = "unknown"


def parse_run_ids(values: Sequence[str]) -> list[str]:
    """Comma-separated or repeated run ids. Path tokens are dropped."""
    found: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        for part in value.split(","):
            token = part.strip()
            if not token or token in seen or not _safe_token(token):
                continue
            seen.add(token)
            found.append(token)
    return found


def readiness_document(
    workspace: Path,
    project_id: str,
    run_ids: Sequence[str],
) -> dict[str, object]:
    """``check_ready`` as JSON. ``reasons`` are the service text, unchanged."""
    from excavation_lab.actions.job import check_ready

    found = check_ready(workspace, project_id, list(run_ids))
    return {
        "ok": found.ok,
        "reasons": list(found.reasons),
        "detector_model_id": found.detector_model_id,
        "analysis_fps": found.analysis_fps,
    }


def start_action_train(
    workspace: Path,
    project_id: str,
    run_ids: Sequence[str],
    fields: Mapping[str, object],
) -> ActionWrite:
    """Queue ``enqueue_action_train`` unless a job already holds the lock.

    The route closes the page-server detector before this runs.
    """
    blocked = blocking_job(workspace)
    if blocked is not None:
        from excavation_lab.errors import JobAlreadyRunningError

        running = JobAlreadyRunningError()
        return ActionWrite(note=ActionNote(text=running.message, code=running.code))
    config = _train_config(fields)
    if isinstance(config, ActionNote):
        return ActionWrite(note=config)
    from excavation_lab.actions.job import enqueue_action_train

    try:
        record = enqueue_action_train(workspace, project_id, list(run_ids), config)
    except LabError as exc:
        return ActionWrite(note=ActionNote(text=exc.message, code=exc.code))
    return ActionWrite(payload={"job_id": record.job_id})


def start_action_infer(
    workspace: Path,
    project_id: str,
    run_id: str,
    action_model_id: str,
    fields: Mapping[str, object],
) -> ActionWrite:
    """Queue inference for one run. Thresholds omitted here use the bundle."""
    blocked = blocking_job(workspace)
    if blocked is not None:
        from excavation_lab.errors import JobAlreadyRunningError

        running = JobAlreadyRunningError()
        return ActionWrite(note=ActionNote(text=running.message, code=running.code))
    if not _safe_token(run_id) or not _safe_token(action_model_id):
        return ActionWrite(
            note=ActionNote(
                text="Инференс действий выполняется на одном прогоне.",
                code="invalid_action_infer",
            )
        )
    confidence = _optional_unit(fields.get("min_confidence"))
    duration = _optional_duration(fields.get("min_duration_s"))
    if confidence is False or duration is False:
        return ActionWrite(
            note=ActionNote(
                text="Порог уверенности или длительности действия задан неверно.",
                code="invalid_action_infer",
            )
        )
    from excavation_lab.actions.infer import enqueue_action_infer

    try:
        record = enqueue_action_infer(
            workspace,
            project_id,
            run_id,
            action_model_id,
            min_confidence=None if confidence is None else float(confidence),
            min_duration_s=None if duration is None else float(duration),
        )
    except LabError as exc:
        return ActionWrite(note=ActionNote(text=exc.message, code=exc.code))
    return ActionWrite(payload={"job_id": record.job_id})


def cancel_action_job(workspace: Path, project_id: str, job_id: str) -> ActionWrite:
    """Cancel the job that holds the lock when ``job_id`` is that job."""
    from excavation_lab.jobs import request_cancel

    blocked = blocking_job(workspace)
    if blocked is None or blocked.job_id != job_id:
        return ActionWrite(payload={"job_id": job_id, "status": "absent"})
    try:
        record = request_cancel(workspace, project_id, job_id)
    except LabError as exc:
        return ActionWrite(note=ActionNote(text=exc.message, code=exc.code))
    return ActionWrite(payload={"job_id": record.job_id, "status": record.status.value})


def action_job_document(workspace: Path, project_id: str, job_id: str) -> dict[str, object]:
    """Status from the job row. Epoch fields from ``progress.json``. History from CSV.

    This is not the detection-training progress endpoint.
    """
    from excavation_lab.jobs import _require_job_id, get_job, run_directory

    _require_job_id(job_id)
    record = get_job(workspace, project_id, job_id)
    run_dir = run_directory(workspace, project_id, job_id)
    progress = _read_json_object(run_dir / _PROGRESS_NAME)
    return {
        "status": record.status.value,
        "epoch": _json_number(None if progress is None else progress.get("epoch")),
        "epochs": _json_number(None if progress is None else progress.get("epochs")),
        "val_macro_f1": _json_number(
            None if progress is None else progress.get("val_macro_f1")
        ),
        "history": _read_history(run_dir / _HISTORY_NAME),
        "error_code": record.error_code,
    }


def action_export_bytes(workspace: Path, project_id: str, job_id: str) -> bytes:
    """UTF-8 ``action_intervals.csv`` for one inference job in this project."""
    from excavation_lab.jobs import _require_job_id

    _require_job_id(job_id)
    directory = _inference_dir(workspace, project_id, job_id)
    if directory is None or not _inside(workspace, directory):
        raise LabError("job_not_found", "Задание инференса действий не найдено.")
    path = directory / _INTERVALS_NAME
    if path.is_symlink() or not path.is_file() or not _inside(workspace, path):
        raise LabError("missing_artifact", "В задании нет action_intervals.csv.")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise LabError("missing_artifact", "В задании нет action_intervals.csv.") from exc


def train_panel(
    workspace: Path,
    project_id: str,
    video: Mapping[str, object],
    blocks: Sequence[RoleBlock],
    book: Catalog,
    *,
    selected_runs: Sequence[str],
    model_id: str | None,
) -> dict[str, object]:
    """Step 3. Disabled text is a service sentence or a catalog empty state."""
    video_id = str(video.get("video_id") or "")
    runs = _custom_runs(workspace, project_id, video_id, blocks, book)
    listed = {row["id"] for row in runs}
    queried = [run_id for run_id in selected_runs if _safe_token(run_id)]
    checked = [run_id for run_id in queried if run_id in listed]
    blocked = blocking_job(workspace)
    reasons: list[str] = []
    ready = False
    if queried:
        document = readiness_document(workspace, project_id, queried)
        reasons = [str(item) for item in document["reasons"]]
        ready = bool(document["ok"])
    if blocked is not None:
        from excavation_lab.errors import JobAlreadyRunningError

        start_reason = JobAlreadyRunningError().message
        start_enabled = False
    elif not runs and not queried:
        start_reason = book.action_no_runs
        start_enabled = False
    elif not queried:
        start_reason = book.action_select_runs
        start_enabled = False
    else:
        start_enabled = ready
        start_reason = "" if ready else (reasons[0] if reasons else book.action_select_runs)
    if queried and ready and not reasons:
        reasons = [book.action_ready_ok]
    models = _model_rows(workspace, project_id, book, model_id)
    chosen = next((row["id"] for row in models if row["selected"]), None)
    detector = next(
        (str(row["detector_model_id"]) for row in models if row["selected"]),
        None,
    )
    inferences = _inference_rows(workspace, project_id, book)
    chart = _model_history(workspace, project_id, str(chosen) if chosen else None)
    return {
        "runs": [{**row, "selected": row["id"] in checked} for row in runs],
        "reasons": reasons,
        "ready": ready and blocked is None,
        "start_enabled": start_enabled,
        "start_reason": start_reason,
        "blocked": blocked is not None,
        "cancel_job_id": None if blocked is None else blocked.job_id,
        "poll_job_id": None if blocked is None else blocked.job_id,
        "poll_status": "" if blocked is None else blocked.status.value,
        "models": models,
        "selected_model_id": chosen,
        "infer_runs": _infer_run_options(runs, detector, book),
        "infer_enabled": chosen is not None and blocked is None and any(
            not row["disabled"] for row in _infer_run_options(runs, detector, book)
        ),
        "infer_reason": _infer_reason(book, chosen, runs, detector, blocked is not None),
        "inferences": inferences,
        "epochs": _DEFAULT_EPOCHS,
        "window_s": _DEFAULT_WINDOW_S,
        "min_confidence": _DEFAULT_CONFIDENCE,
        "min_duration_s": _DEFAULT_DURATION_S,
        "chart_history": chart,
    }


def results_panel(
    workspace: Path,
    project_id: str,
    video: Mapping[str, object],
    blocks: Sequence[RoleBlock],
    intervals: Sequence[ActionInterval],
    classes: Sequence[ActionClass],
    book: Catalog,
    *,
    model_id: str | None,
    inference_id: str | None,
) -> dict[str, object]:
    """Step 4. Test labels follow the inference's model, not a newer bundle."""
    titles = {item.name: item.title for item in classes}
    chosen_model = _selected_model_id(workspace, project_id, model_id)
    inference = _selected_inference(workspace, project_id, inference_id)
    # A selected inference reveals test spans only from its own action model.
    reveal_model = _model_of(inference) if inference is not None else chosen_model
    reveal = _test_file_exists(workspace, project_id, reveal_model)
    comparison = _comparison(intervals, blocks, video, titles, book, reveal_test=reveal)
    predicted = _predicted(workspace, inference, video, titles, book)
    evaluation = _evaluation_view(workspace, project_id, inference, book)
    final_enabled, final_reason, final_view = _final_view(
        workspace,
        project_id,
        inference,
        book,
    )
    return {
        "comparison": comparison["intervals"],
        "comparison_spans": comparison["spans"],
        "hide_test": comparison["hide_test"],
        "predicted": predicted["rows"],
        "predicted_spans": predicted["spans"],
        "unknown_share": predicted["unknown_share"],
        "inference_id": None if inference is None else inference["job_id"],
        "evaluation": evaluation,
        "final_enabled": final_enabled,
        "final_reason": final_reason,
        "final_view": final_view,
    }


def blocking_job(workspace: Path) -> object | None:
    """The queued, running, or cancel-requested job, if the lock is held."""
    from excavation_lab.jobs import reopen

    try:
        jobs = reopen(workspace)
    except LabError:
        return None
    for job in jobs:
        if job.status.value in _ACTIVE:
            return job
    return None


def _train_config(fields: Mapping[str, object]) -> object:
    from pydantic import ValidationError

    from excavation_lab.actions.dataset import WindowConfig
    from excavation_lab.actions.train import ActionTrainConfig

    epochs = _optional_int(fields.get("epochs"), _DEFAULT_EPOCHS)
    window_s = _optional_float(fields.get("window_s"), _DEFAULT_WINDOW_S)
    confidence = _optional_unit(fields.get("min_confidence"))
    duration = _optional_duration(fields.get("min_duration_s"))
    if confidence is None:
        confidence = _DEFAULT_CONFIDENCE
    if duration is None:
        duration = _DEFAULT_DURATION_S
    if epochs is None or window_s is None or confidence is False or duration is False:
        return ActionNote(
            text="Нужна конфигурация обучения действий.",
            code="invalid_action_train",
        )
    try:
        return ActionTrainConfig(
            epochs=epochs,
            window=WindowConfig(window_s=window_s),
            min_confidence=float(confidence),
            min_duration_s=float(duration),
        )
    except ValidationError:
        return ActionNote(
            text="Нужна конфигурация обучения действий.",
            code="invalid_action_train",
        )


def _custom_runs(
    workspace: Path,
    project_id: str,
    video_id: str,
    blocks: Sequence[RoleBlock],
    book: Catalog,
) -> list[dict[str, object]]:
    rows = storage.list_analysis_runs(
        workspace,
        project_id,
        video_id=video_id,
        origin="custom",
    )
    listed: list[dict[str, object]] = []
    for row in rows:
        if not row["completed"]:
            continue
        end_s = row["end_s"]
        start_s = float(row["start_s"])
        end = None if end_s is None else float(end_s)
        roles = book.not_calculated if end is None else _roles_touched(start_s, end, blocks, book)
        end_text = book.not_calculated if end is None else _fmt(end)
        label = (
            f"{row['model_id']} [{_fmt(start_s)}, {end_text}) · "
            f"{_fmt(float(row['analysis_fps']))} {book.action_run_fps} · {roles}"
        )
        listed.append(
            {
                "id": row["run_id"],
                "model_id": row["model_id"],
                "label": label,
                "selected": False,
            }
        )
    return listed


def _infer_run_options(
    runs: Sequence[Mapping[str, object]],
    detector_model_id: str | None,
    book: Catalog,
) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    for row in runs:
        same = detector_model_id is not None and row["model_id"] == detector_model_id
        options.append(
            {
                "id": row["id"],
                "label": row["label"],
                "disabled": not same,
                "note": "" if same else book.action_other_detector,
            }
        )
    return options


def _infer_reason(
    book: Catalog,
    model_id: str | None,
    runs: Sequence[Mapping[str, object]],
    detector_model_id: str | None,
    blocked: bool,
) -> str:
    if blocked:
        from excavation_lab.errors import JobAlreadyRunningError

        return JobAlreadyRunningError().message
    if model_id is None:
        return book.action_no_model
    if not runs:
        return book.action_no_runs
    if detector_model_id is None or not any(
        row["model_id"] == detector_model_id for row in runs
    ):
        return book.action_other_detector
    return ""


def _model_rows(
    workspace: Path,
    project_id: str,
    book: Catalog,
    model_id: str | None,
) -> list[dict[str, object]]:
    rows = storage.list_action_models(workspace, project_id)
    chosen = model_id if any(row["action_model_id"] == model_id for row in rows) else None
    if chosen is None and rows:
        chosen = rows[-1]["action_model_id"]
    listed: list[dict[str, object]] = []
    for row in rows:
        summary = _bundle_summary(Path(row["path"]))
        unstable = summary.get("selection_unstable")
        listed.append(
            {
                "id": row["action_model_id"],
                "detector_model_id": row["detector_model_id"],
                "selected": row["action_model_id"] == chosen,
                "support_train": _support_text(summary.get("support_train"), book),
                "support_validation": _support_text(summary.get("support_validation"), book),
                "classes_scored": _names_text(summary.get("classes_scored"), book),
                "selection_unstable": unstable if isinstance(unstable, bool) else None,
            }
        )
    return listed


def _model_history(
    workspace: Path,
    project_id: str,
    model_id: str | None,
) -> list[dict[str, object]]:
    if not model_id:
        return []
    for row in storage.list_action_models(workspace, project_id):
        if row["action_model_id"] == model_id:
            return _read_history(Path(row["path"]) / _HISTORY_NAME)
    return []


def _inference_rows(
    workspace: Path,
    project_id: str,
    book: Catalog,
) -> list[dict[str, object]]:
    from excavation_lab.jobs import get_job

    listed: list[dict[str, object]] = []
    for row in storage.list_action_inferences(workspace, project_id):
        try:
            status = get_job(workspace, project_id, row["job_id"]).status.value
            status_text = book.job_status_labels.get(status, status)
        except LabError:
            status = ""
            status_text = book.not_calculated
        listed.append(
            {
                "job_id": row["job_id"],
                "action_model_id": row["action_model_id"],
                "source_run_id": row["source_run_id"],
                "status": status,
                "status_text": status_text,
            }
        )
    return listed


def _selected_model_id(workspace: Path, project_id: str, model_id: str | None) -> str | None:
    rows = storage.list_action_models(workspace, project_id)
    if model_id and any(row["action_model_id"] == model_id for row in rows):
        return model_id
    if rows:
        return rows[-1]["action_model_id"]
    return None


def _selected_inference(
    workspace: Path,
    project_id: str,
    inference_id: str | None,
) -> Mapping[str, object] | None:
    rows = storage.list_action_inferences(workspace, project_id)
    if inference_id:
        for row in rows:
            if row["job_id"] == inference_id:
                return row
        return None
    if not rows:
        return None
    return rows[-1]


def _model_of(inference: Mapping[str, object] | None) -> str | None:
    if inference is None:
        return None
    model_id = inference.get("action_model_id")
    if isinstance(model_id, str) and model_id:
        return model_id
    return None


def _test_file_exists(workspace: Path, project_id: str, model_id: str | None) -> bool:
    directory = _model_dir(workspace, project_id, model_id)
    if directory is None:
        return False
    path = directory / _TEST_EVAL
    return path.is_file() and not path.is_symlink()


def _comparison(
    intervals: Sequence[ActionInterval],
    blocks: Sequence[RoleBlock],
    video: Mapping[str, object],
    titles: Mapping[str, str],
    book: Catalog,
    *,
    reveal_test: bool,
) -> dict[str, object]:
    duration_s = _positive(video.get("duration_s"))
    shown: list[dict[str, object]] = []
    hide_test = False
    for item in intervals:
        role = role_of(item, blocks)
        if role == "test" and not reveal_test:
            hide_test = True
            continue
        if role not in {"train", "validation", "test"}:
            continue
        shown.append(
            {
                "role": role,
                "role_label": split_label(role, book),
                "start_s": item.start_s,
                "end_s": item.end_s,
                "action_name": item.action_name,
                "title": titles.get(item.action_name, item.action_name),
                "left": _percent(item.start_s, duration_s),
                "width": _span_width(item.start_s, item.end_s, duration_s),
            }
        )
    return {"intervals": shown, "spans": shown, "hide_test": hide_test}


def _predicted(
    workspace: Path,
    inference: Mapping[str, object] | None,
    video: Mapping[str, object],
    titles: Mapping[str, str],
    book: Catalog,
) -> dict[str, object]:
    empty: dict[str, object] = {"rows": [], "spans": [], "unknown_share": None}
    if inference is None:
        return empty
    directory = Path(str(inference["path"]))
    if not _inside(workspace, directory):
        return empty
    rows = _read_interval_csv(directory / _INTERVALS_NAME, titles, book)
    document = _read_json_object(directory / _RUN_NAME)
    share = None if document is None else _json_number(document.get("unknown_share"))
    duration_s = _positive(video.get("duration_s"))
    spans: list[dict[str, object]] = []
    for row in rows:
        start = row.get("start_value")
        end = row.get("end_value")
        if not isinstance(start, float) or not isinstance(end, float):
            continue
        spans.append(
            {
                **row,
                "left": _percent(start, duration_s),
                "width": _span_width(start, end, duration_s),
            }
        )
    return {"rows": rows, "spans": spans, "unknown_share": share}


def _evaluation_view(
    workspace: Path,
    project_id: str,
    inference: Mapping[str, object] | None,
    book: Catalog,
) -> dict[str, object] | None:
    if inference is None:
        return None
    model_id = _model_of(inference)
    directory = _model_dir(workspace, project_id, model_id)
    if directory is None:
        return None
    path = directory / f"{_VALIDATION_PREFIX}{inference['job_id']}.json"
    document = _read_json_object(path)
    if document is None:
        return None
    return _evaluation_fields(document, book)


def _final_view(
    workspace: Path,
    project_id: str,
    inference: Mapping[str, object] | None,
    book: Catalog,
) -> tuple[bool, str, dict[str, object] | None]:
    if inference is None:
        return False, book.action_no_inference, None
    model_id = _model_of(inference)
    directory = _model_dir(workspace, project_id, model_id)
    if directory is None:
        return False, book.action_no_model, None
    test_path = directory / _TEST_EVAL
    test_document = _read_json_object(test_path) if _test_file_exists(
        workspace, project_id, model_id
    ) else None
    from excavation_lab.actions.evaluate import _FINAL_EXISTS, _match_validation

    if test_path.is_file() and not test_path.is_symlink():
        view = None if test_document is None else _evaluation_fields(test_document, book)
        return False, _FINAL_EXISTS, view
    thresholds = _inference_thresholds(workspace, inference)
    if thresholds is None:
        return False, book.action_predicted_empty, None
    _matched, message = _match_validation(
        workspace,
        project_id,
        directory,
        thresholds["action_model_id"],
        thresholds["min_confidence"],
        thresholds["min_duration_s"],
    )
    if _matched is None:
        return False, message, None
    return True, "", None


def _evaluation_fields(document: Mapping[str, object], book: Catalog) -> dict[str, object]:
    per_class = document.get("per_class")
    rows: list[dict[str, object]] = []
    if isinstance(per_class, Mapping):
        for name, score in per_class.items():
            if not isinstance(name, str) or not isinstance(score, Mapping):
                continue
            rows.append(
                {
                    "name": name,
                    "precision": metric_text(score.get("precision"), book),
                    "recall": metric_text(score.get("recall"), book),
                    "f1": metric_text(score.get("f1"), book),
                    "support_frames": count_text(score.get("support_frames"), book),
                    "support_intervals": count_text(score.get("support_intervals"), book),
                }
            )
    unstable = document.get("selection_unstable")
    return {
        "rows": rows,
        "accuracy": metric_text(document.get("accuracy"), book),
        "macro_f1": metric_text(document.get("macro_f1"), book),
        "classes_scored": _names_text(document.get("classes_scored"), book),
        "evaluated_frames": count_text(document.get("evaluated_frames"), book),
        "unlabelled_frames": count_text(document.get("unlabelled_frames"), book),
        "unknown_rate": metric_text(document.get("unknown_rate"), book),
        "selection_unstable": unstable if isinstance(unstable, bool) else None,
    }


def _inference_thresholds(
    workspace: Path,
    inference: Mapping[str, object],
) -> dict[str, object] | None:
    directory = Path(str(inference["path"]))
    if not _inside(workspace, directory):
        return None
    document = _read_json_object(directory / _RUN_NAME)
    if document is None:
        return None
    model_id = document.get("action_model_id")
    confidence = _json_number(document.get("min_confidence"))
    duration = _json_number(document.get("min_duration_s"))
    if not isinstance(model_id, str) or not model_id:
        return None
    if not isinstance(confidence, (int, float)) or not isinstance(duration, (int, float)):
        return None
    if isinstance(confidence, bool) or isinstance(duration, bool):
        return None
    return {
        "action_model_id": model_id,
        "min_confidence": float(confidence),
        "min_duration_s": float(duration),
    }


def _inference_dir(workspace: Path, project_id: str, job_id: str) -> Path | None:
    for row in storage.list_action_inferences(workspace, project_id):
        if row["job_id"] == job_id:
            return Path(row["path"])
    return None


def _model_dir(workspace: Path, project_id: str, model_id: str | None) -> Path | None:
    if not model_id:
        return None
    for row in storage.list_action_models(workspace, project_id):
        if row["action_model_id"] == model_id:
            return Path(row["path"])
    return None


def _bundle_summary(directory: Path) -> dict[str, object]:
    document = _read_json_object(directory / _BUNDLE_NAME)
    if document is None:
        return {}
    return document


def _read_interval_csv(
    path: Path,
    titles: Mapping[str, str],
    book: Catalog,
) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        raw_rows = list(reader)
        fieldnames = reader.fieldnames or ()
    except (OSError, csv.Error, UnicodeError):
        return []
    required = {"start_s", "end_s", "duration_s", "action_name", "mean_confidence"}
    if not required <= set(fieldnames):
        return []
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        name = (raw.get("action_name") or "").strip()
        start = _csv_float(raw.get("start_s"))
        end = _csv_float(raw.get("end_s"))
        rows.append(
            {
                "start": metric_text(start, book),
                "end": metric_text(end, book),
                "duration": metric_text(_csv_float(raw.get("duration_s")), book),
                "action_name": name,
                "title": name if name == _UNKNOWN else titles.get(name, name),
                "confidence": metric_text(_csv_float(raw.get("mean_confidence")), book),
                "unknown": name == _UNKNOWN,
                "start_value": start,
                "end_value": end,
            }
        )
    return rows


def _read_history(path: Path) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        raw_rows = list(reader)
    except (OSError, csv.Error, UnicodeError):
        return []
    history: list[dict[str, object]] = []
    for raw in raw_rows:
        epoch = _csv_float(raw.get("epoch"))
        loss = _csv_float(raw.get("train_loss"))
        score = _csv_float(raw.get("val_macro_f1"))
        if epoch is None or loss is None or score is None:
            continue
        history.append(
            {
                "epoch": int(epoch) if epoch.is_integer() else epoch,
                "train_loss": loss,
                "val_macro_f1": score,
            }
        )
    return history


def _read_json_object(path: Path) -> dict[str, object] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        if path.stat().st_size > 1_048_576:
            return None
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def metric_text(value: object, book: Catalog) -> str:
    """A finite measurement, or the catalog's missing-value sentence."""
    number = _json_number(value)
    if not isinstance(number, (int, float)) or isinstance(number, bool):
        return book.not_calculated
    return _fmt(float(number))


def count_text(value: object, book: Catalog) -> str:
    """An integer count. ``0`` stays ``0``. Missing stays uncalculated."""
    if isinstance(value, (bool, float)):
        return book.not_calculated
    if isinstance(value, int):
        return str(value)
    return book.not_calculated


def _support_text(value: object, book: Catalog) -> str:
    if not isinstance(value, Mapping) or not value:
        return book.not_calculated
    parts: list[str] = []
    for name in value:
        if not isinstance(name, str):
            continue
        count = value[name]
        if isinstance(count, bool) or not isinstance(count, int):
            parts.append(f"{name}: {book.not_calculated}")
        else:
            parts.append(f"{name}: {count}")
    if not parts:
        return book.not_calculated
    return ", ".join(parts)


def _names_text(value: object, book: Catalog) -> str:
    if not isinstance(value, list) or not value:
        return book.not_calculated
    names = [item for item in value if isinstance(item, str) and item]
    if not names:
        return book.not_calculated
    return ", ".join(names)


def _roles_touched(
    start_s: float,
    end_s: float,
    blocks: Sequence[RoleBlock],
    book: Catalog,
) -> str:
    labels: list[str] = []
    for block in blocks:
        if start_s < block.end_s and end_s > block.start_s:
            label = split_label(block.role, book)
            if label not in labels:
                labels.append(label)
    if not labels:
        return book.not_calculated
    return ", ".join(labels)


def _percent(start_s: float, duration_s: float | None) -> float | None:
    if duration_s is None:
        return None
    return max(0.0, start_s / duration_s * 100.0)


def _span_width(start_s: float, end_s: float, duration_s: float | None) -> float | None:
    if duration_s is None:
        return None
    return max(0.0, (end_s - start_s) / duration_s * 100.0)


def _fmt(value: float) -> str:
    return format(float(value), ".6f").rstrip("0").rstrip(".") or "0"


def _safe_token(value: object) -> bool:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        return False
    return not any(char in value for char in "/\\\x00")


def _inside(workspace: Path, path: Path) -> bool:
    root = Path(workspace).resolve()
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def _optional_int(value: object, default: int) -> int | None:
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 1:
        return None
    return number


def _optional_float(value: object, default: float) -> float | None:
    if value is None or value == "":
        return default
    number = _json_number(value) if not isinstance(value, str) else _csv_float(value)
    if not isinstance(number, float) or number <= 0:
        return None
    return number


def _optional_unit(value: object) -> float | None | bool:
    """``None`` when omitted. ``False`` when the text is not a unit interval."""
    if value is None or value == "":
        return None
    number = _csv_float(value) if isinstance(value, str) else _json_number(value)
    if not isinstance(number, float) or number < 0 or number > 1:
        return False
    return number


def _optional_duration(value: object) -> float | None | bool:
    if value is None or value == "":
        return None
    number = _csv_float(value) if isinstance(value, str) else _json_number(value)
    if not isinstance(number, float) or number <= 0:
        return False
    return number


def _json_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _csv_float(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number
