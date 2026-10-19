"""«Обучение». Start and cancel go through the train service.

The page does not import RF-DETR and does not construct a detector.
``request_training`` is the only enqueue. The metrics log is the loss series.
A header-only log is not a curve.
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
from excavation_lab.errors import JobAlreadyRunningError, LabError
from excavation_lab.jobs import reopen, request_cancel, run_directory
from excavation_lab.models.train import enqueue_train
from excavation_lab.schemas import (
    PROFILE_OVERFIT,
    PROFILE_PILOT_SMALL,
    PROFILE_SMOKE,
    JobRecord,
    JobStatus,
    load_pilot,
)
from excavation_lab.ui import copy as ui_copy

_ACTIVE = frozenset(
    {
        JobStatus.queued,
        JobStatus.running,
        JobStatus.cancel_requested,
    }
)
_PROFILES = (PROFILE_SMOKE, PROFILE_PILOT_SMALL)
_PROFILE_NAMES = frozenset({PROFILE_SMOKE, PROFILE_PILOT_SMALL, PROFILE_OVERFIT})
_DEVICES = ("auto", "cuda", "mps", "cpu")
_VARIANTS = (("Nano", "nano"), ("Small", "small"))
_VARIANT_BY_LABEL = dict(_VARIANTS)
_LABEL_BY_VARIANT = {value: label for label, value in _VARIANTS}
_LOSS_COLUMNS = ("train/loss_epoch", "train/loss", "train_loss", "loss")
_DURATION_COLUMNS = (
    "epoch_duration_s",
    "duration_s",
    "epoch_time_s",
    "step_duration_s",
)
_MIN_ETA_EPOCHS = 3
_MAX_HISTORY_BYTES = 1_048_576

# The page server reopens jobs through this module.
__all__ = ["reopen"]


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    dataset_id: str


@dataclass(frozen=True, slots=True)
class TrainForm:
    dataset_id: str | None
    variant: str
    profile: str
    device: str
    resolution: int
    epochs: int
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    encoder_learning_rate: float
    workers: int
    seed: int


@dataclass(frozen=True, slots=True)
class TrainAttempt:
    """One Start or Cancel click. ``job_id`` is set only after the lock.

    ``notice_id`` and ``error_id`` are stable. The page looks up catalog
    sentences at display time and does not keep a translated sentence to
    compare later.
    """

    job_id: str | None = None
    notice: str | None = None
    error: str | None = None
    code: str | None = None
    notice_id: str | None = None
    error_id: str | None = None


@dataclass(frozen=True, slots=True)
class TrainingLogView:
    """Metrics copied from the metrics log. Missing stays unknown."""

    show_chart: bool
    losses: tuple[float, ...]
    epoch_marks: tuple[int, ...]
    latest_epoch: int | None
    measured_epochs: int
    eta_s: float | None
    readable: bool = True

    @staticmethod
    def absent() -> TrainingLogView:
        return TrainingLogView(False, (), (), None, 0, None, True)

    @staticmethod
    def broken() -> TrainingLogView:
        return TrainingLogView(False, (), (), None, 0, None, False)


def start_enabled(statuses: Sequence[JobStatus]) -> bool:
    """False while a queued, running, or cancel-requested job holds the lock."""
    return not any(status in _ACTIVE for status in statuses)


def should_draw_chart(view: TrainingLogView, error_code: str | None) -> bool:
    """A not-ready dataset never gets a curve, including a sample one."""
    if error_code == "dataset_not_ready":
        return False
    return view.show_chart and view.readable


def training_log_view(
    text: str | None,
    *,
    planned_epochs: int | None,
) -> TrainingLogView:
    """Chart rows and ETA from one CSV body.

    A missing body or a header with no data rows does not chart. ETA needs
    at least three positive epoch durations and a planned epoch count.
    """
    if text is None:
        return TrainingLogView.absent()
    body = text.lstrip("\ufeff")
    if not body.strip():
        return TrainingLogView.absent()
    try:
        reader = csv.DictReader(io.StringIO(body))
        fieldnames = reader.fieldnames
        rows = list(reader)
    except csv.Error:
        return TrainingLogView.broken()
    if not fieldnames:
        return TrainingLogView.absent()

    epoch_column = _named_column(fieldnames, ("epoch",))
    loss_column = _loss_column(fieldnames)
    duration_column = _duration_column(fieldnames)
    losses: list[float] = []
    marks: list[int] = []
    durations: list[float] = []
    latest: int | None = None
    seen = 0
    for row in rows:
        if _blank_row(row):
            continue
        seen += 1
        epoch = _epoch_value(row.get(epoch_column) if epoch_column else None)
        if epoch is None:
            epoch = seen
        latest = epoch
        if loss_column is not None:
            loss = _finite(row.get(loss_column))
            if loss is not None:
                marks.append(epoch)
                losses.append(loss)
        if duration_column is not None:
            duration = _finite(row.get(duration_column))
            if duration is not None and duration > 0:
                durations.append(duration)
    planned = _planned_epochs(planned_epochs)
    return TrainingLogView(
        show_chart=bool(losses),
        losses=tuple(losses),
        epoch_marks=tuple(marks),
        latest_epoch=latest,
        measured_epochs=len(durations),
        eta_s=_eta(durations, planned),
        readable=True,
    )


def form_settings(form: TrainForm) -> dict[str, object]:
    """Profile JSON plus the advanced fields. ``run_test`` stays off."""
    base = load_pilot().train_profile(form.profile).model_dump(mode="json")
    base["variant"] = form.variant
    base["resolution"] = form.resolution
    base["epochs"] = form.epochs
    base["batch_size"] = form.batch_size
    base["gradient_accumulation"] = form.gradient_accumulation
    base["learning_rate"] = form.learning_rate
    base["encoder_learning_rate"] = form.encoder_learning_rate
    base["workers"] = form.workers
    base["seed"] = form.seed
    base["run_test"] = False
    return base


def request_training(
    workspace: Path,
    project_id: str | None,
    form: TrainForm,
    statuses: Sequence[JobStatus],
    text: ui_copy.Catalog | None = None,
) -> TrainAttempt:
    """Enqueue one train job, or return the refusal without a second start.

    No catalog argument stays Russian. A service ``LabError`` message stays
    the service text. Page refusals keep a stable ``error_id``.
    """
    catalog = _resolved(text)
    if not project_id:
        return TrainAttempt(error=catalog.need_project, error_id="need_project")
    if not start_enabled(statuses):
        running = JobAlreadyRunningError()
        return TrainAttempt(code=running.code, error=running.message)
    if not form.dataset_id:
        return TrainAttempt(
            code="dataset_not_ready",
            error=catalog.dataset_not_ready,
            error_id="dataset_not_ready",
        )
    try:
        record = enqueue_train(
            workspace,
            project_id,
            form.dataset_id,
            form.profile,
            device=form.device,
            profile_override=form_settings(form),
        )
    except LabError as exc:
        return TrainAttempt(code=exc.code, error=exc.message)
    return TrainAttempt(
        job_id=record.job_id,
        notice=catalog.train_queued,
        notice_id="queued",
    )


def request_training_cancel(
    workspace: Path,
    project_id: str,
    job_id: str,
    text: ui_copy.Catalog | None = None,
) -> TrainAttempt:
    """``request_cancel`` for the job that holds the workspace lock.

    No catalog argument stays Russian. A service ``LabError`` message stays
    the service text.
    """
    catalog = _resolved(text)
    try:
        record = request_cancel(workspace, project_id, job_id)
    except LabError as exc:
        return TrainAttempt(code=exc.code, error=exc.message)
    return TrainAttempt(
        job_id=record.job_id,
        notice=catalog.train_cancelled,
        notice_id="cancelled",
    )


def list_dataset_versions(workspace: Path, project_id: str) -> list[DatasetVersion]:
    db_path = storage.project_sqlite(workspace, project_id)
    with storage.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT dataset_id FROM dataset_versions
            WHERE project_id = ?
            ORDER BY created_at, dataset_id
            """,
            (project_id,),
        ).fetchall()
    versions: list[DatasetVersion] = []
    for row in rows:
        dataset_id = row["dataset_id"]
        if isinstance(dataset_id, str) and dataset_id:
            versions.append(DatasetVersion(dataset_id))
    return versions


def format_error(code: str | None, message: str) -> str:
    if code and not message.startswith(f"{code}:"):
        return f"{code}: {message}"
    return message


def missing_snapshot_message(
    workspace: Path,
    project_id: str,
    text: ui_copy.Catalog | None = None,
) -> str:
    """Why training cannot start before a snapshot exists.

    The sentence is the data-page gate reason when that gate is closed, so
    the two pages name the same missing part. An open gate with no snapshot
    yet does not claim that reviewed frames are missing.
    """
    from excavation_lab.ui.data_page import snapshot_gate

    catalog = _resolved(text)
    gate = snapshot_gate(workspace, project_id, catalog)
    if gate.enabled:
        return catalog.snapshot_not_built
    if gate.reason:
        return gate.reason
    return catalog.dataset_not_ready


def format_epoch(epoch: int | None, text: ui_copy.Catalog | None = None) -> str:
    """No catalog argument stays Russian."""
    catalog = _resolved(text)
    if epoch is None:
        return catalog.not_calculated
    return str(epoch)


def format_eta(eta_s: float | None, text: ui_copy.Catalog | None = None) -> str:
    """Missing ETA is the catalog phrase, never ``0``.

    No catalog argument stays Russian: ``не рассчитано`` or ``{number} с``.
    """
    catalog = _resolved(text)
    if eta_s is None or isinstance(eta_s, bool) or not isinstance(eta_s, (int, float)):
        return catalog.not_calculated
    if not math.isfinite(float(eta_s)) or float(eta_s) < 0:
        return catalog.not_calculated
    return f"{float(eta_s):.1f} {catalog.seconds_suffix}"


def format_checkpoint(path: Path | None, text: ui_copy.Catalog | None = None) -> str:
    """No catalog argument stays Russian."""
    catalog = _resolved(text)
    if path is None:
        return catalog.not_calculated
    return str(path)


def format_status(status: JobStatus | None, text: ui_copy.Catalog | None = None) -> str:
    """No catalog argument stays Russian. Status keys stay the stored ids."""
    catalog = _resolved(text)
    if status is None:
        return catalog.not_calculated
    return catalog.job_status_labels.get(status.value, catalog.not_calculated)


def _resolved(text: ui_copy.Catalog | None) -> ui_copy.Catalog:
    if text is not None:
        return text
    return ui_copy.catalog("ru")


def profile_field_values(name: str) -> dict[str, object]:
    """Advanced fields ``_apply_profile`` writes. An unknown name uses smoke."""
    chosen = name if name in _PROFILES else PROFILE_SMOKE
    profile = load_pilot().train_profile(chosen)
    variant = profile.variant.value
    return {
        "profile": chosen,
        "variant": variant,
        "variant_label": _LABEL_BY_VARIANT.get(variant, "Nano"),
        "resolution": profile.resolution,
        "epochs": profile.epochs,
        "batch_size": profile.batch_size,
        "gradient_accumulation": profile.gradient_accumulation,
        "learning_rate": profile.learning_rate,
        "encoder_learning_rate": profile.encoder_learning_rate,
        "workers": profile.workers,
        "seed": profile.seed,
    }


def _train_pairs(
    workspace: Path,
    project_id: str | None,
    jobs: Sequence[JobRecord],
) -> list[tuple[JobRecord, Path]]:
    if not project_id:
        return []
    pairs: list[tuple[JobRecord, Path]] = []
    for job in jobs:
        run_dir = _job_run_dir(workspace, project_id, job.job_id)
        if run_dir is None:
            continue
        request = run_dir / "train_request.json"
        try:
            if request.is_file() and not request.is_symlink():
                pairs.append((job, run_dir))
        except OSError:
            continue
    return pairs


def _view_for_run(
    workspace: Path,
    project_id: str,
    job_id: str,
    run_dir: Path,
) -> TrainingLogView:
    request = _read_request(run_dir)
    profile = request.get("profile") if request else None
    text, readable = _history_text(workspace, project_id, job_id, run_dir, profile)
    if not readable:
        return TrainingLogView.broken()
    return training_log_view(text, planned_epochs=_planned_from_request(request, profile))


def _history_text(
    workspace: Path,
    project_id: str,
    job_id: str,
    run_dir: Path,
    profile: object,
) -> tuple[str | None, bool]:
    path = _history_file(workspace, project_id, job_id, run_dir, profile)
    if path is None:
        return None, True
    try:
        if path.is_symlink() or not path.is_file():
            return None, True
        size = path.stat().st_size
    except OSError:
        return None, False
    if size <= 0:
        return None, True
    if size > _MAX_HISTORY_BYTES:
        return None, False
    try:
        return path.read_text(encoding="utf-8"), True
    except (OSError, UnicodeDecodeError):
        return None, False


def _history_file(
    workspace: Path,
    project_id: str,
    job_id: str,
    run_dir: Path,
    profile: object,
) -> Path | None:
    candidates = [run_dir / "metrics_history.csv", run_dir / "metrics.csv"]
    bundle = _bundle_dir(workspace, project_id, profile, job_id)
    if bundle is not None:
        candidates.append(bundle / "metrics_history.csv")
    # Prefer the first regular file that already has a data row. A header-only
    # metrics_history.csv must not hide the CSVLogger file written during the run.
    fallback: Path | None = None
    for path in candidates:
        kind = _csv_kind(path)
        if kind == "skip":
            continue
        if kind == "rows":
            return path
        if fallback is None:
            fallback = path
    return fallback


def _checkpoint_file(
    workspace: Path,
    project_id: str,
    job_id: str,
    run_dir: Path,
) -> Path | None:
    request = _read_request(run_dir)
    profile = request.get("profile") if request else None
    candidates = [run_dir / "weights.pth"]
    bundle = _bundle_dir(workspace, project_id, profile, job_id)
    if bundle is not None:
        candidates.append(bundle / "weights.pth")
    try:
        if run_dir.is_dir() and not run_dir.is_symlink():
            candidates.extend(sorted(run_dir.glob("*.ckpt")))
    except OSError:
        pass
    return _first_file(candidates)


def _bundle_dir(
    workspace: Path,
    project_id: str,
    profile: object,
    job_id: str,
) -> Path | None:
    if not isinstance(profile, str) or profile not in _PROFILE_NAMES:
        return None
    if not job_id.isalnum():
        return None
    model_id = f"{profile}-{job_id}"
    try:
        return storage.recorded_model_directory(workspace, project_id, model_id)
    except LabError:
        return None


def _planned_from_request(request: Mapping[str, object] | None, profile: object) -> int | None:
    if request is not None:
        override = request.get("profile_override")
        if isinstance(override, dict):
            parsed = _planned_epochs(override.get("epochs"))
            if parsed is not None:
                return parsed
    if not isinstance(profile, str) or profile not in _PROFILE_NAMES:
        return None
    try:
        return load_pilot().train_profile(profile).epochs
    except LabError:
        return None


def _read_request(run_dir: Path) -> dict[str, object] | None:
    path = run_dir / "train_request.json"
    try:
        if path.is_symlink() or not path.is_file():
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    return document


def _first_file(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        try:
            if path.is_symlink() or not path.is_file():
                continue
            if path.stat().st_size <= 0:
                continue
        except OSError:
            continue
        return path
    return None


def _job_run_dir(workspace: Path, project_id: str, job_id: str) -> Path | None:
    try:
        run_dir = run_directory(workspace, project_id, job_id)
    except LabError:
        return None
    try:
        if run_dir.is_dir() and not run_dir.is_symlink():
            return run_dir
    except OSError:
        return None
    return None


def _blocking_job(jobs: Sequence[JobRecord]) -> JobRecord | None:
    for job in jobs:
        if job.status in _ACTIVE:
            return job
    return None


def _eta(durations: Sequence[float], planned_epochs: int | None) -> float | None:
    if planned_epochs is None or len(durations) < _MIN_ETA_EPOCHS:
        return None
    remaining = max(planned_epochs - len(durations), 0)
    return (sum(durations) / len(durations)) * remaining


def _planned_epochs(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _named_column(fieldnames: Sequence[str], names: Sequence[str]) -> str | None:
    lookup = {name.strip().lower(): name for name in fieldnames if name and name.strip()}
    for candidate in names:
        found = lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def _csv_kind(path: Path) -> str:
    """``rows`` when the file has a data row, ``empty`` when it is only a header."""
    try:
        if path.is_symlink() or not path.is_file():
            return "skip"
        size = path.stat().st_size
    except OSError:
        return "skip"
    if size <= 0 or size > _MAX_HISTORY_BYTES:
        return "empty"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "empty"
    if _body_has_data_row(text):
        return "rows"
    return "empty"


def _body_has_data_row(text: str) -> bool:
    body = text.lstrip("\ufeff")
    if not body.strip():
        return False
    try:
        reader = csv.DictReader(io.StringIO(body))
    except csv.Error:
        return False
    if not reader.fieldnames:
        return False
    try:
        return any(not _blank_row(row) for row in reader)
    except csv.Error:
        return False


def _loss_column(fieldnames: Sequence[str]) -> str | None:
    found = _named_column(fieldnames, _LOSS_COLUMNS)
    if found is not None:
        return found
    named = [name for name in fieldnames if name]
    train = [name for name in named if "loss" in name.lower() and "train" in name.lower()]
    if train:
        return train[0]
    any_loss = [name for name in named if "loss" in name.lower()]
    if any_loss:
        return any_loss[0]
    return None


def _duration_column(fieldnames: Sequence[str]) -> str | None:
    found = _named_column(fieldnames, _DURATION_COLUMNS)
    if found is not None:
        return found
    for name in fieldnames:
        if name and "duration" in name.lower():
            return name
    return None


def _blank_row(row: Mapping[str, str | None]) -> bool:
    return all(not (value or "").strip() for value in row.values())


def _epoch_value(value: str | None) -> int | None:
    number = _finite(value)
    if number is None or number < 0:
        return None
    epoch = int(number)
    if float(epoch) != number:
        return None
    return epoch


def _finite(value: str | None) -> float | None:
    if value is None:
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
