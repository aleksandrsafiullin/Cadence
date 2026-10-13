"""Training routes. Page helpers stay in ``ui.training_page``."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.routing import Route

from excavation_lab import storage
from excavation_lab.errors import JobAlreadyRunningError, LabError
from excavation_lab.jobs import worker_job
from excavation_lab.schemas import (
    PROFILE_PILOT_SMALL,
    PROFILE_SMOKE,
    JobRecord,
    JobStatus,
)
from excavation_lab.ui import training_page
from excavation_lab.ui.copy import Catalog, catalog
from excavation_lab.ui.locale import resolve_locale
from excavation_lab.ui.status import status_tone
from excavation_lab.ui.video_page import job_status_line
from excavation_lab.web.preferences import choose_project_id, write_flash

_FORM_PROFILES = (PROFILE_SMOKE, PROFILE_PILOT_SMALL)
_DEVICES = ("auto", "cuda", "mps", "cpu")
_VARIANTS = ("nano", "small")
_ACTIVE = frozenset(
    {
        JobStatus.queued,
        JobStatus.running,
        JobStatus.cancel_requested,
    }
)
_JSON_KEYS = (
    "variant",
    "resolution",
    "epochs",
    "batch_size",
    "gradient_accumulation",
    "learning_rate",
    "encoder_learning_rate",
    "workers",
    "seed",
)


def training_routes() -> list[Route]:
    return [
        Route("/training/start", post_training_start, methods=["POST"]),
        Route("/training/cancel", post_training_cancel, methods=["POST"]),
    ]


def training_context(
    request: Request,
    base: dict[str, object],
    flash: dict[str, str],
) -> dict[str, object]:
    """Template fields for ``training.html``. Start stays in ``request_training``."""
    workspace = _workspace(request)
    book = catalog(str(base.get("lang") or "ru"))
    project_raw = base.get("project_id")
    project_id = project_raw if isinstance(project_raw, str) and project_raw else None
    shared: dict[str, object] = {
        "flash_error": flash.get("error"),
        "flash_notice": flash.get("notice"),
        "list_error": None,
        "show_empty": project_id is None,
        "show_form": False,
        "book": book,
    }
    if project_id is None:
        return shared
    shared["show_form"] = True
    shared["show_empty"] = False
    shared.update(_form_context(request, workspace, project_id, book, flash))
    return shared


async def post_training_start(request: Request) -> RedirectResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    form = await request.form()
    if project_id is None:
        _remember(request, {"error": book.need_project})
        return _redirect()
    profile = _text(form.get("profile"))
    if profile not in _FORM_PROFILES:
        _remember(request, {"error": "unknown_profile: Неизвестный профиль обучения."})
        return _redirect()
    try:
        jobs = training_page.reopen(workspace)
    except LabError as exc:
        _remember(request, {"error": training_page.format_error(exc.code, exc.message)})
        return _redirect()
    attempt = training_page.request_training(
        workspace,
        project_id,
        _train_form(workspace, project_id, form, profile),
        [job.status for job in jobs],
        book,
    )
    _remember(request, _attempt_payload(attempt))
    return _redirect()


async def post_training_cancel(request: Request) -> RedirectResponse:
    """Cancel the active job id. A posted PID is not a signal target."""
    workspace = _workspace(request)
    book = _book(workspace)
    form = await request.form()
    try:
        jobs = training_page.reopen(workspace)
    except LabError as exc:
        _remember(request, {"error": training_page.format_error(exc.code, exc.message)})
        return _redirect()
    blocking = _blocking(jobs)
    if blocking is None:
        return _redirect()
    posted = _text(form.get("job_id"))
    if posted and posted != blocking.job_id:
        return _redirect()
    try:
        owner, _stored, _record = worker_job(workspace, blocking.job_id)
    except LabError as exc:
        _remember(request, {"error": training_page.format_error(exc.code, exc.message)})
        return _redirect()
    attempt = training_page.request_training_cancel(
        workspace,
        owner,
        blocking.job_id,
        book,
    )
    _remember(request, _attempt_payload(attempt))
    return _redirect()


def _form_context(
    request: Request,
    workspace: Path,
    project_id: str,
    book: Catalog,
    flash: dict[str, str],
) -> dict[str, object]:
    jobs: list[JobRecord] = []
    list_error = None
    try:
        jobs = list(training_page.reopen(workspace))
    except LabError as exc:
        list_error = training_page.format_error(exc.code, exc.message)
    versions: list[str] = []
    try:
        versions = [
            item.dataset_id
            for item in training_page.list_dataset_versions(workspace, project_id)
        ]
    except LabError as exc:
        list_error = training_page.format_error(exc.code, exc.message)
    profile_name = _requested_profile(request)
    fields = training_page.profile_field_values(profile_name)
    blocking = _blocking(jobs)
    enabled = training_page.start_enabled([job.status for job in jobs])
    pairs = training_page._train_pairs(workspace, project_id, jobs)
    focus = pairs[-1] if pairs else None
    code = "dataset_not_ready" if _flash_not_ready(flash.get("error")) else None
    view = training_page.TrainingLogView.absent()
    checkpoint: Path | None = None
    if focus is not None and code != "dataset_not_ready":
        job, run_dir = focus
        view = training_page._view_for_run(workspace, project_id, job.job_id, run_dir)
        checkpoint = training_page._checkpoint_file(
            workspace,
            project_id,
            job.job_id,
            run_dir,
        )
    shown = blocking if blocking is not None else (focus[0] if focus else None)
    epoch = None if not view.readable else view.latest_epoch
    eta = None if not view.readable else view.eta_s
    cancel_id = None
    if blocking is not None and training_page._job_run_dir(
        workspace,
        project_id,
        blocking.job_id,
    ):
        cancel_id = blocking.job_id
    return {
        "list_error": list_error,
        "versions": versions,
        "not_ready": (
            None
            if versions
            else training_page.format_error("dataset_not_ready", book.dataset_not_ready)
        ),
        "profile": profile_name,
        "profile_options": [
            {"id": name, "label": _profile_label(book, name)} for name in _FORM_PROFILES
        ],
        "profile_help": _profile_label(book, profile_name),
        "variant": str(fields["variant"]),
        "devices": _DEVICES,
        "resolution_value": _shown_number(fields["resolution"]),
        "epochs_value": _shown_number(fields["epochs"]),
        "batch_value": _shown_number(fields["batch_size"]),
        "accumulation_value": _shown_number(fields["gradient_accumulation"]),
        "lr_value": _shown_number(fields["learning_rate"]),
        "encoder_lr_value": _shown_number(fields["encoder_learning_rate"]),
        "workers_value": _shown_number(fields["workers"]),
        "seed_value": _shown_number(fields["seed"]),
        "profiles": _browser_profiles(),
        "start_enabled": enabled,
        "blocked_message": None if enabled else JobAlreadyRunningError().message,
        "cancel_job_id": cancel_id,
        "status_line": (
            job_status_line(shown, book)
            if shown is not None
            else training_page.format_status(None, book)
        ),
        "status_tone": status_tone(shown.status.value if shown is not None else None),
        "poll": shown is not None and shown.status in _ACTIVE,
        "status_value": shown.status.value if shown is not None else "",
        "status_job_id": shown.job_id if shown is not None else "",
        "epoch_text": training_page.format_epoch(epoch, book),
        "checkpoint_text": training_page.format_checkpoint(checkpoint, book),
        "eta_text": training_page.format_eta(eta, book),
        "log_broken": None if view.readable else book.train_log_broken,
        "log_empty": (
            None if training_page.should_draw_chart(view, code) else book.training_empty
        ),
        "job_rows": [
            {"tone": status_tone(job.status.value), "line": job_status_line(job, book)}
            for job, _run_dir in pairs
        ],
    }


def _train_form(
    workspace: Path,
    project_id: str,
    form: object,
    profile: str,
) -> training_page.TrainForm:
    defaults = training_page.profile_field_values(profile)
    variant = _text(_field(form, "variant"))
    if variant not in _VARIANTS:
        variant = str(defaults["variant"])
    device = _text(_field(form, "device"))
    if device not in _DEVICES:
        device = "auto"
    return training_page.TrainForm(
        dataset_id=_dataset_id(workspace, project_id, _field(form, "dataset_id")),
        variant=variant,
        profile=profile,
        device=device,
        resolution=_int(_field(form, "resolution"), int(defaults["resolution"]), minimum=1),
        epochs=_int(_field(form, "epochs"), int(defaults["epochs"]), minimum=1),
        batch_size=_int(_field(form, "batch_size"), int(defaults["batch_size"]), minimum=1),
        gradient_accumulation=_int(
            _field(form, "gradient_accumulation"),
            int(defaults["gradient_accumulation"]),
            minimum=1,
        ),
        learning_rate=_float(_field(form, "learning_rate"), float(defaults["learning_rate"])),
        encoder_learning_rate=_float(
            _field(form, "encoder_learning_rate"),
            float(defaults["encoder_learning_rate"]),
        ),
        workers=_int(_field(form, "workers"), int(defaults["workers"]), minimum=0),
        seed=_int(_field(form, "seed"), int(defaults["seed"]), minimum=0),
    )


def _dataset_id(workspace: Path, project_id: str, raw: object) -> str | None:
    text = _text(raw)
    if not text:
        return None
    try:
        known = {
            item.dataset_id
            for item in training_page.list_dataset_versions(workspace, project_id)
        }
    except LabError:
        return None
    if text in known:
        return text
    return None


def _attempt_payload(attempt: training_page.TrainAttempt) -> dict[str, str]:
    if attempt.error:
        return {"error": training_page.format_error(attempt.code, attempt.error)}
    if not attempt.notice:
        return {}
    text = attempt.notice
    if attempt.job_id:
        text = f"{text} {attempt.job_id}."
    return {"notice": text}


def _browser_profiles() -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    for name in _FORM_PROFILES:
        fields = training_page.profile_field_values(name)
        found[name] = {key: fields[key] for key in _JSON_KEYS}
    return found


def _requested_profile(request: Request) -> str:
    raw = request.query_params.get("profile")
    if isinstance(raw, str) and raw in _FORM_PROFILES:
        return raw
    return PROFILE_SMOKE


def _profile_label(book: Catalog, name: str) -> str:
    if name == PROFILE_SMOKE:
        return book.smoke_help
    return book.pilot_help


def _blocking(jobs: Sequence[JobRecord]) -> JobRecord | None:
    for job in jobs:
        if job.status in _ACTIVE:
            return job
    return None


def _flash_not_ready(text: str | None) -> bool:
    return isinstance(text, str) and text.startswith("dataset_not_ready:")


def _shown_number(value: object) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def _project_id(workspace: Path) -> str | None:
    try:
        projects = storage.list_projects(workspace)
    except LabError:
        return None
    rows = [{"project_id": item["project_id"], "name": item["name"]} for item in projects]
    return choose_project_id(workspace, rows)


def _book(workspace: Path) -> Catalog:
    return catalog(resolve_locale(workspace, session={}))


def _workspace(request: Request) -> Path:
    return Path(request.app.state.workspace)


def _remember(request: Request, payload: dict[str, str]) -> None:
    try:
        write_flash(_workspace(request), payload)
    except Exception:  # noqa: BLE001 — the page still has to show the sentence
        text = payload.get("error")
        if isinstance(text, str) and text:
            request.app.state.pending_notice = text


def _redirect() -> RedirectResponse:
    return RedirectResponse("/training", status_code=303)


def _field(form: object, name: str) -> object:
    getter = getattr(form, "get", None)
    if not callable(getter):
        return None
    return getter(name)


def _text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _int(value: object, fallback: int, *, minimum: int) -> int:
    if isinstance(fallback, bool) or not isinstance(fallback, int):
        fallback = minimum
    if not isinstance(value, str):
        return fallback
    try:
        number = int(value.strip())
    except ValueError:
        return fallback
    if number < minimum:
        return fallback
    return number


def _float(value: object, fallback: float) -> float:
    if isinstance(fallback, bool) or not isinstance(fallback, (int, float)):
        return 0.0
    default = float(fallback)
    if not isinstance(value, str):
        return default
    try:
        number = float(value.strip())
    except ValueError:
        return default
    if not math.isfinite(number) or number <= 0:
        return default
    return number
