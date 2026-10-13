"""Quality routes. Page helpers stay in ``ui.quality_page``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.routing import Route

from excavation_lab import storage
from excavation_lab.errors import LabError
from excavation_lab.ui import quality_page
from excavation_lab.ui.copy import Catalog, catalog
from excavation_lab.ui.locale import resolve_locale
from excavation_lab.web.preferences import choose_project_id, write_flash

_RUN_COOKIE = "lab_quality_run"
_RUN_A_COOKIE = "lab_quality_run_a"
_RUN_B_COOKIE = "lab_quality_run_b"
_DELTA_COOKIE = "lab_quality_delta"


def quality_routes() -> list[Route]:
    return [
        Route("/quality/compare", post_quality_compare, methods=["POST"]),
        Route("/quality/label", post_quality_label, methods=["POST"]),
    ]


def quality_context(
    request: Request,
    base: dict[str, object],
    flash: dict[str, str],
) -> dict[str, object]:
    """Template fields for ``quality.html``. Compare and label stay on POST."""
    workspace = _workspace(request)
    book = catalog(str(base.get("lang") or "ru"))
    project_raw = base.get("project_id")
    project_id = project_raw if isinstance(project_raw, str) and project_raw else None
    body = _blank(book, flash)
    if project_id is None:
        return body
    try:
        runs = quality_page.list_metric_runs(workspace, project_id)
    except LabError as exc:
        body["list_error"] = quality_page._localized_error(exc, book)
        return body
    if not runs:
        report, error = _load(workspace, project_id, None, book)
        if error:
            body["list_error"] = error
        body.update(quality_page.report_fields(report, book))
        body["ap50_spec"] = quality_page.ap50_chart_spec(report, workspace, book)
        return body
    eval_id = _picked(
        request.query_params.get("quality_run"),
        request.cookies.get(_RUN_COOKIE),
        runs,
        0,
    )
    report, error = _load(workspace, project_id, eval_id, book)
    if error:
        body["list_error"] = error
    show_compare = len(runs) >= 2
    run_a = _picked(request.cookies.get(_RUN_A_COOKIE), None, runs, 0) if show_compare else None
    run_b = _picked(request.cookies.get(_RUN_B_COOKIE), None, runs, 1) if show_compare else None
    body.update(quality_page.report_fields(report, book))
    body["ap50_spec"] = quality_page.ap50_chart_spec(report, workspace, book)
    body.update(
        {
            "show_empty": False,
            "show_compare": show_compare,
            "compare_same": show_compare and run_a == run_b,
            "runs": runs,
            "run_options": _options(runs, eval_id),
            "run_a_options": _options(runs, run_a),
            "run_b_options": _options(runs, run_b),
            "eval_id": eval_id,
            "delta_metrics": _delta(request, workspace, project_id, runs, run_a, run_b, book, flash),
            "quality_run_cookie": eval_id,
            "quality_run_a_cookie": run_a,
            "quality_run_b_cookie": run_b,
        }
    )
    return body


async def post_quality_compare(request: Request) -> RedirectResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    form = await request.form()
    if project_id is None:
        _remember(request, {"error": book.quality_empty})
        return _redirect(delta=False)
    try:
        runs = quality_page.list_metric_runs(workspace, project_id)
    except LabError as exc:
        _remember(request, {"error": quality_page._localized_error(exc, book)})
        return _redirect(delta=False)
    eval_id = _picked(form.get("quality_run"), None, runs, 0) if runs else None
    run_a = _token(form.get("quality_run_a"))
    run_b = _token(form.get("quality_run_b"))
    if run_a not in runs or run_b not in runs:
        missing = run_a if run_a not in runs else run_b
        _remember(request, {"error": book.eval_not_found_message(missing or "")})
        return _redirect(eval_id=eval_id, delta=False)
    if run_a == run_b:
        return _redirect(eval_id=eval_id, run_a=run_a, run_b=run_b, delta=False)
    try:
        quality_page.compare(workspace, project_id, run_a, run_b)
    except LabError as exc:
        _remember(request, {"error": quality_page._localized_error(exc, book)})
        return _redirect(eval_id=eval_id, run_a=run_a, run_b=run_b, delta=False)
    return _redirect(eval_id=eval_id, run_a=run_a, run_b=run_b, delta=True)


async def post_quality_label(request: Request) -> RedirectResponse:
    workspace = _workspace(request)
    book = _book(workspace)
    project_id = _project_id(workspace)
    form = await request.form()
    if project_id is None:
        _remember(request, {"error": book.quality_empty})
        return _redirect()
    try:
        runs = quality_page.list_metric_runs(workspace, project_id)
    except LabError as exc:
        _remember(request, {"error": quality_page._localized_error(exc, book)})
        return _redirect()
    eval_id = _token(form.get("quality_run"))
    frame_id = form.get("frame_id")
    frame_text = frame_id if isinstance(frame_id, str) else ""
    if eval_id not in runs or not frame_text:
        return _redirect(eval_id=eval_id if eval_id in runs else None)
    try:
        report = quality_page.load_quality_report(workspace, project_id, eval_id)
    except LabError as exc:
        _remember(request, {"error": quality_page._localized_error(exc, book)})
        return _redirect(eval_id=eval_id)
    row = _row_for_frame(report, frame_text)
    if row is None or not quality_page.send_enabled(row.get("split")):
        return _redirect(eval_id=eval_id)
    try:
        queued = quality_page.send_to_labeling(
            workspace,
            project_id,
            frame_text,
            eval_id=eval_id,
        )
    except LabError as exc:
        _remember(request, {"error": quality_page._localized_error(exc, book)})
        return _redirect(eval_id=eval_id)
    shown = queued.get("frame_id") if isinstance(queued, Mapping) else None
    label = shown if isinstance(shown, str) and shown else frame_text
    _remember(request, {"notice": book.sent_to_labeling_message(label)})
    return _redirect(eval_id=eval_id)


def _blank(book: Catalog, flash: dict[str, str]) -> dict[str, object]:
    body: dict[str, object] = {
        "book": book,
        "flash_error": flash.get("error"),
        "flash_notice": flash.get("notice"),
        "list_error": None,
        "show_empty": True,
        "show_compare": False,
        "compare_same": False,
        "runs": [],
        "run_options": [],
        "run_a_options": [],
        "run_b_options": [],
        "eval_id": None,
        "delta_metrics": [],
        "quality_run_cookie": None,
        "quality_run_a_cookie": None,
        "quality_run_b_cookie": None,
        "ap50_spec": None,
    }
    body.update(quality_page.report_fields(quality_page._unknown_report(), book))
    return body


def _load(
    workspace: Path,
    project_id: str,
    eval_id: str | None,
    book: Catalog,
) -> tuple[dict[str, Any], str | None]:
    try:
        return quality_page.load_quality_report(workspace, project_id, eval_id), None
    except LabError as exc:
        return quality_page._unknown_report(), quality_page._localized_error(exc, book)


def _delta(
    request: Request,
    workspace: Path,
    project_id: str,
    runs: list[str],
    run_a: str | None,
    run_b: str | None,
    book: Catalog,
    flash: dict[str, str],
) -> list[dict[str, str]]:
    if flash.get("error"):
        return []
    if request.cookies.get(_DELTA_COOKIE) != "1":
        return []
    if run_a not in runs or run_b not in runs or run_a == run_b:
        return []
    try:
        result = quality_page.compare(workspace, project_id, run_a, run_b)
    except LabError:
        return []
    if not isinstance(result, Mapping):
        return []
    return quality_page.delta_fields(result, book)


def _row_for_frame(report: Mapping[str, Any], frame_id: str) -> Mapping[str, Any] | None:
    errors = report.get("errors")
    if not isinstance(errors, list):
        return None
    for row in errors:
        if isinstance(row, Mapping) and row.get("frame_id") == frame_id:
            return row
    return None


def _options(runs: list[str], selected: str | None) -> list[dict[str, object]]:
    return [{"id": item, "selected": item == selected} for item in runs]


def _picked(primary: object, secondary: object, runs: list[str], index: int) -> str:
    """A listed id, otherwise the run at ``index``. A stale id does not stick."""
    for candidate in (primary, secondary):
        token = _token(candidate)
        if token in runs:
            return token
    return runs[index]


def _token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or _unsafe(text):
        return None
    return text


def _unsafe(value: str) -> bool:
    if value in {".", ".."}:
        return True
    return any(char in value for char in "/\\\x00\r\n;")


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


def _redirect(
    *,
    eval_id: str | None = None,
    run_a: str | None = None,
    run_b: str | None = None,
    delta: bool | None = None,
) -> RedirectResponse:
    response = RedirectResponse("/quality", status_code=303)
    _stamp(response, _RUN_COOKIE, eval_id)
    _stamp(response, _RUN_A_COOKIE, run_a)
    _stamp(response, _RUN_B_COOKIE, run_b)
    if delta is True:
        _stamp(response, _DELTA_COOKIE, "1")
    elif delta is False:
        response.delete_cookie(_DELTA_COOKIE, path="/")
    return response


def _stamp(response: RedirectResponse, name: str, value: str | None) -> None:
    if not isinstance(value, str) or not value or _unsafe(value):
        return
    response.set_cookie(name, value, path="/", httponly=True, samesite="lax")
