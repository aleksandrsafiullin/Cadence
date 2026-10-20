"""v0.2 events gate: schemas only, indicators unread, CLI refuses."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import pkgutil
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel
from starlette.testclient import TestClient
from typer.testing import CliRunner

from excavation_lab import analytics, storage
from excavation_lab.analytics import EventRow, TruckVisitRow, unavailable_error
from excavation_lab.cli import app
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.ui import copy as ui_copy
from excavation_lab.web.preferences import write_project_file

_RUNNER = CliRunner()
_ZERO_TOKEN = re.compile(r"(?<![\d.])0(?:[.,]0+)?%?(?![\d.])")
_FORBIDDEN_FUNCTION = re.compile(
    r"unload|bucket_count|truck_visit|rule_score|anchor|censored|kpi|propose|export",
    re.IGNORECASE,
)
_EVENT_COLUMNS = (
    "event_id",
    "run_id",
    "video_id",
    "view_id",
    "excavator_id",
    "truck_visit_id_nullable",
    "event_type",
    "start_s",
    "end_s",
    "anchor_s",
    "inference_status",
    "rule_score_nullable",
    "review_status",
    "evidence_observation_ids",
    "left_censored",
    "right_censored",
    "note",
)
_VISIT_COLUMNS = (
    "truck_visit_id",
    "video_id",
    "excavator_id_nullable",
    "linked_track_ids",
    "physical_vehicle_id_nullable",
    "first_seen_s",
    "in_position_s_nullable",
    "loading_start_s_nullable",
    "loading_end_s_nullable",
    "departure_s_nullable",
    "last_seen_s",
    "confirmed_bucket_count",
    "estimated_bucket_count",
    "left_censored",
    "right_censored",
    "identity_uncertain",
    "review_status",
)
_STORAGE_SPIES = (
    "excavation_lab.storage.project_sqlite",
    "excavation_lab.storage.create_project",
    "excavation_lab.storage.list_projects",
    "excavation_lab.storage.project_directory",
    "excavation_lab.storage.get_analysis_run",
    "excavation_lab.storage.connect",
    "excavation_lab.storage.default_workspace_root",
    "excavation_lab.cli.create_project",
    "excavation_lab.cli.default_workspace_root",
    "excavation_lab.cli._workspace",
)
_ABSENT = (
    "tons",
    "fill factor",
    "volume",
    "t/h",
    "тонн",
    "наполнен",
    "объём",
    "объем",
    "т/ч",
)


def _analytics_modules() -> list[object]:
    modules = [analytics]
    for info in pkgutil.iter_modules(analytics.__path__, analytics.__name__ + "."):
        modules.append(importlib.import_module(info.name))
    return modules


def _zero_token(text: str) -> str | None:
    match = _ZERO_TOKEN.search(text)
    if match is None:
        return None
    return match.group(0)


def _is_zero(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value in (0, 0.0) and not isinstance(
        node.value, bool
    )


def _returns_zero(fn: ast.AST) -> bool:
    for child in ast.walk(fn):
        if isinstance(child, ast.Return) and _is_zero(child.value):
            return True
    return False


def test_analytics_has_no_indicator_function() -> None:
    public: list[str] = []
    for module in _analytics_modules():
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_"):
                continue
            if not getattr(obj, "__module__", "").startswith("excavation_lab.analytics"):
                continue
            public.append(f"{module.__name__}.{name}")
            signature = inspect.signature(obj)
            required = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ]
            if required:
                continue
            result = obj()
            assert result != 0
            assert result != 0.0
            assert not isinstance(result, int | float)
        for obj in vars(module).values():
            if not isinstance(obj, type) or not obj.__module__.startswith(
                "excavation_lab.analytics"
            ):
                continue
            if issubclass(obj, BaseModel):
                assert obj.__doc__ is not None
                assert "not implemented in v0.1" in obj.__doc__
            for name, member in vars(obj).items():
                if name.startswith("_"):
                    continue
                assert not isinstance(member, property | staticmethod | classmethod)
                assert not inspect.isfunction(member)
    assert public == ["excavation_lab.analytics.unavailable_error"]
    error = unavailable_error()
    assert error.code == "unavailable"
    assert error.message == "События и показатели в версии 0.1 не рассчитываются."
    package = Path(analytics.__file__).resolve().parent
    for path in package.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "streamlit" not in source
        assert ".to_csv" not in source
        assert "csv.writer" not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                assert _FORBIDDEN_FUNCTION.search(node.name) is None, node.name
                assert not _returns_zero(node), node.name
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in {"confirmed_bucket_count", "estimated_bucket_count"}
            ):
                assert not _is_zero(node.value)


def test_event_schemas_leave_counts_unknown() -> None:
    assert tuple(EventRow.model_fields) == _EVENT_COLUMNS
    assert tuple(TruckVisitRow.model_fields) == _VISIT_COLUMNS
    event = EventRow(
        event_id="event",
        run_id="run",
        video_id="video",
        view_id="view",
        excavator_id="excavator",
    )
    visit = TruckVisitRow(truck_visit_id="visit", video_id="video")
    assert event.rule_score_nullable is None
    assert event.anchor_s is None
    assert event.start_s is None
    assert event.end_s is None
    assert event.truck_visit_id_nullable is None
    assert visit.confirmed_bucket_count is None
    assert visit.estimated_bucket_count is None
    assert visit.first_seen_s is None
    assert visit.last_seen_s is None
    assert visit.left_censored is None
    assert visit.right_censored is None
    for row in (event, visit):
        values = row.model_dump().values()
        assert 0 not in values
        assert 0.0 not in values
    assert EventRow.model_fields["rule_score_nullable"].default is None
    assert TruckVisitRow.model_fields["confirmed_bucket_count"].default is None
    assert TruckVisitRow.model_fields["estimated_bucket_count"].default is None


def test_events_cli_refuses_before_storage(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    workspace = storage.default_workspace_root()
    before = _event_csvs(workspace)

    def spy(label: str):
        def inner(*_args, **_kwargs):
            calls.append(label)
            raise AssertionError(label)

        return inner

    for target in _STORAGE_SPIES:
        monkeypatch.setattr(target, spy(target))
    monkeypatch.chdir(tmp_path)
    flag_sets = (
        [],
        ["--project", "pilot"],
        ["--run", "RUN"],
        ["--project", "pilot", "--run", "RUN"],
    )
    for command in ("propose", "export"):
        for flags in flag_sets:
            result = _RUNNER.invoke(app, ["events", command, *flags, "--json"])
            assert result.exit_code != 0, (command, flags, result.stdout, result.stderr)
            body = json.loads(result.stdout)
            assert body["ok"] is False
            assert body["error"]["code"] == "unavailable"
            assert body["error"]["message"] == unavailable_error().message
        plain = _RUNNER.invoke(
            app,
            ["events", command, "--project", "pilot", "--run", "RUN"],
        )
        assert plain.exit_code != 0
        assert "unavailable" in plain.stderr
    assert calls == []
    assert list(tmp_path.rglob("events.csv")) == []
    assert list(tmp_path.rglob("truck_visits.csv")) == []
    assert _event_csvs(workspace) == before


def test_events_page_shows_uncomputed_indicators(tmp_path: Path) -> None:
    from excavation_lab.analytics import unavailable_error
    from excavation_lab.web.app import create_app

    list_catalogs(tmp_path)
    storage.create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/events")
    assert response.status_code == 200
    text = response.text
    _assert_events_gate(text, ui_copy.catalog("ru"))
    assert "недоступен" in text
    assert ui_copy.NOT_CALCULATED in text
    assert ui_copy.EVENTS_V02_CAPTION in text
    assert "v0.2" in text
    assert _zero_token(text) is None, text
    message = unavailable_error().message
    assert text.count(message) == 1
    folded = text.casefold()
    for token in _ABSENT:
        assert token not in folded
    assert "propose" not in folded
    assert "export" not in folded
    _assert_events_source()

    ui = tmp_path / "ui"
    ui.mkdir(exist_ok=True)
    (ui / "locale.txt").write_text("en\n", encoding="utf-8")
    book = ui_copy.catalog("en")
    with TestClient(create_app(tmp_path)) as client:
        english = client.get("/events")
    assert english.status_code == 200
    body = english.text
    _assert_events_gate(body, book)
    assert unavailable_error().message not in body
    assert book.events_unavailable in body
    assert book.events_unavailable_caption in body
    assert book.events_v02_caption in body
    assert _zero_token(body) is None, body
    leftover = body.replace("Русский", "")
    assert _CYRILLIC.search(leftover) is None, leftover
    main = re.search(r'<main class="lab-main" id="content">(.*)</main>', body, re.DOTALL)
    assert main is not None
    assert _CYRILLIC.search(main.group(1)) is None


_CYRILLIC = re.compile(r"[\u0400-\u04FF]")


def _assert_events_gate(text: str, book: ui_copy.Catalog) -> None:
    labels = book.event_indicator_labels
    assert len(labels) == 12
    assert text.count('class="lab-metric"') == 12
    for label in labels:
        snippet = f"<span>{label}</span><b>{book.not_calculated}</b>"
        assert snippet in text
    lowered = text.lower()
    assert "<video" not in lowered
    assert "<canvas" not in lowered
    assert "lab-chart" not in text
    assert "<svg" not in lowered


def _assert_events_source() -> None:
    from excavation_lab.ui import events_page

    source = Path(events_page.__file__).read_text(encoding="utf-8")
    assert "st.button" not in source
    for control in (
        "Confirm",
        "Reject",
        "Correct time",
        "Assign truck visit",
        "Add missed event",
    ):
        assert control not in source
    template = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "excavation_lab"
        / "web"
        / "templates"
        / "events.html"
    )
    markup = template.read_text(encoding="utf-8")
    assert "unavailable_error" not in markup
    folded = markup.casefold()
    assert "propose" not in folded
    assert "export" not in folded
    assert "<video" not in folded
    assert "<canvas" not in folded


def test_core_does_not_import_analytics() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "excavation_lab"
    allowed = {
        root / "cli.py",
        root / "ui" / "events_page.py",
        root / "analytics" / "__init__.py",
        root / "analytics" / "models.py",
    }
    for path in root.rglob("*.py"):
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        assert "excavation_lab.analytics" not in text, path


def _event_csvs(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return set(root.rglob("events.csv")) | set(root.rglob("truck_visits.csv"))


def _nodes(app_test: Any) -> list[object]:
    return [*app_test.main, *app_test.sidebar]


def _strings(app_test: Any) -> list[str]:
    found: list[str] = []
    for node in _nodes(app_test):
        for attr in ("value", "label", "help", "title"):
            raw = getattr(node, attr, None)
            if isinstance(raw, str) and raw and "--pdm-bg" not in raw:
                found.append(raw)
    return found


def _joined(app_test: Any) -> str:
    return "\n".join(_strings(app_test))


def test_strings_skip_stylesheet_values_and_keep_captions() -> None:
    css = "background: var(--pdm-bg); margin: 0;"
    app = SimpleNamespace(
        main=[
            SimpleNamespace(
                value=css,
                label="Подтверждённые ковши",
                help=None,
                title=None,
            )
        ],
        sidebar=[
            SimpleNamespace(
                value=ui_copy.EVENTS_V02_CAPTION,
                label=None,
                help=None,
                title=None,
            )
        ],
    )
    found = _strings(app)
    assert css not in found
    assert "Подтверждённые ковши" in found
    joined = _joined(app)
    assert css not in joined
    assert ui_copy.EVENTS_V02_CAPTION in joined
