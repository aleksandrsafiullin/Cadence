"""Design-contract locks: theme default, portable CSS, copy, and chart roles."""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

from excavation_lab.ui import charts, events_page
from excavation_lab.ui import copy as ui_copy
from excavation_lab.ui.theme import read_theme_file, resolve_theme, theme_script_lines
from excavation_lab.ui.tokens import LAB_THEME_CSS

REPO_ROOT = Path(__file__).resolve().parents[2]
_DESIGN = REPO_ROOT / "docs" / "lab_design_code.md"
_HEX = re.compile(r"#[0-9A-Fa-f]{3,8}\b")
_CSS_FENCE = re.compile(r"```css\n(.*?)```", re.DOTALL)
_PAGE_TITLES = (
    ("PAGE_VIDEO", "Видео"),
    ("PAGE_DATA", "Данные и разметка"),
    ("PAGE_TRAINING", "Обучение"),
    ("PAGE_ANALYSIS", "Анализ"),
    ("PAGE_QUALITY", "Качество"),
    ("PAGE_EVENTS", "События"),
)
_JOB_STATUSES = (
    "queued",
    "running",
    "completed",
    "failed",
    "cancel_requested",
    "cancelled",
)
_CHART_CALLS = (
    "loss_chart",
    "ap50_chart",
    "line_chart",
    "bar_chart",
    "altair_chart",
    "plotly_chart",
)


def _first_css_fence(markdown: str) -> str:
    match = _CSS_FENCE.search(markdown)
    assert match is not None
    return match.group(1)


def _hexes(text: str) -> set[str]:
    return set(_HEX.findall(text))


def _load_test(name: str) -> ModuleType:
    path = REPO_ROOT / "tests" / "unit" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function_source(name: str) -> str:
    source = Path(charts.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(name)


def test_missing_file_and_session_resolve_dark(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    session: dict[str, object] = {}
    assert not (workspace / "ui" / "theme.txt").exists()
    assert "lab_theme" not in session
    assert read_theme_file(workspace) is None
    assert resolve_theme(workspace, session) == "dark"
    assert "lab_theme" not in session
    assert not (workspace / "ui").exists()


def test_light_file_is_used_on_a_fresh_session(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ui = workspace / "ui"
    ui.mkdir(parents=True)
    (ui / "theme.txt").write_text("light\n", encoding="utf-8")
    session: dict[str, object] = {}
    assert resolve_theme(workspace, session) == "light"
    assert session["lab_theme"] == "light"


@pytest.mark.parametrize("text", ("stopping", "stopped", "idle", ""))
def test_invalid_preference_file_resolves_dark(tmp_path: Path, text: str) -> None:
    workspace = tmp_path / "ws"
    ui = workspace / "ui"
    ui.mkdir(parents=True)
    (ui / "theme.txt").write_text(text, encoding="utf-8")
    session: dict[str, object] = {}
    assert read_theme_file(workspace) is None
    assert resolve_theme(workspace, session) == "dark"
    assert "lab_theme" not in session


def test_portable_css_is_the_design_fence() -> None:
    fence = _first_css_fence(_DESIGN.read_text(encoding="utf-8"))
    portable = LAB_THEME_CSS.read_text(encoding="utf-8")
    assert fence in portable
    assert '[data-theme="dark"]' in portable
    assert '[data-theme="light"]' in portable
    assert "--pdm-accent" in portable
    assert "--lab-accent" not in portable
    assert "--excavation-accent" not in portable
    assert _hexes(portable) <= _hexes(fence)
    assert _hexes(fence)


def test_chrome_css_has_no_hex_and_no_wow_page() -> None:
    css = (
        REPO_ROOT / "src" / "excavation_lab" / "web" / "static" / "lab.css"
    ).read_text(encoding="utf-8")
    assert _HEX.search(css) is None
    assert "wow_page" not in css
    assert "var(--pdm-text)" in css


def test_page_titles_job_keys_and_display_caption() -> None:
    for name, title in _PAGE_TITLES:
        assert getattr(ui_copy, name) == title
    assert list(ui_copy.JOB_STATUS_LABELS) == list(_JOB_STATUSES)
    assert ui_copy.THEME_DISPLAY_ONLY == "Меняет только оформление."


def test_events_page_renders_unavailable_without_the_chart_styler(tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    from excavation_lab.web.app import create_app

    source = Path(events_page.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)
    assert "excavation_lab.ui.charts" not in modules
    for name in _CHART_CALLS:
        assert name not in calls
    from excavation_lab.datasets.catalog import list_catalogs
    from excavation_lab.storage import create_project
    from excavation_lab.web.preferences import write_project_file

    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/events")
    assert page.status_code == 200
    assert "lab-chart" not in page.text
    assert 'id="ap50-chart"' not in page.text
    assert "vega-lite" not in page.text
    assert ui_copy.EVENTS_UNAVAILABLE in page.text or "недоступен" in page.text


def test_stylesheet_skip_does_not_skip_the_metric_zero(tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    from excavation_lab.web.app import create_app

    states = _load_test("test_ui_empty_states")
    css = (
        REPO_ROOT / "src" / "excavation_lab" / "web" / "static" / "lab.css"
    ).read_text(encoding="utf-8")
    assert "margin: 0" in css
    from excavation_lab.datasets.catalog import list_catalogs
    from excavation_lab.storage import create_project
    from excavation_lab.web.preferences import write_project_file

    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/events")
    assert page.status_code == 200
    values = re.findall(
        r'<div class="lab-metric">\s*<span>([^<]*)</span>\s*<b>([^<]*)</b>',
        page.text,
    )
    assert values
    for _label, value in values:
        assert states._zero_token(value) is None
        assert states._NUMERIC_METRIC.match(value.strip()) is None
    assert "margin: 0" not in "\n".join(value for _label, value in values)
    assert states._zero_token("0") == "0"
    assert states._NUMERIC_METRIC.match("0") is not None
    honest = (REPO_ROOT / "tests" / "unit" / "test_ui_empty_states.py").read_text(
        encoding="utf-8"
    )
    assert "for metric in app.metric:" not in honest


def test_loss_stays_observed_and_ap50_stays_accent() -> None:
    loss = _function_source("loss_chart")
    bars = _function_source("ap50_chart")
    loss_mark = _function_source("_loss_mark")
    bar_mark = _function_source("_ap50_mark")
    assert '"--pdm-series-observed"' in loss
    assert "--pdm-series-forecast" not in loss
    assert 'colors["--pdm-series-observed"]' in loss_mark
    assert "--pdm-series-forecast" not in loss_mark
    assert '"--pdm-accent"' in bars
    assert "--pdm-series-forecast" not in bars
    assert 'colors["--pdm-accent"]' in bar_mark
    assert "--pdm-series-forecast" not in bar_mark


def test_current_page_script_remarks_when_href_arrives() -> None:
    script = "\n".join(theme_script_lines("dark"))
    assert script.count("requestAnimationFrame") == 1
    assert script.count("location.pathname") == 1
    assert "MutationObserver" in script
    assert 'attributeFilter: ["href"]' in script
    assert "aria-current" not in script.split("attributeFilter", 1)[1]
    assert "window." not in script
    assert "root.labNavMark = observer;" in script


def test_streamlit_import_stays_in_app_and_ui() -> None:
    scan = _load_test("test_no_streamlit_import")
    roots = (
        REPO_ROOT / "src" / "excavation_lab" / "ui",
        REPO_ROOT / "src" / "excavation_lab" / "web",
    )
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for root in roots
        for path in root.rglob("*.py")
        if scan._imports_streamlit(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
