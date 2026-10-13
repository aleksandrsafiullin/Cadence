"""Token-driven loss and AP50 charts. Synthetic rows stay in this file."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from excavation_lab.ui import charts, quality_page, training_page
from excavation_lab.ui.charts import ap50_chart, loss_chart
from excavation_lab.ui.tokens import theme_custom_properties

REPO_ROOT = Path(__file__).resolve().parents[2]
_UI = REPO_ROOT / "src" / "excavation_lab" / "ui"
_VIZ = REPO_ROOT / "src" / "excavation_lab" / "visualization"
_LOSS_EPOCHS = (1, 2, 3)
_LOSS_VALUES = (0.137, 0.864, 0.421)
_AP50_ROWS = (
    {"класс": "Экскаватор", "AP50": 0.258},
    {"класс": "Ковш", "AP50": 0.613},
)
_BANNED_THEMES = ("plotly_dark", "plotly", "ggplot2")
_OTHER_ROLES = ("--pdm-series-forecast", "--pdm-series-3", "--pdm-success", "--pdm-series-band")


def test_loss_and_ap50_follow_parsed_tokens() -> None:
    for theme_name in ("dark", "light"):
        props = theme_custom_properties(theme_name)
        loss = loss_chart(_LOSS_EPOCHS, _LOSS_VALUES, theme_name).to_dict()
        bars = ap50_chart(_AP50_ROWS, theme_name).to_dict()
        _assert_chrome(loss, props)
        _assert_chrome(bars, props)
        assert loss["mark"]["type"] == "line"
        assert loss["mark"]["color"] == props["--pdm-series-observed"]
        assert loss["mark"]["strokeWidth"] == 2
        assert "strokeDash" not in loss["mark"]
        assert "layer" not in loss
        assert [row["loss"] for row in loss["data"]["values"]] == list(_LOSS_VALUES)
        assert bars["mark"]["type"] == "bar"
        assert bars["mark"]["color"] == props["--pdm-accent"]
        assert bars["mark"]["color"] != props["--pdm-series-3"]
        assert bars["mark"]["color"] != props["--pdm-success"]
        assert "color" not in bars["encoding"]
        assert [row["AP50"] for row in bars["data"]["values"]] == [
            row["AP50"] for row in _AP50_ROWS
        ]
        blob = json.dumps({"loss": loss, "bars": bars})
        for name in _BANNED_THEMES:
            assert name not in blob
    dark = theme_custom_properties("dark")
    light = theme_custom_properties("light")
    assert dark["--pdm-series-observed"] != light["--pdm-series-observed"]
    assert dark["--pdm-accent"] != light["--pdm-accent"]


def test_ap50_uses_accent_not_the_forecast_token(monkeypatch: pytest.MonkeyPatch) -> None:
    real = charts.theme_custom_properties

    def wrapped(theme: str) -> dict[str, str]:
        props = dict(real(theme))
        props["--pdm-series-forecast"] = "forecast-token"
        props["--pdm-series-3"] = "series-3-token"
        props["--pdm-success"] = "success-token"
        return props

    monkeypatch.setattr(charts, "theme_custom_properties", wrapped)
    spec = ap50_chart(_AP50_ROWS, "dark").to_dict()
    blob = json.dumps(spec)
    assert spec["mark"]["color"] == real("dark")["--pdm-accent"]
    assert "forecast-token" not in blob
    assert "series-3-token" not in blob
    assert "success-token" not in blob
    source = Path(charts.__file__).read_text(encoding="utf-8")
    for token in _OTHER_ROLES:
        assert token not in source


def test_loss_has_no_forecast_or_band(monkeypatch: pytest.MonkeyPatch) -> None:
    real = charts.theme_custom_properties

    def wrapped(theme: str) -> dict[str, str]:
        props = dict(real(theme))
        props["--pdm-series-forecast"] = "forecast-token"
        props["--pdm-series-band"] = "band-token"
        props["--pdm-series-3"] = "series-3-token"
        return props

    monkeypatch.setattr(charts, "theme_custom_properties", wrapped)
    spec = loss_chart(_LOSS_EPOCHS, _LOSS_VALUES, "light").to_dict()
    blob = json.dumps(spec)
    assert spec["mark"]["color"] == real("light")["--pdm-series-observed"]
    assert spec["mark"]["strokeWidth"] == 2
    assert "forecast-token" not in blob
    assert "band-token" not in blob
    assert "series-3-token" not in blob
    assert "layer" not in spec


def test_missing_token_fails_the_chart(monkeypatch: pytest.MonkeyPatch) -> None:
    real = charts.theme_custom_properties

    def wrapped(theme: str) -> dict[str, str]:
        props = dict(real(theme))
        del props["--pdm-chart-bg"]
        return props

    monkeypatch.setattr(charts, "theme_custom_properties", wrapped)
    with pytest.raises(KeyError, match="--pdm-chart-bg"):
        loss_chart((1,), (0.137,), "dark")


def test_unknown_theme_fails() -> None:
    with pytest.raises(ValueError, match="unknown theme"):
        ap50_chart(_AP50_ROWS, "stopping")


def test_building_a_chart_restores_the_altair_theme() -> None:
    import altair as alt

    before = alt.theme.active
    loss_chart(_LOSS_EPOCHS, _LOSS_VALUES, "dark")
    assert alt.theme.active == before


def test_named_theme_is_disabled_only_while_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import altair as alt

    before = alt.theme.active
    seen: list[str] = []
    real_enable = alt.theme.enable

    def spy(name: str | None = None, **options: object) -> object:
        result = real_enable(name, **options)
        seen.append(alt.theme.active)
        return result

    monkeypatch.setattr(alt.theme, "enable", spy)
    ap50_chart(_AP50_ROWS, "light")
    assert seen == ["none"]
    assert alt.theme.active == before
    tree = ast.parse(Path(charts.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                assert child.func.attr != "enable"


def test_charts_and_tokens_do_not_import_streamlit() -> None:
    for name in ("charts.py", "tokens.py"):
        source = (_UI / name).read_text(encoding="utf-8")
        assert _imports_streamlit(source) is False


def test_only_training_and_quality_import_charts() -> None:
    progress = REPO_ROOT / "src" / "excavation_lab" / "web" / "routes.py"
    quality = _UI / "quality_page.py"
    training = _UI / "training_page.py"
    assert _imports_charts(progress.read_text(encoding="utf-8"))
    assert _imports_charts(quality.read_text(encoding="utf-8"))
    assert not _imports_charts(training.read_text(encoding="utf-8"))
    for path in sorted(_UI.glob("*.py")):
        if path.name in {"quality_page.py", "charts.py"}:
            continue
        assert not _imports_charts(path.read_text(encoding="utf-8")), path.name
    for path in sorted(_VIZ.rglob("*.py")):
        assert not _imports_charts(path.read_text(encoding="utf-8")), path.name
    web = REPO_ROOT / "src" / "excavation_lab" / "web"
    for path in sorted(web.rglob("*.py")):
        if path.name == "routes.py":
            continue
        assert not _imports_charts(path.read_text(encoding="utf-8")), path.name
    analysis = (_UI / "analysis_page.py").read_text(encoding="utf-8")
    events = (_UI / "events_page.py").read_text(encoding="utf-8")
    assert not _imports_charts(analysis)
    assert not _imports_charts(events)


def test_pages_call_the_styler_only_behind_the_existing_gates() -> None:
    training = Path(training_page.__file__).read_text(encoding="utf-8")
    quality = Path(quality_page.__file__).read_text(encoding="utf-8")
    progress_source = (
        REPO_ROOT / "src" / "excavation_lab" / "web" / "routes.py"
    ).read_text(encoding="utf-8")
    assert "0.137" not in training
    assert "0.864" not in training
    assert "0.258" not in quality
    assert "0.613" not in quality
    assert "st.line_chart" not in training
    assert "st.bar_chart" not in quality
    assert "st.altair_chart" not in training
    assert "st.altair_chart" not in quality
    assert "st.altair_chart" not in progress_source
    progress = _function(ast.parse(progress_source), "get_job_progress")
    gate = _if_calling(progress, "should_draw_chart")
    assert _calls(gate.body, "loss_chart")
    assert not _calls(gate.body, "altair_chart")
    assert not _calls(gate.orelse, "loss_chart")
    assert not _calls(gate.orelse, "altair_chart")
    shown = _if_frame(_function(ast.parse(quality), "ap50_chart_spec"))
    assert _calls(shown.body, "ap50_chart")
    assert not _calls(shown.body, "ap50_chart_frame")
    assert not _calls(shown.body, "altair_chart")
    assert not _calls(shown.orelse, "ap50_chart")
    assert not _calls(shown.orelse, "altair_chart")
    missing = _function(ast.parse(quality), "ap50_chart_frame")
    skipped = _if_name_is_none(missing, "number")
    assert not _calls(skipped.body, "ap50_chart")
    assert not _calls(skipped.body, "altair_chart")


def test_building_a_chart_does_not_import_or_draw_events() -> None:
    events = (_UI / "events_page.py").read_text(encoding="utf-8")
    assert not _imports_charts(events)
    assert "altair_chart" not in events
    assert "loss_chart" not in events
    assert "ap50_chart" not in events
    script = """
import sys
from excavation_lab.ui.charts import ap50_chart, loss_chart
loss_chart((1, 2), (0.137, 0.864), "dark")
ap50_chart(({"класс": "ковш", "AP50": 0.421},), "light")
assert "excavation_lab.ui.events_page" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_vega_container_uses_surface_tokens() -> None:
    css = (
        REPO_ROOT / "src" / "excavation_lab" / "web" / "static" / "lab.css"
    ).read_text(encoding="utf-8")
    start = css.index("#loss-chart")
    block = css[start : css.index("\n}", start) + 2]
    body = block[block.index("{") + 1 :]
    assert "var(--pdm-surface)" in block
    assert "var(--pdm-border)" in block
    assert "border-radius: 12px" in block
    assert "#" not in body


def _assert_chrome(spec: dict[str, object], props: dict[str, str]) -> None:
    blob = json.dumps(spec)
    for name in _BANNED_THEMES:
        assert name not in blob
    assert "title" not in spec
    padding = spec["padding"]
    assert isinstance(padding, dict)
    assert padding["left"] == 56
    assert padding["right"] == 16
    assert padding["bottom"] == 48
    assert padding["top"] == 16
    config = spec["config"]
    assert isinstance(config, dict)
    assert config["font"] == props["--pdm-font"]
    axis = config["axis"]
    assert isinstance(axis, dict)
    assert axis["labelFontSize"] == 12
    assert axis["labelColor"] == props["--pdm-chart-text"]
    legend = config["legend"]
    assert isinstance(legend, dict)
    assert legend["disable"] is True
    encoding = spec["encoding"]
    assert isinstance(encoding, dict)
    assert "legend" not in encoding
    x_axis = encoding["x"]["axis"]
    y_axis = encoding["y"]["axis"]
    assert x_axis["grid"] is False
    assert y_axis["grid"] is True
    assert x_axis["domainWidth"] == 1
    assert y_axis["domainWidth"] == 1
    assert x_axis["domainColor"] == props["--pdm-chart-axis"]
    assert y_axis["domainColor"] == props["--pdm-chart-grid"]
    assert y_axis["gridColor"] == props["--pdm-chart-grid"]
    assert config["axisX"]["grid"] is False
    assert config["axisY"]["grid"] is True
    assert spec["background"] == props["--pdm-chart-bg"]
    view = spec["view"]
    assert isinstance(view, dict)
    assert view["fill"] == props["--pdm-chart-bg"]
    hover = spec["usermeta"]["hover"]
    assert hover["background"] == props["--pdm-surface"]
    assert hover["border"] == props["--pdm-border"]
    assert hover["text"] == props["--pdm-text"]
    assert hover["fontSize"] == 12


def _imports_streamlit(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "streamlit" or alias.name.startswith("streamlit."):
                    return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if module == "streamlit" or module.startswith("streamlit."):
                return True
    return False


def _imports_charts(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "excavation_lab.ui.charts" or node.module.endswith(".charts"):
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "excavation_lab.ui.charts":
                    return True
    return False


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(name)


def _if_calling(fn: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> ast.If:
    found = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If) and _calls([node.test], name)  # type: ignore[list-item]
    ]
    assert len(found) == 1
    return found[0]


def _if_name_is_none(fn: ast.FunctionDef, name: str) -> ast.If:
    found = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == name
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Is)
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value is None
    ]
    assert len(found) == 1
    return found[0]


def _if_frame(fn: ast.FunctionDef) -> ast.If:
    found = [node for node in ast.walk(fn) if isinstance(node, ast.If) and _frame_test(node.test)]
    assert len(found) == 1
    return found[0]


def _frame_test(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "frame"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.IsNot)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def _calls(nodes: list[ast.stmt] | list[ast.expr], name: str) -> bool:
    for node in nodes:
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            if isinstance(func, ast.Name) and func.id == name:
                return True
            if isinstance(func, ast.Attribute) and func.attr == name:
                return True
    return False

