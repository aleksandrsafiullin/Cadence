"""Empty lab screens: no invented metrics, no model boxes, events unavailable."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from starlette.testclient import TestClient

from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.models.quality import load_quality_report
from excavation_lab.storage import create_project
from excavation_lab.ui import (
    analysis_page,
    data_page,
    events_page,
    quality_page,
    training_page,
    video_page,
)
from excavation_lab.ui import copy as ui_copy
from excavation_lab.web.app import create_app
from excavation_lab.web.preferences import write_project_file

REPO_ROOT = Path(__file__).resolve().parents[2]
_PAGES = (
    video_page,
    data_page,
    training_page,
    analysis_page,
    quality_page,
    events_page,
)
_ZERO_TOKEN = re.compile(r"(?<![\d.])0(?:[.,]0+)?%?(?![\d.])")
_NUMERIC_METRIC = re.compile(r"^-?\d+(?:[.,]\d+)?%?$")
_NETWORK_MODULES = {"requests", "httpx", "socket", "urllib"}


def _zero_token(text: str) -> str | None:
    match = _ZERO_TOKEN.search(text)
    if match is None:
        return None
    return match.group(0)


def _nodes(app: object) -> list[object]:
    return [*app.main, *app.sidebar]  # type: ignore[attr-defined]


def _strings(app: object) -> list[str]:
    found: list[str] = []
    for node in _nodes(app):
        for attr in ("value", "label", "help", "title"):
            raw = getattr(node, attr, None)
            if isinstance(raw, str) and raw and "--pdm-bg" not in raw:
                found.append(raw)
        options = getattr(node, "options", None)
        if isinstance(options, list):
            for item in options:
                text = str(item)
                if "--pdm-bg" not in text:
                    found.append(text)
    return found


def _joined(app: object) -> str:
    return "\n".join(_strings(app))


def test_strings_skip_stylesheet_values_and_keep_captions() -> None:
    css = "background: var(--pdm-bg); margin: 0;"
    node = SimpleNamespace(
        value=css,
        label=ui_copy.THEME_LABEL,
        help=None,
        title=None,
        options=[ui_copy.THEME_LIGHT, css],
    )
    caption = SimpleNamespace(
        value=ui_copy.THEME_DISPLAY_ONLY,
        label=None,
        help=None,
        title=None,
    )
    app = SimpleNamespace(main=[node], sidebar=[caption])
    found = _strings(app)  # type: ignore[arg-type]
    assert css not in found
    assert ui_copy.THEME_LABEL in found
    assert ui_copy.THEME_LIGHT in found
    joined = _joined(app)  # type: ignore[arg-type]
    assert css not in joined
    assert ui_copy.THEME_DISPLAY_ONLY in joined


def test_zero_token_is_a_measurement_not_a_version() -> None:
    assert _zero_token("показатели: 0") == "0"
    assert _zero_token("0.0") == "0.0"
    assert _zero_token(" 0 ") == "0"
    assert _zero_token("версии 0.1 не") is None
    assert _zero_token("200 МБ") is None
    assert _zero_token(ui_copy.NOT_CALCULATED) is None
    assert _zero_token("AP50") is None


def test_page_modules_expose_render() -> None:
    for module in _PAGES:
        assert module.__doc__
        assert not hasattr(module, "render")


_SHELL_PAGES = (
    ("/video", "video", ui_copy.PAGE_VIDEO),
    ("/data", "data", ui_copy.PAGE_DATA),
    ("/training", "training", ui_copy.PAGE_TRAINING),
    ("/analysis", "analysis", ui_copy.PAGE_ANALYSIS),
    ("/quality", "quality", ui_copy.PAGE_QUALITY),
    ("/events", "events", ui_copy.PAGE_EVENTS),
)
_EMPTY_ATTR = {
    "data": "data_empty",
    "training": "training_empty",
    "analysis": "analysis_empty",
    "quality": "quality_empty",
}


@pytest.mark.parametrize(
    ("path", "page_id", "title"),
    _SHELL_PAGES,
    ids=[item[1] for item in _SHELL_PAGES],
)
def test_empty_workspace_page(path: str, page_id: str, title: str, tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path)) as client:
        response = client.get(path, follow_redirects=False)
    assert response.status_code == 303
    assert urlparse(response.headers["location"]).path == "/"
    assert page_id
    assert title
    text = response.text
    assert 'class="lab-workflow"' not in text
    assert 'action="/video/choose"' not in text
    assert "<video" not in text.lower()
    assert not (tmp_path / "ui" / "project.txt").exists()


def test_quality_page_without_metrics_json_has_no_zero_chart(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    book = ui_copy.catalog("ru")
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/quality", follow_redirects=False)
    assert response.status_code == 200
    text = response.text
    assert ui_copy.PAGE_QUALITY in text
    assert ui_copy.PAGE_DESCRIPTIONS[ui_copy.PAGE_QUALITY] in text
    assert ui_copy.PRODUCT_TITLE in text
    assert ui_copy.QUALITY_EMPTY in text
    assert ui_copy.NOT_CALCULATED in text
    assert 'id="lab-home"' in text
    assert '<p class="lab-label">pilot</p>' in text
    main = re.search(r"<main\b[^>]*>(.*)</main>", text, re.DOTALL)
    assert main is not None
    assert "<select" not in main.group(1)
    metrics = re.findall(
        r'<div class="lab-metric">\s*<span>([^<]*)</span>\s*<b>([^<]*)</b>\s*</div>',
        text,
    )
    assert metrics
    for _label, value in metrics:
        assert value == ui_copy.NOT_CALCULATED
    lowered = text.lower()
    assert "lab-chart" not in lowered
    assert "<svg" not in lowered
    assert "<canvas" not in lowered
    assert "<img" not in lowered
    assert "altair" not in lowered
    labels = [item.strip() for item in re.findall(r"<button\b[^>]*>([^<]*)</button>", text)]
    assert book.send_to_labeling not in labels
    assert book.train_start not in labels
    assert book.analysis_start not in labels
    assert "xyxy" not in text
    assert "heuristic" not in lowered
    visible = re.sub(r"<[^>]+>", " ", text)
    assert _zero_token(visible) is None, text
    report = load_quality_report(tmp_path, "pilot", None)
    assert quality_page.ap50_chart_frame(report) is None
    zeros = {
        "status": "computed",
        "classes": {name: {"ap50": 0} for name in report["classes"]},
    }
    assert quality_page.ap50_chart_frame(zeros) is None


def test_project_without_video_still_asks_for_a_local_path(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/video", follow_redirects=False)
    assert response.status_code == 200
    text = response.text
    assert ui_copy.PAGE_VIDEO in text
    assert ui_copy.PAGE_DESCRIPTIONS[ui_copy.PAGE_VIDEO] in text
    assert ui_copy.PRODUCT_TITLE in text
    assert ui_copy.CHOOSE_FILE_BUTTON in text
    assert 'action="/video/choose"' in text
    assert 'id="local-path"' in text
    assert 'type="file"' not in text
    assert ui_copy.UPLOAD_LABEL not in text
    assert ui_copy.LOCAL_PATH_LABEL in text
    assert 'id="lab-home"' in text
    assert '<p class="lab-label">pilot</p>' in text
    assert "<video" not in text
    assert "<canvas" not in text
    visible = re.sub(r"<[^>]+>", " ", text)
    assert _zero_token(visible) is None


def test_streamlit_config_is_localhost_only() -> None:
    from excavation_lab.web.server import HOST
    from excavation_lab.web.uploads import UPLOAD_CAP

    assert HOST == "127.0.0.1"
    assert UPLOAD_CAP == 8192 * 1024 * 1024
    environment = (REPO_ROOT / "docs" / "ENVIRONMENT.md").read_text(encoding="utf-8")
    uploads = (
        REPO_ROOT / "src" / "excavation_lab" / "web" / "uploads.py"
    ).read_text(encoding="utf-8")
    assert "local path" in environment or "local path" in uploads


def test_page_server_host_is_not_all_interfaces() -> None:
    from excavation_lab.web.server import HOST

    assert HOST != "0.0.0.0"
    assert HOST == "127.0.0.1"


def test_shell_does_not_call_auth_or_telemetry() -> None:
    ui_root = REPO_ROOT / "src" / "excavation_lab" / "ui"
    web_root = REPO_ROOT / "src" / "excavation_lab" / "web"
    assert ui_root.is_dir()
    assert web_root.is_dir()
    files = [
        path
        for root in (ui_root, web_root)
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".html", ".css", ".js"}
    ]
    for path in files:
        source = path.read_text(encoding="utf-8")
        assert "st.login" not in source
        assert "0.0.0.0" not in source
        assert "telemetry.streamlit.io" not in source
        assert "https://" not in source
        assert "http://" not in source
        if path.suffix != ".py":
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".", 1)[0]]
            else:
                continue
            assert _NETWORK_MODULES.isdisjoint(names), path


def test_app_entrypoint_renders_video(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/video", follow_redirects=False)
    assert response.status_code == 303
    assert urlparse(response.headers["location"]).path == "/"
    assert 'action="/video/choose"' not in response.text
    assert "<video" not in response.text.lower()
    assert not (tmp_path / "ui" / "project.txt").exists()
