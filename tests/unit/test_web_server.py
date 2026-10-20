"""Localhost page server: empty shell, no live port."""

from __future__ import annotations

import re
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest
from starlette.testclient import TestClient

from excavation_lab.ui.copy import PRODUCT_TITLE, catalog
from excavation_lab.ui.tokens import LAB_THEME_CSS
from excavation_lab.web.app import STATIC_DIR, create_app, theme_stylesheet
from excavation_lab.web.server import HOST, serve

_HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{3,8}\b")
_REPO = Path(__file__).resolve().parents[2]


def test_index_is_an_empty_shell(tmp_path: Path) -> None:
    book = catalog("ru")
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/")
        video = client.get("/video", follow_redirects=False)
    assert response.status_code == 200
    text = response.text
    assert PRODUCT_TITLE in text
    assert 'lang="ru"' in text
    assert 'data-theme="dark"' in text
    assert 'href="/static/lab_theme.css"' in text
    assert 'href="/static/lab.css"' in text
    assert 'href="#content"' in text
    assert "К содержанию" in text
    assert "<video" not in text.lower()
    assert 'action="/video/choose"' not in text
    assert "http://" not in text
    assert "https://" not in text
    main = re.search(
        r'<main class="lab-main" id="content">(.*)</main>',
        text,
        re.DOTALL,
    )
    assert main is not None
    assert book.home_title in main.group(1)
    assert book.home_empty in main.group(1)
    assert book.page_video not in main.group(1)
    assert book.lead("video") not in main.group(1)
    assert "<video" not in main.group(1).lower()
    assert video.status_code == 303
    assert urlparse(video.headers["location"]).path == "/"
    assert 'action="/video/choose"' not in video.text
    assert "<video" not in video.text.lower()
    assert 'class="lab-workflow"' not in video.text
    application = create_app(tmp_path)
    assert application.state.workspace == tmp_path


def test_html_autoescape_is_on(tmp_path: Path) -> None:
    application = create_app(tmp_path)
    env = application.state.templates
    assert env.autoescape("base.html") is True
    rendered = env.get_template("base.html").render(
        product_title="<script>alert(1)</script>",
    )
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_static_css_uses_tokens_and_the_theme_file(tmp_path: Path) -> None:
    css_path = STATIC_DIR / "lab.css"
    css = css_path.read_text(encoding="utf-8")
    assert "var(--pdm-" in css
    assert _HEX_COLOR.search(css) is None
    assert not (STATIC_DIR / "lab_theme.css").exists()
    assert LAB_THEME_CSS.parent.name == "ui"
    direct = theme_stylesheet(None)  # type: ignore[arg-type]
    assert Path(direct.path) == LAB_THEME_CSS
    with TestClient(create_app(tmp_path)) as client:
        theme = client.get("/static/lab_theme.css")
        page = client.get("/static/lab.css")
    assert theme.status_code == 200
    assert theme.content == LAB_THEME_CSS.read_bytes()
    assert page.status_code == 200
    assert page.text == css
    assert _HEX_COLOR.search(page.text) is None


@pytest.mark.parametrize("host", ["0.0.0.0", "localhost", "::1"])
def test_other_host_raises_before_the_socket(
    tmp_path: Path,
    host: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def opened(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("socket opened")

    monkeypatch.setattr(socket, "socket", opened)
    with pytest.raises(ValueError, match="127.0.0.1"):
        serve(tmp_path, host=host)


def test_serve_passes_localhost_to_uvicorn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    captured: dict[str, object] = {}

    def fake_run(application: object, **kwargs: object) -> None:
        captured["application"] = application
        captured["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", fake_run)
    serve(tmp_path)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["host"] == HOST
    assert kwargs["port"] == 8501
    assert captured["application"] is not None


def test_doctor_help_does_not_import_starlette() -> None:
    script = """\
import sys
from typer.testing import CliRunner
from excavation_lab.cli import app

def loaded():
    names = ("starlette", "jinja2", "uvicorn")
    return [name for name in names if name in sys.modules]

runner = CliRunner()
doctor = runner.invoke(app, ["doctor", "--help"])
if doctor.exit_code != 0:
    raise SystemExit(doctor.output)
found = loaded()
if found:
    raise SystemExit("doctor --help loaded " + ", ".join(found))
ui = runner.invoke(app, ["ui", "--help"])
if ui.exit_code != 0:
    raise SystemExit(ui.output)
if "ui" not in ui.output:
    raise SystemExit(ui.output)
found = loaded()
if found:
    raise SystemExit("ui --help loaded " + ", ".join(found))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=_REPO,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
