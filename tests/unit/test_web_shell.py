"""Shell routes: navigation, project file, theme, locale, and the events gate."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
from starlette.testclient import TestClient

from excavation_lab.errors import PathNotAllowedError
from excavation_lab.storage import create_project
from excavation_lab.ui.copy import catalog
from excavation_lab.ui.locale import read_locale_file
from excavation_lab.ui.theme import read_theme_file
from excavation_lab.web import app as web_app
from excavation_lab.web.app import create_app
from excavation_lab.web.preferences import (
    consume_flash,
    read_flash,
    read_project_file,
    write_flash,
    write_project_file,
)

_PATHS = ("/", "/data", "/training", "/analysis", "/quality", "/events")
_HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{3,8}\b")
_WEB = Path(__file__).resolve().parents[2] / "src" / "excavation_lab" / "web"


def test_sidebar_marks_one_current_link(tmp_path: Path) -> None:
    book = catalog("ru")
    titles = book.page_titles
    with TestClient(create_app(tmp_path)) as client:
        for path, title in zip(_PATHS, titles, strict=True):
            response = client.get(path)
            assert response.status_code == 200
            links = _nav(response.text)
            assert [item[0] for item in links] == list(_PATHS)
            assert [item[1] for item in links] == list(titles)
            current = [href for href, _label, active in links if active]
            assert current == [path]
            assert title in response.text
            assert 'aria-current="page"' in response.text


def test_reads_do_not_create_preference_files(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert read_theme_file(workspace) is None
    assert read_locale_file(workspace) is None
    assert read_project_file(workspace) is None
    assert read_flash(workspace) is None
    assert not (workspace / "ui").exists()
    with TestClient(create_app(workspace)) as client:
        response = client.get("/quality")
    assert response.status_code == 200
    assert 'data-theme="dark"' in response.text
    assert 'lang="ru"' in response.text
    assert not (workspace / "ui").exists()


def test_project_picker_writes_a_listed_id_and_ignores_the_rest(tmp_path: Path) -> None:
    book = catalog("ru")
    with TestClient(create_app(tmp_path)) as client:
        empty = client.get("/")
        assert book.no_project in empty.text
        select = _select(empty.text)
        assert "disabled" in select[0]
        ignored = client.post(
            "/project",
            data={"project": "pilot", "next": "/"},
            follow_redirects=False,
        )
        assert ignored.status_code == 303
        assert not (tmp_path / "ui" / "project.txt").exists()

        create_project("pilot", tmp_path)
        create_project("west", tmp_path)
        listed = client.get("/data")
        options = _options(listed.text)
        assert options == [("pilot", "pilot", False), ("west", "west", False)]
        chosen = client.post(
            "/project",
            data={"project": "west", "next": "/training"},
            follow_redirects=False,
        )
        assert chosen.status_code == 303
        assert (tmp_path / "ui" / "project.txt").read_bytes() == b"west\n"
        page = client.get("/training")
        assert _options(page.text) == [("pilot", "pilot", False), ("west", "west", True)]
        unknown = client.post(
            "/project",
            data={"project": "other", "next": "/training"},
            follow_redirects=True,
        )
        assert unknown.status_code == 200
        assert (tmp_path / "ui" / "project.txt").read_bytes() == b"west\n"
        assert 'option value="west" selected' in unknown.text or _options(unknown.text)[1][2]


def test_project_write_refuses_an_escaped_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    create_project("pilot", workspace)
    (workspace / "ui").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathNotAllowedError):
        write_project_file(workspace, "pilot")
    assert list(outside.iterdir()) == []
    with TestClient(create_app(workspace)) as client:
        response = client.post(
            "/project",
            data={"project": "pilot", "next": "/"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert list(outside.iterdir()) == []
        page = client.get("/")
        assert page.status_code == 200
        assert catalog("ru").project_write_failed in page.text


def test_unknown_theme_and_locale_do_not_write(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path)) as client:
        theme = client.post(
            "/theme",
            data={"theme": "blue", "next": "/analysis"},
            follow_redirects=False,
        )
        locale = client.post(
            "/locale",
            data={"locale": "de", "next": "https://example.com"},
            follow_redirects=False,
        )
    assert theme.status_code == 303
    assert _path(theme.headers["location"]) == "/analysis"
    assert locale.status_code == 303
    assert _path(locale.headers["location"]) == "/"
    assert "example.com" not in locale.headers["location"]
    assert not (tmp_path / "ui").exists()


def test_lab_css_uses_the_shell_tokens() -> None:
    css = (_WEB / "static" / "lab.css").read_text(encoding="utf-8")
    assert "grid-template-columns: 21rem 1fr" in css
    assert "var(--pdm-sidebar)" in css
    assert "var(--pdm-accent-soft)" in css
    assert "var(--pdm-accent-text)" in css
    assert "var(--pdm-focus-ring)" in css
    assert '[aria-pressed="true"]' in css
    assert "text-decoration: none" in css
    assert ".lab-video" in css
    assert "grid-template-columns: minmax(0, 1fr) minmax(16rem, 22rem)" not in css
    assert _HEX_COLOR.search(css) is None


def test_web_package_does_not_import_fastapi_or_streamlit() -> None:
    offenders: list[str] = []
    for path in _WEB.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".", 1)[0]]
            if "fastapi" in names or "streamlit" in names:
                offenders.append(path.name)
    assert offenders == []
    source = Path(web_app.__file__).read_text(encoding="utf-8")
    assert "st.session_state" not in source
    assert source.count("session={}") >= 2
    assert "write_theme_file(" in source
    assert "write_locale_file(" in source


def test_flash_replaces_one_object_and_refuses_a_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert read_flash(workspace) is None
    assert not (workspace / "ui").exists()
    write_flash(workspace, {"error": "first"})
    write_flash(workspace, {"notice": "second"})
    assert read_flash(workspace) == {"notice": "second"}
    assert consume_flash(workspace) == {"notice": "second"}
    assert read_flash(workspace) is None
    assert not (workspace / "ui" / "flash.json").exists()
    with pytest.raises(TypeError):
        write_flash(workspace, ["error", "notice"])  # type: ignore[arg-type]
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    (escaped / "ui").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathNotAllowedError):
        write_flash(escaped, {"error": "nope"})
    assert list(outside.iterdir()) == []


def test_handlers_pass_an_empty_session_into_the_resolvers() -> None:
    source = Path(web_app.__file__).read_text(encoding="utf-8")
    assert "resolve_theme(workspace, session={})" in source
    assert "resolve_locale(workspace, session={})" in source


def _nav(html: str) -> list[tuple[str, str, bool]]:
    found: list[tuple[str, str, bool]] = []
    for href, attrs, title in re.findall(
        r'<a class="lab-nav-btn" href="([^"]+)"([^>]*)>([^<]+)</a>',
        html,
    ):
        found.append((href, title, 'aria-current="page"' in attrs))
    return found


def _select(html: str) -> tuple[str, str]:
    match = re.search(r"<select\b([^>]*)>(.*?)</select>", html, re.DOTALL)
    assert match is not None
    return match.group(1), match.group(2)


def _options(html: str) -> list[tuple[str, str, bool]]:
    _attrs, body = _select(html)
    found: list[tuple[str, str, bool]] = []
    for attrs, label in re.findall(r"<option\b([^>]*)>([^<]*)</option>", body):
        value = re.search(r'value="([^"]*)"', attrs)
        found.append((value.group(1) if value else "", label, "selected" in attrs))
    return found


def _path(value: str) -> str:
    path = urlparse(value).path
    return path or value
