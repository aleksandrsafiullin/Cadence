"""Theme file, token parser, and the Light/Dark switch."""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

import pytest
from starlette.testclient import TestClient

from excavation_lab.errors import PathNotAllowedError
from excavation_lab.ui import copy as ui_copy
from excavation_lab.ui import theme as theme_mod
from excavation_lab.ui import tokens
from excavation_lab.ui.theme import (
    commit_theme_choice,
    read_theme_file,
    resolve_theme,
    theme_script_lines,
    write_theme_file,
)
from excavation_lab.ui.tokens import combined_css, theme_custom_properties
from excavation_lab.web.app import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_THEME = REPO_ROOT / "workspace" / "ui" / "theme.txt"
_REPO_WORKSPACE = REPO_ROOT / "workspace"
_INVALID = ("stopping", "stopped", "idle", "", "nope", "DARK", "light\ndark")


@pytest.fixture(autouse=True)
def _repo_theme_file_unchanged() -> Iterator[None]:
    before = _repo_theme_bytes()
    yield
    assert _repo_theme_bytes() == before


def _repo_theme_bytes() -> bytes | None:
    if _REPO_THEME.is_file():
        return _REPO_THEME.read_bytes()
    return None


def test_read_does_not_create_the_preference(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert read_theme_file(workspace) is None
    assert resolve_theme(workspace, {}) == "dark"
    assert not (workspace / "ui").exists()


def test_session_overrides_file(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    write_theme_file(workspace, "light")
    session = {"lab_theme": "dark"}
    assert resolve_theme(workspace, session) == "dark"
    assert session["lab_theme"] == "dark"
    assert _preference(workspace).read_text(encoding="utf-8").strip() == "light"


def test_valid_file_is_stored_in_an_empty_session(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    write_theme_file(workspace, "light")
    session: dict[str, object] = {}
    assert resolve_theme(workspace, session) == "light"
    assert session["lab_theme"] == "light"


def test_invalid_session_falls_through_to_the_file(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    write_theme_file(workspace, "light")
    session: dict[str, object] = {"lab_theme": "stopping"}
    assert resolve_theme(workspace, session) == "light"
    assert session["lab_theme"] == "light"


@pytest.mark.parametrize("text", _INVALID)
def test_invalid_preference_is_dark(tmp_path: Path, text: str) -> None:
    workspace = tmp_path / "ws"
    ui = workspace / "ui"
    ui.mkdir(parents=True)
    (ui / "theme.txt").write_text(text, encoding="utf-8")
    session: dict[str, object] = {}
    assert read_theme_file(workspace) is None
    assert resolve_theme(workspace, session) == "dark"
    assert "lab_theme" not in session


def test_stripped_light_line_is_light(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ui = workspace / "ui"
    ui.mkdir(parents=True)
    (ui / "theme.txt").write_text("  light \n", encoding="utf-8")
    assert read_theme_file(workspace) == "light"


def test_write_uses_replace_inside_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy(src: str | Path, dst: str | Path) -> None:
        calls.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(theme_mod.os, "replace", spy)
    before = _repo_theme_bytes()
    write_theme_file(tmp_path, "light")
    written = _preference(tmp_path)
    assert len(calls) == 1
    src, dst = calls[0]
    assert src.name.startswith(".theme-")
    assert src.suffix == ".tmp"
    assert dst == written
    assert not src.exists()
    assert list((tmp_path / "ui").glob(".theme-*")) == []
    assert written.read_text(encoding="utf-8").strip() == "light"
    assert written.resolve().is_relative_to(tmp_path.resolve())
    assert not written.resolve().is_relative_to(_REPO_WORKSPACE.resolve())
    assert _repo_theme_bytes() == before
    write_theme_file(tmp_path, "dark")
    assert written.read_text(encoding="utf-8").strip() == "dark"


def test_unknown_theme_is_not_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown theme"):
        write_theme_file(tmp_path, "stopping")
    assert not (tmp_path / "ui").exists()


def test_unknown_label_does_not_write(tmp_path: Path) -> None:
    session: dict[str, object] = {"lab_theme": "dark"}
    commit_theme_choice(tmp_path, "stopping", session)
    assert session == {"lab_theme": "dark"}
    assert not (tmp_path / "ui").exists()


def test_failed_write_keeps_the_session_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(_workspace: Path, _theme: str) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(theme_mod, "write_theme_file", explode)
    session: dict[str, object] = {"lab_theme": "dark", "lab_appearance": ui_copy.THEME_DARK}
    commit_theme_choice(tmp_path, ui_copy.THEME_LIGHT, session)
    assert session["lab_theme"] == "dark"
    assert session["lab_appearance"] == ui_copy.THEME_DARK
    assert session["lab_theme_write_error"] == ui_copy.THEME_WRITE_FAILED
    assert ui_copy.THEME_WRITE_FAILED != ui_copy.THEME_DISPLAY_ONLY
    assert not (tmp_path / "ui" / "theme.txt").exists()


def test_failed_write_assigns_theme_without_a_prior_valid_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(_workspace: Path, _theme: str) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(theme_mod, "write_theme_file", explode)
    missing: dict[str, object] = {}
    commit_theme_choice(tmp_path, ui_copy.THEME_LIGHT, missing)
    assert missing["lab_theme"] == "dark"
    assert missing["lab_appearance"] == ui_copy.THEME_DARK

    monkeypatch.undo()
    write_theme_file(tmp_path, "light")
    monkeypatch.setattr(theme_mod, "write_theme_file", explode)
    invalid: dict[str, object] = {"lab_theme": "stopping"}
    commit_theme_choice(tmp_path, ui_copy.THEME_DARK, invalid)
    assert invalid["lab_theme"] == "light"
    assert invalid["lab_appearance"] == ui_copy.THEME_LIGHT


def test_successful_write_clears_the_error(tmp_path: Path) -> None:
    session: dict[str, object] = {"lab_theme_write_error": ui_copy.THEME_WRITE_FAILED}
    commit_theme_choice(tmp_path, ui_copy.THEME_LIGHT, session)
    assert session["lab_theme"] == "light"
    assert "lab_theme_write_error" not in session
    assert _preference(tmp_path).read_text(encoding="utf-8").strip() == "light"


def test_symlink_directory_outside_workspace_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "theme.txt").write_text("secret\n", encoding="utf-8")
    (workspace / "ui").symlink_to(outside, target_is_directory=True)
    assert read_theme_file(workspace) is None
    with pytest.raises(PathNotAllowedError) as caught:
        write_theme_file(workspace, "light")
    assert caught.value.code == "path_not_allowed"
    assert (outside / "theme.txt").read_text(encoding="utf-8") == "secret\n"
    assert list(outside.glob(".theme-*")) == []
    session: dict[str, object] = {"lab_theme": "dark"}
    commit_theme_choice(workspace, ui_copy.THEME_LIGHT, session)
    assert session["lab_theme"] == "dark"
    assert session["lab_theme_write_error"] == ui_copy.THEME_WRITE_FAILED
    assert (outside / "theme.txt").read_text(encoding="utf-8") == "secret\n"


def test_symlink_file_outside_workspace_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ui = workspace / "ui"
    ui.mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    (ui / "theme.txt").symlink_to(secret)
    with pytest.raises(PathNotAllowedError):
        write_theme_file(workspace, "dark")
    assert secret.read_text(encoding="utf-8") == "secret"
    assert (ui / "theme.txt").is_symlink()
    assert read_theme_file(workspace) is None


def test_symlink_that_stays_inside_is_written(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    real = workspace / "stored"
    real.mkdir(parents=True)
    (workspace / "ui").symlink_to(real, target_is_directory=True)
    write_theme_file(workspace, "light")
    written = real / "theme.txt"
    assert written.read_text(encoding="utf-8").strip() == "light"
    assert read_theme_file(workspace) == "light"
    assert written.resolve().is_relative_to(workspace.resolve())


def test_unknown_theme_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown theme"):
        theme_custom_properties("stopping")
    with pytest.raises(ValueError, match="unknown theme"):
        theme_script_lines("idle")


def test_light_overrides_dark_and_keeps_font() -> None:
    dark = theme_custom_properties("dark")
    light = theme_custom_properties("light")
    assert dark["--pdm-bg"] == "#0F0F11"
    assert light["--pdm-bg"] == "#F5F5F7"
    assert dark["--pdm-accent"] == "#0A84FF"
    assert light["--pdm-accent"] == "#0071E3"
    assert light["--pdm-font"] == dark["--pdm-font"]
    assert "SF Pro Text" in dark["--pdm-font"]


def test_combined_css_loads_both_stylesheets_by_file_name() -> None:
    source = Path(tokens.__file__).read_text(encoding="utf-8")
    assert 'with_name("lab_theme.css")' in source
    assert "streamlit_chrome.css" not in source
    text = combined_css()
    assert text == tokens.LAB_THEME_CSS.read_text(encoding="utf-8")
    assert '[data-theme="dark"]' in text
    assert '[data-theme="light"]' in text
    assert "stApp" not in text
    assert tokens.LAB_THEME_CSS.is_file()


def test_tokens_module_does_not_import_streamlit() -> None:
    source = Path(tokens.__file__).read_text(encoding="utf-8")
    assert _imports_streamlit(source) is False


def test_theme_script_whitelist_is_one_statement() -> None:
    dark = theme_script_lines("dark")
    light = theme_script_lines("light")
    assert dark[0] == 'document.documentElement.setAttribute("data-theme", "dark");'
    assert light[0] == 'document.documentElement.setAttribute("data-theme", "light");'
    assert dark[1:] == light[1:]
    script = "\n".join(dark)
    assert 'getAttribute("href")' in script
    assert "a.pathname" not in script
    assert script.count("location.pathname") == 1
    assert script.count("requestAnimationFrame") == 1
    assert 'setAttribute("aria-current", "page")' in script
    assert 'removeAttribute("aria-current")' in script
    assert re.search(r"(?<![\d.])0(?:[.,]0+)?%?(?![\d.])", script) is None
    assert "http://" not in script
    assert "https://" not in script


def test_only_the_change_callback_writes_the_preference() -> None:
    source = Path(theme_mod.__file__).read_text(encoding="utf-8")
    assert _callers(source, "write_theme_file") == ["commit_theme_choice"]
    assert _callers(source, "commit_theme_choice") == []
    assert "on_change" not in source
    assert "persist_state" not in source
    assert "query-params" not in source
    assert "bind=" not in source
    assert "st.markdown" not in source
    assert "st.html" not in source
    assert _imports_streamlit(source) is False
    app = (REPO_ROOT / "src" / "excavation_lab" / "web" / "app.py").read_text(encoding="utf-8")
    assert "write_theme_file(" in app
    assert "on_change" not in app


def test_app_injects_css_only_before_navigation() -> None:
    template = (
        REPO_ROOT / "src" / "excavation_lab" / "web" / "templates" / "base.html"
    ).read_text(encoding="utf-8")
    assert template.index('href="/static/lab_theme.css"') < template.index("<nav")
    assert "emit_theme_script" not in template
    app = (REPO_ROOT / "src" / "excavation_lab" / "web" / "app.py").read_text(encoding="utf-8")
    assert "apply_theme" not in app
    assert "emit_theme_script" not in app
    assert "st.set_page_config" not in app
    assert "st.navigation" not in app
    shell = (REPO_ROOT / "src" / "excavation_lab" / "web" / "shell.py").read_text(encoding="utf-8")
    context = shell[shell.index("def page_shell") :]
    assert context.index("resolve_theme(") < context.index("nav_links")
    nav = (REPO_ROOT / "src" / "excavation_lab" / "ui" / "navigation.py").read_text(
        encoding="utf-8"
    )
    assert "PAGE_ROUTES" in nav
    assert "st.navigation" not in nav
    assert "import streamlit" not in nav


def test_config_theme_base_is_dark_without_hex_colors(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert resolve_theme(workspace, {}) == "dark"
    template = (
        REPO_ROOT / "src" / "excavation_lab" / "web" / "templates" / "base.html"
    ).read_text(encoding="utf-8")
    assert "default('dark')" in template
    assert re.search(r"#[0-9A-Fa-f]{3,8}\b", template) is None
    server = (REPO_ROOT / "src" / "excavation_lab" / "web" / "server.py").read_text(
        encoding="utf-8"
    )
    assert 'HOST = "127.0.0.1"' in server
    assert "0.0.0.0" not in server
    assert re.search(r"#[0-9A-Fa-f]{3,8}\b", server) is None
    assert "auth" not in server.lower()


def test_ui_sources_have_no_url_scheme() -> None:
    root = REPO_ROOT / "src" / "excavation_lab" / "ui"
    for path in root.rglob("*"):
        if path.suffix not in {".py", ".css"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "http://" not in text, path
        assert "https://" not in text, path


def test_switch_defaults_dark_and_persists_light(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    preference = workspace / "ui" / "theme.txt"
    with TestClient(create_app(workspace)) as client:
        first = client.get("/")
        assert first.status_code == 200
        _assert_theme_control(first.text, "dark")
        assert ui_copy.THEME_LABEL in first.text
        assert first.text.count(ui_copy.THEME_LABEL) == 1
        assert ui_copy.THEME_DISPLAY_ONLY in first.text
        assert re.findall(r"<script\b[^>]*>", first.text, flags=re.IGNORECASE) == [
            '<script src="/static/project_catalog.js">',
        ]
        assert not preference.exists()
        assert 'name="theme"' in first.text

        switched = client.post(
            "/theme",
            data={"theme": "light", "next": "/"},
            follow_redirects=True,
        )
        assert switched.status_code == 200
        assert switched.history[0].status_code == 303
        assert _location_path(switched.history[0].headers["location"]) == "/"
        assert preference.read_text(encoding="utf-8").strip() == "light"
        _assert_theme_control(switched.text, "light")
        assert 'data-theme="light"' in switched.text

        fresh = client.get("/")
        assert fresh.status_code == 200
        _assert_theme_control(fresh.text, "light")
        assert preference.read_text(encoding="utf-8").strip() == "light"

        darkened = client.post(
            "/theme",
            data={"theme": "dark", "next": "/events"},
            follow_redirects=True,
        )
        assert darkened.status_code == 200
        assert darkened.history[0].headers["location"]
        assert _location_path(darkened.history[0].headers["location"]) == "/events"
        assert preference.read_text(encoding="utf-8").strip() == "dark"
        _assert_theme_control(darkened.text, "dark")
        events = client.get("/events")
        assert events.status_code == 200
        _assert_theme_control(events.text, "dark")
        assert preference.read_text(encoding="utf-8").strip() == "dark"


def test_escaped_symlink_shows_sidebar_error(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "ui").symlink_to(outside, target_is_directory=True)
    with TestClient(create_app(workspace)) as client:
        response = client.post(
            "/theme",
            data={"theme": "light", "next": "/"},
            follow_redirects=True,
        )
    assert response.status_code == 200
    assert response.history[0].status_code == 303
    assert list(outside.iterdir()) == []
    assert ui_copy.THEME_WRITE_FAILED in response.text
    assert 'data-theme="dark"' in response.text
    _assert_theme_control(response.text, "dark")


def _location_path(value: str) -> str:
    path = urlparse(value).path
    return path or value


def _preference(workspace: Path) -> Path:
    return workspace / "ui" / "theme.txt"


def _assert_theme_control(html: str, theme: str) -> None:
    assert f'data-theme="{theme}"' in html
    assert _button_pressed(html, ui_copy.THEME_LIGHT) == ("true" if theme == "light" else "false")
    assert _button_pressed(html, ui_copy.THEME_DARK) == ("true" if theme == "dark" else "false")
    assert 'value="light"' in html
    assert 'value="dark"' in html


def _button_pressed(html: str, label: str) -> str | None:
    for attrs, text in re.findall(r"<button\b([^>]*)>([^<]*)</button>", html):
        if text.strip() != label:
            continue
        match = re.search(r'aria-pressed="(true|false)"', attrs)
        if match is not None:
            return match.group(1)
    return None


def _imports_streamlit(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "streamlit" or alias.name.startswith("streamlit."):
                    return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if module == "streamlit" or module.startswith("streamlit."):
                return True
    return False


def _callers(source: str, name: str) -> list[str]:
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    callers: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name) or func.id != name:
            continue
        current: ast.AST | None = node
        while current is not None and not isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            current = parents.get(current)
        assert isinstance(current, ast.FunctionDef)
        callers.append(current.name)
    return callers
