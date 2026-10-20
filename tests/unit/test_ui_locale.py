"""Locale file, resolver, and the Русский | English switch."""

from __future__ import annotations

import ast
import json
import os
import re
from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path
from urllib.parse import urlparse

import pytest
from starlette.testclient import TestClient

from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.errors import PathNotAllowedError
from excavation_lab.storage import create_project
from excavation_lab.ui import copy as ui_copy
from excavation_lab.ui import locale as locale_mod
from excavation_lab.ui import navigation
from excavation_lab.ui.locale import (
    commit_locale_choice,
    page_id_for,
    preference_path,
    read_locale_file,
    resolve_locale,
    title_for,
    write_locale_file,
)
from excavation_lab.web.app import create_app
from excavation_lab.web.preferences import write_project_file

REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_LOCALE = REPO_ROOT / "workspace" / "ui" / "locale.txt"
_REPO_WORKSPACE = REPO_ROOT / "workspace"
_PAGE_IDS = ("video", "data", "training", "analysis", "quality", "actions", "events")
_INVALID = ("stopping", "", "fr", "EN", "RU", "ru\nen")
_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_ALLOWED_CYRILLIC = ("Русский", "класс")
_ENGLISH_TITLES = (
    "Video",
    "Data and labeling",
    "Training",
    "Analysis",
    "Quality",
    "Actions",
    "Events",
)
_ENGLISH_ROUTES = (
    ("/video", "Video", "video"),
    ("/data", "Data and labeling", "data"),
    ("/training", "Training", "training"),
    ("/analysis", "Analysis", "analysis"),
    ("/quality", "Quality", "quality"),
    ("/actions", "Actions", "actions"),
    ("/events", "Events", "events"),
)
_CATALOG_CALLS: tuple[tuple[str, tuple[object, ...]], ...] = (
    ("not_train_frame_message", ("frame-1", "train")),
    ("frame_not_found_message", ("frame-1",)),
    ("frame_ambiguous_message", ("frame-1",)),
    ("eval_not_found_message", ("eval-1",)),
    ("eval_not_found_message", ("",)),
    ("sent_to_labeling_message", ("frame-1",)),
    ("episode_ambiguous_message", ("episode-1",)),
    ("hardware_caption", ("cpu", False, False, None, False)),
    ("hardware_caption", ("cuda", True, False, 4096, True)),
    ("hardware_unavailable", ("ffprobe missing",)),
    ("weights_warning", ("/cache/weights",)),
    ("allowed_directory_extras", ("clips",)),
    ("allowed_directory_extras", ("clips", "dumps")),
    ("duplicate_video_message", ()),
)
_ZERO_METRIC = frozenset({"0", "0.0", "0,0"})
_INVARIANT_FIELDS = (
    "product_title",
    "kicker_project",
    "kicker_workflow",
    "kicker_appearance",
    "theme_light",
    "theme_dark",
    "not_evaluated",
    "not_evaluated_caption",
    "predicted_heading",
)


@pytest.fixture(autouse=True)
def _repo_locale_file_unchanged() -> Iterator[None]:
    before = _repo_locale_bytes()
    yield
    assert _repo_locale_bytes() == before


def _repo_locale_bytes() -> bytes | None:
    if _REPO_LOCALE.is_file():
        return _REPO_LOCALE.read_bytes()
    return None


def test_read_does_not_create_the_preference(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert preference_path(workspace) == workspace / "ui" / "locale.txt"
    assert read_locale_file(workspace) is None
    session: dict[str, object] = {}
    assert resolve_locale(workspace, session) == "ru"
    assert "lab_locale" not in session
    assert not (workspace / "ui").exists()


def test_session_overrides_file(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    write_locale_file(workspace, "en")
    session = {"lab_locale": "ru"}
    assert resolve_locale(workspace, session) == "ru"
    assert session["lab_locale"] == "ru"
    assert _preference(workspace).read_text(encoding="utf-8") == "en\n"


def test_valid_file_is_stored_in_an_empty_session(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    write_locale_file(workspace, "en")
    session: dict[str, object] = {}
    assert resolve_locale(workspace, session) == "en"
    assert session["lab_locale"] == "en"
    assert _preference(workspace).read_bytes() == b"en\n"


def test_invalid_session_falls_through_to_the_file(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    write_locale_file(workspace, "en")
    session: dict[str, object] = {"lab_locale": "stopping"}
    assert resolve_locale(workspace, session) == "en"
    assert session["lab_locale"] == "en"


@pytest.mark.parametrize("text", _INVALID)
def test_invalid_preference_is_russian(tmp_path: Path, text: str) -> None:
    workspace = tmp_path / "ws"
    ui = workspace / "ui"
    ui.mkdir(parents=True)
    path = ui / "locale.txt"
    path.write_text(text, encoding="utf-8")
    session: dict[str, object] = {}
    assert read_locale_file(workspace) is None
    assert resolve_locale(workspace, session) == "ru"
    assert "lab_locale" not in session
    assert path.read_text(encoding="utf-8") == text


def test_stripped_en_line_is_en(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ui = workspace / "ui"
    ui.mkdir(parents=True)
    (ui / "locale.txt").write_text("  en \n", encoding="utf-8")
    assert read_locale_file(workspace) == "en"


def test_write_uses_replace_inside_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy(src: str | Path, dst: str | Path) -> None:
        calls.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(locale_mod.os, "replace", spy)
    before = _repo_locale_bytes()
    write_locale_file(tmp_path, "en")
    written = _preference(tmp_path)
    assert len(calls) == 1
    src, dst = calls[0]
    assert src.name.startswith(".locale-")
    assert src.suffix == ".tmp"
    assert dst == written
    assert not src.exists()
    assert list((tmp_path / "ui").glob(".locale-*")) == []
    assert written.read_bytes() == b"en\n"
    assert written.resolve().is_relative_to(tmp_path.resolve())
    assert not written.resolve().is_relative_to(_REPO_WORKSPACE.resolve())
    assert _repo_locale_bytes() == before
    write_locale_file(tmp_path, "ru")
    assert written.read_bytes() == b"ru\n"


def test_unknown_locale_is_not_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown locale"):
        write_locale_file(tmp_path, "stopping")
    assert not (tmp_path / "ui").exists()
    with pytest.raises(ValueError, match="unknown locale"):
        write_locale_file(tmp_path, "EN")
    assert not (tmp_path / "ui").exists()


def test_unknown_label_does_not_write(tmp_path: Path) -> None:
    session: dict[str, object] = {
        "lab_locale": "ru",
        "project_picker": "no-project",
        "analysis_time": 3.0,
    }
    commit_locale_choice(tmp_path, "stopping", session)
    commit_locale_choice(tmp_path, None, session)
    assert session == {
        "lab_locale": "ru",
        "project_picker": "no-project",
        "analysis_time": 3.0,
    }
    assert not (tmp_path / "ui").exists()


def test_successful_write_keeps_other_keys(tmp_path: Path) -> None:
    session: dict[str, object] = {
        "lab_locale_write_error": "stale",
        "project_picker": "no-project",
        "analysis_time": 3.0,
    }
    commit_locale_choice(tmp_path, "English", session)
    assert session["lab_locale"] == "en"
    assert session["project_picker"] == "no-project"
    assert session["analysis_time"] == 3.0
    assert "lab_locale_write_error" not in session
    assert "English" not in session.values()
    assert "Русский" not in session.values()
    assert _preference(tmp_path).read_bytes() == b"en\n"


def test_failed_write_keeps_the_session_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(_workspace: Path, _locale: str) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(locale_mod, "write_locale_file", explode)
    session: dict[str, object] = {
        "lab_locale": "en",
        "lab_locale_choice": "English",
        "project_picker": "pilot",
        "analysis_time": 4.0,
    }
    commit_locale_choice(tmp_path, "Русский", session)
    assert session["lab_locale"] == "en"
    assert session["lab_locale_choice"] == "English"
    assert session["project_picker"] == "pilot"
    assert session["analysis_time"] == 4.0
    assert session["lab_locale_write_error"] == ui_copy.catalog("en").locale_write_failed
    assert not (tmp_path / "ui" / "locale.txt").exists()


def test_failed_write_assigns_locale_without_a_prior_valid_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(_workspace: Path, _locale: str) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(locale_mod, "write_locale_file", explode)
    missing: dict[str, object] = {"analysis_time": 1.0}
    commit_locale_choice(tmp_path, "English", missing)
    assert missing["lab_locale"] == "ru"
    assert missing["lab_locale_choice"] == "Русский"
    assert missing["analysis_time"] == 1.0
    assert missing["lab_locale_write_error"] == ui_copy.catalog("ru").locale_write_failed

    monkeypatch.undo()
    write_locale_file(tmp_path, "en")
    monkeypatch.setattr(locale_mod, "write_locale_file", explode)
    invalid: dict[str, object] = {"lab_locale": "stopping"}
    commit_locale_choice(tmp_path, "Русский", invalid)
    assert invalid["lab_locale"] == "en"
    assert invalid["lab_locale_choice"] == "English"
    assert invalid["lab_locale_write_error"] == ui_copy.catalog("en").locale_write_failed
    assert _preference(tmp_path).read_bytes() == b"en\n"


def test_symlink_directory_outside_workspace_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "locale.txt").write_text("secret\n", encoding="utf-8")
    (workspace / "ui").symlink_to(outside, target_is_directory=True)
    assert read_locale_file(workspace) is None
    with pytest.raises(PathNotAllowedError) as caught:
        write_locale_file(workspace, "en")
    assert caught.value.code == "path_not_allowed"
    assert (outside / "locale.txt").read_text(encoding="utf-8") == "secret\n"
    assert list(outside.glob(".locale-*")) == []
    session: dict[str, object] = {"lab_locale": "ru", "project_picker": "no-project"}
    commit_locale_choice(workspace, "English", session)
    assert session["lab_locale"] == "ru"
    assert session["lab_locale_choice"] == "Русский"
    assert session["project_picker"] == "no-project"
    assert session["lab_locale_write_error"] == ui_copy.catalog("ru").locale_write_failed
    assert (outside / "locale.txt").read_text(encoding="utf-8") == "secret\n"


def test_symlink_file_outside_workspace_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ui = workspace / "ui"
    ui.mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    (ui / "locale.txt").symlink_to(secret)
    with pytest.raises(PathNotAllowedError):
        write_locale_file(workspace, "ru")
    assert secret.read_text(encoding="utf-8") == "secret"
    assert (ui / "locale.txt").is_symlink()
    assert read_locale_file(workspace) is None


def test_symlink_that_stays_inside_is_written(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    real = workspace / "stored"
    real.mkdir(parents=True)
    (workspace / "ui").symlink_to(real, target_is_directory=True)
    write_locale_file(workspace, "en")
    written = real / "locale.txt"
    assert written.read_bytes() == b"en\n"
    assert read_locale_file(workspace) == "en"
    assert written.resolve().is_relative_to(workspace.resolve())


def test_only_the_change_callback_writes_the_preference() -> None:
    source = Path(locale_mod.__file__).read_text(encoding="utf-8")
    assert _callers(source, "write_locale_file") == ["commit_locale_choice"]
    assert _callers(source, "commit_locale_choice") == []
    assert "on_change" not in source
    assert "persist_state" not in source
    assert "query-params" not in source
    assert "bind=" not in source
    assert "st.markdown" not in source
    assert _imports_streamlit(source) is False
    copy_source = Path(ui_copy.__file__).read_text(encoding="utf-8")
    assert _imports_streamlit(copy_source) is False
    assert _imports_ui_locale(copy_source) is False
    app = (REPO_ROOT / "src" / "excavation_lab" / "web" / "app.py").read_text(encoding="utf-8")
    assert "write_locale_file(" in app
    assert "on_change" not in app


def test_compute_modules_do_not_import_locale() -> None:
    paths = [REPO_ROOT / "src" / "excavation_lab" / "cli.py"]
    paths.extend(sorted((REPO_ROOT / "src" / "excavation_lab" / "models").rglob("*.py")))
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in paths
        if _imports_ui_locale(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_page_id_for_accepts_ids_and_both_catalogs() -> None:
    ru = ui_copy.catalog("ru")
    en = ui_copy.catalog("en")
    for page_id, ru_title, en_title in zip(
        _PAGE_IDS, ru.page_titles, en.page_titles, strict=True
    ):
        assert page_id_for(page_id) == page_id
        assert page_id_for(ru_title) == page_id
        assert page_id_for(en_title) == page_id
        assert title_for(ru, page_id) == ru_title
        assert title_for(en, page_id) == en_title
    with pytest.raises(ValueError, match="unknown page"):
        page_id_for("nope")
    with pytest.raises(ValueError, match="unknown page"):
        title_for(ru, "nope")


def test_shell_order_and_shared_resolver() -> None:
    template = (
        REPO_ROOT / "src" / "excavation_lab" / "web" / "templates" / "base.html"
    ).read_text(encoding="utf-8")
    order = [
        'id="lab-home"',
        "back_to_projects",
        "kicker_workflow",
        "<nav",
        "{% block toolbar %}",
        'action="/theme"',
        'action="/locale"',
        "theme_display_only",
        "locale_caption",
    ]
    sidebar = template.split("<aside", 1)[1].split("</aside>", 1)[0]
    assert "kicker_appearance" not in sidebar
    assert 'name="theme"' not in sidebar
    assert 'name="locale"' not in sidebar
    positions = [template.index(token) for token in order]
    assert positions == sorted(positions)
    assert "PAGE_DESCRIPTIONS" not in template
    app = (REPO_ROOT / "src" / "excavation_lab" / "web" / "app.py").read_text(encoding="utf-8")
    assert "PAGE_DESCRIPTIONS" not in app
    shell = (REPO_ROOT / "src" / "excavation_lab" / "web" / "shell.py").read_text(encoding="utf-8")
    context = shell[shell.index("def page_shell") :]
    assert context.index("resolve_locale(") < context.index("nav_links")
    nav = Path(navigation.__file__).read_text(encoding="utf-8")
    assert "st.navigation" not in nav
    assert "import streamlit" not in nav
    assert [item[0] for item in navigation.PAGE_ROUTES] == [
        "video",
        "data",
        "training",
        "analysis",
        "quality",
        "actions",
        "events",
    ]
    assert ui_copy.PRODUCT_TITLE == "Cadence"
    assert "<title>{{ product_title }}</title>" in template


@pytest.mark.parametrize("title", [ui_copy.PAGE_VIDEO, "video", "Video"])
def test_render_frame_maps_onto_the_russian_catalog(tmp_path: Path, title: str) -> None:
    assert page_id_for(title) == "video"
    book = ui_copy.catalog("ru")
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/video", follow_redirects=False)
    assert response.status_code == 200
    text = response.text
    assert "<h1" in text
    assert book.page_video in text
    assert book.lead("video") in text
    assert ui_copy.PAGE_DESCRIPTIONS[ui_copy.PAGE_VIDEO] in text
    aside = _aside(text)
    main = _main(text)
    assert text.index('id="lab-home"') < text.index("WORKFLOW")
    assert 'name="theme"' not in aside
    assert 'name="locale"' not in aside
    assert 'name="theme"' in main
    assert 'name="locale"' in main
    assert _button_pressed(text, "Light") == "false"
    assert _button_pressed(text, "Dark") == "true"
    assert _button_pressed(text, "Русский") == "true"
    assert _button_pressed(text, "English") == "false"
    assert 'aria-label="Язык"' in text
    assert text.index(book.theme_display_only) < text.index(book.locale_caption)
    assert not (tmp_path / "ui" / "locale.txt").exists()
    assert '<p class="lab-label">pilot</p>' in text


def test_english_file_renders_english_chrome_without_rewriting(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    ui = workspace / "ui"
    ui.mkdir(parents=True)
    (ui / "locale.txt").write_text("en", encoding="utf-8")
    list_catalogs(workspace)
    create_project("pilot", workspace, catalog_id="excavation")
    write_project_file(workspace, "pilot")
    book = ui_copy.catalog("en")
    with TestClient(create_app(workspace)) as client:
        response = client.get("/video", follow_redirects=False)
    assert response.status_code == 200
    assert (ui / "locale.txt").read_text(encoding="utf-8") == "en"
    text = response.text
    assert 'lang="en"' in text
    assert book.page_video in text
    assert book.lead("video") in text
    assert book.theme_display_only in text
    aside = _aside(text)
    main = _main(text)
    assert text.index(book.theme_display_only) < text.index(book.locale_caption)
    assert text.index('id="lab-home"') < text.index("WORKFLOW")
    assert 'name="theme"' not in aside
    assert 'name="locale"' not in aside
    assert 'name="theme"' in main
    assert 'name="locale"' in main
    assert "Cadence" in text
    assert _button_pressed(text, "Light") == "false"
    assert _button_pressed(text, "Dark") == "true"
    assert _button_pressed(text, "English") == "true"
    assert _button_pressed(text, "Русский") == "false"
    assert 'aria-label="Language"' in text
    assert '<p class="lab-label">pilot</p>' in text


def test_language_switch_on_empty_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    book = ui_copy.catalog("en")
    with TestClient(create_app(workspace)) as client:
        before = client.get("/")
        assert before.status_code == 200
        assert ui_copy.catalog("ru").home_empty in before.text
        assert 'id="project-open-' not in before.text
        assert not (workspace / "ui" / "locale.txt").exists()
        assert not (workspace / "ui" / "theme.txt").exists()
        switched = client.post(
            "/locale",
            data={"locale": "en", "next": "/video"},
            follow_redirects=True,
        )
    assert switched.status_code == 200
    assert switched.history[0].status_code == 303
    assert urlparse(switched.history[0].headers["location"]).path == "/video"
    assert switched.history[1].status_code == 303
    assert urlparse(switched.history[1].headers["location"]).path == "/"
    written = workspace / "ui" / "locale.txt"
    assert written.read_bytes() == b"en\n"
    assert written.resolve().is_relative_to(workspace.resolve())
    assert not written.resolve().is_relative_to(_REPO_WORKSPACE.resolve())
    assert 'lang="en"' in switched.text
    assert 'value="en"' in switched.text
    assert "English" in switched.text
    text = switched.text
    assert _h1(text) == "Projects"
    assert book.lead("video") not in _main(text)
    assert _button_pressed(text, "Dark") == "true"
    assert _button_pressed(text, "English") == "true"
    assert 'name="theme"' in text
    assert text.index(book.theme_display_only) < text.index(book.locale_caption)
    assert book.home_empty in text
    assert not (workspace / "ui" / "theme.txt").exists()
    assert not (workspace / "ui" / "project.txt").exists()


def test_unknown_page_text_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown page"):
        page_id_for("nope")
    with TestClient(create_app(tmp_path)) as client:
        missing = client.get("/nope")
    assert missing.status_code == 404


def test_escaped_symlink_shows_sidebar_error(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "ui").symlink_to(outside, target_is_directory=True)
    book = ui_copy.catalog("ru")
    with TestClient(create_app(workspace)) as client:
        response = client.post(
            "/locale",
            data={"locale": "en", "next": "/events"},
            follow_redirects=True,
        )
    assert response.status_code == 200
    assert book.locale_write_failed in response.text
    assert not (outside / "locale.txt").exists()
    assert list(outside.glob(".locale-*")) == []
    assert _button_pressed(response.text, "Русский") == "true"
    assert _button_pressed(response.text, "English") == "false"
    assert _button_pressed(response.text, "Dark") == "true"


def test_english_catalog_translations_have_no_cyrillic() -> None:
    ru = ui_copy.catalog("ru")
    en = ui_copy.catalog("en")
    assert en.not_calculated == "not calculated"
    assert ru.not_calculated == "не рассчитано"
    assert en.page_titles == _ENGLISH_TITLES
    assert en.language_options == ru.language_options == ("Русский", "English")
    for name in _INVARIANT_FIELDS:
        assert getattr(en, name) == getattr(ru, name), name
    assert en.product_title == "Cadence"
    assert (en.theme_light, en.theme_dark) == ("Light", "Dark")
    assert (en.kicker_project, en.kicker_workflow, en.kicker_appearance) == (
        "PROJECT",
        "WORKFLOW",
        "APPEARANCE",
    )
    assert en.not_evaluated_caption == "not evaluated"
    assert en.seconds_suffix == "s"
    assert ru.seconds_suffix == "с"
    for field in fields(ui_copy.Catalog):
        _assert_translated(getattr(ru, field.name), getattr(en, field.name), field.name)
    for name, args in _CATALOG_CALLS:
        _assert_translated(getattr(ru, name)(*args), getattr(en, name)(*args), name)


def test_quality_constant_ignores_an_english_locale_file(tmp_path: Path) -> None:
    from excavation_lab.models import quality

    workspace = tmp_path / "ws"
    ui = workspace / "ui"
    ui.mkdir(parents=True)
    path = ui / "locale.txt"
    path.write_text("en\n", encoding="utf-8")
    source = Path(quality.__file__).read_text(encoding="utf-8")
    copy_source = Path(ui_copy.__file__).read_text(encoding="utf-8")
    assert _imports_ui_locale(source) is False
    assert _imports_streamlit(copy_source) is False
    session: dict[str, object] = {}
    assert read_locale_file(workspace) == "en"
    assert resolve_locale(workspace, session) == "en"
    assert session["lab_locale"] == "en"
    assert path.read_bytes() == b"en\n"
    assert quality.NOT_CALCULATED == "не рассчитано"
    assert quality.NOT_CALCULATED == ui_copy.catalog("ru").not_calculated
    assert not path.resolve().is_relative_to(_REPO_WORKSPACE.resolve())


def test_english_pages_reject_leftover_cyrillic(tmp_path: Path) -> None:
    from excavation_lab.storage import create_project

    workspace = tmp_path / "ws"
    workspace.mkdir()
    with TestClient(create_app(workspace)) as client:
        first = client.get("/video", follow_redirects=False)
        assert first.status_code == 303
        assert urlparse(first.headers["location"]).path == "/"
        assert not (workspace / "ui" / "locale.txt").exists()
        assert 'class="lab-workflow"' not in first.text
        switched = client.post(
            "/locale",
            data={"locale": "en", "next": "/"},
            follow_redirects=False,
        )
        assert switched.status_code == 303
        written = workspace / "ui" / "locale.txt"
        assert written.read_bytes() == b"en\n"
        assert written.resolve().is_relative_to(workspace.resolve())
        assert not written.resolve().is_relative_to(_REPO_WORKSPACE.resolve())
        list_catalogs(workspace)
        create_project("pilot", workspace, catalog_id="excavation")
        write_project_file(workspace, "pilot")
        opened = client.get("/video", follow_redirects=False)
        assert opened.status_code == 200
        assert 'lang="en"' in opened.text
        assert _h1(opened.text) == "Video"
        assert ui_copy.catalog("en").no_project == "no project"
        _assert_english_chrome_html(opened.text)

        book = ui_copy.catalog("en")
        leftovers: list[str] = []
        seen_titles: list[str] = []
        for path, title, kind in _ENGLISH_ROUTES:
            assert written.read_bytes() == b"en\n"
            page = client.get(path, follow_redirects=False)
            assert page.status_code == 200, title
            html = page.text
            assert written.read_bytes() == b"en\n"
            assert 'lang="en"' in html
            assert _h1(html) == title
            seen_titles.append(title)
            if kind == "video":
                assert book.choose_file_button in html
                assert 'action="/video/choose"' in html
            else:
                assert "not calculated" in html
            _assert_english_chrome_html(html)
            assert _button_pressed(html, "English") == "true"
            assert _locale_labels(html) == ["Русский", "English"]
            leftovers.extend(f"{title}: {hit}" for hit in _leftover_cyrillic(html))
            for label, value in _lab_metrics(html):
                assert value not in _ZERO_METRIC, (title, label, value)
            if kind == "events":
                assert book.events_unavailable in html
                metrics = _lab_metrics(html)
                assert len(metrics) == 12
                assert [item[0] for item in metrics] == list(book.event_indicator_labels)
                assert [item[1] for item in metrics] == [book.not_calculated] * 12
                assert "<button" not in _main(html).split("<h1", 1)[1]
                _assert_no_plot_html(html)
            if kind == "quality":
                assert "not evaluated" in html
                assert book.not_evaluated_caption in html
                metrics = _lab_metrics(html)
                assert metrics
                for label, value in metrics:
                    assert value == book.not_calculated, label
        assert tuple(seen_titles) == _ENGLISH_TITLES
        assert leftovers == []


def _analysis_time(html: str) -> tuple[str, str]:
    labels = _analysis_time_options(html)
    selected = [item for item in labels if item[2]]
    assert len(selected) == 1
    value, label, _chosen = selected[0]
    return value, label


def _analysis_time_labels(html: str) -> list[str]:
    return [label for _value, label, _chosen in _analysis_time_options(html)]


def _analysis_time_options(html: str) -> list[tuple[str, str, bool]]:
    match = re.search(
        r'<select\b[^>]*id="analysis-time"[^>]*>(.*?)</select>',
        html,
        re.DOTALL,
    )
    assert match is not None
    found: list[tuple[str, str, bool]] = []
    for attrs, label in re.findall(r"<option\b([^>]*)>([^<]*)</option>", match.group(1)):
        value = re.search(r'value="([^"]*)"', attrs)
        assert value is not None
        chosen = re.search(r"\sselected(?:\s|>|$)", attrs) is not None
        found.append((value.group(1), label.strip(), chosen))
    return found


def test_analysis_time_survives_the_locale_switch(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    first, later = 1.5, 3.5
    video_id = _finished_run(workspace, (first, later))
    assert video_id
    ru_suffix = ui_copy.catalog("ru").seconds_suffix
    en_suffix = ui_copy.catalog("en").seconds_suffix
    ru_first = f"{first:g} {ru_suffix}"
    ru_later = f"{later:g} {ru_suffix}"
    en_first = f"{first:g} {en_suffix}"
    en_later = f"{later:g} {en_suffix}"
    with TestClient(create_app(workspace)) as client:
        page = client.get("/analysis")
        assert page.status_code == 200
        value, label = _analysis_time(page.text)
        assert value == f"{first:g}"
        assert label == ru_first
        assert _analysis_time_labels(page.text) == [ru_first, ru_later]
        # The stored value is the later source timestamp, not the formatted label.
        chosen = client.get("/analysis", params={"source_time_s": str(later)})
        assert chosen.status_code == 200
        value, label = _analysis_time(chosen.text)
        assert value == f"{later:g}"
        assert value != label
        assert label == ru_later
        assert ru_later in chosen.text
        switched = client.post(
            "/locale",
            data={"locale": "en", "next": "/analysis"},
            follow_redirects=True,
        )
        assert switched.status_code == 200
        value, label = _analysis_time(switched.text)
        assert value == f"{later:g}"
        assert float(value) == later
        assert float(value) != first
        assert label == en_later
        assert _analysis_time_labels(switched.text) == [en_first, en_later]
        assert en_later in switched.text
        assert ru_later not in switched.text
    written = workspace / "ui" / "locale.txt"
    assert written.read_bytes() == b"en\n"
    assert written.resolve().is_relative_to(workspace.resolve())
    assert not written.resolve().is_relative_to(_REPO_WORKSPACE.resolve())


def _preference(workspace: Path) -> Path:
    return workspace / "ui" / "locale.txt"


def _assert_no_project_html(html: str, visible: str) -> None:
    select = re.search(r"<select\b([^>]*)>(.*?)</select>", html, re.DOTALL)
    assert select is not None
    assert "disabled" in select.group(1)
    options = re.findall(r"<option\b([^>]*)>([^<]*)</option>", select.group(2))
    assert len(options) == 1
    attrs, label = options[0]
    assert "disabled" in attrs
    assert label == visible


def _button_pressed(html: str, label: str) -> str | None:
    for attrs, text in re.findall(r"<button\b([^>]*)>([^<]*)</button>", html):
        if text.strip() != label:
            continue
        match = re.search(r'aria-pressed="(true|false)"', attrs)
        if match is not None:
            return match.group(1)
    return None


def _h1(html: str) -> str:
    match = re.search(r'<h1 class="lab-title">([^<]*)</h1>', html)
    assert match is not None
    return match.group(1).strip()


def _aside(html: str) -> str:
    match = re.search(r'<aside class="lab-sidebar">(.*)</aside>', html, re.DOTALL)
    assert match is not None
    return match.group(1)


def _main(html: str) -> str:
    match = re.search(r'<main class="lab-main" id="content">(.*)</main>', html, re.DOTALL)
    assert match is not None
    return match.group(1)


def _locale_labels(html: str) -> list[str]:
    form = re.search(r'<form method="post" action="/locale">(.*?)</form>', html, re.DOTALL)
    assert form is not None
    return [
        text.strip()
        for _attrs, text in re.findall(r"<button\b([^>]*)>([^<]*)</button>", form.group(1))
    ]


def _lab_metrics(html: str) -> list[tuple[str, str]]:
    return re.findall(
        r'<div class="lab-metric">\s*<span>([^<]*)</span>\s*<b>([^<]*)</b>\s*</div>',
        html,
    )


def _leftover_cyrillic(html: str) -> list[str]:
    cleaned = html
    for token in _ALLOWED_CYRILLIC:
        cleaned = cleaned.replace(token, "")
    hits: list[str] = []
    seen: set[str] = set()
    for match in _CYRILLIC.finditer(cleaned):
        start = max(0, match.start() - 48)
        end = min(len(cleaned), match.end() + 48)
        snippet = " ".join(cleaned[start:end].split())
        if snippet in seen:
            continue
        seen.add(snippet)
        hits.append(snippet)
    return hits


def _assert_no_plot_html(html: str) -> None:
    lowered = html.lower()
    assert "<canvas" not in lowered
    assert "<svg" not in lowered
    assert "lab-chart" not in lowered
    assert "<video" not in lowered


def _assert_english_chrome_html(html: str) -> None:
    assert "Cadence" in html
    assert re.findall(r'<p class="lab-kicker">([^<]*)</p>', html) == ["WORKFLOW"]
    assert _button_pressed(html, "Light") in {"true", "false"}
    assert _button_pressed(html, "Dark") in {"true", "false"}


def _assert_translated(ru_value: object, en_value: object, path: str) -> None:
    if isinstance(ru_value, str):
        assert isinstance(en_value, str), path
        if _CYRILLIC.search(ru_value):
            assert _CYRILLIC.search(en_value) is None, (path, en_value)
            assert en_value != ru_value, path
            return
        assert en_value == ru_value, path
        return
    if isinstance(ru_value, dict):
        assert isinstance(en_value, dict), path
        assert list(ru_value) == list(en_value), path
        for key, item in ru_value.items():
            _assert_translated(item, en_value[key], f"{path}.{key}")
        return
    if isinstance(ru_value, tuple):
        assert isinstance(en_value, tuple), path
        assert len(ru_value) == len(en_value), path
        for index, item in enumerate(ru_value):
            _assert_translated(item, en_value[index], f"{path}[{index}]")
        return
    raise AssertionError(f"{path}: unsupported catalog value {type(ru_value).__name__}")


def _finished_run(workspace: Path, stamps: tuple[float, float]) -> str:
    import importlib.util

    import pyarrow as pa
    import pyarrow.parquet as pq

    from excavation_lab.media.probe import add_video
    from excavation_lab.storage import create_project, project_directory

    list_catalogs(workspace)
    create_project("pilot", workspace, catalog_id="excavation")
    write_project_file(workspace, "pilot")
    clip = workspace / "clip.mp4"
    spec_path = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
    spec = importlib.util.spec_from_file_location("locale_synthetic_mp4", spec_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.write_synthetic_mp4(clip, 0.4, 10)
    added = add_video(workspace, "pilot", clip)
    video_id = added["video_id"]
    run = project_directory(workspace, "pilot") / "runs" / "demo"
    frames = [
        {
            "analyzed": True,
            "source_time_s": stamp,
            "bucket_observation_state": "missing",
            "model_version": "bundle-a",
        }
        for stamp in stamps
    ]
    detections = [
        {
            "class_name": "bucket",
            "score": 0.5,
            "x1": 1.0,
            "y1": 1.0,
            "x2": 2.0,
            "y2": 2.0,
            "track_id_nullable": 1,
            "observation_state": "observed",
            "source_time_s": stamps[0],
        }
    ]
    for name, rows in (
        ("frame_status.parquet", frames),
        ("detections.parquet", detections),
    ):
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
    (run / "run.json").write_text(
        json.dumps({"video_id": video_id, "model_origin": "custom"}),
        encoding="utf-8",
    )
    return video_id


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


def _imports_ui_locale(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "excavation_lab.ui.locale":
                return True
            if node.module == "excavation_lab.ui" and any(
                alias.name == "locale" for alias in node.names
            ):
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "excavation_lab.ui.locale":
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
