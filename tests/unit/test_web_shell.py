"""Shell routes: navigation, project file, theme, locale, and the events gate."""

from __future__ import annotations

import ast
import re
import sqlite3
from pathlib import Path
from urllib.parse import urlparse

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient

from excavation_lab.datasets.catalog import list_catalogs, read_catalog
from excavation_lab.errors import PathNotAllowedError
from excavation_lab.jobs import lab_database
from excavation_lab.storage import create_project
from excavation_lab.ui.copy import catalog
from excavation_lab.ui.locale import read_locale_file, write_locale_file
from excavation_lab.ui.navigation import PAGE_ROUTES
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
from excavation_lab.web.shell import RETURN_PATHS, page_shell

_PATHS = ("/video", "/data", "/training", "/analysis", "/quality", "/actions", "/events")
_HEX_COLOR = re.compile(r"#[0-9A-Fa-f]{3,8}\b")
_WEB = Path(__file__).resolve().parents[2] / "src" / "excavation_lab" / "web"
_SHELL_KEYS = {
    "lang",
    "theme",
    "product_title",
    "skip_label",
    "kicker_project",
    "kicker_workflow",
    "kicker_appearance",
    "project_label",
    "project_help",
    "no_project",
    "projects",
    "project_id",
    "project_name",
    "project_error",
    "nav_links",
    "stages",
    "current_path",
    "theme_label",
    "theme_light",
    "theme_dark",
    "theme_display_only",
    "locale_label",
    "locale_caption",
    "language_russian",
    "language_english",
    "notice",
    "page_title",
    "lead",
    "empty_state",
    "events_unavailable",
    "events_unavailable_caption",
    "events_v02_caption",
    "event_indicator_labels",
    "not_calculated",
    "catalogs",
    "catalog_classes",
    "create_project_heading",
    "create_project_name",
    "create_project_button",
    "catalog_existing",
    "catalog_new",
    "catalog_id_label",
    "catalog_copy_label",
    "catalog_copy_none",
    "catalog_none",
    "add_class_row",
    "class_name_column",
    "class_title_column",
    "class_box_column",
    "catalog_classes_heading",
    "rename_project_label",
    "rename_project_button",
    "delete_project_button",
    "delete_project_confirm",
    "back_to_projects",
}


def test_sidebar_marks_one_current_link(tmp_path: Path) -> None:
    book = catalog("ru")
    titles = book.page_titles
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
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


def test_page_shell_keeps_context_keys_and_label_is_not_a_nav_entry(tmp_path: Path) -> None:
    book = catalog("ru")
    english = catalog("en")
    application = create_app(tmp_path)
    data_shell = page_shell(_request(application, "/data"), "data", {})
    assert set(data_shell) == _SHELL_KEYS
    assert data_shell["page_title"] == book.page_data
    assert data_shell["lead"] == book.lead("data")
    assert [item["path"] for item in data_shell["nav_links"]] == list(_PATHS)  # type: ignore[index]
    label_shell = page_shell(_request(application, "/label"), "label", {"error": "сбой"})
    assert set(label_shell) == _SHELL_KEYS
    assert label_shell["page_title"] == book.label_page_title
    assert label_shell["current_path"] == "/label"
    assert {"/", "/label", "/video"} <= RETURN_PATHS
    video_shell = page_shell(_request(application, "/video"), "video", {})
    assert video_shell["current_path"] == "/video"
    assert label_shell["notice"] == "сбой"
    paths = [item["path"] for item in label_shell["nav_links"]]  # type: ignore[index]
    assert paths == [path for _page_id, path, _attr in PAGE_ROUTES]
    assert "/label" not in paths
    current = [item["path"] for item in label_shell["nav_links"] if item["current"]]  # type: ignore[index]
    assert current == ["/data"]
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    with TestClient(application) as client:
        page = client.get("/label?video=clip-1", follow_redirects=False)
        assert page.status_code == 404
        assert page.cookies.get("lab_video") == "clip-1"
        links = _nav(page.text)
        assert [href for href, _title, _active in links] == list(_PATHS)
        assert [href for href, _title, active in links if active] == ["/data"]
        assert book.label_page_title in page.text
        assert book.label_loading in page.text
        assert book.label_back_to_data in page.text
        assert 'href="/data?video=clip-1#step-label"' in page.text
        main = page.text.split("<main", 1)[1]
        assert book.label_loading in main
        refused = client.get("/label?video=a%3Bb", follow_redirects=False)
        assert "lab_video" not in refused.headers.get("set-cookie", "")
        write_locale_file(tmp_path, "en")
        english_page = client.get("/label")
    assert english.label_page_title in english_page.text
    assert english.label_loading in english_page.text
    assert english.label_back_to_data in english_page.text
    assert book.label_loading not in english_page.text
    assert 'lang="en"' in english_page.text


def test_stepper_markup_on_video_data_and_actions(tmp_path: Path) -> None:
    book = catalog("ru")
    english = catalog("en")
    hrefs = ("/video", "/data", "/training", "/analysis", "/actions")
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    with TestClient(create_app(tmp_path)) as client:
        for path in ("/video", "/data", "/actions"):
            page = client.get(path, follow_redirects=False)
            assert page.status_code == 200
            block = _workflow(page.text)
            assert block.count('data-workflow="waiting"') == 4
            assert block.count('data-workflow="current"') == 1
            assert 'data-workflow="done"' not in block
            assert book.add_local_video in block
            assert book.need_project not in block
            assert 'href="/events"' not in block
            for href in hrefs:
                assert f'href="{href}"' in block
        write_locale_file(tmp_path, "en")
        english_page = client.get("/video", follow_redirects=False)
    block = _workflow(english_page.text)
    assert english.add_local_video in block
    assert book.need_project not in block
    assert 'lang="en"' in english_page.text


def test_stepper_highlights_the_open_page(tmp_path: Path) -> None:
    from excavation_lab.datasets.catalog import list_catalogs
    from excavation_lab.storage import create_project

    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    with TestClient(create_app(tmp_path)) as client:
        video = _workflow(client.get("/video", follow_redirects=False).text)
        data = _workflow(client.get("/data").text)
        actions = _workflow(client.get("/actions").text)
        quality = _workflow(client.get("/quality").text)
        events = _workflow(client.get("/events").text)
        label = _workflow(client.get("/label").text)
    assert video.count('data-workflow="current"') == 1
    assert 'aria-current="page"' in _stage_item(video, "/video")
    assert 'aria-current="page"' in _stage_item(data, "/data")
    assert 'aria-current="page"' in _stage_item(actions, "/actions")
    assert 'aria-current="page"' in _stage_item(label, "/data")
    assert 'aria-current="page"' not in quality
    assert 'href="/events"' not in events


def test_reads_do_not_create_preference_files(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    assert read_theme_file(workspace) is None
    assert read_locale_file(workspace) is None
    assert read_project_file(workspace) is None
    assert read_flash(workspace) is None
    assert not (workspace / "ui").exists()
    with TestClient(create_app(workspace)) as client:
        response = client.get("/quality", follow_redirects=False)
        home = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert _path(response.headers["location"]) == "/"
    assert home.status_code == 200
    assert 'data-theme="dark"' in home.text
    assert 'lang="ru"' in home.text
    assert not (workspace / "ui").exists()


def test_project_picker_writes_a_listed_id_and_ignores_the_rest(tmp_path: Path) -> None:
    book = catalog("ru")
    with TestClient(create_app(tmp_path)) as client:
        empty = client.get("/")
        assert book.home_empty in empty.text
        assert 'id="project-create"' in empty.text
        ignored = client.post(
            "/project",
            data={"project": "pilot", "next": "/"},
            follow_redirects=False,
        )
        assert ignored.status_code == 303
        assert not (tmp_path / "ui" / "project.txt").exists()

        list_catalogs(tmp_path)
        create_project("pilot", tmp_path, catalog_id="excavation")
        list_catalogs(tmp_path)
        create_project("west", tmp_path, catalog_id="excavation")
        listed = client.get("/", follow_redirects=False)
        assert 'id="project-open-pilot"' in listed.text
        assert 'id="project-open-west"' in listed.text
        assert book.home_current not in listed.text
        chosen = client.post(
            "/project",
            data={"project": "west", "next": "/training"},
            follow_redirects=False,
        )
        assert chosen.status_code == 303
        assert _path(chosen.headers["location"]) == "/video"
        assert read_project_file(tmp_path) == "west"
        assert (tmp_path / "ui" / "project.txt").read_bytes() == b"west\n"
        page = client.get("/training")
        assert 'id="lab-home"' in page.text
        assert ">west</p>" in page.text
        unknown = client.post(
            "/project",
            data={"project": "other", "next": "/training"},
            follow_redirects=True,
        )
        assert unknown.status_code == 200
        assert unknown.url.path == "/"
        assert (tmp_path / "ui" / "project.txt").read_bytes() == b"west\n"
        assert book.home_current in unknown.text
        assert 'name="project" value="west"' in unknown.text


def test_project_select_submits_on_change_without_a_button(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/video", follow_redirects=False)
        assert 'id="lab-home"' in page.text
        assert 'href="/"' in page.text
        assert 'id="lab-project"' not in page.text
        assert 'id="project-manager"' not in page.text
        assert "project_shell.js" not in page.text
        home = client.get("/", follow_redirects=False)
    form = _form_by_action(home.text, "/project")
    assert 'id="project-open-pilot"' in form
    assert 'type="submit"' in form
    assert "<select" not in form


def test_project_write_refuses_an_escaped_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    list_catalogs(workspace)
    create_project("pilot", workspace, catalog_id="excavation")
    (workspace / "ui").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathNotAllowedError):
        write_project_file(workspace, "pilot")
    assert list(outside.iterdir()) == []
    with TestClient(create_app(workspace)) as client:
        response = client.post(
            "/project",
            data={"project": "pilot", "next": "/video"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert _path(response.headers["location"]) == "/"
        assert list(outside.iterdir()) == []
        page = client.get("/")
        assert page.status_code == 200
        assert catalog("ru").project_write_failed in page.text


def test_create_write_failure_stays_home_even_if_next_is_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk")

    monkeypatch.setattr(web_app, "write_project_file", _boom)
    list_catalogs(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        posted = client.post(
            "/project/create",
            data={
                "name": "pit",
                "catalog_mode": "existing",
                "catalog_id": "excavation",
                "next": "/video",
            },
            follow_redirects=False,
        )
        assert posted.status_code == 303
        location = _path(posted.headers["location"])
        assert location == "/"
        assert location != "/video"
        assert not (tmp_path / "ui" / "project.txt").exists()
        assert (tmp_path / "projects" / "pit").is_dir()
        page = client.get("/", follow_redirects=False)
    assert page.status_code == 200
    error = re.search(
        r'<p class="lab-alert" id="project-home-error"[^>]*>(.*?)</p>',
        page.text,
    )
    assert error is not None
    assert catalog("ru").project_write_failed in error.group(1)


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


def test_label_return_keeps_video_and_frame(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    with TestClient(create_app(tmp_path)) as client:
        page = client.get(
            "/label",
            params={"video": "vid", "frame": "10"},
            follow_redirects=False,
        )
        data = client.get(
            "/data",
            params={"video": "vid", "frame": "10"},
            follow_redirects=False,
        )
        video_only = client.get("/label", params={"video": "vid"}, follow_redirects=False)
        theme = client.post(
            "/theme",
            data={"theme": "light", "next": "/label?video=vid&frame=10"},
            follow_redirects=False,
        )
        locale = client.post(
            "/locale",
            data={"locale": "en", "next": "/label?frame=3&video=vid"},
            follow_redirects=False,
        )
        project = client.post(
            "/locale",
            data={"locale": "ru", "next": "/label?video=vid&frame=0"},
            follow_redirects=False,
        )
        analysis = client.post(
            "/theme",
            data={"theme": "dark", "next": "/analysis"},
            follow_redirects=False,
        )
        extra = client.post(
            "/locale",
            data={"locale": "ru", "next": "/label?video=vid&frame=1&evil=1"},
            follow_redirects=False,
        )
        traversal = client.post(
            "/theme",
            data={"theme": "dark", "next": "/label?video=../secrets&frame=1"},
            follow_redirects=False,
        )
        letters = client.post(
            "/locale",
            data={"locale": "en", "next": "/label?video=vid&frame=1a"},
            follow_redirects=False,
        )
        open_redirect = client.post(
            "/theme",
            data={"theme": "dark", "next": "https://example.com"},
            follow_redirects=False,
        )
    assert page.status_code == 404
    assert 'name="next" value="/label?video=vid&amp;frame=10"' in page.text
    assert 'name="next" value="/label?video=vid"' in video_only.text
    assert 'name="next" value="/data"' in data.text
    assert "/data?video=" not in data.text
    assert theme.headers["location"] == "/label?video=vid&frame=10"
    assert locale.headers["location"] == "/label?video=vid&frame=3"
    assert project.headers["location"] == "/label?video=vid&frame=0"
    assert analysis.headers["location"] == "/analysis"
    assert extra.headers["location"] == "/"
    assert traversal.headers["location"] == "/"
    assert letters.headers["location"] == "/"
    assert open_redirect.headers["location"] == "/"
    assert "example.com" not in open_redirect.headers["location"]


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


def test_open_workspace_lists_excavation_before_a_project(tmp_path: Path) -> None:
    book = catalog("ru")
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/")
    assert page.status_code == 200
    aside, main, after = _regions(page.text)
    assert 'class="lab-workflow"' not in page.text
    assert 'id="project-manager"' not in page.text
    assert 'id="project-create"' not in aside
    assert 'id="project-create"' in main
    assert 'id="project-rename"' not in aside
    assert 'id="project-delete"' not in aside
    assert "data-project-manager-open" not in after
    assert 'id="catalog-new-fields" hidden' in page.text
    values = _option_values(_select_by_id(page.text, "catalog-id"))
    assert values[0] == ""
    assert "excavation" in values
    assert book.catalog_copy_label in page.text
    assert book.catalog_none in page.text
    assert not (tmp_path / "projects").exists()
    assert (tmp_path / "catalogs" / "excavation" / "catalog.yaml").is_file()


def test_create_project_on_excavation_stores_the_id(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path)) as client:
        opened = client.get("/")
        assert "excavation" in _option_values(_select_by_id(opened.text, "catalog-id"))
        created = client.post(
            "/project/create",
            data={
                "name": "pit",
                "catalog_mode": "existing",
                "catalog_id": "excavation",
                "next": "/",
            },
            follow_redirects=True,
        )
    assert created.status_code == 200
    project = tmp_path / "projects" / "pit"
    assert project.is_dir()
    assert list(project.rglob("catalog.yaml")) == []
    assert _catalog_id(project) == "excavation"
    assert read_project_file(tmp_path) == "pit"
    assert created.history[0].status_code == 303
    assert _path(created.history[0].headers["location"]) == "/video"
    assert 'id="lab-home"' in created.text
    assert 'id="project-catalog-classes"' not in created.text
    assert "data-project-manager-open" not in created.text


def test_blank_catalog_keeps_the_name_on_the_create_form(tmp_path: Path) -> None:
    book = catalog("ru")
    with TestClient(create_app(tmp_path)) as client:
        refused = client.post(
            "/project/create",
            data={"name": "pit", "catalog_mode": "existing", "catalog_id": ""},
            follow_redirects=True,
        )
    assert not (tmp_path / "projects" / "pit").exists()
    form = refused.text.split('id="project-create"', 1)[1]
    assert book.catalog_required in form
    assert 'value="pit"' in form


def test_rejected_create_keeps_the_name_next_to_the_button(tmp_path: Path) -> None:
    book = catalog("ru")
    with TestClient(create_app(tmp_path)) as client:
        refused = client.post(
            "/project/create",
            data={"name": "Карьер", "catalog_mode": "existing", "catalog_id": "excavation"},
            follow_redirects=True,
        )
    assert refused.status_code == 200
    assert not (tmp_path / "projects").exists()
    form = refused.text.split('id="project-create"', 1)[1]
    assert book.project_name_required not in form
    assert "Недопустимое имя проекта." in form
    assert 'value="Карьер"' in form
    assert 'id="project-name"' in form
    assert "required" in form


def test_new_catalog_keeps_the_typed_title_in_both_locales(tmp_path: Path) -> None:
    title = "Лопата участка"
    box = "Видимая лопата."
    with TestClient(create_app(tmp_path)) as client:
        created = client.post(
            "/project/create",
            data={
                "name": "east",
                "catalog_mode": "new",
                "new_catalog_id": "pit_tools",
                "copy_from": "",
                "class_name": "shovel",
                "class_title": title,
                "class_box": box,
                "next": "/video",
            },
            follow_redirects=True,
        )
        assert created.status_code == 200
        assert title in created.text
        assert box in created.text
        english = client.post(
            "/locale",
            data={"locale": "en", "next": "/video"},
            follow_redirects=True,
        )
        assert 'lang="en"' in english.text
        assert title in english.text
        assert box in english.text
        assert catalog("en").catalog_classes_heading in english.text
        assert catalog("ru").catalog_classes_heading not in english.text
        russian = client.post(
            "/locale",
            data={"locale": "ru", "next": "/video"},
            follow_redirects=True,
        )
        assert 'lang="ru"' in russian.text
        assert title in russian.text
        assert box in russian.text
    stored = read_catalog(tmp_path, "pit_tools")
    assert [(item.name, item.title, item.box) for item in stored.classes] == [
        ("shovel", title, box)
    ]
    project = tmp_path / "projects" / "east"
    assert list(project.rglob("catalog.yaml")) == []
    assert _catalog_id(project) == "pit_tools"
    seed = tmp_path / "catalogs" / "excavation" / "catalog.yaml"
    assert seed.is_file()
    assert "shovel" not in seed.read_text(encoding="utf-8")
    custom = (tmp_path / "catalogs" / "pit_tools" / "catalog.yaml").read_text(encoding="utf-8")
    for class_name in ("excavator_body", "bucket", "haul_truck", "truck_bed"):
        assert class_name not in custom


def test_create_form_copies_a_catalog_then_adds_a_row(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    source_bytes = (tmp_path / "catalogs" / "excavation" / "catalog.yaml").read_bytes()
    with TestClient(create_app(tmp_path)) as client:
        page = client.post(
            "/project/create",
            data={
                "name": "copied",
                "catalog_mode": "new",
                "new_catalog_id": "pit_plus",
                "copy_from": "excavation",
                "class_name": "marker",
                "class_title": "Метка",
                "class_box": "Видимая метка.",
                "next": "/video",
            },
            follow_redirects=True,
        )
    assert page.status_code == 200
    assert "Метка" in page.text
    assert (tmp_path / "catalogs" / "excavation" / "catalog.yaml").read_bytes() == source_bytes
    source = read_catalog(tmp_path, "excavation")
    copied = read_catalog(tmp_path, "pit_plus")
    assert len(source.classes) == 4
    copied_head = [
        (item.id, item.name, item.title, item.box, item.added_at)
        for item in copied.classes[:4]
    ]
    source_rows = [
        (item.id, item.name, item.title, item.box, item.added_at) for item in source.classes
    ]
    assert copied_head == source_rows
    assert copied.classes[-1].name == "marker"
    assert copied.classes[-1].title == "Метка"
    assert copied.classes[-1].box == "Видимая метка."
    project = tmp_path / "projects" / "copied"
    assert list(project.rglob("catalog.yaml")) == []
    assert _catalog_id(project) == "pit_plus"


def test_submit_without_a_catalog_does_not_create_a_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def _blocked(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        raise AssertionError("create_project")

    monkeypatch.setattr(web_app.storage, "create_project", _blocked)
    with TestClient(create_app(tmp_path)) as client:
        refused = client.post(
            "/project/create",
            data={
                "name": "ghost",
                "catalog_mode": "existing",
                "catalog_id": "",
                "next": "/",
            },
            follow_redirects=False,
        )
        assert refused.status_code == 303
        assert calls == []
        assert not (tmp_path / "projects").exists()
        assert not (tmp_path / "catalogs").exists()
        page = client.get("/")
        write_locale_file(tmp_path, "en")
        english = client.post(
            "/project/create",
            data={
                "name": "ghost",
                "catalog_mode": "new",
                "new_catalog_id": "tools",
                "copy_from": "",
                "class_name": "",
                "class_title": "",
                "class_box": "",
                "next": "/",
            },
            follow_redirects=True,
        )
    assert catalog("ru").catalog_required in page.text
    assert 'id="project-home-error"' in page.text
    assert "data-project-manager-open" not in page.text
    assert english.status_code == 200
    assert catalog("en").catalog_required in english.text
    assert catalog("ru").catalog_required not in english.text
    assert calls == []
    assert not (tmp_path / "projects").exists()
    assert not (tmp_path / "projects" / "ghost").exists()
    tools = tmp_path / "catalogs" / "tools"
    assert not tools.exists()


def test_partial_class_row_does_not_create_a_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def _blocked(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        raise AssertionError("create_project")

    monkeypatch.setattr(web_app.storage, "create_project", _blocked)
    with TestClient(create_app(tmp_path)) as client:
        refused = client.post(
            "/project/create",
            data={
                "name": "ghost",
                "catalog_mode": "new",
                "new_catalog_id": "tools",
                "class_name": "shovel",
                "class_title": "Лопата",
                "class_box": "",
                "next": "/",
            },
            follow_redirects=False,
        )
        assert refused.status_code == 303
        assert not (tmp_path / "projects").exists()
        assert not (tmp_path / "catalogs").exists()
        page = client.get("/")
    assert page.status_code == 200
    assert catalog("ru").class_row_incomplete in page.text
    assert calls == []
    assert not (tmp_path / "projects").exists()
    assert not (tmp_path / "catalogs" / "tools").exists()


def test_rename_updates_the_name_shown_in_the_shell(tmp_path: Path) -> None:
    english = catalog("en")
    list_catalogs(tmp_path)
    create_project("pit", tmp_path, catalog_id="excavation")
    create_project("east", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pit")
    with TestClient(create_app(tmp_path)) as client:
        opened = client.get("/")
        aside, main, _after = _regions(opened.text)
        assert 'id="project-create"' in main
        assert 'action="/project/rename"' not in aside
        assert 'action="/project/delete"' not in aside
        assert 'id="project-rename-pit"' in main
        assert 'id="project-delete-pit"' in main
        assert 'id="project-rename-east"' in main
        assert _input_value(opened.text, "rename-name-pit") == "pit"
        switched = client.post(
            "/locale",
            data={"locale": "en", "next": "/"},
            follow_redirects=True,
        )
        assert english.rename_project_button in switched.text
        assert english.delete_project_button in switched.text
        assert english.delete_project_confirm in switched.text
        renamed = client.post(
            "/project/rename",
            data={"project": "pit", "name": "Pit east", "next": "/"},
            follow_redirects=True,
        )
        data_page = client.get("/data")
        taken = client.post(
            "/project/rename",
            data={"project": "pit", "name": "east", "next": "/"},
            follow_redirects=True,
        )
    assert renamed.status_code == 200
    assert (tmp_path / "projects" / "pit").is_dir()
    assert not (tmp_path / "projects" / "Pit east").exists()
    assert read_project_file(tmp_path) == "pit"
    assert _input_value(renamed.text, "rename-name-pit") == "Pit east"
    assert "Pit east" in renamed.text
    assert english.home_current in renamed.text
    assert "data-project-manager-open" not in renamed.text
    assert "Pit east" in data_page.text
    assert 'id="lab-home"' in data_page.text
    assert 'id="project-rename-pit"' not in data_page.text
    assert "Проект «east» уже существует." in taken.text
    assert 'id="project-home-error"' in taken.text
    assert _input_value(taken.text, "rename-name-pit") == "Pit east"


def test_delete_removes_the_project_and_clears_selection(tmp_path: Path) -> None:
    book = catalog("ru")
    list_catalogs(tmp_path)
    for name in ("alpha", "beta", "gamma"):
        create_project(name, tmp_path, catalog_id="excavation")
    catalog_bytes = (tmp_path / "catalogs" / "excavation" / "catalog.yaml").read_bytes()
    write_project_file(tmp_path, "alpha")
    with TestClient(create_app(tmp_path)) as client:
        opened = client.get("/")
        assert 'name="project" value="alpha"' in opened.text
        assert book.home_current in opened.text
        refused = client.post(
            "/project/delete",
            data={"project": "alpha", "next": "/"},
            follow_redirects=True,
        )
        assert (tmp_path / "projects" / "alpha").is_dir()
        assert book.delete_confirm_required in refused.text
        assert 'id="project-home-error"' in refused.text
        assert "data-project-manager-open" not in refused.text
        assert read_project_file(tmp_path) == "alpha"
        deleted = client.post(
            "/project/delete",
            data={"project": "alpha", "confirm": "yes", "next": "/"},
            follow_redirects=True,
        )
    assert deleted.status_code == 200
    assert not (tmp_path / "projects" / "alpha").exists()
    assert (tmp_path / "projects" / "beta").is_dir()
    assert (tmp_path / "projects" / "gamma").is_dir()
    assert read_project_file(tmp_path) is None
    assert book.home_current not in deleted.text
    assert "data-project-manager-open" not in deleted.text
    assert 'name="project" value="alpha"' not in deleted.text
    aside, main, _after = _regions(deleted.text)
    assert 'action="/project/rename"' not in aside
    assert 'action="/project/delete"' not in aside
    assert 'id="project-rename-beta"' in main
    assert 'id="project-delete-gamma"' in main
    assert 'id="project-create"' not in aside
    assert 'id="project-create"' in main
    assert (tmp_path / "catalogs" / "excavation" / "catalog.yaml").read_bytes() == catalog_bytes


@pytest.mark.parametrize("status", ["queued", "running"])
def test_delete_refuses_while_a_workspace_job_is_active(
    tmp_path: Path,
    status: str,
) -> None:
    list_catalogs(tmp_path)
    create_project("alpha", tmp_path, catalog_id="excavation")
    create_project("beta", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "alpha")
    _hold_lock(tmp_path, "alpha", status)
    with TestClient(create_app(tmp_path)) as client:
        refused = client.post(
            "/project/delete",
            data={"project": "alpha", "confirm": "yes", "next": "/data"},
            follow_redirects=True,
        )
    assert refused.status_code == 200
    assert (tmp_path / "projects" / "alpha").is_dir()
    state = "в очереди" if status == "queued" else "выполняется"
    assert f"Проект «alpha» нельзя удалить: задание {state}." in refused.text
    assert 'id="project-home-error"' in refused.text
    assert "data-project-manager-open" not in refused.text
    assert read_project_file(tmp_path) == "alpha"
    connection = sqlite3.connect(lab_database(tmp_path))
    try:
        row = connection.execute(
            "SELECT project_id, status FROM workspace_lock WHERE singleton = 1"
        ).fetchone()
    finally:
        connection.close()
    assert row == ("alpha", status)


def test_handlers_pass_an_empty_session_into_the_resolvers() -> None:
    source = Path(web_app.__file__).read_text(encoding="utf-8")
    shell = (_WEB / "shell.py").read_text(encoding="utf-8")
    assert "resolve_theme(workspace, session={})" in shell
    assert "resolve_locale(workspace, session={})" in shell
    assert "resolve_locale(workspace, session={})" in source


def _form_by_action(html: str, action: str) -> str:
    match = re.search(
        rf'<form\b[^>]*\baction="{re.escape(action)}"[^>]*>.*?</form>',
        html,
        re.DOTALL,
    )
    assert match is not None
    return match.group(0)


def _regions(html: str) -> tuple[str, str, str]:
    before, _, rest = html.partition("<main")
    main, _, after = rest.partition("</main>")
    return before, main, after


def _request(application: object, path: str) -> Request:
    scope = {
        "type": "http",
        "asgi": {"spec_version": "2.3", "version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 80),
        "root_path": "",
        "app": application,
        "state": {},
    }
    return Request(scope)


def _workflow(html: str) -> str:
    match = re.search(r'<ol class="lab-workflow">.*?</ol>', html, re.DOTALL)
    assert match is not None
    return match.group(0)


def _stage_item(block: str, href: str) -> str:
    for item in re.findall(r"<li\b.*?</li>", block, re.DOTALL):
        if f'href="{href}"' in item:
            return item
    raise AssertionError(href)


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


def _select_by_id(html: str, element_id: str) -> str:
    match = re.search(
        rf'<select\b[^>]*\bid="{element_id}"[^>]*>(.*?)</select>',
        html,
        re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _option_values(body: str) -> list[str]:
    return re.findall(r'value="([^"]*)"', body)


def _hold_lock(workspace: Path, project_id: str, status: str) -> None:
    path = lab_database(workspace)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workspace_lock (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                job_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO workspace_lock (singleton, job_id, project_id, status)
            VALUES (1, 'job-hold', ?, ?)
            """,
            (project_id, status),
        )
        connection.commit()
    finally:
        connection.close()


def _input_value(html: str, element_id: str) -> str:
    match = re.search(rf'<input\b[^>]*\bid="{element_id}"[^>]*>', html)
    assert match is not None
    value = re.search(r'value="([^"]*)"', match.group(0))
    assert value is not None
    return value.group(1)


def _catalog_id(project: Path) -> str:
    connection = sqlite3.connect(project / "project.sqlite")
    try:
        row = connection.execute("SELECT catalog_id FROM projects").fetchone()
    finally:
        connection.close()
    assert row is not None
    return str(row[0])
