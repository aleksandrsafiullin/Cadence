"""Workflow pages render only after project.txt names a listed project."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pytest
from starlette.testclient import TestClient

from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.errors import LabError
from excavation_lab.storage import create_project
from excavation_lab.ui.copy import catalog
from excavation_lab.web.app import create_app
from excavation_lab.web.preferences import (
    choose_project_id,
    read_flash,
    write_flash,
    write_project_file,
)

_PAGES = (
    "/video",
    "/data",
    "/training",
    "/analysis",
    "/quality",
    "/actions",
    "/events",
    "/label",
)


def test_choose_project_id_ignores_a_sole_unentered_project(tmp_path: Path) -> None:
    text = choose_project_id.__doc__ or ""
    assert "empty file" in text
    assert "missing file" in text
    assert "not listed" in text
    assert "does not write" in text
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    projects = [{"project_id": "pilot", "name": "pilot"}]
    assert choose_project_id(tmp_path, projects) is None
    assert not (tmp_path / "ui" / "project.txt").exists()
    write_project_file(tmp_path, "pilot")
    assert choose_project_id(tmp_path, projects) == "pilot"
    write_project_file(tmp_path, "other")
    assert choose_project_id(tmp_path, projects) is None


def test_unentered_pages_redirect_without_consuming_flash(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_flash(tmp_path, {"error": "держится"})
    with TestClient(create_app(tmp_path)) as client:
        for path in _PAGES:
            response = client.get(path, follow_redirects=False)
            assert response.status_code == 303, path
            assert urlparse(response.headers["location"]).path == "/"
            assert 'class="lab-workflow"' not in response.text
            assert 'action="/video/choose"' not in response.text
            assert read_flash(tmp_path) == {"error": "держится"}
        home = client.get("/", follow_redirects=False)
    assert home.status_code == 200
    assert urlparse(home.headers.get("location", "")).path != "/video"
    assert 'id="project-home-error"' in home.text
    assert "держится" in home.text
    assert 'class="lab-workflow"' not in home.text
    assert read_flash(tmp_path) is None
    assert not (tmp_path / "ui" / "project.txt").exists()


def test_video_add_without_a_project_file_bounces_home_with_need_project(
    tmp_path: Path,
) -> None:
    book = catalog("ru")
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    with TestClient(create_app(tmp_path)) as client:
        posted = client.post(
            "/video/add",
            data={"local_path": "clip.mp4"},
            follow_redirects=False,
        )
        assert posted.status_code == 303
        assert urlparse(posted.headers["location"]).path == "/video"
        assert read_flash(tmp_path) == {"error": book.need_project}
        gated = client.get("/video", follow_redirects=False)
        assert gated.status_code == 303
        assert urlparse(gated.headers["location"]).path == "/"
        assert read_flash(tmp_path) == {"error": book.need_project}
        home = client.get("/", follow_redirects=False)
    assert home.status_code == 200
    assert 'id="project-home-error"' in home.text
    assert book.need_project in home.text
    assert 'class="lab-workflow"' not in home.text
    assert read_flash(tmp_path) is None
    assert not (tmp_path / "ui" / "project.txt").exists()


def test_entered_video_page_shows_path_required(tmp_path: Path) -> None:
    book = catalog("ru")
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    with TestClient(create_app(tmp_path)) as client:
        posted = client.post(
            "/video/add",
            data={"local_path": "  "},
            follow_redirects=False,
        )
        assert posted.status_code == 303
        assert urlparse(posted.headers["location"]).path == "/video"
        page = client.get("/video", follow_redirects=False)
    assert page.status_code == 200
    assert book.path_required in page.text
    assert read_flash(tmp_path) is None


def test_entered_pages_stay_on_their_routes(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    with TestClient(create_app(tmp_path)) as client:
        home = client.get("/", follow_redirects=False)
        assert home.status_code == 200
        assert "location" not in home.headers
        for path in _PAGES:
            response = client.get(path, follow_redirects=False)
            assert response.status_code in {200, 404}, path
            assert urlparse(response.headers.get("location", "")).path != "/"


def test_unentered_label_returns_before_label_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")

    def _context(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("label_context")

    monkeypatch.setattr("excavation_lab.web.app.label_context", _context)
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/label?video=clip&frame=1", follow_redirects=False)
    assert response.status_code == 303
    assert urlparse(response.headers["location"]).path == "/"
    assert response.status_code != 404


def test_listing_error_redirects_without_consuming_flash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_flash(tmp_path, {"error": "каталог"})

    def _boom(_workspace: Path) -> list[dict[str, str]]:
        raise LabError("stopped", "остановлено")

    monkeypatch.setattr("excavation_lab.web.app.storage.list_projects", _boom)
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/video", follow_redirects=False)
    assert response.status_code == 303
    assert urlparse(response.headers["location"]).path == "/"
    assert read_flash(tmp_path) == {"error": "каталог"}


def test_frame_and_job_routes_are_not_gated(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    with TestClient(create_app(tmp_path)) as client:
        frame = client.get("/data/frame", follow_redirects=False)
        frames = client.get("/label/frames", follow_redirects=False)
        job = client.get("/jobs/missing/progress", follow_redirects=False)
    assert frame.status_code == 404
    assert "location" not in frame.headers
    assert frames.status_code != 303
    assert frames.headers["content-type"].startswith("application/json")
    assert job.status_code == 404
    assert "location" not in job.headers
    assert not (tmp_path / "ui" / "project.txt").exists()
