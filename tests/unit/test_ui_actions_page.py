"""Actions page: catalog, interval replace, and a stale revision."""

from __future__ import annotations

import re
from pathlib import Path

from starlette.testclient import TestClient

from excavation_lab import storage
from excavation_lab.actions.labels import list_intervals
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.splits import freeze_split, write_blocks
from excavation_lab.ui import navigation
from excavation_lab.ui.copy import action_seed_titles, catalog
from excavation_lab.ui.locale import write_locale_file
from excavation_lab.web.app import create_app
from excavation_lab.web.preferences import write_project_file

_PROJECT = "mvp"
_VIDEO = "vid"
_BLOCKS = [
    RoleBlock(start_s=0, end_s=80, role="train"),
    RoleBlock(start_s=90, end_s=160, role="validation"),
    RoleBlock(start_s=170, end_s=200, role="test"),
]
_CYRILLIC = re.compile(r"[\u0400-\u04FF]")


def test_navigation_docstring_says_seven_pages() -> None:
    text = Path(navigation.__file__).read_text(encoding="utf-8")
    assert "seven pages" in text
    assert [item[0] for item in navigation.PAGE_ROUTES][5] == "actions"


def test_empty_page_without_project_or_video(tmp_path: Path) -> None:
    book = catalog("ru")
    with TestClient(create_app(tmp_path)) as client:
        empty = client.get("/actions", follow_redirects=False)
    assert empty.status_code == 303
    assert empty.headers["location"].endswith("/")
    assert 'id="action-track"' not in empty.text
    assert not (tmp_path / "ui" / "project.txt").exists()
    list_catalogs(tmp_path)
    storage.create_project(_PROJECT, tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, _PROJECT)
    with TestClient(create_app(tmp_path)) as client:
        bare = client.get("/actions")
    assert bare.status_code == 200
    assert book.action_no_video in bare.text
    assert "<video" not in bare.text
    assert book.action_seed in bare.text
    assert book.action_step_train_wait in bare.text
    assert book.action_step_results_wait in bare.text
    train = bare.text.split('id="step-train"', 1)[1].split('id="step-results"', 1)[0]
    assert "<button" not in train


def test_catalog_add_edit_and_seed(tmp_path: Path) -> None:
    book = catalog("ru")
    workspace = _project(tmp_path)
    with TestClient(create_app(workspace)) as client:
        refused = client.post("/actions/class", content=b"{}")
        added = client.post(
            "/actions/class",
            json={"name": "swing_loaded", "title": "Как ввели"},
        )
        page = client.get("/actions")
        edited = client.post(
            "/actions/class",
            json={"name": "swing_loaded", "title": "Новая подпись", "edit": True},
        )
        shown = client.get("/actions")
        seeded = client.post("/actions/seed", json={})
    assert refused.status_code == 415
    assert refused.json()["code"] == "unsupported_media_type"
    assert added.status_code == 200
    assert added.json()["classes"][0]["title"] == "Как ввели"
    assert "Как ввели" in page.text
    assert book.action_seed not in page.text
    assert edited.json()["classes"][0]["title"] == "Новая подпись"
    assert "Новая подпись" in shown.text
    assert "Как ввели" not in shown.text
    names = [item["name"] for item in seeded.json()["classes"]]
    assert names[0] == "swing_loaded"
    assert [item["name"] for item in seeded.json()["classes"][1:]] == [
        "digging",
        "dumping",
        "swing_empty",
        "idle",
    ]
    stored = {item["name"]: item["title"] for item in seeded.json()["classes"]}
    assert stored["swing_loaded"] == "Новая подпись"
    assert stored["digging"] == action_seed_titles["ru"]["digging"]


def test_interval_round_trip_uses_source_seconds(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    _clock(workspace, offset=5.0)
    write_blocks(workspace, _PROJECT, _VIDEO, _BLOCKS)
    with TestClient(create_app(workspace)) as client:
        client.post("/actions/seed", json={})
        loaded = client.get("/actions/intervals", params={"video": _VIDEO})
        revision = loaded.json()["revision"]
        saved = client.post(
            "/actions/intervals",
            params={"video": _VIDEO},
            json={
                "revision": revision,
                "intervals": [
                    {
                        "player_start_s": 10.0,
                        "player_end_s": 20.0,
                        "action_name": "digging",
                    }
                ],
            },
        )
        page = client.get("/actions")
    assert loaded.status_code == 200
    assert revision["count"] == 0
    assert len(revision["sha256"]) == 64
    assert saved.status_code == 200
    row = saved.json()["intervals"][0]
    assert row["start_s"] == 15.0
    assert row["end_s"] == 25.0
    assert row["role"] == "train"
    assert "обучение" in row["role_label"]
    assert "черновик" in row["role_label"]
    stored = list_intervals(workspace, _PROJECT, _VIDEO)
    assert [(item.start_s, item.end_s, item.action_name) for item in stored] == [
        (15.0, 25.0, "digging"),
    ]
    assert "15.000" in page.text
    assert "25.000" in page.text
    assert catalog("ru").action_blocks_unfrozen in page.text
    assert 'id="action-blocks-unfrozen"' in page.text
    assert "Копание" in page.text
    assert "swing_loaded" in page.text
    assert "swing_empty" in page.text
    assert "lab-coverage-missing" in page.text


def test_stale_revision_is_409_and_does_not_write(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    _clock(workspace, offset=0.0)
    path = write_blocks(workspace, _PROJECT, _VIDEO, _BLOCKS)
    freeze_split(workspace, _PROJECT, path)
    with TestClient(create_app(workspace)) as client:
        client.post("/actions/seed", json={})
        first = client.get("/actions/intervals", params={"video": _VIDEO})
        original = first.json()["revision"]
        saved = client.post(
            "/actions/intervals",
            params={"video": _VIDEO},
            json={
                "revision": original,
                "intervals": [
                    {"player_start_s": 1.0, "player_end_s": 4.0, "action_name": "idle"}
                ],
            },
        )
        bare = client.post(
            "/actions/intervals",
            params={"video": _VIDEO},
            content=b"{}",
        )
        stale = client.post(
            "/actions/intervals",
            params={"video": _VIDEO},
            json={
                "revision": original,
                "intervals": [
                    {
                        "player_start_s": 40.0,
                        "player_end_s": 50.0,
                        "action_name": "dumping",
                    }
                ],
            },
        )
        page = client.get("/actions")
    assert saved.status_code == 200
    assert bare.status_code == 415
    assert bare.json()["code"] == "unsupported_media_type"
    assert stale.status_code == 409
    assert stale.json()["code"] == "action_labels_stale"
    assert [(item.start_s, item.end_s) for item in list_intervals(workspace, _PROJECT, _VIDEO)] == [
        (1.0, 4.0),
    ]
    table = page.text.split('id="action-table"', 1)[1].split("</table>", 1)[0]
    assert "черновик" not in table
    assert "обучение" in table
    assert catalog("ru").action_reload in page.text
    assert 'id="step-train"' in page.text
    assert "enqueue" not in page.text


def test_english_page_program_strings_have_no_cyrillic(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    write_locale_file(workspace, "en")
    book = catalog("en")
    with TestClient(create_app(workspace)) as client:
        client.post("/actions/seed", json={})
        page = client.get("/actions")
    text = page.text
    assert 'lang="en"' in text
    assert book.page_actions in text
    assert book.action_block_help == "The training block must contain every action."
    assert book.action_block_help in text
    assert "swing_loaded" in text
    assert "swing_empty" in text
    assert book.action_coverage_hint in text
    assert "Digging" in text
    cleaned = text.replace("Русский", "")
    assert _CYRILLIC.search(cleaned) is None


def _project(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project(_PROJECT, workspace, catalog_id="excavation")
    write_project_file(workspace, _PROJECT)
    storage.insert_video(workspace, _PROJECT, _VIDEO, name="clip.mp4")
    return workspace


def _clock(workspace: Path, *, offset: float) -> None:
    with storage.connect(storage.project_sqlite(workspace, _PROJECT)) as connection:
        connection.execute(
            """
            UPDATE videos
            SET duration_s = ?, fps_nominal = ?, fps_average = ?,
                time_base = ?, start_offset_s = ?
            WHERE video_id = ?
            """,
            (200.0, 10.0, 10.0, "1/1000", offset, _VIDEO),
        )
