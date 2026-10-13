"""Data page: freeze lock, snapshot gate, review counts, contact sheet."""

from __future__ import annotations

import asyncio
import importlib.util
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
from starlette.testclient import TestClient

from excavation_lab.datasets.episodes import EpisodeSpec, add_episode
from excavation_lab.datasets.snapshot import SnapshotResult
from excavation_lab.datasets.splits import (
    SAME_SCENE_MESSAGE,
    freeze_split,
    frozen_manifest_path,
    propose_for_project,
    propose_split,
    require_frozen,
    write_proposal,
)
from excavation_lab.datasets.validate import DatasetValidation, ValidationIssue
from excavation_lab.errors import LabError
from excavation_lab.media.probe import add_video
from excavation_lab.storage import (
    create_project,
    insert_video,
    list_episodes,
    project_directory,
)
from excavation_lab.ui import copy as ui_copy
from excavation_lab.ui import data_page
from excavation_lab.web.app import create_app
from excavation_lab.web.uploads import UPLOAD_CAP

_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SYNTHETIC_SPEC = importlib.util.spec_from_file_location(
    "excavation_lab_synthetic_mp4_data_page",
    _SYNTHETIC_PATH,
)
assert _SYNTHETIC_SPEC is not None and _SYNTHETIC_SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SYNTHETIC_SPEC)
_SYNTHETIC_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_mp4 = _SYNTHETIC.write_synthetic_mp4

_PAGE_SOURCE = Path(data_page.__file__).read_text(encoding="utf-8")
_ZERO_TOKEN = re.compile(r"(?<![\d.])0(?:[.,]0+)?%?(?![\d.])")
_THREE = (
    EpisodeSpec("train-ep", "clip", "g-train", 0, 60, "погрузка"),
    EpisodeSpec("val-ep", "clip", "g-val", 72, 92, "погрузка"),
    EpisodeSpec("test-ep", "clip", "g-test", 104, 124, "погрузка"),
)


def _client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace))


def _post(client: TestClient, path: str, **kwargs: object) -> str:
    response = client.post(path, follow_redirects=False, **kwargs)  # type: ignore[arg-type]
    assert response.status_code == 303, response.text
    location = urlparse(response.headers["location"])
    assert location.path == "/data"
    assert location.query == ""
    page = client.get("/data")
    assert page.status_code == 200
    return page.text


def _button_attrs(html: str, label: str) -> str:
    found: list[str] = []
    for attrs, body in re.findall(r"<button\b([^>]*)>(.*?)</button>", html, re.DOTALL):
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", body)).strip()
        if text == label:
            found.append(attrs)
    assert found, label
    return found[0]


def _button_labels(html: str) -> list[str]:
    labels: list[str] = []
    for _attrs, body in re.findall(r"<button\b([^>]*)>(.*?)</button>", html, re.DOTALL):
        labels.append(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", body)).strip())
    return labels


def _contact_names(html: str) -> list[str]:
    match = re.search(r'id="contact-sheet">(.*?)</ul>', html, re.DOTALL)
    assert match is not None
    return re.findall(r"<li>(.*?)</li>", match.group(1), re.DOTALL)


def _import_body(
    filename: str,
    payload: bytes,
    *,
    video_id: str = "clip",
    split: str = "train",
) -> tuple[bytes, str]:
    boundary = "----labbound"
    text = ""
    for name, value in (("video_id", video_id), ("split", split)):
        text += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        )
    text += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/zip\r\n\r\n"
    )
    body = text.encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _raw(
    app: object,
    method: str,
    path: str,
    headers: list[tuple[bytes, bytes]],
    body: bytes = b"",
) -> int:
    sent: list[dict[str, object]] = []
    delivered = False

    async def receive() -> dict[str, object]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"spec_version": "2.3", "version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 80),
        "root_path": "",
        "state": {},
    }
    asyncio.run(app(scope, receive, send))  # type: ignore[operator]
    status = 500
    for message in sent:
        if message["type"] == "http.response.start":
            status = int(message["status"])  # type: ignore[arg-type]
    return status


def _pilot(tmp_path: Path) -> None:
    create_project("pilot", tmp_path)


def _video(tmp_path: Path, video_id: str = "clip") -> None:
    insert_video(tmp_path, "pilot", video_id, name=f"{video_id}.mp4")


def _episodes(tmp_path: Path, specs: tuple[EpisodeSpec, ...]) -> None:
    for spec in specs:
        add_episode(tmp_path, "pilot", spec)


def _freeze(tmp_path: Path, video_id: str) -> None:
    proposal = propose_split(
        video_id,
        [EpisodeSpec("e1", video_id, "g1", 0, 30, "заезд")],
    )
    path = write_proposal(tmp_path, "pilot", proposal)
    freeze_split(tmp_path, "pilot", path)


def test_page_calls_services_and_does_not_draw_boxes() -> None:
    assert "freeze_split" in _PAGE_SOURCE
    assert "require_frozen" in _PAGE_SOURCE
    assert "propose_for_project" in _PAGE_SOURCE
    assert "sample_frames" in _PAGE_SOURCE
    assert "export_cvat" in _PAGE_SOURCE
    assert "import_coco" in _PAGE_SOURCE
    assert "validate_dataset" in _PAGE_SOURCE
    assert "snapshot_dataset" in _PAGE_SOURCE
    assert "add_episode" in _PAGE_SOURCE
    assert "propose_split" not in _PAGE_SOURCE
    assert "shell=True" not in _PAGE_SOURCE
    assert "subprocess" not in _PAGE_SOURCE
    assert "supervision" not in _PAGE_SOURCE
    assert "rectangle" not in _PAGE_SOURCE
    assert "docker" not in _PAGE_SOURCE
    assert "FREEZE_SPLIT" in _PAGE_SOURCE
    assert "starlette" not in _PAGE_SOURCE
    assert "zipfile" not in _PAGE_SOURCE


def test_snapshot_gate_follows_require_frozen(tmp_path: Path) -> None:
    _pilot(tmp_path)
    closed = data_page.snapshot_gate(tmp_path, "pilot")
    assert closed.enabled is False
    assert closed.reason == ui_copy.SNAPSHOT_LOCKED

    _video(tmp_path, "clip")
    still = data_page.snapshot_gate(tmp_path, "pilot")
    assert still.enabled is False
    assert still.reason == ui_copy.SNAPSHOT_LOCKED

    _freeze(tmp_path, "clip")
    opened = data_page.snapshot_gate(tmp_path, "pilot")
    assert opened.enabled is True
    assert opened.reason == ui_copy.SNAPSHOT_READY

    _video(tmp_path, "other")
    blocked = data_page.snapshot_gate(tmp_path, "pilot")
    assert blocked.enabled is False
    assert blocked.reason == ui_copy.SNAPSHOT_LOCKED
    _freeze(tmp_path, "other")
    assert data_page.snapshot_gate(tmp_path, "pilot").enabled is True


def test_missing_review_counts_are_not_zero() -> None:
    text = data_page.review_count_text(None)
    assert f"{ui_copy.COUNT_REVIEWED}: {ui_copy.NOT_CALCULATED}." in text
    assert f"{ui_copy.COUNT_UNREVIEWED}: {ui_copy.NOT_CALCULATED}." in text
    assert f"{ui_copy.COUNT_EXCLUDED}: {ui_copy.NOT_CALCULATED}." in text
    assert f"{ui_copy.COUNT_REVIEWED_EMPTY}: {ui_copy.NOT_CALCULATED}." in text
    assert _ZERO_TOKEN.search(text) is None


def test_reviewed_empty_does_not_include_unreviewed() -> None:
    result = DatasetValidation(
        dataset_id="ds1",
        project_id="pilot",
        path="/tmp/dataset",
        dataset_hash="ab" * 32,
        reviewed=2,
        unreviewed=5,
        excluded=1,
        reviewed_empty=1,
        negatives=1,
    )
    text = data_page.review_count_text(result)
    assert f"{ui_copy.COUNT_REVIEWED}: 2." in text
    assert f"{ui_copy.COUNT_UNREVIEWED}: 5." in text
    assert f"{ui_copy.COUNT_EXCLUDED}: 1." in text
    assert f"{ui_copy.COUNT_REVIEWED_EMPTY}: 1." in text
    assert f"{ui_copy.COUNT_REVIEWED_EMPTY}: 6." not in text
    assert f"{ui_copy.COUNT_REVIEWED_EMPTY}: 7." not in text


def test_import_error_names_index_or_file() -> None:
    indexed = LabError(
        "image_missing",
        "Кадр clip:7, файл clip_7.jpg: изображения нет в архиве.",
    )
    text = data_page.format_import_error(indexed)
    assert "source_frame_index: 7." in text
    assert "clip_7.jpg" in text
    named = LabError("hash_mismatch", "Файл «frame_a.jpg» не совпал.")
    file_only = data_page.format_import_error(named)
    assert "frame_a.jpg" in file_only
    assert "source_frame_index" not in file_only


def test_same_scene_sentence_comes_from_the_split_service() -> None:
    one = propose_split("clip", [EpisodeSpec("e1", "clip", "g1", 0, 30, "заезд")])
    assert data_page.limitation_text(one) == SAME_SCENE_MESSAGE
    three = propose_split("clip", _THREE)
    assert data_page.limitation_text(three) is None


def test_closed_snapshot_does_not_call_snapshot_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pilot(tmp_path)
    _video(tmp_path)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("snapshot_dataset")

    monkeypatch.setattr(data_page, "snapshot_dataset", fail)
    note = data_page.run_snapshot(tmp_path, "pilot")
    assert note.code == "split_not_frozen"
    assert note.text == ui_copy.SNAPSHOT_LOCKED


def test_no_project_shows_unknown_counts_and_a_locked_snapshot(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        page = client.get("/data")
    assert page.status_code == 200
    text = page.text
    assert ui_copy.DATA_EMPTY in text
    assert data_page.review_count_text(None) in text
    attrs = _button_attrs(text, ui_copy.SNAPSHOT_DATASET)
    assert "disabled" in attrs
    assert ui_copy.SNAPSHOT_LOCKED in attrs
    assert _ZERO_TOKEN.search(text) is None


def test_no_video_shows_unknown_counts_and_a_locked_snapshot(tmp_path: Path) -> None:
    _pilot(tmp_path)
    gate = data_page.snapshot_gate(tmp_path, "pilot")
    with _client(tmp_path) as client:
        page = client.get("/data")
        assert page.status_code == 200
        text = page.text
        assert ui_copy.DATA_EMPTY in text
        assert data_page.review_count_text(None) in text
        attrs = _button_attrs(text, ui_copy.SNAPSHOT_DATASET)
        assert "disabled" in attrs
        assert gate.reason in text
        assert gate.enabled is False
        assert _ZERO_TOKEN.search(text) is None
        forced = client.post("/data/snapshot", follow_redirects=False)
    assert forced.status_code == 303
    assert urlparse(forced.headers["location"]).query == ""


def test_no_import_has_unknown_counts_and_a_locked_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pilot(tmp_path)
    _video(tmp_path)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("snapshot_dataset")

    monkeypatch.setattr(data_page, "snapshot_dataset", fail)
    gate = data_page.snapshot_gate(tmp_path, "pilot")
    with _client(tmp_path) as client:
        page = client.get("/data")
        assert page.status_code == 200
        text = page.text
        forced = client.post("/data/snapshot", follow_redirects=False)
    assert f"{ui_copy.COUNT_REVIEWED}: {ui_copy.NOT_CALCULATED}." in text
    assert f"{ui_copy.COUNT_UNREVIEWED}: {ui_copy.NOT_CALCULATED}." in text
    assert f"{ui_copy.COUNT_EXCLUDED}: {ui_copy.NOT_CALCULATED}." in text
    assert f"{ui_copy.COUNT_REVIEWED_EMPTY}: {ui_copy.NOT_CALCULATED}." in text
    assert f"{ui_copy.COUNT_REVIEWED}: 0" not in text
    assert f"{ui_copy.COUNT_UNREVIEWED}: 0" not in text
    assert f"{ui_copy.COUNT_EXCLUDED}: 0" not in text
    assert f"{ui_copy.COUNT_REVIEWED_EMPTY}: 0" not in text
    assert ui_copy.CONTACT_EMPTY in text
    assert "<img" not in text.lower()
    assert 'action="/data/episode"' in text
    assert 'action="/data/propose"' in text
    assert 'action="/data/freeze"' in text
    assert 'action="/data/sample"' in text
    assert 'action="/data/export"' in text
    assert 'action="/data/import"' in text
    assert 'action="/data/snapshot"' in text
    attrs = _button_attrs(text, ui_copy.SNAPSHOT_DATASET)
    assert "disabled" in attrs
    assert gate.reason == ui_copy.SNAPSHOT_LOCKED
    assert ui_copy.SNAPSHOT_LOCKED in attrs
    assert forced.status_code == 303
    assert urlparse(forced.headers["location"]).query == ""


def test_page_shows_validate_counts_separately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    issue = ValidationIssue(
        code="box_outside",
        message="Рамка вне кадра.",
        image_file="clip_4.jpg",
        source_frame_index=4,
    )

    def fake_validate(_workspace: Path, _project_id: str) -> DatasetValidation:
        return DatasetValidation(
            dataset_id="ds1",
            project_id="pilot",
            path="/tmp/dataset",
            dataset_hash="ab" * 32,
            reviewed=2,
            unreviewed=5,
            excluded=1,
            reviewed_empty=1,
            negatives=1,
            errors=[issue],
        )

    monkeypatch.setattr(data_page, "validate_dataset", fake_validate)
    with _client(tmp_path) as client:
        page = client.get("/data")
    assert page.status_code == 200
    text = page.text
    assert f"{ui_copy.COUNT_REVIEWED}: 2." in text
    assert f"{ui_copy.COUNT_UNREVIEWED}: 5." in text
    assert f"{ui_copy.COUNT_EXCLUDED}: 1." in text
    assert f"{ui_copy.COUNT_REVIEWED_EMPTY}: 1." in text
    assert f"{ui_copy.COUNT_REVIEWED_EMPTY}: 6." not in text
    assert "clip_4.jpg" in text
    assert "source_frame_index 4" in text


def test_add_episode_from_the_form(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    with _client(tmp_path) as client:
        text = _post(
            client,
            "/data/episode",
            data={
                "video_id": "clip",
                "episode_id": "ep-form",
                "split_group_id": "g-form",
                "start_s": "0",
                "end_s": "10",
                "label": "погрузка",
            },
        )
    rows = list_episodes(tmp_path, "pilot", "clip")
    assert len(rows) == 1
    assert rows[0]["episode_id"] == "ep-form"
    assert rows[0]["split_group_id"] == "g-form"
    assert rows[0]["start_s"] == 0.0
    assert rows[0]["end_s"] == 10.0
    assert "split_group_id g-form" in text
    assert "ep-form: 0–10 с" in text


def test_too_few_episodes_show_the_same_scene_sentence(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _episodes(tmp_path, (EpisodeSpec("e1", "clip", "g1", 0, 30, "заезд"),))
    with _client(tmp_path) as client:
        text = _post(client, "/data/propose", data={"video_id": "clip"})
    assert text.count(SAME_SCENE_MESSAGE) == 1
    assert "измеряет переносимость" not in text
    assert "transfer" not in text.lower()
    assert ui_copy.GUARDS_EMPTY in text


def test_freeze_shows_hash_and_a_second_click_does_not_rewrite(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _episodes(tmp_path, _THREE)
    proposal = propose_for_project(tmp_path, "pilot", "clip")
    assert proposal.guards
    with _client(tmp_path) as client:
        opened = client.get("/data")
        assert "disabled" in _button_attrs(opened.text, ui_copy.SNAPSHOT_DATASET)
        proposed = _post(client, "/data/propose", data={"video_id": "clip"})
        for line in data_page.guard_lines(proposal):
            assert line in proposed
        for line in data_page.block_lines(proposal):
            assert line in proposed
        for spec in _THREE:
            assert f"split_group_id {spec.split_group_id}" in proposed
        assert SAME_SCENE_MESSAGE not in proposed
        assert "переносимость" not in proposed

        frozen_page = _post(client, "/data/freeze", data={"video_id": "clip"})
        frozen = require_frozen(tmp_path, "pilot", "clip")
        assert frozen.content_hash in frozen_page
        assert ui_copy.SPLIT_HASH in frozen_page
        assert "split_frozen" not in frozen_page
        snapshot = _button_attrs(frozen_page, ui_copy.SNAPSHOT_DATASET)
        assert "disabled" not in snapshot
        assert ui_copy.SNAPSHOT_READY in snapshot
        labels = _button_labels(frozen_page)
        assert ui_copy.FREEZE_SPLIT in labels
        assert all("перезапис" not in label.lower() for label in labels)
        manifest = frozen_manifest_path(tmp_path, "pilot", "clip")
        before = manifest.read_bytes()

        again = _post(client, "/data/freeze", data={"video_id": "clip"})
    assert manifest.read_bytes() == before
    assert "split_frozen" in again
    assert require_frozen(tmp_path, "pilot", "clip").content_hash == frozen.content_hash


def test_sample_buttons_pass_a_plan_or_a_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    recorded: list[dict[str, object]] = []

    def spy(*_args: object, **kwargs: object) -> None:
        recorded.append(kwargs)
        raise LabError("stopped", "остановлено")

    monkeypatch.setattr(data_page, "sample_frames", spy)
    book = ui_copy.catalog("ru")
    with _client(tmp_path) as client:
        form = client.get("/data")
        assert form.status_code == 200
        for plan_id, label in book.sample_plan_labels.items():
            assert f'<option value="{plan_id}">{label}</option>' in form.text
        planned = _post(
            client,
            "/data/sample",
            data={
                "video_id": "clip",
                "plan": "diagnostic",
                "split": "validation",
                "count": "9",
                "spacing_s": "2",
                "sample_mode": "plan",
            },
        )
        counted = _post(
            client,
            "/data/sample",
            data={
                "video_id": "clip",
                "plan": "content",
                "split": "train",
                "count": "4",
                "spacing_s": "0.5",
                "sample_mode": "count",
            },
        )
    assert recorded[0]["plan"] == "diagnostic"
    assert recorded[0]["split"] is None
    assert recorded[0]["count"] is None
    assert recorded[0]["video_id"] == "clip"
    assert recorded[0]["spacing_s"] == 2.0
    assert "strategy" not in recorded[0]
    assert recorded[1]["plan"] is None
    assert recorded[1]["split"] == "train"
    assert recorded[1]["count"] == 4
    assert recorded[1]["spacing_s"] == 0.5
    assert "strategy" not in recorded[1]
    assert "остановлено" in planned
    assert "остановлено" in counted


def test_import_button_shows_the_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    seen: list[Path] = []

    def boom(*args: object, **_kwargs: object) -> None:
        seen.append(Path(str(args[3])))
        raise LabError(
            "image_missing",
            "Кадр clip:7, файл clip_7.jpg: изображения нет в архиве.",
        )

    monkeypatch.setattr(data_page, "import_coco", boom)
    with _client(tmp_path) as client:
        text = _post(
            client,
            "/data/import",
            data={"video_id": "clip", "split": "train"},
            files={"file": ("batch.zip", b"not-a-zip", "application/zip")},
        )
    assert "source_frame_index: 7." in text
    assert "clip_7.jpg" in text
    assert len(seen) == 1
    stored = seen[0]
    assert stored.name == "batch.zip"
    assert stored.is_file()
    assert ".." not in stored.parts
    assert stored.parent == project_directory(tmp_path, "pilot") / "imports"


def test_enabled_snapshot_calls_snapshot_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _freeze(tmp_path, "clip")
    called: list[str] = []

    def fake_snapshot(_workspace: Path, project_id: str) -> SnapshotResult:
        called.append(project_id)
        return SnapshotResult(
            dataset_id="ds1",
            project_id=project_id,
            path=str(tmp_path),
            dataset_hash="ab" * 32,
            split_content_hash="cd" * 32,
        )

    monkeypatch.setattr(data_page, "snapshot_dataset", fake_snapshot)
    with _client(tmp_path) as client:
        page = client.get("/data")
        attrs = _button_attrs(page.text, ui_copy.SNAPSHOT_DATASET)
        assert "disabled" not in attrs
        assert ui_copy.SNAPSHOT_READY in attrs
        _post(client, "/data/snapshot")
    assert called == ["pilot"]


def test_contact_sheet_shows_written_jpegs_only(tmp_path: Path) -> None:
    _pilot(tmp_path)
    clip = tmp_path / "clip.mp4"
    write_synthetic_mp4(clip, 2.0, 5)
    add_video(tmp_path, "pilot", clip, video_id="clip")
    add_episode(tmp_path, "pilot", EpisodeSpec("ep", "clip", "g1", 0, 30, "заезд"))
    proposal = propose_for_project(tmp_path, "pilot", "clip")
    freeze_split(tmp_path, "pilot", write_proposal(tmp_path, "pilot", proposal))
    with _client(tmp_path) as client:
        _post(
            client,
            "/data/sample",
            data={
                "video_id": "clip",
                "plan": "diagnostic",
                "split": "train",
                "count": "2",
                "spacing_s": "0.4",
                "sample_mode": "count",
            },
        )
    paths, error = data_page.contact_jpegs(tmp_path, "pilot", "clip")
    assert error is None
    assert paths
    assert all(path.suffix == ".jpg" and path.stat().st_size > 0 for path in paths)
    stray = paths[0].parent / "demo_box.jpg"
    stray.write_bytes(b"not-a-jpeg")
    with _client(tmp_path) as client:
        page = client.get("/data")
    assert page.status_code == 200
    assert _contact_names(page.text) == [path.name for path in paths]
    assert "demo_box.jpg" not in page.text
    assert "<img" not in page.text.lower()
    assert "<canvas" not in page.text.lower()
    assert data_page.contact_jpegs(tmp_path, "pilot", "clip")[0] == paths


def test_export_posts_split_and_include_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    recorded: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def spy(*args: object, **kwargs: object) -> None:
        recorded.append((args, kwargs))
        raise LabError("stopped", "остановлено")

    monkeypatch.setattr(data_page, "export_cvat", spy)
    with _client(tmp_path) as client:
        _post(
            client,
            "/data/export",
            data={
                "video_id": "clip",
                "split": "validation",
                "include_predictions": "1",
            },
        )
        _post(
            client,
            "/data/export",
            data={"video_id": "clip", "split": "train"},
        )
    assert recorded[0][0][2] == "clip__validation"
    assert recorded[0][1]["include_predictions"] is True
    assert recorded[1][0][2] == "clip__train"
    assert recorded[1][1]["include_predictions"] is False


def test_import_rejects_parent_segments_before_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pilot(tmp_path)
    _video(tmp_path)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("import_coco")

    monkeypatch.setattr(data_page, "import_coco", fail)
    imports = project_directory(tmp_path, "pilot") / "imports"
    with _client(tmp_path) as client:
        for name in ("../batch.zip", "batch/../../x.zip", "..", "..\\batch.zip"):
            body, content_type = _import_body(name, b"PK\x03\x04")
            refused = client.post(
                "/data/import",
                content=body,
                headers={"content-type": content_type},
                follow_redirects=False,
            )
            assert refused.status_code == 303
            assert urlparse(refused.headers["location"]).query == ""
            page = client.get("/data")
            assert ui_copy.catalog("ru").import_bad_name in page.text
    assert not imports.exists() or list(imports.rglob("*")) == []


def test_import_cap_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pilot(tmp_path)
    _video(tmp_path)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("import_coco")

    monkeypatch.setattr(data_page, "import_coco", fail)
    imports = project_directory(tmp_path, "pilot") / "imports"
    app = create_app(tmp_path)
    status = _raw(
        app,
        "POST",
        "/data/import",
        [
            (b"content-length", str(UPLOAD_CAP + 1).encode()),
            (b"content-type", b"multipart/form-data"),
        ],
    )
    assert status == 413
    assert not imports.exists() or list(imports.rglob("*")) == []
    monkeypatch.setattr("excavation_lab.web.uploads.UPLOAD_CAP", 4)
    monkeypatch.setattr(
        "excavation_lab.web.uploads.declared_over_cap",
        lambda _value: False,
    )
    with _client(tmp_path) as client:
        body, content_type = _import_body("batch.zip", b"12345")
        counted = client.post(
            "/data/import",
            content=body,
            headers={"content-type": content_type},
            follow_redirects=False,
        )
    assert counted.status_code == 413
    assert not imports.exists() or list(imports.rglob("*")) == []


def test_import_route_does_not_open_the_archive() -> None:
    routes = Path(data_page.__file__).resolve().parents[1] / "web" / "data_routes.py"
    uploads = routes.with_name("uploads.py")
    route_source = routes.read_text(encoding="utf-8")
    upload_source = uploads.read_text(encoding="utf-8")
    assert "zipfile" not in route_source
    assert "ZipFile" not in route_source
    assert "request.body(" not in route_source
    assert "ZipFile" not in upload_source
