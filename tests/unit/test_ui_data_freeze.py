"""Data page: freeze lock, snapshot gate, review counts, contact sheet."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
from starlette.testclient import TestClient

from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import (
    create_catalog,
    list_catalogs,
    require_project_catalog,
)
from excavation_lab.datasets.episodes import EpisodeSpec, add_episode
from excavation_lab.datasets.snapshot import SnapshotResult
from excavation_lab.datasets.splits import (
    SAME_SCENE_MESSAGE,
    SplitProposal,
    canonical_json_bytes,
    freeze_split,
    frozen_manifest_path,
    require_frozen,
    write_blocks,
)
from excavation_lab.datasets.validate import DatasetValidation, ValidationIssue
from excavation_lab.errors import LabError
from excavation_lab.media.probe import add_video
from excavation_lab.storage import (
    connect,
    create_project,
    insert_video,
    list_episodes,
    project_directory,
    project_sqlite,
)
from excavation_lab.ui import copy as ui_copy
from excavation_lab.ui import data_page
from excavation_lab.ui.locale import write_locale_file
from excavation_lab.web.app import create_app
from excavation_lab.web.preferences import write_project_file
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
_BLOCKS = (
    RoleBlock(start_s=0, end_s=60, role="train"),
    RoleBlock(start_s=72, end_s=92, role="validation"),
    RoleBlock(start_s=104, end_s=124, role="test"),
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
    names: list[str] = []
    for inner in re.findall(r"<li>(.*?)</li>", match.group(1), re.DOTALL):
        text = re.sub(r"<[^>]+>", " ", inner)
        names.append(re.sub(r"\s+", " ", text).strip())
    return names


def _current_anchors(html: str) -> list[str]:
    block = re.search(r'<ol class="lab-steps">(.*?)</ol>', html, re.DOTALL)
    assert block is not None
    found: list[str] = []
    for attrs, href in re.findall(r'<li([^>]*)>\s*<a href="(#[^"]+)"', block.group(1)):
        if 'aria-current="step"' in attrs:
            found.append(href)
    return found


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
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")


def _video(tmp_path: Path, video_id: str = "clip") -> None:
    insert_video(tmp_path, "pilot", video_id, name=f"{video_id}.mp4")


def _episodes(tmp_path: Path, specs: tuple[EpisodeSpec, ...]) -> None:
    for spec in specs:
        add_episode(tmp_path, "pilot", spec)


def _freeze(tmp_path: Path, video_id: str) -> None:
    path = write_blocks(
        tmp_path,
        "pilot",
        video_id,
        [RoleBlock(start_s=0, end_s=30, role="train")],
    )
    freeze_split(tmp_path, "pilot", path)


def _freeze_roles(tmp_path: Path, video_id: str) -> None:
    path = write_blocks(tmp_path, "pilot", video_id, list(_BLOCKS))
    freeze_split(tmp_path, "pilot", path)


def _store_reviewed(
    tmp_path: Path,
    video_id: str,
    split: str,
    index: int,
    source_time_s: float,
    *,
    status: str = "reviewed",
) -> None:
    """One stored frame. Reviewed with no boxes is trainable. Excluded is not."""
    from excavation_lab.datasets.label_store import FrameWrite, upsert_frames
    from excavation_lab.datasets.labels import LabelRecord
    from excavation_lab.schemas import ReviewStatus, SplitName

    frozen = require_frozen(tmp_path, "pilot", video_id)
    review = ReviewStatus(status)
    record = LabelRecord(
        video_id=video_id,
        source_frame_index=index,
        review_status=review,
    )
    upsert_frames(
        tmp_path,
        "pilot",
        video_id,
        SplitName(split),
        [
            FrameWrite(
                image_file=f"{video_id}_{index}.jpg",
                source_frame_index=index,
                source_pts=index,
                time_base="1/1000",
                source_time_s=source_time_s,
                episode_id="",
                split=SplitName(split),
                split_group_id="",
                width=8,
                height=8,
                payload=b"\xff\xd8\xff\xd9",
                record=record,
                coco_subset="",
                image_attributes={},
                sidecar_boxes=[],
            )
        ],
        frozen=frozen,
        batch_status="labelled",
    )


def test_page_calls_services_and_does_not_draw_boxes() -> None:
    assert "freeze_split" in _PAGE_SOURCE
    assert "require_frozen" in _PAGE_SOURCE
    assert "propose_for_project" in _PAGE_SOURCE
    assert "sample_frames" in _PAGE_SOURCE
    assert "export_cvat" in _PAGE_SOURCE
    assert "import_coco" in _PAGE_SOURCE
    assert "add_class" in _PAGE_SOURCE
    assert "class_box_lines" not in _PAGE_SOURCE
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

    _freeze_roles(tmp_path, "clip")
    held = data_page.snapshot_gate(tmp_path, "pilot")
    assert held.enabled is False
    assert held.refusal == "dataset_not_ready"
    assert held.reason.startswith("Нет проверенных кадров в блоке обучения")
    assert "обучение 0, проверка 0" in held.reason

    _store_reviewed(tmp_path, "clip", "train", 1, 10.0)
    train_only = data_page.snapshot_gate(tmp_path, "pilot")
    assert train_only.enabled is False
    assert train_only.review_ready is False
    assert train_only.reason.startswith("Нет проверенных кадров в блоке проверки")
    assert "обучение 1, проверка 0" in train_only.reason
    english = data_page.snapshot_gate(tmp_path, "pilot", ui_copy.catalog("en"))
    assert english.enabled is False
    assert english.reason.startswith("No reviewed frames in the validation block")
    assert "training 1, validation 0" in english.reason

    _store_reviewed(tmp_path, "clip", "validation", 2, 80.0)
    opened = data_page.snapshot_gate(tmp_path, "pilot")
    assert opened.enabled is True
    assert opened.review_ready is True
    assert opened.reason == ui_copy.SNAPSHOT_READY

    _video(tmp_path, "other")
    blocked = data_page.snapshot_gate(tmp_path, "pilot")
    assert blocked.enabled is False
    assert blocked.reason == ui_copy.SNAPSHOT_LOCKED
    assert blocked.review_ready is True
    _freeze(tmp_path, "other")
    assert data_page.snapshot_gate(tmp_path, "pilot").enabled is True


def test_other_unfrozen_video_is_named_and_a_missing_snapshot_is_quiet(
    tmp_path: Path,
) -> None:
    _pilot(tmp_path)
    _video(tmp_path, "clip")
    _video(tmp_path, "other")
    _episodes(tmp_path, (EpisodeSpec("e1", "clip", "g1", 0, 30, "заезд"),))
    _freeze(tmp_path, "clip")
    with _client(tmp_path) as client:
        page = client.get("/data", params={"video": "clip"})
    assert page.status_code == 200
    text = page.text
    assert "other.mp4 (other)" in text
    assert "Снимок собирается по всему проекту." in text
    attrs = _button_attrs(text, ui_copy.SNAPSHOT_DATASET)
    assert "disabled" in attrs
    assert ui_copy.SNAPSHOT_LOCKED in attrs
    assert "dataset snapshot" not in text
    sample = re.search(r'<details id="step-sample"([^>]*)>', text)
    assert sample is not None and "open" in sample.group(1)
    assert 'id="step-episodes"' not in text


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
    one = SplitProposal(
        video_id="clip",
        blocks=[RoleBlock(start_s=0, end_s=30, role="train")],
    )
    assert one.independence.value == "same_scene"
    assert data_page.limitation_text(one) == SAME_SCENE_MESSAGE
    assert "переносим" not in SAME_SCENE_MESSAGE
    assert "transfer" not in SAME_SCENE_MESSAGE.casefold()


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
        page = client.get("/data", follow_redirects=False)
    assert page.status_code == 303
    assert urlparse(page.headers["location"]).path == "/"
    assert ui_copy.DATA_EMPTY not in page.text
    assert 'class="lab-workflow"' not in page.text
    assert not (tmp_path / "ui" / "project.txt").exists()


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
        assert 'aria-current="step"' not in text
        assert "<img" not in text.lower()
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
    book = ui_copy.catalog("ru")
    assert book.label_counts_this_video in text
    assert book.label_counts_project in text
    assert book.label_counts_snapshot in text
    assert book.snapshot_gate_scope in text
    assert "обучение: не рассчитано." in text
    assert "проверка: не рассчитано." in text
    assert "<img" not in text.lower()
    assert '<video controls preload="metadata" src="/media/pilot/clip">' in text
    assert 'id="block-timeline"' in text
    assert 'id="playback-failed"' in text
    assert 'action="/data/episode"' not in text
    assert 'action="/data/episodes"' not in text
    assert 'name="start_s"' not in text
    assert 'name="end_s"' not in text
    assert 'action="/data/propose"' not in text
    assert book.equipment_block_help in text
    assert book.action_block_help not in text
    assert ui_copy.catalog("en").action_block_help not in text
    assert book.propose_split not in text
    assert book.cut_here not in text
    assert book.concurrent_pass not in text
    assert 'action="/data/freeze"' in text
    assert 'action="/data/sample"' in text
    assert 'action="/data/export"' in text
    assert 'action="/data/import"' in text
    assert 'action="/data/snapshot"' in text
    assert 'action="/data/class"' in text
    assert "disabled" not in _button_attrs(text, book.add_class)
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


def _set_duration(tmp_path: Path, video_id: str, duration_s: float, fps: float | None) -> None:
    with connect(project_sqlite(tmp_path, "pilot")) as connection:
        connection.execute(
            """
            UPDATE videos
            SET duration_s = ?, fps_average = ?
            WHERE video_id = ?
            """,
            (duration_s, fps, video_id),
        )


def _set_clock(
    tmp_path: Path,
    video_id: str,
    *,
    offset_s: float,
    duration_s: float,
    fps: float | None = None,
    time_base: str = "1/1",
) -> None:
    with connect(project_sqlite(tmp_path, "pilot")) as connection:
        connection.execute(
            """
            UPDATE videos
            SET start_offset_s = ?, duration_s = ?, fps_average = ?, time_base = ?
            WHERE video_id = ?
            """,
            (offset_s, duration_s, fps, time_base, video_id),
        )


def _proposal_path(tmp_path: Path, video_id: str) -> Path:
    return project_directory(tmp_path, "pilot") / "split_proposals" / f"{video_id}.json"


def _stored_blocks(tmp_path: Path, video_id: str) -> list[tuple[float, float, str]]:
    path = _proposal_path(tmp_path, video_id)
    if not path.is_file():
        return []
    document = SplitProposal.model_validate_json(path.read_text(encoding="utf-8"))
    return [(block.start_s, block.end_s, block.role) for block in document.blocks]


def test_haul_episode_file_stays_on_disk_and_is_not_called_damaged(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    path = _proposal_path(tmp_path, "clip")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        '{"schema_version":1,"video_id":"clip","frozen":false,'
        '"independence":"episode_split","episodes":[{"episode_id":"1",'
        '"video_id":"clip","split_group_id":"1","start_s":0.0,"end_s":10.0,'
        '"label":"Заезд первого АС","split":"train","concurrent":false}]}\n'
    )
    path.write_text(raw, encoding="utf-8")
    book = ui_copy.catalog("ru")
    with _client(tmp_path) as client:
        text = client.get("/data").text
    assert book.proposal_legacy in text
    assert "Заезд первого АС: 0–10 с." in text
    assert book.proposal_broken not in text
    assert book.proposal_missing not in text
    assert "Защитные интервалы" not in text
    assert 'id="project-create"' not in text.split("<main", 1)[1].split("</main>", 1)[0]
    assert 'id="project-manager"' not in text
    assert 'id="lab-home"' in text
    assert path.read_text(encoding="utf-8") == raw


def test_old_episode_file_with_blocks_is_shown_and_not_rewritten(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    path = _proposal_path(tmp_path, "clip")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = """{
  "schema_version": 1,
  "video_id": "clip",
  "frozen": false,
  "independence": "episode_split",
  "message": null,
  "guard_gap_s": 10.0,
  "episodes": [
    {"episode_id": "1", "video_id": "clip", "split_group_id": "1", "start_s": 0.0, "end_s": 10.0, "label": "Заезд первого АС", "split": "train", "concurrent": false}
  ],
  "blocks": [{"start_s": 0.0, "end_s": 10.0, "role": "train"}]
}
"""
    path.write_text(raw, encoding="utf-8")
    book = ui_copy.catalog("ru")
    with _client(tmp_path) as client:
        text = client.get("/data").text
    assert book.proposal_broken not in text
    assert "Заезд первого АС: 0–10 с." in text
    assert "обучение: 0–10 с." in text
    assert "Независимость:" not in text
    assert path.read_text(encoding="utf-8") == raw


def test_old_frozen_file_with_blocks_opens_and_is_not_rewritten(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    body = {
        "blocks": [{"end_s": 1.0, "role": "train", "start_s": 0.0}],
        "episodes": [
            {
                "concurrent": False,
                "end_s": 1.0,
                "episode_id": "ep1",
                "label": "заезд",
                "split": "train",
                "split_group_id": "g1",
                "start_s": 0.0,
                "video_id": "clip",
            }
        ],
        "frozen": True,
        "guard_gap_s": 10.0,
        "independence": "same_scene",
        "limitation": "too_few_episodes",
        "message": "Мало эпизодов для независимого разбиения.",
        "schema_version": 1,
        "video_id": "clip",
    }
    digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    document = dict(body)
    document["content_hash"] = digest
    path = frozen_manifest_path(tmp_path, "pilot", "clip")
    path.parent.mkdir(parents=True)
    raw = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    path.write_text(raw, encoding="utf-8")
    loaded = require_frozen(tmp_path, "pilot", "clip")
    assert [(block.start_s, block.end_s, block.role) for block in loaded.blocks] == [
        (0.0, 1.0, "train")
    ]
    assert path.read_text(encoding="utf-8") == raw
    with _client(tmp_path) as client:
        text = client.get("/data").text
    assert "не соответствует схеме" not in text
    assert "обучение: 0–1 с." in text


def test_data_page_shows_player_and_timeline(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    with _client(tmp_path) as client:
        text = client.get("/data").text
    assert '<video controls preload="metadata" src="/media/pilot/clip">' in text
    assert 'id="block-timeline"' in text
    assert 'id="playback-failed"' in text
    assert 'action="/data/episode"' not in text
    assert 'name="start_s"' not in text
    assert 'name="end_s"' not in text
    assert 'id="episode-start"' not in text
    assert 'id="episode-end"' not in text


def test_missing_duration_timeline_has_no_fake_ruler(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    with _client(tmp_path) as client:
        text = client.get("/data").text
    match = re.search(
        r'<div class="lab-timeline" id="block-timeline"(.*?)</div>',
        text,
        re.DOTALL,
    )
    assert match is not None
    assert re.search(r"\d", match.group(0)) is None
    assert "data-duration" not in match.group(0)
    assert "Длительность: не рассчитано." in text
    assert "Шаг кадра: не рассчитано." in text


def test_timeline_duration_comes_from_the_video_row(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _set_clock(tmp_path, "clip", offset_s=0, duration_s=95.5, fps=30.0)
    with _client(tmp_path) as client:
        text = client.get("/data").text
    assert 'id="block-timeline" data-duration="95.5"' in text
    assert data_page.frame_step_text(30.0) in text
    window = text[text.find('id="block-timeline"') : text.find('id="block-timeline"') + 80]
    assert "25" not in window
    book = ui_copy.catalog("ru")
    gap = re.search(
        r'class="lab-timeline-block is-unassigned"[^>]*>([^<]*)<',
        text,
    )
    assert gap is not None
    assert gap.group(1) == book.block_unassigned
    assert gap.group(1) != book.split_name_train


def test_post_blocks_replaces_the_set_and_episodes_route_refuses(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _video(tmp_path, "other")
    _set_clock(tmp_path, "clip", offset_s=0, duration_s=40)
    _set_clock(tmp_path, "other", offset_s=0, duration_s=10)
    write_blocks(
        tmp_path,
        "pilot",
        "other",
        [RoleBlock(start_s=1, end_s=2, role="train")],
    )
    add_episode(tmp_path, "pilot", EpisodeSpec("other-ep", "other", "g", 1, 2, "заезд"))
    first = [
        {"start_s": 0, "end_s": 5, "role": "train"},
        {"start_s": 5, "end_s": 10, "role": "train"},
        {"start_s": 10, "end_s": 20, "role": "validation"},
        {"start_s": 20, "end_s": 30, "role": "test"},
    ]
    moved = [
        {"start_s": 0, "end_s": 4, "role": "train"},
        {"start_s": 4, "end_s": 10, "role": "train"},
        {"start_s": 10, "end_s": 20, "role": "validation"},
        {"start_s": 20, "end_s": 30, "role": "test"},
    ]
    with _client(tmp_path) as client:
        created = client.post("/data/blocks", json={"video_id": "clip", "blocks": first})
        replaced = client.post("/data/blocks", json={"video_id": "clip", "blocks": moved})
        refused = client.post(
            "/data/episodes",
            json={"video_id": "clip", "episodes": [{"start_s": 0, "end_s": 1}]},
        )
    assert created.status_code == 200
    assert [
        (row["start_s"], row["end_s"], row["role"]) for row in created.json()["blocks"]
    ] == [(0.0, 5.0, "train"), (5.0, 10.0, "train"), (10.0, 20.0, "validation"), (20.0, 30.0, "test")]
    assert replaced.status_code == 200
    assert _stored_blocks(tmp_path, "clip")[0] == (0.0, 4.0, "train")
    assert _stored_blocks(tmp_path, "other") == [(1.0, 2.0, "train")]
    assert refused.status_code == 400
    assert refused.json()["ok"] is False
    assert "Автоматическое предложение не создаётся" in refused.json()["error"]
    assert list_episodes(tmp_path, "pilot", "clip") == []
    assert list_episodes(tmp_path, "pilot", "other")[0]["episode_id"] == "other-ep"


def test_post_blocks_rejects_overlap_in_the_active_locale(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _set_clock(tmp_path, "clip", offset_s=0, duration_s=30)
    write_blocks(
        tmp_path,
        "pilot",
        "clip",
        [RoleBlock(start_s=0, end_s=10, role="train")],
    )
    before = _proposal_path(tmp_path, "clip").read_bytes()
    overlap = [
        {"start_s": 0, "end_s": 10, "role": "train"},
        {"start_s": 5, "end_s": 15, "role": "validation"},
    ]
    with _client(tmp_path) as client:
        russian = client.post("/data/blocks", json={"video_id": "clip", "blocks": overlap})
        write_locale_file(tmp_path, "en")
        english = client.post("/data/blocks", json={"video_id": "clip", "blocks": overlap})
    assert russian.status_code == 400
    assert russian.json() == {"ok": False, "error": ui_copy.catalog("ru").block_overlap}
    assert english.status_code == 400
    assert english.json() == {"ok": False, "error": ui_copy.catalog("en").block_overlap}
    assert ui_copy.catalog("ru").block_overlap not in english.json()["error"]
    assert _proposal_path(tmp_path, "clip").read_bytes() == before


def test_frozen_split_post_does_not_change_blocks(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _set_clock(tmp_path, "clip", offset_s=0, duration_s=40)
    _freeze(tmp_path, "clip")
    manifest = frozen_manifest_path(tmp_path, "pilot", "clip")
    before = manifest.read_bytes()
    with _client(tmp_path) as client:
        response = client.post(
            "/data/blocks",
            json={
                "video_id": "clip",
                "blocks": [{"start_s": 0, "end_s": 4, "role": "train"}],
            },
        )
        page = client.get("/data")
    assert response.status_code == 400
    assert response.json()["ok"] is False
    assert "Разбиение этого видео уже зафиксировано." in response.json()["error"]
    assert manifest.read_bytes() == before
    assert '"frozen": true' in page.text
    assert 'id="block-timeline"' in page.text
    assert "is-readonly" in page.text
    assert ui_copy.catalog("ru").freeze_once in page.text


def test_second_video_keeps_its_own_blocks(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path, "clip")
    _video(tmp_path, "second")
    _set_clock(tmp_path, "clip", offset_s=0, duration_s=10)
    _set_clock(tmp_path, "second", offset_s=0, duration_s=10)
    with _client(tmp_path) as client:
        second = client.post(
            "/data/blocks",
            json={
                "video_id": "second",
                "blocks": [
                    {"start_s": 0, "end_s": 4, "role": "train"},
                    {"start_s": 4, "end_s": 10, "role": "validation"},
                ],
            },
        )
        clip = client.post(
            "/data/blocks",
            json={
                "video_id": "clip",
                "blocks": [{"start_s": 0, "end_s": 2, "role": "test"}],
            },
        )
    assert second.status_code == 200
    assert clip.status_code == 200
    assert _stored_blocks(tmp_path, "second") == [
        (0.0, 4.0, "train"),
        (4.0, 10.0, "validation"),
    ]
    assert _stored_blocks(tmp_path, "clip") == [(0.0, 2.0, "test")]


def test_frozen_form_post_does_not_add_an_episode(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    add_episode(tmp_path, "pilot", EpisodeSpec("keep", "clip", "g", 0, 10, "заезд"))
    _freeze(tmp_path, "clip")
    before = list_episodes(tmp_path, "pilot", "clip")
    with _client(tmp_path) as client:
        text = _post(
            client,
            "/data/episode",
            data={
                "video_id": "clip",
                "episode_id": "ep-late",
                "split_group_id": "g-late",
                "start_s": "0",
                "end_s": "1",
                "label": "заезд",
            },
        )
    assert list_episodes(tmp_path, "pilot", "clip") == before
    assert "Разбиение этого видео уже зафиксировано." in text
    assert "ep-late" not in text


def test_source_window_uses_start_offset(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _set_clock(tmp_path, "clip", offset_s=300, duration_s=10)
    with _client(tmp_path) as client:
        page = client.get("/data")
        accepted = client.post(
            "/data/blocks",
            json={
                "video_id": "clip",
                "blocks": [
                    {"start_s": 0, "end_s": 5, "role": "train"},
                    {"start_s": 5, "end_s": 10, "role": "validation"},
                ],
            },
        )
    assert '"offset_s": 300' in page.text
    assert accepted.status_code == 200
    assert _stored_blocks(tmp_path, "clip") == [
        (300.0, 305.0, "train"),
        (305.0, 310.0, "validation"),
    ]


def test_missing_offset_does_not_invent_a_zero_ruler(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _set_duration(tmp_path, "clip", 10, 25)
    with _client(tmp_path) as client:
        page = client.get("/data")
        rejected = client.post(
            "/data/blocks",
            json={"video_id": "clip", "blocks": [{"start_s": 0, "end_s": 5, "role": "train"}]},
        )
    text = page.text
    match = re.search(
        r'<div class="lab-timeline" id="block-timeline"(.*?)</div>',
        text,
        re.DOTALL,
    )
    assert match is not None
    assert "data-duration" not in match.group(0)
    assert "data-offset" not in match.group(0)
    assert re.search(r"\d", match.group(0)) is None
    assert "Смещение начала: не рассчитано." in text
    assert rejected.status_code == 400
    assert _stored_blocks(tmp_path, "clip") == []
    note = re.search(r'id="block-offset-note">(.*?)</p>', text)
    assert note is not None
    assert "0" not in note.group(1)


def test_known_zero_offset_stores_a_block_at_zero(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _set_clock(tmp_path, "clip", offset_s=0, duration_s=10)
    with _client(tmp_path) as client:
        page = client.get("/data")
        created = client.post(
            "/data/blocks",
            json={
                "video_id": "clip",
                "blocks": [{"start_s": 0, "end_s": 4, "role": "train"}],
            },
        )
    assert '"offset_s": 0' in page.text
    assert '"offset_s": null' not in page.text
    assert created.status_code == 200
    assert _stored_blocks(tmp_path, "clip") == [(0.0, 4.0, "train")]


def test_unassigned_gaps_are_not_labeled_train(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _set_clock(tmp_path, "clip", offset_s=0, duration_s=10)
    write_blocks(
        tmp_path,
        "pilot",
        "clip",
        [
            RoleBlock(start_s=0, end_s=2, role="train"),
            RoleBlock(start_s=6, end_s=8, role="validation"),
        ],
    )
    book = ui_copy.catalog("ru")
    with _client(tmp_path) as client:
        text = client.get("/data").text
    roles = re.findall(r'class="lab-timeline-block"[^>]*>([^<]+)<', text)
    gaps = re.findall(r'class="lab-timeline-block is-unassigned"[^>]*>([^<]+)<', text)
    assert roles == [book.split_name_train, book.split_name_validation]
    assert gaps == [book.block_unassigned, book.block_unassigned]
    assert book.split_name_train not in gaps


def test_guard_gap_rule_is_stated_and_block_lines_list_the_gaps(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _set_clock(tmp_path, "clip", offset_s=0, duration_s=130)
    write_blocks(tmp_path, "pilot", "clip", list(_BLOCKS))
    proposal = SplitProposal(video_id="clip", blocks=list(_BLOCKS))
    for locale in ("ru", "en"):
        book = ui_copy.catalog(locale)
        write_locale_file(tmp_path, locale)
        with _client(tmp_path) as client:
            text = client.get("/data").text
        assert "10" in book.block_guard_help
        assert 'id="block-guard-help"' in text
        assert book.block_guard_help in text
        lines = data_page.block_lines(proposal, book)
        suffix = book.seconds_suffix
        assert lines == [
            f"{book.split_name_train}: 0–60 {suffix}.",
            f"{book.block_unassigned}: 60–72 {suffix} (12 {suffix}).",
            f"{book.split_name_validation}: 72–92 {suffix}.",
            f"{book.block_unassigned}: 92–104 {suffix} (12 {suffix}).",
            f"{book.split_name_test}: 104–124 {suffix}.",
        ]
        for line in lines:
            assert line in text
    touching = SplitProposal(
        video_id="clip",
        blocks=[
            RoleBlock(start_s=0, end_s=10, role="train"),
            RoleBlock(start_s=10, end_s=20, role="train"),
        ],
    )
    assert all(ui_copy.RU.block_unassigned not in line for line in data_page.block_lines(touching))


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
    assert 'id="step-episodes"' not in text
    assert 'action="/data/episode"' not in text


def test_propose_refuses_and_writes_no_split(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _episodes(tmp_path, (EpisodeSpec("e1", "clip", "g1", 0, 30, "заезд"),))
    with _client(tmp_path) as client:
        text = _post(client, "/data/propose", data={"video_id": "clip"})
    assert "split_propose_refused" in text
    assert "Автоматическое предложение не создаётся" in text
    assert SAME_SCENE_MESSAGE not in text
    project = project_directory(tmp_path, "pilot")
    assert not (project / "split_proposals").exists()
    assert not (project / "split_manifests").exists()


def test_freeze_shows_hash_and_a_second_click_does_not_rewrite(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    write_blocks(tmp_path, "pilot", "clip", list(_BLOCKS))
    proposal = SplitProposal(video_id="clip", blocks=list(_BLOCKS))
    with _client(tmp_path) as client:
        opened = client.get("/data")
        assert "disabled" in _button_attrs(opened.text, ui_copy.SNAPSHOT_DATASET)
        assert SAME_SCENE_MESSAGE in opened.text
        assert "переносим" not in opened.text
        assert "transfer" not in opened.text.lower()
        assert ui_copy.GUARDS_HEADING not in opened.text
        assert ui_copy.GUARDS_EMPTY not in opened.text
        for line in data_page.block_lines(proposal):
            assert line in opened.text

        frozen_page = _post(client, "/data/freeze", data={"video_id": "clip"})
        frozen = require_frozen(tmp_path, "pilot", "clip")
        assert frozen.content_hash in frozen_page
        assert ui_copy.SPLIT_HASH in frozen_page
        assert "split_frozen" not in frozen_page
        snapshot = _button_attrs(frozen_page, ui_copy.SNAPSHOT_DATASET)
        assert "disabled" in snapshot
        assert "Нет проверенных кадров в блоке обучения" in snapshot
        labels = _button_labels(frozen_page)
        assert ui_copy.FREEZE_SPLIT in labels
        assert all("перезапис" not in label.lower() for label in labels)
        manifest = frozen_manifest_path(tmp_path, "pilot", "clip")
        before = manifest.read_bytes()

        again = _post(client, "/data/freeze", data={"video_id": "clip"})
    assert manifest.read_bytes() == before
    assert "split_frozen" in again
    assert require_frozen(tmp_path, "pilot", "clip").content_hash == frozen.content_hash


def test_freeze_warns_about_a_short_guard_gap_but_still_freezes(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    close = [
        RoleBlock(start_s=0, end_s=60, role="train"),
        RoleBlock(start_s=64, end_s=84, role="validation"),
        RoleBlock(start_s=84, end_s=90, role="validation"),
        RoleBlock(start_s=100, end_s=120, role="test"),
    ]
    write_blocks(tmp_path, "pilot", "clip", close)
    write_locale_file(tmp_path, "en")
    with _client(tmp_path) as client:
        frozen_page = _post(client, "/data/freeze", data={"video_id": "clip"})
    frozen = require_frozen(tmp_path, "pilot", "clip")
    book = ui_copy.catalog("en")
    assert frozen.content_hash in frozen_page
    assert book.freeze_done in frozen_page
    expected = book.block_guard_gap.format(
        earlier=book.split_name_train,
        earlier_end="60",
        later=book.split_name_validation,
        later_start="64",
        gap="4",
        minimum="10",
    )
    assert expected in frozen_page
    assert "4 s, under 10 s" in expected
    # validation→validation touching blocks are not a role boundary; 90→100 is exactly 10 s.
    assert frozen_page.count("under 10 s") == 1
    ru_lines = data_page.guard_gap_lines(frozen.blocks)
    assert len(ru_lines) == 1
    assert ru_lines[0] == ui_copy.RU.block_guard_gap.format(
        earlier=ui_copy.RU.split_name_train,
        earlier_end="60",
        later=ui_copy.RU.split_name_validation,
        later_start="64",
        gap="4",
        minimum="10",
    )
    assert data_page.guard_gap_lines(list(_BLOCKS)) == []
    assert data_page.freeze_recorded_proposal(tmp_path, "pilot", "clip").code == "split_frozen"


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
        sample_form = re.search(
            r'<form class="lab-form" method="post" action="/data/sample">(.*?)</form>',
            form.text,
            re.DOTALL,
        )
        assert sample_form is not None
        body = sample_form.group(1)
        assert body.count('name="sample_mode"') == 2
        detail = re.search(r"<details\b[^>]*>(.*?)</details>", body, re.DOTALL)
        assert detail is not None
        assert 'value="count"' in detail.group(1)
        assert 'value="plan"' in body
        assert 'value="plan"' not in detail.group(1)
        for split_name in ("train", "validation", "test"):
            assert f'<option value="{split_name}">' in body
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
    _freeze_roles(tmp_path, "clip")
    _store_reviewed(tmp_path, "clip", "train", 1, 10.0)
    _store_reviewed(tmp_path, "clip", "validation", 2, 80.0)
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
    path = write_blocks(
        tmp_path,
        "pilot",
        "clip",
        [RoleBlock(start_s=0, end_s=30, role="train")],
    )
    freeze_split(tmp_path, "pilot", path)
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
    names = [path.name for path in paths]
    assert _contact_names(page.text) == names
    assert "demo_box.jpg" not in page.text
    assert "<canvas" not in page.text.lower()
    assert page.text.lower().count("<img") == len(names)
    for name in names:
        assert f'alt="{name}"' in page.text
        assert f"name={name}" in page.text
    assert ui_copy.catalog("ru").contact_no_boxes in page.text
    assert data_page.contact_jpegs(tmp_path, "pilot", "clip")[0] == paths


def test_label_link_index_uses_the_trailing_frame_number() -> None:
    from excavation_lab.web.data_routes import _source_frame_index

    video = "v_b91dd7f95a9d46a5ba8a442bbd6314f0"
    assert _source_frame_index(f"{video}_80.jpg", video) == 80
    assert _source_frame_index("clip_3.jpeg", "clip") == 3
    assert _source_frame_index("clip_3.jpg", "other") is None
    assert _source_frame_index("clip_extra_3.jpg", "clip") is None


def test_step_three_and_thumbnails_open_the_labeler(tmp_path: Path) -> None:
    book = ui_copy.catalog("ru")
    english = ui_copy.catalog("en")
    _pilot(tmp_path)
    clip = tmp_path / "clip.mp4"
    write_synthetic_mp4(clip, 2.0, 5)
    add_video(tmp_path, "pilot", clip, video_id="clip")
    path = write_blocks(
        tmp_path,
        "pilot",
        "clip",
        [RoleBlock(start_s=0, end_s=30, role="train")],
    )
    freeze_split(tmp_path, "pilot", path)
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
        page = client.get("/data?video=clip")
        write_locale_file(tmp_path, "en")
        english_page = client.get("/data?video=clip")
    assert page.status_code == 200
    names = _contact_names(page.text)
    assert names
    step = re.search(r'id="step-label"(.*?)</details>', page.text, re.DOTALL)
    assert step is not None
    body = step.group(1)
    assert 'href="/label?video=clip"' in body
    assert book.label_open_app in body
    assert 'action="/data/export"' in body
    assert 'action="/data/import"' in body
    assert book.export_cvat in body
    assert book.import_coco in body
    sheet = re.search(r'id="contact-sheet">(.*?)</ul>', page.text, re.DOTALL)
    assert sheet is not None
    for name in names:
        index = int(name[: -len(".jpg")].rsplit("_", 1)[1])
        linked = re.search(
            rf'<a href="/label\?video=clip&amp;frame={index}">\s*'
            rf'<img src="[^"]*" alt="{re.escape(name)}">',
            sheet.group(1),
        )
        assert linked is not None
    assert english_page.status_code == 200
    assert english.label_open_app in english_page.text
    assert book.label_open_app not in english_page.text
    assert 'href="/label?video=clip"' in english_page.text


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


def _step_titles() -> tuple[str, ...]:
    book = ui_copy.catalog("ru")
    return (
        book.split_heading,
        book.sample_heading,
        book.step_label_title,
        book.step_snapshot_title,
    )


def _steps(
    *,
    episode_count: int,
    split_state: str,
    jpeg_count: int,
    gate: data_page.SnapshotGate,
    snapshot_exists: bool = False,
    review_ready: bool = False,
    label_caption: str = "",
) -> tuple[data_page.WorkflowStep, ...]:
    return data_page.workflow_steps(
        episode_count=episode_count,
        split_state=split_state,
        jpeg_count=jpeg_count,
        review_ready=review_ready,
        gate=gate,
        snapshot_exists=snapshot_exists,
        label_caption=label_caption,
        text=ui_copy.catalog("ru"),
    )


def test_workflow_keeps_one_current_step() -> None:
    book = ui_copy.catalog("ru")
    closed = data_page.SnapshotGate(enabled=False, reason=book.snapshot_locked)
    opened = data_page.SnapshotGate(enabled=True, reason=book.snapshot_ready)

    def states(**kwargs: object) -> list[str]:
        steps = _steps(gate=closed, **kwargs)  # type: ignore[arg-type]
        assert sum(step.state == "current" for step in steps) == 1
        return [step.state for step in steps]

    assert states(
        episode_count=0,
        split_state="missing",
        jpeg_count=0,
    ) == [
        "current",
        "waiting",
        "waiting",
        "waiting",
    ]
    proposed = _steps(
        episode_count=1,
        split_state="proposal",
        jpeg_count=0,
        gate=closed,
    )
    assert [step.state for step in proposed] == [
        "current",
        "waiting",
        "waiting",
        "waiting",
    ]
    assert proposed[0].waiting_reason == ""
    assert proposed[1].waiting_reason == book.step_wait_frozen
    sampled = _steps(
        episode_count=1,
        split_state="frozen",
        jpeg_count=0,
        gate=opened,
    )
    assert [step.state for step in sampled] == [
        "done",
        "current",
        "waiting",
        "waiting",
    ]
    assert sampled[3].status == book.step_wait
    assert sampled[3].waiting_reason == book.step_snapshot_open
    assert "проверенный кадр" not in sampled[3].waiting_reason
    labeling = _steps(
        episode_count=1,
        split_state="frozen",
        jpeg_count=3,
        gate=opened,
        label_caption="обучение: проверено 0 из 3, не проверено 3, исключено 0.",
    )
    assert [step.state for step in labeling] == [
        "done",
        "done",
        "current",
        "waiting",
    ]
    assert labeling[3].waiting_reason == book.step_snapshot_open
    assert "проверено 0 из 3" in labeling[2].waiting_reason
    assert book.not_calculated not in labeling[2].waiting_reason
    early_snapshot = _steps(
        episode_count=1,
        split_state="frozen",
        jpeg_count=3,
        gate=opened,
        snapshot_exists=True,
    )
    assert early_snapshot[3].state == "waiting"
    assert early_snapshot[3].status == book.step_wait
    assert early_snapshot[3].status != book.step_done
    excluded_only = _steps(
        episode_count=1,
        split_state="frozen",
        jpeg_count=3,
        gate=opened,
    )
    assert excluded_only[2].state == "current"
    partial = _steps(
        episode_count=1,
        split_state="frozen",
        jpeg_count=3,
        gate=opened,
        review_ready=False,
    )
    assert partial[2].state == "current"
    unfinished = _steps(
        episode_count=1,
        split_state="frozen",
        jpeg_count=8,
        gate=opened,
        review_ready=True,
        label_caption="обучение: проверено 1 из 8, не проверено 7, исключено 0.",
    )
    assert unfinished[2].state == "done"
    assert unfinished[2].status == book.step_done
    assert "не проверено 7" in unfinished[2].waiting_reason
    ready = _steps(
        episode_count=1,
        split_state="frozen",
        jpeg_count=3,
        gate=opened,
        review_ready=True,
        snapshot_exists=False,
    )
    assert [step.state for step in ready] == [
        "done",
        "done",
        "done",
        "current",
    ]
    assert ready[3].status == book.step_now
    assert ready[3].status != book.step_done
    written = _steps(
        episode_count=1,
        split_state="frozen",
        jpeg_count=3,
        gate=opened,
        review_ready=True,
        snapshot_exists=True,
    )
    assert [step.state for step in written] == [
        "done",
        "done",
        "done",
        "current",
    ]
    assert written[3].status == book.step_done
    assert written[3].waiting_reason == book.step_snapshot_written
    assert "dataset_hash" not in written[3].waiting_reason
    blocked = _steps(
        episode_count=1,
        split_state="frozen",
        jpeg_count=3,
        gate=closed,
        review_ready=True,
    )
    assert [step.state for step in blocked] == [
        "done",
        "done",
        "done",
        "current",
    ]
    assert blocked[3].status == book.step_now
    assert blocked[3].waiting_reason == book.snapshot_locked


def test_page_marks_the_episode_step_until_one_exists(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    book = ui_copy.catalog("ru")
    with _client(tmp_path) as client:
        page = client.get("/data")
    assert page.status_code == 200
    text = page.text
    for title in _step_titles():
        assert title in text
    assert _current_anchors(text) == ["#step-split"]
    assert 'id="step-split" class="lab-step" data-state="current"' in text
    assert 'id="step-episodes"' not in text
    assert text.count('data-state="current"') == 1
    assert text.count('data-state="waiting"') == 3
    assert book.step_label_text in text
    assert book.step_sample_text in text
    assert book.cvat_heading in text
    assert book.equipment_block_help == (
        "В блоке обучения должны быть все единицы и части оборудования, "
        "которые модель должна узнавать."
    )
    assert book.equipment_block_help in text
    english = ui_copy.catalog("en")
    assert english.equipment_block_help == (
        "The training block must contain every equipment unit and every part "
        "the model should recognize."
    )
    assert book.action_block_help == "В блоке обучения должны быть все действия."
    assert english.action_block_help == "The training block must contain every action."
    assert book.action_block_help not in text
    assert english.action_block_help not in text
    assert book.propose_split not in text
    assert book.cut_here not in text
    assert book.concurrent_pass not in text
    assert 'action="/data/freeze"' in text
    assert 'action="/data/sample"' in text
    assert '<video controls preload="metadata" src="/media/pilot/clip">' in text
    assert 'id="block-timeline"' in text
    assert 'id="step-label"' in text
    assert 'id="step-snapshot"' in text
    assert 'name="split_group_id"' not in text
    assert book.column_split_group not in text
    assert "<img" not in text.lower()


def test_one_episode_makes_the_split_step_current(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _episodes(tmp_path, (EpisodeSpec("e1", "clip", "g1", 0, 30, "заезд"),))
    with _client(tmp_path) as client:
        page = client.get("/data")
    assert page.status_code == 200
    assert _current_anchors(page.text) == ["#step-split"]
    assert 'id="step-split" class="lab-step" data-state="current"' in page.text
    assert 'id="step-episodes"' not in page.text
    assert ui_copy.catalog("ru").freeze_once in page.text
    assert ui_copy.FREEZE_NOTE in page.text


def test_frozen_split_makes_sampling_current_and_keeps_the_snapshot_gate(
    tmp_path: Path,
) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _episodes(tmp_path, (EpisodeSpec("e1", "clip", "g1", 0, 30, "заезд"),))
    _freeze(tmp_path, "clip")
    with _client(tmp_path) as client:
        page = client.get("/data")
    assert page.status_code == 200
    text = page.text
    assert _current_anchors(text) == ["#step-sample"]
    assert 'id="step-sample" class="lab-step" data-state="current"' in text
    attrs = _button_attrs(text, ui_copy.SNAPSHOT_DATASET)
    assert "disabled" in attrs
    assert "Нет проверенных кадров в блоке обучения" in attrs
    assert ui_copy.SNAPSHOT_READY not in attrs
    assert ui_copy.catalog("ru").step_snapshot_text in text
    assert ui_copy.catalog("ru").snapshot_gate_scope in text


def test_batch_download_stays_inside_the_project(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    project = project_directory(tmp_path, "pilot")
    batch = project / "annotation_batches" / "clip__train"
    batch.mkdir(parents=True)
    payload = b"PK\x03\x04real-batch"
    (batch / "annotation_batch.zip").write_bytes(payload)
    outside = tmp_path / "secret.zip"
    outside.write_bytes(b"secret")
    (project / "annotation_batch.zip").write_bytes(b"not-in-batch-dir")
    (project / "annotation_batches" / "annotation_batch.zip").write_bytes(b"sibling")
    with _client(tmp_path) as client:
        ok = client.get("/data/batch", params={"video": "clip", "split": "train"})
        assert ok.status_code == 200
        assert ok.content == payload
        disposition = ok.headers["content-disposition"].lower()
        assert "attachment" in disposition
        assert 'filename="annotation_batch.zip"' in disposition
        page = client.get("/data")
        assert ui_copy.catalog("ru").download_batch in page.text
        assert "/data/batch?video=clip&amp;split=train" in page.text
        assert "split=validation" not in page.text
        assert "split=test" not in page.text
        assert str(outside) not in page.text
        assert "not-in-batch-dir" not in page.text
        for params in (
            {"video": "clip", "split": "nope"},
            {"video": "clip", "split": "../train"},
            {"video": "../secret", "split": "train"},
            {"video": "..", "split": "train"},
            {"video": "clip/../../outside", "split": "train"},
            {"video": "clip", "split": "validation"},
        ):
            refused = client.get("/data/batch", params=params)
            assert refused.status_code == 404
            assert refused.content != payload
            assert b"secret" not in refused.content
            assert b"sibling" not in refused.content
        (batch / "annotation_batch.zip").unlink()
        (batch / "annotation_batch.zip").symlink_to(outside)
        leaked = client.get("/data/batch", params={"video": "clip", "split": "train"})
        assert leaked.status_code == 404
        assert leaked.content != b"secret"
        gone = client.get("/data")
        assert "/data/batch?video=clip" not in gone.text


def test_frame_route_serves_only_manifest_jpegs(tmp_path: Path) -> None:
    from excavation_lab.datasets.sampler import SampleResult, SampleRow
    from excavation_lab.schemas import SplitCounts

    _pilot(tmp_path)
    _video(tmp_path)
    project = project_directory(tmp_path, "pilot")
    images = project / "samples" / "clip" / "images"
    images.mkdir(parents=True)
    jpeg = images / "clip_0.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff\xd9")
    stray = images / "stray.jpg"
    stray.write_bytes(b"\xff\xd8\xff\xd9stray")
    written = SplitCounts(train=1, validation=0, test=0)
    empty = SplitCounts(train=0, validation=0, test=0)
    result = SampleResult(
        project_id="pilot",
        video_id="clip",
        split_content_hash="cd" * 32,
        spacing_s=1,
        requested=written,
        written=written,
        shortfall=empty,
        rows=[
            SampleRow(
                video_id="clip",
                source_frame_index=0,
                source_time_s=0,
                episode_id="ep",
                split="train",
                image_sha256="ab" * 32,
                image_file="clip_0.jpg",
                strategy="spaced",
            )
        ],
        manifest_path="sample_manifest.json",
        images_dir=str(images),
    )
    manifest = project / "samples" / "clip" / "sample_manifest.json"
    manifest.write_text(result.model_dump_json(), encoding="utf-8")
    with _client(tmp_path) as client:
        ok = client.get("/data/frame", params={"video": "clip", "name": "clip_0.jpg"})
        assert ok.status_code == 200
        assert ok.headers["content-type"].startswith("image/jpeg")
        assert ok.content == jpeg.read_bytes()
        for name in ("stray.jpg", "../clip_0.jpg", "clip/0.jpg", "clip_0.png", "..", "clip_0.jpg/../../x"):
            refused = client.get("/data/frame", params={"video": "clip", "name": name})
            assert refused.status_code == 404, name
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(b"\xff\xd8\xff\xd9outside")
        jpeg.unlink()
        jpeg.symlink_to(outside)
        leaked = client.get("/data/frame", params={"video": "clip", "name": "clip_0.jpg"})
        assert leaked.status_code == 404
        assert b"outside" not in leaked.content


def test_frame_route_rejects_an_images_dir_outside_the_sample_folder(tmp_path: Path) -> None:
    from excavation_lab.datasets.sampler import SampleResult, SampleRow
    from excavation_lab.schemas import SplitCounts

    _pilot(tmp_path)
    _video(tmp_path)
    project = project_directory(tmp_path, "pilot")
    outside = project / "samples" / "clip"
    outside.mkdir(parents=True)
    jpeg = outside / "clip_0.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff\xd9outside-folder")
    written = SplitCounts(train=1, validation=0, test=0)
    empty = SplitCounts(train=0, validation=0, test=0)
    result = SampleResult(
        project_id="pilot",
        video_id="clip",
        split_content_hash="cd" * 32,
        spacing_s=1,
        requested=written,
        written=written,
        shortfall=empty,
        rows=[
            SampleRow(
                video_id="clip",
                source_frame_index=0,
                source_time_s=0,
                episode_id="ep",
                split="train",
                image_sha256="ab" * 32,
                image_file="clip_0.jpg",
                strategy="spaced",
            )
        ],
        manifest_path="sample_manifest.json",
        images_dir=str(outside),
    )
    manifest = project / "samples" / "clip" / "sample_manifest.json"
    manifest.write_text(result.model_dump_json(), encoding="utf-8")
    with _client(tmp_path) as client:
        refused = client.get("/data/frame", params={"video": "clip", "name": "clip_0.jpg"})
    assert refused.status_code == 404
    assert refused.content != jpeg.read_bytes()


def test_batch_link_encodes_an_ampersand_in_the_video_id(tmp_path: Path) -> None:
    _pilot(tmp_path)
    video_id = "a&b"
    _video(tmp_path, video_id)
    project = project_directory(tmp_path, "pilot")
    batch = project / "annotation_batches" / f"{video_id}__train"
    batch.mkdir(parents=True)
    payload = b"PK\x03\x04amp"
    (batch / "annotation_batch.zip").write_bytes(payload)
    with _client(tmp_path) as client:
        page = client.get("/data")
        assert page.status_code == 200
        assert "video=a%26b&amp;split=train" in page.text
        assert "video=a&amp;split=b" not in page.text
        assert "split=b" not in page.text
        downloaded = client.get(
            "/data/batch",
            params={"video": video_id, "split": "train"},
        )
    assert downloaded.status_code == 200
    assert downloaded.content == payload


def test_label_step_follows_this_video_and_snapshot_stays_current(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from excavation_lab.datasets.sampler import SampleResult, SampleRow
    from excavation_lab.schemas import ReviewStatus, SplitCounts

    _pilot(tmp_path)
    _video(tmp_path)
    _episodes(tmp_path, (EpisodeSpec("e1", "clip", "g1", 0, 30, "заезд"),))
    _freeze(tmp_path, "clip")
    project = project_directory(tmp_path, "pilot")
    images = project / "samples" / "clip" / "images"
    images.mkdir(parents=True)
    (images / "clip_0.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    written_counts = SplitCounts(train=1, validation=0, test=0)
    empty = SplitCounts(train=0, validation=0, test=0)
    manifest = project / "samples" / "clip" / "sample_manifest.json"
    manifest.write_text(
        SampleResult(
            project_id="pilot",
            video_id="clip",
            split_content_hash="cd" * 32,
            spacing_s=1,
            requested=written_counts,
            written=written_counts,
            shortfall=empty,
            rows=[
                SampleRow(
                    video_id="clip",
                    source_frame_index=0,
                    source_time_s=0,
                    episode_id="e1",
                    split="train",
                    image_sha256="ab" * 32,
                    image_file="clip_0.jpg",
                    strategy="spaced",
                )
            ],
            manifest_path=str(manifest),
            images_dir=str(images),
        ).model_dump_json(),
        encoding="utf-8",
    )
    other = SimpleNamespace(
        video_id="other",
        record=SimpleNamespace(review_status=ReviewStatus.reviewed),
    )
    book = ui_copy.catalog("ru")
    monkeypatch.setattr(data_page, "load_imported_frames", lambda *_a, **_k: [other])
    with _client(tmp_path) as client:
        unlabeled = client.get("/data")
    assert unlabeled.status_code == 200
    assert _current_anchors(unlabeled.text) == ["#step-label"]
    assert unlabeled.text.count('aria-current="step"') == 1
    assert data_page.review_count_text(None) in unlabeled.text
    assert book.label_counts_this_video in unlabeled.text
    assert book.label_counts_project in unlabeled.text
    assert "обучение: не рассчитано." in unlabeled.text
    assert f"{ui_copy.COUNT_REVIEWED}: 0" not in unlabeled.text
    assert f"{ui_copy.COUNT_UNREVIEWED}: 0" not in unlabeled.text
    reviewed, unreviewed = data_page.video_review_counts(tmp_path, "pilot", "clip")
    assert reviewed is None and unreviewed is None
    monkeypatch.undo()
    _store_reviewed(tmp_path, "clip", "train", 0, 0.0)
    with _client(tmp_path) as client:
        labeled = client.get("/data")
    assert _current_anchors(labeled.text) == ["#step-label"]
    assert 'id="step-label" class="lab-step" data-state="done"' not in labeled.text
    assert "Нет проверенных кадров в блоке проверки" in labeled.text
    assert "обучение: Проверенные 1," in labeled.text


def test_label_step_uses_the_catalog_and_add_class_stays_enabled(tmp_path: Path) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    book = ui_copy.catalog("ru")
    english = ui_copy.catalog("en")
    template = (
        Path(data_page.__file__).resolve().parents[1] / "web" / "templates" / "data.html"
    )
    template_text = template.read_text(encoding="utf-8")
    assert "class_box_lines" not in template_text
    assert "class_labels" not in template_text
    stored_box = (
        "Кабина, поворотная платформа, противовес и видимая ходовая часть. "
        "Без стрелы, рукояти и ковша."
    )
    project = project_directory(tmp_path, "pilot")
    sentinel = project / "annotation_batches" / "boxes.json"
    sentinel.write_text('{"class_name":"bucket","box":[1,2,3,4]}\n', encoding="utf-8")
    before = sentinel.read_bytes()
    with _client(tmp_path) as client:
        page = client.get("/data")
        assert page.status_code == 200
        text = page.text
        assert 'id="step-label" class="lab-step" data-state="current"' not in text
        assert "<code>excavator_body</code>" in text
        assert "Экскаватор: корпус" in text
        assert stored_box in text
        assert book.class_box_lines["excavator_body"] not in text
        assert book.class_labels["excavator_body"] not in text
        assert "disabled" not in _button_attrs(text, book.add_class)
        assert data_page.snapshot_written(tmp_path, "pilot") is False
        added = _post(
            client,
            "/data/class",
            data={
                "video_id": "clip",
                "class_name": "shovel",
                "title": "Лопата",
                "box": "Видимая лопата.",
            },
        )
        assert book.class_added in added
        assert "<code>shovel</code>" in added
        assert "Лопата" in added
        assert "Видимая лопата." in added
        assert "disabled" not in _button_attrs(added, book.add_class)
        assert sentinel.read_bytes() == before
        names = {item.name for item in require_project_catalog(tmp_path, "pilot").classes}
        assert "shovel" in names
        assert names >= {"excavator_body", "bucket", "haul_truck", "truck_bed"}
        _freeze(tmp_path, "clip")
        versions = project / "dataset_versions" / "ds1"
        versions.mkdir()
        (versions / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
        with connect(project / "project.sqlite") as connection:
            connection.execute(
                """
                INSERT INTO dataset_versions (
                    dataset_id, project_id, content_hash, path, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("ds1", "pilot", "ab" * 32, str(versions), "2020-01-02T00:00:00+00:00"),
            )
        assert data_page.snapshot_written(tmp_path, "pilot") is True
        after = client.get("/data")
        assert after.status_code == 200
        assert "disabled" not in _button_attrs(after.text, book.add_class)
        assert "<code>shovel</code>" in after.text
        assert sentinel.read_bytes() == before
        write_locale_file(tmp_path, "en")
        english_page = client.get("/data")
    assert english.add_class == "Add class"
    assert english.add_class in english_page.text
    assert stored_box in english_page.text
    assert english.class_box_lines["excavator_body"] not in english_page.text
    assert "disabled" not in _button_attrs(english_page.text, english.add_class)


def test_add_class_is_in_the_next_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    seen: list[str] = []

    def spy(workspace: Path, project_id: str, batch_id: str, **kwargs: object) -> None:
        del batch_id, kwargs
        stored = require_project_catalog(workspace, project_id)
        seen.extend(item.name for item in stored.classes)
        raise LabError("stopped", "остановлено")

    monkeypatch.setattr(data_page, "export_cvat", spy)
    with _client(tmp_path) as client:
        _post(
            client,
            "/data/class",
            data={
                "video_id": "clip",
                "class_name": "shovel",
                "title": "Лопата",
                "box": "Видимая лопата.",
            },
        )
        _post(
            client,
            "/data/export",
            data={"video_id": "clip", "split": "train"},
        )
    assert "shovel" in seen


def test_other_catalog_does_not_list_the_excavator_classes(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_catalog(
        tmp_path,
        "pit_tools",
        (("loader", "Погрузчик", "Видимый погрузчик."),),
    )
    create_project("east", tmp_path, catalog_id="pit_tools")
    write_project_file(tmp_path, "east")
    insert_video(tmp_path, "east", "clip", name="clip.mp4")
    with _client(tmp_path) as client:
        page = client.get("/data")
    assert page.status_code == 200
    text = page.text
    assert "<code>loader</code>" in text
    assert "Погрузчик" in text
    assert "Видимый погрузчик." in text
    for name in ("excavator_body", "bucket", "haul_truck", "truck_bed"):
        assert f"<code>{name}</code>" not in text
    assert "Экскаватор: корпус" not in text
    assert "disabled" not in _button_attrs(text, ui_copy.catalog("ru").add_class)
    assert 'action="/data/sample"' in text
    assert 'action="/data/import"' in text
    assert 'action="/data/export"' in text
    assert 'action="/data/snapshot"' in text


def test_in_app_saves_open_the_gate_and_keep_test_frames_out(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    from excavation_lab.datasets.coco_import import load_imported_frames
    from excavation_lab.datasets.sampler import SampleResult, SampleRow
    from excavation_lab.schemas import SplitCounts

    def jpeg(tone: int) -> bytes:
        image = np.full((24, 32, 3), tone, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        assert ok
        return encoded.tobytes()

    def plant(video_id: str, specs: list[tuple[int, float, bytes, str]]) -> None:
        frozen = require_frozen(tmp_path, "pilot", video_id)
        root = project_directory(tmp_path, "pilot") / "samples" / video_id
        images = root / "images"
        images.mkdir(parents=True)
        rows: list[SampleRow] = []
        for index, time_s, payload, split in specs:
            name = f"{video_id}_{index}.jpg"
            (images / name).write_bytes(payload)
            rows.append(
                SampleRow(
                    video_id=video_id,
                    source_frame_index=index,
                    source_time_s=time_s,
                    episode_id="",
                    split=split,
                    image_sha256=hashlib.sha256(payload).hexdigest(),
                    image_file=name,
                    strategy="spaced",
                    source_pts=index * 1000,
                    time_base="1/1000",
                )
            )
        manifest = root / "sample_manifest.json"
        counts = SplitCounts(
            train=sum(1 for row in rows if row.split == "train"),
            validation=sum(1 for row in rows if row.split == "validation"),
            test=sum(1 for row in rows if row.split == "test"),
        )
        body = SampleResult(
            project_id="pilot",
            video_id=video_id,
            split_content_hash=frozen.content_hash,
            spacing_s=1.0,
            requested=counts,
            written=counts,
            shortfall=SplitCounts(train=0, validation=0, test=0),
            rows=rows,
            manifest_path=str(manifest),
            images_dir=str(images),
        )
        manifest.write_text(body.model_dump_json(), encoding="utf-8")

    _pilot(tmp_path)
    _video(tmp_path, "main")
    _freeze_roles(tmp_path, "main")
    _video(tmp_path, "clip")
    _freeze(tmp_path, "clip")
    plant(
        "main",
        [
            (10, 10.0, jpeg(20), "train"),
            (12, 12.0, jpeg(30), "train"),
            (14, 14.0, jpeg(40), "train"),
            (80, 80.0, jpeg(50), "validation"),
            (110, 110.0, jpeg(60), "test"),
        ],
    )
    plant(
        "clip",
        [(index, float(index), jpeg(70 + index), "train") for index in range(1, 5)],
    )
    book = ui_copy.catalog("ru")
    with _client(tmp_path) as client:
        closed = client.get("/data", params={"video": "main"})
        assert "disabled" in _button_attrs(closed.text, book.snapshot_dataset)
        assert "Нет проверенных кадров в блоке обучения" in closed.text
        assert "обучение: не рассчитано." in closed.text
        saved = client.post(
            "/label/frame",
            params={"video": "main", "frame": "10"},
            json={
                "review_status": "reviewed",
                "boxes": [
                    {"class_name": "bucket", "x1": 1, "y1": 1, "x2": 10, "y2": 8}
                ],
            },
        )
        assert saved.status_code == 200
        assert saved.json()["ok"] is True
        train_only = client.get("/data", params={"video": "main"})
        assert "disabled" in _button_attrs(train_only.text, book.snapshot_dataset)
        assert "Нет проверенных кадров в блоке проверки" in train_only.text
        client.post(
            "/label/frame",
            params={"video": "main", "frame": "12"},
            json={
                "review_status": "unreviewed",
                "boxes": [
                    {"class_name": "bucket", "x1": 1, "y1": 1, "x2": 8, "y2": 6}
                ],
            },
        )
        client.post(
            "/label/frame",
            params={"video": "main", "frame": "14"},
            json={"review_status": "excluded", "boxes": []},
        )
        client.post(
            "/label/frame",
            params={"video": "main", "frame": "80"},
            json={"review_status": "reviewed", "boxes": []},
        )
        client.post(
            "/label/frame",
            params={"video": "main", "frame": "110"},
            json={
                "review_status": "reviewed",
                "boxes": [
                    {"class_name": "haul_truck", "x1": 0, "y1": 0, "x2": 20, "y2": 12}
                ],
            },
        )
        opened = client.get("/data", params={"video": "main"})
        assert "disabled" not in _button_attrs(opened.text, book.snapshot_dataset)
        assert book.snapshot_ready in opened.text
        assert book.snapshot_gate_scope in opened.text
        assert 'id="step-label" class="lab-step" data-state="done"' in opened.text
        assert "не проверено 1" in opened.text
        video_counts = opened.text.split('id="label-counts-video"', 1)[1].split("</p>", 1)[0]
        assert "Проверенные 1" in video_counts
        assert "Непроверенные 1" in video_counts
        assert "Исключённые 1" in video_counts
        project_counts = opened.text.split('id="label-counts-project"', 1)[1].split(
            "</p>", 1
        )[0]
        assert "Проверенные 1" in project_counts
        clip_page = client.get("/data", params={"video": "clip"})
        clip_counts = clip_page.text.split('id="label-counts-video"', 1)[1].split("</p>", 1)[0]
        assert "не рассчитано" in clip_counts
        assert "Проверенные 0" not in clip_counts
        assert "disabled" not in _button_attrs(clip_page.text, book.snapshot_dataset)
        write_locale_file(tmp_path, "en")
        english = ui_copy.catalog("en")
        english_page = client.get("/data", params={"video": "main"})
        assert english.snapshot_gate_scope in english_page.text
        assert english.label_counts_this_video in english_page.text
        assert "disabled" not in _button_attrs(english_page.text, english.snapshot_dataset)
        write_locale_file(tmp_path, "ru")
        note = data_page.run_snapshot(tmp_path, "pilot")
    assert note.code is None
    root = Path(note.text.splitlines()[0])
    by_index = {
        frame.source_frame_index: frame.image_file
        for frame in load_imported_frames(tmp_path, "pilot")
        if frame.video_id == "main"
    }

    def names(folder: str) -> set[str]:
        document = json.loads(
            (root / folder / "_annotations.coco.json").read_text(encoding="utf-8")
        )
        return {image["file_name"] for image in document["images"]}

    train_names = names("train")
    valid_names = names("valid")
    test_names = names("test")
    assert train_names == {by_index[10]}
    assert valid_names == {by_index[80]}
    assert test_names == {by_index[110]}
    assert by_index[12] not in train_names | valid_names | test_names
    assert by_index[14] not in train_names | valid_names | test_names
    assert by_index[110] not in train_names | valid_names
    assert by_index[10] not in valid_names | test_names
    assert by_index[80] not in train_names | test_names
    for index in range(1, 5):
        assert f"clip_{index}.jpg" not in train_names | valid_names | test_names
    with _client(tmp_path) as client:
        reviewed = client.get("/data", params={"video": "main"})
        assert "disabled" not in _button_attrs(reviewed.text, book.snapshot_dataset)
        assert "missing_class" in reviewed.text
        training = client.get("/training")
    assert training.status_code == 200
    dataset_id = root.name
    assert dataset_id in training.text
    assert "нет снимка" not in training.text


def test_review_warning_does_not_disable_the_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pilot(tmp_path)
    _video(tmp_path)
    _freeze_roles(tmp_path, "clip")
    _store_reviewed(tmp_path, "clip", "train", 1, 10.0)
    _store_reviewed(tmp_path, "clip", "validation", 2, 80.0)
    warning = ValidationIssue(
        code="test_not_labeled",
        message="Часть test не размечена. Это предупреждение и не отказ.",
        split="test",
    )

    def fake_validate(_workspace: Path, _project_id: str) -> DatasetValidation:
        return DatasetValidation(
            dataset_id="ds1",
            project_id="pilot",
            path="/tmp/dataset",
            dataset_hash="ab" * 32,
            reviewed=2,
            unreviewed=1,
            excluded=0,
            reviewed_empty=1,
            negatives=1,
            warnings=[warning],
        )

    monkeypatch.setattr(data_page, "validate_dataset", fake_validate)
    book = ui_copy.catalog("ru")
    with _client(tmp_path) as client:
        page = client.get("/data")
    assert warning.message in page.text
    attrs = _button_attrs(page.text, book.snapshot_dataset)
    assert "disabled" not in attrs
    assert warning.message not in attrs
    assert book.snapshot_ready in attrs


def test_training_page_repeats_the_snapshot_gate_reason(tmp_path: Path) -> None:
    from excavation_lab.ui import training_page

    _pilot(tmp_path)
    _video(tmp_path)
    _freeze_roles(tmp_path, "clip")
    book = ui_copy.catalog("ru")
    with _client(tmp_path) as client:
        closed = client.get("/training")
        assert "dataset_not_ready" in closed.text
        assert "Нет проверенных кадров в блоке обучения" in closed.text
        assert book.dataset_not_ready not in closed.text
        _store_reviewed(tmp_path, "clip", "train", 1, 10.0)
        _store_reviewed(tmp_path, "clip", "validation", 2, 80.0)
        ready = client.get("/training")
    assert "dataset_not_ready" in ready.text
    assert book.snapshot_not_built in ready.text
    assert "Нет проверенных кадров" not in ready.text
    assert training_page.missing_snapshot_message(tmp_path, "pilot") == book.snapshot_not_built


def test_import_route_does_not_open_the_archive() -> None:
    routes = Path(data_page.__file__).resolve().parents[1] / "web" / "data_routes.py"
    uploads = routes.with_name("uploads.py")
    route_source = routes.read_text(encoding="utf-8")
    upload_source = uploads.read_text(encoding="utf-8")
    assert "zipfile" not in route_source
    assert "ZipFile" not in route_source
    assert "request.body(" not in route_source
    assert "ZipFile" not in upload_source
