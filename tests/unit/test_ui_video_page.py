"""Video page: local add, probe metadata, baseline lock, no invented boxes."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
from html import unescape
from pathlib import Path
from urllib.parse import urlparse

import pytest
from starlette.testclient import TestClient

from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.errors import LabError, PathNotAllowedError
from excavation_lab.hardware import official_nano_weights_path
from excavation_lab.jobs import reopen, request_cancel, start_job
from excavation_lab.media.probe import add_video
from excavation_lab.schemas import DecodeStatus, JobRecord, JobStatus
from excavation_lab.storage import (
    connect,
    create_project,
    list_videos,
    project_directory,
    project_sqlite,
    replace_extra_roots,
)
from excavation_lab.tracking.analyze import FAKE_DETECTOR_ENV, REQUEST_NAME
from excavation_lab.ui import copy as ui_copy
from excavation_lab.ui import video_page
from excavation_lab.ui.locale import write_locale_file
from excavation_lab.web.app import create_app
from excavation_lab.web.choose_file import ChooseFileFailed
from excavation_lab.web.preferences import write_flash, write_project_file
from excavation_lab.web.uploads import UPLOAD_CAP

_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SYNTHETIC_SPEC = importlib.util.spec_from_file_location(
    "excavation_lab_synthetic_mp4_video_page",
    _SYNTHETIC_PATH,
)
assert _SYNTHETIC_SPEC is not None and _SYNTHETIC_SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SYNTHETIC_SPEC)
_SYNTHETIC_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_mp4 = _SYNTHETIC.write_synthetic_mp4

_ZERO_TOKEN = re.compile(r"(?<![\d.])0(?:[.,]0+)?%?(?![\d.])")
_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_PAGE_SOURCE = Path(video_page.__file__).read_text(encoding="utf-8")
_WEB_ROOT = Path(video_page.__file__).resolve().parents[1] / "web"
_ACTIVE = {
    JobStatus.queued,
    JobStatus.running,
    JobStatus.cancel_requested,
}


def _client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace))


def _location(response: object) -> str:
    raw = response.headers["location"]  # type: ignore[attr-defined]
    parsed = urlparse(raw)
    assert parsed.query == ""
    return parsed.path


def _submit_buttons(html: str) -> list[tuple[str, bool]]:
    found: list[tuple[str, bool]] = []
    for attrs, text in re.findall(r"<button\b([^>]*)>(.*?)</button>", html, re.DOTALL):
        if 'type="submit"' not in attrs or "lab-button" not in attrs:
            continue
        found.append((re.sub(r"\s+", " ", text).strip(), "disabled" in attrs.split()))
    return found


def _button_labels(html: str) -> list[str]:
    return [label for label, _disabled in _submit_buttons(html)]


def _input_value(html: str, name: str) -> str | None:
    tag = re.search(rf'<input\b(?=[^>]*\bname="{re.escape(name)}")[^>]*>', html)
    if tag is None:
        raise AssertionError(name)
    value = re.search(r'\bvalue="([^"]*)"', tag.group(0))
    if value is None:
        return None
    return value.group(1)


def _video_options(html: str) -> list[tuple[str, str, bool]]:
    block = re.search(r'<select\b[^>]*\bname="video"[^>]*>(.*?)</select>', html, re.DOTALL)
    assert block is not None
    found: list[tuple[str, str, bool]] = []
    for attrs, label in re.findall(r"<option\b([^>]*)>([^<]*)</option>", block.group(1)):
        value = re.search(r'value="([^"]*)"', attrs)
        found.append(
            (
                value.group(1) if value else "",
                label.strip(),
                "selected" in attrs.split(),
            )
        )
    return found


def _video_src(html: str) -> str | None:
    match = re.search(r'<video\b[^>]*\bsrc="([^"]+)"', html)
    if match is None:
        return None
    return match.group(1)


def _tagged_input(html: str, element_id: str) -> str:
    match = re.search(
        rf'<input\b(?=[^>]*\bid="{re.escape(element_id)}")[^>]*>',
        html,
    )
    assert match is not None, element_id
    return match.group(0)


def _local_path_value(html: str) -> str:
    tag = _tagged_input(html, "local-path")
    value = re.search(r'\bvalue="([^"]*)"', tag)
    assert value is not None
    return unescape(value.group(1))


def _alert_texts(html: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", text).strip()
        for text in re.findall(r'<p class="lab-alert"[^>]*>(.*?)</p>', html, re.DOTALL)
    ]


def _foreign_cyrillic(html: str) -> str:
    return "".join(_CYRILLIC.findall(html.replace(ui_copy.LANGUAGE_RUSSIAN, "")))


def _raw(
    app: object,
    method: str,
    path: str,
    headers: list[tuple[bytes, bytes]],
    body: bytes = b"",
) -> tuple[int, bytes]:
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
    chunks: list[bytes] = []
    for message in sent:
        if message["type"] == "http.response.start":
            status = int(message["status"])  # type: ignore[arg-type]
        elif message["type"] == "http.response.body":
            raw = message.get("body", b"")
            if isinstance(raw, bytes):
                chunks.append(raw)
    return status, b"".join(chunks)


def _stop_jobs(workspace: Path, project_id: str) -> None:
    for job in reopen(workspace):
        if job.status in _ACTIVE:
            request_cancel(workspace, project_id, job.job_id, kill_after_s=0.4)


def _requests(workspace: Path, project_id: str) -> list[Path]:
    runs = project_directory(workspace, project_id) / "runs"
    if not runs.is_dir():
        return []
    return sorted(runs.glob(f"*/{REQUEST_NAME}"))


def _clip(workspace: Path, name: str = "clip.mp4", seconds: float = 0.4) -> Path:
    path = workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_synthetic_mp4(path, seconds, 10)
    return path


def _extra_roots(workspace: Path) -> list[str]:
    with connect(project_sqlite(workspace, "pilot")) as connection:
        row = connection.execute(
            "SELECT extra_roots_json FROM projects WHERE project_id = 'pilot'"
        ).fetchone()
    assert row is not None
    payload = json.loads(row["extra_roots_json"])
    assert isinstance(payload, list)
    return [item for item in payload if isinstance(item, str)]


def test_page_does_not_shell_out_or_build_a_detector() -> None:
    assert "shell=True" not in _PAGE_SOURCE
    assert "subprocess" not in _PAGE_SOURCE
    assert "youtube" not in _PAGE_SOURCE.lower()
    assert re.search(r"maxUploadSize\s*=", _PAGE_SOURCE) is None
    assert "open_detector" not in _PAGE_SOURCE
    assert "RFDETR" not in _PAGE_SOURCE
    assert "add_video" in _PAGE_SOURCE
    assert "reprobe_video" in _PAGE_SOURCE
    assert "enqueue_analyze" in _PAGE_SOURCE
    assert "require_half_open_interval" in _PAGE_SOURCE
    assert "player_file" in _PAGE_SOURCE
    assert "import streamlit" not in _PAGE_SOURCE
    assert "from streamlit" not in _PAGE_SOURCE
    assert "import starlette" not in _PAGE_SOURCE
    assert "from starlette" not in _PAGE_SOURCE
    assert "App" + "Test" not in Path(__file__).read_text(encoding="utf-8")
    routes = (_WEB_ROOT / "routes.py").read_text(encoding="utf-8")
    assert "copy=False" in routes
    assert "/video/choose" in routes
    assert "/video/upload" not in routes
    assert "post_video_upload" not in routes
    assert "stream_upload" not in routes
    assert "store_browser_upload" not in routes
    for path in _WEB_ROOT.rglob("*.py"):
        assert "request.body(" not in path.read_text(encoding="utf-8")


def test_unknown_metadata_is_not_zero() -> None:
    blank = {
        "duration_s": None,
        "fps_nominal": None,
        "fps_average": 0,
        "width": 0,
        "height": None,
        "codec": "",
        "sha256": None,
    }
    text = "\n".join(video_page.metadata_lines(blank, None))
    text = f"{text}\n{video_page.rotation_line(blank)}"
    assert text.count(ui_copy.NOT_CALCULATED) >= 7
    assert ui_copy.META_FPS_NOMINAL in text
    assert ui_copy.META_FPS_AVERAGE in text
    assert _ZERO_TOKEN.search(text) is None


def test_both_fps_values_are_shown_when_present() -> None:
    text = "\n".join(
        video_page.metadata_lines(
            {
                "duration_s": 1.5,
                "fps_nominal": 30,
                "fps_average": 29.97,
                "width": 64,
                "height": 48,
                "codec": "h264",
                "sha256": "abc",
            },
            "cpu",
        )
    )
    assert "номинальная 30" in text
    assert "средняя 29.97" in text
    assert "64×48" in text
    assert "Устройство: cpu" in text


def test_empty_project_names_the_workspace_root(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    with _client(tmp_path) as client:
        page = client.get("/video")
        refused = client.post("/video/add", data={"local_path": "  "}, follow_redirects=False)
        shown = client.get("/video")
        again = client.get("/video")
    assert page.status_code == 200
    text = page.text
    field = _tagged_input(text, "local-path")
    assert ui_copy.CHOOSE_FILE_BUTTON in text
    assert ui_copy.ADD_LOCAL_VIDEO in text
    assert ui_copy.ALLOWED_DIRECTORY in text
    assert "корень workspace" in text
    assert ui_copy.LOCAL_PATH_LABEL in text
    assert ui_copy.LOCAL_PATH_PLACEHOLDER not in text
    assert ui_copy.UPLOAD_LABEL not in text
    assert ui_copy.UPLOAD_BUTTON not in text
    assert ui_copy.ADD_ANOTHER not in text
    assert ui_copy.ADD_VIDEO_BUTTON not in text
    assert ui_copy.PATH_HELP not in text
    assert ui_copy.UPLOAD_HELP not in text
    assert ui_copy.SOURCE_URL_HELP not in text
    assert ui_copy.SOURCE_URL_LABEL not in text
    assert "<summary>Дополнительно</summary>" not in text
    assert text.count("<summary>") == 0
    assert 'id="project-create-panel"' not in text
    assert 'id="project-create"' not in text.split("<main", 1)[1].split("</main>", 1)[0]
    assert 'id="project-manager"' not in text
    assert 'id="lab-project"' not in text
    assert 'id="lab-home"' in text
    assert 'type="file"' not in text
    assert 'name="file"' not in text
    assert 'name="local_path"' not in text
    assert 'action="/video/upload"' not in text
    assert "multipart/form-data" not in text
    assert 'action="/video/choose"' in text
    assert text.count('id="local-path"') == 1
    assert 'type="text"' in field
    assert "readonly" in field
    assert "name=" not in field
    assert _local_path_value(text) == ""
    assert ui_copy.VIDEO_METADATA_UNKNOWN not in text
    assert "ffprobe" not in text
    assert "youtube" not in text.lower()
    assert "<video" not in text
    labels = _button_labels(text)
    assert ui_copy.CHOOSE_FILE_BUTTON in labels
    assert ui_copy.ADD_VIDEO_BUTTON not in labels
    assert ui_copy.UPLOAD_BUTTON not in labels
    assert ui_copy.BASELINE_BUTTON not in labels
    assert (
        f'<button type="submit" class="lab-button">{ui_copy.CHOOSE_FILE_BUTTON}</button>'
        in text
    )
    assert _location(refused) == "/video"
    assert ui_copy.PATH_REQUIRED in shown.text
    assert ui_copy.PATH_REQUIRED not in again.text
    assert list_videos(tmp_path, "pilot") == []


def test_extra_allowed_root_is_named(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    extra = tmp_path / "clips"
    extra.mkdir()

    def roots(workspace: Path, project_id: str) -> list[Path]:
        del workspace, project_id
        return [tmp_path, extra]

    monkeypatch.setattr(
        "excavation_lab.storage.allowed_media_roots",
        roots,
    )
    text = video_page.allowed_directory_caption(tmp_path, "pilot")
    assert "корень workspace" in text
    assert "clips" in text
    assert "youtube" not in text.lower()


def test_path_outside_the_root_is_rejected_in_russian(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    with _client(tmp_path) as client:
        response = client.post(
            "/video/add",
            data={"local_path": "/tmp/outside-clip.mp4"},
            follow_redirects=True,
        )
    assert response.status_code == 200
    assert "Путь вне разрешённых каталогов" in response.text
    assert list_videos(tmp_path, "pilot") == []
    assert "<video" not in response.text


def test_added_file_shows_probe_metadata_and_plays_from_disk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    clip = _clip(tmp_path)
    calls = {"n": 0}
    real = video_page.reprobe_video

    def wrapped(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(video_page, "reprobe_video", wrapped)
    with _client(tmp_path) as client:
        response = client.post(
            "/video/add",
            data={"local_path": str(clip)},
            follow_redirects=True,
        )
        videos = list_videos(tmp_path, "pilot")
        assert len(videos) == 1
        text = response.text
        device = video_page.playback_device()
        for line in video_page.metadata_lines(videos[0], device):
            assert line in text
        assert video_page.rotation_line(videos[0]) in text
        assert ui_copy.VIDEO_REGISTERED not in text
        assert ui_copy.PROBE_BUTTON not in _button_labels(text)
        assert ui_copy.CHOOSE_FILE_BUTTON in text
        assert ui_copy.UPLOAD_LABEL not in text
        assert _local_path_value(text) == str(videos[0]["path"])
        assert ui_copy.BASELINE_CAPTION in text
        assert ui_copy.PLAYBACK_FILE not in text
        assert ui_copy.PLAYBACK_PREVIEW not in text
        assert ui_copy.PLAYBACK_FAILED in text
        assert 'class="lab-aside"' not in text
        assert 'src="/static/lab.js"' not in text
        assert 'addEventListener("error"' in text
        video_at = text.find("<video")
        meta_at = text.find('class="lab-meta"')
        form_at = text.find('action="/video/baseline"')
        assert video_at != -1 and meta_at != -1 and form_at != -1
        assert video_at < meta_at < form_at
        failed = re.search(r'<p\b[^>]*\bid="playback-failed"[^>]*>', text)
        assert failed is not None and re.search(r"\bhidden\b", failed.group(0))
        assert ui_copy.RUNS_EMPTY in text
        assert "Экскаватор" not in text
        runs = re.search(r'<section class="lab-runs">(.*)</section>', text, re.DOTALL)
        assert runs is not None
        assert "<table" not in runs.group(1)
        src = _video_src(text)
        video_id = videos[0]["video_id"]
        assert src == f"/media/pilot/{video_id}"
        assert calls["n"] == 1
        decoy = tmp_path / "not-registered.bin"
        decoy.write_bytes(b"QUERY-PATH-BYTES")
        media = client.get(src, params={"path": str(decoy)})
        assert media.status_code == 200
        assert media.content == Path(str(videos[0]["path"])).read_bytes()
        assert b"QUERY-PATH-BYTES" not in media.content
        ranged = client.get(src, headers={"Range": "bytes=0-3"})
        assert ranged.status_code == 206
        assert ranged.headers["content-range"].startswith("bytes 0-3/")
        assert ranged.content == Path(str(videos[0]["path"])).read_bytes()[:4]
        assert calls["n"] == 1
        disposition = ranged.headers.get("content-disposition", "")
        assert "attachment" not in disposition.lower()


def test_second_added_file_becomes_the_player(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    first = _clip(tmp_path, "clip.mp4")
    second = _clip(tmp_path, "haul.mp4")
    add_video(tmp_path, "pilot", first)
    with _client(tmp_path) as client:
        response = client.post(
            "/video/add",
            data={"local_path": str(second)},
            follow_redirects=True,
        )
    added = next(item for item in list_videos(tmp_path, "pilot") if item["name"] == "haul.mp4")
    selected = [label for _value, label, on in _video_options(response.text) if on]
    assert selected == [video_page.video_choice_label(added)]
    assert _video_src(response.text) == f"/media/pilot/{added['video_id']}"


def test_second_chosen_file_becomes_the_player(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    first = _clip(tmp_path, "clip.mp4")
    second = _clip(tmp_path, "haul.mp4")
    original = add_video(tmp_path, "pilot", first)

    def choose(*, prompt: str) -> Path:
        assert prompt == ui_copy.CHOOSE_FILE_PROMPT
        return second

    monkeypatch.setattr("excavation_lab.web.routes.choose_video_path", choose)
    with _client(tmp_path) as client:
        response = client.post("/video/choose", follow_redirects=True)
    videos = list_videos(tmp_path, "pilot")
    assert len(videos) == 2
    kept = next(item for item in videos if item["video_id"] == original["video_id"])
    assert kept["path"] == original["path"]
    added = next(item for item in videos if item["name"] == "haul.mp4")
    assert Path(str(added["path"])) == second.resolve()
    selected = [label for _value, label, on in _video_options(response.text) if on]
    assert selected == [video_page.video_choice_label(added)]
    assert _video_src(response.text) == f"/media/pilot/{added['video_id']}"
    field = _tagged_input(response.text, "local-path")
    assert 'type="text"' in field
    assert "readonly" in field
    assert "name=" not in field
    assert _local_path_value(response.text) == str(added["path"])


def test_source_url_is_stored_and_not_fetched(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    clip = _clip(tmp_path)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fetched")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    marker = "https://example.invalid/clip.mp4"
    with _client(tmp_path) as client:
        page = client.get("/video")
        added = client.post(
            "/video/add",
            data={"local_path": str(clip), "source_url": marker},
            follow_redirects=True,
        )
    for html in (page.text, added.text):
        assert "source-url" not in html
        assert "source-url-more" not in html
        assert ui_copy.SOURCE_URL_LABEL not in html
        assert 'name="source_url"' not in html
    videos = list_videos(tmp_path, "pilot")
    assert videos[0]["source_url"] is None


def test_upload_cap_and_traversal_do_not_write(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    assert video_page.upload_rejection(0) == ui_copy.UPLOAD_EMPTY
    assert video_page.upload_rejection(video_page._DIRECT_PLAYBACK_MAX_BYTES + 1) is None
    assert video_page.upload_rejection(8 * 1024 * 1024 * 1024) is None
    with pytest.raises(LabError, match="Недопустимое имя"):
        video_page.store_browser_upload(tmp_path, "pilot", "../clip.mp4", b"x")
    with pytest.raises(LabError):
        video_page.store_browser_upload(tmp_path, "pilot", "clip.mp4", b"not-a-video")
    media = project_directory(tmp_path, "pilot") / "media"
    assert list(media.glob("*.mp4")) == []
    assert list_videos(tmp_path, "pilot") == []


def test_real_camera_name_is_a_media_leaf() -> None:
    catalog = ui_copy.catalog("ru")
    assert video_page._upload_leaf("CAT 336 [Veek].mp4", catalog) == "CAT 336 [Veek].mp4"
    assert video_page._upload_leaf("ковш 12.MP4", catalog) == "ковш 12.mp4"
    with pytest.raises(LabError, match="Недопустимое имя"):
        video_page._upload_leaf("notes.txt", catalog)
    with pytest.raises(LabError, match="Недопустимое имя"):
        video_page._upload_leaf("../clip.mp4", catalog)


def test_small_upload_is_registered_by_add_video(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    source = _clip(tmp_path, "source.mp4", 0.3)
    video_page.store_browser_upload(
        tmp_path,
        "pilot",
        "take.mp4",
        source.read_bytes(),
    )
    videos = list_videos(tmp_path, "pilot")
    assert len(videos) == 1
    assert videos[0]["width"] == 64
    assert videos[0]["height"] == 48
    assert videos[0]["sha256"]
    stored = Path(str(videos[0]["path"]))
    assert stored.is_file()
    assert stored.parent == project_directory(tmp_path, "pilot") / "media"


def test_chooser_registers_the_chosen_file_in_place(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    source = _clip(tmp_path, "source.mp4", 0.3)
    seen: list[dict[str, object]] = []
    real = add_video

    def spy(*args: object, **kwargs: object) -> object:
        seen.append(dict(kwargs))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    def choose(*, prompt: str) -> Path:
        assert prompt == ui_copy.CHOOSE_FILE_PROMPT
        return source

    monkeypatch.setattr("excavation_lab.ui.video_page.add_video", spy)
    monkeypatch.setattr("excavation_lab.web.routes.choose_video_path", choose)
    with _client(tmp_path) as client:
        posted = client.post("/video/choose", follow_redirects=False)
        videos = list_videos(tmp_path, "pilot")
        assert len(videos) == 1
        video_id = str(videos[0]["video_id"])
        assert _location(posted) == "/video"
        assert f"lab_video={video_id}" in posted.headers["set-cookie"]
        page = client.get("/video")
    assert page.status_code == 200
    assert seen == [{"copy": False}]
    stored = Path(str(videos[0]["path"]))
    assert stored == source.resolve()
    assert stored.is_file()
    media = project_directory(tmp_path, "pilot") / "media"
    assert not media.exists() or list(media.iterdir()) == []
    assert videos[0]["source_url"] is None
    assert videos[0]["size_bytes"] == source.stat().st_size
    assert videos[0]["sha256"]
    assert ui_copy.VIDEO_ADDED in page.text
    assert _video_src(page.text) == f"/media/pilot/{video_id}"
    assert _local_path_value(page.text) == str(stored)


def test_choose_cancel_bad_suffix_and_probe_failure_write_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    notes = tmp_path / "notes.txt"
    notes.write_bytes(b"notes")
    broken = tmp_path / "broken.mp4"
    payload = b"not-a-video"
    broken.write_bytes(payload)
    seen: list[dict[str, object]] = []
    real = add_video

    def spy(*args: object, **kwargs: object) -> object:
        seen.append(dict(kwargs))
        return real(*args, **kwargs)  # type: ignore[arg-type]

    state: dict[str, object] = {"kind": "cancel"}

    def choose(*, prompt: str) -> Path | None:
        assert prompt == ui_copy.CHOOSE_FILE_PROMPT
        kind = state["kind"]
        if kind == "cancel":
            return None
        if kind == "fail":
            raise ChooseFileFailed
        path = state["path"]
        assert isinstance(path, Path)
        return path

    monkeypatch.setattr("excavation_lab.ui.video_page.add_video", spy)
    monkeypatch.setattr("excavation_lab.web.routes.choose_video_path", choose)
    with _client(tmp_path) as client:
        before = client.get("/video")
        cancelled = client.post("/video/choose", follow_redirects=True)
        assert _alert_texts(cancelled.text) == _alert_texts(before.text)
        assert ui_copy.CHOOSE_FAILED not in cancelled.text
        assert ui_copy.UPLOAD_BAD_NAME not in cancelled.text
        assert list_videos(tmp_path, "pilot") == []
        assert seen == []

        state["kind"] = "fail"
        failed_dialog = client.post("/video/choose", follow_redirects=True)
        assert ui_copy.CHOOSE_FAILED in failed_dialog.text
        assert list_videos(tmp_path, "pilot") == []
        assert seen == []

        state["kind"] = "path"
        state["path"] = notes
        refused = client.post("/video/choose", follow_redirects=True)
        assert ui_copy.UPLOAD_BAD_NAME in refused.text
        assert notes.read_bytes() == b"notes"
        assert list_videos(tmp_path, "pilot") == []
        assert seen == []

        state["path"] = broken
        probed = client.post("/video/choose", follow_redirects=True)
    assert probed.status_code == 200
    assert seen == [{"copy": False}]
    assert broken.is_file()
    assert broken.read_bytes() == payload
    assert list_videos(tmp_path, "pilot") == []
    assert "Файл не декодируется как видео" in probed.text
    media = project_directory(tmp_path, "pilot") / "media"
    assert not media.exists() or list(media.iterdir()) == []


def test_choose_outside_workspace_registers_the_file_in_place(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    create_project("pilot", workspace, catalog_id="excavation")
    write_project_file(workspace, "pilot")
    kept = tmp_path / "kept"
    kept.mkdir()
    replace_extra_roots(workspace, "pilot", [kept])
    library = tmp_path / "library"
    clip = _clip(library, "haul.mp4", 0.3)

    def choose(*, prompt: str) -> Path:
        assert prompt == ui_copy.CHOOSE_FILE_PROMPT
        return clip

    monkeypatch.setattr("excavation_lab.web.routes.choose_video_path", choose)
    with _client(workspace) as client:
        response = client.post("/video/choose", follow_redirects=True)
    videos = list_videos(workspace, "pilot")
    assert len(videos) == 1
    assert Path(str(videos[0]["path"])) == clip.resolve()
    assert videos[0]["source_url"] is None
    media = project_directory(workspace, "pilot") / "media"
    assert list(media.iterdir()) == []
    assert clip.is_file()
    extras = _extra_roots(workspace)
    assert extras == [str(kept.resolve()), str(library.resolve())]
    assert _video_src(response.text) == f"/media/pilot/{videos[0]['video_id']}"


def test_choose_undecodable_outside_file_restores_extra_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    create_project("pilot", workspace, catalog_id="excavation")
    write_project_file(workspace, "pilot")
    kept = tmp_path / "kept"
    kept.mkdir()
    replace_extra_roots(workspace, "pilot", [kept])
    library = tmp_path / "library"
    library.mkdir()
    broken = library / "broken.mp4"
    broken.write_bytes(b"not-a-video")
    before = _extra_roots(workspace)

    def choose(*, prompt: str) -> Path:
        assert prompt == ui_copy.CHOOSE_FILE_PROMPT
        return broken

    monkeypatch.setattr("excavation_lab.web.routes.choose_video_path", choose)
    with _client(workspace) as client:
        page = client.post("/video/choose", follow_redirects=True)
    assert "Файл не декодируется как видео" in page.text
    assert _extra_roots(workspace) == before
    assert str(library.resolve()) not in before
    assert list_videos(workspace, "pilot") == []
    assert broken.read_bytes() == b"not-a-video"
    media = project_directory(workspace, "pilot") / "media"
    assert not media.exists() or list(media.iterdir()) == []


def test_typed_path_outside_roots_does_not_widen_them(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    create_project("pilot", workspace, catalog_id="excavation")
    write_project_file(workspace, "pilot")
    kept = tmp_path / "kept"
    kept.mkdir()
    replace_extra_roots(workspace, "pilot", [kept])
    library = tmp_path / "library"
    clip = _clip(library, "haul.mp4", 0.3)
    before = _extra_roots(workspace)
    with _client(workspace) as client:
        page = client.post(
            "/video/add",
            data={"local_path": str(clip)},
            follow_redirects=True,
        )
    assert PathNotAllowedError(str(clip.resolve())).message in page.text
    assert _extra_roots(workspace) == before
    assert str(library.resolve()) not in before
    assert list_videos(workspace, "pilot") == []
    media = project_directory(workspace, "pilot") / "media"
    assert list(media.iterdir()) == []


def test_choose_registered_path_selects_the_existing_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    create_project("pilot", workspace, catalog_id="excavation")
    write_project_file(workspace, "pilot")
    clip = _clip(workspace, "clip.mp4", 0.3)
    original = add_video(workspace, "pilot", clip)

    def choose(*, prompt: str) -> Path:
        assert prompt == ui_copy.CHOOSE_FILE_PROMPT
        return clip

    monkeypatch.setattr("excavation_lab.web.routes.choose_video_path", choose)
    with _client(workspace) as client:
        posted = client.post("/video/choose", follow_redirects=False)
        page = client.get("/video")
    videos = list_videos(workspace, "pilot")
    assert [item["video_id"] for item in videos] == [original["video_id"]]
    assert Path(str(videos[0]["path"])) == clip.resolve()
    assert f"lab_video={original['video_id']}" in posted.headers["set-cookie"]
    assert _location(posted) == "/video"
    assert _video_src(page.text) == f"/media/pilot/{original['video_id']}"
    assert _input_value(page.text, "video_id") == original["video_id"]
    assert _extra_roots(workspace) == []


def test_declared_and_counted_caps_write_nothing() -> None:
    from excavation_lab.web.uploads import declared_over_cap

    assert UPLOAD_CAP == 8192 * 1024 * 1024
    assert declared_over_cap(str(UPLOAD_CAP + 1)) is True
    assert declared_over_cap(str(UPLOAD_CAP)) is False
    assert declared_over_cap(None) is False


def test_player_file_keeps_mp4_off_the_preview_branch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("playback_path")

    monkeypatch.setattr(video_page, "playback_path", boom)
    path, kind = video_page.player_file(
        tmp_path,
        "pilot",
        {
            "path": "media/clip.mp4",
            "size_bytes": video_page._DIRECT_PLAYBACK_MAX_BYTES + 1,
        },  # type: ignore[arg-type]
        ui_copy.catalog("ru"),
    )
    assert path == Path("media/clip.mp4")
    assert kind == "playback_file"


def test_player_file_delegates_other_suffixes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    source = tmp_path / "clip.mov"
    source.write_bytes(b"x")
    preview = tmp_path / "previews" / "v.mp4"
    preview.parent.mkdir()
    preview.write_bytes(b"p")

    def fake(
        workspace: Path,
        project_id: str,
        video: object,
        text: object = None,
    ) -> Path:
        del workspace, project_id, text
        size = video["size_bytes"]  # type: ignore[index]
        if size > video_page._DIRECT_PLAYBACK_MAX_BYTES:
            return preview
        return source

    monkeypatch.setattr(video_page, "playback_path", fake)
    small, kind = video_page.player_file(
        tmp_path,
        "pilot",
        {"path": str(source), "size_bytes": 1},  # type: ignore[arg-type]
        ui_copy.catalog("ru"),
    )
    assert kind == "playback_file"
    assert small.resolve() == source.resolve()
    large, kind = video_page.player_file(
        tmp_path,
        "pilot",
        {
            "path": str(source),
            "size_bytes": video_page._DIRECT_PLAYBACK_MAX_BYTES + 1,
        },  # type: ignore[arg-type]
        ui_copy.catalog("ru"),
    )
    assert kind == "playback_preview"
    assert large == preview


def test_large_mp4_is_served_without_a_preview(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("write_preview")

    monkeypatch.setattr("excavation_lab.media.encoder.write_preview", boom)
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    clip = _clip(tmp_path)
    added = add_video(tmp_path, "pilot", clip)
    with connect(project_sqlite(tmp_path, "pilot")) as connection:
        connection.execute(
            "UPDATE videos SET size_bytes = ? WHERE video_id = ?",
            (video_page._DIRECT_PLAYBACK_MAX_BYTES + 1, added["video_id"]),
        )
    with _client(tmp_path) as client:
        page = client.get("/video")
        assert ui_copy.PLAYBACK_FILE not in page.text
        assert ui_copy.PLAYBACK_PREVIEW not in page.text
        assert "Файл изменился" in page.text
        video_at = page.text.find("<video")
        note_at = page.text.find("Файл изменился")
        meta_at = page.text.find('class="lab-meta"')
        form_at = page.text.find('action="/video/baseline"')
        assert video_at < note_at < meta_at < form_at
        assert "не перезаписаны" in page.text
        src = _video_src(page.text)
        assert src == f"/media/pilot/{added['video_id']}"
        ranged = client.get(str(src), headers={"Range": "bytes=0-3"})
    previews = project_directory(tmp_path, "pilot") / "media" / "previews"
    assert not previews.exists()
    assert ranged.status_code == 206
    assert ranged.content == clip.read_bytes()[:4]
    stored = list_videos(tmp_path, "pilot")[0]
    assert stored["size_bytes"] == video_page._DIRECT_PLAYBACK_MAX_BYTES + 1


def test_large_mov_uses_the_short_preview(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    clip = _clip(tmp_path, "clip.mov", 0.4)
    added = add_video(tmp_path, "pilot", clip)
    with connect(project_sqlite(tmp_path, "pilot")) as connection:
        connection.execute(
            "UPDATE videos SET size_bytes = ? WHERE video_id = ?",
            (video_page._DIRECT_PLAYBACK_MAX_BYTES + 1, added["video_id"]),
        )
    with _client(tmp_path) as client:
        page = client.get("/video")
        assert ui_copy.PLAYBACK_PREVIEW not in page.text
        assert ui_copy.PLAYBACK_FILE not in page.text
        src = _video_src(page.text)
        assert src == f"/media/pilot/{added['video_id']}"
        media = client.get(str(src))
    preview = (
        project_directory(tmp_path, "pilot")
        / "media"
        / "previews"
        / f"{added['video_id']}.mp4"
    )
    assert preview.is_file()
    assert media.status_code == 200
    assert media.content == preview.read_bytes()
    assert media.content != clip.read_bytes()


def test_symlink_and_dotdot_are_not_a_media_path(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    clip = _clip(tmp_path)
    added = add_video(tmp_path, "pilot", clip)
    stored = Path(str(added["path"]))
    secret = tmp_path.parent / f"secret-{tmp_path.name}.bin"
    secret.write_bytes(b"SECRET-BYTES-OUTSIDE")
    stored.unlink()
    stored.symlink_to(secret)
    app = create_app(tmp_path)
    with _client(tmp_path) as client:
        escaped = client.get(f"/media/pilot/{added['video_id']}")
        queried = client.get(
            f"/media/pilot/{added['video_id']}",
            params={"path": str(secret)},
        )
    assert escaped.status_code == 404
    assert b"SECRET-BYTES-OUTSIDE" not in escaped.content
    assert queried.status_code == 404
    assert b"SECRET-BYTES-OUTSIDE" not in queried.content
    for path in (f"/media/pilot/{added['video_id']}/..", "/media/pilot/..", "/media/pilot/%2e%2e"):
        status, body = _raw(app, "GET", path, [])
        assert status == 404
        assert b"SECRET-BYTES-OUTSIDE" not in body
    secret.unlink(missing_ok=True)


def test_escaped_non_mp4_is_not_opened_or_previewed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = {"n": 0}
    real = video_page.playback_path

    def wrapped(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(video_page, "playback_path", wrapped)
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    added = add_video(tmp_path, "pilot", _clip(tmp_path, "clip.mov", 0.3))
    with connect(project_sqlite(tmp_path, "pilot")) as connection:
        connection.execute(
            "UPDATE videos SET size_bytes = ? WHERE video_id = ?",
            (video_page._DIRECT_PLAYBACK_MAX_BYTES + 1, added["video_id"]),
        )
    stored = Path(str(added["path"]))
    outside = tmp_path.parent / f"outside-mov-{tmp_path.name}"
    outside.mkdir()
    secret = outside / "secret.mov"
    secret.write_bytes(b"SECRET-MOV-OUTSIDE")
    stored.unlink()
    stored.symlink_to(secret)
    try:
        with _client(tmp_path) as client:
            page = client.get("/video")
            media = client.get(f"/media/pilot/{added['video_id']}")
        assert calls["n"] == 0
        assert page.status_code == 200
        assert "<video" not in page.text
        assert b"SECRET-MOV-OUTSIDE" not in page.content
        assert media.status_code == 404
        assert b"SECRET-MOV-OUTSIDE" not in media.content
        assert not (outside / "previews").exists()
        assert list(outside.rglob("*.mp4")) == []
        assert not (project_directory(tmp_path, "pilot") / "media" / "previews").exists()
    finally:
        secret.unlink(missing_ok=True)
        for path in sorted(outside.rglob("*"), reverse=True):
            if path.is_dir() and not path.is_symlink():
                path.rmdir()
            else:
                path.unlink(missing_ok=True)
        outside.rmdir()


def test_short_preview_does_not_follow_an_escaped_media_symlink(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    clip = tmp_path / "clip.mov"
    clip.write_bytes(b"inside-source")
    outside = tmp_path.parent / f"outside-media-{tmp_path.name}"
    outside.mkdir()
    media = project_directory(tmp_path, "pilot") / "media"
    media.rmdir()
    media.symlink_to(outside, target_is_directory=True)
    record = {
        "video_id": "v_escape",
        "path": str(clip),
        "size_bytes": video_page._DIRECT_PLAYBACK_MAX_BYTES + 1,
        "duration_s": 10.0,
        "start_offset_s": 0.0,
    }
    try:
        with pytest.raises(LabError):
            video_page.playback_path(tmp_path, "pilot", record)  # type: ignore[arg-type]
        assert not (outside / "previews").exists()
        assert list(outside.iterdir()) == []
    finally:
        media.unlink(missing_ok=True)
        for path in sorted(outside.rglob("*"), reverse=True):
            if path.is_dir() and not path.is_symlink():
                path.rmdir()
            else:
                path.unlink(missing_ok=True)
        outside.rmdir()


def test_short_preview_does_not_follow_an_escaped_previews_symlink(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    add_video(tmp_path, "pilot", _clip(tmp_path))
    record = list_videos(tmp_path, "pilot")[0]
    record["size_bytes"] = video_page._DIRECT_PLAYBACK_MAX_BYTES + 1
    outside = tmp_path.parent / f"outside-previews-{tmp_path.name}"
    outside.mkdir()
    previews = project_directory(tmp_path, "pilot") / "media" / "previews"
    previews.symlink_to(outside, target_is_directory=True)
    try:
        with pytest.raises(LabError):
            video_page.playback_path(tmp_path, "pilot", record)
        assert list(outside.iterdir()) == []
    finally:
        previews.unlink(missing_ok=True)
        for path in sorted(outside.rglob("*"), reverse=True):
            if path.is_dir() and not path.is_symlink():
                path.rmdir()
            else:
                path.unlink(missing_ok=True)
        outside.rmdir()


def test_large_source_plays_a_short_encoder_preview(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    clip = _clip(tmp_path)
    added = add_video(tmp_path, "pilot", clip)
    record = list_videos(tmp_path, "pilot")[0]
    record["size_bytes"] = video_page._DIRECT_PLAYBACK_MAX_BYTES + 1
    preview = video_page.playback_path(tmp_path, "pilot", record)
    assert preview.name == f"{added['video_id']}.mp4"
    assert preview.parent.name == "previews"
    assert preview.is_file()
    assert preview.stat().st_size > 0
    assert preview.resolve() != Path(str(record["path"])).resolve()
    assert (preview.parent / f"{added['video_id']}.csv").is_file()
    again = video_page.playback_path(tmp_path, "pilot", record)
    assert again == preview


def test_bad_interval_does_not_enqueue(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    added = add_video(tmp_path, "pilot", _clip(tmp_path))
    with _client(tmp_path) as client:
        response = client.post(
            "/video/baseline",
            data={"video_id": added["video_id"], "start_s": "4", "end_s": "1"},
            follow_redirects=True,
        )
    assert "полуоткрытый" in response.text or "раньше начала" in response.text
    assert _requests(tmp_path, "pilot") == []
    assert "Экскаватор" not in response.text


def test_missing_weights_show_the_doctor_notice(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    added = add_video(tmp_path, "pilot", _clip(tmp_path))
    assert not official_nano_weights_path(tmp_path).is_file()
    with _client(tmp_path) as client:
        page = client.get("/video")
        assert "Официальные веса" not in page.text
        assert "Official RF-DETR weights" not in page.text
        assert "локальный кэш" not in page.text
        buttons = dict(_submit_buttons(page.text))
        assert buttons[ui_copy.BASELINE_BUTTON] is True
        done = client.post(
            "/video/baseline",
            data={
                "video_id": added["video_id"],
                "start_s": _input_value(page.text, "start_s"),
                "end_s": _input_value(page.text, "end_s"),
            },
            follow_redirects=True,
        )
    assert _requests(tmp_path, "pilot") == []
    assert "Экскаватор" not in done.text
    assert "Официальные веса" not in done.text
    assert "Official RF-DETR weights" not in done.text
    assert dict(_submit_buttons(done.text))[ui_copy.BASELINE_BUTTON] is True
    outcome = video_page.request_baseline_run(
        tmp_path,
        "pilot",
        added["video_id"],
        float(_input_value(page.text, "start_s") or "0"),
        float(_input_value(page.text, "end_s") or "0"),
    )
    assert outcome.job_id is None
    assert outcome.notice is None
    assert outcome.error is None


def test_running_job_disables_baseline_and_blocks_a_second_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FAKE_DETECTOR_ENV, "1")
    calls = {"n": 0}
    real = video_page.enqueue_analyze

    def wrapped(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(video_page, "enqueue_analyze", wrapped)
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    added = add_video(tmp_path, "pilot", _clip(tmp_path))
    weights = official_nano_weights_path(tmp_path)
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"cached-name-only")
    job = start_job(
        tmp_path,
        "pilot",
        kind="analyze",
        worker="sleep",
        video_duration_s=30,
        step_s=1,
        sleep_s=2,
    )
    try:
        with _client(tmp_path) as client:
            page = client.get("/video")
            assert page.status_code == 200
            buttons = dict(_submit_buttons(page.text))
            assert buttons[ui_copy.BASELINE_BUTTON] is True
            assert "выполняется" in page.text
            assert ui_copy.RUNS_NO_BOXES in page.text
            again = client.get("/video")
            assert dict(_submit_buttons(again.text))[ui_copy.BASELINE_BUTTON] is True
            posted = client.post(
                "/video/baseline",
                data={
                    "video_id": added["video_id"],
                    "start_s": "0",
                    "end_s": "0.2",
                },
                follow_redirects=True,
            )
        assert len(reopen(tmp_path)) == 1
        assert "уже выполняется задание" in posted.text
        outcome = video_page.request_baseline_run(
            tmp_path,
            "pilot",
            added["video_id"],
            0.0,
            0.2,
        )
        assert outcome.job_id is None
        assert outcome.error is not None
        assert "уже выполняется задание" in outcome.error
        assert calls["n"] == 0
        assert _requests(tmp_path, "pilot") == []
        assert weights.read_bytes() == b"cached-name-only"
    finally:
        _stop_jobs(tmp_path, "pilot")
        assert job.job_id


def test_baseline_enqueues_once_and_a_rerun_does_not_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FAKE_DETECTOR_ENV, "1")
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    added = add_video(tmp_path, "pilot", _clip(tmp_path))
    weights = official_nano_weights_path(tmp_path)
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"cached-name-only")
    try:
        with _client(tmp_path) as client:
            done = client.post(
                "/video/baseline",
                data={
                    "video_id": added["video_id"],
                    "start_s": "0",
                    "end_s": "0.3",
                },
                follow_redirects=True,
            )
            requests = _requests(tmp_path, "pilot")
            assert len(requests) == 1
            body = json.loads(requests[0].read_text(encoding="utf-8"))
            assert body["model_id"] == "baseline_coco"
            assert body["model_origin"] == "baseline_coco"
            assert body["video_id"] == added["video_id"]
            assert body["start_s"] == 0.0
            assert body["end_s"] == 0.3
            client.get("/video")
            assert len(_requests(tmp_path, "pilot")) == 1
        assert weights.read_bytes() == b"cached-name-only"
        assert ui_copy.BASELINE_CAPTION in done.text
        assert "Экскаватор" not in done.text
        assert ui_copy.RUNS_NO_BOXES in done.text or ui_copy.BASELINE_QUEUED in done.text
        jobs = reopen(tmp_path)
        assert jobs
        assert ui_copy.BASELINE_QUEUED in done.text
        assert jobs[0].job_id in done.text
    finally:
        _stop_jobs(tmp_path, "pilot")


def test_interval_bounds_use_start_offset_and_preview_seeks_there(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = {
        "video_id": "v_late",
        "start_offset_s": 300,
        "duration_s": 10,
        "size_bytes": video_page._DIRECT_PLAYBACK_MAX_BYTES + 1,
        "path": str(tmp_path / "late.mp4"),
    }
    assert video_page.interval_bounds(record) == (300.0, 310.0)
    assert video_page.interval_bounds({"start_offset_s": 0, "duration_s": 10}) == (
        0.0,
        10.0,
    )
    assert video_page.interval_bounds({"start_offset_s": None, "duration_s": 10}) == (
        None,
        None,
    )
    assert video_page.interval_bounds({"start_offset_s": -4, "duration_s": 10}) == (
        None,
        None,
    )
    missing = video_page.offset_line({"start_offset_s": None})
    assert ui_copy.NOT_CALCULATED in missing
    assert _ZERO_TOKEN.search(missing) is None
    assert _ZERO_TOKEN.search(video_page.offset_line({"start_offset_s": 0}))

    calls: dict[str, object] = {}

    class _Reader:
        def __init__(self, path: Path, *, video_id: str) -> None:
            calls["video_id"] = video_id
            calls["path"] = Path(path)

        def iterate(self, start_s: float, end_s: float | None = None):
            calls["iterate"] = (start_s, end_s)
            yield _Held()

    def _write(
        frames: object,
        output_path: Path,
        *,
        output_fps: float,
        start_s: float,
        end_s: float,
        map_path: Path | None = None,
    ) -> list[object]:
        del frames, output_fps
        calls["write"] = (start_s, end_s)
        Path(output_path).write_bytes(b"preview")
        if map_path is not None:
            Path(map_path).write_text("map", encoding="utf-8")
        return []

    monkeypatch.setattr("excavation_lab.media.reader.FrameReader", _Reader)
    monkeypatch.setattr("excavation_lab.media.encoder.write_preview", _write)
    (tmp_path / "late.mp4").write_bytes(b"stub")
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    preview = video_page.playback_path(tmp_path, "pilot", record)  # type: ignore[arg-type]
    assert calls["iterate"] == (300.0, 302.0)
    assert calls["write"] == (300.0, 302.0)
    assert preview.name == "v_late.mp4"


class _Held:
    decode_status = DecodeStatus.ok
    source_frame_index = 0
    source_time_s = 300.0
    image = (((0, 0, 0),),)


def test_offset_widgets_default_to_the_source_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    late = {
        "video_id": "v_late",
        "name": "clip.mp4",
        "path": str(tmp_path / "missing.mp4"),
        "size_bytes": 128,
        "duration_s": 10.0,
        "start_offset_s": 300.0,
        "sha256": None,
        "rotation_deg": None,
        "source_url": None,
    }
    monkeypatch.setattr(video_page.storage, "list_videos", lambda *_args: [late])
    with _client(tmp_path) as client:
        page = client.get("/video")
    assert page.status_code == 200
    assert _input_value(page.text, "start_s") == "300"
    assert _input_value(page.text, "end_s") == "310"
    assert f"{ui_copy.OFFSET_LABEL}: 300 с." in page.text

    blank = dict(late)
    blank["video_id"] = "v_blank"
    blank["start_offset_s"] = None
    monkeypatch.setattr(video_page.storage, "list_videos", lambda *_args: [blank])
    with _client(tmp_path) as client:
        empty = client.get("/video")
    assert _input_value(empty.text, "start_s") is None
    assert _input_value(empty.text, "end_s") is None
    offset = re.search(rf"{re.escape(ui_copy.OFFSET_LABEL)}:[^<]*", empty.text)
    assert offset is not None
    assert ui_copy.NOT_CALCULATED in offset.group(0)
    assert _ZERO_TOKEN.search(offset.group(0)) is None


def test_same_filename_selectbox_includes_video_id(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    left = _clip(tmp_path, "left/clip.mp4", 0.2)
    right = _clip(tmp_path, "right/clip.mp4", 0.2)
    first = add_video(tmp_path, "pilot", left)
    second = add_video(tmp_path, "pilot", right)
    assert video_page.video_choice_label(
        {"name": "clip.mp4", "video_id": first["video_id"]}
    ) != video_page.video_choice_label(
        {"name": "clip.mp4", "video_id": second["video_id"]}
    )
    with _client(tmp_path) as client:
        page = client.get("/video")
    options = _video_options(page.text)
    assert len(options) == 2
    labels = [label for _value, label, _on in options]
    assert any(first["video_id"] in item for item in labels)
    assert any(second["video_id"] in item for item in labels)


def test_video_query_selects_on_the_video_page_only(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    add_video(tmp_path, "pilot", _clip(tmp_path, "one.mp4", 0.2))
    second = add_video(tmp_path, "pilot", _clip(tmp_path, "two.mp4", 0.2))
    video_id = str(second["video_id"])
    with _client(tmp_path) as client:
        chosen = client.get("/video", params={"video": video_id})
        home = client.get("/", params={"video": video_id})
    assert chosen.status_code == 200
    selected = [value for value, _label, on in _video_options(chosen.text) if on]
    assert selected == [video_id]
    assert _input_value(chosen.text, "video_id") == video_id
    assert f"/media/pilot/{video_id}" in chosen.text
    assert home.status_code == 200
    assert "<video" not in home.text.lower()
    assert f"/media/{video_id}" not in home.text
    assert f"/media/pilot/{video_id}" not in home.text
    assert 'action="/video/choose"' not in home.text


def test_cached_weights_clear_a_stored_download_notice(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    add_video(tmp_path, "pilot", _clip(tmp_path, "clip.mp4", 0.2))
    weights = official_nano_weights_path(tmp_path)
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"cached-name-only")
    write_flash(tmp_path, {"notice": "скачайте-заново-маркер"})
    with _client(tmp_path) as client:
        page = client.get("/video")
    assert "скачайте-заново-маркер" not in page.text
    assert "Официальные веса" not in page.text


def test_english_video_page_keeps_only_the_language_endonym(tmp_path: Path) -> None:
    write_locale_file(tmp_path, "en")
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    write_project_file(tmp_path, "pilot")
    book = ui_copy.catalog("en")
    with _client(tmp_path) as client:
        empty = client.get("/video")
        labels = _button_labels(empty.text)
        assert book.choose_file_button in labels
        assert book.add_local_video in empty.text
        assert book.local_path_label in empty.text
        assert book.local_path_placeholder not in empty.text
        assert book.upload_label not in empty.text
        assert book.upload_button not in labels
        assert book.add_video_button not in labels
        assert book.add_another not in empty.text
        assert "<summary>Дополнительно</summary>" not in empty.text
        assert "<summary>Advanced</summary>" not in empty.text
        assert empty.text.count("<summary>") == 0
        assert 'id="project-create-panel"' not in empty.text
        assert 'id="project-manager"' not in empty.text
        assert 'id="lab-home"' in empty.text
        assert book.baseline_button not in labels
        assert _foreign_cyrillic(empty.text) == ""
        added = client.post(
            "/video/add",
            data={"local_path": str(_clip(tmp_path))},
            follow_redirects=True,
        )
    added_labels = _button_labels(added.text)
    assert book.baseline_button in added_labels
    assert book.choose_file_button in added_labels
    assert book.add_video_button not in added_labels
    assert book.upload_button not in added_labels
    assert book.upload_label not in added.text
    assert book.add_another not in added.text
    assert "<summary>Advanced</summary>" not in added.text
    assert added.text.count("<summary>") == 0
    assert 'id="project-create-panel"' not in added.text
    assert 'id="project-manager"' not in added.text
    assert 'id="lab-home"' in added.text
    assert 'class="lab-video"' in added.text
    assert "Официальные веса" not in added.text
    assert "Official RF-DETR weights" not in added.text
    assert _foreign_cyrillic(added.text) == ""


def test_english_orphaned_partial_status_has_one_period() -> None:
    job = JobRecord(
        job_id="job-1",
        status=JobStatus.failed,
        orphaned=True,
        partial=True,
        analyzed_end_s=1.5,
    )
    line = video_page.job_status_line(job, ui_copy.catalog("en"))
    assert ".." not in line
    assert line == (
        "job-1: Failed. The process disappeared. The result is partial. 1.500 s."
    )
