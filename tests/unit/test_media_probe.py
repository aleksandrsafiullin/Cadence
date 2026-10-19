"""ffprobe import: container duration, provenance URL, and failure codes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Self

import av
import pytest
from typer.testing import CliRunner

from excavation_lab.cli import app
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.errors import (
    CorruptMediaError,
    FfmpegMissingError,
    MediaNotFoundError,
    PathNotAllowedError,
)
from excavation_lab.media import probe
from excavation_lab.storage import (
    create_project,
    get_video,
    project_directory,
    replace_extra_roots,
)

# pytest puts tests/unit on sys.path, and site-packages already owns a
# namespace package named tests. Load the fixture file directly.
_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SYNTHETIC_SPEC = importlib.util.spec_from_file_location(
    "excavation_lab_synthetic_mp4",
    _SYNTHETIC_PATH,
)
assert _SYNTHETIC_SPEC is not None and _SYNTHETIC_SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SYNTHETIC_SPEC)
_SYNTHETIC_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_mp4 = _SYNTHETIC.write_synthetic_mp4

_RUNNER = CliRunner()
_PILOT_URL = "https://www.youtube.com/watch?v=kjWsR2lVeek"


def _cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)


def _project(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    create_project("pilot", workspace, catalog_id="excavation")
    return workspace


def _container_duration_s(path: Path) -> float:
    """Header duration. Does not decode the picture stream."""
    with av.open(str(path)) as container:
        assert container.duration is not None
        # container.duration is in AV_TIME_BASE units. This PyAV exposes that
        # base as an int (1000000); older builds expose a Fraction.
        base = av.time_base
        if isinstance(base, int):
            return container.duration / base
        return float(container.duration * base)


def _ffprobe_payload(monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]) -> None:
    def which(name: str, *_args: object, **_kwargs: object) -> str | None:
        assert name == "ffprobe"
        return "ffprobe"

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert isinstance(args, list)
        assert kwargs.get("shell", False) is False
        body = json.dumps(payload).encode("utf-8")
        return subprocess.CompletedProcess(args, 0, stdout=body, stderr=b"")

    monkeypatch.setattr(probe.shutil, "which", which)
    monkeypatch.setattr(probe.subprocess, "run", run)


def test_fixture_docstring_says_synthetic_not_accuracy_evidence() -> None:
    text = write_synthetic_mp4.__doc__ or ""
    assert "synthetic" in text
    assert "not accuracy evidence" in text


def test_synthetic_probe_uses_container_duration(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    seconds = 1.0
    fps = 10
    write_synthetic_mp4(path, seconds, fps)
    container_s = _container_duration_s(path)
    metadata = probe.probe_media(path, roots=[tmp_path], video_id="clip")

    assert metadata["duration_s"] == pytest.approx(container_s, abs=0.05)
    assert metadata["duration_s"] == pytest.approx(seconds, abs=2 / fps)
    assert metadata["rotation_deg"] is None
    assert metadata["sar"] is None
    assert metadata["dar"] is None
    assert metadata["width"] == 64
    assert metadata["height"] == 48
    assert metadata["codec"] == "h264"
    assert metadata["fps_nominal"] == pytest.approx(fps)
    assert metadata["fps_average"] == pytest.approx(fps)
    assert metadata["time_base"].count("/") == 1
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["size_bytes"] == path.stat().st_size
    assert metadata["source_url"] is None
    assert metadata["path"] == str(path.resolve())
    assert metadata["name"] == "clip.mp4"


def test_ffprobe_uses_argument_array(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "clip.mp4"
    write_synthetic_mp4(path, 0.4, 10)
    calls: list[tuple[list[str], dict[str, object]]] = []
    real_run = subprocess.run

    def wrapped(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert isinstance(args, list)
        calls.append((list(args), dict(kwargs)))
        return real_run(args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(probe.subprocess, "run", wrapped)
    probe.probe_media(path, roots=[tmp_path], video_id="clip")

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert kwargs.get("shell", False) is False
    assert command[0] == probe.shutil.which("ffprobe")
    assert command[0] is not None
    assert "-print_format" in command
    assert "json" in command
    assert "-show_format" in command
    assert "-show_streams" in command
    assert command[-1] == str(path.resolve())


def test_sha256_reads_in_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"abcdefghijklmnopqrstuvwxyz012345"
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    monkeypatch.setattr(probe, "_HASH_CHUNK_BYTES", 5)
    sizes: list[int] = []

    class _Chunks:
        def __init__(self) -> None:
            self.pos = 0

        def read(self, size: int = -1) -> bytes:
            take = len(payload) - self.pos if size is None or size < 0 else size
            chunk = payload[self.pos : self.pos + take]
            sizes.append(len(chunk))
            self.pos += len(chunk)
            return chunk

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    real_open = Path.open

    def wrapped(self: Path, mode: str = "r", *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if self == path and mode == "rb":
            return _Chunks()
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", wrapped)
    assert probe.file_sha256(path) == hashlib.sha256(payload).hexdigest()
    assert sizes
    assert max(size for size in sizes if size) <= 5
    assert sum(sizes) == len(payload)


def test_absent_rotation_and_sar_stay_null(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"header-only")
    _ffprobe_payload(
        monkeypatch,
        {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 64,
                    "height": 48,
                    "sample_aspect_ratio": "N/A",
                    "display_aspect_ratio": "0:1",
                    "r_frame_rate": "0/0",
                    "avg_frame_rate": "10/1",
                    "time_base": "1/10240",
                }
            ],
            "format": {"duration": "1.5"},
        },
    )
    metadata = probe.probe_media(path, roots=[tmp_path], video_id="clip")
    assert metadata["rotation_deg"] is None
    assert metadata["sar"] is None
    assert metadata["dar"] is None
    assert metadata["fps_nominal"] is None
    assert metadata["fps_average"] == pytest.approx(10)
    assert metadata["duration_s"] == pytest.approx(1.5)
    assert metadata["start_offset_s"] is None


def test_rotation_and_sar_come_from_ffprobe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"header-only")
    _ffprobe_payload(
        monkeypatch,
        {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 64,
                    "height": 48,
                    "sample_aspect_ratio": "16:9",
                    "display_aspect_ratio": "16:9",
                    "r_frame_rate": "30000/1001",
                    "avg_frame_rate": "30/1",
                    "time_base": "1/90000",
                    "side_data_list": [
                        {"side_data_type": "Display Matrix", "rotation": -90}
                    ],
                    "tags": {"rotate": "90"},
                }
            ],
            "format": {"duration": "2.0", "start_time": "0.040000"},
        },
    )
    metadata = probe.probe_media(path, roots=[tmp_path], video_id="clip")
    assert metadata["rotation_deg"] == pytest.approx(-90)
    assert metadata["sar"] == "16:9"
    assert metadata["dar"] == "16:9"
    assert metadata["start_offset_s"] == pytest.approx(0.04)
    assert metadata["fps_nominal"] == pytest.approx(30000 / 1001)


def test_explicit_zero_rotation_is_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"header-only")
    _ffprobe_payload(
        monkeypatch,
        {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 32,
                    "height": 32,
                    "r_frame_rate": "25/1",
                    "avg_frame_rate": "25/1",
                    "time_base": "1/12800",
                    "side_data_list": [
                        {"side_data_type": "Display Matrix", "rotation": 0}
                    ],
                }
            ],
            "format": {"duration": "0.2", "start_time": "0.000000"},
        },
    )
    metadata = probe.probe_media(path, roots=[tmp_path], video_id="clip")
    assert metadata["rotation_deg"] == 0.0
    assert metadata["start_offset_s"] == 0.0


def test_rotate_tag_is_used_when_side_data_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"header-only")
    _ffprobe_payload(
        monkeypatch,
        {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 32,
                    "height": 32,
                    "time_base": "1/25",
                    "tags": {"rotate": "90", "sample_aspect_ratio": "4:3"},
                }
            ],
            "format": {"duration": "0.5"},
        },
    )
    metadata = probe.probe_media(path, roots=[tmp_path], video_id="clip")
    assert metadata["rotation_deg"] == pytest.approx(90)
    assert metadata["sar"] == "4:3"


def test_random_bytes_are_corrupt_media(tmp_path: Path) -> None:
    path = tmp_path / "noise.mp4"
    path.write_bytes(os.urandom(256))
    with pytest.raises(CorruptMediaError) as caught:
        probe.probe_media(path, roots=[tmp_path], video_id="noise")
    assert caught.value.code == "corrupt_media"
    assert caught.value.code.isascii()
    assert _cyrillic(caught.value.message)
    assert "Замените" in caught.value.message


def test_missing_file_is_media_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "absent.mp4"
    with pytest.raises(MediaNotFoundError) as caught:
        probe.probe_media(missing, roots=[tmp_path], video_id="absent")
    assert caught.value.code == "media_not_found"
    assert caught.value.code.isascii()
    assert _cyrillic(caught.value.message)
    assert str(missing.resolve()) in caught.value.message
    assert "не скачивает" in caught.value.message


def test_missing_ffprobe_uses_which(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"present")
    seen: list[str] = []

    def which(name: str, *_args: object, **_kwargs: object) -> None:
        seen.append(name)

    monkeypatch.setattr(probe.shutil, "which", which)
    with pytest.raises(FfmpegMissingError) as caught:
        probe.probe_media(path, roots=[tmp_path], video_id="clip")
    assert seen == ["ffprobe"]
    assert caught.value.code == "ffmpeg_missing"
    assert caught.value.code.isascii()
    assert _cyrillic(caught.value.message)
    assert "ffprobe" in caught.value.message


def test_path_outside_roots_is_rejected(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    outside = tmp_path / "outside" / "clip.mp4"
    outside.parent.mkdir()
    outside.write_bytes(b"nope")
    with pytest.raises(PathNotAllowedError) as caught:
        probe.add_video(workspace, "pilot", outside)
    assert caught.value.code == "path_not_allowed"
    assert _cyrillic(caught.value.message)


def test_add_registers_path_without_copying(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    source = workspace / "incoming" / "pilot.mp4"
    source.parent.mkdir()
    write_synthetic_mp4(source, 0.5, 10)
    metadata = probe.add_video(workspace, "pilot", source, video_id="pilot-clip")
    media = project_directory(workspace, "pilot") / "media"
    assert list(media.iterdir()) == []
    assert metadata["path"] == str(source.resolve())
    assert source.is_file()
    row = get_video(workspace, "pilot", "pilot-clip")
    assert row["path"] == str(source.resolve())
    assert row["sha256"] == metadata["sha256"]
    assert row["duration_s"] == pytest.approx(metadata["duration_s"])
    assert row["source_url"] is None


def test_failed_copy_is_removed(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    source = workspace / "noise.mp4"
    source.write_bytes(os.urandom(128))
    with pytest.raises(CorruptMediaError):
        probe.add_video(workspace, "pilot", source, copy=True, video_id="noise")
    media = project_directory(workspace, "pilot") / "media"
    assert list(media.iterdir()) == []
    assert source.is_file()


def test_copy_only_when_flag_is_set(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    source = workspace / "incoming" / "pilot.mp4"
    source.parent.mkdir()
    write_synthetic_mp4(source, 0.5, 10)
    metadata = probe.add_video(
        workspace,
        "pilot",
        source,
        copy=True,
        video_id="copied",
    )
    media = project_directory(workspace, "pilot") / "media"
    copied = Path(metadata["path"])
    assert copied.parent == media.resolve()
    assert copied != source.resolve()
    assert source.is_file()
    assert copied.is_file()
    assert metadata["name"] == source.name
    assert metadata["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert get_video(workspace, "pilot", "copied")["path"] == str(copied)


def test_source_url_is_stored_and_not_fetched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = _project(tmp_path)
    source = workspace / "pilot.mp4"
    write_synthetic_mp4(source, 0.3, 10)

    def urlopen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("source URL was fetched")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    metadata = probe.add_video(
        workspace,
        "pilot",
        source,
        source_url=_PILOT_URL,
        video_id="pilot",
    )
    assert metadata["source_url"] == _PILOT_URL
    assert get_video(workspace, "pilot", "pilot")["source_url"] == _PILOT_URL
    assert metadata["path"] == str(source.resolve())


def test_extra_root_can_register_a_file(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    library = tmp_path / "library"
    library.mkdir()
    clip = library / "yard.mp4"
    write_synthetic_mp4(clip, 0.3, 10)
    with pytest.raises(PathNotAllowedError):
        probe.add_video(workspace, "pilot", clip, video_id="yard")
    replace_extra_roots(workspace, "pilot", [library])
    metadata = probe.add_video(workspace, "pilot", clip, video_id="yard")
    assert metadata["path"] == str(clip.resolve())


def test_reprobe_reports_mismatch_without_overwrite(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    path = workspace / "clip.mp4"
    other = workspace / "other.mp4"
    write_synthetic_mp4(path, 0.4, 10)
    write_synthetic_mp4(other, 0.8, 10)
    added = probe.add_video(workspace, "pilot", path, video_id="clip")
    same = probe.reprobe_video(workspace, "pilot", "clip")
    assert same["mismatch"] is False
    assert same["mismatched_fields"] == []
    assert same["video"]["sha256"] == added["sha256"]

    path.write_bytes(other.read_bytes())
    changed = probe.reprobe_video(workspace, "pilot", "clip")
    assert changed["mismatch"] is True
    assert changed["mismatched_fields"] == ["size_bytes", "sha256"]
    assert changed["observed"]["sha256"] != added["sha256"]
    stored = get_video(workspace, "pilot", "clip")
    assert stored["sha256"] == added["sha256"]
    assert stored["size_bytes"] == added["size_bytes"]
    assert stored["duration_s"] == pytest.approx(added["duration_s"])


def test_cli_json_add_and_probe(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    path = workspace / "pilot.mp4"
    write_synthetic_mp4(path, 0.5, 10)
    added = _RUNNER.invoke(
        app,
        [
            "video",
            "add",
            "--project",
            "pilot",
            "--path",
            str(path),
            "--workspace",
            str(workspace),
            "--video",
            "clip",
            "--source-url",
            _PILOT_URL,
            "--json",
        ],
    )
    assert added.exit_code == 0
    assert "Traceback" not in added.output
    payload = json.loads(added.stdout)
    assert list(payload) == ["ok", "data"]
    assert payload["ok"] is True
    assert payload["data"]["video_id"] == "clip"
    assert payload["data"]["source_url"] == _PILOT_URL
    assert payload["data"]["rotation_deg"] is None
    assert payload["data"]["path"] == str(path.resolve())
    media_copy = project_directory(workspace, "pilot") / "media" / "pilot.mp4"
    assert not media_copy.exists()

    probed = _RUNNER.invoke(
        app,
        [
            "video",
            "probe",
            "--project",
            "pilot",
            "--video",
            "clip",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert probed.exit_code == 0
    report = json.loads(probed.stdout)
    assert report["ok"] is True
    assert report["data"]["mismatch"] is False
    assert report["data"]["observed"]["sha256"] == payload["data"]["sha256"]


def test_cli_json_failure_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = _project(tmp_path)
    missing = _RUNNER.invoke(
        app,
        [
            "video",
            "add",
            "--project",
            "pilot",
            "--path",
            str(workspace / "missing.mp4"),
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert missing.exit_code == 1
    assert "Traceback" not in missing.output
    missing_body = json.loads(missing.stdout)
    assert missing_body["ok"] is False
    assert missing_body["error"]["code"] == "media_not_found"
    assert _cyrillic(missing_body["error"]["message"])

    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"nope")
    denied = _RUNNER.invoke(
        app,
        [
            "video",
            "add",
            "--project",
            "pilot",
            "--path",
            str(outside),
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert denied.exit_code == 1
    assert json.loads(denied.stdout)["error"]["code"] == "path_not_allowed"

    noise = workspace / "noise.mp4"
    noise.write_bytes(os.urandom(128))
    corrupt = _RUNNER.invoke(
        app,
        [
            "video",
            "add",
            "--project",
            "pilot",
            "--path",
            str(noise),
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert corrupt.exit_code == 1
    assert "Traceback" not in corrupt.output
    assert json.loads(corrupt.stdout)["error"]["code"] == "corrupt_media"

    present = workspace / "present.mp4"
    present.write_bytes(b"present")
    monkeypatch.setattr(probe.shutil, "which", lambda *_args, **_kwargs: None)
    missing_tool = _RUNNER.invoke(
        app,
        [
            "video",
            "add",
            "--project",
            "pilot",
            "--path",
            str(present),
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert missing_tool.exit_code == 1
    assert "Traceback" not in missing_tool.output
    tool_body = json.loads(missing_tool.stdout)
    assert tool_body["error"]["code"] == "ffmpeg_missing"
    assert _cyrillic(tool_body["error"]["message"])


def test_cli_text_mismatch_exits_zero(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    path = workspace / "clip.mp4"
    other = workspace / "other.mp4"
    write_synthetic_mp4(path, 0.4, 10)
    write_synthetic_mp4(other, 0.8, 10)
    added = _RUNNER.invoke(
        app,
        [
            "video",
            "add",
            "--project",
            "pilot",
            "--path",
            str(path),
            "--video",
            "clip",
            "--workspace",
            str(workspace),
        ],
    )
    assert added.exit_code == 0
    assert "Длительность" in added.stdout
    assert _cyrillic(added.stdout)
    path.write_bytes(other.read_bytes())
    probed = _RUNNER.invoke(
        app,
        [
            "video",
            "probe",
            "--project",
            "pilot",
            "--video",
            "clip",
            "--workspace",
            str(workspace),
        ],
    )
    assert probed.exit_code == 0
    assert "Файл изменился" in probed.stdout
    stored = get_video(workspace, "pilot", "clip")["sha256"]
    assert stored != hashlib.sha256(path.read_bytes()).hexdigest()
