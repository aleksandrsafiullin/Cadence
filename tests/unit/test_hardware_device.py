"""Inference device order and the missing-FFmpeg doctor branch."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from excavation_lab import hardware
from excavation_lab.cli import app
from excavation_lab.errors import FfmpegMissingError, LabError

_RUNNER = CliRunner()
_NOT_RUN = ("model_load", "inference", "training_step")


def _cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)


def _patch_quiet_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a report test off the real GPU and off a guessed FFmpeg path."""
    monkeypatch.setattr(hardware, "probe_cuda", lambda: False)
    monkeypatch.setattr(hardware, "probe_mps", lambda: False)
    monkeypatch.setattr(hardware, "probe_cpu", lambda: True)
    monkeypatch.setattr(hardware, "mps_built", lambda: False)

    def which(name: str, mode: int = os.F_OK, path: str | None = None) -> str | None:
        assert path is None
        if name in {"ffmpeg", "ffprobe"}:
            return name
        return None

    monkeypatch.setattr(hardware.shutil, "which", which)


def test_select_inference_device_order(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    flags = {"cuda": True, "mps": True, "cpu": True}

    def probe_cuda() -> bool:
        seen.append("cuda")
        return flags["cuda"]

    def probe_mps() -> bool:
        seen.append("mps")
        return flags["mps"]

    def probe_cpu() -> bool:
        seen.append("cpu")
        return flags["cpu"]

    monkeypatch.setattr(hardware, "probe_cuda", probe_cuda)
    monkeypatch.setattr(hardware, "probe_mps", probe_mps)
    monkeypatch.setattr(hardware, "probe_cpu", probe_cpu)

    assert hardware.select_inference_device() == "cuda"
    assert seen == ["cuda"]

    flags["cuda"] = False
    seen.clear()
    assert hardware.select_inference_device() == "mps"
    assert seen == ["cuda", "mps"]

    flags["mps"] = False
    seen.clear()
    assert hardware.select_inference_device() == "cpu"
    assert seen == ["cuda", "mps", "cpu"]

    flags["cpu"] = False
    seen.clear()
    with pytest.raises(LabError) as caught:
        hardware.select_inference_device()
    assert seen == ["cuda", "mps", "cpu"]
    assert caught.value.code == "device_unavailable"
    assert _cyrillic(caught.value.message)


def test_ram_bytes_are_page_size_times_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    def sysconf(name: str) -> int:
        values = {"SC_PAGE_SIZE": 16384, "SC_PHYS_PAGES": 1024}
        return values[name]

    monkeypatch.setattr(hardware.os, "sysconf", sysconf)
    assert hardware.read_ram_bytes() == 16384 * 1024


def test_unknown_ram_is_null_and_not_a_guess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def sysconf(name: str) -> int:
        raise ValueError(name)

    monkeypatch.setattr(hardware.os, "sysconf", sysconf)
    _patch_quiet_machine(monkeypatch)
    report = hardware.run_doctor(tmp_path / "environment_report.json")
    assert report["ram_bytes"] is None
    assert report["ram_message"] == "не рассчитано"
    text = (tmp_path / "environment_report.json").read_text(encoding="utf-8")
    assert "не рассчитано" in text
    assert "ram_gb" not in report


@pytest.mark.parametrize("found", [set(), {"ffmpeg"}, {"ffprobe"}])
def test_missing_ffmpeg_or_ffprobe(
    found: set[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def which(name: str, mode: int = os.F_OK, path: str | None = None) -> str | None:
        assert path is None
        if name in found:
            return name
        return None

    monkeypatch.setattr(hardware.shutil, "which", which)
    monkeypatch.setattr(hardware, "probe_cuda", lambda: False)
    monkeypatch.setattr(hardware, "probe_mps", lambda: False)
    monkeypatch.setattr(hardware, "probe_cpu", lambda: True)
    monkeypatch.setattr(hardware, "mps_built", lambda: False)
    destination = tmp_path / "environment_report.json"
    with pytest.raises(FfmpegMissingError) as caught:
        hardware.run_doctor(destination)
    error = caught.value
    assert isinstance(error, LabError)
    assert error.code == "ffmpeg_missing"
    assert error.code.isascii()
    assert "ffprobe" in error.message
    assert "ffmpeg" in error.message
    assert _cyrillic(error.message)
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["ffmpeg_available"] is ("ffmpeg" in found)
    assert saved["ffprobe_available"] is ("ffprobe" in found)
    for key in _NOT_RUN:
        assert saved[key] == "not_run"


def test_report_does_not_claim_runtime_probes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_quiet_machine(monkeypatch)
    destination = tmp_path / "environment_report.json"
    report = hardware.run_doctor(destination)
    on_disk = json.loads(destination.read_text(encoding="utf-8"))
    assert on_disk == report
    assert report["inference_device"] == "cpu"
    assert report["cuda_available"] is False
    assert report["mps_available"] is False
    assert report["ffmpeg_available"] is True
    assert report["ffprobe_available"] is True
    assert set(report["versions"]) == {
        "torch",
        "torchvision",
        "rfdetr",
        "trackers",
        "supervision",
        "av",
        "streamlit",
    }
    for key in _NOT_RUN:
        assert report[key] == "not_run"
    assert "passed" not in json.dumps(report)
    assert not (tmp_path / "environment_report.json.tmp").exists()


def test_ffmpeg_lookup_uses_which_without_a_fixed_path() -> None:
    source = Path(hardware.__file__).read_text(encoding="utf-8")
    assert "shutil.which" in source
    assert "/usr/bin/ffmpeg" not in source
    assert "/opt/homebrew" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert not any(name == "streamlit" or name.startswith("streamlit.") for name in names)


def test_doctor_json_returns_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_quiet_machine(monkeypatch)
    destination = tmp_path / "environment_report.json"
    result = _RUNNER.invoke(
        app,
        ["doctor", "--json", "--output", str(destination)],
    )
    assert result.exit_code == 0
    assert "Traceback" not in result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["report_path"] == str(destination.resolve())
    assert payload["data"]["inference_device"] == "cpu"
    assert payload["data"]["model_load"] == "not_run"
    assert payload["data"]["inference"] == "not_run"
    assert payload["data"]["training_step"] == "not_run"


def test_doctor_prints_path_and_russian_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_quiet_machine(monkeypatch)
    destination = tmp_path / "environment_report.json"
    result = _RUNNER.invoke(app, ["doctor", "--output", str(destination)])
    assert result.exit_code == 0
    assert str(destination.resolve()) in result.stdout
    assert "Устройство: cpu" in result.stdout
    assert "не запускались" in result.stdout
    assert _cyrillic(result.stdout)


def test_doctor_json_missing_ffmpeg_is_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def which(name: str, mode: int = os.F_OK, path: str | None = None) -> None:
        assert path is None
        assert name

    monkeypatch.setattr(hardware.shutil, "which", which)
    monkeypatch.setattr(hardware, "probe_cuda", lambda: False)
    monkeypatch.setattr(hardware, "probe_mps", lambda: False)
    monkeypatch.setattr(hardware, "probe_cpu", lambda: True)
    monkeypatch.setattr(hardware, "mps_built", lambda: False)
    destination = tmp_path / "environment_report.json"
    result = _RUNNER.invoke(
        app,
        ["doctor", "--json", "--output", str(destination)],
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "ffmpeg_missing"
    assert "ffprobe" in payload["error"]["message"]
    assert "ffmpeg" in payload["error"]["message"]
    assert _cyrillic(payload["error"]["message"])
