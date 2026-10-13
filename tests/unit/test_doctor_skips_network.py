"""Doctor stays offline unless ``--load-weights`` is passed.

The weight-download test is ``manual``. Default pytest addopts exclude that
marker, so this module does not download Nano weights.
"""

from __future__ import annotations

import ast
import json
import os
import socket
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from excavation_lab import hardware
from excavation_lab.cli import app
from excavation_lab.models.adapter import FakeDetector

_RUNNER = CliRunner()
_NOT_RUN = ("model_load", "inference", "training_step")


def _cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)


def _patch_quiet_machine(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("doctor touched the network")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)


def _completed_step(device: str) -> dict[str, object]:
    return {
        "completed": True,
        "method": hardware.TRAINING_STEP_METHOD,
        "peak_memory_bytes": None,
        "latency_s": None,
        "device": device,
    }


class _EmptyPredict:
    size = "rfdetr-nano"
    resolution = 384
    class_names = ("person",)

    def predict(
        self,
        frame: np.ndarray,
        threshold: float,
        include_source_image: bool,
    ) -> SimpleNamespace:
        assert frame.dtype == np.uint8
        assert frame.shape == (48, 64, 3)
        assert frame[:, :, 0].max() == 0
        assert include_source_image is False
        assert threshold == 0.0
        return SimpleNamespace(xyxy=np.zeros((0, 4), dtype=np.float64))


def test_doctor_without_flags_does_not_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def blocked_load(*_args: object, **_kwargs: object) -> None:
        calls.append("load")
        raise AssertionError("offline doctor constructed a model")

    def blocked_step(*_args: object, **_kwargs: object) -> None:
        calls.append("step")
        raise AssertionError("offline doctor ran a training step")

    _patch_quiet_machine(monkeypatch)
    _block_network(monkeypatch)
    monkeypatch.setattr(hardware, "load_official_nano", blocked_load)
    monkeypatch.setattr(hardware, "tensor_optimizer_step", blocked_step)
    destination = tmp_path / "environment_report.json"
    report = hardware.run_doctor(destination, workspace=tmp_path)
    assert calls == []
    assert not (tmp_path / "cache").exists()
    for key in _NOT_RUN:
        assert report[key] == "not_run"
    on_disk = json.loads(destination.read_text(encoding="utf-8"))
    assert on_disk["model_load"] == "not_run"
    assert on_disk["inference"] == "not_run"
    assert on_disk["training_step"] == "not_run"


def test_doctor_json_leaves_three_probes_not_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("offline doctor constructed a model")

    _patch_quiet_machine(monkeypatch)
    _block_network(monkeypatch)
    monkeypatch.setattr(hardware, "load_official_nano", blocked)
    destination = tmp_path / "environment_report.json"
    result = _RUNNER.invoke(
        app,
        ["doctor", "--json", "--output", str(destination), "--workspace", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "Официальные веса" not in result.stdout
    assert "Официальные веса" not in result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["model_load"] == "not_run"
    assert payload["data"]["inference"] == "not_run"
    assert payload["data"]["training_step"] == "not_run"


def test_failed_load_is_failed_not_a_mock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = Path(hardware.__file__).read_text(encoding="utf-8")
    assert "FakeDetector" not in source

    def explode(self: FakeDetector, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("FakeDetector was constructed")

    def explode_open(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("open_detector was called after a failed load")

    def explode_load(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("checkpoint download failed")

    seen: list[str] = []

    def step(device: str) -> dict[str, object]:
        seen.append(device)
        return _completed_step(device)

    _patch_quiet_machine(monkeypatch)
    _block_network(monkeypatch)
    monkeypatch.setattr(FakeDetector, "__init__", explode)
    monkeypatch.setattr(
        "excavation_lab.models.baseline.open_detector",
        explode_open,
    )
    monkeypatch.setattr(hardware, "load_official_nano", explode_load)
    monkeypatch.setattr(hardware, "tensor_optimizer_step", step)
    destination = tmp_path / "environment_report.json"
    report = hardware.run_doctor(
        destination,
        load_weights=True,
        allow_cpu_fallback=False,
        workspace=tmp_path,
    )
    assert seen == []
    load = report["model_load"]
    assert isinstance(load, dict)
    assert load["status"] == "failed"
    assert load["loaded"] is False
    assert load["returned_boxes"] is None
    assert load["peak_memory_bytes"] is None
    assert "checkpoint download failed" in str(load["exception"])
    assert load["class_name"] == "RFDETRNano"
    assert str(load["weights_path"]).endswith("cache/weights/rf-detr-nano.pth")
    assert report["inference"] == "not_run"
    step = report["training_step"]
    assert isinstance(step, dict)
    assert step["device"] is None
    assert step["status"] == "failed"
    cpu = next(item for item in step["attempts"] if item["device"] == "cpu")
    assert cpu["status"] == "not_run"
    assert "FakeDetector" not in json.dumps(report)
    assert not (tmp_path / "cache" / "weights" / "rf-detr-nano.pth").exists()
    text = destination.read_text(encoding="utf-8")
    assert '"status": "failed"' in text
    assert '"peak_memory_bytes": null' in text
    assert '"peak_memory_bytes": 0' not in text


def test_load_weights_prints_russian_notice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def explode_load(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("checkpoint download failed")

    _patch_quiet_machine(monkeypatch)
    _block_network(monkeypatch)
    monkeypatch.setattr(hardware, "load_official_nano", explode_load)
    monkeypatch.setattr(hardware, "tensor_optimizer_step", _completed_step)
    destination = tmp_path / "environment_report.json"
    result = _RUNNER.invoke(
        app,
        [
            "doctor",
            "--load-weights",
            "--json",
            "--output",
            str(destination),
            "--workspace",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert "Официальные веса" in result.stderr
    assert "локальный кэш" in result.stderr
    assert _cyrillic(result.stderr)
    assert "cache/weights" in result.stderr
    payload = json.loads(result.stdout)
    assert payload["data"]["model_load"]["status"] == "failed"
    assert "Traceback" not in result.stdout


def test_null_measurements_stay_null_on_a_patched_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_quiet_machine(monkeypatch)
    _block_network(monkeypatch)
    monkeypatch.setattr(hardware, "load_official_nano", lambda *_a, **_k: _EmptyPredict())
    monkeypatch.setattr(hardware, "tensor_optimizer_step", _completed_step)
    destination = tmp_path / "environment_report.json"
    report = hardware.run_doctor(
        destination,
        load_weights=True,
        allow_cpu_fallback=True,
        workspace=tmp_path,
    )
    load = report["model_load"]
    inference = report["inference"]
    step = report["training_step"]
    assert isinstance(load, dict)
    assert isinstance(inference, dict)
    assert isinstance(step, dict)
    assert load["status"] == "passed"
    assert load["returned_boxes"] is None
    assert load["peak_memory_bytes"] is None
    assert inference["status"] == "passed"
    assert inference["returned_boxes"] is False
    assert inference["frame"] == "synthetic_rgb"
    assert inference["peak_memory_bytes"] is None
    assert step["status"] == "passed"
    assert step["method"] == "tensor_optimizer_step"
    assert step["device"] == "cpu"
    assert step["peak_memory_bytes"] is None
    assert step["latency_s"] is None
    on_disk = json.loads(destination.read_text(encoding="utf-8"))
    assert on_disk["model_load"]["peak_memory_bytes"] is None
    assert on_disk["model_load"]["returned_boxes"] is None
    assert on_disk["inference"]["peak_memory_bytes"] is None
    assert on_disk["training_step"]["peak_memory_bytes"] is None
    assert on_disk["training_step"]["latency_s"] is None
    assert on_disk["inference"]["returned_boxes"] is False
    assert not (tmp_path / "cache" / "weights" / "rf-detr-nano.pth").exists()


def test_loader_imports_rfdetr_nano_only_inside_the_function() -> None:
    source = Path(hardware.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        for name in modules:
            assert name.split(".", 1)[0] != "rfdetr"
    loaders = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_official_nano"
    ]
    assert len(loaders) == 1
    imported: list[str] = []
    for node in ast.walk(loaders[0]):
        if isinstance(node, ast.ImportFrom) and node.module == "rfdetr":
            imported.extend(alias.name for alias in node.names)
    assert imported == ["RFDETRNano"]
    assert "pretrain_weights" in ast.get_source_segment(source, loaders[0])


def test_weights_path_stays_under_workspace_cache(tmp_path: Path) -> None:
    path = hardware.official_nano_weights_path(tmp_path)
    assert path == tmp_path.resolve() / "cache" / "weights" / "rf-detr-nano.pth"
    assert not path.exists()
    notice = hardware.weights_download_notice(path.parent)
    assert "Официальные веса" in notice
    assert str(path.parent) in notice


def test_select_training_device_order_and_cpu_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    available = {"cuda": True, "mps": True, "cpu": True}
    outcomes = {"cuda": True, "mps": True, "cpu": True}

    monkeypatch.setattr(hardware, "probe_cuda", lambda: available["cuda"])
    monkeypatch.setattr(hardware, "probe_mps", lambda: available["mps"])
    monkeypatch.setattr(hardware, "probe_cpu", lambda: available["cpu"])

    def step(device: str) -> dict[str, object]:
        seen.append(device)
        if not outcomes[device]:
            raise RuntimeError(f"{device} step failed")
        return _completed_step(device)

    monkeypatch.setattr(hardware, "tensor_optimizer_step", step)

    chosen = hardware.select_training_device(allow_cpu_fallback=False)
    assert chosen["device"] == "cuda"
    assert chosen["status"] == "passed"
    assert chosen["method"] == "tensor_optimizer_step"
    assert seen == ["cuda"]
    assert chosen["peak_memory_bytes"] is None
    assert chosen["latency_s"] is None

    outcomes["cuda"] = False
    seen.clear()
    chosen = hardware.select_training_device(allow_cpu_fallback=True)
    assert seen == ["cuda", "mps"]
    assert chosen["device"] == "mps"
    assert chosen["status"] == "passed"

    outcomes["mps"] = False
    seen.clear()
    chosen = hardware.select_training_device(allow_cpu_fallback=False)
    assert seen == ["cuda", "mps"]
    assert chosen["device"] is None
    assert chosen["status"] == "failed"
    assert chosen["allow_cpu_fallback"] is False
    mps = next(item for item in chosen["attempts"] if item["device"] == "mps")
    assert mps["status"] == "failed"
    assert mps["exception"] == "RuntimeError: mps step failed"
    assert mps["config"]["device"] == "mps"
    assert mps["config"]["method"] == "tensor_optimizer_step"
    assert mps["config"]["batch_size"] == 1
    assert mps["peak_memory_bytes"] is None
    cpu = next(item for item in chosen["attempts"] if item["device"] == "cpu")
    assert cpu["status"] == "not_run"
    assert cpu["config"] is None
    assert cpu["reason"] == "cpu_fallback_not_requested"
    assert chosen["suggestion"] is not None
    assert _cyrillic(str(chosen["suggestion"]))
    assert "allow-cpu-fallback" in str(chosen["suggestion"])

    seen.clear()
    chosen = hardware.select_training_device(allow_cpu_fallback=True)
    assert seen == ["cuda", "mps", "cpu"]
    assert chosen["device"] == "cpu"
    assert chosen["status"] == "passed"
    assert chosen["allow_cpu_fallback"] is True
    mps = next(item for item in chosen["attempts"] if item["device"] == "mps")
    assert "mps step failed" in str(mps["exception"])
    assert mps["config"]["tensor_shape"] == [1]

    available["cuda"] = False
    outcomes["mps"] = True
    seen.clear()
    chosen = hardware.select_training_device(allow_cpu_fallback=False)
    assert seen == ["mps"]
    assert chosen["device"] == "mps"
    skipped = next(item for item in chosen["attempts"] if item["device"] == "cuda")
    assert skipped["status"] == "unavailable"


def test_incomplete_step_is_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware, "probe_cuda", lambda: True)
    monkeypatch.setattr(hardware, "probe_mps", lambda: False)
    monkeypatch.setattr(hardware, "probe_cpu", lambda: True)

    def step(_device: str) -> dict[str, object]:
        return {"completed": False, "peak_memory_bytes": 0, "latency_s": 0}

    monkeypatch.setattr(hardware, "tensor_optimizer_step", step)
    chosen = hardware.select_training_device(allow_cpu_fallback=False)
    assert chosen["device"] is None
    assert chosen["status"] == "failed"
    assert chosen["peak_memory_bytes"] is None
    assert chosen["latency_s"] is None
    cuda = next(item for item in chosen["attempts"] if item["device"] == "cuda")
    assert cuda["status"] == "failed"
    assert "did not complete" in str(cuda["exception"])
    cpu = next(item for item in chosen["attempts"] if item["device"] == "cpu")
    assert cpu["status"] == "not_run"


def test_cpu_tensor_step_runs_and_leaves_peak_null() -> None:
    measured = hardware.tensor_optimizer_step("cpu")
    assert measured["completed"] is True
    assert measured["method"] == "tensor_optimizer_step"
    assert measured["peak_memory_bytes"] is None
    assert isinstance(measured["latency_s"], float)
    text = json.dumps(measured)
    assert '"peak_memory_bytes": null' in text
    assert '"peak_memory_bytes": 0' not in text


@pytest.mark.manual
def test_second_load_weights_uses_local_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Download official Nano weights once into ``workspace/cache/weights``.

    Not collected by default (``addopts = -m 'not manual'``). Do not run this
    from the offline suite.
    """
    from rfdetr.assets import model_weights

    downloads: list[str] = []
    real_download = model_weights._download_file

    def counted(
        url: str,
        filename: str,
        expected_md5: str | None = None,
        timeout: float = 600,
    ) -> None:
        downloads.append(filename)
        real_download(url, filename, expected_md5, timeout=timeout)

    monkeypatch.setattr(model_weights, "_download_file", counted)
    workspace = tmp_path / "workspace"
    destination = workspace / "environment_report.json"
    hardware.run_doctor(
        destination,
        load_weights=True,
        allow_cpu_fallback=True,
        workspace=workspace,
    )
    assert downloads
    first = len(downloads)
    cached = hardware.official_nano_weights_path(workspace)
    assert cached.is_file()
    hardware.run_doctor(
        destination,
        load_weights=True,
        allow_cpu_fallback=True,
        workspace=workspace,
    )
    assert len(downloads) == first
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["model_load"]["status"] == "passed"
    assert report["model_load"]["class_name"] == "RFDETRNano"
    assert report["inference"]["status"] == "passed"
    assert isinstance(report["inference"]["returned_boxes"], bool)
