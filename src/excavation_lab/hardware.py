"""Machine report, inference device, and optional doctor runtime probes.

Device order is CUDA, then MPS, then CPU. Without ``load_weights``, the model
load, the inference probe, and the training step stay ``not_run`` and this
module does not construct ``RFDETRNano`` or open a socket.

``RFDETR.train`` is a full Lightning fit, not a public micro-step.
``tensor_optimizer_step`` is the probe ``training_step.method`` names.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import time
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, TypedDict

from excavation_lab.errors import FfmpegMissingError, LabError
from excavation_lab.schemas import SCHEMA_VERSION
from excavation_lab.storage import default_workspace_root

ENVIRONMENT_REPORT_NAME = "environment_report.json"
UNKNOWN_MEASUREMENT = "не рассчитано"
NOT_RUN = "not_run"
OFFICIAL_NANO_CLASS = "RFDETRNano"
NANO_WEIGHTS_FILENAME = "rf-detr-nano.pth"
TRAINING_STEP_METHOD = "tensor_optimizer_step"
InferenceDevice = Literal["cuda", "mps", "cpu"]
_PROBE_KEYS = ("model_load", "inference", "training_step")
_DEVICE_ORDER: tuple[InferenceDevice, ...] = ("cuda", "mps", "cpu")
_MPS_SUGGESTION = (
    "Шаг на MPS не выполнен. Исключение и конфигурация записаны. "
    "Отдельный запуск: Nano, batch 1, меньшее допустимое разрешение "
    "или явно выбранный CPU (--allow-cpu-fallback). "
    "Длинное обучение на CPU само не подставляется."
)

_VERSION_PACKAGES: tuple[tuple[str, str], ...] = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("rfdetr", "rfdetr"),
    ("trackers", "trackers"),
    ("supervision", "supervision"),
    ("av", "av"),
    ("streamlit", "streamlit"),
)


class PackageVersions(TypedDict):
    torch: str | None
    torchvision: str | None
    rfdetr: str | None
    trackers: str | None
    supervision: str | None
    av: str | None
    streamlit: str | None


class EnvironmentReport(TypedDict):
    schema_version: int
    report_path: str
    os: str
    python: str
    cpu_architecture: str
    ram_bytes: int | None
    ram_message: str | None
    versions: PackageVersions
    cuda_available: bool
    mps_built: bool
    mps_available: bool
    ffmpeg_available: bool
    ffprobe_available: bool
    inference_device: InferenceDevice
    model_load: str | dict[str, Any]
    inference: str | dict[str, Any]
    training_step: str | dict[str, Any]


def default_report_path() -> Path:
    """Gitignored ``workspace/environment_report.json``."""
    return default_workspace_root() / ENVIRONMENT_REPORT_NAME


def read_ram_bytes() -> int | None:
    """Physical RAM from ``os.sysconf`` on Darwin and Linux.

    Unknown, missing, or non-positive counters stay ``None``. No guessed size.
    """
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None
    try:
        page_size = sysconf("SC_PAGE_SIZE")
        phys_pages = sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None
    if isinstance(page_size, bool) or isinstance(phys_pages, bool):
        return None
    if not isinstance(page_size, int) or not isinstance(phys_pages, int):
        return None
    if page_size <= 0 or phys_pages <= 0:
        return None
    return page_size * phys_pages


def package_versions() -> PackageVersions:
    """Installed distribution versions. Does not import the packages."""
    found: dict[str, str | None] = {}
    for key, distribution in _VERSION_PACKAGES:
        try:
            found[key] = version(distribution)
        except PackageNotFoundError:
            found[key] = None
    return PackageVersions(
        torch=found["torch"],
        torchvision=found["torchvision"],
        rfdetr=found["rfdetr"],
        trackers=found["trackers"],
        supervision=found["supervision"],
        av=found["av"],
        streamlit=found["streamlit"],
    )


def probe_cuda() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def probe_mps() -> bool:
    """MPS is usable. Built-but-unavailable does not count."""
    import torch

    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    return bool(backend.is_available())


def probe_cpu() -> bool:
    return True


def mps_built() -> bool:
    import torch

    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    return bool(backend.is_built())


def ffmpeg_available() -> bool:
    return _command_on_path("ffmpeg")


def ffprobe_available() -> bool:
    return _command_on_path("ffprobe")


def select_inference_device() -> InferenceDevice:
    """CUDA, then MPS, then CPU. Does not load weights or start a job."""
    cuda = probe_cuda()
    mps = False if cuda else probe_mps()
    cpu = False if cuda or mps else probe_cpu()
    return choose_device(cuda=cuda, mps=mps, cpu=cpu)


def choose_device(*, cuda: bool, mps: bool, cpu: bool) -> InferenceDevice:
    if cuda:
        return "cuda"
    if mps:
        return "mps"
    if cpu:
        return "cpu"
    raise LabError(
        "device_unavailable",
        "Нет доступного устройства вывода: CUDA, MPS и CPU недоступны.",
    )


def build_environment_report(path: Path) -> EnvironmentReport:
    destination = Path(path).resolve()
    ram = read_ram_bytes()
    cuda = probe_cuda()
    mps = probe_mps()
    cpu = probe_cpu()
    return EnvironmentReport(
        schema_version=SCHEMA_VERSION,
        report_path=str(destination),
        os=platform.system(),
        python=platform.python_version(),
        cpu_architecture=platform.machine(),
        ram_bytes=ram,
        ram_message=None if ram is not None else UNKNOWN_MEASUREMENT,
        versions=package_versions(),
        cuda_available=cuda,
        mps_built=mps_built(),
        mps_available=mps,
        ffmpeg_available=ffmpeg_available(),
        ffprobe_available=ffprobe_available(),
        inference_device=choose_device(cuda=cuda, mps=mps, cpu=cpu),
        model_load=NOT_RUN,
        inference=NOT_RUN,
        training_step=NOT_RUN,
    )


def write_environment_report(path: Path) -> EnvironmentReport:
    report = build_environment_report(path)
    _write_json(Path(report["report_path"]), report)
    return report


def weights_cache_dir(workspace: Path | None = None) -> Path:
    """``<workspace>/cache/weights``. Does not create the directory."""
    root = default_workspace_root() if workspace is None else Path(workspace)
    return root.resolve() / "cache" / "weights"


def official_nano_weights_path(workspace: Path | None = None) -> Path:
    """Official Nano file inside the workspace cache. Does not download it."""
    return weights_cache_dir(workspace) / NANO_WEIGHTS_FILENAME


def weights_download_notice(cache_dir: Path) -> str:
    """Russian notice printed before ``--load-weights`` may touch the network."""
    return (
        "Официальные веса RF-DETR могут быть скачаны один раз в "
        f"{cache_dir}. Повторный запуск использует этот локальный кэш."
    )


def run_doctor(
    path: Path | None = None,
    *,
    load_weights: bool = False,
    allow_cpu_fallback: bool = False,
    workspace: Path | None = None,
) -> EnvironmentReport:
    """Write the report, then fail if ``ffprobe`` or ``ffmpeg`` is missing.

    ``load_weights`` is the only switch that constructs ``RFDETRNano``.
    The default call leaves the three probes ``not_run``.
    """
    destination = path if path is not None else default_report_path()
    report = build_environment_report(destination)
    if load_weights:
        _apply_runtime_probes(
            report,
            workspace=workspace,
            allow_cpu_fallback=allow_cpu_fallback,
        )
    _write_json(Path(report["report_path"]), report)
    if not report["ffmpeg_available"] or not report["ffprobe_available"]:
        raise FfmpegMissingError()
    return report


def russian_summary(report: Mapping[str, object]) -> str:
    ram = report["ram_bytes"]
    ram_text = UNKNOWN_MEASUREMENT if ram is None else f"{ram} байт"
    cuda = "да" if report["cuda_available"] else "нет"
    mps = "да" if report["mps_available"] else "нет"
    tools_ok = bool(report["ffmpeg_available"]) and bool(report["ffprobe_available"])
    tools = "найдены" if tools_ok else "не найдены"
    return "\n".join(
        (
            f"Отчёт: {report['report_path']}",
            f"Устройство: {report['inference_device']}. CUDA: {cuda}. MPS: {mps}.",
            f"Память: {ram_text}. ffprobe и ffmpeg: {tools}.",
            _probe_summary(report),
        )
    )


def load_official_nano(weights_path: Path, device: InferenceDevice) -> Any:
    """Construct ``RFDETRNano`` with weights under the workspace cache.

    An absolute ``pretrain_weights`` path is used as-is. The library downloads
    that file when it is missing and skips the network when the MD5 matches.
    A bare filename would land in ``~/.roboflow/models`` instead.
    """
    from rfdetr import RFDETRNano

    path = Path(weights_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return RFDETRNano(pretrain_weights=str(path), device=device)


def synthetic_rgb_frame() -> Any:
    """One in-memory RGB frame. Not a pilot video and not accuracy evidence.

    Size matches the synthetic fixture writer: 64×48. Nothing is read from disk.
    """
    import numpy as np

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, :, 1] = 16
    return frame


def tensor_optimizer_step(device: InferenceDevice) -> dict[str, Any]:
    """One forward, backward, and SGD step on a single parameter.

    ``RFDETR.train`` starts a Lightning fit. ``RFDETRModelModule.training_step``
    is a hook that needs a trainer and a dataset batch. This function is the
    public micro-step those APIs do not provide. ``completed`` is set only
    after the parameter value changes.
    """
    import torch

    torch_device = torch.device(device)
    _reset_peak_memory(device)
    start = time.perf_counter()
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float32, device=torch_device))
    optimizer = torch.optim.SGD([parameter], lr=0.5)
    optimizer.zero_grad(set_to_none=True)
    loss = (parameter * parameter).sum()
    loss.backward()
    if parameter.grad is None:
        raise RuntimeError("backward did not produce a gradient")
    before = parameter.detach().to("cpu").clone()
    optimizer.step()
    _synchronize(device)
    latency_s = _measured_latency_s(start)
    after = parameter.detach().to("cpu")
    if torch.equal(before, after):
        raise RuntimeError("optimizer step did not change the parameter")
    return {
        "completed": True,
        "method": TRAINING_STEP_METHOD,
        "peak_memory_bytes": _read_peak_memory_bytes(device),
        "latency_s": latency_s,
    }


def select_training_device(*, allow_cpu_fallback: bool = False) -> dict[str, Any]:
    """First of CUDA, MPS, CPU that finishes one optimizer step.

    CPU is attempted only when ``allow_cpu_fallback`` is true. A failed MPS
    step keeps its exception and config. This does not switch a long train
    onto CPU by itself.
    """
    attempts: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    for device in _DEVICE_ORDER:
        if device == "cpu" and not allow_cpu_fallback:
            attempts.append(_cpu_not_run_attempt())
            break
        if not _training_backend_available(device):
            attempts.append(_unavailable_attempt(device))
            continue
        config = _step_config(device)
        try:
            measured = tensor_optimizer_step(device)
            _require_completed_step(measured)
        except Exception as exc:  # noqa: BLE001 — any backend failure is the probe result
            attempts.append(_failed_attempt(device, config, exc))
            continue
        chosen = _passed_attempt(device, config, measured)
        attempts.append(chosen)
        break
    return _training_selection(
        chosen,
        attempts,
        allow_cpu_fallback=allow_cpu_fallback,
    )


def _apply_runtime_probes(
    report: EnvironmentReport,
    *,
    workspace: Path | None,
    allow_cpu_fallback: bool,
) -> None:
    device = report["inference_device"]
    weights_path = official_nano_weights_path(workspace)
    loaded, module = _probe_model_load(weights_path, device)
    report["model_load"] = loaded
    if module is None:
        report["inference"] = NOT_RUN
    else:
        report["inference"] = _probe_inference(module, device)
    report["training_step"] = select_training_device(allow_cpu_fallback=allow_cpu_fallback)


def _probe_model_load(
    weights_path: Path,
    device: InferenceDevice,
) -> tuple[dict[str, Any], Any | None]:
    record: dict[str, Any] = {
        "status": "failed",
        "device": device,
        "class_name": OFFICIAL_NANO_CLASS,
        "weights_path": str(weights_path),
        "returned_boxes": None,
        "loaded": False,
        "peak_memory_bytes": None,
        "latency_s": None,
        "exception": None,
    }
    _reset_peak_memory(device)
    start = time.perf_counter()
    try:
        module = load_official_nano(weights_path, device)
    except Exception as exc:  # noqa: BLE001 — a failed load is recorded, not replaced
        record["exception"] = _exception_text(exc)
        record["latency_s"] = _measured_latency_s(start)
        return record, None
    _synchronize(device)
    record["status"] = "passed"
    record["loaded"] = True
    record["class_name"] = type(module).__name__
    record["latency_s"] = _measured_latency_s(start)
    record["peak_memory_bytes"] = _read_peak_memory_bytes(device)
    return record, module


def _probe_inference(module: Any, device: InferenceDevice) -> dict[str, Any]:
    from excavation_lab.models.baseline import open_detector

    record: dict[str, Any] = {
        "status": "failed",
        "device": device,
        "returned_boxes": None,
        "peak_memory_bytes": None,
        "latency_s": None,
        "exception": None,
        "frame": "synthetic_rgb",
    }
    try:
        detector = open_detector("baseline_coco", model=module)
    except Exception as exc:  # noqa: BLE001 — inference setup failure stays in the report
        record["exception"] = _exception_text(exc)
        return record
    start = time.perf_counter()
    try:
        frame = synthetic_rgb_frame()
        _reset_peak_memory(device)
        detections = detector.predict(frame, 0.0)
        _synchronize(device)
        record["latency_s"] = _measured_latency_s(start)
        record["peak_memory_bytes"] = _read_peak_memory_bytes(device)
        record["returned_boxes"] = int(detections.xyxy.shape[0]) > 0
        record["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 — a failed frame stays failed, not a mock
        record["exception"] = _exception_text(exc)
        if record["latency_s"] is None:
            record["latency_s"] = _measured_latency_s(start)
    finally:
        detector.close()
    return record


def _probe_summary(report: Mapping[str, object]) -> str:
    if all(report[key] == NOT_RUN for key in _PROBE_KEYS):
        return "Загрузка модели, проверка вывода и шаг обучения: не запускались."
    load = _status_word(report["model_load"])
    inference = _status_word(report["inference"])
    step = _status_word(report["training_step"])
    return (
        f"Загрузка модели: {load}. Проверка вывода: {inference}. "
        f"Шаг обучения: {step}."
    )


def _status_word(value: object) -> str:
    if value == NOT_RUN:
        return "не запускалась"
    if isinstance(value, Mapping):
        status = value.get("status")
        if status == "passed":
            return "пройдена"
        if status == "failed":
            return "ошибка"
    return UNKNOWN_MEASUREMENT


def _training_backend_available(device: InferenceDevice) -> bool:
    if device == "cuda":
        return probe_cuda()
    if device == "mps":
        return probe_mps()
    return probe_cpu()


def _step_config(device: InferenceDevice) -> dict[str, Any]:
    return {
        "method": TRAINING_STEP_METHOD,
        "device": device,
        "batch_size": 1,
        "dtype": "float32",
        "tensor_shape": [1],
    }


def _require_completed_step(measured: object) -> Mapping[str, Any]:
    if not isinstance(measured, Mapping) or measured.get("completed") is not True:
        raise RuntimeError("optimizer step did not complete")
    return measured


def _passed_attempt(
    device: InferenceDevice,
    config: dict[str, Any],
    measured: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "device": device,
        "status": "passed",
        "exception": None,
        "config": config,
        "peak_memory_bytes": _optional_int(measured.get("peak_memory_bytes")),
        "latency_s": _optional_float(measured.get("latency_s")),
    }


def _failed_attempt(
    device: InferenceDevice,
    config: dict[str, Any],
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "device": device,
        "status": "failed",
        "exception": _exception_text(exc),
        "config": config,
        "peak_memory_bytes": None,
        "latency_s": None,
    }


def _unavailable_attempt(device: InferenceDevice) -> dict[str, Any]:
    return {
        "device": device,
        "status": "unavailable",
        "exception": None,
        "config": None,
        "peak_memory_bytes": None,
        "latency_s": None,
    }


def _cpu_not_run_attempt() -> dict[str, Any]:
    return {
        "device": "cpu",
        "status": NOT_RUN,
        "exception": None,
        "config": None,
        "peak_memory_bytes": None,
        "latency_s": None,
        "reason": "cpu_fallback_not_requested",
    }


def _training_selection(
    chosen: dict[str, Any] | None,
    attempts: list[dict[str, Any]],
    *,
    allow_cpu_fallback: bool,
) -> dict[str, Any]:
    mps_failed = any(
        item["device"] == "mps" and item["status"] == "failed" for item in attempts
    )
    if chosen is None:
        return {
            "status": "failed",
            "method": TRAINING_STEP_METHOD,
            "device": None,
            "peak_memory_bytes": None,
            "latency_s": None,
            "exception": _overall_exception(attempts),
            "config": _failed_config(attempts),
            "attempts": attempts,
            "suggestion": _MPS_SUGGESTION if mps_failed else None,
            "allow_cpu_fallback": allow_cpu_fallback,
        }
    return {
        "status": "passed",
        "method": TRAINING_STEP_METHOD,
        "device": chosen["device"],
        "peak_memory_bytes": chosen["peak_memory_bytes"],
        "latency_s": chosen["latency_s"],
        "exception": None,
        "config": chosen["config"],
        "attempts": attempts,
        "suggestion": _MPS_SUGGESTION if mps_failed else None,
        "allow_cpu_fallback": allow_cpu_fallback,
    }


def _failed_config(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    for device in ("mps", "cuda", "cpu"):
        for item in attempts:
            if item["device"] == device and item["status"] == "failed":
                config = item["config"]
                if isinstance(config, dict):
                    return config
    return None


def _overall_exception(attempts: list[dict[str, Any]]) -> str:
    for device in ("mps", "cuda", "cpu"):
        for item in attempts:
            if item["device"] == device and item["status"] == "failed":
                text = item["exception"]
                if isinstance(text, str) and text:
                    return text
    return "Ни один backend не завершил шаг обучения."


def _exception_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".strip()


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _measured_latency_s(start: float) -> float | None:
    elapsed = time.perf_counter() - start
    if not math.isfinite(elapsed) or elapsed < 0:
        return None
    return elapsed


def _reset_peak_memory(device: InferenceDevice) -> None:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _synchronize(device: InferenceDevice) -> None:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _read_peak_memory_bytes(device: InferenceDevice) -> int | None:
    """CUDA peak allocation. MPS and CPU do not expose a peak counter.

    ``torch.mps.current_allocated_memory`` and ``driver_allocated_memory`` are
    not a peak for this step, so they are not stored as one. Unknown stays
    ``None``.
    """
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        return None
    value = torch.cuda.max_memory_allocated()
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _command_on_path(name: str) -> bool:
    return shutil.which(name) is not None


def _write_json(path: Path, report: EnvironmentReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
