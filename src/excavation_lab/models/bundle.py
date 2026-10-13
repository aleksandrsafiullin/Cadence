"""Write a fine-tune directory and check its checksums.

``weights.pth`` is the inference file. ``training_checkpoint.ckpt`` is copied
only when the library wrote a ``.ckpt``. A missing resume file is allowed and
is not a successful resume point. Official COCO weights are not copied into
this directory and named as a custom detector.

``class_names`` is the class order. Sorting JSON object keys does not define
that order. An overfit run keeps ``metric_role`` ``diagnostic_overfit`` and
is not written as a test-quality card.

``load_bundle`` reads ``weights.pth`` through the library safe loader and
does not open ``training_checkpoint.ckpt``. A metadata-only state dict is
not an RF-DETR graph, so the Nano or Small module is built on ``predict``.
Score thresholds are not changed here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from excavation_lab.datasets.coco_import import build_class_mapping
from excavation_lab.datasets.splits import SAME_SCENE_MESSAGE
from excavation_lab.errors import (
    CheckpointMissingError,
    IncompatibleCheckpointError,
    LabError,
    map_os_error,
)
from excavation_lab.hardware import (
    build_environment_report,
    official_nano_weights_path,
    package_versions,
    weights_cache_dir,
)
from excavation_lab.models.adapter import (
    CheckpointOrigin,
    DetectorAdapter,
    ModelMetadata,
)
from excavation_lab.models.baseline import LIBRARY_SCORE_THRESHOLD, _frame_from_raw
from excavation_lab.models.preprocess import model_rgb_frame
from excavation_lab.models.train_config import (
    METRIC_ROLE_DIAGNOSTIC_OVERFIT,
    ResolvedTrainConfig,
)
from excavation_lab.schemas import SCHEMA_VERSION
from excavation_lab.storage import connect, project_directory, project_sqlite

NOT_CALCULATED = "не рассчитано"
CHECKSUMS_NAME = "checksums.json"
RESUME_NAME = "training_checkpoint.ckpt"
WEIGHTS_NAME = "weights.pth"
EMPTY_HISTORY_HEADER = "epoch"
OTHER_CAMERA_ABSENT = "Другая камера не проверялась."
RESUME_ABSENT = "Отсутствие файла не является успешной точкой возобновления."
CLASS_NAMES: tuple[str, ...] = (
    "excavator_body",
    "bucket",
    "haul_truck",
    "truck_bed",
)
BUNDLE_FILENAMES: tuple[str, ...] = (
    WEIGHTS_NAME,
    "model_config.json",
    "class_mapping.json",
    "resolved_train_config.json",
    "dataset_manifest.json",
    "environment_report.json",
    "metrics_history.csv",
    "validation_metrics.json",
    "model_card.md",
)
_HEX = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SPLIT_INDEPENDENCE = frozenset({"episode_split", "same_scene"})
_TEST_SPLITS = frozenset({"test", "testing"})
_QUALITY_CLAIM = (
    "Диагностика overfit не записывается как качество модели."
)
_DIAGNOSTIC_USE = (
    "Диагностика overfit: проверка, что данные и классы позволяют модели "
    "подогнаться под небольшой набор train-кадров. "
    "Это не карточка качества модели и не оценка на отложенном test."
)
_QUALITY_USE = (
    "Дообученный детектор четырёх классов на видео экскавации: "
    "корпус экскаватора, ковш, самосвал и кузов. "
    "Карточка описывает этот bundle. "
    "Она не подтверждает разгрузку и не заменяет отдельную оценку на test."
)
_DIAGNOSTIC_ROLE = (
    "Роль метрик: diagnostic_overfit. "
    "Оценка качества на test в эту карточку не входит."
)


class BundleChecksumError(LabError):
    """A bundle file does not match ``checksums.json``."""

    def __init__(self, message: str) -> None:
        super().__init__("checksum_mismatch", message)


@dataclass(frozen=True, slots=True)
class ModelBundleSpec:
    """Inputs for one bundle directory. Weights are the fine-tune bytes."""

    variant: str
    resolution: int
    dataset_hash: str
    library_versions: Mapping[str, str | None]
    resolved_train_config: Mapping[str, Any]
    dataset_manifest: Mapping[str, Any]
    environment_report: Mapping[str, Any]
    validation_metrics: Mapping[str, Any]
    weights: bytes
    class_mapping: Mapping[str, Any] | None = None
    metrics_log: bytes | None = None
    training_checkpoint: bytes | None = None
    view_id: str | None = None
    equipment_family: str | None = None
    split_independence: str | None = None
    excluded_conditions: str | None = None
    other_camera_tested: bool = False


def write_model_bundle(directory: Path, spec: ModelBundleSpec) -> Path:
    """Write ``directory`` and ``checksums.json``. Does not load the weights."""
    prepared = _prepare(spec)
    final = Path(directory)
    if final.exists() or final.is_symlink():
        raise LabError(
            "model_exists",
            f"Каталог модели уже есть: {final}. Bundle повторно не записывается.",
        )
    parent = final.parent
    partial = parent / f".{final.name}.partial"
    with map_os_error(str(final)):
        parent.mkdir(parents=True, exist_ok=True)
        if partial.exists() or partial.is_symlink():
            shutil.rmtree(partial)
        partial.mkdir()
        try:
            _write_bytes(partial / WEIGHTS_NAME, prepared["weights"])
            resume = prepared["training_checkpoint"]
            if isinstance(resume, bytes):
                _write_bytes(partial / RESUME_NAME, resume)
            _write_json(partial / "model_config.json", prepared["model_config"])
            _write_json(partial / "class_mapping.json", prepared["class_mapping"])
            _write_json(
                partial / "resolved_train_config.json",
                prepared["resolved_train_config"],
            )
            _write_json(partial / "dataset_manifest.json", prepared["dataset_manifest"])
            _write_json(
                partial / "environment_report.json",
                prepared["environment_report"],
            )
            _write_text(partial / "metrics_history.csv", prepared["metrics_history"])
            _write_json(
                partial / "validation_metrics.json",
                prepared["validation_metrics"],
            )
            _write_text(partial / "model_card.md", prepared["model_card"])
            _write_json(partial / CHECKSUMS_NAME, _checksum_document(partial))
            verify_bundle_checksums(partial)
            os.replace(partial, final)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
    return final


def verify_bundle_checksums(directory: Path) -> None:
    """Recompute SHA-256 for every file except ``checksums.json``.

    A changed byte, a missing required file, or a file the checksum list
    does not name fails. The resume flag must match whether
    ``training_checkpoint.ckpt`` is in the directory.
    """
    root = Path(directory)
    checksum_path = root / CHECKSUMS_NAME
    if not checksum_path.is_file() or checksum_path.is_symlink():
        raise BundleChecksumError(
            "В bundle нет checksums.json. Соберите каталог модели заново.",
        )
    try:
        document = json.loads(checksum_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleChecksumError(
            "checksums.json повреждён. Соберите каталог модели заново.",
        ) from exc
    if not isinstance(document, dict) or document.get("algorithm") != "sha256":
        raise BundleChecksumError(
            "checksums.json должен описывать алгоритм sha256.",
        )
    listed = document.get("files")
    if not isinstance(listed, dict):
        raise BundleChecksumError(
            "В checksums.json нет объекта files. Соберите каталог модели заново.",
        )
    actual = _regular_files(root)
    missing = [name for name in BUNDLE_FILENAMES if name not in actual]
    if missing:
        names = ", ".join(missing)
        raise BundleChecksumError(
            f"В bundle нет обязательных файлов: {names}. Соберите каталог заново.",
        )
    listed_names = set(listed)
    actual_names = set(actual)
    if listed_names != actual_names:
        extra = sorted(actual_names - listed_names)
        absent = sorted(listed_names - actual_names)
        detail = ", ".join([*extra, *absent])
        raise BundleChecksumError(
            "checksums.json не покрывает файлы каталога "
            f"({detail}). Соберите каталог модели заново.",
        )
    for name, path in actual.items():
        expected = listed.get(name)
        digest = _sha256(path)
        if not isinstance(expected, str) or expected != digest:
            raise BundleChecksumError(
                f"Контрольная сумма не сходится для «{name}». "
                "Файл изменён после записи bundle. Соберите каталог заново.",
            )
    _check_resume_flag(root)


class CustomDetector:
    """``custom`` bundle. Class indexes stay the bundle order.

    ``weights.pth`` is checked before this object exists. The RF-DETR module
    is built on the first ``predict``: constructing Nano or Small needs a real
    state dict, and a metadata probe is not one. ``close`` drops that module.
    There is no process-wide cache and no COCO stand-in.
    """

    def __init__(
        self,
        *,
        weights_path: Path,
        class_names: tuple[str, ...],
        variant: str,
        resolution: int,
        model: Any | None = None,
    ) -> None:
        self._weights_path = weights_path
        self._class_names = class_names
        self._variant = variant
        self._resolution = resolution
        self._model: Any | None = model
        self._closed = False
        self.boundary_frames: list[NDArray[np.uint8]] = []

    def predict(
        self,
        frame_rgb: NDArray[np.uint8],
        timestamp_s: float,
    ) -> Any:
        model = self._ensure_model()
        frame = model_rgb_frame(frame_rgb)
        self.boundary_frames = [frame]
        raw = model.predict(
            frame,
            threshold=LIBRARY_SCORE_THRESHOLD,
            include_source_image=False,
        )
        return _frame_from_raw(
            raw,
            frame=frame,
            timestamp_s=timestamp_s,
            origin=CheckpointOrigin.custom,
            class_names=self._class_names,
        )

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            origin=CheckpointOrigin.custom,
            class_names=self._class_names,
            boxes_are_original=True,
            variant=self._variant,
            resolution=self._resolution,
        )

    def close(self) -> None:
        self._model = None
        self._closed = True

    def _ensure_model(self) -> Any:
        if self._closed:
            raise LabError(
                "detector_closed",
                "Детектор закрыт. Откройте модель заново; общий экземпляр не хранится.",
            )
        if self._model is None:
            loaded = _load_custom_module(
                self._weights_path,
                variant=self._variant,
                resolution=self._resolution,
                class_count=len(self._class_names),
            )
            _require_loaded_names(loaded, self._class_names)
            self._model = loaded
        return self._model


def load_bundle(path: str | Path, *, model: Any | None = None) -> DetectorAdapter:
    """Open ``weights.pth`` from a custom bundle directory.

    Does not read ``training_checkpoint.ckpt``. Does not construct the COCO
    baseline and does not download weights. An injected ``model`` is only for
    tests and does not skip the file or class-order checks. Thresholds on the
    checkpoint are left as stored; this function does not write them.
    """
    checked = _check_bundle(Path(path))
    return CustomDetector(
        weights_path=checked.weights,
        class_names=checked.class_names,
        variant=checked.variant,
        resolution=checked.resolution,
        model=model,
    )


def publish_train_bundle(
    workspace: Path,
    project_id: str,
    job_id: str,
    *,
    dataset_id: str,
    dataset_dir: Path,
    run_dir: Path,
    profile: str,
    resolved: ResolvedTrainConfig,
) -> Path:
    """Copy a finished run into ``models/<profile>-<job_id>/``.

    Does not download weights and does not read the official COCO file as
    the custom detector. Does not reload the checkpoint.
    """
    version = _model_version(profile, job_id)
    destination = project_directory(workspace, project_id) / "models" / version
    weights_path = run_dir / WEIGHTS_NAME
    _reject_official_weights(weights_path, workspace)
    if weights_path.is_symlink() or not weights_path.is_file():
        raise LabError(
            "empty_checkpoint",
            "weights.pth нет в каталоге задания. Bundle не записывается.",
        )
    weights = weights_path.read_bytes()
    if not weights:
        raise LabError(
            "empty_checkpoint",
            "weights.pth пуст. Bundle не записывается.",
        )
    dataset_hash, manifest = _dataset_manifest(
        workspace,
        project_id,
        dataset_id,
        dataset_dir,
    )
    resolved_document = _require_object(
        run_dir / "resolved_train_config.json",
        "В каталоге задания нет resolved_train_config.json.",
    )
    metrics = _require_object(
        run_dir / "validation_metrics.json",
        "В каталоге задания нет validation_metrics.json.",
    )
    report = build_environment_report(destination / "environment_report.json")
    view_id, equipment_family = _view_context(workspace, project_id)
    spec = ModelBundleSpec(
        variant=_variant(resolved),
        resolution=_resolution(resolved),
        dataset_hash=dataset_hash,
        library_versions=dict(package_versions()),
        resolved_train_config=resolved_document,
        dataset_manifest=manifest,
        environment_report=dict(report),
        validation_metrics=metrics,
        weights=weights,
        class_mapping=_optional_object(dataset_dir / "class_mapping.json"),
        metrics_log=_metrics_log(run_dir),
        training_checkpoint=_resume_bytes(run_dir),
        view_id=view_id,
        equipment_family=equipment_family,
        split_independence=_split_independence(dataset_dir / "split_manifest.json"),
        other_camera_tested=False,
    )
    written = write_model_bundle(destination, spec)
    try:
        _record_model(
            workspace,
            project_id,
            model_id=version,
            dataset_id=dataset_id,
            variant=spec.variant,
            path=written,
        )
    except Exception:
        shutil.rmtree(written, ignore_errors=True)
        raise
    return written


def _prepare(spec: ModelBundleSpec) -> dict[str, Any]:
    variant = _require_variant(spec.variant)
    resolution = _require_resolution(spec.resolution)
    dataset_hash = _require_hash(spec.dataset_hash)
    if not isinstance(spec.weights, bytes) or not spec.weights:
        raise LabError(
            "empty_checkpoint",
            "В bundle нет байтов weights.pth. Официальные веса COCO не подставляются.",
        )
    if not isinstance(spec.other_camera_tested, bool):
        raise LabError(
            "config_invalid",
            "Флаг проверки другой камеры должен быть логическим значением.",
        )
    independence = _require_independence(spec.split_independence)
    diagnostic = _is_diagnostic(spec.validation_metrics, spec.resolved_train_config)
    _reject_diagnostic_quality_claim(
        diagnostic,
        spec.validation_metrics,
        spec.resolved_train_config,
    )
    metrics = _stored_metrics(spec.validation_metrics)
    resolved = _stored_resolved(spec.resolved_train_config)
    if diagnostic:
        metrics["metric_role"] = METRIC_ROLE_DIAGNOSTIC_OVERFIT
        metrics["diagnostic"] = True
        resolved["metric_role"] = METRIC_ROLE_DIAGNOSTIC_OVERFIT
        resolved["diagnostic"] = True
        metrics.pop("quality_claim", None)
        resolved.pop("quality_claim", None)
    history, history_rows = _history_text(spec.metrics_log)
    resume = _resume_payload(spec.training_checkpoint)
    mapping = _class_mapping_document(spec.class_mapping)
    manifest = _manifest_document(spec.dataset_manifest, dataset_hash)
    versions = _library_versions(spec.library_versions)
    environment = _object_copy(spec.environment_report, "environment_report.json")
    model_config = {
        "schema_version": SCHEMA_VERSION,
        "origin": "custom",
        "variant": variant,
        "resolution": resolution,
        "class_names": list(CLASS_NAMES),
        "library_versions": versions,
        "dataset_hash": dataset_hash,
        "weights_file": WEIGHTS_NAME,
        "resume_checkpoint": resume is not None,
    }
    card = _render_card(
        diagnostic=diagnostic,
        variant=variant,
        resolution=resolution,
        view_id=_optional_text(spec.view_id),
        equipment_family=_optional_text(spec.equipment_family),
        split_independence=independence,
        excluded_conditions=_optional_text(spec.excluded_conditions),
        other_camera_tested=spec.other_camera_tested,
        box_map=_metric(metrics, "box_map_50_95"),
        bucket_ap=_metric(metrics, "bucket_ap"),
        validation_images=_count(metrics, "validation_images"),
        selection_unstable=_flag(metrics, "selection_unstable"),
        history_rows=history_rows,
        resume_checkpoint=resume is not None,
    )
    return {
        "weights": spec.weights,
        "training_checkpoint": resume,
        "model_config": model_config,
        "class_mapping": mapping,
        "resolved_train_config": resolved,
        "dataset_manifest": manifest,
        "environment_report": environment,
        "metrics_history": history,
        "validation_metrics": metrics,
        "model_card": card,
    }


def _render_card(
    *,
    diagnostic: bool,
    variant: str,
    resolution: int,
    view_id: str | None,
    equipment_family: str | None,
    split_independence: str | None,
    excluded_conditions: str | None,
    other_camera_tested: bool,
    box_map: float | None,
    bucket_ap: float | None,
    validation_images: int | None,
    selection_unstable: bool | None,
    history_rows: int,
    resume_checkpoint: bool,
) -> str:
    if diagnostic:
        intended = _DIAGNOSTIC_USE
        role = _DIAGNOSTIC_ROLE
    else:
        intended = _QUALITY_USE
        role = "Роль метрик: validation. Это не результат на test."
    if split_independence == "same_scene":
        split_line = f"Независимость разбиения: same_scene. {SAME_SCENE_MESSAGE}"
    elif split_independence == "episode_split":
        split_line = "Независимость разбиения: episode_split."
    else:
        split_line = f"Независимость разбиения: {NOT_CALCULATED}."
    if other_camera_tested:
        cameras = (
            "Другая камера проверялась отдельным прогоном. "
            "Этот файл сам по себе проверку не доказывает."
        )
    else:
        cameras = OTHER_CAMERA_ABSENT
    if resume_checkpoint:
        resume = (
            "training_checkpoint.ckpt скопирован из checkpoint библиотеки. "
            "Это не weights.pth. Загрузка этого файла здесь не проверяется."
        )
    else:
        resume = (
            "training_checkpoint.ckpt нет. "
            f"{RESUME_ABSENT}."
        )
    if selection_unstable is True:
        stable = "Выбор checkpoint по validation неустойчив."
    elif selection_unstable is False:
        stable = "Выбор checkpoint по validation не помечен как неустойчивый."
    else:
        stable = f"Устойчивость выбора checkpoint: {NOT_CALCULATED}."
    if history_rows > 0:
        history = f"в журнале есть записи ({history_rows})"
    else:
        history = NOT_CALCULATED
    excluded = excluded_conditions if excluded_conditions else NOT_CALCULATED
    lines = (
        "# Карточка модели",
        "",
        "## Область применения",
        "",
        intended,
        "",
        role,
        "",
        "## Ракурс",
        "",
        f"view_id: {_shown(view_id)}",
        "",
        "## Тип техники",
        "",
        f"equipment_family: {_shown(equipment_family)}",
        "",
        "## Разбиение",
        "",
        "Поле независимости разбиения принимает значение episode_split или same_scene.",
        split_line,
        "",
        "## Исключённые условия",
        "",
        excluded,
        "",
        "## Метрики",
        "",
        (
            "Числа скопированы из validation_metrics.json. "
            f"Пустое значение — «{NOT_CALCULATED}», не 0."
        ),
        f"- box mAP50:95: {_format_metric(box_map)}",
        f"- AP ковша: {_format_metric(bucket_ap)}",
        f"- Изображений validation: {_format_count(validation_images)}",
        f"- История метрик: {history}",
        stable,
        f"Вариант: {variant}. Разрешение: {resolution}.",
        "",
        "## Другие камеры",
        "",
        cameras,
        "",
        "## Веса и resume",
        "",
        "weights.pth — веса для инференса.",
        resume,
        "",
    )
    return "\n".join(lines)


def _is_diagnostic(metrics: Mapping[str, Any], resolved: Mapping[str, Any]) -> bool:
    if metrics.get("metric_role") == METRIC_ROLE_DIAGNOSTIC_OVERFIT:
        return True
    if metrics.get("diagnostic") is True:
        return True
    if resolved.get("metric_role") == METRIC_ROLE_DIAGNOSTIC_OVERFIT:
        return True
    return resolved.get("diagnostic") is True or resolved.get("profile") == "overfit"


def _reject_diagnostic_quality_claim(
    diagnostic: bool,
    metrics: Mapping[str, Any],
    resolved: Mapping[str, Any],
) -> None:
    if not diagnostic:
        return
    if metrics.get("quality_claim") is True or resolved.get("quality_claim") is True:
        raise LabError("quality_claim", _QUALITY_CLAIM)
    if _has_test_claim(metrics) or _has_test_claim(resolved):
        raise LabError("quality_claim", _QUALITY_CLAIM)


def _has_test_claim(node: object) -> bool:
    if isinstance(node, dict):
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            if key in {"quality_claim", "test"} or key.startswith("test/"):
                if key == "quality_claim" and value is not True:
                    continue
                return True
            if key == "split" and value in _TEST_SPLITS:
                return True
            if _has_test_claim(value):
                return True
        return False
    if isinstance(node, list):
        return any(_has_test_claim(item) for item in node)
    return False


def _stored_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    document = _strip_test_claim(_object_copy(metrics, "validation_metrics.json"))
    if not isinstance(document, dict):
        raise LabError("invalid_snapshot", "validation_metrics.json должен быть объектом.")
    document.pop("quality_claim", None)
    return document


def _stored_resolved(resolved: Mapping[str, Any]) -> dict[str, Any]:
    document = _strip_test_claim(_object_copy(resolved, "resolved_train_config.json"))
    if not isinstance(document, dict):
        raise LabError(
            "invalid_snapshot",
            "resolved_train_config.json должен быть объектом.",
        )
    document.pop("quality_claim", None)
    return document


def _strip_test_claim(node: object) -> object:
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            if key in {"test"} or key.startswith("test/"):
                continue
            if key == "split" and value in _TEST_SPLITS:
                continue
            cleaned[key] = _strip_test_claim(value)
        return cleaned
    if isinstance(node, list):
        kept: list[Any] = []
        for item in node:
            if isinstance(item, dict) and item.get("split") in _TEST_SPLITS:
                continue
            kept.append(_strip_test_claim(item))
        return kept
    return node


def _class_mapping_document(source: Mapping[str, Any] | None) -> dict[str, Any]:
    if source is None:
        document = build_class_mapping()
    else:
        document = _object_copy(source, "class_mapping.json")
    by_name = document.get("by_name")
    if not isinstance(by_name, dict):
        raise LabError("class_order", "В class_mapping.json нет by_name.")
    expected = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES)), strict=True))
    if set(by_name) != set(expected):
        raise LabError(
            "class_order",
            "class_mapping.json должен содержать ровно четыре класса: "
            "excavator_body, bucket, haul_truck, truck_bed.",
        )
    for name, class_id in expected.items():
        row = by_name.get(name)
        found = row.get("internal_id") if isinstance(row, dict) else None
        if isinstance(found, bool) or found != class_id:
            raise LabError(
                "class_order",
                f"Порядок классов нарушен: «{name}» должен иметь internal id {class_id}.",
            )
    document["schema_version"] = SCHEMA_VERSION
    document["class_names"] = list(CLASS_NAMES)
    return document


def _manifest_document(manifest: Mapping[str, Any], dataset_hash: str) -> dict[str, Any]:
    document = _object_copy(manifest, "dataset_manifest.json")
    if document.get("dataset_hash") != dataset_hash:
        raise LabError(
            "dataset_hash",
            "dataset_hash в dataset_manifest.json не совпадает с model_config.json.",
        )
    return document


def _library_versions(versions: Mapping[str, str | None]) -> dict[str, str | None]:
    if isinstance(versions, (str, bytes)) or not isinstance(versions, Mapping):
        raise LabError("library_versions", "library_versions должен быть объектом.")
    cleaned: dict[str, str | None] = {}
    for key, value in versions.items():
        if not isinstance(key, str) or not key.isascii() or not key:
            raise LabError("library_versions", "Ключ версии библиотеки должен быть английским.")
        if value is not None and not isinstance(value, str):
            raise LabError(
                "library_versions",
                f"Версия «{key}» должна быть строкой или null, не числом.",
            )
        cleaned[key] = value
    if "rfdetr" not in cleaned:
        raise LabError("library_versions", "В library_versions нет версии rfdetr.")
    return cleaned


def _history_text(metrics_log: bytes | None) -> tuple[str, int]:
    if metrics_log is None or not metrics_log.strip():
        return f"{EMPTY_HISTORY_HEADER}\n", 0
    if not isinstance(metrics_log, bytes):
        raise LabError("metrics_history", "Журнал метрик должен быть байтами CSV.")
    try:
        text = metrics_log.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LabError(
            "metrics_history",
            "metrics.csv не читается как UTF-8. История не подменяется нулями.",
        ) from exc
    if not text.endswith("\n"):
        text += "\n"
    rows = _csv_data_rows(text)
    if rows == 0 and not _header_line(text):
        return f"{EMPTY_HISTORY_HEADER}\n", 0
    return text, rows


def _header_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _csv_data_rows(text: str) -> int:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return 0
    return len(lines) - 1


def _resume_payload(payload: bytes | None) -> bytes | None:
    if payload is None:
        return None
    if not isinstance(payload, bytes) or not payload:
        raise LabError(
            "resume_checkpoint",
            "Пустой training_checkpoint.ckpt не является точкой возобновления.",
        )
    return payload


def _checksum_document(directory: Path) -> dict[str, Any]:
    files = {
        name: _sha256(path)
        for name, path in sorted(_regular_files(directory).items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "sha256",
        "files": files,
    }


def _regular_files(directory: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not directory.is_dir() or directory.is_symlink():
        raise BundleChecksumError(
            "Каталог bundle не читается. Соберите его заново.",
        )
    for path in directory.iterdir():
        if path.name == CHECKSUMS_NAME:
            if path.is_symlink() or not path.is_file():
                raise BundleChecksumError(
                    "checksums.json должен быть обычным файлом.",
                )
            continue
        if path.is_symlink():
            raise BundleChecksumError(
                f"«{path.name}» — ссылка. Bundle хранит копии файлов, не ссылки на COCO.",
            )
        if path.is_dir() or not path.is_file():
            raise BundleChecksumError(
                f"«{path.name}» не является файлом bundle. Соберите каталог заново.",
            )
        found[path.name] = path
    return found


def _check_resume_flag(directory: Path) -> None:
    config_path = directory / "model_config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise BundleChecksumError("В bundle нет model_config.json.")
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleChecksumError("model_config.json повреждён.") from exc
    if not isinstance(document, dict):
        raise BundleChecksumError("model_config.json должен быть объектом.")
    resume_path = directory / RESUME_NAME
    has_resume = resume_path.is_file() and not resume_path.is_symlink()
    if bool(document.get("resume_checkpoint")) != has_resume:
        if has_resume:
            detail = "Файл training_checkpoint.ckpt есть, а флаг resume_checkpoint выключен."
        else:
            detail = (
                "Флаг resume_checkpoint включён, файла training_checkpoint.ckpt нет. "
                f"{RESUME_ABSENT}."
            )
        raise BundleChecksumError(detail)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_version(profile: str, job_id: str) -> str:
    version = f"{profile}-{job_id}"
    if _VERSION.fullmatch(version) is None:
        raise LabError("invalid_model_version", "Недопустимое имя версии модели.")
    return version


def _variant(resolved: ResolvedTrainConfig) -> str:
    value = resolved.constructor.get("variant")
    return _require_variant(value if isinstance(value, str) else "")


def _resolution(resolved: ResolvedTrainConfig) -> int:
    value = resolved.constructor.get("resolution")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    noted = resolved.not_applied.get("resolution")
    raw = None if noted is None else noted.value
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    raise LabError(
        "resolution_missing",
        "В профиле нет разрешения. Bundle не записывается.",
    )


def _require_variant(value: object) -> str:
    if value not in {"nano", "small"}:
        raise LabError(
            "rfdetr_variant",
            "Bundle принимает только варианты nano и small.",
        )
    return str(value)


def _require_resolution(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LabError("resolution_missing", "Разрешение модели должно быть целым числом.")
    return value


def _require_hash(value: object) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise LabError(
            "dataset_hash",
            "dataset_hash должен быть SHA-256. Подставной хеш не записывается.",
        )
    return value


def _require_independence(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _SPLIT_INDEPENDENCE:
        raise LabError(
            "split_independence",
            "Независимость разбиения — episode_split или same_scene.",
        )
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _shown(value: str | None) -> str:
    return value if value else NOT_CALCULATED


def _format_metric(value: float | None) -> str:
    if value is None:
        return NOT_CALCULATED
    return format(value, ".12g")


def _format_count(value: int | None) -> str:
    if value is None:
        return NOT_CALCULATED
    return str(value)


def _metric(document: Mapping[str, Any], key: str) -> float | None:
    if key not in document:
        return None
    value = document[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _count(document: Mapping[str, Any], key: str) -> int | None:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _flag(document: Mapping[str, Any], key: str) -> bool | None:
    value = document.get(key)
    if isinstance(value, bool):
        return value
    return None


def _object_copy(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Mapping):
        raise LabError("invalid_snapshot", f"{name} должен быть объектом.")
    try:
        text = json.dumps(value, ensure_ascii=False)
        document = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise LabError("invalid_snapshot", f"{name} не сериализуется в JSON.") from exc
    if not isinstance(document, dict):
        raise LabError("invalid_snapshot", f"{name} должен быть объектом.")
    return document


def _dataset_manifest(
    workspace: Path,
    project_id: str,
    dataset_id: str,
    dataset_dir: Path,
) -> tuple[str, dict[str, Any]]:
    manifest = _optional_object(dataset_dir / "dataset_manifest.json")
    if manifest is not None:
        dataset_hash = manifest.get("dataset_hash")
        if not isinstance(dataset_hash, str) or _HEX.fullmatch(dataset_hash) is None:
            raise LabError("invalid_snapshot", "В dataset_manifest.json нет dataset_hash.")
        return dataset_hash, manifest
    dataset_hash = _sqlite_hash(workspace, project_id, dataset_id)
    return dataset_hash, {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "project_id": project_id,
        "dataset_hash": dataset_hash,
    }


def _sqlite_hash(workspace: Path, project_id: str, dataset_id: str) -> str:
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT content_hash FROM dataset_versions
            WHERE project_id = ? AND dataset_id = ?
            """,
            (project_id, dataset_id),
        ).fetchone()
    value = None if row is None else row["content_hash"]
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise LabError(
            "dataset_hash",
            "У снимка нет dataset_hash. Bundle не записывается.",
        )
    return value


def _split_independence(path: Path) -> str | None:
    """``episode_split`` or ``same_scene`` from a snapshot manifest, else unknown."""
    document = _optional_object(path)
    if document is None:
        return None
    direct = document.get("independence")
    if direct in _SPLIT_INDEPENDENCE:
        return str(direct)
    videos = document.get("videos")
    if not isinstance(videos, dict) or not videos:
        return None
    found: set[str] = set()
    for item in videos.values():
        if not isinstance(item, dict):
            return None
        value = item.get("independence")
        if value not in _SPLIT_INDEPENDENCE:
            return None
        found.add(str(value))
    if len(found) != 1:
        return None
    return next(iter(found))


def _optional_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _require_object(path, f"{path.name} не читается.")


def _require_object(path: Path, missing: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LabError("invalid_snapshot", missing)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError("invalid_snapshot", f"{path.name} повреждён.") from exc
    if not isinstance(document, dict):
        raise LabError("invalid_snapshot", f"{path.name} должен быть объектом.")
    return document


def _metrics_log(run_dir: Path) -> bytes | None:
    path = run_dir / "metrics.csv"
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        return None
    return path.read_bytes()


def _resume_bytes(run_dir: Path) -> bytes | None:
    source = _resume_source(run_dir)
    if source is None:
        return None
    payload = source.read_bytes()
    if not payload:
        return None
    return payload


def _resume_source(run_dir: Path) -> Path | None:
    """Library ``.ckpt`` only. A ``.pth`` file is not a resume checkpoint."""
    found: list[Path] = []
    if not run_dir.is_dir():
        return None
    for path in run_dir.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix != ".ckpt" or path.stat().st_size <= 0:
            continue
        found.append(path)
    if not found:
        return None
    for path in found:
        if path.name == "last.ckpt":
            return path

    def rank(path: Path) -> tuple[int, str]:
        match = re.fullmatch(r"checkpoint_(\d+)", path.stem)
        epoch = int(match.group(1)) if match else -1
        return epoch, path.name

    return max(found, key=rank)


def _view_context(workspace: Path, project_id: str) -> tuple[str | None, str | None]:
    """One saved view can be named. Several views are not guessed."""
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT view_id, equipment_family FROM views
            WHERE project_id = ?
            ORDER BY view_id
            """,
            (project_id,),
        ).fetchall()
    if len(rows) != 1:
        return None, None
    view_id = rows[0]["view_id"]
    family = rows[0]["equipment_family"]
    if not isinstance(view_id, str) or not view_id.strip():
        return None, None
    if not isinstance(family, str) or not family.strip():
        return view_id, None
    return view_id, family


def _reject_official_weights(path: Path, workspace: Path) -> None:
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise LabError(
            "baseline_weights",
            "Путь к весам не читается. Официальные веса COCO не копируются.",
        ) from exc
    official = official_nano_weights_path(workspace)
    try:
        official_resolved = official.resolve()
    except OSError:
        official_resolved = official
    cache = weights_cache_dir(workspace).resolve()
    copied_from_cache = cache in resolved.parents and resolved.name.startswith("rf-detr-")
    if resolved == official_resolved or copied_from_cache:
        raise LabError(
            "baseline_weights",
            "Официальные веса COCO не копируются в bundle как обученная модель. "
            "Нужен результат дообучения.",
        )


def _record_model(
    workspace: Path,
    project_id: str,
    *,
    model_id: str,
    dataset_id: str,
    variant: str,
    path: Path,
) -> None:
    db_path = project_sqlite(workspace, project_id)
    try:
        with connect(db_path) as connection:
            connection.execute(
                """
                INSERT INTO model_versions (
                    model_id, project_id, dataset_id, variant, path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    project_id,
                    dataset_id,
                    variant,
                    str(path.resolve()),
                    datetime.now(UTC).isoformat(),
                ),
            )
    except LabError:
        raise
    except Exception as exc:
        raise LabError(
            "model_record",
            "Каталог модели записан не был: строка model_versions не сохранилась.",
        ) from exc


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    _write_text(
        path,
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_text(path: Path, text: str) -> None:
    _write_bytes(path, text.encode("utf-8"))


def _write_bytes(path: Path, payload: bytes) -> None:
    tmp_path: Path | None = None
    try:
        with map_os_error(str(path)):
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=".bundle-",
                suffix=".tmp",
                dir=path.parent,
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class _CheckedBundle:
    weights: Path
    class_names: tuple[str, ...]
    variant: str
    resolution: int


_VARIANT_TOKENS = {
    "nano": "nano",
    "rfdetrnano": "nano",
    "rf-detr-nano": "nano",
    "rfdetr-nano": "nano",
    "small": "small",
    "rfdetrsmall": "small",
    "rf-detr-small": "small",
    "rfdetr-small": "small",
}


def _check_bundle(path: Path) -> _CheckedBundle:
    """Select ``weights.pth`` and refuse a class, variant, or resolution clash.

    The resume file is hashed with the rest of the directory and is not
    unpickled. A missing ``weights.pth`` does not construct the COCO baseline.
    """
    directory = _bundle_directory(path)
    weights = _require_weights_file(directory)
    verify_bundle_checksums(directory)
    config = _require_object(
        directory / "model_config.json",
        "В bundle нет model_config.json.",
    )
    mapping = _require_object(
        directory / "class_mapping.json",
        "В bundle нет class_mapping.json.",
    )
    if config.get("origin") != CheckpointOrigin.custom:
        raise IncompatibleCheckpointError(
            "model_config.json должен описывать origin=custom. "
            "Базовая COCO-модель не подставляется.",
        )
    if config.get("weights_file", WEIGHTS_NAME) != WEIGHTS_NAME:
        raise IncompatibleCheckpointError(
            "Инференс читает только weights.pth. "
            "training_checkpoint.ckpt не подставляется.",
        )
    expected_variant = _require_variant(config.get("variant"))
    expected_resolution = _require_resolution(config.get("resolution"))
    checkpoint = _read_checkpoint(weights)
    class_names = _require_class_order(mapping, config, checkpoint)
    found_variant = _checkpoint_variant(checkpoint)
    found_resolution = _checkpoint_resolution(checkpoint)
    if found_variant != expected_variant:
        raise IncompatibleCheckpointError(
            f"Вариант weights.pth ({found_variant}) не совпадает с "
            f"model_config.json ({expected_variant}). "
            "Nano и Small не подменяют друг друга.",
        )
    if found_resolution != expected_resolution:
        raise IncompatibleCheckpointError(
            f"Разрешение weights.pth ({found_resolution}) не совпадает с "
            f"model_config.json ({expected_resolution}). "
            "Модель не загружается с другим разрешением.",
        )
    return _CheckedBundle(
        weights=weights,
        class_names=class_names,
        variant=expected_variant,
        resolution=expected_resolution,
    )


def _bundle_directory(path: Path) -> Path:
    if path.name == RESUME_NAME:
        raise IncompatibleCheckpointError(
            "training_checkpoint.ckpt не загружается как детектор. "
            "Нужен каталог bundle и файл weights.pth.",
        )
    if path.name == WEIGHTS_NAME:
        return path.parent
    return path


def _require_weights_file(directory: Path) -> Path:
    weights = directory / WEIGHTS_NAME
    if weights.is_symlink():
        raise IncompatibleCheckpointError(
            "weights.pth является ссылкой. "
            "Загрузка не переходит по ссылке на веса COCO.",
        )
    if not weights.is_file():
        raise CheckpointMissingError(str(weights))
    return weights


def _read_checkpoint(path: Path) -> dict[str, Any]:
    """Library safe load. Full pickle stays off (``trust=False``)."""
    from rfdetr.utilities.io import _safe_torch_load

    try:
        loaded = _safe_torch_load(path, trust=False)
    except Exception as exc:
        raise IncompatibleCheckpointError(
            "weights.pth не открывается безопасной загрузкой. "
            "Полная распаковка pickle не включается, "
            "training_checkpoint.ckpt не подставляется.",
        ) from exc
    if not isinstance(loaded, dict):
        raise IncompatibleCheckpointError(
            "weights.pth не является словарём чекпоинта.",
        )
    return loaded


def _load_custom_module(
    weights: Path,
    *,
    variant: str,
    resolution: int,
    class_count: int,
) -> Any:
    """Build Nano or Small on the bundle file.

    ``trust_checkpoint`` stays false. No score threshold is passed: load does
    not retune the checkpoint, and ``predict`` keeps the raw score floor.
    """
    if variant == "nano":
        from rfdetr import RFDETRNano as builder
    elif variant == "small":
        from rfdetr import RFDETRSmall as builder
    else:
        raise IncompatibleCheckpointError(
            f"Вариант «{variant}» не загружается. Допустимы nano и small.",
        )
    return builder(
        pretrain_weights=os.fspath(weights.resolve()),
        resolution=resolution,
        num_classes=class_count,
        trust_checkpoint=False,
    )


def _require_loaded_names(model: Any, class_names: tuple[str, ...]) -> None:
    reported = _reported_loaded_names(model)
    if reported is None or reported == class_names:
        return
    raise IncompatibleCheckpointError(
        "Имена классов загруженной модели не совпадают с class_mapping.json. "
        f"В модели: {', '.join(reported)}. "
        "Имена COCO не подставляются.",
    )


def _reported_loaded_names(model: Any) -> tuple[str, ...] | None:
    """Names stored on the module, not the COCO fallback property."""
    context = getattr(model, "model", None)
    if context is not None and hasattr(context, "class_names"):
        parsed = _string_list(context.class_names)
        if parsed:
            return parsed
    raw = getattr(model, "__dict__", {}).get("class_names")
    return _string_list(raw)


def _require_class_order(
    mapping: Mapping[str, Any],
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> tuple[str, ...]:
    mapping_names = _mapping_class_names(mapping)
    checkpoint_names = _checkpoint_class_names(checkpoint)
    if mapping_names != checkpoint_names:
        raise IncompatibleCheckpointError(
            "Порядок классов в class_mapping.json не совпадает с метаданными "
            "weights.pth. "
            f"class_mapping.json: {', '.join(mapping_names)}. "
            f"weights.pth: {', '.join(checkpoint_names)}. "
            "Индексы не переставляются.",
        )
    if mapping_names != CLASS_NAMES:
        raise IncompatibleCheckpointError(
            "Порядок классов должен быть excavator_body, bucket, haul_truck, "
            "truck_bed. Другой порядок не загружается.",
        )
    config_names = _string_list(config.get("class_names"))
    if config_names is not None and config_names != mapping_names:
        raise IncompatibleCheckpointError(
            "Порядок class_names в model_config.json не совпадает с "
            "class_mapping.json.",
        )
    return mapping_names


def _mapping_class_names(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Index order is ``internal_id``, not JSON object key order."""
    by_name = document.get("by_name")
    if isinstance(by_name, (str, bytes)) or not isinstance(by_name, Mapping):
        raise IncompatibleCheckpointError(
            "В class_mapping.json нет by_name. Порядок классов не читается.",
        )
    slots: list[str | None] = [None] * len(CLASS_NAMES)
    for name, row in by_name.items():
        if not isinstance(name, str) or isinstance(row, (str, bytes)) or not isinstance(row, Mapping):
            raise IncompatibleCheckpointError(
                "class_mapping.json содержит строку класса без имени.",
            )
        class_id = row.get("internal_id")
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise IncompatibleCheckpointError(
                f"У класса «{name}» в class_mapping.json нет целого internal id.",
            )
        if class_id < 0 or class_id >= len(slots) or slots[class_id] is not None:
            raise IncompatibleCheckpointError(
                f"internal id {class_id} класса «{name}» выходит за порядок "
                "из четырёх классов.",
            )
        slots[class_id] = name
    if any(slot is None for slot in slots):
        raise IncompatibleCheckpointError(
            "В class_mapping.json не хватает класса. Порядок не дополняется.",
        )
    ordered = tuple(slot for slot in slots if slot is not None)
    listed = document.get("class_names")
    if listed is not None and _string_list(listed) != ordered:
        raise IncompatibleCheckpointError(
            "Порядок class_names в class_mapping.json не совпадает с internal id.",
        )
    return ordered


def _checkpoint_class_names(checkpoint: Mapping[str, Any]) -> tuple[str, ...]:
    found: list[tuple[str, ...]] = []
    args = checkpoint.get("args")
    if args is not None:
        parsed = _string_list(_args_value(args, "class_names"))
        if parsed is not None:
            found.append(parsed)
    model_config = checkpoint.get("model_config")
    if isinstance(model_config, Mapping) and "class_names" in model_config:
        parsed = _string_list(model_config.get("class_names"))
        if parsed is None:
            raise IncompatibleCheckpointError(
                "class_names в метаданных weights.pth не является списком имён.",
            )
        if parsed not in found:
            found.append(parsed)
    if len(found) != 1:
        raise IncompatibleCheckpointError(
            "В weights.pth нет порядка классов. "
            "Список из class_mapping.json не подставляется без метаданных чекпоинта.",
        )
    return found[0]


def _checkpoint_variant(checkpoint: Mapping[str, Any]) -> str:
    candidates: list[object] = [checkpoint.get("model_name")]
    model_config = checkpoint.get("model_config")
    if isinstance(model_config, Mapping):
        candidates.append(model_config.get("model_name"))
        candidates.append(model_config.get("variant"))
        candidates.append(model_config.get("size"))
    args = checkpoint.get("args")
    if args is not None:
        candidates.append(_args_value(args, "model_name"))
        candidates.append(_args_value(args, "variant"))
        pretrain = _args_value(args, "pretrain_weights")
        if isinstance(pretrain, str):
            candidates.append(Path(pretrain).name)
    found: list[str] = []
    for candidate in candidates:
        token = _variant_token(candidate)
        if token is not None and token not in found:
            found.append(token)
    if len(found) != 1:
        raise IncompatibleCheckpointError(
            "В weights.pth не указан один вариант nano или small. "
            "Вариант из model_config.json не подставляется вслепую.",
        )
    return found[0]


def _checkpoint_resolution(checkpoint: Mapping[str, Any]) -> int:
    found: list[int] = []
    model_config = checkpoint.get("model_config")
    if isinstance(model_config, Mapping):
        number = _positive_int(model_config.get("resolution"))
        if number is not None:
            found.append(number)
    args = checkpoint.get("args")
    if args is not None:
        number = _positive_int(_args_value(args, "resolution"))
        if number is not None and number not in found:
            found.append(number)
    if len(found) != 1:
        raise IncompatibleCheckpointError(
            "В weights.pth нет разрешения, которое можно сверить с model_config.json.",
        )
    return found[0]


def _variant_token(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _VARIANT_TOKENS.get(value.strip().lower().replace("_", ""))


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _string_list(value: object) -> tuple[str, ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return None
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            return None
        names.append(item)
    return tuple(names)


def _args_value(args: object, key: str) -> object:
    if isinstance(args, Mapping):
        return args.get(key)
    return getattr(args, key, None)
