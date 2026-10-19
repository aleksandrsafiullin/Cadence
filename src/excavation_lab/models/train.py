"""Fine-tune RF-DETR through the installed train API.

The parent enqueues a job and does not construct ``RFDETRNano`` or
``RFDETRSmall``. The worker builds a directory of symlinks to ``train`` and
``valid`` only, then calls ``model.train``. The project's ``test`` split stays
on disk and is not an entry in that view.

``augmentation_train`` and ``augmentation_eval`` stay on the payload the test
double records. They are not ``TrainConfig`` fields, so the library call drops
them and sends the off switch (``aug_config={}``, ``scale_jitter=False``,
``multi_scale=False``) with ``run_test=False``. This module does not call
``augment.apply``. It does not freeze the backbone and it does not train only
the last layer. A failed device is written onto the job row. Training is not
started again on CPU.

When a local pretrain file is already in the workspace cache, the worker
rewrites it to ``class_head.pth`` in the run directory and passes that path
as ``pretrain_weights`` with ``num_classes`` equal to the catalog length.
The module RF-DETR builds inside ``train`` then loads the same width, so
``_resize_linear`` does not repeat or truncate rows. ``class_head.pth`` is
an input to that call. ``publish_train_bundle`` still publishes the finished
``weights.pth`` and is what marks a bundle custom.

Before ``model.train``, ``SetCriterion.loss_labels`` is wrapped so an
unsupervised class column is zeroed on ``pos_weights`` and ``neg_weights``
before that method reduces them. The original method is put back when
``model.train`` returns. ``run_test`` stays false.

``overfit`` is a data and class-wiring check on 20–40 reviewed frames that
``require_frozen`` assigns to train. Its metrics are ``diagnostic_overfit``.
They are not a test score. A finished run writes ``models/<profile>-<job_id>/``.
``weights.pth`` there is the inference file. ``training_checkpoint.ckpt`` is
copied only when the library wrote a ``.ckpt``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from excavation_lab.datasets.blocks import assignment_at as role_at
from excavation_lab.datasets.catalog import require_project_catalog
from excavation_lab.datasets.coco_import import ImportedFrame, load_imported_frames
from excavation_lab.datasets.splits import FrozenSplit, require_frozen
from excavation_lab.errors import LabError
from excavation_lab.hardware import weights_cache_dir
from excavation_lab.jobs import (
    PROGRESS_FILENAME,
    get_job,
    mark_completed,
    mark_failed,
    start_job,
    worker_job,
    write_progress,
)
from excavation_lab.models.augment import library_off_switch
from excavation_lab.models.bundle import publish_train_bundle
from excavation_lab.models.class_head import rewrite_class_head
from excavation_lab.models.supervision import (
    catalog_class_names,
    observe_unsupervised,
    wrap_loss_labels,
)
from excavation_lab.models.train_config import (
    METRIC_ROLE_DIAGNOSTIC_OVERFIT,
    METRIC_ROLE_SYNTHETIC_WIRING,
    _installed_train_api,
    resolve_train_profile,
    write_resolved_train_config,
)
from excavation_lab.schemas import (
    PROFILE_OVERFIT,
    PROFILE_PILOT_SMALL,
    PROFILE_SMOKE,
    SCHEMA_VERSION,
    JobStatus,
    ReviewStatus,
    load_pilot,
)
from excavation_lab.storage import connect, project_sqlite

FAKE_TRAINER_ENV = "EXCAVATION_LAB_FAKE_TRAINER"
MANUAL_SMOKE_ENV = "EXCAVATION_LAB_MANUAL_SMOKE"
# First substantive validation plan in the spec. Fewer images is not a
# stable checkpoint pick. This is not a quality threshold.
STABLE_VALIDATION_IMAGES = 60
OVERFIT_MIN_FRAMES = 20
OVERFIT_MAX_FRAMES = 40
OVERFIT_OUTPUT_NAME = "output.json"
_PROFILES = frozenset({PROFILE_SMOKE, PROFILE_PILOT_SMALL, PROFILE_OVERFIT})
_DEVICES = frozenset({"auto", "cuda", "mps", "cpu"})
_EXPLICIT_DEVICES = frozenset({"cuda", "mps", "cpu"})
_FAKE_MODES = frozenset({"hold", "empty", "fail", "metrics", "reload"})
_OFFICIAL_WEIGHTS = {
    "nano": "rf-detr-nano.pth",
    "small": "rf-detr-small.pth",
}
CLASS_HEAD_NAME = "class_head.pth"
_PROJECT_AUGMENT_KEYS = frozenset({"augmentation_train", "augmentation_eval"})
_VAL_MAP_KEYS = ("val/mAP_50_95", "val/map_50_95", "validation/mAP_50_95")
_BUCKET_KEYS = ("val/AP/bucket", "val/ap/bucket")
_NOT_READY = (
    "В train или validation нет проверенных изображений. "
    "Обучение не запускается, weights.pth не записывается."
)
_NOT_FOUND = "Снимок датасета не найден. Сначала соберите снимок."
_HOLDS = 300.0
_START_WAIT_S = 10.0
TrainFn = Callable[..., Mapping[str, Any] | None]


class _WorkerStopped(Exception):
    """The job was cancelled while the test double was holding the lock."""


def enqueue_train(
    workspace: Path,
    project_id: str,
    dataset_id: str,
    profile: str,
    *,
    device: str | None = None,
    profile_override: Mapping[str, Any] | None = None,
) -> Any:
    """Queue one train job. Does not construct an RF-DETR model.

    ``device`` is ``auto``, ``cuda``, ``mps``, or ``cpu``. Omitted and
    ``auto`` leave the choice to the worker. ``profile_override`` replaces
    the YAML profile in that worker. It is not applied in this process.
    """
    if profile not in _PROFILES:
        raise LabError("unknown_profile", f"Неизвестный профиль обучения: {profile}")
    if device is not None and device not in _DEVICES:
        raise LabError(
            "unknown_device",
            "Устройство должно быть auto, cuda, mps или cpu.",
        )
    override = None if profile_override is None else _json_object(profile_override)
    if override is not None:
        _reject_variant_override(profile, override)
    dataset_key = _require_dataset_id(dataset_id)
    dataset_dir = _dataset_directory(workspace, project_id, dataset_key)
    if profile == PROFILE_OVERFIT:
        _select_overfit_frames(workspace, project_id, dataset_dir)
    else:
        _require_reviewed_images(dataset_dir)
    mode = os.environ.get(FAKE_TRAINER_ENV)

    def write_request(run_dir: Path) -> None:
        body: dict[str, Any] = {"dataset_id": dataset_key, "profile": profile}
        if device is not None:
            body["device"] = device
        if override is not None:
            body["profile_override"] = override
        if mode:
            body["fake_trainer"] = mode
        _atomic_json(run_dir / "train_request.json", body)

    return start_job(
        workspace,
        project_id,
        kind="train",
        worker="rfdetr",
        on_run_dir=write_request,
    )


def run_train_job(
    workspace: Path,
    job_id: str,
    *,
    trainer: TrainFn | None = None,
) -> int:
    """Worker entry. Loads the model only when ``trainer`` is the library."""
    root = Path(workspace).resolve()
    project_id, run_dir, _record = worker_job(root, job_id)
    if not _wait_until_running(root, project_id, job_id):
        record = get_job(root, project_id, job_id)
        if record.status in {JobStatus.queued, JobStatus.running}:
            _fail(root, project_id, job_id, run_dir, "worker_start_timeout")
        return 1
    try:
        return _execute(root, project_id, job_id, run_dir, trainer)
    except _WorkerStopped:
        return 0
    except LabError as exc:
        _fail(root, project_id, job_id, run_dir, exc.code)
        return 1
    except Exception as exc:  # noqa: BLE001 — the job row keeps the backend exception
        _fail(root, project_id, job_id, run_dir, _exception_text(exc))
        return 1


def choose_train_device() -> str:
    """One device: CUDA, otherwise MPS, otherwise CPU.

    Does not probe a second device and does not restart on CPU.
    """
    import torch

    cuda = getattr(torch, "cuda", None)
    if cuda is not None and bool(cuda.is_available()):
        return "cuda"
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is not None and bool(mps.is_available()):
        return "mps"
    return "cpu"


def build_validation_metrics(
    reported: Mapping[str, Any] | None,
    *,
    validation_images: int,
) -> dict[str, Any]:
    """Validation box mAP50:95 and bucket AP. Test numbers are not copied."""
    body = dict(reported or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_metric": "val/mAP_50_95",
        "box_map_50_95": _box_map(body),
        "bucket_ap": _bucket_ap(body),
        "selection_unstable": validation_images < STABLE_VALIDATION_IMAGES,
        "validation_images": validation_images,
    }


def _execute(
    workspace: Path,
    project_id: str,
    job_id: str,
    run_dir: Path,
    trainer: TrainFn | None,
) -> int:
    request = _read_request(run_dir)
    dataset_dir = _dataset_directory(workspace, project_id, request["dataset_id"])
    profile = request["profile"]
    if profile not in _PROFILES:
        raise LabError("unknown_profile", f"Неизвестный профиль обучения: {profile}")
    frames: list[_ChosenFrame] | None = None
    categories: list[Any] = []
    if profile == PROFILE_OVERFIT:
        frames, categories = _select_overfit_frames(workspace, project_id, dataset_dir)
    else:
        _require_reviewed_images(dataset_dir)
    resolved = _resolved_for_request(str(profile), request)
    device = _device_for_request(request)
    batch_size = int(resolved.train_kwargs["batch_size"])
    diagnostic = " diagnostic=true" if profile == PROFILE_OVERFIT else ""
    _append_log(
        run_dir,
        (
            f"profile={profile} device={device} "
            f"batch_size={batch_size} run_test=false{diagnostic}"
        ),
    )
    write_resolved_train_config(run_dir / "resolved_train_config.json", resolved)
    view: Path | None = None
    try:
        if frames is None:
            validation_images = _image_count(dataset_dir / "valid")
            view = _hide_test(dataset_dir, run_dir)
        else:
            validation_images = len(frames)
            view = _overfit_view(run_dir, frames, categories)
        payload = _call_payload(
            resolved,
            dataset_dir=view,
            output_dir=run_dir,
            device=device,
        )
        _prepare_class_head(workspace, project_id, run_dir, resolved, payload)
        _note_progress(
            run_dir,
            job_id=job_id,
            status=JobStatus.running.value,
            profile=str(profile),
            device=device,
        )
        train_fn = trainer if trainer is not None else _trainer_for_request(
            request,
            resolved,
            workspace,
            project_id,
            job_id,
        )
        try:
            reported = train_fn(**payload)
        except _WorkerStopped:
            raise
        except Exception as exc:  # noqa: BLE001 — one attempt, then the job row
            _fail(workspace, project_id, job_id, run_dir, _exception_text(exc))
            return 1
        if not reported:
            reported = _read_library_metrics(run_dir)
        metrics = build_validation_metrics(
            reported if isinstance(reported, Mapping) else None,
            validation_images=validation_images,
        )
        if isinstance(reported, Mapping) and reported.get("synthetic") is True:
            metrics = _label_synthetic_metrics(metrics)
        elif frames is not None:
            metrics = _label_overfit_metrics(metrics)
            _atomic_json(run_dir / OVERFIT_OUTPUT_NAME, _overfit_output(metrics, frames))
        _atomic_json(run_dir / "validation_metrics.json", metrics)
        weights = _publish_weights(run_dir)
        if weights is None:
            _fail(workspace, project_id, job_id, run_dir, "empty_checkpoint")
            return 1
        bundle = publish_train_bundle(
            workspace,
            project_id,
            job_id,
            dataset_id=str(request["dataset_id"]),
            dataset_dir=dataset_dir,
            run_dir=run_dir,
            profile=str(profile),
            resolved=resolved,
        )
        mark_completed(
            workspace,
            project_id,
            job_id,
            [weights, bundle / "checksums.json", bundle / "weights.pth"],
        )
        _note_progress(
            run_dir,
            job_id=job_id,
            status=JobStatus.completed.value,
            profile=str(profile),
            device=device,
        )
        _append_log(run_dir, f"checkpoint={weights.name}")
        _append_log(run_dir, f"bundle={bundle.name}")
        return 0
    finally:
        if view is not None:
            _remove_view(view)


def _call_payload(
    resolved: Any,
    *,
    dataset_dir: Path,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Resolved train kwargs plus the run paths.

    Includes ``augmentation_train`` and ``augmentation_eval``. The library
    call strips those two names.
    """
    payload = dict(resolved.train_kwargs)
    payload.update(library_off_switch())
    payload["dataset_dir"] = str(dataset_dir)
    payload["output_dir"] = str(output_dir)
    payload["device"] = device
    payload["run_test"] = False
    payload["batch_size"] = int(resolved.train_kwargs["batch_size"])
    fields = _installed_train_api().train_fields
    extras = {
        "log_per_class_metrics": True,
        "tensorboard": False,
        "wandb": False,
        "mlflow": False,
        "clearml": False,
        "progress_bar": None,
    }
    for key, value in extras.items():
        if key in fields:
            payload[key] = value
    if "gradient_checkpointing" in payload:
        raise LabError(
            "train_kwargs_split",
            "gradient_checkpointing относится к конструктору и не входит в train kwargs.",
        )
    return payload


def _prepare_class_head(
    workspace: Path,
    project_id: str,
    run_dir: Path,
    resolved: Any,
    payload: dict[str, Any],
) -> None:
    """Point the constructor at a catalog-width copy of the local pretrain file.

    No local file means this does not download weights. The library trainer
    still receives ``run_test=False``. The rewritten file is not a published
    bundle.
    """
    variant = resolved.constructor.get("variant")
    source = _local_pretrain_file(workspace, variant)
    if source is None:
        return
    names = _catalog_names(workspace, project_id)
    destination = run_dir / CLASS_HEAD_NAME
    rewrite_class_head(source, destination, names, workspace=workspace)
    payload["pretrain_weights"] = os.fspath(destination.resolve())
    payload["num_classes"] = len(names)


def _local_pretrain_file(workspace: Path, variant: object) -> Path | None:
    if not isinstance(variant, str):
        return None
    filename = _OFFICIAL_WEIGHTS.get(variant)
    if filename is None:
        return None
    path = weights_cache_dir(workspace) / filename
    if path.is_symlink() or not path.is_file():
        return None
    return path


def _constructor_kwargs(resolved: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Model constructor kwargs. ``num_classes`` is the catalog length."""
    constructor = dict(resolved.constructor_kwargs())
    pretrain = payload.get("pretrain_weights")
    if isinstance(pretrain, str) and pretrain:
        constructor["pretrain_weights"] = pretrain
    class_count = payload.get("num_classes")
    if isinstance(class_count, int) and not isinstance(class_count, bool):
        constructor["num_classes"] = class_count
    return constructor


def _library_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    """``model.train`` kwargs. Project augmentation names are omitted."""
    fields = _installed_train_api().train_fields
    allowed = set(fields) | {"device"}
    library = {
        key: value
        for key, value in payload.items()
        if key in allowed and key not in _PROJECT_AUGMENT_KEYS
    }
    library.update(library_off_switch())
    library["run_test"] = False
    library["batch_size"] = int(payload["batch_size"])
    library["device"] = payload["device"]
    return library


def _trainer_for_request(
    request: Mapping[str, Any],
    resolved: Any,
    workspace: Path,
    project_id: str,
    job_id: str,
) -> TrainFn:
    mode = request.get("fake_trainer")
    if not isinstance(mode, str) or not mode:
        mode = os.environ.get(FAKE_TRAINER_ENV)
    if not mode:
        return _library_trainer_for(resolved, workspace, project_id)
    if mode not in _FAKE_MODES:
        raise LabError("invalid_job", f"Неизвестный тестовый trainer: {mode}")
    return _fake_trainer(mode, workspace, project_id, job_id)


def _catalog_names(workspace: Path, project_id: str) -> list[str]:
    return catalog_class_names(require_project_catalog(workspace, project_id))


def _library_trainer_for(resolved: Any, workspace: Path, project_id: str) -> TrainFn:
    def _call(**kwargs: Any) -> dict[str, Any] | None:
        _reject_unit_test_train()
        from rfdetr import RFDETRNano, RFDETRSmall

        classes = {"RFDETRNano": RFDETRNano, "RFDETRSmall": RFDETRSmall}
        class_name = resolved.constructor["class_name"]
        cls = classes.get(class_name)
        if cls is None:
            raise LabError(
                "rfdetr_variant",
                f"Нет конструктора RF-DETR для класса {class_name}.",
            )
        model = cls(device=str(kwargs["device"]), **_constructor_kwargs(resolved, kwargs))
        names = _catalog_names(workspace, project_id)
        with wrap_loss_labels(Path(kwargs["dataset_dir"]), names):
            model.train(**_library_kwargs(kwargs))
        return _read_library_metrics(Path(kwargs["output_dir"]))

    return _call


def _reject_unit_test_train() -> None:
    if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get(MANUAL_SMOKE_ENV) != "1":
        raise RuntimeError("unit tests must not call RF-DETR train")


def _fake_trainer(
    mode: str,
    workspace: Path,
    project_id: str,
    job_id: str,
) -> TrainFn:
    def _call(**kwargs: Any) -> dict[str, Any] | None:
        output_dir = Path(kwargs["output_dir"])
        dataset_dir = Path(kwargs["dataset_dir"])
        opened = _opened_paths(dataset_dir)
        seen = observe_unsupervised(dataset_dir, _catalog_names(workspace, project_id))
        _append_trainer_call(output_dir, kwargs, opened, seen)
        if mode == "hold":
            _hold(workspace, project_id, job_id)
            raise _WorkerStopped()
        if mode == "fail":
            raise RuntimeError("mps backend failed: simulated")
        if mode == "metrics":
            (output_dir / "weights.pth").write_bytes(b"checkpoint")
        if mode == "reload":
            _write_reloadable_checkpoint(output_dir, workspace, project_id)
            return {
                "synthetic": True,
                "metric_role": METRIC_ROLE_SYNTHETIC_WIRING,
            }
        return {
            "val/mAP_50_95": 0.5,
            "test/mAP_50_95": 0.125,
            "val/AP/bucket": 0.25,
            "test/AP/bucket": 0.75,
        }

    return _call


def _hold(workspace: Path, project_id: str, job_id: str) -> None:
    deadline = time.monotonic() + _HOLDS
    while time.monotonic() < deadline:
        record = get_job(workspace, project_id, job_id)
        if record.status in {
            JobStatus.cancel_requested,
            JobStatus.cancelled,
            JobStatus.failed,
            JobStatus.completed,
        }:
            return
        time.sleep(0.05)


def _opened_paths(dataset_dir: Path) -> list[str]:
    """Open every directory entry the train view exposes."""
    opened: list[str] = []
    root = dataset_dir.resolve()
    opened.append(str(root))
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        opened.append(str(child.resolve()))
        if not child.is_dir():
            continue
        for item in sorted(child.iterdir(), key=lambda entry: entry.name):
            if not item.is_file():
                continue
            with item.open("rb") as handle:
                handle.read(1)
            opened.append(str(item.resolve()))
    return opened


def _append_trainer_call(
    output_dir: Path,
    payload: Mapping[str, Any],
    opened: list[str],
    supervision_seen: list[dict[str, Any]],
) -> None:
    path = output_dir / "trainer_calls.json"
    calls: list[Any] = []
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            calls = loaded
    recorded = json.loads(json.dumps(dict(payload), ensure_ascii=False))
    calls.append(
        {
            "kwargs": recorded,
            "opened": opened,
            "model_constructed": False,
            "supervision_seen": supervision_seen,
        }
    )
    _atomic_json(path, calls)


def _hide_test(dataset_dir: Path, run_dir: Path) -> Path:
    """Symlink ``train`` and ``valid``. Do not link ``test`` and do not delete it."""
    view = Path(tempfile.mkdtemp(prefix="dataset-view-", dir=run_dir))
    for name in ("train", "valid"):
        source = (dataset_dir / name).resolve()
        if not source.is_dir():
            raise LabError("dataset_not_ready", _NOT_READY)
        os.symlink(source, view / name, target_is_directory=True)
    if (view / "test").exists() or (view / "test").is_symlink():
        raise LabError(
            "test_visible",
            "Каталог test попал в каталог, который открывает обучение.",
        )
    return view


def _remove_view(view: Path) -> None:
    """Drop the view symlinks. The dataset directories they point at stay."""
    if not view.exists():
        return
    try:
        for child in list(view.iterdir()):
            if child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        view.rmdir()
    except OSError:
        return


@dataclass(frozen=True, slots=True)
class _ChosenFrame:
    file_name: str
    source: Path
    video_id: str
    source_frame_index: int
    width: int
    height: int
    annotations: tuple[dict[str, Any], ...]


def _select_overfit_frames(
    workspace: Path,
    project_id: str,
    dataset_dir: Path,
) -> tuple[list[_ChosenFrame], list[Any]]:
    """Reviewed train frames listed in the train COCO.

    ``require_frozen`` assigns the split. The stored split string on the
    import is not the authority. A validation, test, or unreviewed
    candidate fails the run. Fewer than 20 unique train frames fails
    without copying or inventing images. The project's ``test`` directory
    is not read and is not deleted.
    """
    document = _read_train_coco(dataset_dir)
    images = document.get("images")
    if not isinstance(images, list):
        images = []
    annotations = _annotations_by_image(document)
    categories = document.get("categories")
    if not isinstance(categories, list):
        categories = []
    imported, duplicated = _imported_by_file(workspace, project_id)
    frozen_by_video: dict[str, FrozenSplit] = {}
    rows: list[_ChosenFrame] = []
    train_root = (dataset_dir / "train").resolve()
    for image in images:
        if not isinstance(image, Mapping):
            _reject_overfit_frame("?", "Запись изображения в train COCO повреждена.")
        file_name = _overfit_image_name(image)
        if file_name in duplicated:
            _reject_overfit_frame(
                file_name,
                "Кадр импортирован дважды. Вторая копия не создаётся.",
            )
        frame = imported.get(file_name)
        if frame is None:
            _reject_overfit_frame(
                file_name,
                "Кадр не сопоставлен с импортом. require_frozen не подтвердил train.",
            )
        role = _frozen_role(workspace, project_id, frame, frozen_by_video)
        _reject_unless_reviewed_train(file_name, frame, role)
        source = (train_root / file_name).resolve()
        if not source.is_relative_to(train_root) or not source.is_file():
            _reject_overfit_frame(
                file_name,
                "Файла нет в каталоге train. Копия не создаётся.",
            )
        rows.append(
            _ChosenFrame(
                file_name=file_name,
                source=source,
                video_id=frame.video_id,
                source_frame_index=frame.source_frame_index,
                width=_positive_size(image.get("width"), frame.width),
                height=_positive_size(image.get("height"), frame.height),
                annotations=tuple(annotations.get(image.get("id"), ())),
            )
        )
    rows.sort(key=lambda item: (item.video_id, item.source_frame_index, item.file_name))
    unique: list[_ChosenFrame] = []
    seen: set[tuple[str, int]] = set()
    for item in rows:
        identity = (item.video_id, item.source_frame_index)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(item)
    if len(unique) < OVERFIT_MIN_FRAMES:
        raise LabError(
            "not_enough_train_frames",
            f"Проверенных train-кадров {len(unique)}, нужно от {OVERFIT_MIN_FRAMES} "
            f"до {OVERFIT_MAX_FRAMES}. "
            "Недостающие кадры не копируются и не выдумываются.",
        )
    return unique[:OVERFIT_MAX_FRAMES], categories


def _overfit_view(
    run_dir: Path,
    frames: Sequence[_ChosenFrame],
    categories: list[Any],
) -> Path:
    """Symlink the selected train frames into ``train`` and ``valid``.

    ``valid`` points at those same reviewed train frames. It is not the
    project's validation split. The links are not extra images added to
    reach 20. ``test`` is not linked and is not deleted.
    """
    view = Path(tempfile.mkdtemp(prefix="overfit-view-", dir=run_dir))
    document = _subset_coco(categories, frames)
    for name in ("train", "valid"):
        directory = view / name
        directory.mkdir()
        for frame in frames:
            os.symlink(frame.source, directory / frame.file_name)
        _atomic_json(directory / "_annotations.coco.json", document)
    if (view / "test").exists() or (view / "test").is_symlink():
        raise LabError(
            "test_visible",
            "Каталог test попал в каталог, который открывает обучение.",
        )
    return view


def _label_synthetic_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Wiring-run metrics. Numbers from the double are not measurements."""
    labeled = dict(metrics)
    labeled["synthetic"] = True
    labeled["metric_role"] = METRIC_ROLE_SYNTHETIC_WIRING
    labeled["box_map_50_95"] = None
    labeled["bucket_ap"] = None
    labeled.pop("quality_claim", None)
    return labeled


def _label_overfit_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    labeled = dict(metrics)
    labeled["metric_role"] = METRIC_ROLE_DIAGNOSTIC_OVERFIT
    labeled["diagnostic"] = True
    labeled.pop("quality_claim", None)
    labeled.pop("split", None)
    return _without_test_claim(labeled)


def _overfit_output(
    metrics: Mapping[str, Any],
    frames: Sequence[_ChosenFrame],
) -> dict[str, Any]:
    body = {
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE_OVERFIT,
        "diagnostic": True,
        "metric_role": METRIC_ROLE_DIAGNOSTIC_OVERFIT,
        "frame_count": len(frames),
        "frames": [frame.file_name for frame in frames],
        "selection_metric": metrics.get("selection_metric"),
        "box_map_50_95": metrics.get("box_map_50_95"),
        "bucket_ap": metrics.get("bucket_ap"),
        "selection_unstable": metrics.get("selection_unstable"),
        "validation_images": metrics.get("validation_images"),
    }
    return _without_test_claim(body)


def _without_test_claim(document: dict[str, Any]) -> dict[str, Any]:
    cleaned = _strip_test_claim(document)
    if not isinstance(cleaned, dict) or _claim_has_test_split(cleaned):
        raise LabError(
            "quality_claim",
            "Диагностика overfit не записывается как качество модели.",
        )
    return cleaned


def _strip_test_claim(node: object) -> object:
    if isinstance(node, dict):
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key in {"quality_claim", "test"} or (
                isinstance(key, str) and key.startswith("test/")
            ):
                continue
            if key == "split" and value in {"test", "testing"}:
                continue
            cleaned[str(key)] = _strip_test_claim(value)
        return cleaned
    if isinstance(node, list):
        kept: list[Any] = []
        for item in node:
            if isinstance(item, dict) and item.get("split") in {"test", "testing"}:
                continue
            kept.append(_strip_test_claim(item))
        return kept
    return node


def _claim_has_test_split(node: object) -> bool:
    if isinstance(node, dict):
        if node.get("quality_claim") is True or node.get("split") in {"test", "testing"}:
            return True
        return any(_claim_has_test_split(value) for value in node.values())
    if isinstance(node, list):
        return any(_claim_has_test_split(item) for item in node)
    return False


def _read_train_coco(dataset_dir: Path) -> dict[str, Any]:
    """Train COCO only. Does not open ``valid`` or ``test``."""
    path = dataset_dir / "train" / "_annotations.coco.json"
    if not path.is_file():
        return {"images": [], "annotations": [], "categories": []}
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise LabError("dataset_not_ready", _NOT_READY)
    return document


def _annotations_by_image(
    document: Mapping[str, Any],
) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = {}
    rows = document.get("annotations")
    if not isinstance(rows, list):
        return grouped
    for row in rows:
        if not isinstance(row, dict):
            continue
        grouped.setdefault(row.get("image_id"), []).append(dict(row))
    return grouped


def _imported_by_file(
    workspace: Path,
    project_id: str,
) -> tuple[dict[str, ImportedFrame], set[str]]:
    found: dict[str, ImportedFrame] = {}
    duplicated: set[str] = set()
    for frame in load_imported_frames(workspace, project_id):
        if frame.image_file in found or frame.image_file in duplicated:
            duplicated.add(frame.image_file)
            found.pop(frame.image_file, None)
            continue
        found[frame.image_file] = frame
    return found, duplicated


def _frozen_role(
    workspace: Path,
    project_id: str,
    frame: ImportedFrame,
    cache: dict[str, FrozenSplit],
) -> str | None:
    """Split named by ``require_frozen``. The import's split string is not used."""
    frozen = cache.get(frame.video_id)
    if frozen is None:
        frozen = require_frozen(workspace, project_id, frame.video_id)
        cache[frame.video_id] = frozen
    return role_at(frozen.blocks, frame.source_time_s)


def _reject_unless_reviewed_train(
    file_name: str,
    frame: ImportedFrame,
    role: str | None,
) -> None:
    # ``frame.split`` is ignored. require_frozen is the assignment.
    if frame.record.review_status is not ReviewStatus.reviewed:
        _reject_overfit_frame(
            file_name,
            "Кадр не проверен. Непроверенный кадр в overfit не входит.",
        )
    if role == "validation":
        _reject_overfit_frame(
            file_name,
            "По зафиксированному разбиению кадр относится к validation, не к train.",
        )
    if role == "test":
        _reject_overfit_frame(
            file_name,
            "По зафиксированному разбиению кадр относится к test, не к train.",
        )
    if role != "train":
        _reject_overfit_frame(
            file_name,
            "По зафиксированному разбиению кадр не относится к train.",
        )


def _reject_overfit_frame(file_name: str, detail: str) -> NoReturn:
    raise LabError("overfit_frame_rejected", f"Кадр «{file_name}» не входит в overfit. {detail}")


def _overfit_image_name(image: Mapping[str, Any]) -> str:
    file_name = image.get("file_name")
    if (
        not isinstance(file_name, str)
        or not file_name
        or file_name in {".", ".."}
        or "/" in file_name
        or "\\" in file_name
        or "\x00" in file_name
    ):
        shown = file_name if isinstance(file_name, str) else "?"
        _reject_overfit_frame(shown, "Имя файла кадра некорректно. Файл не создаётся.")
    return file_name


def _positive_size(value: object, fallback: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return fallback
    return value


def _subset_coco(
    categories: list[Any],
    frames: Sequence[_ChosenFrame],
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for new_id, frame in enumerate(frames, start=1):
        images.append(
            {
                "id": new_id,
                "file_name": frame.file_name,
                "width": frame.width,
                "height": frame.height,
            }
        )
        for ann in frame.annotations:
            copied = dict(ann)
            copied.pop("split", None)
            copied["id"] = len(annotations) + 1
            copied["image_id"] = new_id
            annotations.append(copied)
    return {
        "categories": list(categories),
        "images": images,
        "annotations": annotations,
    }


def _require_reviewed_images(dataset_dir: Path) -> None:
    """Train and validation each need one COCO image. ``test`` is not opened."""
    if _image_count(dataset_dir / "train") < 1 or _image_count(dataset_dir / "valid") < 1:
        raise LabError("dataset_not_ready", _NOT_READY)


def _image_count(split_dir: Path) -> int:
    document_path = split_dir / "_annotations.coco.json"
    if not document_path.is_file():
        return 0
    document = json.loads(document_path.read_text(encoding="utf-8"))
    images = document.get("images") if isinstance(document, dict) else None
    if not isinstance(images, list):
        return 0
    return len(images)


def _dataset_directory(workspace: Path, project_id: str, dataset_id: str) -> Path:
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT path FROM dataset_versions
            WHERE project_id = ? AND dataset_id = ?
            """,
            (project_id, dataset_id),
        ).fetchone()
    path_text = None if row is None else row["path"]
    if not isinstance(path_text, str) or not path_text:
        raise LabError("dataset_not_found", _NOT_FOUND)
    dataset_dir = Path(path_text)
    if not dataset_dir.is_dir():
        raise LabError("dataset_not_found", f"Каталог снимка не найден: {dataset_dir}.")
    return dataset_dir


def _require_dataset_id(dataset_id: str) -> str:
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or dataset_id in {".", ".."}
        or "/" in dataset_id
        or "\\" in dataset_id
        or "\x00" in dataset_id
    ):
        raise LabError("dataset_not_found", _NOT_FOUND)
    return dataset_id


def _json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=False)
        parsed = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise LabError(
            "config_invalid",
            "Настройки профиля не сериализуются.",
        ) from exc
    if not isinstance(parsed, dict):
        raise LabError("config_invalid", "Настройки профиля должны быть объектом.")
    return parsed


def _reject_variant_override(profile: str, override: Mapping[str, Any]) -> None:
    variant = override.get("variant")
    if not isinstance(variant, str):
        return
    expected = load_pilot().train_profile(profile).variant.value
    if variant != expected:
        raise LabError(
            "profile_variant",
            f"Профиль {profile} ожидает вариант {expected}, указан {variant}.",
        )


def _resolved_for_request(profile: str, request: Mapping[str, Any]) -> Any:
    override = request.get("profile_override")
    if isinstance(override, dict) and override:
        return resolve_train_profile(profile, profile=override)
    return resolve_train_profile(profile)


def _device_for_request(request: Mapping[str, Any]) -> str:
    """``auto`` and a missing device pick once. An explicit device is not probed."""
    raw = request.get("device", "auto")
    if raw is None or raw == "auto":
        return choose_train_device()
    if isinstance(raw, str) and raw in _EXPLICIT_DEVICES:
        return raw
    raise LabError(
        "unknown_device",
        "Устройство должно быть auto, cuda, mps или cpu.",
    )


def _read_request(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "train_request.json"
    if not path.is_file():
        raise LabError("invalid_job", "В каталоге задания нет train_request.json.")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise LabError("invalid_job", "train_request.json должен быть объектом.")
    dataset_id = document.get("dataset_id")
    profile = document.get("profile")
    if not isinstance(dataset_id, str) or not isinstance(profile, str):
        raise LabError("invalid_job", "train_request.json не содержит датасет и профиль.")
    return document


def _wait_until_running(workspace: Path, project_id: str, job_id: str) -> bool:
    deadline = time.monotonic() + _START_WAIT_S
    while time.monotonic() < deadline:
        record = get_job(workspace, project_id, job_id)
        if record.status == JobStatus.running:
            return True
        if record.status in {
            JobStatus.failed,
            JobStatus.cancelled,
            JobStatus.completed,
            JobStatus.cancel_requested,
        }:
            return False
        time.sleep(0.01)
    return False


def _note_progress(run_dir: Path, **fields: object) -> None:
    """Merge one progress object. ``run_test`` stays false on every write."""
    current: dict[str, Any] = {}
    path = run_dir / PROGRESS_FILENAME
    if path.is_file() and not path.is_symlink():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            current = loaded
    current.update(fields)
    current["run_test"] = False
    write_progress(run_dir, current)


def _write_reloadable_checkpoint(
    output_dir: Path,
    workspace: Path,
    project_id: str,
) -> None:
    """Metadata-only ``weights.pth`` the bundle loader can open.

    The bytes are not a trained network. They carry the catalog class order
    and the resolved variant and resolution so ``load_bundle`` can check them.
    """
    import torch

    document = json.loads(
        (output_dir / "resolved_train_config.json").read_text(encoding="utf-8")
    )
    constructor = document.get("constructor")
    if not isinstance(constructor, dict):
        constructor = {}
    variant = constructor.get("variant")
    resolution = constructor.get("resolution")
    if variant not in {"nano", "small"}:
        raise LabError("rfdetr_variant", "Тестовый чекпоинт не знает вариант профиля.")
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
        raise LabError("resolution_missing", "Тестовый чекпоинт не знает разрешение профиля.")
    names = _catalog_names(workspace, project_id)
    model_name = "RFDETRNano" if variant == "nano" else "RFDETRSmall"
    torch.save(
        {
            "model": {"probe": torch.zeros(1)},
            "args": {"class_names": list(names), "num_classes": len(names)},
            "model_name": model_name,
            "model_config": {"resolution": resolution},
            "synthetic": True,
            "metric_role": METRIC_ROLE_SYNTHETIC_WIRING,
        },
        output_dir / "weights.pth",
    )


def _fail(
    workspace: Path,
    project_id: str,
    job_id: str,
    run_dir: Path,
    code: str,
) -> None:
    _append_log(run_dir, f"failed {code}")
    _note_progress(run_dir, job_id=job_id, status=JobStatus.failed.value, error=code)
    mark_failed(workspace, project_id, job_id, code)


def _exception_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    if not text:
        return "train_failed"
    return text[:2000]


def _append_log(run_dir: Path, line: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _box_map(reported: Mapping[str, Any]) -> float | None:
    for key in _VAL_MAP_KEYS:
        number = _number(reported.get(key))
        if number is not None:
            return number
    return None


def _bucket_ap(reported: Mapping[str, Any]) -> float | None:
    for key in _BUCKET_KEYS:
        number = _number(reported.get(key))
        if number is not None:
            return number
    rows = reported.get("per_class")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict) or row.get("name") != "bucket":
            continue
        if row.get("split") in {"test", "testing"}:
            continue
        number = _number(row.get("ap"))
        if number is not None:
            return number
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        return _number_text(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _number_text(value: object) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _read_library_metrics(output_dir: Path) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if not output_dir.is_dir():
        return found
    for path in output_dir.rglob("*.csv"):
        if path.is_symlink() or not path.is_file():
            continue
        _take_csv_metrics(path, found)
    return found


def _take_csv_metrics(path: Path, found: dict[str, Any]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return
        for row in reader:
            for key in ("val/mAP_50_95", "val/AP/bucket"):
                number = _number_text(row.get(key))
                if number is not None:
                    found[key] = number


def _publish_weights(output_dir: Path) -> Path | None:
    files = _checkpoint_files(output_dir)
    if not files:
        return None
    source = _prefer_checkpoint(files)
    target = output_dir / "weights.pth"
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    if not target.is_file() or target.stat().st_size <= 0:
        return None
    return target


def _checkpoint_files(output_dir: Path) -> list[Path]:
    found: list[Path] = []
    for path in output_dir.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix == ".pth" or path.name.endswith(".ckpt"):
            found.append(path)
    return found


def _prefer_checkpoint(files: list[Path]) -> Path:
    def rank(path: Path) -> tuple[int, str]:
        name = path.name
        if name == "weights.pth":
            return (0, name)
        if "best" in name and "total" in name:
            return (1, name)
        if "best" in name:
            return (2, name)
        return (3, name)

    return min(files, key=rank)


def _atomic_json(path: Path, payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".train-", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
