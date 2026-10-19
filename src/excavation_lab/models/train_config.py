"""Translate a pilot profile into RF-DETR constructor and train payloads.

``RFDETR.train`` is ``train(self, **kwargs)``. ``_prepare_run_config`` pops
``device`` and ``resolution``, then builds ``TrainConfig``. ``TrainConfig``
rejects unknown keys. This module maps project names onto that signature.
It does not call ``train`` and it does not splat the YAML mapping into
``train``.

``gradient_checkpointing`` is a ``ModelConfig`` field. It stays on the
constructor. ``amp_dtype=None`` is FP32. The installed default is ``"auto"``,
which is not what we send.

``augmentation_train`` is the mild preset body from ``models/augment.py``.
``augmentation_eval`` is deterministic preprocess plus the library off
switch. Those two names are project keys, not ``TrainConfig`` fields.
Installed defaults are ``aug_config=None`` (torchvision
``RandomHorizontalFlip(p=0.5)``), ``scale_jitter=True``, and
``multi_scale=per-batch``. This module sends ``aug_config={}``,
``scale_jitter=False``, and ``multi_scale=False`` on the train payload and
on the validation and test payloads. ``augment.py`` applies the mild preset
for the train split. Validation and test stay on ``model_rgb_frame``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from excavation_lab.errors import LabError
from excavation_lab.models.augment import (
    EVAL_AUGMENTATION,
    MILD_PRESET,
    eval_augmentation,
    library_off_switch,
    mild_preset,
)
from excavation_lab.schemas import (
    PROFILE_OVERFIT,
    PROFILE_PILOT_SMALL,
    PROFILE_SMOKE,
    SCHEMA_VERSION,
    ModelVariant,
    StrictModel,
    TrainProfile,
    pilot_path,
    read_yaml_mapping,
)

AUGMENTATION_TRAIN_KEY = "augmentation_train"
AUGMENTATION_EVAL_KEY = "augmentation_eval"
AUGMENTATION_TRAIN = "mild"
AUGMENTATION_EVAL = "deterministic"
METRIC_ROLE_DIAGNOSTIC_OVERFIT = "diagnostic_overfit"
METRIC_ROLE_SYNTHETIC_WIRING = "synthetic_wiring"
# Smoke network fields. Overfit may change epochs and must set diagnostic.
_SMOKE_NETWORK_KEYS = (
    "variant",
    "resolution",
    "batch_size",
    "gradient_accumulation",
    "learning_rate",
    "encoder_learning_rate",
    "early_stopping",
    "patience",
    "min_delta",
    "best_skip",
    "workers",
    "amp_dtype",
    "seed",
    "run_test",
)

# Project YAML name -> TrainConfig field. Names that differ are intentional.
_TRAIN_FIELD_MAP: dict[str, str] = {
    "epochs": "epochs",
    "batch_size": "batch_size",
    "gradient_accumulation": "grad_accum_steps",
    "learning_rate": "lr",
    "encoder_learning_rate": "lr_encoder",
    "early_stopping": "early_stopping",
    "patience": "early_stopping_patience",
    "min_delta": "early_stopping_min_delta",
    "best_skip": "skip_best_epochs",
    "workers": "num_workers",
    "amp_dtype": "amp_dtype",
    "seed": "seed",
    "run_test": "run_test",
}
# Null in YAML when early stopping is off. The library fields are ints.
_NULL_WHEN_EARLY_STOP_OFF = frozenset({"patience", "min_delta"})
_PROFILE_VARIANT = {
    PROFILE_SMOKE: ModelVariant.nano,
    PROFILE_PILOT_SMALL: ModelVariant.small,
    PROFILE_OVERFIT: ModelVariant.nano,
}
_VARIANT_CLASS = {
    ModelVariant.nano.value: "RFDETRNano",
    ModelVariant.small.value: "RFDETRSmall",
}
_VARIANT_SIZE = {
    ModelVariant.nano.value: "rfdetr-nano",
    ModelVariant.small.value: "rfdetr-small",
}
# Detection box mAP50:95. Trainer monitors val/mAP_50_95 for this value.
_BOX_MAP_METRIC = "map"
_TEST_PATH_KEYS = frozenset(
    {
        "test_dataset",
        "test_dataset_dir",
        "test_dir",
        "test_path",
        "dataset_test",
    }
)
_DISABLED_STRINGS = frozenset({"fp8", "fp16", "bf16", "auto"})


class NotApplied(StrictModel):
    """A spec value that is not sent as a train kwarg."""

    value: Any = None
    reason: str = Field(min_length=1)


class ResolvedTrainConfig(StrictModel):
    """One profile, split into constructor and train payloads."""

    schema_version: int = SCHEMA_VERSION
    profile: Literal["smoke", "pilot-small", "overfit"]
    rfdetr_version: str = Field(min_length=1)
    run_test: Literal[False] = False
    diagnostic: bool = False
    constructor: dict[str, Any]
    train_kwargs: dict[str, Any]
    validation_kwargs: dict[str, Any]
    test_kwargs: dict[str, Any]
    ignored_keys: list[str]
    not_applied: dict[str, NotApplied]

    def constructor_kwargs(self) -> dict[str, Any]:
        """Kwargs accepted by the variant config. Excludes the class selector."""
        fields = _installed_train_api().model_fields
        return {key: value for key, value in self.constructor.items() if key in fields}

    def library_train_kwargs(self) -> dict[str, Any]:
        """``TrainConfig`` fields only.

        ``augmentation_train`` and ``augmentation_eval`` remain on
        ``train_kwargs``. They are not fields of the installed train config.
        ``aug_config``, ``scale_jitter``, and ``multi_scale`` are included:
        they are the explicit off switch. ``gradient_checkpointing`` is not
        included.
        """
        fields = _installed_train_api().train_fields
        return {key: value for key, value in self.train_kwargs.items() if key in fields}


@dataclass(frozen=True, slots=True)
class _InstalledTrainApi:
    version: str
    train_fields: frozenset[str]
    model_fields: frozenset[str]
    resolution_block: dict[str, int]
    patience_default: object
    min_delta_default: object


@lru_cache(maxsize=1)
def _installed_train_api() -> _InstalledTrainApi:
    """Read TrainConfig and the Nano/Small classes. Does not construct a model."""
    try:
        from rfdetr import variants
        from rfdetr.config import (
            ModelConfig,
            RFDETRNanoConfig,
            RFDETRSmallConfig,
            TrainConfig,
        )
    except ImportError as exc:
        raise LabError(
            "rfdetr_missing",
            "Пакет rfdetr не установлен. Профиль обучения не собран.",
        ) from exc

    installed = version("rfdetr")
    blocks: dict[str, int] = {}
    for variant, config_cls in (
        (ModelVariant.nano.value, RFDETRNanoConfig),
        (ModelVariant.small.value, RFDETRSmallConfig),
    ):
        class_name = _VARIANT_CLASS[variant]
        cls = getattr(variants, class_name, None)
        if not isinstance(cls, type) or getattr(cls, "size", None) != _VARIANT_SIZE[variant]:
            raise LabError(
                "rfdetr_variant",
                f"В установленном rfdetr нет класса {class_name} для варианта {variant}.",
            )
        train_cls = getattr(cls, "_train_config_class", None)
        if train_cls is not TrainConfig:
            raise LabError(
                "rfdetr_variant",
                f"{class_name} использует другой train-конфиг, не TrainConfig.",
            )
        patch = int(config_cls.model_fields["patch_size"].default)
        windows = int(config_cls.model_fields["num_windows"].default)
        blocks[variant] = patch * windows

    return _InstalledTrainApi(
        version=installed,
        train_fields=frozenset(TrainConfig.model_fields),
        model_fields=frozenset(ModelConfig.model_fields),
        resolution_block=blocks,
        patience_default=TrainConfig.model_fields["early_stopping_patience"].default,
        min_delta_default=TrainConfig.model_fields["early_stopping_min_delta"].default,
    )


def resolve_train_profile(
    name: str,
    *,
    profile: Mapping[str, Any] | None = None,
    pilot_file: Path | None = None,
) -> ResolvedTrainConfig:
    """Build ``smoke``, ``pilot-small``, or the ``overfit`` diagnostic.

    ``profile`` replaces the ``profiles.<name>`` mapping. Keys this module
    does not recognize are listed on ``ignored_keys`` and dropped.
    ``overfit`` reuses the smoke network and records its own epoch count.
    """
    if name not in _PROFILE_VARIANT:
        raise LabError("unknown_profile", f"Неизвестный профиль обучения: {name}")

    raw = _raw_profile(name, profile=profile, pilot_file=pilot_file)
    if name == PROFILE_OVERFIT:
        _require_smoke_network(raw, pilot_file=pilot_file)
    recognized = _recognized_keys()
    ignored = sorted(key for key in raw if key not in recognized)
    known = {key: raw[key] for key in raw if key in recognized}

    aug_train = _coerce_train_augmentation(
        known.pop(AUGMENTATION_TRAIN_KEY, AUGMENTATION_TRAIN),
    )
    aug_eval = _coerce_eval_augmentation(
        known.pop(AUGMENTATION_EVAL_KEY, AUGMENTATION_EVAL),
    )
    if aug_train == aug_eval:
        raise LabError(
            "eval_uses_train_preset",
            "Validation и test получили train-пресет аугментации. "
            "Для них остаётся только детерминированный режим.",
        )

    gradient_checkpointing = known.pop("gradient_checkpointing", False)
    if not isinstance(gradient_checkpointing, bool):
        raise LabError(
            "config_invalid",
            "gradient_checkpointing должен быть логическим значением.",
        )

    _reject_auto_batch(known.get("batch_size"))
    parsed = TrainProfile.model_validate(known)
    if parsed.variant is not _PROFILE_VARIANT[name]:
        raise LabError(
            "profile_variant",
            f"Профиль {name} ожидает вариант {_PROFILE_VARIANT[name].value}, "
            f"в YAML указан {parsed.variant.value}.",
        )
    if parsed.run_test:
        raise LabError(
            "run_test_enabled",
            "Профиль включает test во время обучения. "
            "Стартовые профили держат run_test выключенным.",
        )
    diagnostic = name == PROFILE_OVERFIT
    if diagnostic and not parsed.diagnostic:
        raise LabError(
            "config_invalid",
            "Профиль overfit помечается diagnostic: true и не входит в таблицу качества.",
        )
    if not diagnostic and parsed.diagnostic:
        raise LabError(
            "config_invalid",
            "diagnostic: true допустим только у профиля overfit.",
        )

    api = _installed_train_api()
    _check_resolution(api, parsed.variant.value, parsed.resolution)
    not_applied: dict[str, NotApplied] = {}
    train_kwargs: dict[str, Any] = {}
    for project_key, library_key in _TRAIN_FIELD_MAP.items():
        value = getattr(parsed, project_key)
        if project_key in _NULL_WHEN_EARLY_STOP_OFF and not parsed.early_stopping:
            not_applied[project_key] = _omitted_early_stop_field(
                api,
                library_key,
                value,
            )
            continue
        _assign_train_field(
            api,
            train_kwargs,
            not_applied,
            project_key,
            library_key,
            value,
        )

    if "best_model_metric" in api.train_fields:
        train_kwargs["best_model_metric"] = _BOX_MAP_METRIC
    else:
        not_applied["best_model_metric"] = NotApplied(
            value=_BOX_MAP_METRIC,
            reason=(
                f"rfdetr {api.version} TrainConfig has no best_model_metric. "
                "Box mAP50:95 was not renamed onto another argument. "
                "The profile epoch count is unchanged."
            ),
        )

    train_kwargs[AUGMENTATION_TRAIN_KEY] = aug_train
    train_kwargs[AUGMENTATION_EVAL_KEY] = aug_eval
    # aug_config=None selects torchvision RandomHorizontalFlip(p=0.5).
    # scale_jitter defaults to True. multi_scale defaults to per-batch.
    # The empty dict and the two false values are the installed off switch.
    train_kwargs.update(library_off_switch())
    not_applied["bucket_metric"] = NotApplied(
        value=None,
        reason=(
            f"rfdetr {api.version} best_model_metric accepts map or mar. "
            "map monitors val/mAP_50_95 for detection, which is box mAP50:95. "
            "There is no train argument for per-class bucket AP."
        ),
    )

    if "gradient_checkpointing" in train_kwargs:
        raise LabError(
            "train_kwargs_split",
            "gradient_checkpointing относится к конструктору и не входит в train kwargs.",
        )

    variant = parsed.variant.value
    constructor = {
        "variant": variant,
        "class_name": _VARIANT_CLASS[variant],
        "resolution": parsed.resolution,
        "gradient_checkpointing": gradient_checkpointing,
    }
    if "resolution" not in api.model_fields:
        not_applied["resolution"] = NotApplied(
            value=parsed.resolution,
            reason=(
                f"rfdetr {api.version} ModelConfig has no resolution. "
                "The profile value was not rewritten."
            ),
        )
        constructor.pop("resolution")
    if "gradient_checkpointing" not in api.model_fields:
        not_applied["gradient_checkpointing"] = NotApplied(
            value=gradient_checkpointing,
            reason=(
                f"rfdetr {api.version} ModelConfig has no gradient_checkpointing. "
                "It was not moved onto train kwargs."
            ),
        )
        constructor.pop("gradient_checkpointing")

    validation_kwargs = _eval_payload()
    test_kwargs = _eval_payload()
    _reject_disabled_options(constructor)
    _reject_disabled_options(train_kwargs)
    _reject_disabled_options(validation_kwargs)
    _reject_disabled_options(test_kwargs)

    return ResolvedTrainConfig(
        profile=name,  # type: ignore[arg-type]
        rfdetr_version=api.version,
        diagnostic=diagnostic,
        constructor=constructor,
        train_kwargs=train_kwargs,
        validation_kwargs=validation_kwargs,
        test_kwargs=test_kwargs,
        ignored_keys=ignored,
        not_applied=not_applied,
    )


def resolved_train_document(resolved: ResolvedTrainConfig) -> dict[str, Any]:
    """JSON-ready document. No test-dataset path.

    Smoke and pilot-small keep the previous keys. Overfit adds
    ``diagnostic`` and ``metric_role`` so the file is not a quality row.
    """
    document = resolved.model_dump(mode="json")
    diagnostic = bool(document.pop("diagnostic", False))
    document.pop("quality_claim", None)
    if diagnostic:
        document["diagnostic"] = True
        document["metric_role"] = METRIC_ROLE_DIAGNOSTIC_OVERFIT
    _reject_test_dataset_path(document)
    if document.get("quality_claim") is True or _contains_test_split(document):
        raise LabError(
            "quality_claim",
            "Диагностика overfit не записывается как качество модели.",
        )
    return document


def write_resolved_train_config(path: Path, resolved: ResolvedTrainConfig) -> None:
    """Write ``resolved_train_config.json`` for a run. Does not train."""
    document = resolved_train_document(resolved)
    text = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def _recognized_keys() -> frozenset[str]:
    return frozenset(TrainProfile.model_fields) | {
        AUGMENTATION_TRAIN_KEY,
        AUGMENTATION_EVAL_KEY,
        "gradient_checkpointing",
    }


def _require_smoke_network(raw: Mapping[str, Any], *, pilot_file: Path | None) -> None:
    """Overfit keeps the smoke network. Epochs are the one schedule change."""
    smoke = _raw_profile(PROFILE_SMOKE, profile=None, pilot_file=pilot_file)
    for key in _SMOKE_NETWORK_KEYS:
        if raw.get(key) != smoke.get(key):
            raise LabError(
                "config_invalid",
                "Профиль overfit повторяет сеть smoke "
                "(Nano, 384, FP32, run_test false). "
                f"Поле «{key}» не совпадает. Меняется только число эпох.",
            )


def _raw_profile(
    name: str,
    *,
    profile: Mapping[str, Any] | None,
    pilot_file: Path | None,
) -> dict[str, Any]:
    if profile is not None:
        if any(not isinstance(key, str) for key in profile):
            raise LabError(
                "config_invalid",
                "Ключи профиля должны быть строками.",
            )
        return dict(profile)
    document = read_yaml_mapping(pilot_file or pilot_path())
    profiles = document.get("profiles")
    if not isinstance(profiles, dict) or name not in profiles:
        raise LabError("unknown_profile", f"Неизвестный профиль обучения: {name}")
    raw = profiles[name]
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise LabError(
            "config_invalid",
            f"Профиль {name} должен быть YAML-объектом.",
        )
    return dict(raw)


def _eval_payload() -> dict[str, Any]:
    payload = {AUGMENTATION_EVAL_KEY: eval_augmentation()}
    payload.update(library_off_switch())
    return payload


def _coerce_train_augmentation(value: object) -> dict[str, Any]:
    if value == AUGMENTATION_TRAIN or value == MILD_PRESET:
        return mild_preset()
    raise LabError(
        "unknown_augmentation",
        "augmentation_train должен быть пресетом mild. "
        "Именованные пресеты библиотеки не подставляются.",
    )


def _coerce_eval_augmentation(value: object) -> dict[str, Any]:
    if (
        value == AUGMENTATION_EVAL
        or value == "deterministic_preprocess"
        or value == EVAL_AUGMENTATION
    ):
        return eval_augmentation()
    raise LabError(
        "unknown_augmentation",
        "augmentation_eval должен быть детерминированным режимом. "
        "Train-пресет на validation и test не ставится.",
    )


def _reject_auto_batch(raw_batch: object) -> None:
    if isinstance(raw_batch, bool) or raw_batch == "auto":
        raise LabError(
            "batch_size_auto",
            'batch_size="auto" не используется. Укажите целое число. '
            "Для CPU и MPS стартовое значение — 1.",
        )


def _check_resolution(api: _InstalledTrainApi, variant: str, resolution: int) -> None:
    block = api.resolution_block[variant]
    if resolution % block != 0:
        raise LabError(
            "resolution_block",
            f"resolution {resolution} не делится на patch_size * num_windows "
            f"({block}) для варианта {variant}.",
        )


def _omitted_early_stop_field(
    api: _InstalledTrainApi,
    library_key: str,
    value: object,
) -> NotApplied:
    if library_key == "early_stopping_patience":
        default = api.patience_default
    else:
        default = api.min_delta_default
    if library_key not in api.train_fields:
        reason = (
            f"rfdetr {api.version} TrainConfig has no {library_key}. "
            "The null profile value is unchanged, and epochs is not rewritten."
        )
    else:
        reason = (
            "early_stopping is false. "
            f"TrainConfig.{library_key} is not nullable "
            f"(installed default {default!r}). "
            "That default is omitted from train_kwargs. "
            "The profile epoch count is unchanged."
        )
    return NotApplied(value=value, reason=reason)


def _assign_train_field(
    api: _InstalledTrainApi,
    train_kwargs: dict[str, Any],
    not_applied: dict[str, NotApplied],
    project_key: str,
    library_key: str,
    value: object,
) -> None:
    if library_key in api.train_fields:
        train_kwargs[library_key] = value
        return
    not_applied[project_key] = NotApplied(
        value=value,
        reason=(
            f"rfdetr {api.version} TrainConfig has no {library_key}. "
            "The profile value was not renamed onto another argument, "
            "and epochs was not changed to compensate."
        ),
    )


def _reject_disabled_options(payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        lowered = key.lower()
        if "onnx" in lowered or "coreml" in lowered:
            raise LabError(
                "export_disabled",
                "ONNX и CoreML в профиль обучения v0.1 не входят.",
            )
        if key in {"compile", "cuda_graphs"} and value:
            raise LabError(
                "cuda_option_disabled",
                "torch.compile и CUDA graphs в профиль v0.1 не входят.",
            )
        if key == "amp_dtype" and value is not None:
            raise LabError(
                "amp_dtype",
                "Точность v0.1 — FP32: amp_dtype должен быть null.",
            )
        if isinstance(value, str) and value in _DISABLED_STRINGS:
            raise LabError(
                "precision_disabled",
                "FP8, FP16, BF16 и batch_size auto в профиль v0.1 не входят.",
            )
        if isinstance(value, Mapping):
            _reject_disabled_options(value)


def _contains_test_split(node: object) -> bool:
    """True when a document assigns ``split`` to the test split."""
    if isinstance(node, dict):
        if node.get("split") in {"test", "testing"}:
            return True
        return any(_contains_test_split(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_test_split(item) for item in node)
    return False


def _reject_test_dataset_path(node: object) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _TEST_PATH_KEYS:
                raise LabError(
                    "test_path_in_train_config",
                    "resolved_train_config содержит путь к test. "
                    "Каталог test в этот файл не пишется.",
                )
            if key == "dataset_dir" and isinstance(value, str) and _looks_like_test_dir(value):
                raise LabError(
                    "test_path_in_train_config",
                    "dataset_dir указывает на каталог test. "
                    "Train его не открывает.",
                )
            if isinstance(value, str) and _looks_like_test_dir(value):
                raise LabError(
                    "test_path_in_train_config",
                    "resolved_train_config содержит путь к каталогу test.",
                )
            _reject_test_dataset_path(value)
    elif isinstance(node, list):
        for item in node:
            _reject_test_dataset_path(item)


def _looks_like_test_dir(value: str) -> bool:
    if "/" not in value and "\\" not in value:
        return False
    normalized = value.replace("\\", "/").rstrip("/")
    return normalized == "test" or normalized.endswith("/test")
