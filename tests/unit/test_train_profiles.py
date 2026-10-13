"""Smoke and pilot-small payloads. Does not call RF-DETR train."""

from __future__ import annotations

import json
from importlib.metadata import version

import pytest
from pydantic import ValidationError

from excavation_lab.errors import LabError
from excavation_lab.models.augment import EVAL_AUGMENTATION, MILD_PRESET
from excavation_lab.models.train_config import (
    resolve_train_profile,
    write_resolved_train_config,
)
from excavation_lab.schemas import pilot_path, read_yaml_mapping

_SMOKE_TRAIN_KEYS = {
    "epochs",
    "batch_size",
    "grad_accum_steps",
    "lr",
    "lr_encoder",
    "early_stopping",
    "skip_best_epochs",
    "num_workers",
    "amp_dtype",
    "seed",
    "run_test",
    "best_model_metric",
    "augmentation_train",
    "augmentation_eval",
    "aug_config",
    "scale_jitter",
    "multi_scale",
}
_PILOT_TRAIN_KEYS = _SMOKE_TRAIN_KEYS | {
    "early_stopping_patience",
    "early_stopping_min_delta",
}


def test_smoke_and_pilot_small_numbers() -> None:
    smoke = resolve_train_profile("smoke")
    pilot = resolve_train_profile("pilot-small")

    assert smoke.profile == "smoke"
    assert smoke.constructor["variant"] == "nano"
    assert smoke.constructor["class_name"] == "RFDETRNano"
    assert smoke.constructor["resolution"] == 384
    assert smoke.constructor["gradient_checkpointing"] is False
    assert smoke.train_kwargs["epochs"] == 2
    assert smoke.train_kwargs["batch_size"] == 1
    assert smoke.train_kwargs["grad_accum_steps"] == 1
    assert smoke.train_kwargs["early_stopping"] is False
    assert smoke.train_kwargs["skip_best_epochs"] == 0
    assert "early_stopping_patience" not in smoke.train_kwargs
    assert "early_stopping_min_delta" not in smoke.train_kwargs
    assert set(smoke.not_applied) == {
        "patience",
        "min_delta",
        "bucket_metric",
    }
    assert smoke.not_applied["patience"].value is None
    assert smoke.not_applied["min_delta"].value is None

    assert pilot.profile == "pilot-small"
    assert pilot.constructor["variant"] == "small"
    assert pilot.constructor["class_name"] == "RFDETRSmall"
    assert pilot.constructor["resolution"] == 512
    assert pilot.train_kwargs["epochs"] == 50
    assert pilot.train_kwargs["batch_size"] == 1
    assert pilot.train_kwargs["grad_accum_steps"] == 4
    assert pilot.train_kwargs["early_stopping"] is True
    assert pilot.train_kwargs["early_stopping_patience"] == 12
    assert pilot.train_kwargs["early_stopping_min_delta"] == 0.001
    assert pilot.train_kwargs["skip_best_epochs"] == 3
    assert "patience" not in pilot.not_applied
    assert "min_delta" not in pilot.not_applied

    for resolved in (smoke, pilot):
        assert resolved.rfdetr_version == version("rfdetr") == "1.11.2"
        assert resolved.run_test is False
        assert resolved.train_kwargs["lr"] == 1e-4
        assert resolved.train_kwargs["lr_encoder"] == 1e-5
        assert resolved.train_kwargs["num_workers"] == 0
        assert resolved.train_kwargs["seed"] == 42
        assert resolved.train_kwargs["run_test"] is False
        assert resolved.train_kwargs["best_model_metric"] == "map"
        assert resolved.train_kwargs["batch_size"] != "auto"
        assert type(resolved.train_kwargs["batch_size"]) is int
        assert resolved.ignored_keys == []
        assert resolved.constructor_kwargs() == {
            "resolution": resolved.constructor["resolution"],
            "gradient_checkpointing": False,
        }


def test_gradient_checkpointing_stays_on_the_constructor() -> None:
    raw = dict(read_yaml_mapping(pilot_path())["profiles"]["smoke"])
    raw["gradient_checkpointing"] = True
    resolved = resolve_train_profile("smoke", profile=raw)

    assert resolved.constructor["gradient_checkpointing"] is True
    assert resolved.constructor["variant"] == "nano"
    assert resolved.constructor["resolution"] == 384
    assert "gradient_checkpointing" not in resolved.train_kwargs
    assert "gradient_checkpointing" not in resolved.library_train_kwargs()
    assert "gradient_checkpointing" not in resolved.validation_kwargs
    assert "gradient_checkpointing" not in resolved.test_kwargs
    assert "variant" not in resolved.train_kwargs
    assert "resolution" not in resolved.train_kwargs


@pytest.mark.parametrize("name", ["smoke", "pilot-small"])
def test_amp_dtype_is_fp32_null(name: str) -> None:
    resolved = resolve_train_profile(name)
    assert resolved.train_kwargs["amp_dtype"] is None
    assert resolved.train_kwargs["amp_dtype"] not in {"fp16", "bf16", "fp8", "auto"}


@pytest.mark.parametrize("precision", ["fp16", "bf16", "fp8", "auto"])
def test_mixed_precision_profile_is_rejected(precision: str) -> None:
    raw = dict(read_yaml_mapping(pilot_path())["profiles"]["pilot-small"])
    raw["amp_dtype"] = precision
    with pytest.raises(ValidationError):
        resolve_train_profile("pilot-small", profile=raw)


def test_cuda_only_and_export_are_unset() -> None:
    for name in ("smoke", "pilot-small"):
        resolved = resolve_train_profile(name)
        payloads = (
            resolved.constructor,
            resolved.train_kwargs,
            resolved.library_train_kwargs(),
            resolved.validation_kwargs,
            resolved.test_kwargs,
        )
        for payload in payloads:
            assert payload.get("compile") is not True
            assert payload.get("cuda_graphs") is not True
            assert "compile" not in payload
            assert "cuda_graphs" not in payload
            assert "onnx" not in payload
            assert "coreml" not in payload
            assert "fp8" not in payload.values()
            assert "fp16" not in payload.values()
            assert "bf16" not in payload.values()
            if "amp_dtype" in payload:
                assert payload["amp_dtype"] is None


def test_eval_augmentation_is_not_the_train_preset() -> None:
    for name in ("smoke", "pilot-small"):
        resolved = resolve_train_profile(name)
        assert resolved.train_kwargs["augmentation_train"] == MILD_PRESET
        assert resolved.train_kwargs["augmentation_eval"] == EVAL_AUGMENTATION
        assert resolved.train_kwargs["aug_config"] == {}
        assert resolved.train_kwargs["scale_jitter"] is False
        assert resolved.train_kwargs["multi_scale"] is False
        assert set(resolved.train_kwargs) == (
            _SMOKE_TRAIN_KEYS if name == "smoke" else _PILOT_TRAIN_KEYS
        )
        for payload in (resolved.validation_kwargs, resolved.test_kwargs):
            assert payload == {
                "augmentation_eval": EVAL_AUGMENTATION,
                "aug_config": {},
                "scale_jitter": False,
                "multi_scale": False,
            }
            assert "augmentation_train" not in payload
            assert "mild" not in payload.values()
            assert "mild" not in json.dumps(payload)
        library = resolved.library_train_kwargs()
        assert "augmentation_train" not in library
        assert "augmentation_eval" not in library
        assert library["aug_config"] == {}
        assert library["scale_jitter"] is False
        assert library["multi_scale"] is False
        assert "aug_config" not in resolved.not_applied
        dumped = json.dumps(resolved.model_dump(mode="json"))
        assert "no library preset is selected" not in dumped
        assert "aug_config is unset" not in dumped


def test_unknown_yaml_keys_are_not_forwarded(tmp_path) -> None:
    raw = dict(read_yaml_mapping(pilot_path())["profiles"]["smoke"])
    raw["epoch"] = 99
    raw["demo_metric"] = 0
    raw["wandb"] = True
    raw["test_dir"] = "/tmp/dataset/test"
    raw["augmentation_train"] = "mild"
    raw["augmentation_eval"] = "deterministic"
    resolved = resolve_train_profile("smoke", profile=raw)

    assert resolved.ignored_keys == ["demo_metric", "epoch", "test_dir", "wandb"]
    assert resolved.train_kwargs["epochs"] == 2
    assert "augmentation_train" not in resolved.ignored_keys
    assert "augmentation_eval" not in resolved.ignored_keys
    forwarded = {
        **resolved.constructor,
        **resolved.train_kwargs,
        **resolved.validation_kwargs,
        **resolved.test_kwargs,
        **resolved.library_train_kwargs(),
    }
    assert "demo_metric" not in forwarded
    assert "wandb" not in forwarded
    assert "epoch" not in forwarded
    assert "test_dir" not in forwarded
    assert forwarded["augmentation_train"] == MILD_PRESET

    path = tmp_path / "resolved_train_config.json"
    write_resolved_train_config(path, resolved)
    text = path.read_text(encoding="utf-8")
    assert "/tmp/dataset/test" not in text
    assert "test_dir" in text


def test_resolved_json_has_run_test_false_and_no_test_path(tmp_path) -> None:
    resolved = resolve_train_profile("pilot-small")
    path = tmp_path / "resolved_train_config.json"
    write_resolved_train_config(path, resolved)
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["schema_version"] == 1
    assert document["run_test"] is False
    assert document["train_kwargs"]["run_test"] is False
    assert document["train_kwargs"]["epochs"] == 50
    assert document["train_kwargs"]["amp_dtype"] is None
    assert "dataset_dir" not in document["train_kwargs"]
    assert "dataset_dir" not in document["test_kwargs"]
    assert "dataset_dir" not in document["validation_kwargs"]
    assert set(document) == {
        "constructor",
        "ignored_keys",
        "not_applied",
        "profile",
        "rfdetr_version",
        "run_test",
        "schema_version",
        "test_kwargs",
        "train_kwargs",
        "validation_kwargs",
    }
    assert "dataset/test" not in path.read_text(encoding="utf-8")


def test_batch_size_auto_is_rejected() -> None:
    raw = dict(read_yaml_mapping(pilot_path())["profiles"]["smoke"])
    raw["batch_size"] = "auto"
    with pytest.raises(LabError, match="auto") as raised:
        resolve_train_profile("smoke", profile=raw)
    assert raised.value.code == "batch_size_auto"


def test_library_kwargs_fit_installed_configs_without_calling_train(tmp_path) -> None:
    from rfdetr.config import (
        MultiScale,
        RFDETRNanoConfig,
        RFDETRSmallConfig,
        TrainConfig,
        _resolve_amp_dtype,
    )

    smoke = resolve_train_profile("smoke")
    pilot = resolve_train_profile("pilot-small")
    dataset_dir = tmp_path / "dataset_root"

    smoke_model = RFDETRNanoConfig(**smoke.constructor_kwargs())
    pilot_model = RFDETRSmallConfig(**pilot.constructor_kwargs())
    assert smoke_model.resolution == 384
    assert pilot_model.resolution == 512
    assert smoke_model.gradient_checkpointing is False
    assert pilot_model.gradient_checkpointing is False
    assert smoke_model.compile is False
    assert smoke_model.cuda_graphs is False
    assert pilot_model.compile is False
    assert pilot_model.cuda_graphs is False

    smoke_train = TrainConfig(dataset_dir=dataset_dir, **smoke.library_train_kwargs())
    pilot_train = TrainConfig(dataset_dir=dataset_dir, **pilot.library_train_kwargs())
    assert smoke_train.epochs == 2
    assert smoke_train.grad_accum_steps == 1
    assert smoke_train.skip_best_epochs == 0
    assert smoke_train.early_stopping is False
    assert pilot_train.epochs == 50
    assert pilot_train.grad_accum_steps == 4
    assert pilot_train.early_stopping_patience == 12
    assert pilot_train.early_stopping_min_delta == 0.001
    assert pilot_train.skip_best_epochs == 3
    for train_config, model_config in (
        (smoke_train, smoke_model),
        (pilot_train, pilot_model),
    ):
        assert train_config.batch_size == 1
        assert train_config.lr == 1e-4
        assert train_config.lr_encoder == 1e-5
        assert train_config.num_workers == 0
        assert train_config.seed == 42
        assert train_config.run_test is False
        assert train_config.amp_dtype is None
        assert train_config.aug_config == {}
        assert train_config.scale_jitter is False
        assert train_config.multi_scale == MultiScale.OFF
        assert train_config.best_model_metric == "map"
        assert _resolve_amp_dtype(model_config, train_config) is None


def test_resolve_does_not_call_model_train(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("model.train was called")

    monkeypatch.setattr("rfdetr.detr.RFDETR.train", fail)
    smoke = resolve_train_profile("smoke")
    pilot = resolve_train_profile("pilot-small")
    assert smoke.train_kwargs["epochs"] == 2
    assert pilot.train_kwargs["epochs"] == 50
