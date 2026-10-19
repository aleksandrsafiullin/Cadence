"""Class-head rewrite. Tiny tensors only. No RF-DETR constructor and no train."""

from __future__ import annotations

import hashlib
import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from excavation_lab import jobs, storage
from excavation_lab.datasets.catalog import catalog_from_seed, list_catalogs
from excavation_lab.datasets.coco_import import build_class_mapping
from excavation_lab.errors import LabError, UnknownCategoryError
from excavation_lab.models import train as train_mod
from excavation_lab.models.adapter import CheckpointOrigin
from excavation_lab.models.bundle import (
    CLASS_NAMES,
    ModelBundleSpec,
    _mapping_class_names,
    load_bundle,
    write_model_bundle,
)
from excavation_lab.models.class_head import rewrite_class_head
from excavation_lab.models.evaluate import _mapping_names, _require_known_name
from excavation_lab.models.train_config import resolve_train_profile

_FOUR = ["excavator_body", "bucket", "haul_truck", "truck_bed"]
_NON_HEAD = "bbox_embed.layers.0.weight"
_ENC_WEIGHT = "transformer.enc_out_class_embed.0.weight"
_ENC_BIAS = "transformer.enc_out_class_embed.0.bias"
_HASH = "ab" * 32


def _checkpoint(
    foreground: int,
    labels: list[str] | None,
    *,
    include_encoder: bool = True,
) -> dict[str, object]:
    count = foreground + 1
    weight = torch.arange(count * 2, dtype=torch.float32).reshape(count, 2) + 10
    bias = torch.arange(count, dtype=torch.float32) + 100
    model: dict[str, torch.Tensor] = {
        "class_embed.weight": weight,
        "class_embed.bias": bias,
        _NON_HEAD: torch.tensor([[1.25, -2.5], [3.5, 4.75]]),
    }
    if include_encoder:
        model[_ENC_WEIGHT] = weight + 1000
        model[_ENC_BIAS] = bias + 7
    checkpoint: dict[str, object] = {"model": model}
    if labels is not None:
        checkpoint["args"] = {"class_names": list(labels), "num_classes": len(labels)}
    return checkpoint


def _save(path: Path, checkpoint: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _load(path: Path) -> dict[str, object]:
    loaded = torch.load(path, weights_only=True)
    assert isinstance(loaded, dict)
    return loaded


def _model(checkpoint: dict[str, object]) -> dict[str, torch.Tensor]:
    model = checkpoint["model"]
    assert isinstance(model, dict)
    return model


def test_excavation_four_class_rows_stay_on_their_names(tmp_path: Path) -> None:
    source = tmp_path / "source.pth"
    destination = tmp_path / "run" / "class_head.pth"
    checkpoint = _checkpoint(4, _FOUR)
    _save(source, checkpoint)
    before = source.read_bytes()
    rewrite_class_head(source, destination, _FOUR, workspace=tmp_path)
    assert source.read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == hashlib.sha256(source.read_bytes()).hexdigest()
    loaded = _load(destination)
    original = _model(checkpoint)
    written = _model(loaded)
    assert torch.equal(written[_NON_HEAD], original[_NON_HEAD])
    for key in ("class_embed.weight", "class_embed.bias", _ENC_WEIGHT, _ENC_BIAS):
        assert torch.equal(written[key], original[key])
    assert written["class_embed.bias"].shape[0] == len(_FOUR) + 1
    args = loaded["args"]
    assert isinstance(args, dict)
    assert args["class_names"] == _FOUR
    assert loaded.get("origin") != "custom"


def test_reordered_rows_follow_catalog_ids(tmp_path: Path) -> None:
    source_names = ["bucket", "excavator_body", "haul_truck", "truck_bed"]
    source = tmp_path / "source.pth"
    destination = tmp_path / "class_head.pth"
    checkpoint = _checkpoint(4, source_names)
    _save(source, checkpoint)
    rewrite_class_head(source, destination, _FOUR, workspace=tmp_path)
    written = _model(_load(destination))
    original = _model(checkpoint)
    for key in ("class_embed.weight", "class_embed.bias", _ENC_WEIGHT, _ENC_BIAS):
        dest = written[key]
        src = original[key]
        assert torch.equal(dest[0], src[1])
        assert torch.equal(dest[1], src[0])
        assert torch.equal(dest[2], src[2])
        assert torch.equal(dest[3], src[3])
        assert torch.equal(dest[-1], src[-1])
    assert torch.equal(written[_NON_HEAD], original[_NON_HEAD])


def test_five_class_expansion_keeps_old_rows_and_adds_shovel(tmp_path: Path) -> None:
    catalog = [*_FOUR, "shovel"]
    source = tmp_path / "source.pth"
    destination = tmp_path / "class_head.pth"
    checkpoint = _checkpoint(4, _FOUR)
    _save(source, checkpoint)
    rewrite_class_head(source, destination, catalog, workspace=tmp_path)
    written = _model(_load(destination))
    original = _model(checkpoint)
    for key in ("class_embed.weight", "class_embed.bias", _ENC_WEIGHT, _ENC_BIAS):
        dest = written[key]
        src = original[key]
        assert dest.shape[0] == len(catalog) + 1
        for index in range(len(_FOUR)):
            assert torch.equal(dest[index], src[index])
            assert not torch.equal(dest[index], torch.zeros_like(dest[index]))
        assert torch.equal(dest[-1], src[-1])
        shovel = dest[len(_FOUR)]
        for row in range(src.shape[0]):
            assert not torch.equal(shovel, src[row])
    assert torch.equal(written[_NON_HEAD], original[_NON_HEAD])


def test_coco_file_copies_background_and_not_class_zero(tmp_path: Path) -> None:
    source = tmp_path / "coco.pth"
    destination = tmp_path / "class_head.pth"
    checkpoint = _checkpoint(2, ["person", "bicycle"])
    _save(source, checkpoint)
    before = source.read_bytes()
    rewrite_class_head(source, destination, _FOUR, workspace=tmp_path)
    assert source.read_bytes() == before
    written = _model(_load(destination))
    original = _model(checkpoint)
    dest = written["class_embed.weight"]
    src = original["class_embed.weight"]
    assert dest.shape[0] == len(_FOUR) + 1
    assert torch.equal(dest[-1], src[-1])
    assert not torch.equal(dest[0], src[0])
    for row in dest[:-1]:
        for index in range(src.shape[0]):
            assert not torch.equal(row, src[index])
    assert torch.equal(written[_NON_HEAD], original[_NON_HEAD])
    assert torch.equal(written[_ENC_WEIGHT][-1], original[_ENC_WEIGHT][-1])
    assert not torch.equal(written[_ENC_WEIGHT][0], original[_ENC_WEIGHT][0])

    nameless = tmp_path / "nameless.pth"
    plain = _checkpoint(2, None, include_encoder=False)
    _save(nameless, plain)
    nameless_out = tmp_path / "nameless-out.pth"
    rewrite_class_head(nameless, nameless_out, _FOUR, workspace=tmp_path)
    loaded = _model(_load(nameless_out))
    assert _ENC_WEIGHT not in loaded
    plain_weight = _model(plain)["class_embed.weight"]
    assert torch.equal(loaded["class_embed.weight"][-1], plain_weight[-1])
    assert not torch.equal(loaded["class_embed.weight"][0], plain_weight[0])


def test_rewrite_does_not_edit_source_or_the_official_cache(tmp_path: Path) -> None:
    source = tmp_path / "source.pth"
    checkpoint = _checkpoint(4, _FOUR)
    _save(source, checkpoint)
    before = source.read_bytes()
    cache_dest = tmp_path / "cache" / "weights" / "class_head.pth"
    with pytest.raises(LabError) as caught:
        rewrite_class_head(source, cache_dest, _FOUR, workspace=tmp_path)
    assert caught.value.code == "baseline_weights"
    assert not cache_dest.exists()
    assert source.read_bytes() == before
    with pytest.raises(LabError) as in_place:
        rewrite_class_head(source, source, _FOUR, workspace=tmp_path)
    assert in_place.value.code == "class_head"
    assert source.read_bytes() == before


def test_state_dict_keeps_non_head_tensor(tmp_path: Path) -> None:
    weight = torch.tensor([[10.0, 11.0], [20.0, 21.0], [90.0, 91.0]])
    bias = torch.tensor([1.0, 2.0, 9.0])
    backbone = torch.tensor([[4.0, 5.0]])
    source = tmp_path / "state.pth"
    _save(
        source,
        {
            "state_dict": {
                "model.class_embed.weight": weight,
                "model.class_embed.bias": bias,
                "model.bbox_embed.layers.0.weight": backbone,
            },
            "args": {"class_names": ["bucket", "excavator_body"]},
        },
    )
    destination = tmp_path / "class_head.pth"
    rewrite_class_head(
        source,
        destination,
        ["excavator_body", "bucket"],
        workspace=tmp_path,
    )
    loaded = _load(destination)
    state = loaded["state_dict"]
    assert isinstance(state, dict)
    written = state["model.class_embed.weight"]
    assert torch.equal(written[0], weight[1])
    assert torch.equal(written[1], weight[0])
    assert torch.equal(written[-1], weight[-1])
    assert torch.equal(state["model.bbox_embed.layers.0.weight"], backbone)


def test_constructor_width_matches_catalog_and_run_test_stays_false() -> None:
    resolved = resolve_train_profile("smoke")
    payload = {
        "pretrain_weights": "/tmp/run/class_head.pth",
        "num_classes": 5,
        "device": "cpu",
        "batch_size": 1,
        "run_test": False,
    }
    constructor = train_mod._constructor_kwargs(resolved, payload)
    library = train_mod._library_kwargs(payload)
    assert constructor["pretrain_weights"] == "/tmp/run/class_head.pth"
    assert constructor["num_classes"] == 5
    assert library["run_test"] is False
    assert "pretrain_weights" not in library
    assert "num_classes" not in library
    text = Path(train_mod.__file__).read_text(encoding="utf-8")
    assert "reinitialize_detection_head" not in text
    assert "model.model" not in text


def test_fake_trainer_records_the_rewritten_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    dataset = workspace / "dataset"
    _write_split(dataset / "train")
    _write_split(dataset / "valid")
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            """
            INSERT INTO dataset_versions (
                dataset_id, project_id, content_hash, path, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("ds1", "pilot", _HASH, str(dataset), datetime.now(UTC).isoformat()),
        )
    source_names = ["bucket", "excavator_body", "haul_truck", "truck_bed"]
    cache = workspace / "cache" / "weights" / "rf-detr-nano.pth"
    checkpoint = _checkpoint(4, source_names)
    _save(cache, checkpoint)
    before = cache.read_bytes()
    monkeypatch.setenv(train_mod.FAKE_TRAINER_ENV, "fail")
    monkeypatch.setattr(jobs, "_spawn", _dummy_spawn)
    started = train_mod.enqueue_train(workspace, "pilot", "ds1", "smoke")
    assert train_mod.run_train_job(workspace, started.job_id) == 1
    run_dir = jobs.run_directory(workspace, "pilot", started.job_id)
    calls_path = run_dir / "trainer_calls.json"
    if not calls_path.is_file():
        log_path = run_dir / "train.log"
        log = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        raise AssertionError(log)
    calls = json.loads(calls_path.read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert calls[0]["model_constructed"] is False
    kwargs = calls[0]["kwargs"]
    assert kwargs["run_test"] is False
    assert kwargs["num_classes"] == 4
    recorded = Path(kwargs["pretrain_weights"])
    assert recorded.name == "class_head.pth"
    assert recorded.parent == run_dir.resolve()
    assert cache.resolve() not in recorded.resolve().parents
    assert cache.read_bytes() == before
    loaded = _model(_load(recorded))
    original = _model(checkpoint)
    assert torch.equal(loaded[_NON_HEAD], original[_NON_HEAD])
    weight = loaded["class_embed.weight"]
    source_weight = original["class_embed.weight"]
    assert weight.shape[0] == kwargs["num_classes"] + 1
    assert torch.equal(weight[0], source_weight[1])
    assert torch.equal(weight[1], source_weight[0])
    assert torch.equal(weight[-1], source_weight[-1])
    assert _load(recorded).get("origin") != "custom"
    assert not (workspace / "weights").exists()


def test_five_name_mapping_loads_and_unknown_name_is_not_glued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = [*_FOUR, "shovel"]
    mapping = build_class_mapping(catalog_from_seed())
    mapping["by_name"]["shovel"] = {"internal_id": 4, "coco_category_id": 4}
    bundle = write_model_bundle(
        tmp_path / "five",
        _bundle_spec(_bundle_bytes(names), mapping),
    )
    stored = json.loads((bundle / "class_mapping.json").read_text(encoding="utf-8"))
    assert stored["class_names"] == names
    assert stored["class_names"] != sorted(names)
    assert _mapping_class_names(stored) == tuple(names)
    loaded_names = _mapping_names(stored)
    assert loaded_names["shovel"] == 4
    assert loaded_names["excavator_body"] == 0
    assert list(loaded_names) == names
    assert _require_known_name("shovel", loaded_names, "frame-1") == "shovel"
    with pytest.raises(UnknownCategoryError) as unknown:
        _require_known_name("widget", loaded_names, "frame-1")
    assert unknown.value.class_id is None
    assert "widget" in unknown.value.message
    assert "bucket" not in unknown.value.message

    missing = json.loads(json.dumps(stored))
    del missing["by_name"]["shovel"]
    with pytest.raises(LabError) as absent:
        _mapping_names(missing)
    assert absent.value.code == "class_order"
    assert "shovel" in absent.value.message

    extra = json.loads(json.dumps(stored))
    extra["by_name"]["widget"] = {"internal_id": 1, "coco_category_id": 1}
    with pytest.raises(LabError) as glued:
        _mapping_names(extra)
    assert glued.value.code == "class_order"
    assert "widget" in glued.value.message

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("constructed an RF-DETR model")

    import rfdetr

    monkeypatch.setattr(rfdetr, "RFDETRNano", _boom)
    monkeypatch.setattr(rfdetr, "RFDETRSmall", _boom)
    adapter = load_bundle(bundle)
    assert list(adapter.metadata().class_names) == names
    assert adapter.metadata().origin == CheckpointOrigin.custom
    config = json.loads((bundle / "model_config.json").read_text(encoding="utf-8"))
    assert config["class_names"] == names
    assert config["origin"] == "custom"


def _bundle_bytes(names: list[str]) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "model": {"probe": torch.zeros(1)},
            "args": {"class_names": list(names), "num_classes": len(names)},
            "model_name": "RFDETRNano",
            "model_config": {"resolution": 384},
        },
        buffer,
    )
    return buffer.getvalue()


def _bundle_spec(weights: bytes, mapping: dict[str, object]) -> ModelBundleSpec:
    return ModelBundleSpec(
        variant="nano",
        resolution=384,
        dataset_hash=_HASH,
        library_versions={"rfdetr": "1.11.2", "torch": "2.14.1"},
        resolved_train_config={"schema_version": 1, "profile": "smoke", "run_test": False},
        dataset_manifest={
            "schema_version": 1,
            "dataset_id": "ds1",
            "project_id": "pilot",
            "dataset_hash": _HASH,
        },
        environment_report={"schema_version": 1, "inference_device": "cpu"},
        validation_metrics={"schema_version": 1, "box_map_50_95": None},
        weights=weights,
        class_mapping=mapping,
    )


def _write_split(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "frame.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (directory / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "annotations": [],
                "categories": [],
                "images": [{"id": 1, "file_name": "frame.jpg", "width": 8, "height": 8}],
            }
        ),
        encoding="utf-8",
    )


class _DummyProcess:
    def __init__(self) -> None:
        self.pid = os.getpid()

    def poll(self) -> None:
        return None


def _dummy_spawn(*_args: object, **_kwargs: object) -> _DummyProcess:
    return _DummyProcess()


def test_class_names_constant_is_the_four_excavation_names() -> None:
    assert list(CLASS_NAMES) == _FOUR
