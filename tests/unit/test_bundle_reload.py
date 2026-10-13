"""Reload weights.pth in-process and in a second interpreter.

A metadata-only state dict is not an RF-DETR graph. Class order and file
selection run here. The network constructor is not called. A real checkpoint
reload is marked manual and is not part of this run.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from excavation_lab.errors import (
    CheckpointMissingError,
    IncompatibleCheckpointError,
    LabError,
)
from excavation_lab.models import adapter as adapter_module
from excavation_lab.models import baseline as baseline_module
from excavation_lab.models import bundle as bundle_mod
from excavation_lab.models.adapter import DISPLAY_SCORE_THRESHOLD, CheckpointOrigin
from excavation_lab.models.baseline import LIBRARY_SCORE_THRESHOLD
from excavation_lab.models.bundle import (
    CLASS_NAMES,
    ModelBundleSpec,
    load_bundle,
    write_model_bundle,
)

_NAMES = ["excavator_body", "bucket", "haul_truck", "truck_bed"]
_CHILD = """
import json
import sys

from excavation_lab.models.bundle import load_bundle

adapter = load_bundle(sys.argv[1])
names = list(adapter.metadata().class_names)
print(json.dumps({
    "names": names,
    "bucket": names.index("bucket"),
    "streamlit": "streamlit" in sys.modules,
    "module": load_bundle.__module__,
}))
"""


def _cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)


_HASH = "ab" * 32


def _spec(**overrides: object) -> ModelBundleSpec:
    payload: dict[str, object] = {
        "variant": "nano",
        "resolution": 384,
        "dataset_hash": _HASH,
        "library_versions": {"rfdetr": "1.11.2", "torch": "2.14.1"},
        "resolved_train_config": {"schema_version": 1, "profile": "smoke", "run_test": False},
        "dataset_manifest": {
            "schema_version": 1,
            "dataset_id": "ds1",
            "project_id": "pilot",
            "dataset_hash": _HASH,
        },
        "environment_report": {"schema_version": 1, "inference_device": "cpu"},
        "validation_metrics": {"schema_version": 1, "box_map_50_95": None},
        "weights": _checkpoint_bytes(_NAMES),
    }
    payload.update(overrides)
    return ModelBundleSpec(**payload)  # type: ignore[arg-type]


def _checkpoint_bytes(
    class_names: list[str],
    *,
    model_name: str = "RFDETRNano",
    resolution: int = 384,
) -> bytes:
    payload = io.BytesIO()
    torch.save(
        {
            "model": {"probe": torch.zeros(1)},
            "args": {
                "class_names": list(class_names),
                "num_classes": len(class_names),
            },
            "model_name": model_name,
            "model_config": {"resolution": resolution},
        },
        payload,
    )
    return payload.getvalue()


def _bundle(tmp_path: Path, name: str = "model", **overrides: object) -> Path:
    weights = overrides.pop("weights", _checkpoint_bytes(_NAMES))
    return write_model_bundle(tmp_path / name, _spec(weights=weights, **overrides))  # type: ignore[arg-type]


def _rewrite_checksums(bundle: Path) -> None:
    document = json.loads((bundle / "checksums.json").read_text(encoding="utf-8"))
    files: dict[str, str] = {}
    for path in bundle.iterdir():
        if path.name == "checksums.json" or path.is_symlink() or not path.is_file():
            continue
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    document["files"] = files
    (bundle / "checksums.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _swap_mapping_order(bundle: Path) -> None:
    path = bundle / "class_mapping.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["class_names"] = [
        "excavator_body",
        "haul_truck",
        "bucket",
        "truck_bed",
    ]
    document["by_name"]["haul_truck"]["internal_id"] = 1
    document["by_name"]["bucket"]["internal_id"] = 2
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_checksums(bundle)


def _refuse_stand_ins(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    constructed: list[str] = []

    def _boom(label: str):
        def _call(*_args: object, **_kwargs: object) -> None:
            constructed.append(label)
            raise AssertionError(label)

        return _call

    monkeypatch.setattr(baseline_module, "BaselineCoco", _boom("BaselineCoco"))
    monkeypatch.setattr(baseline_module, "_load_rfdetr_nano", _boom("_load_rfdetr_nano"))
    monkeypatch.setattr(adapter_module, "FakeDetector", _boom("FakeDetector"))
    import rfdetr

    monkeypatch.setattr(rfdetr, "RFDETRNano", _boom("RFDETRNano"))
    monkeypatch.setattr(rfdetr, "RFDETRSmall", _boom("RFDETRSmall"))
    return constructed


def test_load_reads_weights_and_keeps_class_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = _refuse_stand_ins(monkeypatch)
    poison = _checkpoint_bytes(
        ["bucket", "excavator_body", "truck_bed", "haul_truck"],
        model_name="RFDETRSmall",
        resolution=512,
    )
    bundle = _bundle(tmp_path, training_checkpoint=poison)
    seen: list[str] = []
    real = bundle_mod._read_checkpoint

    def _spy(path: Path) -> dict[str, object]:
        seen.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(bundle_mod, "_read_checkpoint", _spy)
    adapter = load_bundle(bundle)
    mapping = json.loads((bundle / "class_mapping.json").read_text(encoding="utf-8"))
    names = list(adapter.metadata().class_names)
    assert seen == ["weights.pth"]
    assert (bundle / "training_checkpoint.ckpt").is_file()
    assert names == _NAMES
    assert names == mapping["class_names"]
    assert names.index("bucket") == 1
    assert adapter.metadata().origin == CheckpointOrigin.custom
    assert adapter.metadata().origin == "custom"
    assert adapter.metadata().variant == "nano"
    assert adapter.metadata().resolution == 384
    assert constructed == []


def test_fresh_process_returns_mapping_names(tmp_path: Path) -> None:
    poison = _checkpoint_bytes(
        ["bucket", "excavator_body", "truck_bed", "haul_truck"],
        model_name="RFDETRSmall",
        resolution=512,
    )
    bundle = _bundle(tmp_path, "fresh", training_checkpoint=poison)
    command = [sys.executable, "-c", _CHILD, str(bundle)]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert not isinstance(completed.args, str)
    assert completed.returncode == 0, completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["names"] == _NAMES
    assert payload["bucket"] == 1
    assert payload["streamlit"] is False
    assert payload["module"] == "excavation_lab.models.bundle"


def test_class_mapping_order_must_match_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = _refuse_stand_ins(monkeypatch)
    bundle = _bundle(tmp_path, "order")
    config = json.loads((bundle / "model_config.json").read_text(encoding="utf-8"))
    assert config["class_names"] == _NAMES
    _swap_mapping_order(bundle)
    with pytest.raises(IncompatibleCheckpointError) as mismatch:
        load_bundle(bundle)
    assert mismatch.value.code == "incompatible_checkpoint"
    assert isinstance(mismatch.value, LabError)
    assert _cyrillic(mismatch.value.message)
    assert "class_mapping.json: excavator_body, haul_truck, bucket, truck_bed" in (
        mismatch.value.message
    )
    assert "weights.pth: excavator_body, bucket, haul_truck, truck_bed" in mismatch.value.message
    assert constructed == []


def test_resolution_must_match_model_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = _refuse_stand_ins(monkeypatch)
    bundle = _bundle(tmp_path, "resolution", weights=_checkpoint_bytes(_NAMES, resolution=512))
    with pytest.raises(IncompatibleCheckpointError) as mismatch:
        load_bundle(bundle)
    assert mismatch.value.code == "incompatible_checkpoint"
    assert _cyrillic(mismatch.value.message)
    assert "512" in mismatch.value.message
    assert "384" in mismatch.value.message
    assert constructed == []


def test_variant_must_match_model_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = _refuse_stand_ins(monkeypatch)
    bundle = _bundle(
        tmp_path,
        "variant",
        weights=_checkpoint_bytes(_NAMES, model_name="RFDETRSmall"),
    )
    with pytest.raises(IncompatibleCheckpointError) as mismatch:
        load_bundle(bundle)
    assert mismatch.value.code == "incompatible_checkpoint"
    assert _cyrillic(mismatch.value.message)
    assert "small" in mismatch.value.message
    assert "nano" in mismatch.value.message
    assert constructed == []


def test_missing_weights_does_not_construct_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = _refuse_stand_ins(monkeypatch)
    bundle = _bundle(tmp_path, "missing", training_checkpoint=b"resume-not-inference")
    weights = bundle / "weights.pth"
    weights.unlink()
    assert (bundle / "training_checkpoint.ckpt").is_file()
    with pytest.raises(CheckpointMissingError) as missing:
        load_bundle(bundle, model=object())
    assert missing.value.code == "checkpoint_missing"
    assert _cyrillic(missing.value.message)
    assert str(weights) in missing.value.message
    assert "COCO" in missing.value.message
    assert constructed == []

    ckpt = bundle / "training_checkpoint.ckpt"
    with pytest.raises(IncompatibleCheckpointError) as resume:
        load_bundle(ckpt)
    assert resume.value.code == "incompatible_checkpoint"
    assert _cyrillic(resume.value.message)
    assert constructed == []


def test_loader_uses_safe_path_and_does_not_retune(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = inspect.getsource(bundle_mod._read_checkpoint)
    source += inspect.getsource(bundle_mod._load_custom_module)
    assert "_safe_torch_load" in source
    assert "weights_only=False" not in source
    assert "trust=True" not in source
    assert "trust_checkpoint=True" not in source

    captured: dict[str, object] = {}
    built: list[str] = []

    class _Nano:
        def __init__(self, **kwargs: object) -> None:
            captured["nano"] = kwargs
            built.append("nano")

    class _Small:
        def __init__(self, **kwargs: object) -> None:
            captured["small"] = kwargs
            built.append("small")

    import rfdetr

    monkeypatch.setattr(rfdetr, "RFDETRNano", _Nano)
    monkeypatch.setattr(rfdetr, "RFDETRSmall", _Small)
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"metadata")
    bundle_mod._load_custom_module(
        weights,
        variant="nano",
        resolution=384,
        class_count=4,
    )
    kwargs = captured["nano"]
    assert isinstance(kwargs, dict)
    assert set(kwargs) == {
        "pretrain_weights",
        "resolution",
        "num_classes",
        "trust_checkpoint",
    }
    assert kwargs["trust_checkpoint"] is False
    assert kwargs["resolution"] == 384
    assert kwargs["num_classes"] == 4
    assert str(kwargs["pretrain_weights"]).endswith("weights.pth")
    assert "threshold" not in kwargs
    assert "small" not in captured

    calls: list[tuple[str, bool]] = []
    from rfdetr.utilities import io

    real_load = io._safe_torch_load

    def _spy(path: object, *, trust: bool = False) -> object:
        calls.append((Path(str(path)).name, trust))
        return real_load(path, trust=trust)

    monkeypatch.setattr(io, "_safe_torch_load", _spy)
    bundle = _bundle(tmp_path, "safe")
    load_bundle(bundle)
    assert calls == [("weights.pth", False)]
    assert built == ["nano"]


def test_predict_keeps_raw_score_floor(tmp_path: Path) -> None:
    class _Stub:
        def __init__(self) -> None:
            self.thresholds: list[float] = []
            self.class_names = tuple(CLASS_NAMES)

        def predict(
            self,
            image: np.ndarray,
            threshold: float = 0.5,
            **kwargs: object,
        ) -> SimpleNamespace:
            self.thresholds.append(float(threshold))
            assert kwargs.get("include_source_image") is False
            assert image.dtype == np.uint8
            return SimpleNamespace(
                xyxy=np.zeros((0, 4)),
                class_id=np.zeros((0,)),
                confidence=np.zeros((0,)),
                data={},
            )

    stub = _Stub()
    adapter = load_bundle(_bundle(tmp_path, "predict"), model=stub)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result = adapter.predict(frame, 1.5)
    assert stub.thresholds == [LIBRARY_SCORE_THRESHOLD]
    assert LIBRARY_SCORE_THRESHOLD != DISPLAY_SCORE_THRESHOLD
    assert LIBRARY_SCORE_THRESHOLD != 0.5
    assert result.origin == CheckpointOrigin.custom
    assert result.class_name == ()
    adapter.close()
    with pytest.raises(LabError) as closed:
        adapter.predict(frame, 0.0)
    assert closed.value.code == "detector_closed"


@pytest.mark.manual
def test_real_bundle_reloads_when_present() -> None:
    """Reload one on-disk bundle through the real constructor.

    Not collected by default (``addopts = -m 'not manual'``). Does not download
    weights and does not run in the unit suite.
    """
    repo = Path(__file__).resolve().parents[2]
    workspace = repo / "workspace"
    found: Path | None = None
    if workspace.is_dir():
        for weights in workspace.glob("**/weights.pth"):
            if weights.is_symlink() or not weights.is_file():
                continue
            if (weights.parent / "model_config.json").is_file():
                found = weights.parent
                break
    if found is None:
        pytest.skip("no real bundle under workspace")
    detector = load_bundle(found)
    assert list(detector.metadata().class_names) == _NAMES
    assert detector.metadata().origin == CheckpointOrigin.custom
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detector.predict(frame, 0.0)
    detector.close()
