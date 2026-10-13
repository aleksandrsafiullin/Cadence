"""Checkpoint names stay checkpoint names. No weight download."""

from __future__ import annotations

import ast
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from excavation_lab.errors import CheckpointMissingError
from excavation_lab.models import baseline as baseline_module
from excavation_lab.models.adapter import (
    DISPLAY_SCORE_THRESHOLD,
    FakeDetector,
    ScriptedDetection,
)
from excavation_lab.models.baseline import (
    LIBRARY_SCORE_THRESHOLD,
    BaselineCoco,
    open_detector,
)

_REPO = Path(__file__).resolve().parents[2]


def _cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)


def test_fake_checkpoint_truck_is_not_renamed() -> None:
    detector = FakeDetector(
        class_names=("truck",),
        scripted=(
            ScriptedDetection(
                xyxy=(10.0, 10.0, 20.0, 20.0),
                class_id=0,
                score=0.11,
                class_name="truck",
            ),
        ),
    )
    frame = np.zeros((80, 30, 3), dtype=np.uint8)
    result = detector.predict(frame, 1.25)
    assert result.origin == "baseline_coco"
    assert result.class_name == ("truck",)
    assert result.class_name[0] != "excavator_body"
    assert result.class_id.tolist() == [0]
    assert result.score.tolist() == [0.11]
    assert result.score[0] < DISPLAY_SCORE_THRESHOLD
    assert result.xyxy.tolist() == [[10.0, 10.0, 20.0, 20.0]]
    assert detector.metadata().class_names == ("truck",)
    assert detector.metadata().boxes_are_original is True


def test_contiguous_checkpoint_index_keeps_truck() -> None:
    detector = FakeDetector(
        class_names=("car", "truck"),
        scripted=(ScriptedDetection(xyxy=(0.0, 0.0, 1.0, 1.0), class_id=1, score=0.2),),
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result = detector.predict(frame, 0.0)
    assert result.class_name == ("truck",)
    assert result.class_name[0] != "bucket"
    assert result.class_name[0] != "excavator_body"


def test_baseline_adapter_emits_reported_truck_and_skips_display_threshold() -> None:
    class _Checkpoint:
        size = "fake"
        resolution = 384
        class_names = ("truck",)

        def __init__(self) -> None:
            self.thresholds: list[float] = []

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
                xyxy=np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float64),
                class_id=np.array([0]),
                confidence=np.array([0.11]),
                data={"class_name": np.array(["truck"], dtype=object)},
            )

    module = _Checkpoint()
    detector = BaselineCoco(module)
    frame = np.zeros((80, 30, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)
    result = detector.predict(frame, 0.5)
    assert module.thresholds == [LIBRARY_SCORE_THRESHOLD]
    assert LIBRARY_SCORE_THRESHOLD != DISPLAY_SCORE_THRESHOLD
    assert result.class_name == ("truck",)
    assert result.origin == "baseline_coco"
    assert result.score.tolist() == [0.11]
    assert result.xyxy.tolist() == [[10.0, 10.0, 20.0, 20.0]]
    assert detector.metadata().boxes_are_original is True
    assert detector.metadata().class_names == ("truck",)
    assert detector.boundary_frames[0][0, 0].tolist() == [255, 0, 0]


def test_baseline_index_fallback_uses_checkpoint_order() -> None:
    class _Checkpoint:
        class_names = ("car", "truck")

        def predict(
            self,
            image: np.ndarray,
            threshold: float = 0.5,
            **kwargs: object,
        ) -> SimpleNamespace:
            del image, threshold, kwargs
            return SimpleNamespace(
                xyxy=np.array([[0.0, 0.0, 2.0, 2.0]]),
                class_id=np.array([1]),
                confidence=np.array([0.04]),
                data={},
            )

    result = BaselineCoco(_Checkpoint()).predict(np.zeros((4, 4, 3), dtype=np.uint8), 0.0)
    assert result.class_name == ("truck",)
    assert result.score.tolist() == [0.04]


def test_custom_without_weights_does_not_construct_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[str] = []

    def _boom(*_args: object, **_kwargs: object) -> None:
        constructed.append("called")
        raise AssertionError("baseline constructed")

    monkeypatch.setattr(baseline_module, "BaselineCoco", _boom)
    monkeypatch.setattr(baseline_module, "_load_rfdetr_nano", _boom)

    with pytest.raises(CheckpointMissingError) as missing:
        open_detector("custom")
    assert missing.value.code == "checkpoint_missing"
    assert _cyrillic(missing.value.message)
    assert "COCO" in missing.value.message

    absent = tmp_path / "missing" / "weights.pth"
    with pytest.raises(CheckpointMissingError) as named:
        open_detector("custom", weights_path=absent)
    assert named.value.code == "checkpoint_missing"
    assert str(absent) in named.value.message
    assert constructed == []


def test_close_drops_the_module_reference() -> None:
    class _Checkpoint:
        class_names = ("truck",)

        def predict(self, image: np.ndarray, threshold: float = 0.5, **kwargs: object) -> None:
            del image, threshold, kwargs
            raise AssertionError("predict is not part of this test")

    module = _Checkpoint()
    detector = BaselineCoco(module)
    ref = weakref.ref(module)
    del module
    assert ref() is not None
    detector.close()
    gc.collect()
    assert ref() is None
    with pytest.raises(Exception) as closed:
        detector.predict(np.zeros((4, 4, 3), dtype=np.uint8), 0.0)
    assert closed.value.code == "detector_closed"


def test_fake_detectors_are_not_a_shared_singleton() -> None:
    first = FakeDetector(class_names=("truck",))
    second = FakeDetector(class_names=("car",))
    first.close()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    second.predict(frame, 0.0)
    assert second.metadata().class_names == ("car",)
    assert first.metadata().class_names == ("truck",)


def test_rfdetr_import_stays_inside_the_loader() -> None:
    path = _REPO / "src" / "excavation_lab" / "models" / "baseline.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        for name in modules:
            assert name.split(".", 1)[0] not in {
                "rfdetr",
                "inference",
                "ultralytics",
                "streamlit",
            }
    loaders = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_rfdetr_nano"
    ]
    assert len(loaders) == 1
    imported: list[str] = []
    for node in ast.walk(loaders[0]):
        if isinstance(node, ast.ImportFrom) and node.module == "rfdetr":
            imported.extend(alias.name for alias in node.names)
    assert imported == ["RFDETRNano"]
