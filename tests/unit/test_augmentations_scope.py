"""Spec §19 item 4: train augmentation stays off validation and test.

Synthetic images only. Does not call ``RFDETR.train`` and does not open a
test dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from excavation_lab.errors import LabError
from excavation_lab.models import augment
from excavation_lab.models.augment import (
    BRIGHTNESS_LIMIT,
    CONTRAST_LIMIT,
    EVAL_AUGMENTATION,
    HORIZONTAL_FLIP_PROBABILITY,
    MILD_PRESET,
    SCALE_LIMIT,
    TRAIN_TRANSFORMS,
    apply,
    brightness_contrast,
    crop_to_window,
    save_examples,
)
from excavation_lab.models.preprocess import model_rgb_frame
from excavation_lab.models.train_config import resolve_train_profile
from excavation_lab.schemas import pilot_path, read_yaml_mapping


def test_validation_and_test_match_and_skip_horizontal_flip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"flip": 0, "photo": 0, "crop": 0}
    original_flip = augment.horizontal_flip
    original_photo = augment.brightness_contrast
    original_crop = augment.crop_to_window

    def flip_spy(image: np.ndarray, boxes: np.ndarray):
        calls["flip"] += 1
        return original_flip(image, boxes)

    def photo_spy(image: np.ndarray, *, brightness: float, contrast: float):
        calls["photo"] += 1
        return original_photo(image, brightness=brightness, contrast=contrast)

    def crop_spy(image: np.ndarray, boxes: np.ndarray, window: tuple[int, int, int, int]):
        calls["crop"] += 1
        return original_crop(image, boxes, window)

    monkeypatch.setattr(augment, "horizontal_flip", flip_spy)
    monkeypatch.setattr(augment, "brightness_contrast", photo_spy)
    monkeypatch.setattr(augment, "crop_to_window", crop_spy)

    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image[:, 0, 0] = 7
    image[:, -1, 0] = 200
    boxes = np.array([[1.0, 2.0, 6.0, 10.0]], dtype=np.float64)
    image_before = image.copy()
    boxes_before = boxes.copy()
    overrides = {
        "seed": 1,
        "flip_horizontal": True,
        "brightness": BRIGHTNESS_LIMIT,
        "contrast": -CONTRAST_LIMIT,
        "crop_window": (2, 2, 8, 8),
    }
    validation = apply(image, boxes, "validation", **overrides)
    test = apply(image, boxes, "test", **overrides)

    np.testing.assert_array_equal(validation.image, test.image)
    np.testing.assert_array_equal(validation.boxes, test.boxes)
    np.testing.assert_array_equal(validation.image, model_rgb_frame(image))
    np.testing.assert_array_equal(validation.boxes, boxes)
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(boxes, boxes_before)
    assert validation.image[0, 0, 0] == 7
    assert validation.horizontal_flip_applied is False
    assert test.horizontal_flip_applied is False
    assert validation.crop_applied is False
    assert test.crop_applied is False
    assert calls == {"flip": 0, "photo": 0, "crop": 0}
    assert validation.image is not test.image
    assert validation.boxes is not test.boxes


def test_train_horizontal_flip_updates_box_x() -> None:
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[:, 0, 0] = 5
    image[:, -1, 0] = 9
    boxes = np.array([[2.0, 4.0, 10.0, 12.0]], dtype=np.float64)
    image_before = image.copy()
    boxes_before = boxes.copy()
    sample = apply(
        image,
        boxes,
        "train",
        flip_horizontal=True,
        brightness=0.0,
        contrast=0.0,
        crop_window=(0, 0, 40, 20),
    )
    assert sample.horizontal_flip_applied is True
    assert sample.image[0, 0, 0] == 9
    assert sample.boxes.tolist() == [[30.0, 4.0, 38.0, 12.0]]
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(boxes, boxes_before)


def test_horizontal_flip_is_not_forced_on_every_train_sample() -> None:
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image[:, 0, 0] = 3
    image[:, -1, 0] = 9
    boxes = np.array([[1.0, 1.0, 4.0, 4.0]], dtype=np.float64)
    flags: set[bool] = set()
    for seed in range(40):
        sample = apply(
            image,
            boxes,
            "train",
            seed=seed,
            brightness=0.0,
            contrast=0.0,
            crop_window=(0, 0, 32, 16),
        )
        flags.add(sample.horizontal_flip_applied)
        if sample.horizontal_flip_applied:
            assert sample.image[0, 0, 0] == 9
            assert sample.boxes[0, 0] == 32 - 4
            assert sample.boxes[0, 2] == 32 - 1
        else:
            assert sample.image[0, 0, 0] == 3
            assert sample.boxes.tolist() == boxes.tolist()
    assert flags == {True, False}
    assert HORIZONTAL_FLIP_PROBABILITY == 0.5


def test_vertical_flip_is_not_enabled() -> None:
    source = Path(augment.__file__).read_text(encoding="utf-8").lower()
    assert "vertical" not in source
    for token in ("flipud", "perspective", "cutout", "coarsedropout"):
        assert token not in source
    assert "vertical" not in json.dumps(list(TRAIN_TRANSFORMS)).lower()
    assert "vertical" not in json.dumps(MILD_PRESET).lower()
    assert "vertical" not in json.dumps(EVAL_AUGMENTATION).lower()
    for name in ("smoke", "pilot-small"):
        resolved = resolve_train_profile(name)
        blob = json.dumps(resolved.model_dump(mode="json")).lower()
        assert "vertical" not in blob
        assert "perspective" not in blob
        assert "cutout" not in blob
        body = resolved.train_kwargs["augmentation_train"]
        assert body == MILD_PRESET
        assert body["brightness_limit"] == BRIGHTNESS_LIMIT == 0.2
        assert body["contrast_limit"] == CONTRAST_LIMIT == 0.2
        assert body["scale_limit"] == SCALE_LIMIT == 0.1
        assert body["horizontal_flip_p"] == 0.5
        assert "aug_config" not in resolved.not_applied
        assert "no library preset is selected" not in blob
        assert "aug_config is unset" not in blob


def test_crop_on_the_boundary_stays_positive_and_a_deleted_box_rejects_the_crop() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[10, 70] = (1, 2, 3)
    window = (0, 0, 90, 100)
    on_edge = np.array([[70.0, 10.0, 90.0, 40.0]], dtype=np.float64)
    cropped, boxes, applied = crop_to_window(image, on_edge, window)
    assert applied is True
    assert cropped.shape == (100, 90, 3)
    assert cropped[10, 70].tolist() == [1, 2, 3]
    assert boxes.tolist() == [[70.0, 10.0, 90.0, 40.0]]
    assert boxes[0, 2] > boxes[0, 0]
    assert boxes[0, 3] > boxes[0, 1]

    straddle = np.array([[85.0, 10.0, 99.0, 40.0]], dtype=np.float64)
    _cropped, clipped, applied = crop_to_window(image, straddle, window)
    assert applied is True
    assert clipped.tolist() == [[85.0, 10.0, 90.0, 40.0]]
    assert clipped[0, 2] > clipped[0, 0]

    outside = np.array([[90.0, 10.0, 100.0, 40.0]], dtype=np.float64)
    rejected, rejected_boxes, applied = crop_to_window(image, outside, window)
    assert applied is False
    np.testing.assert_array_equal(rejected, image)
    np.testing.assert_array_equal(rejected_boxes, outside)
    assert rejected is not image
    assert rejected_boxes is not outside

    both = np.array(
        [[10.0, 10.0, 20.0, 20.0], [90.0, 10.0, 100.0, 40.0]],
        dtype=np.float64,
    )
    _image, kept, applied = crop_to_window(image, both, window)
    assert applied is False
    np.testing.assert_array_equal(kept, both)

    with pytest.raises(ValueError, match="scale limit"):
        crop_to_window(image, on_edge, (0, 0, 80, 100))


def test_train_crop_stays_inside_the_scale_limit() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    boxes = np.array([[20.0, 20.0, 80.0, 70.0]], dtype=np.float64)
    for seed in range(30):
        sample = apply(
            image,
            boxes,
            "train",
            seed=seed,
            flip_horizontal=False,
            brightness=0.0,
            contrast=0.0,
        )
        assert sample.crop_applied is True
        assert 100 - sample.image.shape[0] <= int(np.floor(100 * SCALE_LIMIT))
        assert 120 - sample.image.shape[1] <= int(np.floor(120 * SCALE_LIMIT))
        assert sample.boxes[0, 2] > sample.boxes[0, 0]
        assert sample.boxes[0, 3] > sample.boxes[0, 1]


def test_photometric_gains_stay_inside_the_preset() -> None:
    image = np.full((4, 4, 3), 80, dtype=np.uint8)
    out = brightness_contrast(image, brightness=0.2, contrast=-0.2)
    assert out.dtype == np.uint8
    assert int(out[0, 0, 0]) == 96
    with pytest.raises(ValueError):
        brightness_contrast(image, brightness=0.21, contrast=0.0)
    with pytest.raises(ValueError):
        brightness_contrast(image, brightness=0.0, contrast=-0.21)

    image_span = np.zeros((8, 8, 3), dtype=np.uint8)
    image_span[:, :4, :] = 40
    image_span[:, 4:, :] = 160
    boxes = np.zeros((0, 4), dtype=np.float64)
    for seed in range(20):
        sample = apply(
            image_span,
            boxes,
            "train",
            seed=seed,
            flip_horizontal=False,
            crop_window=(0, 0, 8, 8),
        )
        assert sample.image.dtype == np.uint8
        assert int(sample.image.min()) >= 0
        assert int(sample.image.max()) <= 255


def test_save_examples_writes_a_pair_without_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("model.train was called")

    monkeypatch.setattr("rfdetr.detr.RFDETR.train", fail)
    image = np.zeros((32, 40, 3), dtype=np.uint8)
    image[:, :20, 0] = 20
    image[:, 20:, 1] = 200
    boxes = np.array([[4.0, 4.0, 12.0, 16.0]], dtype=np.float64)
    before, after = save_examples(tmp_path, image=image, boxes=boxes)
    assert before.name == "before.png"
    assert after.name == "after.png"
    assert before.parent == tmp_path
    saved_before = np.asarray(Image.open(before))
    saved_after = np.asarray(Image.open(after))
    assert saved_before.shape == image.shape
    assert saved_after.shape[2] == 3
    assert not np.array_equal(saved_before, saved_after)


def test_unknown_split_and_library_preset_are_rejected() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply(image, np.zeros((0, 4)), "val")

    raw = dict(read_yaml_mapping(pilot_path())["profiles"]["smoke"])
    raw["augmentation_train"] = "AUG_AGGRESSIVE"
    with pytest.raises(LabError) as raised:
        resolve_train_profile("smoke", profile=raw)
    assert raised.value.code == "unknown_augmentation"

    raw["augmentation_train"] = {"VerticalFlip": {"p": 0.5}}
    with pytest.raises(LabError) as raised:
        resolve_train_profile("smoke", profile=raw)
    assert raised.value.code == "unknown_augmentation"


def test_eval_payload_disables_library_defaults(tmp_path: Path) -> None:
    from rfdetr.config import MultiScale, TrainConfig

    resolved = resolve_train_profile("pilot-small")
    for payload in (resolved.validation_kwargs, resolved.test_kwargs, resolved.train_kwargs):
        fields = {key: value for key, value in payload.items() if key in TrainConfig.model_fields}
        config = TrainConfig(dataset_dir=tmp_path, **fields)
        assert config.aug_config == {}
        assert config.scale_jitter is False
        assert config.multi_scale == MultiScale.OFF
    for payload in (resolved.validation_kwargs, resolved.test_kwargs):
        assert "augmentation_train" not in payload
        assert payload["augmentation_eval"] == EVAL_AUGMENTATION
        assert payload["augmentation_eval"]["scale_jitter"] is False
        assert payload["augmentation_eval"]["multi_scale"] is False
        assert payload["augmentation_eval"]["aug_config"] == {}
