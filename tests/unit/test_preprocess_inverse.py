"""Letterbox inverse. Synthetic geometry, not pit-video accuracy."""

from __future__ import annotations

import numpy as np

from excavation_lab.models.adapter import CheckpointOrigin, ModelMetadata
from excavation_lab.models.preprocess import inverse_boxes


def test_letterbox_inverse_uses_known_pad_and_scale() -> None:
    # Original 20×40 (w×h) on an 80 canvas: scale 2, pad_x 20, pad_y 0.
    # A second row uses the same canvas and a non-zero pad_y case is below.
    letterboxed = np.array(
        [
            [28.0, 12.0, 40.0, 36.0],
            [20.0, 0.0, 24.0, 6.0],
        ],
        dtype=np.float64,
    )
    restored = inverse_boxes(
        letterboxed,
        scale=2.0,
        pad_x=20.0,
        pad_y=0.0,
        boxes_are_original=False,
    )
    assert restored.tolist() == [
        [4.0, 6.0, 10.0, 18.0],
        [0.0, 0.0, 2.0, 3.0],
    ]

    # Original 40×20 on a 100 canvas: scale 2.5, pad_x 0, pad_y 25.
    portrait = np.array([[20.0, 35.0, 40.0, 55.0]], dtype=np.float64)
    restored_portrait = inverse_boxes(
        portrait,
        scale=2.5,
        pad_x=0.0,
        pad_y=25.0,
        boxes_are_original=False,
    )
    assert restored_portrait.tolist() == [[8.0, 4.0, 16.0, 12.0]]


def test_inverse_is_noop_when_boxes_are_already_original() -> None:
    letterboxed = np.array([[28.0, 12.0, 40.0, 36.0]], dtype=np.float64)
    metadata = ModelMetadata(
        origin=CheckpointOrigin.baseline_coco,
        class_names=("truck",),
        boxes_are_original=True,
        variant="rfdetr-nano",
        resolution=384,
    )
    restored = inverse_boxes(
        letterboxed,
        scale=2.0,
        pad_x=20.0,
        pad_y=25.0,
        boxes_are_original=metadata.boxes_are_original,
    )
    assert metadata.boxes_are_original is True
    np.testing.assert_array_equal(restored, letterboxed)
    restored[0, 0] = 0.0
    assert letterboxed[0, 0] == 28.0
