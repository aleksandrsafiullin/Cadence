"""COCO box conversion, class ids, and the letterbox inverse.

Synthetic rectangles. They check geometry, not pit-video accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavation_lab.datasets.coco_import import (
    build_class_mapping,
    resolve_category_name,
)
from excavation_lab.datasets.geometry import xywh_to_xyxy, xyxy_to_xywh
from excavation_lab.errors import UnknownCategoryError
from excavation_lab.models.preprocess import inverse_boxes


def test_known_rectangle_converts_both_ways() -> None:
    xyxy = (10.0, 20.0, 40.0, 55.0)
    xywh = (10.0, 20.0, 30.0, 35.0)
    assert xyxy_to_xywh(xyxy, image_width=100, image_height=80) == xywh
    assert xywh_to_xyxy(xywh, image_width=100, image_height=80) == xyxy
    assert xywh_to_xyxy((0, 0, 100, 80), image_width=100, image_height=80) == (
        0.0,
        0.0,
        100.0,
        80.0,
    )
    with pytest.raises(ValueError, match="positive"):
        xyxy_to_xywh((5, 5, 5, 9), image_width=100, image_height=80)
    with pytest.raises(ValueError, match="inside"):
        xywh_to_xyxy((90, 0, 20, 10), image_width=100, image_height=80)


def test_letterbox_inverse_restores_original_coordinates() -> None:
    # Original (3, 4, 9, 10) on a canvas with scale 2 and pad (5, 7).
    original = (3.0, 4.0, 9.0, 10.0)
    scale = 2.0
    pad_x = 5.0
    pad_y = 7.0
    canvas = (
        original[0] * scale + pad_x,
        original[1] * scale + pad_y,
        original[2] * scale + pad_x,
        original[3] * scale + pad_y,
    )
    assert canvas == (11.0, 15.0, 23.0, 27.0)
    restored = inverse_boxes(
        np.array([canvas], dtype=np.float64),
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        boxes_are_original=False,
    )
    assert restored.tolist() == [list(original)]


def test_class_mapping_uses_the_name() -> None:
    assert resolve_category_name("bucket", 99, frame_id="pit:1") == 1
    assert resolve_category_name("excavator_body", 99, frame_id="pit:3") == 0
    mapping = build_class_mapping()
    assert mapping["by_name"]["bucket"]["internal_id"] == 1
    assert mapping["by_name"]["excavator_body"]["internal_id"] == 0
    assert "dump" not in mapping["by_name"]
    with pytest.raises(UnknownCategoryError) as caught:
        resolve_category_name("dump", 0, frame_id="pit:2")
    error = caught.value
    assert error.category_name == "dump"
    assert error.class_id is None
    assert error.frame_id == "pit:2"
