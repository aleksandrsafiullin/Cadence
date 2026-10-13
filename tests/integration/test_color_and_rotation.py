"""RGB boundary, one BGR conversion, and display-matrix boxes.

Synthetic pixels. Not accuracy evidence for pit video.
"""

from __future__ import annotations

import numpy as np
import pytest

from excavation_lab.errors import ColorOrderError
from excavation_lab.media.reader import map_coded_box, map_coded_image
from excavation_lab.models.adapter import FakeDetector
from excavation_lab.models.preprocess import bgr_to_rgb


def test_rgb_red_pixel_stays_red_at_the_adapter_boundary() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)
    detector = FakeDetector()
    detector.predict(frame, 0.0)
    seen = detector.boundary_frames[0]
    assert seen[0, 0, 0] == 255
    assert seen[0, 0, 2] == 0
    assert seen[0, 0].tolist() == [255, 0, 0]
    assert seen[0, 0, 0] == frame[0, 0, 0]
    assert seen[0, 0, 2] == frame[0, 0, 2]


def test_bgr_buffer_is_not_passed_through_as_rgb() -> None:
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr[0, 0] = (0, 0, 255)
    converted = bgr_to_rgb(bgr)
    assert converted[0, 0, 0] == bgr[0, 0, 2]
    assert converted[0, 0, 2] == bgr[0, 0, 0]
    assert converted[0, 0].tolist() == [255, 0, 0]

    detector = FakeDetector()
    detector.predict(converted, 0.0)
    seen = detector.boundary_frames[0]
    assert seen[0, 0, 0] == 255
    assert seen[0, 0, 2] == 0
    assert seen[0, 0, 0] == converted[0, 0, 0]
    assert seen[0, 0, 2] == converted[0, 0, 2]
    assert seen[0, 0].tolist() != bgr[0, 0].tolist()

    reversed_view = bgr[:, :, ::-1]
    assert int(reversed_view.strides[2]) < 0
    rejected = FakeDetector()
    with pytest.raises(ColorOrderError) as caught:
        rejected.predict(reversed_view, 0.0)
    assert caught.value.code == "color_order"
    assert rejected.boundary_frames == []


def test_display_rotation_moves_the_box_with_the_pixels() -> None:
    coded_height = 8
    coded_width = 16
    x1, y1, x2, y2 = 2, 1, 5, 4
    coded = np.zeros((coded_height, coded_width, 3), dtype=np.uint8)
    coded[y1:y2, x1:x2, 0] = 255
    upright = map_coded_image(coded, 90)
    expected = map_coded_box(x1, y1, x2, y2, coded_width, coded_height, 90)
    detector = FakeDetector(locate_red=True, class_names=("truck",))
    result = detector.predict(upright, 0.4)
    assert result.class_name == ("truck",)
    assert result.xyxy.shape == (1, 4)
    assert result.xyxy[0].tolist() == list(expected)
    assert result.xyxy[0].tolist() != [float(x1), float(y1), float(x2), float(y2)]
    assert result.timestamp_s == 0.4
    assert result.frame_width == upright.shape[1]
    assert result.frame_height == upright.shape[0]
    left = int(result.xyxy[0, 0])
    top = int(result.xyxy[0, 1])
    assert upright[top, left, 0] == 255
    assert upright[top, left, 2] == 0
