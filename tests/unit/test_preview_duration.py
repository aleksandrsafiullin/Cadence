"""CFR preview duration stays on the source interval.

Both clips are synthetic fixtures. They are not accuracy evidence for pit
video or for the detector. Timestamps come from the PTS reader.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import av
import numpy as np
import pytest

from excavation_lab.media.encoder import (
    GAP_SLATE_TEXT,
    PreviewFrame,
    render_gap_slate,
    write_preview,
)
from excavation_lab.media.probe import ffprobe_executable
from excavation_lab.media.reader import FrameReader, ReadFrame

_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SYNTHETIC_SPEC = importlib.util.spec_from_file_location(
    "excavation_lab_synthetic_mp4",
    _SYNTHETIC_PATH,
)
assert _SYNTHETIC_SPEC is not None and _SYNTHETIC_SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SYNTHETIC_SPEC)
_SYNTHETIC_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_mp4 = _SYNTHETIC.write_synthetic_mp4

_MAP_COLUMNS = [
    "preview_time_s",
    "source_time_s",
    "source_frame_index",
    "repeated",
    "gap_slate",
]
_START_S = 0.5
_END_S = 1.5
_OUTPUT_FPS = 10.0
_SOURCE_FPS = 30


def test_uniform_5hz_grid_keeps_interval_duration(tmp_path: Path) -> None:
    """Five kept frames over [0.5, 1.5) at 10 fps preview.

    The analysis step is 0.2 s. That is under the 0.5 s gap threshold, so
    this clip checks duration, the map, and the repeat flag only.
    """
    source = tmp_path / "uniform_2s.mp4"
    preview = tmp_path / "uniform_preview.mp4"
    map_path = tmp_path / "preview_time_map.csv"
    write_synthetic_mp4(source, 2.0, _SOURCE_FPS)
    targets = _analysis_grid(_START_S, _END_S, 5.0)
    assert len(targets) == 5
    read = _read_targets(source, targets, _START_S, _END_S)
    kept = [PreviewFrame.from_read_frame(frame) for frame in read]

    write_preview(
        kept,
        preview,
        output_fps=_OUTPUT_FPS,
        start_s=_START_S,
        end_s=_END_S,
        map_path=map_path,
    )

    duration_s = _assert_silent_cfr(preview, _OUTPUT_FPS)
    assert duration_s == pytest.approx(1.0, abs=1.0 / _OUTPUT_FPS)
    rows = _read_map(map_path)
    assert len(rows) == 10
    assert rows[0]["preview_time_s"] == pytest.approx(0.0, abs=1e-6)
    assert rows[0]["source_time_s"] == pytest.approx(kept[0].source_time_s, abs=1e-6)
    assert rows[0]["source_time_s"] == pytest.approx(0.5, abs=1e-3)
    observations = []
    for index, row in enumerate(rows):
        assert row["preview_time_s"] == pytest.approx(index / _OUTPUT_FPS, abs=1e-6)
        assert _START_S <= row["source_time_s"] < _END_S
        assert row["gap_slate"] is False
        assert row["source_frame_index"] == kept[index // 2].source_frame_index
        assert row["source_time_s"] == pytest.approx(kept[index // 2].source_time_s)
        assert row["repeated"] is (index % 2 == 1)
        if not row["repeated"]:
            observations.append(row["source_frame_index"])
    assert observations == [frame.source_frame_index for frame in kept]
    assert len(set(observations)) == 5


def test_large_gap_slate_is_not_an_observation(tmp_path: Path) -> None:
    """Kept frames at 0.5 s and 1.3 s. The 0.8 s hole is a slate.

    The output frame on each kept timestamp stays that observation. The
    slate is every other frame in the hole, so its length is 0.8 s within
    one output frame.
    """
    source = tmp_path / "gap_2s.mp4"
    preview = tmp_path / "gap_preview.mp4"
    map_path = tmp_path / "preview_time_map.csv"
    write_synthetic_mp4(source, 2.0, _SOURCE_FPS)
    read = _read_targets(source, [0.5, 1.3], _START_S, _END_S)
    assert read[1].source_time_s - read[0].source_time_s == pytest.approx(0.8, abs=1e-4)
    colors = ((220, 30, 30), (30, 30, 220))
    kept = [_paint(frame, color) for frame, color in zip(read, colors, strict=True)]

    write_preview(
        kept,
        preview,
        output_fps=_OUTPUT_FPS,
        start_s=_START_S,
        end_s=_END_S,
        map_path=map_path,
    )

    duration_s = _assert_silent_cfr(preview, _OUTPUT_FPS)
    assert duration_s == pytest.approx(_END_S - _START_S, abs=1.0 / _OUTPUT_FPS)
    rows = _read_map(map_path)
    assert rows[0]["preview_time_s"] == pytest.approx(0.0, abs=1e-6)
    assert rows[0]["source_time_s"] == pytest.approx(0.5, abs=1e-3)
    assert rows[0]["gap_slate"] is False
    for row in rows:
        assert _START_S <= row["source_time_s"] < _END_S

    slate = [row for row in rows if row["gap_slate"]]
    gap_s = kept[1].source_time_s - kept[0].source_time_s
    slate_s = len(slate) / _OUTPUT_FPS
    frame_s = 1.0 / _OUTPUT_FPS
    assert slate
    assert gap_s == pytest.approx(0.8, abs=1e-4)
    # One output frame of tolerance. ``pytest.approx`` rejects the exact
    # boundary when ``0.8 - 0.7`` is a hair over ``0.1``.
    assert abs(slate_s - gap_s) <= frame_s + 1e-9
    assert abs(slate_s - 0.8) <= frame_s + 1e-9
    for previous, current in pairwise(slate):
        step = current["preview_time_s"] - previous["preview_time_s"]
        assert step == pytest.approx(1.0 / _OUTPUT_FPS, abs=1e-6)
        assert kept[0].source_time_s < current["source_time_s"] < kept[1].source_time_s
    assert kept[0].source_time_s < slate[0]["source_time_s"] < kept[1].source_time_s

    observations = [
        row for row in rows if row["gap_slate"] is False and row["repeated"] is False
    ]
    assert [row["source_frame_index"] for row in observations] == [
        kept[0].source_frame_index,
        kept[1].source_frame_index,
    ]
    assert GAP_SLATE_TEXT == "нет наблюдения"
    for row in slate:
        assert row["gap_slate"] is True
        assert row["repeated"] is False
        assert row["source_frame_index"] is None
        assert row["source_frame_index"] not in {
            kept[0].source_frame_index,
            kept[1].source_frame_index,
        }

    decoded = _decoded_rgb(preview)
    assert len(decoded) == len(rows)
    expected = render_gap_slate(decoded[0].shape[1], decoded[0].shape[0])
    assert float(expected.std()) > 10
    pictures = {
        kept[0].source_frame_index: kept[0].image,
        kept[1].source_frame_index: kept[1].image,
    }
    for index, row in enumerate(rows):
        if row["gap_slate"]:
            assert _mae(decoded[index], expected) < 20
            assert _mae(decoded[index], kept[0].image) > 40
            assert _mae(decoded[index], kept[1].image) > 40
            continue
        picture = pictures[row["source_frame_index"]]
        assert _mae(decoded[index], picture) < 40
        assert _mae(decoded[index], expected) > 20


def _analysis_grid(start_s: float, end_s: float, analysis_fps: float) -> list[float]:
    start = Fraction(start_s).limit_denominator(1_000_000)
    end = Fraction(end_s).limit_denominator(1_000_000)
    step = Fraction(1) / Fraction(analysis_fps).limit_denominator(1_000_000)
    times: list[float] = []
    cursor = start
    while cursor < end:
        times.append(float(cursor))
        cursor += step
    return times


def _read_targets(
    path: Path,
    targets_s: list[float],
    start_s: float,
    end_s: float,
) -> list[ReadFrame]:
    """Stream the interval and keep the frame on each target PTS."""
    tolerance_s = 0.5 / _SOURCE_FPS
    pending = list(targets_s)
    kept: list[ReadFrame] = []
    reader = FrameReader(path, video_id="synthetic-preview")
    for frame in reader.iterate(start_s, end_s):
        if frame.source_time_s is None or not pending:
            continue
        if abs(frame.source_time_s - pending[0]) <= tolerance_s:
            kept.append(frame)
            pending.pop(0)
    assert pending == []
    return kept


def _paint(frame: ReadFrame, color: tuple[int, int, int]) -> PreviewFrame:
    preview = PreviewFrame.from_read_frame(frame)
    image = np.array(preview.image, copy=True)
    image[:, :] = color
    return PreviewFrame(preview.source_frame_index, preview.source_time_s, image)


def _read_map(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == _MAP_COLUMNS
        rows: list[dict[str, object]] = []
        for raw in reader:
            index_text = raw["source_frame_index"]
            rows.append(
                {
                    "preview_time_s": float(raw["preview_time_s"]),
                    "source_time_s": float(raw["source_time_s"]),
                    "source_frame_index": None if index_text == "" else int(index_text),
                    "repeated": _csv_bool(raw["repeated"]),
                    "gap_slate": _csv_bool(raw["gap_slate"]),
                }
            )
    return rows


def _csv_bool(text: str) -> bool:
    assert text in {"true", "false"}
    return text == "true"


def _assert_silent_cfr(path: Path, output_fps: float) -> float:
    command = [
        ffprobe_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    assert completed.returncode == 0
    payload = json.loads(completed.stdout.decode("utf-8"))
    streams = payload["streams"]
    video = [item for item in streams if item["codec_type"] == "video"]
    audio = [item for item in streams if item["codec_type"] == "audio"]
    assert len(video) == 1
    assert audio == []
    assert _ratio(video[0]["avg_frame_rate"]) == pytest.approx(output_fps)
    assert int(video[0]["nb_frames"]) >= 1
    return float(payload["format"]["duration"])


def _decoded_rgb(path: Path) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with av.open(str(path)) as container:
        assert not container.streams.audio
        for frame in container.decode(container.streams.video[0]):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def _mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.int16) - right.astype(np.int16))))


def _ratio(text: str) -> float:
    numerator, denominator = text.split("/", 1)
    return float(numerator) / float(denominator)
