"""Spec §19 item 18: a long clip is streamed.

The MP4 is synthetic. It is not accuracy evidence for pit video.
"""

from __future__ import annotations

import importlib.util
import tracemalloc
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from excavation_lab.media.reader import FrameReader
from excavation_lab.models.adapter import (
    CheckpointOrigin,
    FakeDetector,
    FrameDetections,
)
from excavation_lab.tracking.analyze import AnalyzePipeline

_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SYNTHETIC_SPEC = importlib.util.spec_from_file_location(
    "excavation_lab_synthetic_mp4",
    _SYNTHETIC_PATH,
)
assert _SYNTHETIC_SPEC is not None and _SYNTHETIC_SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SYNTHETIC_SPEC)
_SYNTHETIC_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_mp4 = _SYNTHETIC.write_synthetic_mp4

_FRAMES = 300
_FPS = 30
_ANALYSIS_FPS = 5.0
_WIDTH = 64
_HEIGHT = 48


class CountingReader:
    """Counts yields. Does not keep the frames."""

    def __init__(self, path: Path) -> None:
        self._reader = FrameReader(path, video_id="stream")
        self.yielded = 0
        self.samples: list[int] = []

    def iterate(self, start_s: float = 0.0, end_s: float | None = None):
        for frame in self._reader.iterate(start_s, end_s):
            self.yielded += 1
            if self.yielded in {30, _FRAMES}:
                self.samples.append(tracemalloc.get_traced_memory()[0])
            yield frame


class CountingDetector:
    """Counts ``predict`` calls. Does not keep the RGB arrays."""

    def __init__(self) -> None:
        self.pipeline: AnalyzePipeline | None = None
        self.calls = 0
        self.timestamps: list[float] = []
        self.saw_buffered_row = False

    def predict(self, frame_rgb: np.ndarray, timestamp_s: float) -> FrameDetections:
        self.calls += 1
        self.timestamps.append(float(timestamp_s))
        assert frame_rgb.ndim == 3
        assert frame_rgb.shape[2] == 3
        pipeline = self.pipeline
        assert pipeline is not None
        assert pipeline.max_retained_frames <= 1
        assert pipeline._current_image is frame_rgb
        tables = pipeline.tables
        assert tables is not None
        assert tables.chunk_count == 0
        assert tables.retained_rows < _FRAMES
        if tables.retained_rows > 0:
            self.saw_buffered_row = True
        for value in vars(pipeline).values():
            if isinstance(value, list):
                assert len(value) < _FRAMES
        count = 0
        return FrameDetections(
            timestamp_s=float(timestamp_s),
            frame_width=int(frame_rgb.shape[1]),
            frame_height=int(frame_rgb.shape[0]),
            origin=CheckpointOrigin.baseline_coco,
            xyxy=np.zeros((count, 4), dtype=np.float64),
            class_id=np.zeros((count,), dtype=np.int64),
            class_name=(),
            score=np.zeros((count,), dtype=np.float64),
        )

    def metadata(self):
        raise AssertionError("metadata is not required to stream frames")

    def close(self) -> None:
        return None


def test_long_clip_does_not_retain_every_frame(tmp_path: Path) -> None:
    path = tmp_path / "long.mp4"
    write_synthetic_mp4(path, _FRAMES / _FPS, _FPS)
    reader = CountingReader(path)
    detector = CountingDetector()
    pipeline = AnalyzePipeline(
        reader,
        detector,
        run_id="stream",
        output_dir=tmp_path / "out",
        analysis_fps=_ANALYSIS_FPS,
        model_version="baseline_coco",
    )
    detector.pipeline = pipeline

    tracemalloc.start()
    try:
        pipeline.run()
    finally:
        tracemalloc.stop()

    assert reader.yielded >= _FRAMES
    assert detector.calls > 0
    assert detector.calls < reader.yielded
    assert pipeline.max_retained_frames == 1
    assert pipeline._current_image is None
    assert len(detector.timestamps) == detector.calls
    # Growth between an early frame and the last frame. A list of 300 RGB
    # images would move this by about ``yielded * width * height * 3``.
    frame_bytes = _WIDTH * _HEIGHT * 3
    assert len(reader.samples) == 2
    assert reader.samples[1] - reader.samples[0] < 200 * frame_bytes
    _assert_one_chunk(pipeline, tmp_path / "out")
    assert detector.saw_buffered_row

    fake = FakeDetector()
    fake_reader = CountingReader(path)
    fake_pipeline = AnalyzePipeline(
        fake_reader,
        fake,
        run_id="stream-fake",
        output_dir=tmp_path / "fake-out",
        analysis_fps=_ANALYSIS_FPS,
        model_version="baseline_coco",
    )
    tracemalloc.start()
    try:
        fake_pipeline.run()
    finally:
        tracemalloc.stop()
    assert fake_reader.yielded >= _FRAMES
    assert len(fake.boundary_frames) <= 1
    assert len(fake_reader.samples) == 2
    assert fake_reader.samples[1] - fake_reader.samples[0] < 200 * frame_bytes
    _assert_one_chunk(fake_pipeline, tmp_path / "fake-out")


def _assert_one_chunk(pipeline: AnalyzePipeline, output_dir: Path) -> None:
    """The 10 s clip is one 30 s chunk, and that chunk is not 300 source rows."""
    tables = pipeline.tables
    assert tables is not None
    written = (
        pq.read_table(output_dir / "frame_status.parquet").num_rows
        + pq.read_table(output_dir / "detections.parquet").num_rows
        + pq.read_table(output_dir / "track_states.parquet").num_rows
    )
    assert tables.chunk_count == 1
    assert tables.retained_rows == 0
    assert tables.chunk_s == 30.0
    assert written == tables.rows_written == tables.peak_retained_rows
    assert 0 < written < _FRAMES
