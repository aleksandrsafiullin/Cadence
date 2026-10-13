"""PTS clock and the streaming reader.

Clips written here are synthetic fixtures. They are not accuracy evidence
for pit video or for the detector.

``write_synthetic_mp4`` has a single leading keyframe. ``InputContainer.seek``
is still given the stream timestamp; the demuxer then returns to that
keyframe, and frames before ``start_s`` are not yielded. An all-keyframe
clip in this file can land on the requested tick.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import av
import numpy as np
import pytest

from excavation_lab.errors import DuplicateFrameError, TimebaseError
from excavation_lab.media import reader as reader_module
from excavation_lab.media.reader import (
    FrameReader,
    ReadFrame,
    map_coded_box,
    map_coded_image,
    upright_size,
)
from excavation_lab.media.timebase import (
    TimeBase,
    TimeMode,
    cfr_fallback_time_s,
    frame_time_s,
    in_half_open,
    is_constant_frame_rate,
    require_half_open_interval,
    require_monotonic,
    require_non_negative_duration,
    resolve_time_mode,
    source_time_s,
)
from excavation_lab.schemas import DecodeStatus

_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SYNTHETIC_SPEC = importlib.util.spec_from_file_location(
    "excavation_lab_synthetic_mp4",
    _SYNTHETIC_PATH,
)
assert _SYNTHETIC_SPEC is not None and _SYNTHETIC_SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SYNTHETIC_SPEC)
_SYNTHETIC_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_mp4 = _SYNTHETIC.write_synthetic_mp4

_VFR_DURATIONS_S = (0.2, 0.5)


def _cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)


def _write_clip(
    path: Path,
    *,
    fps: int,
    frames: int,
    pts_origin: int = 0,
    rotation: int | None = None,
    width: int = 32,
    height: int = 24,
    vfr_ticks: list[int] | None = None,
) -> None:
    """Synthetic MP4. Not accuracy evidence.

    ``pts_origin`` is in units of ``1/fps`` before the encoder rescales it.
    ``vfr_ticks`` are milliseconds at a 1/1000 time base.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if vfr_ticks is not None:
        _write_vfr(path, vfr_ticks, width, height)
        return
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "g": "1",
            "keyint": "1",
            "min-keyint": "1",
            "scenecut": "0",
        }
        if rotation is not None:
            stream.set_display_rotation(rotation)
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(blank, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            frame.pts = pts_origin + index
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def _write_vfr(path: Path, durations_ms: list[int], width: int, height: int) -> None:
    from fractions import Fraction

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(0))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.options = {"preset": "ultrafast", "g": "1", "keyint": "1", "scenecut": "0"}
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        cursor = 0
        for duration_ms in durations_ms:
            frame = av.VideoFrame.from_ndarray(blank, format="rgb24")
            frame = frame.reformat(format="yuv420p")
            frame.pts = cursor
            frame.time_base = Fraction(1, 1000)
            cursor += duration_ms
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def test_source_time_lands_on_300_seconds() -> None:
    time_base = TimeBase(1, 90_000)
    at_three_hundred = source_time_s(27_000_000, time_base, 0.0)
    assert at_three_hundred == 300.0
    assert at_three_hundred != 0

    # PTS already at 300 s, and the container start is the same mark.
    # Subtracting the offset would return 0. The offset is included.
    start_pts = 27_000_000
    included = source_time_s(start_pts, time_base, 300.0, start_pts=start_pts)
    assert included == 300.0
    assert included != 0


def test_vfr_stays_on_pts_and_cfr_fallback_is_rejected() -> None:
    durations = list(_VFR_DURATIONS_S)
    assert is_constant_frame_rate(durations) is False
    assert resolve_time_mode(durations, TimeMode.pts) is TimeMode.pts
    time_base = TimeBase(1, 1000)
    pts_times = [
        frame_time_s(
            pts=pts,
            time_base=time_base,
            start_offset_s=0.0,
            durations_s=durations,
            frame_index=index,
            fps=5.0,
            mode=TimeMode.pts,
        )
        for index, pts in enumerate((0, 200, 700))
    ]
    assert pts_times == [0.0, 0.2, 0.7]
    fallback = [index / 5.0 for index in range(3)]
    assert pts_times != fallback
    with pytest.raises(TimebaseError) as raised:
        frame_time_s(
            pts=700,
            time_base=time_base,
            start_offset_s=0.0,
            durations_s=durations,
            frame_index=2,
            fps=5.0,
            mode=TimeMode.cfr_fallback,
        )
    assert raised.value.code == "timebase"
    assert _cyrillic(raised.value.message)


def test_cfr_fallback_matches_index_over_fps_only_after_the_check() -> None:
    durations = [0.2, 0.2, 0.2]
    assert resolve_time_mode(durations) is TimeMode.pts
    assert resolve_time_mode(durations, TimeMode.cfr_fallback) is TimeMode.cfr_fallback
    time_base = TimeBase(1, 1000)
    assert (
        frame_time_s(
            pts=400,
            time_base=time_base,
            start_offset_s=0.0,
            durations_s=durations,
            frame_index=2,
            fps=5.0,
            mode=TimeMode.cfr_fallback,
        )
        == 0.4
    )
    # Same ticks through the default clock, which stays on PTS.
    assert (
        frame_time_s(
            pts=400,
            time_base=time_base,
            start_offset_s=0.0,
            durations_s=durations,
            frame_index=2,
            fps=5.0,
        )
        == 0.4
    )


def test_cfr_fallback_keeps_300s_start_offset(tmp_path: Path) -> None:
    durations = [0.1, 0.1, 0.1]
    time_base = TimeBase(1, 10)
    start_pts = 3_000
    assert (
        frame_time_s(
            pts=start_pts,
            time_base=time_base,
            start_offset_s=300.0,
            start_pts=start_pts,
            durations_s=durations,
            frame_index=0,
            fps=10.0,
            mode=TimeMode.cfr_fallback,
        )
        == 300.0
    )
    assert cfr_fallback_time_s(1, 10.0, 300.0) == pytest.approx(300.1)
    assert cfr_fallback_time_s(1, 10.0, 300.0) - 300.0 == pytest.approx(0.1)
    assert (
        frame_time_s(
            pts=start_pts,
            time_base=time_base,
            start_offset_s=300.0,
            start_pts=start_pts,
            durations_s=durations,
            frame_index=0,
            fps=10.0,
        )
        == 300.0
    )
    with pytest.raises(TimebaseError):
        frame_time_s(
            pts=700,
            time_base=TimeBase(1, 1000),
            start_offset_s=0.0,
            durations_s=list(_VFR_DURATIONS_S),
            frame_index=2,
            fps=5.0,
            mode=TimeMode.cfr_fallback,
        )

    path = tmp_path / "fragment-cfr.mp4"
    _write_clip(path, fps=10, frames=5, pts_origin=300 * 10)
    frames = list(
        FrameReader(path, video_id="fragment", time_mode=TimeMode.cfr_fallback).iterate(
            300.0,
            300.35,
        )
    )
    assert frames
    assert frames[0].source_time_s == pytest.approx(300.0)
    assert frames[0].source_time_s != 0
    assert frames[0].span_end_s == pytest.approx(300.1)
    assert frames[0].span_end_s - frames[0].source_time_s == pytest.approx(0.1)
    assert frames[1].source_time_s == pytest.approx(300.1)
    pts_frames = list(FrameReader(path, video_id="fragment").iterate(300.0, 300.15))
    assert pts_frames[0].source_time_s == pytest.approx(300.0)
    assert pts_frames[0].time_mode is TimeMode.pts


def test_variable_durations_keep_distinct_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "variable-ticks.mp4"
    _write_clip(path, fps=10, frames=3)
    with av.open(str(path)) as container:
        start_pts = container.streams.video[0].start_time or 0
    ticks = (
        (start_pts, 100),
        (start_pts + 100, 100),
        (start_pts + 200, 200),
    )
    emitted = {"done": False}

    def variable(packet: av.Packet) -> object:
        if emitted["done"]:
            return iter(())
        emitted["done"] = True

        def frames() -> object:
            for pts, duration in ticks:
                yield _TickFrame(pts, duration)

        return frames()

    monkeypatch.setattr(reader_module, "decode_packet", variable)
    got = list(FrameReader(path, video_id="ticks").iterate())
    assert [frame.source_frame_index for frame in got] == [0, 1, 2]


class _TickFrame:
    """Coded frame with an explicit duration. Not a picture from the file."""

    def __init__(self, pts: int, duration: int) -> None:
        self.pts = pts
        self.duration = duration
        self.rotation = 0

    def to_ndarray(self, **_kwargs: object) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)


def test_reader_cfr_fallback_after_constant_durations(tmp_path: Path) -> None:
    path = tmp_path / "constant.mp4"
    _write_clip(path, fps=10, frames=6)
    frames = list(
        FrameReader(path, video_id="constant", time_mode=TimeMode.cfr_fallback).iterate(
            0.0,
            0.35,
        )
    )
    assert frames
    assert all(frame.time_mode is TimeMode.cfr_fallback for frame in frames)
    for frame in frames:
        assert frame.source_frame_index is not None
        assert frame.source_time_s == pytest.approx(frame.source_frame_index / 10)


def test_negative_duration_and_decreasing_timestamps() -> None:
    with pytest.raises(TimebaseError) as negative:
        require_half_open_interval(2.0, 1.0)
    assert negative.value.code == "timebase"
    assert "Отрицательная длительность" in negative.value.message
    assert _cyrillic(negative.value.message)

    with pytest.raises(TimebaseError) as decreasing:
        require_monotonic(1.0, 0.25)
    assert "убыва" in decreasing.value.message

    with pytest.raises(TimebaseError):
        require_non_negative_duration(-0.2)
    with pytest.raises(TimebaseError):
        source_time_s(0, TimeBase(1, 90_000), 0.0, start_pts=90_000)

    with pytest.raises(TimebaseError):
        list(FrameReader(Path("unused.mp4"), video_id="clip").iterate(2.0, 1.0))


def test_half_open_interval_and_container_seek(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert in_half_open(1.0, 1.0, 1.5) is True
    assert in_half_open(1.5, 1.0, 1.5) is False
    assert in_half_open(0.0, 1.0, 1.5) is False
    assert in_half_open(1.0, 1.0, 1.0) is False

    path = tmp_path / "clip.mp4"
    seconds = 2.0
    fps = 10
    write_synthetic_mp4(path, seconds, fps)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        assert time_base is not None
        expected_offset = round(1.0 * time_base.denominator / time_base.numerator)

    calls: list[tuple[int, object]] = []
    original = av.container.InputContainer.seek

    def spy(self: object, offset: int, *args: object, **kwargs: object) -> None:
        calls.append((offset, kwargs.get("stream")))
        return original(self, offset, *args, **kwargs)

    monkeypatch.setattr(av.container.InputContainer, "seek", spy)
    frames = list(FrameReader(path, video_id="clip").iterate(1.0, 1.5))

    assert calls, "seek was not called; the reader decoded from timestamp 0"
    assert calls[0][0] == expected_offset
    assert calls[0][0] != 0
    assert calls[0][1] is not None
    assert frames
    assert frames[0].source_time_s is not None
    assert frames[0].source_time_s >= 1.0
    assert frames[0].source_time_s != 0
    assert all(frame.source_time_s is not None and frame.source_time_s < 1.5 for frame in frames)
    assert all(frame.time_mode is TimeMode.pts for frame in frames)
    assert frames[0].time_base == (
        f"{frames[0].time_base_numerator}/{frames[0].time_base_denominator}"
    )
    assert frames[0].image is not None
    assert frames[0].image.ndim == 3
    assert frames[0].image.shape[2] == 3
    assert frames[0].decode_status is DecodeStatus.ok
    assert list(FrameReader(path, video_id="clip").iterate(1.0, 1.0)) == []


def test_keyframe_seek_does_not_decode_the_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "keyframes.mp4"
    _write_clip(path, fps=10, frames=30)
    calls = {"n": 0}
    real = reader_module.decode_packet

    def counting(packet: av.Packet) -> object:
        calls["n"] += 1
        return real(packet)

    monkeypatch.setattr(reader_module, "decode_packet", counting)
    gen = FrameReader(path, video_id="clip").iterate(1.0, 1.5)
    assert inspect.isgenerator(gen)
    assert calls["n"] == 0
    first = next(gen)
    assert calls["n"] <= 2
    gen.close()
    assert first.source_time_s == pytest.approx(1.0)
    assert first.source_frame_index == 10
    assert first.source_time_s != 0


def test_nonzero_fragment_start_is_not_rebased(tmp_path: Path) -> None:
    path = tmp_path / "fragment.mp4"
    fps = 10
    _write_clip(path, fps=fps, frames=5, pts_origin=300 * fps)
    frames = list(FrameReader(path, video_id="fragment").iterate(300.0, 300.3))
    assert frames
    assert frames[0].source_time_s == pytest.approx(300.0)
    assert frames[0].source_time_s != 0
    assert frames[0].start_offset_s == pytest.approx(300.0)
    assert all(
        frame.source_time_s is not None and 300.0 <= frame.source_time_s < 300.3
        for frame in frames
    )
    assert all(frame.time_mode is TimeMode.pts for frame in frames)


def test_reader_vfr_file_rejects_cfr_fallback(tmp_path: Path) -> None:
    path = tmp_path / "vfr.mp4"
    _write_clip(path, fps=10, frames=1, vfr_ticks=[200, 500, 200])
    frames = list(FrameReader(path, video_id="vfr").iterate())
    assert len(frames) >= 3
    assert {frame.time_mode for frame in frames} == {TimeMode.pts}
    assert frames[0].source_time_s == pytest.approx(0.0)
    gaps = [
        frames[index + 1].source_time_s - frames[index].source_time_s  # type: ignore[operator]
        for index in range(2)
    ]
    assert abs(gaps[0] - gaps[1]) > 0.1
    with pytest.raises(TimebaseError) as raised:
        list(FrameReader(path, video_id="vfr", time_mode=TimeMode.cfr_fallback).iterate())
    assert raised.value.code == "timebase"
    assert _cyrillic(raised.value.message)


def test_duplicate_source_frame_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "dup.mp4"
    _write_clip(path, fps=10, frames=3)
    real = reader_module.decode_packet
    state = {"done": False}

    def twice(packet: av.Packet) -> object:
        frames = list(real(packet))
        if frames and not state["done"]:
            state["done"] = True

            def duplicate() -> object:
                yield frames[0]
                yield frames[0]

            return duplicate()
        return iter(frames)

    monkeypatch.setattr(reader_module, "decode_packet", twice)
    with pytest.raises(DuplicateFrameError) as raised:
        list(FrameReader(path, video_id="dup").iterate())
    assert raised.value.code == "duplicate_frame"
    assert raised.value.source_frame_index == 0
    assert _cyrillic(raised.value.message)


def test_corrupt_span_continues_without_downtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "corrupt.mp4"
    _write_clip(path, fps=10, frames=5)
    real = reader_module.decode_packet
    state = {"n": 0}

    def flaky(packet: av.Packet) -> object:
        state["n"] += 1
        if state["n"] == 2:
            raise av.error.InvalidDataError(
                int(av.error.ErrorType.INVALIDDATA),
                "damaged packet",
            )
        return real(packet)

    monkeypatch.setattr(reader_module, "decode_packet", flaky)
    frames = list(FrameReader(path, video_id="corrupt").iterate())
    statuses = [frame.decode_status for frame in frames]
    assert DecodeStatus.corrupt in statuses
    assert DecodeStatus.ok in statuses
    corrupt_at = statuses.index(DecodeStatus.corrupt)
    assert DecodeStatus.ok in statuses[corrupt_at + 1 :]
    corrupt = frames[corrupt_at]
    assert corrupt.image is None
    assert corrupt.decode_status is DecodeStatus.corrupt
    for frame in frames:
        assert not hasattr(frame, "downtime")
        assert not hasattr(frame, "idle")
        assert not hasattr(frame, "event")
    assert "downtime" not in ReadFrame.__slots__
    assert "idle" not in ReadFrame.__slots__
    assert "event" not in ReadFrame.__slots__


def test_display_rotation_maps_coded_pixels_and_box(tmp_path: Path) -> None:
    path = tmp_path / "rotated.mp4"
    _write_clip(path, fps=5, frames=1, rotation=90, width=16, height=8)
    sample = next(FrameReader(path, video_id="rotated").iterate(0.0, 1.0))
    assert sample.rotation_deg == 90
    assert sample.coded_width == 16
    assert sample.coded_height == 8
    assert (sample.width, sample.height) == upright_size(16, 8, sample.rotation_deg)
    assert sample.image is not None
    assert sample.image.shape == (sample.height, sample.width, 3)

    coded = np.zeros((sample.coded_height, sample.coded_width, 3), dtype=np.uint8)
    coded[0, 0] = (9, 8, 7)
    upright = map_coded_image(coded, sample.rotation_deg)
    box = map_coded_box(
        0,
        0,
        1,
        1,
        sample.coded_width,
        sample.coded_height,
        sample.rotation_deg,
    )
    # 90° counterclockwise: coded top-left lands on the upright bottom-left.
    assert upright.shape == (16, 8, 3)
    assert upright[-1, 0].tolist() == [9, 8, 7]
    assert box == (0.0, 15.0, 1.0, 16.0)
    assert upright[int(box[1]), int(box[0])].tolist() == [9, 8, 7]


def test_reader_does_not_buffer_the_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "stream.mp4"
    _write_clip(path, fps=10, frames=20)
    calls = {"n": 0}
    real = reader_module.decode_packet

    def counting(packet: av.Packet) -> object:
        calls["n"] += 1
        return real(packet)

    monkeypatch.setattr(reader_module, "decode_packet", counting)
    gen = FrameReader(path, video_id="stream").iterate()
    assert inspect.isgenerator(gen)
    assert calls["n"] == 0
    first = next(gen)
    assert calls["n"] <= 2
    assert first.image is not None
    gen.close()
