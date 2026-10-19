"""PTS-only tick pass: CFR parity, B-frames, VFR refusal, seek-decode writer, manifest fields."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from itertools import pairwise
from pathlib import Path

import av
import numpy as np
import pytest
from pydantic import ValidationError

from excavation_lab import storage
from excavation_lab.datasets import sampler as sampler_mod
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.sampler import (
    SAMPLE_SCHEMA_VERSION,
    SampleResult,
    decode_frame_at,
    format_sample_summary,
    probe_is_cfr,
    sample_frames,
)
from excavation_lab.datasets.splits import (
    SplitProposal,
    freeze_split,
    require_frozen,
    write_proposal,
)
from excavation_lab.errors import LabError, VfrSamplingUnsupportedError
from excavation_lab.media.probe import add_video
from excavation_lab.media.reader import FrameReader, PtsTick
from excavation_lab.schemas import SplitName

_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SPEC = importlib.util.spec_from_file_location("excavation_lab_synthetic_pts", _SYNTHETIC_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_mp4 = _SYNTHETIC.write_synthetic_mp4
write_synthetic_vfr_mp4 = _SYNTHETIC.write_synthetic_vfr_mp4

_BLOCKS = (
    RoleBlock(start_s=0, end_s=60, role="train"),
    RoleBlock(start_s=72, end_s=92, role="validation"),
    RoleBlock(start_s=104, end_s=124, role="test"),
)
_VFR_BLOCKS = (
    RoleBlock(start_s=0, end_s=0.5, role="train"),
    RoleBlock(start_s=0.9, end_s=1.1, role="validation"),
)
_BFRAME_BLOCKS = (
    RoleBlock(start_s=0, end_s=12, role="train"),
    RoleBlock(start_s=22, end_s=30, role="validation"),
)


@pytest.fixture(scope="module")
def cfr_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("cfr") / "cfr.mp4"
    write_synthetic_mp4(path, 126, 2)
    return path


@pytest.fixture(scope="module")
def vfr_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("vfr") / "vfr.mp4"
    write_synthetic_vfr_mp4(path)
    return path


@pytest.fixture(scope="module")
def bframe_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("bframes") / "bframes.mp4"
    write_synthetic_mp4(path, 30, 10, bframes=True)
    return path


def _bind(tmp_path: Path, clip: Path, blocks: tuple[RoleBlock, ...], *, probe: bool) -> Path:
    """Project with one frozen video. ``probe`` fills the fps columns through ffprobe."""
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    storage.replace_extra_roots(workspace, "pilot", [clip.parent])
    if probe:
        add_video(workspace, "pilot", clip, video_id="vid")
    else:
        storage.insert_video(workspace, "pilot", "vid", path=clip, name=clip.name)
    proposal = SplitProposal(video_id="vid", blocks=list(blocks))
    freeze_split(workspace, "pilot", write_proposal(workspace, "pilot", proposal))
    return workspace


def _spy(monkeypatch: pytest.MonkeyPatch, name: str, calls: list[tuple[float, float | None]]):
    original = getattr(FrameReader, name)

    def wrapped(self: FrameReader, start_s: float = 0.0, end_s: float | None = None):
        calls.append((start_s, end_s))
        return original(self, start_s, end_s)

    monkeypatch.setattr(FrameReader, name, wrapped)


def _demux_pts(clip: Path) -> list[int]:
    with av.open(str(clip)) as container:
        stream = container.streams.video[0]
        return [packet.pts for packet in container.demux(stream) if packet.pts is not None]


def _assert_pts_pass_matches_decode(
    workspace: Path, clip: Path, blocks: tuple[RoleBlock, ...]
) -> None:
    frozen = require_frozen(workspace, "pilot", "vid")
    reader = FrameReader(clip, video_id="vid")
    roles = {SplitName(block.role) for block in blocks}
    for split in sorted(roles, key=lambda item: item.value):
        windows = sampler_mod._windows(frozen, split)
        decoded = sampler_mod._collect_ticks(reader, frozen, split, windows)
        by_pts = sampler_mod._collect_ticks(reader, frozen, split, windows, prefer_pts=True)
        assert decoded.used_pts is False
        assert decoded.reason is None
        assert by_pts.used_pts is True
        assert by_pts.reason is None
        assert decoded.ticks
        assert by_pts.ticks == decoded.ticks
        assert all(tick.source_pts is not None for tick in by_pts.ticks)
        assert all(tick.span_end_s is not None for tick in by_pts.ticks)


def test_pts_only_ticks_equal_decoded_ticks_on_the_cfr_fixture(
    tmp_path: Path, cfr_clip: Path
) -> None:
    workspace = _bind(tmp_path, cfr_clip, _BLOCKS, probe=False)
    _assert_pts_pass_matches_decode(workspace, cfr_clip, _BLOCKS)
    reader = FrameReader(cfr_clip, video_id="vid")
    plain = [
        (tick.source_frame_index, tick.source_pts, tick.source_time_s, tick.span_end_s)
        for tick in reader.iterate_pts(10.0, 14.0)
    ]
    decoded_plain = [
        (frame.source_frame_index, frame.source_pts, frame.source_time_s, frame.span_end_s)
        for frame in reader.iterate(10.0, 14.0)
    ]
    assert plain == decoded_plain
    assert all(isinstance(tick, PtsTick) for tick in reader.iterate_pts(10.0, 11.0))
    demuxed = _demux_pts(cfr_clip)
    assert all(later > earlier for earlier, later in pairwise(demuxed))


def test_bframe_clip_demuxes_out_of_order_and_still_matches_decode(
    tmp_path: Path, bframe_clip: Path
) -> None:
    demuxed = _demux_pts(bframe_clip)
    assert len(demuxed) == 300
    assert any(later < earlier for earlier, later in pairwise(demuxed))
    reader = FrameReader(bframe_clip, video_id="vid")
    full = [
        (frame.source_frame_index, frame.source_pts, frame.source_time_s, frame.span_end_s)
        for frame in reader.iterate()
    ]
    assert [item[0] for item in full] == list(range(300))
    by_pts = [
        (tick.source_frame_index, tick.source_pts, tick.source_time_s, tick.span_end_s)
        for tick in reader.iterate_pts()
    ]
    assert by_pts == full
    workspace = _bind(tmp_path, bframe_clip, _BFRAME_BLOCKS, probe=False)
    _assert_pts_pass_matches_decode(workspace, bframe_clip, _BFRAME_BLOCKS)


def test_seek_decode_returns_the_same_pixels_as_the_full_decode(bframe_clip: Path) -> None:
    reader = FrameReader(bframe_clip, video_id="vid")
    full = {frame.source_pts: frame for frame in reader.iterate()}
    ticks = list(reader.iterate_pts())
    # Keyframe, frames right after it, B-frames deep in a GOP, the last frame.
    picked = [ticks[0], ticks[1], ticks[2], ticks[31], ticks[47], ticks[158], ticks[-1]]
    lumas = set()
    for tick in picked:
        frame = decode_frame_at(
            reader,
            source_frame_index=tick.source_frame_index,
            source_time_s=tick.source_time_s,
            source_pts=tick.source_pts,
            span_end_s=tick.span_end_s,
        )
        assert frame is not None
        assert frame.source_pts == tick.source_pts
        assert frame.source_frame_index == tick.source_frame_index
        expected = full[tick.source_pts]
        assert expected.image is not None and frame.image is not None
        assert np.array_equal(frame.image, expected.image)
        lumas.add(int(frame.image[0, 0, 0]))
    # Frames differ from each other, so equality above is not a blank match.
    assert len(lumas) == len(picked)
    # Without a PTS the match falls back to the source index.
    tick = ticks[77]
    by_index = decode_frame_at(
        reader,
        source_frame_index=tick.source_frame_index,
        source_time_s=tick.source_time_s,
    )
    assert by_index is not None
    assert by_index.source_pts == tick.source_pts
    assert np.array_equal(by_index.image, full[tick.source_pts].image)
    # Asking for the wrong index at that time yields nothing, not a neighbour.
    wrong = decode_frame_at(
        reader,
        source_frame_index=tick.source_frame_index + 1,
        source_time_s=tick.source_time_s,
    )
    assert wrong is None


def test_vfr_fixture_refuses_the_pts_pass_and_the_sampler_decodes_instead(
    tmp_path: Path,
    vfr_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = FrameReader(vfr_clip, video_id="vid")
    with pytest.raises(VfrSamplingUnsupportedError) as caught:
        list(reader.iterate_pts())
    assert caught.value.code == "vfr_sampling_unsupported"
    assert isinstance(caught.value, LabError)
    decoded_times = [frame.source_time_s for frame in reader.iterate()]
    assert len(decoded_times) == 8
    steps = {round(later - earlier, 3) for earlier, later in pairwise(decoded_times)}
    assert len(steps) > 1

    workspace = _bind(tmp_path, vfr_clip, _VFR_BLOCKS, probe=True)
    frozen = require_frozen(workspace, "pilot", "vid")
    windows = sampler_mod._windows(frozen, SplitName.train)
    collected = sampler_mod._collect_ticks(
        reader, frozen, SplitName.train, windows, prefer_pts=True
    )
    assert collected.used_pts is False
    assert collected.name == "decode"
    assert collected.reason == caught.value.message
    assert [tick.source_time_s for tick in collected.ticks] == pytest.approx(
        [0.0, 0.1, 0.2, 0.45]
    )

    pts_calls: list[tuple[float, float | None]] = []
    decode_calls: list[tuple[float, float | None]] = []
    _spy(monkeypatch, "iterate_pts", pts_calls)
    _spy(monkeypatch, "iterate", decode_calls)
    monkeypatch.setattr(sampler_mod, "probe_is_cfr", lambda _video: True)
    result = sample_frames(workspace, "pilot", split="train", count=4, spacing_s=0.05)
    assert result.decode_failed == []
    assert result.written.train == 4
    assert result.tick_pass == "decode"
    assert result.tick_pass_reason == caught.value.message
    assert [row.source_time_s for row in result.rows] == pytest.approx([0.0, 0.1, 0.2, 0.45])
    assert [row.source_pts for row in result.rows] == [0, 1600, 3200, 7200]
    assert {row.time_base for row in result.rows} == {"1/16000"}
    assert (0.0, 0.5) in decode_calls
    assert pts_calls
    summary = format_sample_summary(result)
    assert "Отбор по PTS без декодирования не выполнен" in summary
    assert caught.value.message in summary

    # A probe that does not say CFR skips the attempt and says so.
    monkeypatch.setattr(sampler_mod, "probe_is_cfr", lambda _video: False)
    pts_calls.clear()
    plain = sample_frames(workspace, "pilot", split="train", count=4, spacing_s=0.05)
    assert plain.tick_pass == "decode"
    assert plain.tick_pass_reason == sampler_mod._PROBE_NOT_CFR
    assert pts_calls == []


def test_cfr_probe_switches_the_sampler_to_the_pts_pass(
    tmp_path: Path,
    cfr_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _bind(tmp_path, cfr_clip, _BLOCKS, probe=True)
    video = storage.get_video(workspace, "pilot", "vid")
    assert probe_is_cfr(video) is True
    assert probe_is_cfr({"fps_nominal": 30.0, "fps_average": 29.5}) is False
    assert probe_is_cfr({"fps_nominal": None, "fps_average": 30.0}) is False
    pts_calls: list[tuple[float, float | None]] = []
    decode_calls: list[tuple[float, float | None]] = []
    _spy(monkeypatch, "iterate_pts", pts_calls)
    _spy(monkeypatch, "iterate", decode_calls)
    result = sample_frames(workspace, "pilot", plan="diagnostic")
    assert (0.0, 60.0) in pts_calls
    assert (72.0, 92.0) in pts_calls
    assert (0.0, 60.0) not in decode_calls
    assert (72.0, 92.0) not in decode_calls
    assert len(decode_calls) == len(result.rows)
    for start_s, end_s in decode_calls:
        assert end_s is not None
        assert end_s - start_s == pytest.approx(0.5, abs=1e-6)
    assert result.sample_schema_version == SAMPLE_SCHEMA_VERSION == 2
    assert result.tick_pass == "pts"
    assert result.tick_pass_reason is None
    assert "Отбор по PTS" not in format_sample_summary(result)
    assert result.written.validation == 20
    assert all(row.source_pts is not None and row.time_base is not None for row in result.rows)
    for row in result.rows:
        assert row.source_pts is not None
        assert row.source_pts * 1 / 16384 == pytest.approx(row.source_time_s, abs=1e-6)
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["sample_schema_version"] == 2
    assert manifest["decode_failed"] == []
    assert manifest["tick_pass"] == "pts"
    assert all("source_pts" in row and "time_base" in row for row in manifest["rows"])
    reread = SampleResult.model_validate(manifest)
    assert reread.rows[0].time_base == "1/16384"


def test_a_tick_without_pixels_is_decode_failed_not_a_neighbour(
    tmp_path: Path,
    cfr_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _bind(tmp_path, cfr_clip, _BLOCKS, probe=False)
    frozen = require_frozen(workspace, "pilot", "vid")
    reader = FrameReader(cfr_clip, video_id="vid")
    windows = sampler_mod._windows(frozen, SplitName.validation)
    ticks = sampler_mod._collect_ticks(reader, frozen, SplitName.validation, windows).ticks
    victim = ticks[3].source_frame_index
    original = FrameReader.iterate

    def blanked(self: FrameReader, start_s: float = 0.0, end_s: float | None = None):
        # Only the per-tick read loses pixels; the tick pass still lists the frame.
        per_tick = end_s is not None and end_s - start_s < 1.0
        for frame in original(self, start_s, end_s):
            if per_tick and frame.source_frame_index == victim:
                yield dataclasses.replace(frame, image=None)
            else:
                yield frame

    monkeypatch.setattr(FrameReader, "iterate", blanked)
    result = sample_frames(workspace, "pilot", split="validation", count=len(ticks))
    assert result.decode_failed == [victim]
    assert result.written.validation == len(ticks) - 1
    assert result.shortfall.validation == 1
    assert victim not in {row.source_frame_index for row in result.rows}
    assert not (Path(result.images_dir) / f"vid_{victim}.jpg").exists()
    summary = format_sample_summary(result)
    assert "decode_failed" in summary
    assert str(victim) in summary
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["decode_failed"] == [victim]


def test_legacy_manifest_reads_back_with_null_pts(tmp_path: Path, cfr_clip: Path) -> None:
    workspace = _bind(tmp_path, cfr_clip, (_BLOCKS[0],), probe=False)
    result = sample_frames(workspace, "pilot", split="train", count=3)
    document = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    document.pop("sample_schema_version")
    document.pop("decode_failed")
    document.pop("tick_pass")
    document.pop("tick_pass_reason")
    for row in document["rows"]:
        row.pop("source_pts")
        row.pop("time_base")
    legacy = SampleResult.model_validate(document)
    assert legacy.sample_schema_version == 1
    assert legacy.decode_failed == []
    assert legacy.tick_pass is None
    assert legacy.tick_pass_reason is None
    assert all(row.source_pts is None and row.time_base is None for row in legacy.rows)
    document["sample_schema_version"] = 2
    with pytest.raises(ValidationError, match="source_pts"):
        SampleResult.model_validate(document)
    half = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    half["rows"][0]["time_base"] = None
    with pytest.raises(ValidationError, match="together"):
        SampleResult.model_validate(half)
    bad_base = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    bad_base["rows"][0]["time_base"] = "0/10"
    with pytest.raises(ValidationError):
        SampleResult.model_validate(bad_base)
    stray = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert stray["tick_pass"] == "decode"
    assert stray["tick_pass_reason"] == sampler_mod._PROBE_NOT_CFR
    stray["tick_pass"] = "pts"
    with pytest.raises(ValidationError, match="tick_pass_reason"):
        SampleResult.model_validate(stray)
