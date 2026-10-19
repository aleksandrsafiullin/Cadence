"""Synthetic MP4 writer. These files are not accuracy evidence."""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import av
import pyarrow as pa
import pyarrow.parquet as pq

from excavation_lab.tracking.export_tables import (
    DETECTIONS_NAME,
    DETECTIONS_SCHEMA,
    FRAME_STATUS_NAME,
    FRAME_STATUS_SCHEMA,
    TRACK_STATE_SCHEMA,
    TRACK_STATES_NAME,
)


def write_synthetic_mp4(path: Path, seconds: float, fps: int, *, bframes: bool = False) -> None:
    """Write a synthetic MP4 with PyAV.

    With ``bframes`` the encoder emits a fixed three B-frames between
    references, so packets leave the muxer in decode order and PTS are not
    monotonic on demux. Each frame then carries its own luma level so that a
    seek-decoded frame can be compared pixel by pixel to a full decode.

    The clip is a test fixture. It is synthetic and is not accuracy evidence
    for pit video or for the detector.
    """
    if isinstance(seconds, bool) or isinstance(fps, bool) or not isinstance(fps, int):
        raise TypeError("seconds and fps must be numbers")
    if seconds <= 0 or fps <= 0:
        raise ValueError("seconds and fps must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, round(seconds * fps))
    width = 64
    height = 48
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        if bframes:
            stream.options = {"preset": "medium", "bf": "3", "b_strategy": "0", "g": "30"}
        else:
            stream.options = {"preset": "ultrafast"}
        for index in range(frame_count):
            frame = av.VideoFrame(width, height, "yuv420p")
            if bframes:
                luma, chroma_u, chroma_v = frame.planes
                luma.update(bytes([16 + (index * 7) % 220]) * luma.buffer_size)
                chroma_u.update(bytes([128]) * chroma_u.buffer_size)
                chroma_v.update(bytes([128]) * chroma_v.buffer_size)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def write_synthetic_vfr_mp4(
    path: Path,
    pts_ticks: tuple[int, ...] = (0, 100, 200, 450, 500, 900, 1000, 1050),
    time_base_denominator: int = 1000,
) -> None:
    """Write a synthetic MP4 whose frames sit on irregular PTS.

    ``pts_ticks`` are strictly increasing timestamps in ``1/time_base_denominator``
    seconds. The packet durations therefore differ, which is what the PTS-only
    sampling pass must refuse. The clip is synthetic and is not accuracy
    evidence for pit video or for the detector.
    """
    if len(pts_ticks) < 3:
        raise ValueError("pts_ticks needs at least three frames")
    steps = {later - earlier for earlier, later in pairwise(pts_ticks)}
    if any(step <= 0 for step in steps):
        raise ValueError("pts_ticks must increase")
    if len(steps) < 2:
        raise ValueError("pts_ticks must have at least two different steps")
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 64
    height = 48
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=time_base_denominator)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, time_base_denominator)
        stream.codec_context.time_base = Fraction(1, time_base_denominator)
        stream.options = {"preset": "ultrafast", "g": "1"}
        for pts in pts_ticks:
            frame = av.VideoFrame(width, height, "yuv420p")
            frame.pts = pts
            frame.time_base = Fraction(1, time_base_denominator)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


_BOX_CLASSES = (
    ("excavator_body", 0),
    ("bucket", 1),
    ("haul_truck", 2),
    ("truck_bed", 3),
)


def write_synthetic_run(
    directory: Path,
    *,
    run_id: str,
    video_id: str,
    start_s: float,
    end_s: float,
    fps: float = 5.0,
) -> None:
    """Write frame, detection, and track parquet for one analysis slice.

    Bucket and bed boxes move with source time. The tables are synthetic
    and are not accuracy evidence for pit video or for a detector.
    """
    if (
        isinstance(start_s, bool)
        or isinstance(end_s, bool)
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
    ):
        raise TypeError("start_s, end_s, and fps must be numbers")
    if fps <= 0 or end_s <= start_s:
        raise ValueError("fps must be positive and end_s must exceed start_s")
    if not run_id or not video_id:
        raise ValueError("run_id and video_id must be non-empty strings")
    count = round((end_s - start_s) * float(fps))
    if count < 1:
        raise ValueError("the slice needs at least one frame")
    times = [round(float(start_s) + index / float(fps), 6) for index in range(count)]
    if times[-1] >= end_s:
        raise ValueError("the last frame must stay inside the half-open slice")
    directory.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, object]] = []
    detections: list[dict[str, object]] = []
    tracks: list[dict[str, object]] = []
    for index, stamp in enumerate(times):
        boxes = _moving_boxes(stamp)
        frames.append(_frame_row(run_id, video_id, index, stamp, len(boxes)))
        for name, class_id, box in boxes:
            detections.append(
                _detection_row(run_id, video_id, index, stamp, name, class_id, box)
            )
            tracks.append(_track_row(run_id, name, stamp, box))
    _write_table(directory / FRAME_STATUS_NAME, FRAME_STATUS_SCHEMA, frames)
    _write_table(directory / DETECTIONS_NAME, DETECTIONS_SCHEMA, detections)
    _write_table(directory / TRACK_STATES_NAME, TRACK_STATE_SCHEMA, tracks)


def _moving_boxes(time_s: float) -> list[tuple[str, int, tuple[float, float, float, float]]]:
    """Body stays left. The bucket sweeps right. The bed rides in the truck."""
    body_x = 4.0 + 0.05 * time_s
    bucket_x = 6.0 + 0.45 * time_s
    truck_x = 28.0 + 2.0 * math.sin(time_s * 0.2)
    bed_x = truck_x + 6.0
    layout = {
        "excavator_body": (body_x, 10.0, body_x + 18.0, 40.0),
        "bucket": (bucket_x, 16.0, bucket_x + 8.0, 26.0),
        "haul_truck": (truck_x, 8.0, truck_x + 30.0, 42.0),
        "truck_bed": (bed_x, 12.0, bed_x + 18.0, 34.0),
    }
    return [(name, class_id, layout[name]) for name, class_id in _BOX_CLASSES]


def _frame_row(
    run_id: str,
    video_id: str,
    index: int,
    stamp: float,
    detections: int,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "video_id": video_id,
        "view_id": None,
        "source_frame_index": index,
        "source_time_s": stamp,
        "decode_status": "ok",
        "analyzed": True,
        "model_version": "det_custom",
        "inference_ms": None,
        "num_detections": detections,
        "bucket_observation_state": "observed",
        "diagnostic_flags": [],
        "source_pts": index,
        "time_base": "1/5",
        "analysis_index": index,
        "width": 64,
        "height": 48,
        "decode_ms": None,
        "encode_ms": None,
    }


def _detection_row(
    run_id: str,
    video_id: str,
    index: int,
    stamp: float,
    name: str,
    class_id: int,
    box: tuple[float, float, float, float],
) -> dict[str, object]:
    x1, y1, x2, y2 = box
    return {
        "run_id": run_id,
        "observation_id": f"{run_id}-{index}-{name}",
        "video_id": video_id,
        "view_id": None,
        "source_frame_index": index,
        "source_time_s": stamp,
        "class_id": class_id,
        "class_name": name,
        "score": 0.9,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "frame_width": 64,
        "frame_height": 48,
        "track_id_nullable": f"{run_id}-{name}",
        "parent_track_id_nullable": None,
        "observation_state": "observed",
        "association_score_nullable": None,
        "model_version": "det_custom",
        "source_pts": index,
        "time_base": "1/5",
        "analysis_index": index,
        "width": 64,
        "height": 48,
    }


def _track_row(
    run_id: str,
    name: str,
    stamp: float,
    box: tuple[float, float, float, float],
) -> dict[str, object]:
    x1, y1, x2, y2 = box
    return {
        "run_id": run_id,
        "track_id": f"{run_id}-{name}",
        "class_name": name,
        "source_time_s": stamp,
        "state": "observed",
        "last_observed_time_s": stamp,
        "is_observed": True,
        "x1_nullable": x1,
        "y1_nullable": y1,
        "x2_nullable": x2,
        "y2_nullable": y2,
        "parent_track_id_nullable": None,
        "identity_uncertain": False,
    }


def _write_table(path: Path, schema: pa.Schema, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
