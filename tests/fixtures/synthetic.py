"""Synthetic MP4 writer. These files are not accuracy evidence."""

from __future__ import annotations

from pathlib import Path

import av


def write_synthetic_mp4(path: Path, seconds: float, fps: int) -> None:
    """Write a synthetic MP4 with PyAV.

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
        stream.options = {"preset": "ultrafast"}
        for _ in range(frame_count):
            frame = av.VideoFrame(width, height, "yuv420p")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
