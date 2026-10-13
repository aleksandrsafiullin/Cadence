"""Synthetic end-to-end v0.1 workflow.

The clip is synthetic and is not accuracy evidence.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest

from excavation_lab import jobs
from excavation_lab.datasets.coco_import import import_coco, load_imported_frames
from excavation_lab.datasets.cvat_export import MANIFEST_COLUMNS
from excavation_lab.datasets.sampler import sample_frames
from excavation_lab.datasets.splits import (
    TimelineBlock,
    freeze_split,
    load_frozen,
    propose_for_project,
    write_proposal,
)
from excavation_lab.media.probe import add_video, ffprobe_json
from excavation_lab.media.reader import FrameReader
from excavation_lab.models import baseline as baseline_mod
from excavation_lab.models import bundle as bundle_mod
from excavation_lab.models.adapter import (
    CheckpointOrigin,
    FakeDetector,
    ScriptedDetection,
)
from excavation_lab.schemas import SplitName
from excavation_lab.storage import create_project
from excavation_lab.tracking import analyze as analyze_mod
from excavation_lab.tracking.analyze import FAKE_DETECTOR_ENV, AnalyzePipeline
from excavation_lab.visualization.export_video import encode_annotated_video

# Two 10 s guards. Assignable time is duration minus those gaps, and the
# second guard has to end before the probed duration.
_SECONDS = 25.0
_SOURCE_FPS = 2
_START_S = 0.0
_END_S = 1.0
_EXPORT_FPS = 10.0
_TRAIN_COUNT = 60
_TRUCK = (4.0, 4.0, 60.0, 44.0)
_BED = (12.0, 8.0, 40.0, 28.0)
_TRUCK_SCORE = 0.91
_BED_SCORE = 0.62
_BOX_XYWH = (8.0, 6.0, 20.0, 12.0)
_IMAGE_WIDTH = 64
_IMAGE_HEIGHT = 48

_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SYNTHETIC_SPEC = importlib.util.spec_from_file_location(
    "excavation_lab_synthetic_mp4_workflow",
    _SYNTHETIC_PATH,
)
assert _SYNTHETIC_SPEC is not None and _SYNTHETIC_SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SYNTHETIC_SPEC)
_SYNTHETIC_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_mp4 = _SYNTHETIC.write_synthetic_mp4


def test_synthetic_clip_reaches_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe, technical freeze, train sample, one COCO box, fake analyze, preview."""
    _refuse_real_detector(monkeypatch)
    assert _BED[0] > _TRUCK[0] and _BED[1] > _TRUCK[1]
    assert _BED[2] < _TRUCK[2] and _BED[3] < _TRUCK[3]
    assert _TRUCK[2] <= _IMAGE_WIDTH and _TRUCK[3] <= _IMAGE_HEIGHT

    workspace = tmp_path / "workspace"
    create_project("pilot", workspace)
    clip = workspace / "clip.mp4"
    write_synthetic_mp4(clip, _SECONDS, _SOURCE_FPS)
    video = add_video(workspace, "pilot", clip)
    probed = video["duration_s"]
    assert probed > _END_S
    video_id = video["video_id"]

    proposal = propose_for_project(workspace, "pilot", video_id)
    assert proposal.episodes == []
    assert proposal.duration_s == pytest.approx(probed)
    proposal_path = write_proposal(workspace, "pilot", proposal)
    frozen, _manifest = freeze_split(workspace, "pilot", proposal_path)
    assert frozen.episodes == []
    digest = frozen.content_hash

    sampled = sample_frames(
        workspace,
        "pilot",
        video_id=video_id,
        split="train",
        count=_TRAIN_COUNT,
    )
    assert sampled.requested.train == _TRAIN_COUNT
    assert sampled.requested.validation == 0
    assert sampled.requested.test == 0
    assert sampled.written.validation == 0
    assert sampled.written.test == 0
    assert sampled.shortfall.train == _TRAIN_COUNT - sampled.written.train
    train_blocks = [block for block in frozen.blocks if block.role == "train"]
    test_blocks = [block for block in frozen.blocks if block.role == "test"]
    assert train_blocks
    assert test_blocks
    for row in sampled.rows:
        assert row.split is SplitName.train
        assert _covers(train_blocks, row.source_time_s)
        assert not _covers(test_blocks, row.source_time_s)

    imported = import_coco(
        workspace,
        "pilot",
        f"{video_id}__train",
        _write_coco(workspace, video_id, train_blocks[0].start_s, train_blocks[0].end_s),
    )
    assert imported.image_count == 1
    stored = load_imported_frames(workspace, "pilot")
    assert len(stored) == 1
    assert len(stored[0].record.boxes) == 1
    box = stored[0].record.boxes[0]
    x, y, width, height = _BOX_XYWH
    assert box.class_name == "haul_truck"
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((x, y, x + width, y + height))
    assert load_frozen(workspace, "pilot", video_id).content_hash == digest

    # enqueue_analyze spawns a child that opens a real detector.
    detector = _scripted_detector()
    tables = tmp_path / "analysis"
    try:
        pipeline = AnalyzePipeline(
            FrameReader(Path(video["path"]), video_id=video_id),
            detector,
            run_id="synthetic",
            output_dir=tables,
            analysis_fps=5,
            model_version="fake-detector",
            start_s=_START_S,
            end_s=_END_S,
        )
        frame_path, detection_path, track_path = pipeline.run()
    finally:
        detector.close()

    frames = pq.read_table(frame_path).to_pylist()
    detections = pq.read_table(detection_path).to_pylist()
    tracks = pq.read_table(track_path).to_pylist()
    assert isinstance(tracks, list)
    analyzed = [row for row in frames if row["analyzed"] is True]
    assert analyzed
    assert all(row["num_detections"] == 2 for row in analyzed)
    assert all(_START_S <= float(row["source_time_s"]) < _END_S for row in analyzed)
    by_frame: dict[int, list[dict[str, object]]] = {}
    for row in detections:
        by_frame.setdefault(int(row["source_frame_index"]), []).append(row)
    assert set(by_frame) == {int(row["source_frame_index"]) for row in analyzed}
    for rows in by_frame.values():
        assert len(rows) == 2
        found = {str(row["class_name"]): row for row in rows}
        assert set(found) == {"haul_truck", "truck_bed"}
        _assert_scripted(found["haul_truck"], _TRUCK, 2, _TRUCK_SCORE)
        _assert_scripted(found["truck_bed"], _BED, 3, _BED_SCORE)

    preview = tmp_path / "preview" / "annotated.mp4"
    exported = encode_annotated_video(
        video_path=Path(video["path"]),
        video_id=video_id,
        tables_dir=tables,
        output_path=preview,
        start_s=_START_S,
        end_s=_END_S,
        export_fps=_EXPORT_FPS,
        baseline=False,
    )
    assert exported.video_path == preview
    duration_s = float(ffprobe_json(preview)["format"]["duration"])
    assert duration_s == pytest.approx(_END_S - _START_S, abs=1.0 / _EXPORT_FPS)
    names = {path.name for path in tmp_path.rglob("*") if path.is_file()}
    assert "events.csv" not in names
    assert "truck_visits.csv" not in names


def _refuse_real_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FAKE_DETECTOR_ENV, raising=False)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("real detector or child worker")

    monkeypatch.setattr(analyze_mod, "open_analysis_detector", refuse)
    monkeypatch.setattr(baseline_mod, "open_detector", refuse)
    monkeypatch.setattr(bundle_mod, "load_bundle", refuse)
    monkeypatch.setattr(jobs, "_spawn", refuse)


def _scripted_detector() -> FakeDetector:
    return FakeDetector(
        class_names=("excavator_body", "bucket", "haul_truck", "truck_bed"),
        scripted=(
            ScriptedDetection(_TRUCK, 2, _TRUCK_SCORE, "haul_truck"),
            ScriptedDetection(_BED, 3, _BED_SCORE, "truck_bed"),
        ),
        origin=CheckpointOrigin.custom,
    )


def _covers(blocks: list[TimelineBlock], time_s: float) -> bool:
    return any(block.start_s <= time_s < block.end_s for block in blocks)


def _assert_scripted(
    row: dict[str, object],
    xyxy: tuple[float, float, float, float],
    class_id: int,
    score: float,
) -> None:
    assert int(row["class_id"]) == class_id
    assert float(row["score"]) == pytest.approx(score)
    assert [float(row[name]) for name in ("x1", "y1", "x2", "y2")] == pytest.approx(list(xyxy))


def _write_coco(workspace: Path, video_id: str, start_s: float, end_s: float) -> Path:
    stamp = (start_s + end_s) / 2.0
    image = np.full((_IMAGE_HEIGHT, _IMAGE_WIDTH, 3), 40, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    payload = encoded.tobytes()
    image_file = "frame.jpg"
    digest = hashlib.sha256(payload).hexdigest()
    x, y, width, height = _BOX_XYWH
    document = {
        "categories": [{"id": 7, "name": "haul_truck"}],
        "images": [
            {
                "id": 1,
                "file_name": f"images/{image_file}",
                "width": _IMAGE_WIDTH,
                "height": _IMAGE_HEIGHT,
                "attributes": {"review_status": "reviewed"},
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 7,
                "bbox": [x, y, width, height],
            }
        ],
    }
    manifest = _manifest_csv(
        image_file=image_file,
        video_id=video_id,
        stamp=stamp,
        digest=digest,
        size_bytes=len(payload),
    )
    destination = workspace / "labels.zip"
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("frames_manifest.csv", manifest)
        archive.writestr(
            "annotations/instances_default.json",
            json.dumps(document) + "\n",
        )
        archive.writestr(f"images/{image_file}", payload)
    return destination


def _manifest_csv(
    *,
    image_file: str,
    video_id: str,
    stamp: float,
    digest: str,
    size_bytes: int,
) -> str:
    fields = [*MANIFEST_COLUMNS, "size_bytes"]
    row = {
        "image_file": image_file,
        "video_id": video_id,
        "source_frame_index": "0",
        "source_pts": "0",
        "time_base": "1/2",
        "source_time_s": f"{stamp:.6f}",
        "episode_id": "",
        "split": "train",
        "split_group_id": "",
        "image_sha256": digest,
        "width": str(_IMAGE_WIDTH),
        "height": str(_IMAGE_HEIGHT),
        "size_bytes": str(size_bytes),
    }
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return buffer.getvalue()
