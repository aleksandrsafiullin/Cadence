"""Active learning adds train frames only and does not reshuffle the freeze."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from excavation_lab import storage
from excavation_lab.cli import app
from excavation_lab.datasets import active_learning as active_mod
from excavation_lab.datasets import splits as split_mod
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.sampler import (
    _ACTIVE_REASONS,
    SampleResult,
    format_sample_summary,
    sample_frames,
)
from excavation_lab.datasets.splits import (
    freeze_split,
    frozen_manifest_path,
    require_frozen,
    write_blocks,
)
from excavation_lab.datasets.zones import load_view_zones_file
from excavation_lab.errors import LabError
from excavation_lab.media.reader import FrameReader
from excavation_lab.models.quality import _shown_split
from excavation_lab.schemas import DecodeStatus, SplitName, load_pilot

_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SYNTHETIC_SPEC = importlib.util.spec_from_file_location(
    "excavation_lab_synthetic_mp4",
    _SYNTHETIC_PATH,
)
assert _SYNTHETIC_SPEC is not None and _SYNTHETIC_SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SYNTHETIC_SPEC)
_SYNTHETIC_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_mp4 = _SYNTHETIC.write_synthetic_mp4

_RUNNER = CliRunner()
_MODEL = "model-a"
_BLOCKS = (
    RoleBlock(start_s=0, end_s=60, role="train"),
    RoleBlock(start_s=72, end_s=92, role="validation"),
    RoleBlock(start_s=104, end_s=124, role="test"),
)
_FRAME = 1000
_RIGHT_ZONE = """
schema_version: 1
view_id: pit
site_id: null
camera_id: null
equipment_family: null
measured: false
working_area:
  - {x: 0.5, y: 0.0}
  - {x: 1.0, y: 0.0}
  - {x: 1.0, y: 1.0}
  - {x: 0.5, y: 1.0}
dig_zone: []
loading_zone: []
entry_zone: []
exit_zone: []
"""
_DIG_ONLY = """
schema_version: 1
view_id: pit
site_id: null
camera_id: null
equipment_family: null
measured: false
working_area: []
dig_zone:
  - {x: 0.0, y: 0.0}
  - {x: 0.5, y: 0.0}
  - {x: 0.5, y: 1.0}
  - {x: 0.0, y: 1.0}
loading_zone: []
entry_zone: []
exit_zone: []
"""


def _forbid_propose(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("propose_split")


def _freeze(tmp_path: Path, clip: Path | None = None) -> Path:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    if clip is None:
        storage.insert_video(workspace, "pilot", "vid", name="pilot")
    else:
        storage.replace_extra_roots(workspace, "pilot", [clip.parent])
        storage.insert_video(workspace, "pilot", "vid", path=clip, name=clip.name)
    path = write_blocks(workspace, "pilot", "vid", list(_BLOCKS))
    freeze_split(workspace, "pilot", path)
    return workspace


def _clock(clip: Path) -> list[tuple[int, float]]:
    found: list[tuple[int, float]] = []
    for frame in FrameReader(clip, video_id="vid").iterate():
        if frame.decode_status is not DecodeStatus.ok:
            continue
        if frame.source_frame_index is None or frame.source_time_s is None:
            continue
        found.append((frame.source_frame_index, float(frame.source_time_s)))
    return found


def _between(
    frames: list[tuple[int, float]],
    start_s: float,
    end_s: float,
) -> list[tuple[int, float]]:
    return [item for item in frames if start_s <= item[1] < end_s]


def _box(x1: float, y1: float, x2: float, y2: float) -> dict[str, float | int]:
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "frame_width": _FRAME,
        "frame_height": _FRAME,
    }


def _shifted(motion: float) -> dict[str, float | int]:
    center_x = (0.1 + motion) * _FRAME
    return _box(center_x - 100, 0.0, center_x + 100, 200.0)


def _det(
    index: int,
    time_s: float,
    **updates: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": "run-a",
        "video_id": "vid",
        "source_frame_index": index,
        "source_time_s": time_s,
        "class_name": "haul_truck",
        "class_id": 2,
        "score": 0.9,
        "observation_state": "observed",
        "model_version": _MODEL,
        "track_id_nullable": None,
        "episode_id": "train-ep",
        "view_id": None,
        "split": "train",
        **_box(0.0, 0.0, 40.0, 40.0),
    }
    row.update(updates)
    return row


def _status(
    index: int,
    time_s: float,
    *,
    bucket: str = "observed",
    view_id: str | None = None,
    episode_id: str = "train-ep",
) -> dict[str, object]:
    return {
        "run_id": "run-a",
        "video_id": "vid",
        "source_frame_index": index,
        "source_time_s": time_s,
        "analyzed": True,
        "bucket_observation_state": bucket,
        "view_id": view_id,
        "model_version": _MODEL,
        "width": _FRAME,
        "height": _FRAME,
        "episode_id": episode_id,
        "split": "train",
    }


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _plant_coco(workspace: Path) -> Path:
    path = (
        storage.project_directory(workspace, "pilot")
        / "datasets"
        / "v1"
        / "train"
        / "_annotations.coco.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text('{"annotations":[{"bbox":[0,0,1,1]}]}\n', encoding="utf-8")
    return path


def _invoke(workspace: Path, *extra: str):
    result = _RUNNER.invoke(
        app,
        [
            "dataset",
            "sample",
            "--project",
            "pilot",
            "--workspace",
            str(workspace),
            "--json",
            *extra,
        ],
    )
    return result


@pytest.fixture(scope="module")
def long_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("active") / "clip.mp4"
    write_synthetic_mp4(path, 130, 2)
    return path


def test_active_module_does_not_call_propose_split_or_a_detector() -> None:
    source = Path(active_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules: list[str] = []
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    assert "require_frozen" in called
    assert "propose_split" not in called
    assert "streamlit" not in modules
    assert not any(
        name == "excavation_lab.analytics" or name.startswith("excavation_lab.analytics.")
        for name in modules
    )
    assert "shell=True" not in source
    assert "ByteTrack" not in source
    assert "load_bundle" not in source
    assert "DetectorAdapter" not in source
    assert "dig_zone" not in source
    assert "0.5" not in source
    assert set(active_mod.ACTIVE_REASONS) == _ACTIVE_REASONS


def test_active_without_a_model_stays_unknown_and_other_splits_are_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(LabError) as missing:
        sample_frames(tmp_path, "pilot", split="train", count=1, strategy="active")
    assert missing.value.code == "unknown_strategy"
    with pytest.raises(LabError) as held_out:
        sample_frames(
            tmp_path,
            "pilot",
            split="test",
            strategy="active",
            model=_MODEL,
        )
    assert held_out.value.code == "sample_split"
    with pytest.raises(LabError) as planned:
        sample_frames(
            tmp_path,
            "pilot",
            split="train",
            plan="diagnostic",
            strategy="active",
            model=_MODEL,
        )
    assert planned.value.code == "sample_args"


def test_shown_split_does_not_treat_a_missing_role_as_train(tmp_path: Path) -> None:
    workspace = _freeze(tmp_path)
    frozen = require_frozen(workspace, "pilot", "vid")
    assert _shown_split(frozen, "train-ep", 10.0) == "train"
    assert _shown_split(frozen, None, 10.0) == "train"
    assert _shown_split(frozen, "test-ep", 10.0) == "train"
    assert _shown_split(frozen, "train-ep", 110.0) == "test"
    assert _shown_split(frozen, "train-ep", 80.0) == "validation"
    assert _shown_split(frozen, "train-ep", 65.0) == "unknown"
    assert _shown_split(frozen, "train-ep", 124.0) == "unknown"
    assert _shown_split(frozen, "train-ep", 200.0) == "unknown"


def test_selector_keeps_train_failures_and_uses_jump_norm(tmp_path: Path) -> None:
    workspace = _freeze(tmp_path)
    frozen = require_frozen(workspace, "pilot", "vid")
    pilot = load_pilot()
    size = math.hypot(0.2, 0.2)
    above = pilot.jump_norm * size + 0.02
    below = pilot.jump_norm * size - 0.02
    base = _box(0.0, 0.0, 200.0, 200.0)
    detections = [
        _det(1, 10.0, track_id_nullable="jump", **base),
        _det(2, 12.0, track_id_nullable="jump", **_shifted(above)),
        _det(3, 14.0, track_id_nullable="still", **base),
        _det(4, 16.0, track_id_nullable="still", **_shifted(below)),
        _det(5, 18.0, score=0.2, observation_state="weak", class_name="bucket"),
        _det(6, 20.0, score=0.9, observation_state="observed"),
        _det(7, 22.0, score=0.05, observation_state=None),
        _det(8, 24.0, score=None, observation_state=None),
        _det(
            9,
            26.0,
            class_name="excavator_body",
            class_id=0,
            view_id="pit",
            **_box(650, 400, 850, 600),
        ),
        _det(
            10,
            28.0,
            class_name="excavator_body",
            class_id=0,
            view_id="pit",
            **_box(100, 400, 300, 600),
        ),
        _det(
            11,
            30.0,
            class_name="truck_bed",
            class_id=3,
            observation_state="ambiguous",
            score=0.95,
        ),
        _det(
            12,
            32.0,
            class_name="excavator_body",
            class_id=0,
            view_id=None,
            **_box(100, 400, 300, 600),
        ),
        _det(13, 10.0, episode_id="test-ep", score=0.2, observation_state="weak"),
        _det(14, 110.0, episode_id="train-ep", score=0.2, observation_state="weak"),
        _det(15, 80.0, episode_id="train-ep", score=0.2, observation_state="weak"),
        _det(16, 65.0, episode_id="train-ep", score=0.2, observation_state="weak"),
        _det(17, 124.0, episode_id="train-ep", score=0.2, observation_state="weak"),
        _det(18, 200.0, episode_id="train-ep", score=0.2, observation_state="weak"),
        _det(
            19,
            110.0,
            episode_id="train-ep",
            track_id_nullable="jump-test",
            **base,
        ),
        _det(
            20,
            112.0,
            episode_id="train-ep",
            track_id_nullable="jump-test",
            **_shifted(above),
        ),
    ]
    frames = [
        _status(index, time_s, episode_id=str(row["episode_id"]))
        for row in detections
        if isinstance(row["source_frame_index"], int)
        for index, time_s in ((row["source_frame_index"], row["source_time_s"]),)
    ]
    for frame, bucket, view in (
        (9, "missing", "pit"),
        (10, "missing", "pit"),
        (12, "missing", None),
        (14, "missing", "pit"),
    ):
        matched = next(item for item in frames if item["source_frame_index"] == frame)
        matched["bucket_observation_state"] = bucket
        matched["view_id"] = view
    frames.append(
        _status(21, 34.0, bucket="observed", episode_id="train-ep"),
    )
    zone_path = tmp_path / "pit.yaml"
    zone_path.write_text(_RIGHT_ZONE, encoding="utf-8")
    zones = {"pit": load_view_zones_file(zone_path)}
    false_positives = [
        {
            "video_id": "vid",
            "source_frame_index": 22,
            "source_time_s": 36.0,
            "episode_id": "train-ep",
            "split": "test",
            "class_name": "bucket",
            "model_version": _MODEL,
            "xyxy": [1, 2, 8, 9],
        },
        {
            "video_id": "vid",
            "source_frame_index": 23,
            "source_time_s": 110.0,
            "episode_id": "train-ep",
            "split": "train",
            "class_name": "bucket",
            "model_version": _MODEL,
            "xyxy": [1, 2, 8, 9],
        },
    ]
    tracks = [
        {
            "run_id": "run-a",
            "track_id": "bed-1",
            "class_name": "truck_bed",
            "source_time_s": 34.0,
            "state": "ambiguous",
            "identity_uncertain": True,
            "model_version": _MODEL,
        }
    ]
    selected = active_mod.select_train_frames(
        frozen,
        video_id="vid",
        model_id=_MODEL,
        detections=detections,
        frame_rows=frames,
        track_rows=tracks,
        false_positives=false_positives,
        zones_by_view=zones,
        video_view_id=None,
        jump_norm=pilot.jump_norm,
        detection_floor=pilot.detection_floor,
        display_threshold=pilot.display_threshold,
    )
    by_index = {pick.source_frame_index: pick for pick in selected.picks}
    assert by_index[2].selection_reason == "coordinate_jump"
    assert 4 not in by_index
    assert by_index[5].selection_reason == "low_confidence"
    assert 6 not in by_index
    assert 7 not in by_index
    assert 8 not in by_index
    assert by_index[9].selection_reason == "bucket_missing_in_working_area"
    assert 10 not in by_index
    assert by_index[11].selection_reason == "ambiguous_bed"
    assert 12 not in by_index
    assert by_index[13].selection_reason == "low_confidence"
    assert by_index[21].selection_reason == "ambiguous_bed"
    assert by_index[22].selection_reason == "false_positive"
    for excluded in (14, 15, 16, 17, 18, 19, 20, 23):
        assert excluded not in by_index
    assert all(pick.model_version == _MODEL for pick in selected.picks)
    assert selected.reason_reports == {
        "bucket_missing_in_working_area": "zone_missing",
    }
    tighter = active_mod.select_train_frames(
        frozen,
        video_id="vid",
        model_id=_MODEL,
        detections=[
            _det(3, 14.0, track_id_nullable="still", **base),
            _det(4, 16.0, track_id_nullable="still", **_shifted(below)),
        ],
        frame_rows=[_status(3, 14.0), _status(4, 16.0)],
        track_rows=[],
        false_positives=[],
        zones_by_view={},
        video_view_id=None,
        jump_norm=pilot.jump_norm * 0.5,
        detection_floor=pilot.detection_floor,
        display_threshold=pilot.display_threshold,
    )
    assert [pick.source_frame_index for pick in tighter.picks] == [4]
    assert tighter.picks[0].selection_reason == "coordinate_jump"

    dig_path = tmp_path / "dig.yaml"
    dig_path.write_text(_DIG_ONLY, encoding="utf-8")
    dig = load_view_zones_file(dig_path)
    assert len(dig.dig_zone) >= 3
    assert len(dig.working_area) < 3
    missed = active_mod.select_train_frames(
        frozen,
        video_id="vid",
        model_id=_MODEL,
        detections=[
            _det(
                12,
                32.0,
                class_name="excavator_body",
                class_id=0,
                view_id="pit",
                **_box(100, 400, 300, 600),
            )
        ],
        frame_rows=[_status(12, 32.0, bucket="missing", view_id="pit")],
        track_rows=[],
        false_positives=[],
        zones_by_view={"pit": dig},
        video_view_id=None,
        jump_norm=pilot.jump_norm,
        detection_floor=pilot.detection_floor,
        display_threshold=pilot.display_threshold,
    )
    assert missed.picks == ()
    assert missed.reason_reports == {
        "bucket_missing_in_working_area": "zone_missing",
    }


def test_command_samples_train_only_and_leaves_the_manifest(
    tmp_path: Path,
    long_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _freeze(tmp_path, long_clip)
    manifest = frozen_manifest_path(workspace, "pilot", "vid")
    before = manifest.read_bytes()
    content_hash = json.loads(before)["content_hash"]
    coco = _plant_coco(workspace)
    coco_before = coco.read_bytes()
    frames = _clock(long_clip)
    train = _between(frames, 1.0, 50.0)
    test = _between(frames, 106.0, 122.0)
    guard = _between(frames, 62.0, 70.0)
    validation = _between(frames, 74.0, 90.0)
    assert len(train) >= 16
    assert len(test) >= 6
    assert guard and validation
    pilot = load_pilot()
    size = math.hypot(0.2, 0.2)
    above = pilot.jump_norm * size + 0.02
    below = pilot.jump_norm * size - 0.02
    base = _box(0.0, 0.0, 200.0, 200.0)
    jump_a, jump_b = train[0], train[1]
    still_a, still_b = train[2], train[3]
    low, high, under, blank = train[4], train[5], train[6], train[7]
    inside, outside = train[8], train[9]
    bed, positive = train[10], train[11]
    negative, contradict = train[12], train[13]
    unzoned = train[14]
    test_low, test_jump_a, test_jump_b = test[0], test[1], test[2]
    test_inside, test_bed, test_positive = test[3], test[4], test[5]
    detections = [
        _det(*jump_a, track_id_nullable="jump", **base),
        _det(*jump_b, track_id_nullable="jump", **_shifted(above)),
        _det(*still_a, track_id_nullable="still", **base),
        _det(*still_b, track_id_nullable="still", **_shifted(below)),
        _det(*low, score=0.2, observation_state="weak", class_name="bucket", class_id=1),
        _det(*high, score=0.9, observation_state="observed"),
        _det(*under, score=0.05, observation_state=None),
        _det(*blank, score=None, observation_state=None),
        _det(
            *inside,
            class_name="excavator_body",
            class_id=0,
            view_id="pit",
            **_box(650, 400, 850, 600),
        ),
        _det(
            *outside,
            class_name="excavator_body",
            class_id=0,
            view_id="pit",
            **_box(100, 400, 300, 600),
        ),
        _det(
            *bed,
            class_name="truck_bed",
            class_id=3,
            observation_state="ambiguous",
            score=0.95,
        ),
        _det(*contradict, episode_id="test-ep", score=0.2, observation_state="weak"),
        _det(
            *unzoned,
            class_name="excavator_body",
            class_id=0,
            view_id=None,
            **_box(100, 400, 300, 600),
        ),
        _det(*test_low, episode_id="train-ep", score=0.2, observation_state="weak"),
        _det(*test_jump_a, episode_id="train-ep", track_id_nullable="jump-test", **base),
        _det(
            *test_jump_b,
            episode_id="train-ep",
            track_id_nullable="jump-test",
            **_shifted(above),
        ),
        _det(
            *test_inside,
            episode_id="train-ep",
            class_name="excavator_body",
            class_id=0,
            view_id="pit",
            **_box(650, 400, 850, 600),
        ),
        _det(
            *test_bed,
            episode_id="train-ep",
            class_name="truck_bed",
            class_id=3,
            observation_state="ambiguous",
            score=0.95,
        ),
        _det(*guard[0], episode_id="train-ep", score=0.2, observation_state="weak"),
        _det(*validation[0], episode_id="train-ep", score=0.2, observation_state="weak"),
        _det(90001, 124.0, episode_id="train-ep", score=0.2, observation_state="weak"),
        _det(90002, 200.0, episode_id="train-ep", score=0.2, observation_state="weak"),
    ]
    frame_rows = [
        _status(
            int(row["source_frame_index"]),
            float(row["source_time_s"]),
            episode_id=str(row["episode_id"]),
            view_id=row["view_id"] if isinstance(row["view_id"], str) else None,
        )
        for row in detections
    ]
    for index, view in ((inside[0], "pit"), (outside[0], "pit"), (test_inside[0], "pit")):
        matched = next(item for item in frame_rows if item["source_frame_index"] == index)
        matched["bucket_observation_state"] = "missing"
        matched["view_id"] = view
    unzoned_row = next(item for item in frame_rows if item["source_frame_index"] == unzoned[0])
    unzoned_row["bucket_observation_state"] = "missing"
    unzoned_row["view_id"] = None
    project_dir = storage.project_directory(workspace, "pilot")
    run_dir = project_dir / "runs" / "run-a"
    _write_parquet(run_dir / "detections.parquet", detections)
    _write_parquet(run_dir / "frame_status.parquet", frame_rows)
    (run_dir / "run.json").write_text(
        json.dumps({"model_id": _MODEL, "video_id": "vid"}) + "\n",
        encoding="utf-8",
    )
    eval_dir = project_dir / "runs" / "eval-a"
    eval_dir.mkdir(parents=True)
    (eval_dir / "metrics.json").write_text(
        json.dumps({"model_id": _MODEL}) + "\n",
        encoding="utf-8",
    )
    (eval_dir / "matches.json").write_text(
        json.dumps(
            {
                "model_id": _MODEL,
                "video_id": "vid",
                "false_positives": [
                    {
                        "source_frame_index": positive[0],
                        "source_time_s": positive[1],
                        "episode_id": "train-ep",
                        "split": "test",
                        "class_name": "bucket",
                        "xyxy": [1, 2, 8, 9],
                    },
                    {
                        "source_frame_index": test_positive[0],
                        "source_time_s": test_positive[1],
                        "episode_id": "train-ep",
                        "split": "train",
                        "class_name": "bucket",
                        "xyxy": [1, 2, 8, 9],
                    },
                    {
                        "source_frame_index": 90001,
                        "source_time_s": 124.0,
                        "episode_id": "train-ep",
                        "split": "train",
                        "class_name": "bucket",
                        "xyxy": [0, 0, 2, 2],
                    },
                    {
                        "source_frame_index": 90002,
                        "source_time_s": 200.0,
                        "episode_id": "train-ep",
                        "split": "train",
                        "class_name": "bucket",
                    },
                ],
                "false_negatives": [
                    {
                        "source_frame_index": negative[0],
                        "source_time_s": negative[1],
                        "episode_id": "train-ep",
                        "split": "train",
                        "class_name": "haul_truck",
                        "xyxy": [0, 0, 4, 4],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (project_dir / "views").mkdir()
    (project_dir / "views" / "pit.yaml").write_text(_RIGHT_ZONE, encoding="utf-8")
    monkeypatch.setattr(split_mod, "propose_split", _forbid_propose)
    result = _invoke(
        workspace,
        "--split",
        "train",
        "--strategy",
        "active",
        "--model",
        _MODEL,
    )
    assert result.exit_code == 0, (result.stdout, result.exception)
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    data = payload["data"]
    rows = data["rows"]
    by_index = {row["source_frame_index"]: row for row in rows}
    assert by_index[jump_b[0]]["selection_reason"] == "coordinate_jump"
    assert still_b[0] not in by_index
    assert by_index[low[0]]["selection_reason"] == "low_confidence"
    assert high[0] not in by_index
    assert under[0] not in by_index
    assert blank[0] not in by_index
    assert by_index[inside[0]]["selection_reason"] == "bucket_missing_in_working_area"
    assert outside[0] not in by_index
    assert unzoned[0] not in by_index
    assert by_index[bed[0]]["selection_reason"] == "ambiguous_bed"
    assert by_index[positive[0]]["selection_reason"] == "false_positive"
    assert by_index[contradict[0]]["selection_reason"] == "low_confidence"
    assert negative[0] not in by_index
    excluded = [
        test_low[0],
        test_jump_a[0],
        test_jump_b[0],
        test_inside[0],
        test_bed[0],
        test_positive[0],
        guard[0][0],
        validation[0][0],
        90001,
        90002,
    ]
    assert set(by_index).isdisjoint(excluded)
    assert {row["selection_reason"] for row in rows} == set(active_mod.ACTIVE_REASONS)
    for row in rows:
        assert row["split"] == "train"
        assert row["episode_id"] == ""
        assert row["strategy"] == "active"
        assert row["model_version"] == _MODEL
        assert row["video_id"] == "vid"
        assert 0 <= row["source_time_s"] < 60
    assert data["written"]["validation"] == 0
    assert data["written"]["test"] == 0
    assert data["reason_reports"]["bucket_missing_in_working_area"] == "zone_missing"
    summary = format_sample_summary(SampleResult.model_validate(data))
    assert "zone_missing" in summary
    assert "не отобрано" not in summary
    assert any(row["selection_reason"] == "bucket_missing_in_working_area" for row in rows)
    assert data["split_content_hash"] == content_hash
    images = Path(data["images_dir"])
    for row in rows:
        blob = (images / row["image_file"]).read_bytes()
        assert blob.startswith(b"\xff\xd8")
        assert hashlib.sha256(blob).hexdigest() == row["image_sha256"]
    assert manifest.read_bytes() == before
    assert json.loads(manifest.read_bytes())["content_hash"] == content_hash
    assert coco.read_bytes() == coco_before
    assert list(project_dir.rglob("_annotations.coco.json")) == [coco]
    sample_text = Path(data["manifest_path"]).read_text(encoding="utf-8")
    assert "bbox" not in sample_text
    assert '"score": 0' not in sample_text


def test_model_required_does_not_synthesize_boxes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _freeze(tmp_path)
    manifest = frozen_manifest_path(workspace, "pilot", "vid")
    before = manifest.read_bytes()
    monkeypatch.setattr(split_mod, "propose_split", _forbid_propose)
    result = _invoke(
        workspace,
        "--split",
        "train",
        "--strategy",
        "active",
        "--model",
        _MODEL,
    )
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "model_required"
    assert any("\u0400" <= char <= "\u04FF" for char in payload["error"]["message"])
    project_dir = storage.project_directory(workspace, "pilot")
    assert manifest.read_bytes() == before
    assert not (project_dir / "samples").exists()
    assert list(project_dir.rglob("detections.parquet")) == []
    assert list(project_dir.rglob("_annotations.coco.json")) == []
    assert list(project_dir.rglob("*.jpg")) == []

    baseline = _invoke(
        workspace,
        "--split",
        "train",
        "--strategy",
        "active",
        "--model",
        "baseline_coco",
    )
    assert baseline.exit_code == 1
    assert json.loads(baseline.stdout)["error"]["code"] == "model_required"
    assert manifest.read_bytes() == before


def test_custom_bundle_without_detections_does_not_synthesize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _freeze(tmp_path)
    manifest = frozen_manifest_path(workspace, "pilot", "vid")
    before = manifest.read_bytes()
    weights = _store_weights(workspace, _MODEL, b"not-a-real-checkpoint")
    monkeypatch.setattr(split_mod, "propose_split", _forbid_propose)
    result = _invoke(
        workspace,
        "--split",
        "train",
        "--strategy",
        "active",
        "--model",
        _MODEL,
    )
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "detections_missing"
    assert weights.read_bytes() == b"not-a-real-checkpoint"
    assert manifest.read_bytes() == before
    project_dir = storage.project_directory(workspace, "pilot")
    assert list(project_dir.rglob("_annotations.coco.json")) == []
    assert list(project_dir.rglob("*.jpg")) == []


def test_match_file_needs_a_bundle_and_then_keeps_only_train(
    tmp_path: Path,
    long_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _freeze(tmp_path, long_clip)
    manifest = frozen_manifest_path(workspace, "pilot", "vid")
    before = manifest.read_bytes()
    frames = _clock(long_clip)
    train = _between(frames, 5.0, 20.0)
    test = _between(frames, 108.0, 118.0)
    assert train and test
    project_dir = storage.project_directory(workspace, "pilot")
    eval_dir = project_dir / "runs" / "eval-only"
    eval_dir.mkdir(parents=True)
    (eval_dir / "matches.json").write_text(
        json.dumps(
            {
                "model_id": _MODEL,
                "video_id": "vid",
                "false_positives": [
                    {
                        "source_frame_index": train[0][0],
                        "source_time_s": train[0][1],
                        "episode_id": "train-ep",
                        "split": "test",
                        "class_name": "bucket",
                        "xyxy": [1, 1, 4, 4],
                    },
                    {
                        "source_frame_index": test[0][0],
                        "source_time_s": test[0][1],
                        "episode_id": "train-ep",
                        "split": "train",
                        "class_name": "bucket",
                        "xyxy": [1, 1, 4, 4],
                    },
                ],
                "false_negatives": [
                    {
                        "source_frame_index": train[1][0],
                        "source_time_s": train[1][1],
                        "episode_id": "train-ep",
                        "class_name": "bucket",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(split_mod, "propose_split", _forbid_propose)
    refused = _invoke(
        workspace,
        "--split",
        "train",
        "--strategy",
        "active",
        "--model",
        _MODEL,
    )
    assert refused.exit_code == 1
    assert json.loads(refused.stdout)["error"]["code"] == "model_required"
    assert list(project_dir.rglob("*.jpg")) == []
    assert manifest.read_bytes() == before

    weights = _store_weights(workspace, _MODEL, b"not-a-real-checkpoint")
    result = _invoke(
        workspace,
        "--split",
        "train",
        "--strategy",
        "active",
        "--model",
        _MODEL,
    )
    assert result.exit_code == 0, (result.stdout, result.exception)
    rows = json.loads(result.stdout)["data"]["rows"]
    assert [row["source_frame_index"] for row in rows] == [train[0][0]]
    assert rows[0]["selection_reason"] == "false_positive"
    assert rows[0]["split"] == SplitName.train.value
    assert rows[0]["model_version"] == _MODEL
    assert manifest.read_bytes() == before
    assert weights.read_bytes() == b"not-a-real-checkpoint"
    assert list(project_dir.rglob("_annotations.coco.json")) == []


    assert list(project_dir.rglob("_annotations.coco.json")) == []


def test_foreign_catalog_skips_bucket_and_bed_reasons(tmp_path: Path) -> None:
    workspace = _freeze(tmp_path)
    frozen = require_frozen(workspace, "pilot", "vid")
    pilot = load_pilot()
    detections = [
        _det(
            9,
            26.0,
            class_name="excavator_body",
            class_id=0,
            view_id="pit",
            **_box(650, 400, 850, 600),
        ),
        _det(
            11,
            30.0,
            class_name="truck_bed",
            class_id=3,
            observation_state="ambiguous",
            score=0.95,
        ),
        _det(5, 18.0, score=0.2, observation_state="weak", class_name="bucket"),
    ]
    before = [dict(row) for row in detections]
    frames = [
        _status(9, 26.0, bucket="missing", view_id="pit"),
        _status(11, 30.0),
        _status(5, 18.0),
    ]
    zone_path = tmp_path / "pit.yaml"
    zone_path.write_text(_RIGHT_ZONE, encoding="utf-8")
    zones = {"pit": load_view_zones_file(zone_path)}
    kwargs = {
        "video_id": "vid",
        "model_id": _MODEL,
        "detections": detections,
        "frame_rows": frames,
        "track_rows": [
            {
                "run_id": "run-a",
                "track_id": "bed-1",
                "class_name": "truck_bed",
                "source_time_s": 30.0,
                "state": "ambiguous",
                "identity_uncertain": True,
                "model_version": _MODEL,
            }
        ],
        "false_positives": [],
        "zones_by_view": zones,
        "video_view_id": None,
        "jump_norm": pilot.jump_norm,
        "detection_floor": pilot.detection_floor,
        "display_threshold": pilot.display_threshold,
    }
    foreign = active_mod.select_train_frames(
        frozen,
        class_names=frozenset({"shovel"}),
        **kwargs,
    )
    reasons = {pick.selection_reason for pick in foreign.picks}
    assert "bucket_missing_in_working_area" not in reasons
    assert "ambiguous_bed" not in reasons
    assert "bucket_missing_in_working_area" not in foreign.reason_reports
    assert "low_confidence" in reasons
    assert detections == before
    kept = active_mod.select_train_frames(
        frozen,
        class_names=frozenset({"bucket", "truck_bed"}),
        **kwargs,
    )
    kept_reasons = {pick.selection_reason for pick in kept.picks}
    assert "bucket_missing_in_working_area" in kept_reasons
    assert "ambiguous_bed" in kept_reasons


def test_sample_active_passes_the_project_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _freeze(tmp_path)
    seen: dict[str, object] = {}

    def evidence(*_args: object, **_kwargs: object) -> active_mod._Evidence:
        return active_mod._Evidence((), (), (), (), 1, 0)

    def capture(*_args: object, **kwargs: object) -> active_mod.ActiveSelection:
        seen["class_names"] = kwargs.get("class_names")
        return active_mod.ActiveSelection(picks=(), reason_reports={})

    monkeypatch.setattr(active_mod, "_load_evidence", evidence)
    monkeypatch.setattr(active_mod, "select_train_frames", capture)
    active_mod.sample_active(
        workspace,
        "pilot",
        model_id=_MODEL,
        video_id="vid",
        split="train",
        count=1,
    )
    assert seen["class_names"] == frozenset(
        {"excavator_body", "bucket", "haul_truck", "truck_bed"}
    )
    active_mod.sample_active(
        workspace,
        "pilot",
        model_id="baseline_coco",
        video_id="vid",
        split="train",
        count=1,
    )
    assert seen["class_names"] == frozenset()


def test_active_reports_a_tick_that_did_not_decode_as_shortfall(
    tmp_path: Path,
    long_clip: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses

    workspace = _freeze(tmp_path, long_clip)
    train = _between(_clock(long_clip), 2.0, 40.0)
    chosen = [train[0], train[6], train[12], train[18]]
    victim = chosen[2][0]
    picks = tuple(
        active_mod.ActivePick(
            video_id="vid",
            source_frame_index=index,
            source_time_s=time_s,
            selection_reason="low_confidence",
            model_version=_MODEL,
        )
        for index, time_s in chosen
    )

    def evidence(*_args: object, **_kwargs: object) -> active_mod._Evidence:
        return active_mod._Evidence((), (), (), (), 1, 0)

    def selection(*_args: object, **_kwargs: object) -> active_mod.ActiveSelection:
        return active_mod.ActiveSelection(picks=picks, reason_reports={})

    monkeypatch.setattr(active_mod, "_load_evidence", evidence)
    monkeypatch.setattr(active_mod, "select_train_frames", selection)
    original = FrameReader.iterate

    def blanked(self: FrameReader, start_s: float = 0.0, end_s: float | None = None):
        # Only the per-tick seek loses pixels; the tick pass still lists the frame.
        per_tick = end_s is not None and end_s - start_s < 1.0
        for frame in original(self, start_s, end_s):
            if per_tick and frame.source_frame_index == victim:
                yield dataclasses.replace(frame, image=None)
            else:
                yield frame

    monkeypatch.setattr(FrameReader, "iterate", blanked)
    result = active_mod.sample_active(
        workspace,
        "pilot",
        model_id=_MODEL,
        video_id="vid",
        split="train",
        count=None,
    )
    assert result.decode_failed == [victim]
    assert result.requested.train == 4
    assert result.written.train == 3
    assert result.shortfall.train == 1
    assert victim not in {row.source_frame_index for row in result.rows}
    assert {row.strategy for row in result.rows} == {"active"}
    assert result.tick_pass == "decode"
    assert result.tick_pass_reason is not None
    assert not (Path(result.images_dir) / f"vid_{victim}.jpg").exists()
    summary = format_sample_summary(result)
    assert "decode_failed" in summary
    assert str(victim) in summary
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["decode_failed"] == [victim]
    assert manifest["shortfall"]["train"] == 1


def _store_weights(workspace: Path, model_id: str, payload: bytes) -> Path:
    directory = workspace / "weights" / model_id
    directory.mkdir(parents=True)
    weights = directory / "weights.pth"
    weights.write_bytes(payload)
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            """
            INSERT INTO model_versions (
                model_id, project_id, dataset_id, variant, path, created_at
            ) VALUES (?, 'pilot', NULL, 'nano', ?, ?)
            """,
            (model_id, str(directory.resolve()), datetime.now(UTC).isoformat()),
        )
    return weights
