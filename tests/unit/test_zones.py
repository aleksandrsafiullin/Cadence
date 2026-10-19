"""Camera zones are normalized polygons. A view change drops the old one."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.zones import (
    ZONE_INVALID_FOR_VIEW,
    ZONE_VALID,
    CameraZones,
    ZoneBinding,
    active_working_area_pixels,
    annotation_polygons,
    load_view_zones,
    load_view_zones_file,
    point_in_polygon,
    polygon_pixels,
)
from excavation_lab.errors import LabError
from excavation_lab.models.adapter import CheckpointOrigin, FrameDetections
from excavation_lab.schemas import ObservationState, example_view_path
from excavation_lab.storage import (
    connect,
    create_project,
    get_analysis_run,
    get_video,
    insert_analysis_run,
    insert_video,
    project_sqlite,
)
from excavation_lab.tracking.bucket import (
    CONTAINMENT_BONUS,
    BucketAssociator,
    normalized_center_distance,
    pair_score,
)
from excavation_lab.tracking.session import UNSET_VIEW_ID, public_track_id
from excavation_lab.visualization.annotate import (
    AnnotateToggles,
    ZonePolygon,
    annotate_rgb,
)

_FRAME_W = 1000
_FRAME_H = 800
_BODY = 500.0
_SCORE = 0.9
_Y = 400.0
_RUN = "run-z"
_VIDEO = "clip"
_ZONE_NAMES = (
    "working_area",
    "dig_zone",
    "loading_zone",
    "entry_zone",
    "exit_zone",
)


@dataclass(frozen=True, slots=True)
class _Body:
    track_id: str
    xyxy: tuple[float, float, float, float]
    class_name: str = "excavator_body"


def test_view_file_loads_identity_and_five_polygons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaders: list[object] = []
    real_load = yaml.load

    def spy(stream: object, loader: object = None) -> object:
        loaders.append(loader)
        return real_load(stream, loader)  # type: ignore[arg-type]

    monkeypatch.setattr(yaml, "load", spy)
    shifts = {
        "working_area": 0.0,
        "dig_zone": 0.15,
        "loading_zone": 0.30,
        "entry_zone": 0.45,
        "exit_zone": 0.60,
    }
    _write_view(
        tmp_path,
        "pit-east",
        site_id="north-pit",
        camera_id="cam-7",
        equipment_family="hydraulic_shovel",
        **{name: _triangle(shift) for name, shift in shifts.items()},
    )
    loaded = load_view_zones("pit-east", directory=tmp_path)
    assert loaded.status == ZONE_VALID
    assert loaded.view_id == "pit-east"
    assert loaded.site_id == "north-pit"
    assert loaded.camera_id == "cam-7"
    assert loaded.equipment_family == "hydraulic_shovel"
    assert loaded.site_id != "quarry"
    for name, shift in shifts.items():
        assert loaded.polygon(name) == tuple((point["x"], point["y"]) for point in _triangle(shift))
    assert len({loaded.polygon(name) for name in shifts}) == len(shifts)

    template = load_view_zones_file(example_view_path())
    assert template.view_id == "example"
    assert template.site_id is None
    assert template.camera_id is None
    assert template.equipment_family is None
    assert template.measured is False
    for name in _ZONE_NAMES:
        assert template.polygon(name) == ()
    assert active_working_area_pixels(template, frame_width=640, frame_height=360) is None
    assert loaders
    assert all(loader is yaml.SafeLoader for loader in loaders)


def test_polygons_reject_range_line_and_zero_area(tmp_path: Path) -> None:
    _write_view(
        tmp_path,
        "too-far",
        working_area=[
            {"x": 0.0, "y": 0.0},
            {"x": 1.2, "y": 0.0},
            {"x": 0.0, "y": 1.0},
        ],
    )
    with pytest.raises(LabError) as out_of_range:
        load_view_zones("too-far", directory=tmp_path)
    assert out_of_range.value.code == "invalid_zone"
    assert "от 0 до 1" in out_of_range.value.message

    _write_view(
        tmp_path,
        "line",
        working_area=[{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}],
    )
    with pytest.raises(LabError) as line:
        load_view_zones("line", directory=tmp_path)
    assert line.value.code == "invalid_zone"
    assert "трёх" in line.value.message

    _write_view(
        tmp_path,
        "spike",
        working_area=[
            {"x": 0.0, "y": 0.2},
            {"x": 0.8, "y": 0.2},
            {"x": 0.4, "y": 0.2},
        ],
    )
    with pytest.raises(LabError) as spike:
        load_view_zones("spike", directory=tmp_path)
    assert spike.value.code == "invalid_zone"
    assert "нулю" in spike.value.message


def test_pixels_use_the_frame_width() -> None:
    vertices = ((0.5, 0.25), (1.0, 0.25), (1.0, 0.75))
    narrow = polygon_pixels(vertices, frame_width=640, frame_height=360)
    wide = polygon_pixels(vertices, frame_width=800, frame_height=360)
    assert narrow[0] == (320.0, 90.0)
    assert wide[0] == (400.0, 90.0)
    assert narrow[0][0] != 0.5 * 1920
    assert wide[0][0] != 0.5 * 1920


def test_working_area_is_the_polygon_on_either_half(tmp_path: Path) -> None:
    right = _load_half(tmp_path, "right-half", 0.5, 1.0)
    left = _load_half(tmp_path, "left-half", 0.0, 0.5)
    assert point_in_polygon(0.75, 0.5, right.working_area)
    assert not point_in_polygon(0.25, 0.5, right.working_area)
    assert point_in_polygon(0.25, 0.5, left.working_area)
    assert not point_in_polygon(0.75, 0.5, left.working_area)

    right_pixels = active_working_area_pixels(right, frame_width=_FRAME_W, frame_height=_FRAME_H)
    assert right_pixels is not None
    assert right_pixels[0][0] == pytest.approx(0.5 * _FRAME_W)
    assert right_pixels[0][0] != pytest.approx(0.5 * 1920)

    bare_right = _associate(None, outside_x=495, inside_x=630, bucket_x=520, view_id="right-half")
    zoned_right = _associate(
        right,
        outside_x=495,
        inside_x=630,
        bucket_x=520,
        view_id="right-half",
    )
    assert bare_right.observations[0].parent_track_id == "outside"
    assert zoned_right.observations[0].parent_track_id == "inside"
    assert zoned_right.observations[0].parent_track_id != "outside"

    bare_left = _associate(None, outside_x=505, inside_x=370, bucket_x=480, view_id="left-half")
    zoned_left = _associate(
        left,
        outside_x=505,
        inside_x=370,
        bucket_x=480,
        view_id="left-half",
    )
    assert bare_left.observations[0].parent_track_id == "outside"
    assert zoned_left.observations[0].parent_track_id == "inside"
    assert zoned_left.observations[0].association_state is ObservationState.observed


def test_view_change_drops_the_previous_polygon(tmp_path: Path) -> None:
    east = _load_half(tmp_path, "east", 0.5, 1.0)
    binding = ZoneBinding(current=east)
    associator = BucketAssociator.from_pilot()
    usable = binding.bind_view("east")
    assert usable is not None
    assert usable.status == ZONE_VALID
    first = _update(
        associator,
        usable,
        outside_x=495,
        inside_x=630,
        bucket_x=520,
        view_id="east",
        stamp=0.0,
    )
    assert first.observations[0].parent_track_id == "inside"
    assert first.observations[0].bucket_observation_id == f"{_RUN}:{_VIDEO}:east:bucket:0"
    assert first.observations[0].bucket_observation_id == public_track_id(
        _RUN, _VIDEO, "east", "bucket", 0
    )

    passed = binding.bind_view("west")
    assert passed is None
    assert binding.dropped is not None
    assert binding.dropped.status == ZONE_INVALID_FOR_VIEW
    assert binding.dropped.view_id == "east"
    assert binding.dropped.working_area == east.working_area
    assert active_working_area_pixels(
        binding.dropped,
        frame_width=_FRAME_W,
        frame_height=_FRAME_H,
    ) is None
    associator.reset()
    second = _update(
        associator,
        passed,
        outside_x=495,
        inside_x=630,
        bucket_x=520,
        view_id="west",
        stamp=1.0,
    )
    assert second.observations[0].parent_track_id == "outside"
    assert second.observations[0].parent_track_id != "inside"
    assert second.observations[0].bucket_observation_id == f"{_RUN}:{_VIDEO}:west:bucket:1"

    ignored = _update(
        BucketAssociator.from_pilot(),
        binding.dropped,
        outside_x=495,
        inside_x=630,
        bucket_x=520,
        view_id="west",
        stamp=0.0,
    )
    assert ignored.observations[0].parent_track_id == "outside"


def test_containment_bonus_does_not_break_an_ambiguous_tie() -> None:
    bucket = (0.0, 0.0, 10.0, 10.0)
    worse_box = (-33.0, -45.0, 67.0, 55.0)
    better_box = (-40.0, -45.0, 60.0, 55.0)
    zones = CameraZones(
        view_id="tight",
        site_id=None,
        camera_id=None,
        equipment_family=None,
        working_area=((0.0, 0.0), (0.14, 0.0), (0.14, 0.12), (0.0, 0.12)),
        dig_zone=(),
        loading_zone=(),
        entry_zone=(),
        exit_zone=(),
    )
    pixels = active_working_area_pixels(zones, frame_width=100, frame_height=100)
    assert pixels is not None
    assert point_in_polygon(5.0, 5.0, pixels)
    assert point_in_polygon(10.0, 5.0, pixels)
    assert not point_in_polygon(17.0, 5.0, pixels)

    associator = BucketAssociator.from_pilot()
    worse_score = pair_score(
        _SCORE,
        normalized_center_distance(bucket, worse_box),
        continued=False,
    )
    better_score = pair_score(
        _SCORE,
        normalized_center_distance(bucket, better_box),
        continued=False,
    )
    gap = better_score - worse_score
    assert gap > 0.0
    assert gap <= associator.ambiguity_margin
    assert better_score + CONTAINMENT_BONUS - worse_score > associator.ambiguity_margin

    detections = _frame(bucket, 100, 100, 1.0)
    before = np.array(detections.xyxy, copy=True)
    result = associator.update(
        detections,
        bodies=(_Body("worse", worse_box), _Body("better", better_box)),
        source_time_s=1.0,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id="tight",
        zones=zones,
    )
    assert np.array_equal(detections.xyxy, before)
    observation = result.observations[0]
    assert result.bucket_observation_state is ObservationState.ambiguous
    assert observation.association_state is ObservationState.ambiguous
    assert observation.parent_track_id is None
    assert observation.parent_track_id != "better"
    assert observation.association_score is None
    assert observation.reason is not None
    assert observation.reason.value == "ambiguous_association"


def test_missing_view_uses_the_unset_track_token() -> None:
    result = BucketAssociator.from_pilot().update(
        _frame((0.0, 0.0, 10.0, 10.0), 100, 100, 0.0),
        bodies=(_Body("body", (0.0, 0.0, 40.0, 40.0)),),
        source_time_s=0.0,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=None,
        zones=None,
    )
    assert result.observations[0].bucket_observation_id == f"{_RUN}:{_VIDEO}:{UNSET_VIEW_ID}:bucket:0"
    assert UNSET_VIEW_ID == "unset"


def test_video_and_run_store_null_identity(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    create_project("pilot", workspace, catalog_id="excavation")
    insert_video(workspace, "pilot", "bare")
    bare = get_video(workspace, "pilot", "bare")
    assert bare["site_id"] is None
    assert bare["camera_id"] is None
    assert bare["view_id"] is None
    assert bare["equipment_family"] is None
    assert "quarry" not in bare.values()

    insert_video(
        workspace,
        "pilot",
        "placed",
        site_id="north-pit",
        camera_id="cam-7",
        view_id="east",
        equipment_family="hydraulic_shovel",
    )
    placed = get_video(workspace, "pilot", "placed")
    assert placed["site_id"] == "north-pit"
    assert placed["camera_id"] == "cam-7"
    assert placed["view_id"] == "east"
    assert placed["equipment_family"] == "hydraulic_shovel"
    assert get_video(workspace, "pilot", "bare")["site_id"] is None

    with pytest.raises(LabError) as blank:
        insert_video(workspace, "pilot", "blank-site", site_id="")
    assert blank.value.code == "invalid_video"

    with connect(project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                job_id, project_id, status, pid, partial,
                analyzed_end_s, error_code, created_at
            ) VALUES (?, ?, 'completed', NULL, 0, NULL, NULL, ?)
            """,
            ("job-1", "pilot", "2026-10-10T00:00:00+00:00"),
        )
    _insert_run(workspace, "run-null", "bare", view_id=None)
    unknown = get_analysis_run(workspace, "pilot", "run-null")
    assert unknown["view_id"] is None
    assert unknown["site_id"] is None
    assert unknown["camera_id"] is None
    assert unknown["equipment_family"] is None

    _insert_run(
        workspace,
        "run-east",
        "placed",
        view_id="east",
        site_id="north-pit",
        camera_id="cam-7",
        equipment_family="hydraulic_shovel",
    )
    known = get_analysis_run(workspace, "pilot", "run-east")
    assert known["view_id"] == "east"
    assert known["site_id"] == "north-pit"
    assert known["camera_id"] == "cam-7"
    assert known["equipment_family"] == "hydraulic_shovel"
    assert get_analysis_run(workspace, "pilot", "run-null")["site_id"] is None

    with pytest.raises(LabError) as empty_family:
        _insert_run(
            workspace,
            "run-blank",
            "bare",
            view_id=None,
            equipment_family=" ",
        )
    assert empty_family.value.code == "invalid_run"


def test_annotation_hook_allows_an_empty_zone_list(tmp_path: Path) -> None:
    image = np.zeros((_FRAME_H // 10, _FRAME_W // 10, 3), dtype=np.uint8)
    toggles = AnnotateToggles(
        boxes=False,
        class_names=False,
        scores=False,
        track_ids=False,
        zones=True,
    )
    untouched = annotate_rgb(image, [], toggles=toggles, zones=())
    assert np.array_equal(untouched, image)

    zones = _load_half(tmp_path, "draw-east", 0.5, 1.0)
    width = int(image.shape[1])
    height = int(image.shape[0])
    specs = annotation_polygons(zones, frame_width=width, frame_height=height)
    assert specs
    assert specs[0][0] == "working_area"
    assert specs[0][1][0][0] == pytest.approx(0.5 * width)
    painted = annotate_rgb(
        image,
        [],
        toggles=toggles,
        zones=(ZonePolygon(specs[0][0], specs[0][1]),),
    )
    assert not np.array_equal(painted, image)
    assert annotation_polygons(zones.invalidated(), frame_width=width, frame_height=height) == ()


def _load_half(directory: Path, view_id: str, x0: float, x1: float) -> CameraZones:
    _write_view(directory, view_id, working_area=_rect(x0, x1))
    return load_view_zones(view_id, directory=directory)


def _associate(
    zones: CameraZones | None,
    *,
    outside_x: float,
    inside_x: float,
    bucket_x: float,
    view_id: str,
):
    return _update(
        BucketAssociator.from_pilot(),
        zones,
        outside_x=outside_x,
        inside_x=inside_x,
        bucket_x=bucket_x,
        view_id=view_id,
        stamp=0.0,
    )


def _update(
    associator: BucketAssociator,
    zones: CameraZones | None,
    *,
    outside_x: float,
    inside_x: float,
    bucket_x: float,
    view_id: str,
    stamp: float,
):
    bucket = _box(bucket_x, _Y, 40.0)
    outside = _box(outside_x, _Y, _BODY)
    inside = _box(inside_x, _Y, _BODY)
    outside_score = pair_score(
        _SCORE,
        normalized_center_distance(bucket, outside),
        continued=False,
    )
    inside_score = pair_score(
        _SCORE,
        normalized_center_distance(bucket, inside),
        continued=False,
    )
    margin = associator.ambiguity_margin
    assert outside_score - inside_score > margin
    assert inside_score + CONTAINMENT_BONUS - outside_score > margin
    detections = _frame(bucket, _FRAME_W, _FRAME_H, stamp)
    before = np.array(detections.xyxy, copy=True)
    result = associator.update(
        detections,
        bodies=(_Body("outside", outside), _Body("inside", inside)),
        source_time_s=stamp,
        run_id=_RUN,
        video_id=_VIDEO,
        view_id=view_id,
        zones=zones,
    )
    assert np.array_equal(detections.xyxy, before)
    assert detections.class_name == ("bucket",)
    return result


def _frame(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    stamp: float,
) -> FrameDetections:
    return FrameDetections(
        timestamp_s=stamp,
        frame_width=width,
        frame_height=height,
        origin=CheckpointOrigin.custom,
        xyxy=np.array([box], dtype=np.float64),
        class_id=np.array([1], dtype=np.int64),
        class_name=("bucket",),
        score=np.array([_SCORE], dtype=np.float64),
    )


def _box(center_x: float, center_y: float, side: float) -> tuple[float, float, float, float]:
    half = side / 2.0
    return (center_x - half, center_y - half, center_x + half, center_y + half)


def _rect(x0: float, x1: float) -> list[dict[str, float]]:
    return [
        {"x": x0, "y": 0.0},
        {"x": x1, "y": 0.0},
        {"x": x1, "y": 1.0},
        {"x": x0, "y": 1.0},
    ]


def _triangle(shift: float) -> list[dict[str, float]]:
    return [
        {"x": 0.10 + shift, "y": 0.10},
        {"x": 0.20 + shift, "y": 0.10},
        {"x": 0.15 + shift, "y": 0.30},
    ]


def _write_view(directory: Path, view_id: str, **overrides: object) -> None:
    document: dict[str, object] = {
        "schema_version": 1,
        "view_id": view_id,
        "site_id": None,
        "camera_id": None,
        "equipment_family": None,
        "measured": False,
        "working_area": [],
        "dig_zone": [],
        "loading_zone": [],
        "entry_zone": [],
        "exit_zone": [],
    }
    document.update(overrides)
    path = directory / f"{view_id}.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _insert_run(
    workspace: Path,
    run_id: str,
    video_id: str,
    *,
    view_id: str | None,
    site_id: str | None = None,
    camera_id: str | None = None,
    equipment_family: str | None = None,
) -> None:
    insert_analysis_run(
        workspace,
        "pilot",
        run_id=run_id,
        job_id="job-1",
        video_id=video_id,
        view_id=view_id,
        model_id="model-1",
        model_origin="custom",
        analysis_fps=5.0,
        source_fps=None,
        export_fps=None,
        start_s=0.0,
        end_s=None,
        decode_ms=None,
        encode_ms=None,
        site_id=site_id,
        camera_id=camera_id,
        equipment_family=equipment_family,
    )
