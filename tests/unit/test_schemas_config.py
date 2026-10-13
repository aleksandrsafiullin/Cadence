"""Labels, pilot settings, the example view, and serialized record versions."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from excavation_lab import schemas
from excavation_lab.errors import LabError


def test_labels_list_four_classes() -> None:
    labels = schemas.load_labels()
    dumped = labels.model_dump()
    assert dumped["schema_version"] == schemas.SCHEMA_VERSION == 1
    assert [(item.id, item.name.value, item.title) for item in labels.classes] == [
        (0, "excavator_body", "Экскаватор: корпус"),
        (1, "bucket", "Ковш"),
        (2, "haul_truck", "Самосвал"),
        (3, "truck_bed", "Кузов самосвала"),
    ]
    assert dumped["classes"][0]["name"] == "excavator_body"
    assert [item.value for item in labels.visibility] == [
        "clear",
        "partial",
        "unlabelable",
    ]
    assert [item.value for item in labels.review_status] == [
        "unreviewed",
        "reviewed",
        "excluded",
    ]
    assert tuple(labels.attributes) == schemas.LABEL_ATTRIBUTES
    names = {item.name.value for item in labels.classes}
    assert names == {"excavator_body", "bucket", "haul_truck", "truck_bed"}
    assert labels.title_for(schemas.DetectionClassName.bucket) == "Ковш"


def test_unknown_detection_class_is_rejected() -> None:
    raw = schemas.read_yaml_mapping(schemas.labels_path())
    raw["classes"].append({"id": 1, "name": "dump", "title": "Разгрузка"})
    with pytest.raises(ValidationError, match="dump"):
        schemas.LabelsConfig.model_validate(raw)

    payload = _detection_payload(class_name="dump")
    with pytest.raises(ValidationError, match="dump"):
        schemas.Detection.model_validate(payload)


def test_duplicate_class_ids_are_rejected() -> None:
    raw = schemas.read_yaml_mapping(schemas.labels_path())
    raw["classes"].append(
        {"id": 0, "name": "excavator_body", "title": "Экскаватор: корпус"}
    )
    with pytest.raises(ValidationError, match="duplicate class id"):
        schemas.LabelsConfig.model_validate(raw)


def test_class_id_is_tied_to_name() -> None:
    raw = schemas.read_yaml_mapping(schemas.labels_path())
    raw["classes"][1]["id"] = 0
    with pytest.raises(ValidationError, match="bucket must use id 1"):
        schemas.LabelsConfig.model_validate(raw)

    with pytest.raises(ValidationError, match="bucket must use id 1"):
        schemas.Detection.model_validate(_detection_payload(class_id=0))


def test_pilot_starting_numbers() -> None:
    pilot = schemas.load_pilot()
    smoke = pilot.profiles.smoke
    pilot_small = pilot.train_profile("pilot-small")

    assert pilot.schema_version == 1
    assert pilot.jump_norm == 0.5
    assert pilot_small.run_test is False
    assert smoke.epochs == 2
    assert smoke.run_test is False

    assert pilot.analysis_fps == 5.0
    assert pilot.guard_gap_s == 10
    assert pilot.split_fractions.train == 0.60
    assert pilot.split_fractions.validation == 0.20
    assert pilot.split_fractions.test == 0.20
    assert pilot.split_fractions.unit == "episode_duration"
    assert pilot.sample_spacing_s == 2.0
    assert pilot.detection_floor == 0.10
    assert pilot.display_threshold == 0.35
    assert (pilot.sample_plans.diagnostic.train, pilot.sample_plans.diagnostic.validation) == (
        60,
        20,
    )
    assert pilot.sample_plans.diagnostic.test == 0
    assert (
        pilot.sample_plans.content.train,
        pilot.sample_plans.content.validation,
        pilot.sample_plans.content.test,
    ) == (200, 60, 60)
    assert (
        pilot.sample_plans.expanded.train,
        pilot.sample_plans.expanded.validation,
        pilot.sample_plans.expanded.test,
    ) == (360, 120, 120)

    assert pilot.bytetrack.high_confidence == 0.35
    assert pilot.bytetrack.activation == 0.45
    assert pilot.bytetrack.lost_buffer_s == 2.0
    assert pilot.bytetrack.reference_fps == 30
    assert pilot.bytetrack.min_observations == 2
    assert pilot.bucket.trail_s == 5.0
    assert pilot.bucket.trail_gap_s == 0.5
    assert pilot.bucket.extrapolation_s == 0.4
    assert pilot.bucket.ambiguity_margin == 0.1
    assert pilot.bed.min_containment == 0.5
    assert pilot.bed.ambiguity_margin == 0.1
    assert pilot.eval.score_floor == 0.001
    assert pilot.eval.iou == 0.5
    assert pilot.eval.max_detections == 100

    assert smoke.variant is schemas.ModelVariant.nano
    assert smoke.resolution == 384
    assert smoke.batch_size == 1
    assert smoke.gradient_accumulation == 1
    assert smoke.learning_rate == 1e-4
    assert smoke.encoder_learning_rate == 1e-5
    assert smoke.early_stopping is False
    assert smoke.patience is None
    assert smoke.min_delta is None
    assert smoke.best_skip == 0
    assert smoke.workers == 0
    assert smoke.amp_dtype is None
    assert smoke.seed == 42

    assert pilot_small.variant is schemas.ModelVariant.small
    assert pilot_small.resolution == 512
    assert pilot_small.epochs == 50
    assert pilot_small.batch_size == 1
    assert pilot_small.gradient_accumulation == 4
    assert pilot_small.learning_rate == 1e-4
    assert pilot_small.encoder_learning_rate == 1e-5
    assert pilot_small.early_stopping is True
    assert pilot_small.patience == 12
    assert pilot_small.min_delta == 0.001
    assert pilot_small.best_skip == 3
    assert pilot_small.workers == 0
    assert pilot_small.amp_dtype is None
    assert pilot_small.seed == 42

    dumped = pilot.model_dump()
    assert dumped["schema_version"] == 1
    assert dumped["profiles"]["pilot-small"]["run_test"] is False
    again = schemas.PilotConfig.model_validate(dumped)
    assert again.train_profile("pilot-small").run_test is False


def test_amp_dtype_null_is_fp32() -> None:
    text = schemas.pilot_path().read_text(encoding="utf-8")
    assert "fp16" not in text
    assert "bf16" not in text
    raw = schemas.read_yaml_mapping(schemas.pilot_path())
    assert raw["profiles"]["smoke"]["amp_dtype"] is None
    assert raw["profiles"]["pilot-small"]["amp_dtype"] is None
    raw["profiles"]["smoke"]["amp_dtype"] = "fp16"
    with pytest.raises(ValidationError, match="FP32"):
        schemas.PilotConfig.model_validate(raw)


def test_unknown_profile_and_variant_are_rejected() -> None:
    pilot = schemas.load_pilot()
    with pytest.raises(LabError) as caught:
        pilot.train_profile("pilot-xl")
    assert caught.value.code == "unknown_profile"

    raw = schemas.read_yaml_mapping(schemas.pilot_path())
    raw["profiles"]["smoke"]["variant"] = "xl"
    with pytest.raises(ValidationError):
        schemas.PilotConfig.model_validate(raw)
    raw = schemas.read_yaml_mapping(schemas.pilot_path())
    raw["not_a_setting"] = 1
    with pytest.raises(ValidationError):
        schemas.PilotConfig.model_validate(raw)


def test_example_view_is_not_a_measured_camera() -> None:
    view = schemas.load_example_view()
    dumped = view.model_dump()
    assert dumped["schema_version"] == 1
    assert view.view_id == "example"
    assert view.site_id is None
    assert view.camera_id is None
    assert view.equipment_family is None
    assert view.measured is False
    assert view.is_measured_camera() is False
    for name in schemas.ZONE_NAMES:
        assert view.zone(name) == []
        assert dumped[name] == []


def test_view_rejects_a_short_or_out_of_range_polygon() -> None:
    raw = schemas.read_yaml_mapping(schemas.example_view_path())
    raw["working_area"] = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}]
    with pytest.raises(ValidationError, match="at least 3"):
        schemas.ViewConfig.model_validate(raw)

    raw = schemas.read_yaml_mapping(schemas.example_view_path())
    raw["working_area"] = [
        {"x": 0.0, "y": 0.0},
        {"x": 1.2, "y": 0.0},
        {"x": 0.0, "y": 1.0},
    ]
    with pytest.raises(ValidationError):
        schemas.ViewConfig.model_validate(raw)


def test_serialized_records_include_schema_version() -> None:
    frame = schemas.FrameIdentity(
        video_id="video-1",
        source_frame_index=3,
        source_pts=90000,
        time_base="1/30000",
        source_time_s=3.0,
        analysis_index=15,
        width=1920,
        height=1080,
    )
    assert frame.model_dump()["schema_version"] == 1
    assert frame.source_time_s == 3.0

    job = schemas.JobRecord(job_id="job-1", status=schemas.JobStatus.queued)
    dumped_job = job.model_dump()
    assert dumped_job["schema_version"] == 1
    assert dumped_job["status"] == "queued"
    assert dumped_job["analyzed_end_s"] is None

    detection = schemas.Detection.model_validate(_detection_payload())
    assert detection.model_dump()["schema_version"] == 1
    assert detection.model_dump()["parent_track_id"] is None

    track = schemas.TrackState(
        run_id="run-1",
        track_id="track-1",
        class_name="haul_truck",
        source_time_s=1.0,
        state=schemas.ObservationState.missing,
        is_observed=False,
    )
    dumped_track = track.model_dump()
    assert dumped_track["schema_version"] == 1
    assert dumped_track["parent_track_id"] is None
    assert dumped_track["x1"] is None

    with pytest.raises(ValidationError, match="schema_version"):
        schemas.LabelsConfig.model_validate(
            {**schemas.read_yaml_mapping(schemas.labels_path()), "schema_version": 2}
        )


def test_job_statuses() -> None:
    assert [status.value for status in schemas.JobStatus] == [
        "queued",
        "running",
        "completed",
        "failed",
        "cancel_requested",
        "cancelled",
    ]
    with pytest.raises(ValidationError, match="analyzed_end_s"):
        schemas.JobRecord(
            job_id="job-1",
            status=schemas.JobStatus.cancelled,
            partial=True,
        )
    with pytest.raises(ValidationError, match="not completed"):
        schemas.JobRecord(
            job_id="job-1",
            status=schemas.JobStatus.completed,
            partial=True,
            analyzed_end_s=1.5,
        )


def test_episode_interval_is_half_open() -> None:
    with pytest.raises(ValidationError, match="half-open"):
        schemas.Episode(
            episode_id="episode-1",
            video_id="video-1",
            split_group_id="group-1",
            start_s=1.0,
            end_s=1.0,
            label="ожидание",
        )
    episode = schemas.Episode(
        episode_id="episode-1",
        video_id="video-1",
        split_group_id="group-1",
        start_s=0.0,
        end_s=10.0,
        label="заезд",
    )
    assert episode.model_dump()["schema_version"] == 1


def test_schema_and_configs_omit_mass_throughput() -> None:
    banned = ("fill_factor", "fill factor", "t/h", "tonnes_per_hour", "тонн")
    paths = [
        Path(schemas.__file__),
        schemas.labels_path(),
        schemas.pilot_path(),
        schemas.example_view_path(),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text
        assert re.search(r"\btons\b", text) is None


def _detection_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": "run-1",
        "observation_id": "obs-1",
        "video_id": "video-1",
        "source_frame_index": 0,
        "source_time_s": 0.0,
        "class_id": 1,
        "class_name": "bucket",
        "score": 0.5,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 11.0,
        "y2": 12.0,
        "frame_width": 100,
        "frame_height": 80,
        "observation_state": "observed",
        "model_version": "model-1",
    }
    payload.update(overrides)
    return payload
