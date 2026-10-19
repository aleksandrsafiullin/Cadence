"""Pydantic models for labels, pilot settings, views, and stored rows.

Class 0 on the excavation seed is ``excavator_body``, not background.
A workspace catalog may use other names; id 0 is still not background.
COCO category ids are mapped by name elsewhere. ``amp_dtype: null`` is FP32.
Null measurements stay null. Adapter ``FrameDetections`` lives with
``DetectorAdapter``, not here.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from excavation_lab.errors import LabError

SCHEMA_VERSION = 1
PROFILE_SMOKE = "smoke"
PROFILE_PILOT_SMALL = "pilot-small"
PROFILE_OVERFIT = "overfit"
ZONE_NAMES = (
    "working_area",
    "dig_zone",
    "loading_zone",
    "entry_zone",
    "exit_zone",
)
LABEL_ATTRIBUTES = (
    "truncated",
    "instance_local_id",
    "annotator",
    "reviewed_at",
    "note",
)
CATALOG_CLASS_LIMIT = 16
_CLASS_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TIME_BASE_RE = re.compile(r"^[1-9][0-9]*/[1-9][0-9]*$")
_STRICT = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StrictModel(BaseModel):
    model_config = _STRICT


class VersionedModel(StrictModel):
    schema_version: int = SCHEMA_VERSION

    @field_validator("schema_version")
    @classmethod
    def _supported_schema_version(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {value}")
        return value


class DetectionClassName(StrEnum):
    excavator_body = "excavator_body"
    bucket = "bucket"
    haul_truck = "haul_truck"
    truck_bed = "truck_bed"


# Internal ids. Not COCO category_id, and 0 is not background.
CANONICAL_CLASS_IDS: dict[DetectionClassName, int] = {
    DetectionClassName.excavator_body: 0,
    DetectionClassName.bucket: 1,
    DetectionClassName.haul_truck: 2,
    DetectionClassName.truck_bed: 3,
}


class Visibility(StrEnum):
    clear = "clear"
    partial = "partial"
    unlabelable = "unlabelable"


class ReviewStatus(StrEnum):
    unreviewed = "unreviewed"
    reviewed = "reviewed"
    excluded = "excluded"


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancel_requested = "cancel_requested"
    cancelled = "cancelled"


class ObservationState(StrEnum):
    observed = "observed"
    weak = "weak"
    predicted = "predicted"
    missing = "missing"
    ambiguous = "ambiguous"


class ObservationReason(StrEnum):
    no_detection = "no_detection"
    low_confidence = "low_confidence"
    occluded_if_known = "occluded_if_known"
    out_of_frame_if_known = "out_of_frame_if_known"
    decode_gap = "decode_gap"
    ambiguous_association = "ambiguous_association"


class DecodeStatus(StrEnum):
    ok = "ok"
    corrupt = "corrupt"


class SplitName(StrEnum):
    train = "train"
    validation = "validation"
    test = "test"


class SplitIndependence(StrEnum):
    episode_split = "episode_split"
    same_scene = "same_scene"


class SplitLimitation(StrEnum):
    too_few_episodes = "too_few_episodes"


class ModelVariant(StrEnum):
    nano = "nano"
    small = "small"


def canonical_class_id(name: DetectionClassName) -> int:
    return CANONICAL_CLASS_IDS[name]


def _enum_values(kind: type[StrEnum]) -> list[str]:
    return [item.value for item in kind]


def _validate_time_base(value: str) -> str:
    # ffprobe rational. source_time_s is seconds; this string is not a float FPS.
    if _TIME_BASE_RE.fullmatch(value) is None:
        raise ValueError("time_base must be a positive rational numerator/denominator")
    return value


def _require_positive_box(x1: float, y1: float, x2: float, y2: float) -> None:
    coords = (x1, y1, x2, y2)
    if not all(math.isfinite(value) for value in coords):
        raise ValueError("box coordinates must be finite")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive width and height")


def _complete_box(
    coords: tuple[float | None, float | None, float | None, float | None],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = coords
    if x1 is None or y1 is None or x2 is None or y2 is None:
        raise ValueError("track box coordinates are all set or all null")
    return x1, y1, x2, y2


def _require_class_pair(name: DetectionClassName, class_id: int) -> None:
    expected = canonical_class_id(name)
    if class_id != expected:
        raise ValueError(f"{name.value} must use id {expected}, not {class_id}")


def _require_fixed_label_lists(
    visibility: list[Visibility],
    review_status: list[ReviewStatus],
    attributes: list[str],
) -> None:
    if _enum_values(Visibility) != [item.value for item in visibility]:
        raise ValueError("visibility must be clear, partial, unlabelable")
    expected_review = _enum_values(ReviewStatus)
    if expected_review != [item.value for item in review_status]:
        raise ValueError("review_status must be unreviewed, reviewed, excluded")
    if tuple(attributes) != LABEL_ATTRIBUTES:
        raise ValueError("label attributes do not match the spec list")


class LabelClass(StrictModel):
    id: int = Field(ge=0, le=3)
    name: DetectionClassName
    title: str = Field(min_length=1)
    box: str = Field(min_length=1)

    @model_validator(mode="after")
    def id_matches_name(self) -> Self:
        _require_class_pair(self.name, self.id)
        return self


class LabelsConfig(VersionedModel):
    classes: list[LabelClass]
    visibility: list[Visibility]
    review_status: list[ReviewStatus]
    attributes: list[str]

    @model_validator(mode="after")
    def classes_are_the_four(self) -> Self:
        ids = [item.id for item in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate class id")
        names = [item.name.value for item in self.classes]
        if len(names) != len(set(names)):
            raise ValueError("duplicate class name")
        if ids != [0, 1, 2, 3]:
            raise ValueError("classes must be listed in id order 0..3")
        _require_fixed_label_lists(self.visibility, self.review_status, self.attributes)
        return self

    def title_for(self, name: DetectionClassName) -> str:
        for item in self.classes:
            if item.name == name:
                return item.title
        raise LabError("unknown_class", "Неизвестный класс детекции.")


class CatalogClass(StrictModel):
    """One workspace-catalog class. Id 0 is a real class, not background.

    ``name`` is the COCO token. ``title`` and ``box`` are operator text.
    """

    id: int = Field(ge=0)
    name: str
    title: str = Field(min_length=1)
    box: str = Field(min_length=1)

    @field_validator("name", mode="before")
    @classmethod
    def _ascii_token(cls, value: object) -> str:
        if isinstance(value, str) and _CLASS_TOKEN_RE.fullmatch(value) is not None:
            return value
        raise ValueError("class name must be an ASCII token ^[a-z][a-z0-9_]{0,63}$")


class CatalogConfig(VersionedModel):
    """A catalog of 1 to 16 classes. Not the excavation seed.

    ``load_labels`` still returns ``LabelsConfig`` for the four seed names.
    """

    classes: list[CatalogClass]
    visibility: list[Visibility]
    review_status: list[ReviewStatus]
    attributes: list[str]

    @model_validator(mode="after")
    def classes_fit_the_catalog(self) -> Self:
        count = len(self.classes)
        if count < 1:
            raise ValueError("a catalog needs at least 1 class")
        if count > CATALOG_CLASS_LIMIT:
            raise ValueError(f"a catalog has at most {CATALOG_CLASS_LIMIT} classes")
        ids = [item.id for item in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate class id")
        names = [item.name for item in self.classes]
        if len(names) != len(set(names)):
            raise ValueError("duplicate class name")
        _require_fixed_label_lists(self.visibility, self.review_status, self.attributes)
        return self


class SplitCounts(StrictModel):
    train: int = Field(ge=0)
    validation: int = Field(ge=0)
    test: int = Field(ge=0)


class SamplePlans(StrictModel):
    diagnostic: SplitCounts
    content: SplitCounts
    expanded: SplitCounts


class SplitFractions(StrictModel):
    train: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)
    unit: Literal["episode_duration"] = "episode_duration"

    @model_validator(mode="after")
    def sums_to_one(self) -> Self:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        return self


class ByteTrackConfig(StrictModel):
    high_confidence: float = Field(ge=0, le=1)
    activation: float = Field(ge=0, le=1)
    lost_buffer_s: float = Field(gt=0)
    reference_fps: int = Field(gt=0)
    min_observations: int = Field(ge=1)

    @model_validator(mode="after")
    def activation_not_below_high_confidence(self) -> Self:
        if self.activation < self.high_confidence:
            raise ValueError("activation must be >= high_confidence")
        return self


class BucketConfig(StrictModel):
    trail_s: float = Field(gt=0)
    trail_gap_s: float = Field(gt=0)
    extrapolation_s: float = Field(gt=0, le=0.4)
    ambiguity_margin: float = Field(ge=0, lt=1)


class BedConfig(StrictModel):
    min_containment: float = Field(gt=0, le=1)
    ambiguity_margin: float = Field(ge=0, lt=1)


class EvalConfig(StrictModel):
    score_floor: float = Field(gt=0, lt=1)
    iou: float = Field(gt=0, le=1)
    max_detections: int = Field(ge=1)


class TrainProfile(StrictModel):
    """Project training profile. Do not pass this object raw into ``model.train``."""

    variant: ModelVariant
    resolution: int = Field(ge=1)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    gradient_accumulation: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    encoder_learning_rate: float = Field(gt=0)
    early_stopping: bool
    patience: int | None = None
    min_delta: float | None = None
    best_skip: int = Field(ge=0)
    workers: int = Field(ge=0)
    # None means FP32. v0.1 does not store fp16 or bf16 as the Mac default.
    amp_dtype: None = None
    seed: int
    run_test: bool
    # True only on the overfit diagnostic. Smoke and pilot-small stay false.
    diagnostic: bool = False

    @field_validator("amp_dtype", mode="before")
    @classmethod
    def _fp32_only(cls, value: object) -> None:
        if value is not None:
            raise ValueError("amp_dtype must be null (FP32)")

    @model_validator(mode="after")
    def early_stop_fields(self) -> Self:
        if self.early_stopping:
            if self.patience is None or self.min_delta is None:
                raise ValueError("early stopping requires patience and min_delta")
            if self.patience < 1:
                raise ValueError("patience must be >= 1")
            if self.min_delta <= 0:
                raise ValueError("min_delta must be > 0")
        elif self.patience is not None or self.min_delta is not None:
            raise ValueError(
                "patience and min_delta must be null when early stopping is off"
            )
        return self


class TrainProfiles(StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    smoke: TrainProfile
    pilot_small: TrainProfile = Field(alias=PROFILE_PILOT_SMALL)
    overfit: TrainProfile

    @model_validator(mode="after")
    def overfit_is_not_a_quality_profile(self) -> Self:
        if not self.overfit.diagnostic:
            raise ValueError("overfit must set diagnostic: true")
        if self.smoke.diagnostic or self.pilot_small.diagnostic:
            raise ValueError("smoke and pilot-small must not set diagnostic")
        return self


class PilotConfig(VersionedModel):
    analysis_fps: float = Field(gt=0)
    guard_gap_s: float = Field(ge=10)
    split_fractions: SplitFractions
    sample_plans: SamplePlans
    sample_spacing_s: float = Field(gt=0)
    detection_floor: float = Field(ge=0, le=1)
    display_threshold: float = Field(ge=0, le=1)
    # Null leaves same-class duplicates in place. Cross-class pairs are never suppressed.
    same_class_nms_iou: float | None = Field(default=None, gt=0, le=1)
    bytetrack: ByteTrackConfig
    bucket: BucketConfig
    bed: BedConfig
    jump_norm: float = Field(gt=0)
    eval: EvalConfig
    profiles: TrainProfiles

    @model_validator(mode="after")
    def floors_do_not_clip_eval(self) -> Self:
        if self.display_threshold < self.detection_floor:
            raise ValueError("display_threshold must be >= detection_floor")
        if self.eval.score_floor >= self.detection_floor:
            raise ValueError("eval score_floor must stay below detection_floor")
        return self

    def train_profile(self, name: str) -> TrainProfile:
        if name == PROFILE_SMOKE:
            return self.profiles.smoke
        if name == PROFILE_PILOT_SMALL:
            return self.profiles.pilot_small
        if name == PROFILE_OVERFIT:
            return self.profiles.overfit
        raise LabError("unknown_profile", f"Неизвестный профиль обучения: {name}")

    def quality_profiles(self) -> dict[str, TrainProfile]:
        """Profiles whose metrics may be quoted. The overfit diagnostic is omitted."""
        named = {
            PROFILE_SMOKE: self.profiles.smoke,
            PROFILE_PILOT_SMALL: self.profiles.pilot_small,
        }
        return {name: profile for name, profile in named.items() if not profile.diagnostic}


class ZoneVertex(StrictModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


def polygon_area(vertices: list[ZoneVertex]) -> float:
    """Absolute shoelace area in normalized coordinates."""
    total = 0.0
    count = len(vertices)
    for index, current in enumerate(vertices):
        nxt = vertices[(index + 1) % count]
        total += current.x * nxt.y - nxt.x * current.y
    return abs(total) / 2.0


def _require_zone(name: str, vertices: list[ZoneVertex]) -> None:
    if not vertices:
        return
    if len(vertices) < 3:
        raise ValueError(f"{name} needs at least 3 vertices")
    if polygon_area(vertices) <= 1e-12:
        raise ValueError(f"{name} has zero area")


class ViewConfig(VersionedModel):
    """Camera view. ``measured`` false is not a surveyed camera."""

    view_id: str = Field(min_length=1)
    site_id: str | None = None
    camera_id: str | None = None
    equipment_family: str | None = None
    measured: bool = False
    working_area: list[ZoneVertex] = Field(default_factory=list)
    dig_zone: list[ZoneVertex] = Field(default_factory=list)
    loading_zone: list[ZoneVertex] = Field(default_factory=list)
    entry_zone: list[ZoneVertex] = Field(default_factory=list)
    exit_zone: list[ZoneVertex] = Field(default_factory=list)

    @field_validator("site_id", "camera_id", "equipment_family")
    @classmethod
    def _blank_is_unknown(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value == "":
            raise ValueError("use null for an unknown id, not an empty string")
        return value

    @model_validator(mode="after")
    def zones_are_empty_or_polygons(self) -> Self:
        for name in ZONE_NAMES:
            _require_zone(name, list(getattr(self, name)))
        return self

    def is_measured_camera(self) -> bool:
        return self.measured

    def zone(self, name: str) -> list[ZoneVertex]:
        if name not in ZONE_NAMES:
            raise LabError("unknown_zone", f"Неизвестная зона: {name}")
        return list(getattr(self, name))


class FrameIdentity(VersionedModel):
    """One analyzed source frame.

    ``source_time_s`` is seconds from the start of the source video.
    ``time_base`` is the ffprobe rational ``numerator/denominator``.
    ``source_pts`` is ticks in that time base, not a second count.
    """

    video_id: str = Field(min_length=1)
    source_frame_index: int = Field(ge=0)
    source_pts: int
    time_base: str
    source_time_s: float = Field(ge=0)
    analysis_index: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @field_validator("time_base")
    @classmethod
    def _check_time_base(cls, value: str) -> str:
        return _validate_time_base(value)


class JobRecord(VersionedModel):
    job_id: str = Field(min_length=1)
    status: JobStatus
    pid: int | None = Field(default=None, gt=0)
    partial: bool = False
    analyzed_end_s: float | None = Field(default=None, ge=0)
    error_code: str | None = Field(default=None, min_length=1)
    orphaned: bool = False

    @model_validator(mode="after")
    def partial_and_orphan_flags(self) -> Self:
        if self.partial and self.analyzed_end_s is None:
            raise ValueError("partial job requires analyzed_end_s")
        if self.partial and self.status is JobStatus.completed:
            raise ValueError("a partial job is not completed")
        if self.orphaned and self.status is not JobStatus.failed:
            raise ValueError("orphaned is only set on a failed job")
        return self


class Episode(VersionedModel):
    episode_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    split_group_id: str = Field(min_length=1)
    start_s: float = Field(ge=0)
    end_s: float
    label: str = Field(min_length=1)

    @model_validator(mode="after")
    def half_open_interval(self) -> Self:
        if self.end_s <= self.start_s:
            raise ValueError("episode interval must be half-open with end_s > start_s")
        return self


class EpisodeSplit(VersionedModel):
    """One ``split_group_id`` belongs to one split. Assignment only."""

    split_group_id: str = Field(min_length=1)
    split: SplitName


class AnnotationBox(StrictModel):
    """Stored label box. ``class_name`` is a catalog token, not a detection enum."""

    class_name: str
    class_id: int = Field(ge=0)
    visibility: Visibility
    truncated: bool = False
    instance_local_id: str | None = Field(default=None, min_length=1)
    x1: float
    y1: float
    x2: float
    y2: float
    note: str | None = None

    @field_validator("class_name", mode="before")
    @classmethod
    def _catalog_token(cls, value: object) -> str:
        token = value.value if isinstance(value, DetectionClassName) else value
        if isinstance(token, str) and _CLASS_TOKEN_RE.fullmatch(token) is not None:
            return token
        raise ValueError("class name must be an ASCII token ^[a-z][a-z0-9_]{0,63}$")

    @model_validator(mode="after")
    def positive_box(self) -> Self:
        _require_positive_box(self.x1, self.y1, self.x2, self.y2)
        return self


class AnnotatedFrame(VersionedModel):
    video_id: str = Field(min_length=1)
    source_frame_index: int = Field(ge=0)
    review_status: ReviewStatus
    visibility: Visibility | None = None
    annotator: str | None = Field(default=None, min_length=1)
    reviewed_at: datetime | None = None
    note: str | None = None
    boxes: list[AnnotationBox] = Field(default_factory=list)


class FrameStatus(VersionedModel):
    """Flat ``frame_status`` row: spec §17.1 plus the §6.2 identity columns."""

    run_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    view_id: str | None = None
    source_frame_index: int = Field(ge=0)
    source_time_s: float = Field(ge=0)
    source_pts: int
    time_base: str
    analysis_index: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    decode_status: DecodeStatus
    analyzed: bool
    model_version: str | None = Field(default=None, min_length=1)
    inference_ms: float | None = Field(default=None, ge=0)
    num_detections: int = Field(ge=0)
    bucket_observation_state: ObservationState | None = None
    diagnostic_flags: list[str] = Field(default_factory=list)

    @field_validator("time_base")
    @classmethod
    def _check_time_base(cls, value: str) -> str:
        return _validate_time_base(value)


class Detection(VersionedModel):
    """One detector box for one source frame.

    Predicted boxes do not belong here. Null track ids and scores stay null.
    """

    run_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    view_id: str | None = None
    source_frame_index: int = Field(ge=0)
    source_time_s: float = Field(ge=0)
    class_id: int
    class_name: DetectionClassName
    score: float = Field(ge=0, le=1)
    x1: float
    y1: float
    x2: float
    y2: float
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    track_id: str | None = None
    parent_track_id: str | None = None
    observation_state: ObservationState
    association_score: float | None = None
    model_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def box_matches_class(self) -> Self:
        _require_class_pair(self.class_name, self.class_id)
        _require_positive_box(self.x1, self.y1, self.x2, self.y2)
        return self


class TrackState(VersionedModel):
    """Track row. ``is_observed`` is false for a predicted or missing step."""

    run_id: str = Field(min_length=1)
    track_id: str = Field(min_length=1)
    class_name: DetectionClassName
    source_time_s: float = Field(ge=0)
    state: ObservationState
    last_observed_time_s: float | None = Field(default=None, ge=0)
    is_observed: bool
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    parent_track_id: str | None = None
    identity_uncertain: bool = False

    @model_validator(mode="after")
    def boxes_all_or_nothing(self) -> Self:
        coords = (self.x1, self.y1, self.x2, self.y2)
        if all(value is None for value in coords):
            return self
        x1, y1, x2, y2 = _complete_box(coords)
        _require_positive_box(x1, y1, x2, y2)
        return self


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        has_project = (candidate / "pyproject.toml").is_file()
        has_configs = (candidate / "configs").is_dir()
        if has_project and has_configs:
            return candidate
    raise LabError(
        "config_missing",
        "Не найден корень репозитория с каталогом configs.",
    )


def configs_dir() -> Path:
    return repo_root() / "configs"


def labels_path() -> Path:
    return configs_dir() / "labels.yaml"


def pilot_path() -> Path:
    return configs_dir() / "pilot.yaml"


def example_view_path() -> Path:
    return configs_dir() / "views" / "example_view.yaml"


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LabError("config_missing", f"Нет файла конфигурации: {path.name}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LabError(
            "config_invalid",
            f"Не удалось прочитать YAML: {path.name}",
        ) from exc
    if not isinstance(loaded, dict) or any(not isinstance(key, str) for key in loaded):
        raise LabError(
            "config_invalid",
            f"Файл конфигурации должен быть YAML-объектом: {path.name}",
        )
    return loaded


def load_labels(path: Path | None = None) -> LabelsConfig:
    """Read the excavation seed (``configs/labels.yaml`` by default).

    This is not a fallback for a project that has no catalog.
    """
    return LabelsConfig.model_validate(read_yaml_mapping(path or labels_path()))


def load_pilot(path: Path | None = None) -> PilotConfig:
    return PilotConfig.model_validate(read_yaml_mapping(path or pilot_path()))


def load_view(path: Path) -> ViewConfig:
    return ViewConfig.model_validate(read_yaml_mapping(path))


def load_example_view() -> ViewConfig:
    """Load the empty template. This is not a measured camera."""
    view = load_view(example_view_path())
    filled = any(view.zone(name) for name in ZONE_NAMES)
    if view.is_measured_camera() or filled:
        raise LabError(
            "view_template",
            "example_view.yaml — шаблон с пустыми зонами, это не измеренная камера.",
        )
    return view
