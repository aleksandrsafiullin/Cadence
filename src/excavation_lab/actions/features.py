"""Per-frame features for one custom analysis run.

Rows outside ``keep_ranges`` are dropped before any delta. A missing box is
NaN with mask 0, never a zero at the origin. ``track_states.parquet`` is not
an observation: predicted boxes stay out of the feature columns.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, ValidationError, field_validator, model_validator

from excavation_lab.errors import LabError, map_os_error
from excavation_lab.schemas import StrictModel
from excavation_lab.tracking.analyze import RunTables
from excavation_lab.tracking.export_tables import (
    DETECTIONS_COLUMNS,
    FRAME_STATUS_COLUMNS,
)

FEATURES_NAME = "features.parquet"
FEATURE_SPEC_NAME = "feature_spec.json"
FEATURE_SCHEMA_VERSION = 1

KEY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "run_index",
    "run_start",
    "source_frame_index",
    "source_time_s",
)
FRAME_FIELDS: tuple[str, ...] = (
    "source_frame_index",
    "source_time_s",
    "decode_status",
    "analyzed",
    "width",
    "height",
)
DETECTION_FIELDS: tuple[str, ...] = (
    "source_frame_index",
    "source_time_s",
    "class_name",
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "frame_width",
    "frame_height",
    "observation_state",
)

_CLASS_SUFFIXES: tuple[str, ...] = (
    "present",
    "cx",
    "cy",
    "w",
    "h",
    "score",
    "dcx",
    "dcy",
    "dw",
    "dh",
    "mask",
)
_DELTA_SUFFIXES: tuple[str, ...] = ("dcx", "dcy", "dw", "dh")
_RELATION_COLUMNS: tuple[str, ...] = (
    "bucket_in_bed",
    "bucket_bed_dist",
    "relation_mask",
)
_TAIL_COLUMNS: tuple[str, ...] = ("dt_s", "frame_mask")
_LIVE_STATES = frozenset({"observed", "weak"})
_BUCKET = "bucket"
_BED = "truck_bed"
_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

_SPEC_MISMATCH = "Спецификации признаков не совпадают. Прогоны не склеиваются."
_FPS_MISMATCH = "Частота analysis_fps не совпадает. Прогоны не склеиваются."
_COLUMNS_MISMATCH = "Колонки features.parquet не совпадают со спецификацией."
_BAD_RANGE = "Интервал keep_ranges должен быть полуоткрытым: конец больше начала."
_BAD_TIME = "У кадра нет конечного source_time_s."
_BAD_SPEC = "Нужна FeatureSpec с порядком классов и analysis_fps."
_MISSING_FEATURES = "Нет features.parquet или feature_spec.json."
_BAD_SPEC_FILE = "Не читается feature_spec.json."


def _feature_columns(class_names: Sequence[str], *, relation: bool) -> tuple[str, ...]:
    columns: list[str] = []
    for name in class_names:
        columns.extend(f"{name}_{suffix}" for suffix in _CLASS_SUFFIXES)
    if relation:
        columns.extend(_RELATION_COLUMNS)
    columns.extend(_TAIL_COLUMNS)
    if len(columns) != len(set(columns)) or set(columns) & set(KEY_COLUMNS):
        raise ValueError("имена классов пересекаются с колонками признаков")
    return tuple(columns)


def _relation_on(class_names: Sequence[str]) -> bool:
    names = set(class_names)
    return _BUCKET in names and _BED in names


def _as_names(value: object) -> tuple[str, ...] | None:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return None
    names: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        names.append(item)
    return tuple(names)


class FeatureSpec(StrictModel):
    """Class order, generated columns, and the run's analysis fps.

    ``columns`` is produced from ``class_names``. Relation columns exist only
    when both ``bucket`` and ``truck_bed`` are in that order. JSON from
    ``model_dump(mode="json")`` is the bundle form.
    """

    class_names: tuple[str, ...]
    columns: tuple[str, ...]
    relation: bool
    analysis_fps: float = Field(gt=0, allow_inf_nan=False)
    schema_version: int = FEATURE_SCHEMA_VERSION

    @model_validator(mode="before")
    @classmethod
    def _generate(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        names = _as_names(data.get("class_names"))
        if names is None:
            return data
        relation = data.get("relation")
        if relation is None:
            relation = _relation_on(names)
            data = {**data, "relation": relation}
        if data.get("columns") is None and isinstance(relation, bool):
            try:
                columns = _feature_columns(names, relation=relation)
            except ValueError:
                return data
            data = {**data, "columns": columns}
        return data

    @field_validator("schema_version")
    @classmethod
    def _version(cls, value: int) -> int:
        if value != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {value}")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> FeatureSpec:
        if not self.class_names:
            raise ValueError("в спецификации нужен хотя бы один класс")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("повторяющееся имя класса")
        for name in self.class_names:
            if _TOKEN.fullmatch(name) is None:
                raise ValueError(f"имя класса «{name}» не токен")
        if self.relation != _relation_on(self.class_names):
            raise ValueError("relation задаётся именами bucket и truck_bed")
        if self.columns != _feature_columns(self.class_names, relation=self.relation):
            raise ValueError("columns собираются из class_names")
        return self


class FeatureReport(StrictModel):
    """Detections whose ``class_name`` is outside the spec. Not an exception.

    ``ignored_counts`` is the per-class count. ``ignored_class_names`` is that
    map's first-seen order. ``ignored_detections`` is the sum.
    """

    ignored_detections: int = Field(ge=0)
    ignored_class_names: tuple[str, ...]
    ignored_counts: dict[str, int] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Obs:
    cx: float
    cy: float
    w: float
    h: float
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True, slots=True)
class _Frame:
    frame_index: int
    time_s: float
    analyzed: bool
    decode_status: str | None
    width: float | None
    height: float | None
    range_id: int

    @property
    def ok(self) -> bool:
        return self.analyzed and self.decode_status == "ok"


def build_features(
    run_tables: RunTables,
    *,
    run_id: str,
    run_index: int = 0,
    keep_ranges: Sequence[tuple[float, float]] | None = None,
    spec: FeatureSpec,
) -> pd.DataFrame:
    """One row per kept ``frame_status`` row, deltas only inside a kept range.

    ``keep_ranges`` is half-open source time. ``None`` keeps every frame.
    The first row of the run and the first row of each kept range have
    ``run_start`` True, NaN deltas, and class masks 0. The caller passes
    the ranges; this function does not read a test block.

    Unknown detection classes are counted on ``frame.attrs["feature_report"]``.
    """
    if not isinstance(spec, FeatureSpec):
        raise LabError("feature_spec_mismatch", _BAD_SPEC)
    if not isinstance(run_id, str) or not run_id:
        raise LabError("invalid_run_id", "У прогона нет run_id.")
    ranges = _validate_ranges(keep_ranges)
    frames = _load_frames(run_tables.frame_status, ranges)
    kept_keys = {(item.frame_index, item.time_s) for item in frames}
    observations, report = _load_observations(
        run_tables.detections,
        spec,
        kept_keys,
        {item.frame_index: (item.width, item.height) for item in frames},
    )
    rows = _rows(frames, observations, spec, run_id=run_id, run_index=run_index)
    frame = _as_frame(rows, spec)
    frame.attrs["feature_report"] = report
    return frame


def concat_runs(
    frames: Sequence[pd.DataFrame],
    specs: Sequence[FeatureSpec],
) -> tuple[pd.DataFrame, FeatureSpec]:
    """Stack per-run tables. The first row of each run is a boundary.

    Differing specs or ``analysis_fps`` raise ``feature_spec_mismatch``.
    Deltas already stop at ``run_start``, so a later window cannot cross runs.
    ``pd.concat`` drops attrs, so the result gets a merged ``feature_report``.
    """
    if len(frames) != len(specs) or not specs:
        raise LabError("feature_spec_mismatch", _SPEC_MISMATCH)
    reference = specs[0]
    if not isinstance(reference, FeatureSpec):
        raise LabError("feature_spec_mismatch", _BAD_SPEC)
    for spec in specs[1:]:
        _require_same_spec(reference, spec)
    parts = [_mark_boundary(frame, reference) for frame in frames]
    stacked = pd.concat(parts, ignore_index=True) if parts else _as_frame([], reference)
    _require_frame_columns(stacked, reference)
    stacked.attrs["feature_report"] = _merge_reports(frames)
    return stacked, reference


def _merge_reports(frames: Sequence[pd.DataFrame]) -> FeatureReport:
    """Sum ignored detections per class. First-seen order is kept."""
    counts: dict[str, int] = {}
    for frame in frames:
        report = frame.attrs.get("feature_report")
        if not isinstance(report, FeatureReport):
            continue
        for name, count in report.ignored_counts.items():
            counts[name] = counts.get(name, 0) + count
    return FeatureReport(
        ignored_detections=sum(counts.values()),
        ignored_class_names=tuple(counts),
        ignored_counts=counts,
    )


def write_features(frame: pd.DataFrame, spec: FeatureSpec, directory: Path) -> None:
    """Write ``features.parquet`` and ``feature_spec.json``. NaN stays null."""
    if not isinstance(spec, FeatureSpec):
        raise LabError("feature_spec_mismatch", _BAD_SPEC)
    _require_frame_columns(frame, spec)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(directory / FEATURES_NAME, _parquet_bytes(frame, spec))
    payload = spec.model_dump(mode="json")
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_text(directory / FEATURE_SPEC_NAME, body)


def read_features(directory: Path) -> tuple[pd.DataFrame, FeatureSpec]:
    """Load a feature directory and refuse a spec that does not match the table."""
    spec_path = directory / FEATURE_SPEC_NAME
    frame_path = directory / FEATURES_NAME
    if not spec_path.is_file() or not frame_path.is_file():
        raise LabError("features_missing", _MISSING_FEATURES)
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        spec = FeatureSpec.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise LabError("feature_spec_mismatch", _BAD_SPEC_FILE) from exc
    try:
        frame = pq.read_table(frame_path).to_pandas()
    except (OSError, pa.ArrowException) as exc:
        raise LabError("features_missing", _MISSING_FEATURES) from exc
    _require_frame_columns(frame, spec)
    return frame, spec


def _validate_ranges(
    keep_ranges: Sequence[tuple[float, float]] | None,
) -> tuple[tuple[float, float], ...] | None:
    if keep_ranges is None:
        return None
    checked: list[tuple[float, float]] = []
    for item in keep_ranges:
        if isinstance(item, str) or not isinstance(item, Sequence) or len(item) != 2:
            raise LabError("invalid_keep_ranges", _BAD_RANGE)
        start_s = _finite(item[0])
        end_s = _finite(item[1])
        if start_s is None or end_s is None or not end_s > start_s:
            raise LabError("invalid_keep_ranges", _BAD_RANGE)
        checked.append((start_s, end_s))
    return tuple(checked)


def _range_id(
    time_s: float,
    ranges: tuple[tuple[float, float], ...] | None,
) -> int | None:
    if ranges is None:
        return 0
    for index, (start, end) in enumerate(ranges):
        if start <= time_s < end:
            return index
    return None


def _load_frames(
    path: Path,
    ranges: tuple[tuple[float, float], ...] | None,
) -> list[_Frame]:
    table = _read_columns(path, FRAME_FIELDS, FRAME_STATUS_COLUMNS)
    frames: list[_Frame] = []
    for row in table.itertuples(index=False):
        time_s = _finite(row.source_time_s)
        if time_s is None:
            raise LabError("feature_time_missing", _BAD_TIME)
        range_id = _range_id(time_s, ranges)
        if range_id is None:
            continue
        frames.append(
            _Frame(
                frame_index=_frame_index(row.source_frame_index),
                time_s=time_s,
                analyzed=_flag(row.analyzed),
                decode_status=_text(row.decode_status),
                width=_positive(row.width),
                height=_positive(row.height),
                range_id=range_id,
            )
        )
    frames.sort(key=lambda item: (item.time_s, item.frame_index))
    return frames


def _load_observations(
    path: Path,
    spec: FeatureSpec,
    kept_keys: set[tuple[int, float]],
    sizes: Mapping[int, tuple[float | None, float | None]],
) -> tuple[dict[tuple[int, float], dict[str, _Obs]], FeatureReport]:
    table = _read_columns(path, DETECTION_FIELDS, DETECTIONS_COLUMNS)
    known = set(spec.class_names)
    chosen: dict[tuple[int, float, str], _Obs] = {}
    ignored_counts: dict[str, int] = {}
    for row in table.itertuples(index=False):
        key = _detection_key(row)
        if key is None or key not in kept_keys:
            continue
        name = _text(row.class_name)
        if name is None or name not in known:
            if name is not None:
                ignored_counts[name] = ignored_counts.get(name, 0) + 1
            continue
        if _text(row.observation_state) not in _LIVE_STATES:
            continue
        obs = _observation(row, sizes.get(key[0], (None, None)))
        if obs is None:
            continue
        slot = (key[0], key[1], name)
        current = chosen.get(slot)
        if current is None or obs.score > current.score:
            chosen[slot] = obs
    grouped: dict[tuple[int, float], dict[str, _Obs]] = {}
    for (frame_index, time_s, name), obs in chosen.items():
        grouped.setdefault((frame_index, time_s), {})[name] = obs
    report = FeatureReport(
        ignored_detections=sum(ignored_counts.values()),
        ignored_class_names=tuple(ignored_counts),
        ignored_counts=ignored_counts,
    )
    return grouped, report


def _rows(
    frames: Sequence[_Frame],
    observations: Mapping[tuple[int, float], Mapping[str, _Obs]],
    spec: FeatureSpec,
    *,
    run_id: str,
    run_index: int,
) -> list[dict[str, object]]:
    previous: dict[str, _Obs | None] = {}
    previous_time: float | None = None
    active: int | None = None
    rows: list[dict[str, object]] = []
    for item in frames:
        # A new kept range forgets every box from the previous range.
        if active is None or item.range_id != active:
            previous = {}
            previous_time = None
            active = item.range_id
            started = True
        else:
            started = False
        live = observations.get((item.frame_index, item.time_s), {})
        if not item.ok:
            live = {}
        row = _blank(spec)
        row["run_id"] = run_id
        row["run_index"] = run_index
        row["run_start"] = started
        row["source_frame_index"] = item.frame_index
        row["source_time_s"] = item.time_s
        row["frame_mask"] = 1 if item.ok else 0
        if item.ok and previous_time is not None:
            row["dt_s"] = item.time_s - previous_time
        for name in spec.class_names:
            _write_class(row, name, live.get(name), previous.get(name))
        if spec.relation and item.ok:
            _write_relation(row, live.get(_BUCKET), live.get(_BED))
        if item.ok:
            previous = {name: live.get(name) for name in spec.class_names}
            previous_time = item.time_s
        rows.append(row)
    return rows


def _write_class(
    row: dict[str, object],
    name: str,
    obs: _Obs | None,
    prev: _Obs | None,
) -> None:
    if obs is None:
        return
    row[f"{name}_present"] = 1
    row[f"{name}_cx"] = obs.cx
    row[f"{name}_cy"] = obs.cy
    row[f"{name}_w"] = obs.w
    row[f"{name}_h"] = obs.h
    row[f"{name}_score"] = obs.score
    if prev is None:
        return
    row[f"{name}_dcx"] = obs.cx - prev.cx
    row[f"{name}_dcy"] = obs.cy - prev.cy
    row[f"{name}_dw"] = obs.w - prev.w
    row[f"{name}_dh"] = obs.h - prev.h
    row[f"{name}_mask"] = 1


def _write_relation(row: dict[str, object], bucket: _Obs | None, bed: _Obs | None) -> None:
    if bucket is None or bed is None:
        return
    measured = _bucket_bed(bucket, bed)
    if measured is None:
        return
    contain, dist = measured
    row["bucket_in_bed"] = contain
    row["bucket_bed_dist"] = dist
    row["relation_mask"] = 1


def _bucket_bed(bucket: _Obs, bed: _Obs) -> tuple[float, float] | None:
    """Containment is the share of the bucket box inside the bed, not IoU."""
    area = (bucket.x2 - bucket.x1) * (bucket.y2 - bucket.y1)
    diagonal = math.hypot(bed.x2 - bed.x1, bed.y2 - bed.y1)
    if area <= 0.0 or diagonal <= 0.0:
        return None
    width = min(bucket.x2, bed.x2) - max(bucket.x1, bed.x1)
    height = min(bucket.y2, bed.y2) - max(bucket.y1, bed.y1)
    overlap = width * height if width > 0.0 and height > 0.0 else 0.0
    contain = overlap / area
    dist = math.hypot(
        (bucket.x1 + bucket.x2) / 2.0 - (bed.x1 + bed.x2) / 2.0,
        (bucket.y1 + bucket.y2) / 2.0 - (bed.y1 + bed.y2) / 2.0,
    )
    dist /= diagonal
    if not math.isfinite(contain) or not math.isfinite(dist):
        return None
    return contain, dist


def _blank(spec: FeatureSpec) -> dict[str, object]:
    row: dict[str, object] = {}
    for name in spec.columns:
        if _is_mask_column(name):
            row[name] = 0
        else:
            row[name] = math.nan
    return row


def _as_frame(rows: Sequence[Mapping[str, object]], spec: FeatureSpec) -> pd.DataFrame:
    columns = list(KEY_COLUMNS) + list(spec.columns)
    frame = pd.DataFrame.from_records(list(rows), columns=columns)
    _cast(frame, spec)
    return frame


def _cast(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    frame["run_id"] = frame["run_id"].astype("string")
    frame["run_index"] = frame["run_index"].astype("int64")
    frame["run_start"] = frame["run_start"].astype("bool")
    frame["source_frame_index"] = frame["source_frame_index"].astype("int64")
    frame["source_time_s"] = frame["source_time_s"].astype("float64")
    for name in spec.columns:
        if _is_mask_column(name):
            frame[name] = frame[name].astype("int64")
        else:
            frame[name] = frame[name].astype("float64")


def _mark_boundary(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    _require_frame_columns(frame, spec)
    out = frame.copy()
    if out.empty:
        return out
    index = out.index[0]
    out.at[index, "run_start"] = True
    out.at[index, "dt_s"] = math.nan
    for name in spec.class_names:
        for suffix in _DELTA_SUFFIXES:
            out.at[index, f"{name}_{suffix}"] = math.nan
        out.at[index, f"{name}_mask"] = 0
    return out


def _require_same_spec(reference: FeatureSpec, spec: FeatureSpec) -> None:
    if not isinstance(spec, FeatureSpec):
        raise LabError("feature_spec_mismatch", _BAD_SPEC)
    if not math.isclose(reference.analysis_fps, spec.analysis_fps, rel_tol=0.0, abs_tol=1e-9):
        raise LabError("feature_spec_mismatch", _FPS_MISMATCH)
    if (
        spec.class_names != reference.class_names
        or spec.columns != reference.columns
        or spec.relation != reference.relation
        or spec.schema_version != reference.schema_version
    ):
        raise LabError("feature_spec_mismatch", _SPEC_MISMATCH)


def _require_frame_columns(frame: pd.DataFrame, spec: FeatureSpec) -> None:
    expected = list(KEY_COLUMNS) + list(spec.columns)
    if list(frame.columns) != expected:
        raise LabError("feature_spec_mismatch", _COLUMNS_MISMATCH)


def _observation(
    row: Any,
    fallback: tuple[float | None, float | None],
) -> _Obs | None:
    width = _positive(row.frame_width) or fallback[0]
    height = _positive(row.frame_height) or fallback[1]
    if width is None or height is None:
        return None
    x1 = _finite(row.x1)
    y1 = _finite(row.y1)
    x2 = _finite(row.x2)
    y2 = _finite(row.y2)
    score = _finite(row.score)
    if None in (x1, y1, x2, y2, score):
        return None
    assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
    assert score is not None
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0.0 or box_h <= 0.0:
        return None
    return _Obs(
        cx=((x1 + x2) / 2.0) / width,
        cy=((y1 + y2) / 2.0) / height,
        w=box_w / width,
        h=box_h / height,
        score=score,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def _detection_key(row: Any) -> tuple[int, float] | None:
    time_s = _finite(row.source_time_s)
    if time_s is None:
        return None
    try:
        frame_index = _frame_index(row.source_frame_index)
    except LabError:
        return None
    return frame_index, time_s


def _read_columns(
    path: Path,
    fields: Sequence[str],
    defined: Sequence[str],
) -> pd.DataFrame:
    missing = [name for name in fields if name not in defined]
    if missing:
        raise LabError(
            "run_incomplete",
            "В схеме таблицы нет колонок: " + ", ".join(missing) + ".",
        )
    if not path.is_file():
        raise LabError("run_incomplete", f"В прогоне нет таблицы: {path.name}.")
    try:
        table = pq.read_table(path, columns=list(fields))
    except (OSError, pa.ArrowException, KeyError, ValueError) as exc:
        raise LabError("run_incomplete", f"Не читается таблица {path.name}.") from exc
    return table.to_pandas()


def _frame_index(value: object) -> int:
    if isinstance(value, bool) or _missing(value):
        raise LabError("feature_time_missing", "У кадра нет source_frame_index.")
    if isinstance(value, float) and not value.is_integer():
        raise LabError("feature_time_missing", "У кадра нет source_frame_index.")
    try:
        index = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise LabError("feature_time_missing", "У кадра нет source_frame_index.") from exc
    if index < 0:
        raise LabError("feature_time_missing", "У кадра нет source_frame_index.")
    return index


def _parquet_bytes(frame: pd.DataFrame, spec: FeatureSpec) -> bytes:
    schema = _arrow_schema(spec)
    records = []
    for raw in frame.to_dict(orient="records"):
        records.append({key: _arrow_value(key, value) for key, value in raw.items()})
    table = pa.Table.from_pylist(records, schema=schema)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def _arrow_schema(spec: FeatureSpec) -> pa.Schema:
    fields = [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("run_index", pa.int64(), nullable=False),
        pa.field("run_start", pa.bool_(), nullable=False),
        pa.field("source_frame_index", pa.int64(), nullable=False),
        pa.field("source_time_s", pa.float64(), nullable=False),
    ]
    for name in spec.columns:
        if _is_mask_column(name):
            fields.append(pa.field(name, pa.int64(), nullable=False))
        else:
            fields.append(pa.field(name, pa.float64(), nullable=True))
    return pa.schema(fields)


def _arrow_value(column: str, value: object) -> object:
    if column == "run_id":
        return str(value)
    if column == "run_start":
        return bool(value)
    if column in {"run_index", "source_frame_index"} or _is_mask_column(column):
        return int(value)  # type: ignore[arg-type]
    if column == "source_time_s":
        return float(value)  # type: ignore[arg-type]
    if _missing(value):
        return None
    return float(value)  # type: ignore[arg-type]


def _is_mask_column(column: str) -> bool:
    return column.endswith(("_present", "_mask"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with map_os_error(str(path)):
            tmp.write_bytes(payload)
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_text(path: Path, payload: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with map_os_error(str(path)):
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or _missing(value):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _positive(value: object) -> float | None:
    number = _finite(value)
    if number is None or number <= 0.0:
        return None
    return number


def _flag(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true"}
    if _missing(value):
        return False
    return bool(value)


def _text(value: object) -> str | None:
    if _missing(value):
        return None
    text = str(value).strip()
    return text or None


def _missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, float) and math.isnan(value)
