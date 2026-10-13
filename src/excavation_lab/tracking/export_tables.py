"""Parquet and CSV results for one analysis run.

This is the only writer for ``frame_status.parquet``, ``detections.parquet``,
and ``track_states.parquet``. Column names are spec §17.1–§17.3, plus the
§6.2 identity fields on the frame and detection tables. ``decode_ms`` and
``encode_ms`` stay null until those stages are timed. Null is stored as null.

Rows are flushed by source-time chunk. A partial close keeps rows through
``analyzed_end_s`` and does not invent the rest of the video. CSV uses a
decimal point and UTF-8. ``--excel`` adds a BOM. Notes keep their text in
SQLite; a formula prefix is quoted only in the CSV cell.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from excavation_lab.errors import LabError, map_os_error
from excavation_lab.schemas import SCHEMA_VERSION
from excavation_lab.tracking.states import TrackStateRow

FRAME_STATUS_NAME = "frame_status.parquet"
DETECTIONS_NAME = "detections.parquet"
TRACK_STATES_NAME = "track_states.parquet"
TABLES_SIDECAR = "tables.json"
EXPORT_SIDECAR = "export.json"
CHUNK_S = 30.0

FRAME_STATUS_CSV = "frame_status.csv"
DETECTIONS_CSV = "detections.csv"
TRACK_STATES_CSV = "track_states.csv"
NOTES_CSV = "notes.csv"

# Spec §17.1, then §6.2 identity, then timings that are not measured yet.
FRAME_STATUS_COLUMNS: tuple[str, ...] = (
    "run_id",
    "video_id",
    "view_id",
    "source_frame_index",
    "source_time_s",
    "decode_status",
    "analyzed",
    "model_version",
    "inference_ms",
    "num_detections",
    "bucket_observation_state",
    "diagnostic_flags",
    "source_pts",
    "time_base",
    "analysis_index",
    "width",
    "height",
    "decode_ms",
    "encode_ms",
)

# Spec §17.2 names, including the nullable track columns, then §6.2 identity.
DETECTIONS_COLUMNS: tuple[str, ...] = (
    "run_id",
    "observation_id",
    "video_id",
    "view_id",
    "source_frame_index",
    "source_time_s",
    "class_id",
    "class_name",
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "frame_width",
    "frame_height",
    "track_id_nullable",
    "parent_track_id_nullable",
    "observation_state",
    "association_score_nullable",
    "model_version",
    "source_pts",
    "time_base",
    "analysis_index",
    "width",
    "height",
)

# Spec §17.3. Nullable coordinates keep the ``_nullable`` suffix.
TRACK_STATE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "track_id",
    "class_name",
    "source_time_s",
    "state",
    "last_observed_time_s",
    "is_observed",
    "x1_nullable",
    "y1_nullable",
    "x2_nullable",
    "y2_nullable",
    "parent_track_id_nullable",
    "identity_uncertain",
)

NOTE_COLUMNS: tuple[str, ...] = (
    "review_id",
    "video_id",
    "source_frame_index",
    "review_status",
    "note",
)

_STRING = pa.string()
_INT = pa.int64()
_FLOAT = pa.float64()
_PREDICTED = "predicted"
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_KIND_ORDER = ("frame", "detection", "track")

FRAME_STATUS_SCHEMA = pa.schema(
    [
        ("run_id", _STRING),
        ("video_id", _STRING),
        ("view_id", _STRING),
        ("source_frame_index", _INT),
        ("source_time_s", _FLOAT),
        ("decode_status", _STRING),
        ("analyzed", pa.bool_()),
        ("model_version", _STRING),
        ("inference_ms", _FLOAT),
        ("num_detections", _INT),
        ("bucket_observation_state", _STRING),
        ("diagnostic_flags", pa.list_(_STRING)),
        ("source_pts", _INT),
        ("time_base", _STRING),
        ("analysis_index", _INT),
        ("width", _INT),
        ("height", _INT),
        ("decode_ms", _FLOAT),
        ("encode_ms", _FLOAT),
    ]
)

DETECTIONS_SCHEMA = pa.schema(
    [
        ("run_id", _STRING),
        ("observation_id", _STRING),
        ("video_id", _STRING),
        ("view_id", _STRING),
        ("source_frame_index", _INT),
        ("source_time_s", _FLOAT),
        ("class_id", _INT),
        ("class_name", _STRING),
        ("score", _FLOAT),
        ("x1", _FLOAT),
        ("y1", _FLOAT),
        ("x2", _FLOAT),
        ("y2", _FLOAT),
        ("frame_width", _INT),
        ("frame_height", _INT),
        ("track_id_nullable", _STRING),
        ("parent_track_id_nullable", _STRING),
        ("observation_state", _STRING),
        ("association_score_nullable", _FLOAT),
        ("model_version", _STRING),
        ("source_pts", _INT),
        ("time_base", _STRING),
        ("analysis_index", _INT),
        ("width", _INT),
        ("height", _INT),
    ]
)

TRACK_STATE_SCHEMA = pa.schema(
    [
        pa.field("run_id", _STRING, nullable=False),
        pa.field("track_id", _STRING, nullable=False),
        pa.field("class_name", _STRING, nullable=False),
        pa.field("source_time_s", _FLOAT, nullable=False),
        pa.field("state", _STRING, nullable=False),
        pa.field("last_observed_time_s", _FLOAT, nullable=True),
        pa.field("is_observed", pa.bool_(), nullable=False),
        pa.field("x1_nullable", _FLOAT, nullable=True),
        pa.field("y1_nullable", _FLOAT, nullable=True),
        pa.field("x2_nullable", _FLOAT, nullable=True),
        pa.field("y2_nullable", _FLOAT, nullable=True),
        pa.field("parent_track_id_nullable", _STRING, nullable=True),
        pa.field("identity_uncertain", pa.bool_(), nullable=False),
    ]
)

_NAMES = {
    "frame": FRAME_STATUS_NAME,
    "detection": DETECTIONS_NAME,
    "track": TRACK_STATES_NAME,
}
_SCHEMAS = {
    "frame": FRAME_STATUS_SCHEMA,
    "detection": DETECTIONS_SCHEMA,
    "track": TRACK_STATE_SCHEMA,
}
_CSV_NAMES = {
    "frame": FRAME_STATUS_CSV,
    "detection": DETECTIONS_CSV,
    "track": TRACK_STATES_CSV,
}
_CSV_COLUMNS = {
    "frame": FRAME_STATUS_COLUMNS,
    "detection": DETECTIONS_COLUMNS,
    "track": TRACK_STATE_COLUMNS,
}


def track_state_record(row: TrackStateRow) -> dict[str, Any]:
    """One §17.3 row. A null box stays null. A missing parent stays null."""
    return {
        "run_id": row.run_id,
        "track_id": row.track_id,
        "class_name": row.class_name,
        "source_time_s": row.source_time_s,
        "state": row.state.value,
        "last_observed_time_s": row.last_observed_time_s,
        "is_observed": row.is_observed,
        "x1_nullable": row.x1,
        "y1_nullable": row.y1,
        "x2_nullable": row.x2,
        "y2_nullable": row.y2,
        "parent_track_id_nullable": row.parent_track_id,
        "identity_uncertain": row.identity_uncertain,
    }


def write_analysis_tables(
    directory: Path,
    frame_rows: Sequence[Mapping[str, Any]] | None = None,
    detection_rows: Sequence[Mapping[str, Any]] | None = None,
    track_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    chunk_s: float = CHUNK_S,
    partial: bool = False,
    analyzed_end_s: float | None = None,
) -> AnalysisTableWriter | tuple[Path, Path, Path]:
    """Open a chunked writer, or write the three files when rows are passed.

    Passing no row sequences returns the writer. Analyze adds one frame at
    a time and closes it. An empty detection list still writes a file.
    """
    writer = AnalysisTableWriter(directory, chunk_s=chunk_s)
    if frame_rows is None and detection_rows is None and track_rows is None:
        return writer
    for row in () if frame_rows is None else frame_rows:
        writer.add_frame(row)
    for row in () if detection_rows is None else detection_rows:
        writer.add_detection(row)
    for row in () if track_rows is None else track_rows:
        writer.add_track_state(row)
    return writer.close(partial=partial, analyzed_end_s=analyzed_end_s)


class AnalysisTableWriter:
    """Chunked writer. Retains the open chunk, not the whole video."""

    def __init__(self, directory: Path, *, chunk_s: float = CHUNK_S) -> None:
        self.directory = Path(directory)
        self.chunk_s = _require_chunk(chunk_s)
        self.retained_rows = 0
        self.peak_retained_rows = 0
        self.rows_written = 0
        self.chunk_count = 0
        self.last_source_time_s: float | None = None
        self._buffers: dict[str, list[dict[str, Any]]] = {kind: [] for kind in _KIND_ORDER}
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._tmp: dict[str, Path] = {}
        self._chunk_start: float | None = None
        self._partial = False
        self._analyzed_end_s: float | None = None
        self._closed = False

    def add_frame(self, row: Mapping[str, Any]) -> None:
        self._add("frame", row)

    def add_detection(self, row: Mapping[str, Any]) -> None:
        state = row.get("observation_state")
        if state == _PREDICTED or getattr(state, "value", None) == _PREDICTED:
            raise LabError(
                "predicted_detection",
                "Предсказанная рамка не пишется в detections.parquet. "
                "Она хранится в track_states с is_observed=false.",
            )
        self._add("detection", row)

    def add_track_state(self, row: Mapping[str, Any]) -> None:
        state = row.get("state")
        predicted = state == _PREDICTED or getattr(state, "value", None) == _PREDICTED
        if predicted and row.get("is_observed") is not False:
            raise LabError(
                "predicted_track_state",
                "Предсказанная рамка в track_states должна иметь is_observed=false.",
            )
        self._add("track", row)

    def close(
        self,
        *,
        partial: bool = False,
        analyzed_end_s: float | None = None,
    ) -> tuple[Path, Path, Path]:
        """Flush the open chunk and write ``tables.json``."""
        if self._closed:
            return self.paths
        if partial:
            end = _require_end(analyzed_end_s)
            self._trim(end)
            self._partial = True
            self._analyzed_end_s = end
        self._flush(final=True)
        self._seal_all()
        self._write_sidecar()
        self._closed = True
        return self.paths

    def abort(self) -> None:
        """Drop an unfinished chunk. Does not write a final file."""
        if self._closed:
            return
        self._discard_all()
        self._buffers = {kind: [] for kind in _KIND_ORDER}
        self._note_retained()
        self._closed = True

    @property
    def paths(self) -> tuple[Path, Path, Path]:
        return tuple(self._path(kind) for kind in _KIND_ORDER)

    def _add(self, kind: str, row: Mapping[str, Any]) -> None:
        if self._closed:
            raise LabError("invalid_table", "Запись таблиц уже закрыта.")
        copied = _copy_row(row)
        stamp = _stamp(copied)
        if (
            stamp is not None
            and self._chunk_start is not None
            and stamp >= self._chunk_start + self.chunk_s
        ):
            self._flush(final=False)
            self._chunk_start = stamp
        elif stamp is not None and self._chunk_start is None:
            self._chunk_start = stamp
        self._buffers[kind].append(copied)
        if stamp is not None:
            self.last_source_time_s = stamp
        self._note_retained()

    def _trim(self, end_s: float) -> None:
        kept: dict[str, list[dict[str, Any]]] = {}
        for kind, rows in self._buffers.items():
            kept[kind] = [row for row in rows if _kept_time(row, end_s)]
        self._buffers = kept
        self._note_retained()

    def _flush(self, *, final: bool) -> None:
        try:
            for kind in _KIND_ORDER:
                self._emit(kind, final=final)
        except Exception:
            self._discard_all()
            raise
        self.chunk_count += 1
        self._chunk_start = None

    def _emit(self, kind: str, *, final: bool) -> None:
        rows = self._buffers[kind]
        path = self._path(kind)
        schema = _SCHEMAS[kind]
        if not rows:
            if final and kind not in self._writers:
                _write_parquet(path, _table((), schema))
            return
        table = _table(rows, schema)
        if kind in self._writers or not final:
            self._append(kind, table)
        else:
            _write_parquet(path, table)
        self.rows_written += len(rows)
        self._buffers[kind] = []
        self._note_retained()

    def _append(self, kind: str, table: pa.Table) -> None:
        path = self._path(kind)
        with map_os_error(str(path)):
            writer = self._writers.get(kind)
            if writer is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(
                    prefix=".table-",
                    suffix=".parquet.tmp",
                    dir=path.parent,
                )
                os.close(fd)
                tmp = Path(name)
                self._tmp[kind] = tmp
                writer = pq.ParquetWriter(tmp, table.schema)
                self._writers[kind] = writer
            writer.write_table(table)

    def _seal_all(self) -> None:
        for kind in _KIND_ORDER:
            writer = self._writers.get(kind)
            tmp = self._tmp.get(kind)
            if writer is None or tmp is None:
                continue
            path = self._path(kind)
            try:
                with map_os_error(str(path)):
                    writer.close()
                    os.replace(tmp, path)
            except Exception:
                self._discard(kind)
                raise
            self._writers.pop(kind, None)
            self._tmp.pop(kind, None)

    def _write_sidecar(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "partial": self._partial,
            "analyzed_end_s": self._analyzed_end_s if self._partial else None,
            "boxes": "pixels x1,y1,x2,y2",
            "time": "seconds from source start",
            "chunk_s": self.chunk_s,
        }
        body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _write_bytes(self.directory / TABLES_SIDECAR, body.encode("utf-8"))

    def _path(self, kind: str) -> Path:
        return self.directory / _NAMES[kind]

    def _note_retained(self) -> None:
        self.retained_rows = sum(len(rows) for rows in self._buffers.values())
        self.peak_retained_rows = max(self.peak_retained_rows, self.retained_rows)

    def _discard_all(self) -> None:
        for kind in _KIND_ORDER:
            self._discard(kind)

    def _discard(self, kind: str) -> None:
        writer = self._writers.pop(kind, None)
        tmp = self._tmp.pop(kind, None)
        if writer is not None:
            with suppress(OSError, pa.ArrowException):
                writer.close()
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def export_project_run(
    workspace: Path,
    project_id: str,
    run_id: str,
    *,
    excel: bool = False,
) -> dict[str, Any]:
    """Write UTF-8 CSV for one stored run. Does not write event tables."""
    from excavation_lab.storage import (
        get_analysis_run,
        list_reviews,
        project_directory,
    )

    run = get_analysis_run(workspace, project_id, run_id)
    project_dir = project_directory(workspace, project_id)
    job_id = run["job_id"]
    if not isinstance(job_id, str) or not job_id:
        raise LabError("run_not_found", "Прогон не найден в этом проекте.")
    source = project_dir / "runs" / job_id
    destination = project_dir / "exports" / run_id
    return export_directory(
        source,
        destination,
        notes=list_reviews(workspace, project_id),
        excel=excel,
        run_id=run_id,
        project_id=project_id,
    )


def export_directory(
    source: Path,
    destination: Path,
    *,
    notes: Sequence[Mapping[str, Any]],
    excel: bool,
    run_id: str,
    project_id: str,
) -> dict[str, Any]:
    """Read the three parquet files and write CSV. Null stays an empty cell."""
    source = Path(source)
    destination = Path(destination)
    for kind in _KIND_ORDER:
        path = source / _NAMES[kind]
        if not path.is_file():
            raise LabError("missing_artifact", f"В прогоне нет файла {path.name}.")
    sidecar = _read_sidecar(source)
    for kind in _KIND_ORDER:
        _write_csv(
            destination / _CSV_NAMES[kind],
            _CSV_COLUMNS[kind],
            _parquet_rows(source / _NAMES[kind]),
            excel=excel,
        )
    _write_csv(destination / NOTES_CSV, NOTE_COLUMNS, notes, excel=excel)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "project_id": project_id,
        "run_id": run_id,
        "partial": sidecar["partial"],
        "analyzed_end_s": sidecar["analyzed_end_s"],
        "excel_bom": excel,
        "directory": str(destination),
        "files": [
            FRAME_STATUS_CSV,
            DETECTIONS_CSV,
            TRACK_STATES_CSV,
            NOTES_CSV,
        ],
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_bytes(destination / EXPORT_SIDECAR, body.encode("utf-8"))
    return payload


def format_csv_export_summary(result: Mapping[str, Any]) -> str:
    directory = result.get("directory", "")
    return f"CSV записан: {directory}"


def _table(rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> pa.Table:
    if not rows:
        arrays = [pa.array([], type=field.type) for field in schema]
        return pa.table(arrays, schema=schema)
    arrays: list[pa.Array] = []
    for field in schema:
        values = [_cell(row, field.name) for row in rows]
        arrays.append(pa.array(values, type=field.type))
    return pa.table(arrays, schema=schema)


def _cell(row: Mapping[str, Any], name: str) -> Any:
    if name not in row:
        raise LabError("invalid_table", f"В строке результата нет столбца {name}.")
    value = row[name]
    if name == "diagnostic_flags" and value is None:
        return []
    return value


def _write_parquet(path: Path, table: pa.Table) -> None:
    """Write ``table`` via a temp file. No-space errors become ``disk_full``."""
    tmp_path: Path | None = None
    try:
        with map_os_error(str(path)):
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=".table-",
                suffix=".parquet.tmp",
                dir=path.parent,
            )
            os.close(fd)
            tmp_path = Path(tmp_name)
            pq.write_table(table, tmp_path)
            os.replace(tmp_path, path)
            tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    excel: bool,
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(columns),
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _csv_cell(name, row.get(name)) for name in columns})
    payload = buffer.getvalue().encode("utf-8")
    if excel:
        payload = b"\xef\xbb\xbf" + payload
    _write_bytes(path, payload)


def _write_bytes(path: Path, payload: bytes) -> None:
    tmp_path: Path | None = None
    try:
        with map_os_error(str(path)):
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=".export-", suffix=".tmp", dir=path.parent)
            os.close(fd)
            tmp_path = Path(tmp_name)
            _dump_payload(tmp_path, payload)
            os.replace(tmp_path, path)
            tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _dump_payload(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _csv_cell(name: str, value: Any) -> str:
    """Empty string for null. Never a zero stand-in. Notes get a formula guard."""
    if value is None:
        return ""
    if name == "note":
        text = value if isinstance(value, str) else str(value)
        return _guard_formula(text)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".15g")
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    return str(value)


def _guard_formula(text: str) -> str:
    if text.startswith(_FORMULA_PREFIXES):
        return "'" + text
    return text


def _parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _read_sidecar(source: Path) -> dict[str, Any]:
    path = source / TABLES_SIDECAR
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "partial": False, "analyzed_end_s": None}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabError("invalid_table", "tables.json повреждён.") from exc
    if not isinstance(loaded, dict):
        raise LabError("invalid_table", "tables.json должен быть объектом.")
    partial = loaded.get("partial") is True
    end = loaded.get("analyzed_end_s")
    end_value: float | None = None
    if partial and end is not None and not isinstance(end, bool) and isinstance(end, (int, float)):
        end_value = float(end)
    version = loaded.get("schema_version", SCHEMA_VERSION)
    return {
        "schema_version": version,
        "partial": partial,
        "analyzed_end_s": end_value,
    }


def _copy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    flags = copied.get("diagnostic_flags")
    if isinstance(flags, list):
        copied["diagnostic_flags"] = list(flags)
    return copied


def _stamp(row: Mapping[str, Any]) -> float | None:
    value = row.get("source_time_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _kept_time(row: Mapping[str, Any], end_s: float) -> bool:
    stamp = _stamp(row)
    if stamp is None:
        return True
    return stamp <= end_s + 1e-9


def _require_chunk(value: float) -> float:
    number = float(value)
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0.0:
        raise ValueError("chunk_s must be > 0")
    return number


def _require_end(value: float | None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError(
            "invalid_partial",
            "Частичный прогон должен содержать analyzed_end_s.",
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise LabError(
            "invalid_partial",
            "Частичный прогон должен содержать analyzed_end_s.",
        )
    return number
