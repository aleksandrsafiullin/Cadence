"""Stream a browser upload into one project media leaf.

The declared ``Content-Length`` and the counted file bytes stop at
``UPLOAD_CAP`` (the retired 8192 MiB browser limit). Chunks come from
``request.stream``. The whole body is not buffered first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header
from starlette.requests import Request

from excavation_lab import storage
from excavation_lab.errors import LabError, map_os_error
from excavation_lab.ui import copy as ui_copy
from excavation_lab.ui.video_page import _upload_leaf

UPLOAD_CAP = 8192 * 1024 * 1024


@dataclass
class StreamedUpload:
    """One streamed file part. ``status`` is 413 when the cap is crossed."""

    status: int = 0
    destination: Path | None = None
    size: int = 0
    error: str | None = None
    missing: bool = False
    empty: bool = False


def declared_over_cap(content_length: str | None) -> bool:
    """True when the request declares more than ``UPLOAD_CAP`` bytes."""
    if content_length is None:
        return False
    try:
        declared = int(content_length)
    except ValueError:
        return False
    return declared > UPLOAD_CAP


async def stream_upload(
    request: Request,
    workspace: Path,
    project_id: str,
    catalog: ui_copy.Catalog,
) -> StreamedUpload:
    """Write the file part into project media, or refuse before a lasting file.

    A declared size over the cap returns 413 and does not create a leaf.
    Counted file bytes over the cap unlink anything already opened and
    return 413. An empty part and a rejected name write nothing.
    """
    if declared_over_cap(request.headers.get("content-length")):
        return StreamedUpload(status=413)
    content_type, params = parse_options_header(request.headers.get("content-type"))
    boundary = params.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary:
        return StreamedUpload(missing=True)
    sink = _LeafSink(workspace, project_id, catalog)
    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": sink.begin,
            "on_header_field": sink.header_field,
            "on_header_value": sink.header_value,
            "on_header_end": sink.header_end,
            "on_headers_finished": sink.headers_finished,
            "on_part_data": sink.part_data,
            "on_part_end": sink.part_end,
        },
    )
    try:
        async for chunk in request.stream():
            if sink.too_big:
                break
            if chunk:
                parser.write(chunk)
    except Exception:
        sink.abort()
        raise
    finally:
        sink.close()
    if sink.too_big:
        sink.abort()
        return StreamedUpload(status=413)
    if sink.error is not None:
        sink.abort()
        return StreamedUpload(error=sink.error)
    if sink.destination is None:
        return StreamedUpload(missing=True)
    if sink.size <= 0:
        sink.abort()
        return StreamedUpload(empty=True)
    return StreamedUpload(destination=sink.destination, size=sink.size)


class _LeafSink:
    """Multipart callbacks that write one accepted file part."""

    def __init__(self, workspace: Path, project_id: str, catalog: ui_copy.Catalog) -> None:
        self.workspace = workspace
        self.project_id = project_id
        self.catalog = catalog
        self.destination: Path | None = None
        self.size = 0
        self.error: str | None = None
        self.too_big = False
        self._handle = None
        self._field = bytearray()
        self._value = bytearray()
        self._headers: dict[bytes, bytes] = {}
        self._capturing = False
        self._opened = False

    def begin(self) -> None:
        self._field.clear()
        self._value.clear()
        self._headers = {}
        self._capturing = False

    def header_field(self, data: bytes, start: int, end: int) -> None:
        self._field += data[start:end]

    def header_value(self, data: bytes, start: int, end: int) -> None:
        self._value += data[start:end]

    def header_end(self) -> None:
        name = bytes(self._field).lower()
        self._headers[name] = bytes(self._value)
        self._field.clear()
        self._value.clear()

    def headers_finished(self) -> None:
        if self.destination is not None or self.error is not None or self.too_big:
            return
        disposition = self._headers.get(b"content-disposition", b"")
        _kind, params = parse_options_header(disposition)
        raw_name = params.get(b"filename")
        if raw_name is None:
            return
        filename = raw_name.decode("utf-8", "replace")
        try:
            leaf = _upload_leaf(filename, self.catalog)
        except LabError as exc:
            self.error = exc.message
            return
        directory = storage.project_directory(self.workspace, self.project_id) / "media"
        candidate = directory / leaf
        try:
            stored = storage.resolve_project_path(self.workspace, self.project_id, candidate)
        except LabError as exc:
            self.error = exc.message
            return
        if stored.exists():
            self.error = self.catalog.duplicate_video_message()
            return
        self.destination = stored
        self._capturing = True

    def part_data(self, data: bytes, start: int, end: int) -> None:
        if not self._capturing or self.too_big or self.error is not None:
            return
        chunk = data[start:end]
        if not chunk:
            return
        if self.size + len(chunk) > UPLOAD_CAP:
            self.too_big = True
            self.abort()
            return
        self._ensure_open()
        assert self._handle is not None
        with map_os_error(str(self.destination)):
            self._handle.write(chunk)
        self.size += len(chunk)

    def part_end(self) -> None:
        self.close()
        self._capturing = False

    def _ensure_open(self) -> None:
        if self._opened or self.destination is None:
            return
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.destination.open("wb")
        self._opened = True

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            handle.close()

    def abort(self) -> None:
        self.close()
        if self.destination is not None:
            self.destination.unlink(missing_ok=True)
            self.destination = None
            self._opened = False


_TEXT_CAP = 64 * 1024


@dataclass
class StreamedImport:
    """One streamed ZIP plus the small text fields posted with it."""

    status: int = 0
    destination: Path | None = None
    size: int = 0
    error: str | None = None
    missing: bool = False
    empty: bool = False
    fields: dict[str, str] = field(default_factory=dict)


def import_leaf(filename: str) -> str | None:
    """One ZIP leaf, or ``None`` when the name contains ``..``.

    The check runs on the raw browser name. ``Path.name`` is not used first:
    it would drop a ``../`` prefix and hide the segment.
    """
    if (
        not isinstance(filename, str)
        or not filename
        or ".." in filename
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or any(ord(char) < 32 for char in filename)
        or filename != Path(filename).name
        or filename in {".", ".."}
    ):
        return None
    return filename


async def stream_import(
    request: Request,
    workspace: Path,
    project_id: str,
    catalog: ui_copy.Catalog,
) -> StreamedImport:
    """Write one ZIP under the project ``imports`` directory.

    Declared or counted size above ``UPLOAD_CAP`` returns 413 and leaves no
    file. The archive is not opened. Text fields stay small; file bytes are
    written as they arrive and are not kept past the cap.
    """
    if declared_over_cap(request.headers.get("content-length")):
        return StreamedImport(status=413)
    content_type, params = parse_options_header(request.headers.get("content-type"))
    boundary = params.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary:
        return StreamedImport(missing=True)
    sink = _ImportSink(workspace, project_id, catalog)
    parser = MultipartParser(
        boundary,
        {
            "on_part_begin": sink.begin,
            "on_header_field": sink.header_field,
            "on_header_value": sink.header_value,
            "on_header_end": sink.header_end,
            "on_headers_finished": sink.headers_finished,
            "on_part_data": sink.part_data,
            "on_part_end": sink.part_end,
        },
    )
    try:
        async for chunk in request.stream():
            if sink.too_big:
                break
            if chunk:
                parser.write(chunk)
    except Exception:
        sink.abort()
        raise
    finally:
        sink.close()
    if sink.too_big:
        sink.abort()
        return StreamedImport(status=413, fields=dict(sink.fields))
    if sink.error is not None:
        sink.abort()
        return StreamedImport(error=sink.error, fields=dict(sink.fields))
    if sink.destination is None:
        return StreamedImport(missing=True, fields=dict(sink.fields))
    if sink.size <= 0:
        sink.abort()
        return StreamedImport(empty=True, fields=dict(sink.fields))
    return StreamedImport(
        destination=sink.destination,
        size=sink.size,
        fields=dict(sink.fields),
    )


class _ImportSink:
    """Multipart callbacks. A rejected name never opens a file."""

    def __init__(self, workspace: Path, project_id: str, catalog: ui_copy.Catalog) -> None:
        self.workspace = workspace
        self.project_id = project_id
        self.catalog = catalog
        self.destination: Path | None = None
        self.size = 0
        self.error: str | None = None
        self.too_big = False
        self.fields: dict[str, str] = {}
        self._handle = None
        self._field = bytearray()
        self._value = bytearray()
        self._headers: dict[bytes, bytes] = {}
        self._part_name = ""
        self._text = bytearray()
        self._text_part = False
        self._capturing = False
        self._opened = False

    def begin(self) -> None:
        self._field.clear()
        self._value.clear()
        self._headers = {}
        self._part_name = ""
        self._text = bytearray()
        self._text_part = False
        self._capturing = False

    def header_field(self, data: bytes, start: int, end: int) -> None:
        self._field += data[start:end]

    def header_value(self, data: bytes, start: int, end: int) -> None:
        self._value += data[start:end]

    def header_end(self) -> None:
        name = bytes(self._field).lower()
        self._headers[name] = bytes(self._value)
        self._field.clear()
        self._value.clear()

    def headers_finished(self) -> None:
        disposition = self._headers.get(b"content-disposition", b"")
        _kind, params = parse_options_header(disposition)
        raw_name = params.get(b"name", b"")
        self._part_name = raw_name.decode("utf-8", "replace")
        raw_file = params.get(b"filename")
        if raw_file is None or raw_file == b"":
            self._text_part = raw_file is None
            return
        if self.destination is not None or self.error is not None or self.too_big:
            return
        filename = raw_file.decode("utf-8", "replace")
        leaf = import_leaf(filename)
        if leaf is None:
            self.error = self.catalog.import_bad_name
            return
        directory = storage.project_directory(self.workspace, self.project_id) / "imports"
        candidate = directory / leaf
        try:
            stored = storage.resolve_project_path(self.workspace, self.project_id, candidate)
        except LabError as exc:
            self.error = exc.message
            return
        if stored.is_symlink():
            self.error = self.catalog.import_bad_name
            return
        self.destination = stored
        self._capturing = True

    def part_data(self, data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if not chunk:
            return
        if self._text_part:
            if len(self._text) + len(chunk) > _TEXT_CAP:
                self.error = self.catalog.import_bad_name
                self._text_part = False
                self._text.clear()
                return
            self._text += chunk
            return
        if not self._capturing or self.too_big or self.error is not None:
            return
        if self.size + len(chunk) > UPLOAD_CAP:
            self.too_big = True
            self.abort()
            return
        self._ensure_open()
        assert self._handle is not None
        with map_os_error(str(self.destination)):
            self._handle.write(chunk)
        self.size += len(chunk)

    def part_end(self) -> None:
        if self._text_part and self._part_name:
            self.fields[self._part_name] = bytes(self._text).decode("utf-8", "replace")
        self._text_part = False
        self._text.clear()
        self.close()
        self._capturing = False

    def _ensure_open(self) -> None:
        if self._opened or self.destination is None:
            return
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.destination.open("wb")
        self._opened = True

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            handle.close()

    def abort(self) -> None:
        self.close()
        if self.destination is not None:
            self.destination.unlink(missing_ok=True)
            self.destination = None
            self._opened = False
