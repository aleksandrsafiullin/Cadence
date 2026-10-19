"""Spec §19 item 14: path, disk, corrupt media, FFmpeg, checkpoint.

Disk-full is an injected ``OSError(ENOSPC)`` inside the bundle writer.
The test does not fill the disk. Corrupt media and a missing ``ffprobe``
go through the probe functions. An incompatible bundle goes through
``load_bundle``.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
from pathlib import Path
from typing import BinaryIO, Self

import pytest
import torch
from typer.testing import CliRunner

from excavation_lab import cli as cli_module
from excavation_lab.cli import app
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.errors import (
    CorruptMediaError,
    DiskFullError,
    FfmpegMissingError,
    IncompatibleCheckpointError,
    LabError,
    MediaNotFoundError,
    PathNotAllowedError,
)
from excavation_lab.media import probe
from excavation_lab.models import bundle as bundle_mod
from excavation_lab.models.bundle import (
    ModelBundleSpec,
    load_bundle,
    write_model_bundle,
)
from excavation_lab.storage import create_project

_RUNNER = CliRunner()
_HASH = "ab" * 32
_NAMES = ["excavator_body", "bucket", "haul_truck", "truck_bed"]
_NEXT_STEP = {
    "path_not_allowed": "Выберите путь",
    "media_not_found": "Положите",
    "disk_full": "Освободите место",
    "corrupt_media": "Замените",
    "ffmpeg_missing": "Установите",
    "incompatible_checkpoint": "Обучите модель заново",
}


def _cyrillic(text: str) -> bool:
    return any("\u0400" <= char <= "\u04FF" for char in text)


def _assert_message(code: str, message: str) -> None:
    assert message.strip()
    assert _cyrillic(message)
    assert _NEXT_STEP[code] in message
    assert "успешно" not in message.casefold()


def _assert_error(error: LabError, code: str) -> None:
    assert isinstance(error, LabError)
    assert error.code == code
    assert error.code.isascii()
    _assert_message(code, error.message)


def _assert_cli_json(result: object, code: str) -> None:
    exit_code = result.exit_code
    output = result.output
    stdout = result.stdout
    assert exit_code == 1
    assert "Traceback" not in output
    assert stdout.strip().startswith("{")
    payload = json.loads(stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == code
    _assert_message(code, payload["error"]["message"])


def _spec(**overrides: object) -> ModelBundleSpec:
    payload: dict[str, object] = {
        "variant": "nano",
        "resolution": 384,
        "dataset_hash": _HASH,
        "library_versions": {"rfdetr": "1.11.2"},
        "resolved_train_config": {
            "schema_version": 1,
            "profile": "smoke",
            "run_test": False,
        },
        "dataset_manifest": {
            "schema_version": 1,
            "dataset_id": "ds1",
            "project_id": "pilot",
            "dataset_hash": _HASH,
        },
        "environment_report": {"schema_version": 1},
        "validation_metrics": {"schema_version": 1},
        "weights": b"fine-tune-weights",
    }
    payload.update(overrides)
    return ModelBundleSpec(**payload)  # type: ignore[arg-type]


def _checkpoint_bytes(class_names: list[str], *, resolution: int = 384) -> bytes:
    payload = io.BytesIO()
    torch.save(
        {
            "model": {"probe": torch.zeros(1)},
            "args": {
                "class_names": list(class_names),
                "num_classes": len(class_names),
            },
            "model_name": "RFDETRNano",
            "model_config": {"resolution": resolution},
        },
        payload,
    )
    return payload.getvalue()


def _rewrite_checksums(bundle: Path) -> None:
    document = json.loads((bundle / "checksums.json").read_text(encoding="utf-8"))
    files: dict[str, str] = {}
    for path in bundle.iterdir():
        if path.name == "checksums.json" or path.is_symlink() or not path.is_file():
            continue
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    document["files"] = files
    (bundle / "checksums.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _block_constructors(monkeypatch: pytest.MonkeyPatch) -> None:
    import rfdetr

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("RF-DETR constructor")

    monkeypatch.setattr(rfdetr, "RFDETRNano", _boom)
    monkeypatch.setattr(rfdetr, "RFDETRSmall", _boom)


class _RaisingFile:
    def __init__(self, raw: BinaryIO, error: OSError) -> None:
        self._raw = raw
        self._error = error

    def write(self, _payload: bytes) -> None:
        raise self._error

    def flush(self) -> None:
        pass

    def fileno(self) -> int:
        return self._raw.fileno()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self._raw.close()


def _raise_on_write(monkeypatch: pytest.MonkeyPatch, error: OSError) -> None:
    real_fdopen = bundle_mod.os.fdopen

    def fdopen(fd: int, mode: str = "r", *args: object, **kwargs: object) -> _RaisingFile:
        raw = real_fdopen(fd, mode, *args, **kwargs)
        return _RaisingFile(raw, error)

    monkeypatch.setattr(bundle_mod.os, "fdopen", fdopen)


def _video_add(workspace: Path, path: Path):
    return _RUNNER.invoke(
        app,
        [
            "video",
            "add",
            "--project",
            "pilot",
            "--path",
            str(path),
            "--workspace",
            str(workspace),
            "--json",
        ],
    )


def _train(workspace: Path):
    return _RUNNER.invoke(
        app,
        [
            "train",
            "--project",
            "pilot",
            "--dataset",
            "ds1",
            "--profile",
            "smoke",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )


def test_path_outside_roots_and_missing_file_are_distinct(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    clip = outside / "clip.mp4"
    clip.write_bytes(b"present")
    missing = allowed / "absent.mp4"

    with pytest.raises(PathNotAllowedError) as denied:
        probe.probe_media(clip, roots=[allowed])
    _assert_error(denied.value, "path_not_allowed")
    assert str(clip.resolve()) in denied.value.message

    with pytest.raises(MediaNotFoundError) as absent:
        probe.probe_media(missing, roots=[allowed])
    _assert_error(absent.value, "media_not_found")
    assert str(missing.resolve()) in absent.value.message
    assert denied.value.code == "path_not_allowed"
    assert absent.value.code == "media_not_found"


def test_enospc_inside_bundle_writer_is_disk_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _raise_on_write(monkeypatch, OSError(errno.ENOSPC, "No space left on device"))
    with pytest.raises(DiskFullError) as caught:
        write_model_bundle(tmp_path / "model", _spec())
    _assert_error(caught.value, "disk_full")
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.__cause__.errno == errno.ENOSPC
    assert "weights.pth" in caught.value.message
    assert not (tmp_path / "model").exists()
    assert not (tmp_path / ".model.partial").exists()


def test_other_oserror_inside_bundle_writer_is_not_disk_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _raise_on_write(monkeypatch, OSError(errno.EACCES, "Permission denied"))
    with pytest.raises(OSError) as caught:
        write_model_bundle(tmp_path / "model", _spec())
    assert not isinstance(caught.value, DiskFullError)
    assert caught.value.errno == errno.EACCES
    assert not (tmp_path / "model").exists()


def test_random_mp4_bytes_are_corrupt_media(tmp_path: Path) -> None:
    path = tmp_path / "noise.mp4"
    path.write_bytes(os.urandom(256))
    assert path.suffix == ".mp4"
    with pytest.raises(CorruptMediaError) as caught:
        probe.probe_media(path, roots=[tmp_path])
    _assert_error(caught.value, "corrupt_media")
    assert str(path.resolve()) in caught.value.message


def test_ffprobe_which_none_is_ffmpeg_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"present")
    seen: list[str] = []

    def which(name: str, *_args: object, **_kwargs: object) -> None:
        seen.append(name)

    monkeypatch.setattr(probe.shutil, "which", which)
    with pytest.raises(FfmpegMissingError) as caught:
        probe.probe_media(path, roots=[tmp_path])
    assert seen == ["ffprobe"]
    _assert_error(caught.value, "ffmpeg_missing")
    assert "ffprobe" in caught.value.message
    assert "ffmpeg" in caught.value.message


def test_class_order_mismatch_is_incompatible_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_constructors(monkeypatch)
    bundle = write_model_bundle(
        tmp_path / "order",
        _spec(weights=_checkpoint_bytes(_NAMES)),
    )
    mapping_path = bundle / "class_mapping.json"
    document = json.loads(mapping_path.read_text(encoding="utf-8"))
    document["class_names"] = [
        "excavator_body",
        "haul_truck",
        "bucket",
        "truck_bed",
    ]
    document["by_name"]["haul_truck"]["internal_id"] = 1
    document["by_name"]["bucket"]["internal_id"] = 2
    mapping_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_checksums(bundle)
    with pytest.raises(IncompatibleCheckpointError) as caught:
        load_bundle(bundle)
    _assert_error(caught.value, "incompatible_checkpoint")
    assert "haul_truck, bucket" in caught.value.message


def test_resolution_mismatch_is_incompatible_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_constructors(monkeypatch)
    bundle = write_model_bundle(
        tmp_path / "resolution",
        _spec(weights=_checkpoint_bytes(_NAMES, resolution=512)),
    )
    with pytest.raises(IncompatibleCheckpointError) as caught:
        load_bundle(bundle)
    _assert_error(caught.value, "incompatible_checkpoint")
    assert "512" in caught.value.message
    assert "384" in caught.value.message


def test_enospc_mapping_lives_only_in_map_os_error() -> None:
    root = Path(bundle_mod.__file__).resolve().parents[1]
    hits = [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*.py"))
        if "ENOSPC" in path.read_text(encoding="utf-8")
    ]
    assert hits == ["errors.py"]
    source = (root / "errors.py").read_text(encoding="utf-8")
    assert source.count("exc.errno == errno.ENOSPC") == 1
    bundle_source = Path(bundle_mod.__file__).read_text(encoding="utf-8")
    assert "map_os_error(" in bundle_source
    assert "ENOSPC" not in bundle_source


def test_cli_json_media_failures_have_no_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    create_project("pilot", workspace, catalog_id="excavation")

    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"nope")
    denied = _video_add(workspace, outside)
    _assert_cli_json(denied, "path_not_allowed")

    missing = _video_add(workspace, workspace / "missing.mp4")
    _assert_cli_json(missing, "media_not_found")

    noise = workspace / "noise.mp4"
    noise.write_bytes(os.urandom(128))
    corrupt = _video_add(workspace, noise)
    _assert_cli_json(corrupt, "corrupt_media")

    present = workspace / "present.mp4"
    present.write_bytes(b"present")
    monkeypatch.setattr(probe.shutil, "which", lambda *_args, **_kwargs: None)
    missing_tool = _video_add(workspace, present)
    _assert_cli_json(missing_tool, "ffmpeg_missing")


def test_cli_json_disk_full_has_no_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _raise_on_write(monkeypatch, OSError(errno.ENOSPC, "No space left on device"))

    def fail(*_args: object, **_kwargs: object) -> None:
        write_model_bundle(tmp_path / "model", _spec())

    monkeypatch.setattr(cli_module, "enqueue_train", fail)
    _assert_cli_json(_train(tmp_path), "disk_full")
    assert not (tmp_path / "model").exists()


def test_cli_json_incompatible_checkpoint_has_no_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_constructors(monkeypatch)
    bundle = write_model_bundle(
        tmp_path / "resolution",
        _spec(weights=_checkpoint_bytes(_NAMES, resolution=512)),
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        load_bundle(bundle)

    monkeypatch.setattr(cli_module, "enqueue_train", fail)
    _assert_cli_json(_train(tmp_path), "incompatible_checkpoint")


def test_enospc_inside_parquet_writer_is_disk_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from excavation_lab.tracking import export_tables

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(export_tables.pq, "write_table", boom)
    with pytest.raises(DiskFullError) as caught:
        export_tables.write_analysis_tables(tmp_path / "run", [_frame_status_row()], [])
    _assert_error(caught.value, "disk_full")
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.__cause__.errno == errno.ENOSPC
    assert not (tmp_path / "run" / "frame_status.parquet").exists()
    assert not list((tmp_path / "run").glob("*.tmp"))
    source = Path(export_tables.__file__).read_text(encoding="utf-8")
    assert "map_os_error(" in source
    assert "ENOSPC" not in source


def test_other_oserror_inside_parquet_writer_is_not_disk_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from excavation_lab.tracking import export_tables

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(export_tables.pq, "write_table", boom)
    with pytest.raises(OSError) as caught:
        export_tables.write_analysis_tables(tmp_path / "run", [_frame_status_row()], [])
    assert not isinstance(caught.value, DiskFullError)
    assert caught.value.errno == errno.EACCES


def test_enospc_on_parquet_chunk_flush_is_disk_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from excavation_lab.tracking import export_tables

    writer = export_tables.AnalysisTableWriter(tmp_path / "run", chunk_s=1.0)
    first = _frame_status_row()
    second = _frame_status_row()
    second["source_time_s"] = 1.0
    second["source_frame_index"] = 1
    writer.add_frame(first)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(export_tables.pq.ParquetWriter, "write_table", boom)
    with pytest.raises(DiskFullError) as caught:
        writer.add_frame(second)
    _assert_error(caught.value, "disk_full")
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.__cause__.errno == errno.ENOSPC
    assert not (tmp_path / "run" / "frame_status.parquet").exists()
    assert not list((tmp_path / "run").glob("*.tmp"))


def test_enospc_inside_csv_export_is_disk_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from excavation_lab.tracking import export_tables

    source = tmp_path / "run"
    export_tables.write_analysis_tables(source, [_frame_status_row()], [])

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(export_tables, "_dump_payload", boom)
    destination = tmp_path / "csv"
    with pytest.raises(DiskFullError) as caught:
        export_tables.export_directory(
            source,
            destination,
            notes=[],
            excel=False,
            run_id="run",
            project_id="pilot",
        )
    _assert_error(caught.value, "disk_full")
    assert isinstance(caught.value.__cause__, OSError)
    assert not list(destination.glob("*.csv"))
    assert not list(destination.glob("*.tmp"))
    assert not (destination / "events.csv").exists()
    assert not (destination / "truck_visits.csv").exists()


def _frame_status_row() -> dict[str, object]:
    from excavation_lab.tracking.export_tables import FRAME_STATUS_COLUMNS

    row: dict[str, object] = {name: None for name in FRAME_STATUS_COLUMNS}
    row.update(
        {
            "run_id": "run",
            "video_id": "clip",
            "source_frame_index": 0,
            "source_time_s": 0.0,
            "decode_status": "ok",
            "analyzed": True,
            "model_version": "baseline_coco",
            "inference_ms": 1.25,
            "num_detections": 0,
            "diagnostic_flags": [],
            "source_pts": 0,
            "time_base": "1/30",
            "analysis_index": 0,
            "width": 8,
            "height": 8,
            "decode_ms": None,
            "encode_ms": None,
        }
    )
    return row
