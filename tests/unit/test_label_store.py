"""Upsert-by-frame label store. Synthetic JPEGs, not pit-video accuracy."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from excavation_lab import storage
from excavation_lab.datasets import label_store
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import (
    add_class,
    list_catalogs,
    require_project_catalog,
)
from excavation_lab.datasets.class_supervision import class_supervision
from excavation_lab.datasets.coco_import import (
    format_import_summary,
    import_coco,
    load_imported_frames,
)
from excavation_lab.datasets.cvat_export import MANIFEST_COLUMNS
from excavation_lab.datasets.label_store import (
    FrameWrite,
    batch_directory,
    frozen_fields_for,
    sample_rows_with_role,
    save_frame_label,
    upsert_frames,
)
from excavation_lab.datasets.labels import (
    BoxDraft,
    FrameDraft,
    LabelRecord,
    LabelState,
    validate_frame,
)
from excavation_lab.datasets.sampler import SampleResult, SampleRow
from excavation_lab.datasets.snapshot import snapshot_dataset
from excavation_lab.datasets.splits import freeze_split, require_frozen, write_blocks
from excavation_lab.datasets.validate import validate_dataset
from excavation_lab.errors import LabError, SplitNotFrozenError
from excavation_lab.schemas import ReviewStatus, SplitCounts, SplitName, Visibility

_BLOCKS = (
    RoleBlock(start_s=0, end_s=60, role="train"),
    RoleBlock(start_s=72, end_s=92, role="validation"),
    RoleBlock(start_s=104, end_s=124, role="test"),
)
_VIDEO = "vid"
_PROJECT = "pilot"


def _lab(tmp_path: Path) -> tuple[Path, object]:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project(_PROJECT, workspace, catalog_id="excavation")
    clip = workspace / "clip.mp4"
    clip.write_bytes(b"synthetic-not-a-video")
    storage.insert_video(workspace, _PROJECT, _VIDEO, path=clip, name="clip.mp4")
    freeze_split(
        workspace,
        _PROJECT,
        write_blocks(workspace, _PROJECT, _VIDEO, list(_BLOCKS)),
    )
    return workspace, require_frozen(workspace, _PROJECT, _VIDEO)


def _frame(
    index: int,
    payload: bytes,
    *,
    note: str | None = None,
    image_file: str | None = None,
    record: LabelRecord | None = None,
    attributes: dict[str, object] | None = None,
    boxes: list[dict[str, object]] | None = None,
    source_time_s: float = 10.0,
    split: SplitName = SplitName.train,
) -> FrameWrite:
    stored = record or LabelRecord(
        video_id=_VIDEO,
        source_frame_index=index,
        review_status=ReviewStatus.reviewed,
        note=note,
    )
    return FrameWrite(
        image_file=image_file or f"f_{index:04d}.jpg",
        source_frame_index=index,
        source_pts=index,
        time_base="1/1000",
        source_time_s=source_time_s,
        episode_id="",
        split=split,
        split_group_id="",
        width=32,
        height=24,
        payload=payload,
        record=stored,
        coco_subset="",
        image_attributes=dict(attributes or ({"note": note} if note else {})),
        sidecar_boxes=list(boxes or []),
    )


def _save(workspace: Path, frozen: object, writes: list[FrameWrite], *, status: str = "labelled"):
    return upsert_frames(
        workspace,
        _PROJECT,
        _VIDEO,
        SplitName.train,
        writes,
        frozen=frozen,
        batch_status=status,
    )


def _labels(workspace: Path) -> dict[str, object]:
    path = batch_directory(workspace, _PROJECT, _VIDEO, SplitName.train) / "imported_labels.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _jpeg(width: int, height: int, tone: int) -> bytes:
    image = np.full((height, width, 3), tone, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return encoded.tobytes()


def test_batch_directory_and_frozen_fields(tmp_path: Path) -> None:
    workspace, frozen = _lab(tmp_path)
    directory = batch_directory(workspace, _PROJECT, _VIDEO, "train")
    assert directory.name == "vid__train"
    assert directory.parent.name == "annotation_batches"
    fields = frozen_fields_for(frozen, 10.0)
    assert fields is not None
    assert fields[0] is SplitName.train
    assert fields[1] == ""
    assert fields[2:] == (0.0, 60.0)
    assert frozen_fields_for(frozen, 65.0) is None
    validation = frozen_fields_for(frozen, 80.0)
    assert validation is not None
    assert validation[0] is SplitName.validation
    assert validation[2:] == (72.0, 92.0)
    absolute = directory.resolve()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(directory.parent)
    try:
        relative = Path(directory.name)
        dotted = relative / ".." / directory.name
        assert label_store._lock_for(absolute) is label_store._lock_for(relative)
        assert label_store._lock_for(absolute) is label_store._lock_for(dotted)
    finally:
        monkeypatch.undo()


def test_upsert_keeps_other_frames(tmp_path: Path) -> None:
    workspace, frozen = _lab(tmp_path)
    first = b"jpeg-frame-1"
    second = b"jpeg-frame-2"
    _save(workspace, frozen, [_frame(1, first, note="one"), _frame(2, second, note="two")])
    updated = b"jpeg-frame-1-b"
    result = _save(workspace, frozen, [_frame(1, updated, note="one-b")])
    assert result.frame_count == 2
    assert result.status == "labelled"
    assert result.batch_id == "vid__train"
    document = _labels(workspace)
    frames = document["frames"]
    assert isinstance(frames, list)
    by_index = {frame["source_frame_index"]: frame for frame in frames}
    assert set(by_index) == {1, 2}
    assert by_index[2]["record"]["note"] == "two"
    assert by_index[1]["record"]["note"] == "one-b"
    assert by_index[1]["image_sha256"] == hashlib.sha256(updated).hexdigest()
    assert by_index[2]["image_sha256"] == hashlib.sha256(second).hexdigest()
    assert by_index[2]["image_file"] == "f_0002.jpg"
    assert by_index[1]["image_file"] != "f_0001.jpg"
    images = batch_directory(workspace, _PROJECT, _VIDEO, "train") / "images"
    assert (images / str(by_index[1]["image_file"])).read_bytes() == updated
    assert not (images / "f_0001.jpg").exists()
    assert (images / "f_0002.jpg").read_bytes() == second
    sidecar = json.loads(
        (images.parent / "attributes_sidecar.json").read_text(encoding="utf-8")
    )
    side = {frame["source_frame_index"]: frame for frame in sidecar["frames"]}
    assert side[2]["attributes"]["note"] == "two"
    assert side[1]["attributes"]["note"] == "one-b"
    stored = load_imported_frames(workspace, _PROJECT)
    assert {frame.source_frame_index for frame in stored} == {1, 2}
    with storage.connect(storage.project_sqlite(workspace, _PROJECT)) as connection:
        row = connection.execute(
            "SELECT status, path FROM annotation_batches WHERE batch_id = ?",
            ("vid__train",),
        ).fetchone()
    assert row["status"] == "labelled"
    assert str(row["path"]).endswith("imported_labels.json")


def test_upsert_one_frame_of_600_touches_one_jpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, frozen = _lab(tmp_path)
    writes = [
        _frame(index, b"\xff\xd8" + index.to_bytes(2, "big") + b"\xff\xd9")
        for index in range(600)
    ]
    _save(workspace, frozen, writes)
    images = batch_directory(workspace, _PROJECT, _VIDEO, SplitName.train) / "images"
    past = 1_700_000_000
    for path in images.iterdir():
        os.utime(path, (past, past))
    before = {path.name: path.stat().st_mtime_ns for path in images.iterdir()}
    untouched = (images / "f_0000.jpg").read_bytes()
    replacement = b"\xff\xd8\x00\x2a-replaced\xff\xd9"
    real_read = Path.read_bytes
    guard = {"on": False}

    def _spy_read(self: Path, *args: object, **kwargs: object) -> bytes:
        if guard["on"] and self.parent == images:
            raise AssertionError(f"read {self.name}")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _spy_read)
    guard["on"] = True
    try:
        result = _save(workspace, frozen, [_frame(42, replacement)])
    finally:
        guard["on"] = False
    assert result.frame_count == 600
    after = {path.name: path.stat().st_mtime_ns for path in images.iterdir()}
    survivors = set(before) & set(after)
    assert survivors == set(before) - {"f_0042.jpg"}
    assert all(after[name] == before[name] for name in survivors)
    added = set(after) - set(before)
    assert len(added) == 1
    new_name = next(iter(added))
    assert (images / new_name).read_bytes() == replacement
    assert not (images / "f_0042.jpg").exists()
    assert (images / "f_0000.jpg").read_bytes() == untouched
    assert [path.name for path in images.iterdir() if path.name.endswith(".tmp")] == []
    document = _labels(workspace)
    frames = document["frames"]
    assert isinstance(frames, list)
    assert len(frames) == 600
    assert {frame["source_frame_index"] for frame in frames} == set(range(600))
    by_index = {frame["source_frame_index"]: frame for frame in frames}
    assert by_index[42]["image_file"] == new_name
    assert by_index[0]["image_file"] == "f_0000.jpg"


def test_cvat_import_keeps_in_app_frame(tmp_path: Path) -> None:
    workspace, frozen = _lab(tmp_path)
    payload = _jpeg(32, 24, 90)
    reviewed_at = datetime(2026, 3, 1, tzinfo=UTC)
    record = validate_frame(
        FrameDraft(
            video_id=_VIDEO,
            source_frame_index=9,
            review_status=ReviewStatus.reviewed,
            annotator="operator",
            reviewed_at=reviewed_at,
            note="in-app",
            boxes=[BoxDraft(category_name="bucket", x1=1, y1=1, x2=5, y2=5)],
        ),
        require_project_catalog(workspace, _PROJECT),
    )
    _save(
        workspace,
        frozen,
        [
            _frame(
                9,
                payload,
                image_file="vid_9.jpg",
                record=record,
                attributes={"note": "in-app", "annotator": "operator"},
                boxes=[{"annotation_id": 1, "attributes": {"visibility": "clear"}}],
            )
        ],
    )
    zip_path = workspace / "incoming.zip"
    zip_payload = _jpeg(32, 24, 20)
    _write_zip(zip_path, _zip_members(zip_payload))
    imported = import_coco(workspace, _PROJECT, "vid__train", zip_path)
    assert imported.image_count == 1
    assert format_import_summary(imported).startswith("Кадров из архива: 1.")
    document = _labels(workspace)
    frames = document["frames"]
    assert isinstance(frames, list)
    by_index = {frame["source_frame_index"]: frame for frame in frames}
    assert set(by_index) == {1, 9}
    kept = by_index[9]["record"]
    assert isinstance(kept, dict)
    assert kept["note"] == "in-app"
    assert LabelRecord.model_validate(kept).reviewed_at == reviewed_at
    assert kept["boxes"][0]["class_name"] == "bucket"
    images = batch_directory(workspace, _PROJECT, _VIDEO, "train") / "images"
    assert (images / "vid_9.jpg").read_bytes() == payload
    assert (images / "vid_1.jpg").read_bytes() == zip_payload
    sidecar = json.loads(
        (images.parent / "attributes_sidecar.json").read_text(encoding="utf-8")
    )
    side = {frame["source_frame_index"]: frame for frame in sidecar["frames"]}
    assert side[9]["attributes"]["note"] == "in-app"
    with storage.connect(storage.project_sqlite(workspace, _PROJECT)) as connection:
        row = connection.execute(
            "SELECT status, path FROM annotation_batches WHERE batch_id = ?",
            ("vid__train",),
        ).fetchone()
    assert row["status"] == "imported"
    assert str(row["path"]).endswith("source.zip")


def test_crash_between_jpeg_and_json_leaves_previous_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, frozen = _lab(tmp_path)
    original = b"jpeg-0"
    _save(workspace, frozen, [_frame(0, original, note="kept")])
    batch = batch_directory(workspace, _PROJECT, _VIDEO, SplitName.train)
    labels_path = batch / "imported_labels.json"
    previous = labels_path.read_text(encoding="utf-8")
    real_replace = label_store.os.replace

    def _boom(src: object, dst: object, *args: object, **kwargs: object) -> None:
        if Path(str(dst)).name in {"imported_labels.json", "attributes_sidecar.json"}:
            raise RuntimeError("simulated crash")
        real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(label_store.os, "replace", _boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _save(workspace, frozen, [_frame(0, b"jpeg-0-replaced", note="lost")])
    assert labels_path.read_text(encoding="utf-8") == previous
    assert (batch / "images" / "f_0000.jpg").read_bytes() == original
    stored = load_imported_frames(workspace, _PROJECT)
    assert [frame.source_frame_index for frame in stored] == [0]
    assert stored[0].record.note == "kept"
    assert stored[0].image_file == "f_0000.jpg"
    assert stored[0].jpeg_path.read_bytes() == original
    assert stored[0].image_sha256 == hashlib.sha256(original).hexdigest()


def test_corrupt_existing_json_leaves_images_untouched(tmp_path: Path) -> None:
    workspace, frozen = _lab(tmp_path)
    payload = b"jpeg-0"
    _save(workspace, frozen, [_frame(0, payload, note="kept")])
    batch = batch_directory(workspace, _PROJECT, _VIDEO, SplitName.train)
    images = batch / "images"
    before = {path.name: path.read_bytes() for path in images.iterdir()}
    labels_path = batch / "imported_labels.json"
    previous = labels_path.read_text(encoding="utf-8")
    labels_path.write_text("{", encoding="utf-8")
    with pytest.raises(LabError) as caught:
        _save(workspace, frozen, [_frame(0, b"jpeg-0-replaced", note="lost")])
    assert caught.value.code == "invalid_import"
    assert {path.name: path.read_bytes() for path in images.iterdir()} == before
    labels_path.write_text(previous, encoding="utf-8")
    sidecar = batch / "attributes_sidecar.json"
    sidecar.write_text("{", encoding="utf-8")
    with pytest.raises(LabError) as caught:
        _save(workspace, frozen, [_frame(0, b"jpeg-0-replaced", note="lost")])
    assert caught.value.code == "invalid_import"
    assert {path.name: path.read_bytes() for path in images.iterdir()} == before
    assert labels_path.read_text(encoding="utf-8") == previous


def test_sidecar_replace_failure_restores_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, frozen = _lab(tmp_path)
    original = b"jpeg-0"
    _save(workspace, frozen, [_frame(0, original, note="kept")])
    batch = batch_directory(workspace, _PROJECT, _VIDEO, SplitName.train)
    labels_path = batch / "imported_labels.json"
    sidecar_path = batch / "attributes_sidecar.json"
    previous_labels = labels_path.read_text(encoding="utf-8")
    previous_sidecar = sidecar_path.read_text(encoding="utf-8")
    real_replace = label_store.os.replace

    def _boom(src: object, dst: object, *args: object, **kwargs: object) -> None:
        if Path(str(dst)).name == "attributes_sidecar.json":
            raise RuntimeError("sidecar crash")
        real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(label_store.os, "replace", _boom)
    with pytest.raises(RuntimeError, match="sidecar crash"):
        _save(workspace, frozen, [_frame(0, b"jpeg-0-replaced", note="lost")])
    assert labels_path.read_text(encoding="utf-8") == previous_labels
    assert sidecar_path.read_text(encoding="utf-8") == previous_sidecar
    assert not (labels_path.with_name(labels_path.name + ".bak")).exists()
    assert not (sidecar_path.with_name(sidecar_path.name + ".bak")).exists()
    stored = load_imported_frames(workspace, _PROJECT)
    assert stored[0].jpeg_path.read_bytes() == original
    assert stored[0].record.note == "kept"


def test_upsert_removes_uncited_jpeg(tmp_path: Path) -> None:
    workspace, frozen = _lab(tmp_path)
    kept = b"jpeg-kept"
    other = b"jpeg-two"
    _save(workspace, frozen, [_frame(1, kept, note="one"), _frame(2, other, note="two")])
    images = batch_directory(workspace, _PROJECT, _VIDEO, SplitName.train) / "images"
    stray = images / "f_0001.0123456789abcdef.jpg"
    stray.write_bytes(b"stray-jpeg")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"outside")
    link = images / "linked.jpg"
    link.symlink_to(outside)
    nested = images / "nested"
    nested.mkdir()
    (nested / "hidden.jpg").write_bytes(b"hidden")
    _save(workspace, frozen, [_frame(1, b"jpeg-kept-b", note="one-b")])
    assert not stray.exists()
    assert outside.read_bytes() == b"outside"
    assert link.is_symlink()
    assert (nested / "hidden.jpg").read_bytes() == b"hidden"
    document = _labels(workspace)
    frames = document["frames"]
    assert isinstance(frames, list)
    by_index = {frame["source_frame_index"]: frame for frame in frames}
    assert (images / str(by_index[2]["image_file"])).read_bytes() == other
    assert (images / str(by_index[1]["image_file"])).read_bytes() == b"jpeg-kept-b"
    cited = {str(frame["image_file"]) for frame in frames}
    regular = {path.name for path in images.iterdir() if path.is_file() and not path.is_symlink()}
    assert regular == cited


def test_commit_faults_restore_the_previous_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, frozen = _lab(tmp_path)
    original = b"jpeg-0"
    _save(workspace, frozen, [_frame(0, original, note="kept")])
    batch = batch_directory(workspace, _PROJECT, _VIDEO, SplitName.train)
    labels_path = batch / "imported_labels.json"
    sidecar_path = batch / "attributes_sidecar.json"
    previous_labels = labels_path.read_text(encoding="utf-8")
    previous_sidecar = sidecar_path.read_text(encoding="utf-8")
    real_replace = label_store.os.replace

    def _boom_at(name: str, message: str):
        def _boom(src: object, dst: object, *args: object, **kwargs: object) -> None:
            if Path(str(dst)).name == name:
                raise RuntimeError(message)
            real_replace(src, dst, *args, **kwargs)

        return _boom

    monkeypatch.setattr(label_store.os, "replace", _boom_at("attributes_sidecar.json", "sidecar replace"))
    with pytest.raises(RuntimeError, match="sidecar replace"):
        _save(workspace, frozen, [_frame(0, b"jpeg-0-replaced", note="lost")])
    assert labels_path.read_text(encoding="utf-8") == previous_labels
    assert sidecar_path.read_text(encoding="utf-8") == previous_sidecar
    assert not (sidecar_path.with_name(sidecar_path.name + ".bak")).exists()

    monkeypatch.setattr(label_store.os, "replace", _boom_at("imported_labels.json", "labels replace"))
    with pytest.raises(RuntimeError, match="labels replace"):
        _save(workspace, frozen, [_frame(0, b"jpeg-0-replaced", note="lost")])
    assert labels_path.read_text(encoding="utf-8") == previous_labels
    assert sidecar_path.read_text(encoding="utf-8") == previous_sidecar
    assert not (sidecar_path.with_name(sidecar_path.name + ".bak")).exists()
    assert not (labels_path.with_name(labels_path.name + ".bak")).exists()
    stored = load_imported_frames(workspace, _PROJECT)
    assert stored[0].record.note == "kept"
    assert stored[0].jpeg_path.read_bytes() == original

    (sidecar_path.with_name(sidecar_path.name + ".bak")).write_text(
        '{"frames":[{"image_file":"evil.jpg"}]}',
        encoding="utf-8",
    )
    (labels_path.with_name(labels_path.name + ".bak")).write_text("STALE-LABELS", encoding="utf-8")
    monkeypatch.undo()
    _save(workspace, frozen, [_frame(0, b"jpeg-0-fresh", note="fresh")])
    fresh_labels = labels_path.read_text(encoding="utf-8")
    fresh_sidecar = sidecar_path.read_text(encoding="utf-8")
    assert "fresh" in fresh_labels
    assert "STALE-LABELS" not in fresh_labels
    assert "evil.jpg" not in fresh_sidecar
    assert not (sidecar_path.with_name(sidecar_path.name + ".bak")).exists()
    assert not (labels_path.with_name(labels_path.name + ".bak")).exists()
    reloaded = load_imported_frames(workspace, _PROJECT)
    assert reloaded[0].record.note == "fresh"
    assert reloaded[0].jpeg_path.read_bytes() == b"jpeg-0-fresh"


def test_save_ignores_manifest_split(tmp_path: Path) -> None:
    workspace, frozen = _lab(tmp_path)
    payload = _jpeg(32, 24, 70)
    _plant_sample(
        workspace,
        frozen.content_hash,
        [(15, 10.0, payload, 1500)],
        split=SplitName.validation,
    )
    listed = sample_rows_with_role(workspace, _PROJECT, _VIDEO)
    assert listed[0].row.split is SplitName.validation and listed[0].role is SplitName.train
    _save_label(
        workspace,
        15,
        review_status=ReviewStatus.reviewed,
        boxes=[],
        annotator="operator",
    )
    assert (batch_directory(workspace, _PROJECT, _VIDEO, "train") / "imported_labels.json").is_file()
    assert not (batch_directory(workspace, _PROJECT, _VIDEO, "validation") / "imported_labels.json").exists()


def test_test_block_frame_cannot_be_filed_under_train(tmp_path: Path) -> None:
    workspace, frozen = _lab(tmp_path)
    batch = batch_directory(workspace, _PROJECT, _VIDEO, SplitName.train)
    mislabeled = _frame(3, b"nope", source_time_s=110.0)
    with pytest.raises(LabError) as caught:
        _save(workspace, frozen, [mislabeled])
    assert caught.value.code == "frame_split_mismatch"
    assert not (batch / "imported_labels.json").exists()
    assert not (batch / "images").exists()
    wrong_split = _frame(4, b"also-nope", split=SplitName.test)
    with pytest.raises(LabError) as other:
        _save(workspace, frozen, [wrong_split])
    assert other.value.code == "frame_split_mismatch"


def test_concurrent_upserts_keep_both_frames(tmp_path: Path) -> None:
    workspace, frozen = _lab(tmp_path)
    barrier = threading.Barrier(2)

    def _run(item: FrameWrite) -> None:
        barrier.wait(timeout=5)
        _save(workspace, frozen, [item])

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_run, _frame(1, b"jpeg-1", note="one")),
            pool.submit(_run, _frame(2, b"jpeg-2", note="two")),
        ]
        for future in futures:
            future.result(timeout=10)
    document = _labels(workspace)
    frames = document["frames"]
    assert isinstance(frames, list)
    assert {frame["source_frame_index"] for frame in frames} == {1, 2}
    stored = load_imported_frames(workspace, _PROJECT)
    assert {frame.source_frame_index: frame.record.note for frame in stored} == {
        1: "one",
        2: "two",
    }


def _zip_members(payload: bytes) -> dict[str, bytes]:
    image_file = "vid_1.jpg"
    width, height = 32, 24
    row = {
        "image_file": image_file,
        "video_id": _VIDEO,
        "source_frame_index": "1",
        "source_pts": "1",
        "time_base": "1/1000",
        "source_time_s": "10.000000",
        "episode_id": "train-ep",
        "split": "train",
        "split_group_id": "g-train",
        "image_sha256": hashlib.sha256(payload).hexdigest(),
        "width": str(width),
        "height": str(height),
        "size_bytes": str(len(payload)),
    }
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=[*MANIFEST_COLUMNS, "size_bytes"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerow(row)
    document = {
        "categories": [{"id": 99, "name": "bucket", "supercategory": ""}],
        "images": [
            {
                "id": 1,
                "file_name": f"images/{image_file}",
                "width": width,
                "height": height,
                "attributes": {"review_status": "reviewed"},
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 99,
                "bbox": [1, 1, 4, 4],
                "area": 16,
                "iscrowd": 0,
            }
        ],
    }
    return {
        "frames_manifest.csv": buffer.getvalue().encode("utf-8"),
        "annotations/instances_default.json": (
            json.dumps(document, ensure_ascii=False) + "\n"
        ).encode("utf-8"),
        f"images/{image_file}": payload,
    }


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _sample_row(
    index: int,
    time_s: float,
    payload: bytes,
    pts: int | None,
    *,
    split: SplitName = SplitName.train,
) -> SampleRow:
    return SampleRow(
        video_id=_VIDEO,
        source_frame_index=index,
        source_time_s=time_s,
        episode_id="",
        split=split,
        image_sha256=hashlib.sha256(payload).hexdigest(),
        image_file=f"{_VIDEO}_{index}.jpg",
        strategy="spaced",
        source_pts=pts,
        time_base=None if pts is None else "1/30000",
    )


def _plant_sample(
    workspace: Path,
    content_hash: str,
    specs: list[tuple[int, float, bytes, int | None]],
    *,
    schema_version: int = 2,
    split: SplitName = SplitName.train,
) -> None:
    root = storage.project_directory(workspace, _PROJECT) / "samples" / _VIDEO
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = [
        _sample_row(index, time_s, payload, pts, split=split)
        for index, time_s, payload, pts in specs
    ]
    for row, (_index, _time_s, payload, _pts) in zip(rows, specs, strict=True):
        (images / row.image_file).write_bytes(payload)
    manifest = root / "sample_manifest.json"
    count = len(rows)
    requested = SplitCounts(
        train=count if split is SplitName.train else 0,
        validation=count if split is SplitName.validation else 0,
        test=count if split is SplitName.test else 0,
    )
    result = SampleResult(
        project_id=_PROJECT,
        video_id=_VIDEO,
        split_content_hash=content_hash,
        spacing_s=2.0,
        requested=requested,
        written=requested,
        shortfall=SplitCounts(train=0, validation=0, test=0),
        rows=rows,
        manifest_path=str(manifest),
        images_dir=str(images),
        sample_schema_version=schema_version,
    )
    manifest.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _save_label(
    workspace: Path,
    index: int,
    *,
    review_status: ReviewStatus,
    boxes: list[BoxDraft],
    annotator: str | None,
    now: datetime | None = None,
    visibility: Visibility | None = None,
) -> LabelRecord:
    return save_frame_label(
        workspace,
        _PROJECT,
        _VIDEO,
        index,
        review_status=review_status,
        boxes=boxes,
        annotator=annotator,
        visibility=visibility,
        now=now,
    )


def _no_batch(workspace: Path) -> None:
    root = storage.project_directory(workspace, _PROJECT) / "annotation_batches"
    assert list(root.iterdir()) == []


def test_in_app_save_matches_snapshot_supervision(tmp_path: Path) -> None:
    workspace, frozen = _lab(tmp_path)
    assert sample_rows_with_role(workspace, _PROJECT, _VIDEO) == []
    payload = _jpeg(32, 24, 40)
    _plant_sample(
        workspace,
        frozen.content_hash,
        [
            (10, 10.0, payload, 10_000),
            (11, 11.0, payload, 11_000),
            (20, 20.0, payload, 20_000),
            (30, 30.0, payload, 30_000),
        ],
    )
    listed = {item.row.source_frame_index: item.role for item in sample_rows_with_role(workspace, _PROJECT, _VIDEO)}
    assert listed == {
        10: SplitName.train,
        11: SplitName.train,
        20: SplitName.train,
        30: SplitName.train,
    }
    catalog = require_project_catalog(workspace, _PROJECT)
    seed_at = max(item.added_at for item in catalog.classes)
    full = BoxDraft(category_name="bucket", x1=0, y1=0, x2=32, y2=24)
    inner = BoxDraft(category_name="bucket", x1=1, y1=1, x2=8, y2=6)
    reviewed = _save_label(
        workspace,
        10,
        review_status=ReviewStatus.reviewed,
        boxes=[full],
        annotator="operator",
        now=seed_at,
    )
    excluded = _save_label(
        workspace,
        11,
        review_status=ReviewStatus.excluded,
        boxes=[],
        annotator=None,
    )
    unreviewed = _save_label(
        workspace,
        20,
        review_status=ReviewStatus.unreviewed,
        boxes=[inner],
        annotator="  ",
        now=seed_at,
    )
    assert reviewed.reviewed_at == seed_at
    assert reviewed.reviewed_at is not None
    assert reviewed.reviewed_at.utcoffset() is not None
    assert reviewed.annotator == "operator"
    assert reviewed.trainable is True
    assert excluded.review_status is ReviewStatus.excluded
    assert excluded.reviewed_at is not None
    assert excluded.reviewed_at.utcoffset() is not None
    assert excluded.annotator is None
    assert excluded.trainable is False
    assert excluded.is_negative is False
    assert unreviewed.review_status is ReviewStatus.unreviewed
    assert unreviewed.reviewed_at is None
    assert unreviewed.annotator is None
    assert unreviewed.trainable is False
    assert unreviewed.is_negative is False
    assert len(unreviewed.boxes) == 1
    assert unreviewed.boxes[0].class_name == "bucket"
    updated = add_class(workspace, "excavation", "shovel", "Лопата", "Видимая лопата.")
    shovel = next(item for item in updated.classes if item.name == "shovel")
    assert shovel.added_at > seed_at
    empty = _save_label(
        workspace,
        30,
        review_status=ReviewStatus.reviewed,
        boxes=[],
        annotator="operator",
        now=shovel.added_at,
    )
    assert empty.label_state is LabelState.reviewed_empty
    assert empty.is_negative is True
    assert empty.trainable is True
    assert empty.reviewed_at == shovel.added_at
    loaded = {frame.source_frame_index: frame for frame in load_imported_frames(workspace, _PROJECT)}
    assert set(loaded) == {10, 11, 20, 30}
    live = require_project_catalog(workspace, _PROJECT)
    before = class_supervision(loaded[10].record, live)
    after = class_supervision(loaded[30].record, live)
    assert "shovel" in before.unsupervised
    assert "shovel" not in before.negatives
    assert "bucket" in before.positives
    assert "excavator_body" in before.negatives
    assert "shovel" in after.negatives
    assert "shovel" not in after.unsupervised
    assert loaded[20].record.trainable is False
    assert loaded[20].record.boxes[0].class_name == "bucket"
    document = _labels(workspace)
    frames = document["frames"]
    assert isinstance(frames, list)
    stored = {frame["source_frame_index"]: frame for frame in frames}
    assert stored[10]["source_pts"] == 10_000
    assert stored[10]["time_base"] == "1/30000"
    assert stored[10]["source_time_s"] == 10.0
    assert stored[10]["coco_subset"] == ""
    sidecar = json.loads(
        (batch_directory(workspace, _PROJECT, _VIDEO, "train") / "attributes_sidecar.json").read_text(
            encoding="utf-8"
        )
    )
    side = {frame["source_frame_index"]: frame for frame in sidecar["frames"]}
    assert side[10]["attributes"] == {}
    assert side[10]["boxes"] == []
    assert side[20]["coco_subset"] == ""
    with storage.connect(storage.project_sqlite(workspace, _PROJECT)) as connection:
        row = connection.execute(
            "SELECT status FROM annotation_batches WHERE batch_id = ?",
            ("vid__train",),
        ).fetchone()
    assert row["status"] == "labelled"
    result = snapshot_dataset(workspace, _PROJECT)
    root = Path(result.path)
    train = json.loads((root / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    names = {image["file_name"] for image in train["images"]}
    assert names == {"vid_10.jpg", "vid_30.jpg"}
    for folder in ("train", "valid", "test"):
        coco = json.loads(
            (root / folder / "_annotations.coco.json").read_text(encoding="utf-8")
        )
        files = {image["file_name"] for image in coco["images"]}
        assert "vid_20.jpg" not in files
        assert "vid_11.jpg" not in files
        assert not (root / folder / "vid_20.jpg").exists()
        assert not (root / folder / "vid_11.jpg").exists()
    supervision = json.loads(
        (root / "train" / "class_supervision.json").read_text(encoding="utf-8")
    )
    before_side = supervision["images"]["vid_10.jpg"]
    after_side = supervision["images"]["vid_30.jpg"]
    assert set(before_side["unsupervised"]) == set(before.unsupervised)
    assert set(before_side["negatives"]) == set(before.negatives)
    assert set(before_side["positives"]) == set(before.positives)
    assert set(after_side["negatives"]) == set(after.negatives)
    assert "shovel" in after_side["negatives"]
    assert "vid_20.jpg" not in supervision["images"]
    report = validate_dataset(workspace, _PROJECT)
    assert report.reviewed == 2
    assert report.unreviewed == 1
    assert report.excluded == 1
    assert report.reviewed_empty == 1
    assert report.negatives == 1


def test_sample_without_pts_refuses_legacy_row(tmp_path: Path) -> None:
    workspace, frozen = _lab(tmp_path)
    payload = _jpeg(32, 24, 20)
    _plant_sample(
        workspace,
        frozen.content_hash,
        [(4, 8.0, payload, None)],
        schema_version=1,
    )
    listed = sample_rows_with_role(workspace, _PROJECT, _VIDEO)
    assert len(listed) == 1
    assert listed[0].row.source_pts is None
    assert listed[0].role is SplitName.train
    with pytest.raises(LabError) as caught:
        _save_label(
            workspace,
            4,
            review_status=ReviewStatus.reviewed,
            boxes=[],
            annotator="operator",
        )
    assert caught.value.code == "sample_without_pts"
    assert "Повторите отбор" in caught.value.message
    _no_batch(workspace)


def test_role_none_row_is_listed_and_refused(tmp_path: Path) -> None:
    workspace, frozen = _lab(tmp_path)
    payload = _jpeg(32, 24, 30)
    _plant_sample(
        workspace,
        frozen.content_hash,
        [(3, 10.0, payload, 3000), (65, 65.0, payload, 6500)],
    )
    listed = {item.row.source_frame_index: item.role for item in sample_rows_with_role(workspace, _PROJECT, _VIDEO)}
    assert listed[3] is SplitName.train
    assert listed[65] is None
    with pytest.raises(LabError) as caught:
        _save_label(
            workspace,
            65,
            review_status=ReviewStatus.reviewed,
            boxes=[],
            annotator="operator",
        )
    assert caught.value.code == "frame_unassigned"
    assert "vid:65" in caught.value.message
    _no_batch(workspace)


def test_save_frame_label_refuses_missing_box_and_hash(tmp_path: Path) -> None:
    workspace, frozen = _lab(tmp_path)
    payload = _jpeg(32, 24, 50)
    _plant_sample(workspace, frozen.content_hash, [(7, 12.0, payload, 7000)])
    with pytest.raises(LabError) as missing:
        _save_label(
            workspace,
            8,
            review_status=ReviewStatus.reviewed,
            boxes=[],
            annotator="operator",
        )
    assert missing.value.code == "frame_not_sampled"
    outside = BoxDraft(category_name="bucket", x1=0, y1=0, x2=33, y2=10)
    with pytest.raises(LabError) as wide:
        _save_label(
            workspace,
            7,
            review_status=ReviewStatus.reviewed,
            boxes=[outside],
            annotator="operator",
        )
    assert wide.value.code == "invalid_box"
    assert "vid:7" in wide.value.message
    flat = BoxDraft(category_name="bucket", x1=1, y1=1, x2=1, y2=4)
    with pytest.raises(LabError) as degenerate:
        _save_label(
            workspace,
            7,
            review_status=ReviewStatus.reviewed,
            boxes=[flat],
            annotator="operator",
        )
    assert degenerate.value.code == "invalid_box"
    with pytest.raises(LabError) as naive:
        _save_label(
            workspace,
            7,
            review_status=ReviewStatus.reviewed,
            boxes=[],
            annotator="operator",
            now=datetime(2026, 1, 1),  # noqa: DTZ001
        )
    assert naive.value.code == "invalid_reviewed_at"
    image = (
        storage.project_directory(workspace, _PROJECT)
        / "samples"
        / _VIDEO
        / "images"
        / "vid_7.jpg"
    )
    image.write_bytes(payload + b"x")
    with pytest.raises(LabError) as hashed:
        _save_label(
            workspace,
            7,
            review_status=ReviewStatus.reviewed,
            boxes=[],
            annotator="operator",
        )
    assert hashed.value.code == "image_hash_mismatch"
    _no_batch(workspace)


def test_unlabelable_frame_is_excluded_with_reviewed_at(tmp_path: Path) -> None:
    workspace, frozen = _lab(tmp_path)
    payload = _jpeg(32, 24, 60)
    _plant_sample(workspace, frozen.content_hash, [(6, 14.0, payload, 6000)])
    when = datetime(2026, 4, 1, tzinfo=UTC)
    record = _save_label(
        workspace,
        6,
        review_status=ReviewStatus.unreviewed,
        boxes=[],
        annotator="  ",
        visibility=Visibility.unlabelable,
        now=when,
    )
    assert record.review_status is ReviewStatus.excluded
    assert record.label_state is LabelState.excluded
    assert record.reviewed_at == when
    assert record.annotator is None
    assert record.trainable is False
    assert record.is_negative is False


def test_save_frame_label_requires_frozen_split(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project(_PROJECT, workspace, catalog_id="excavation")
    clip = workspace / "clip.mp4"
    clip.write_bytes(b"synthetic-not-a-video")
    storage.insert_video(workspace, _PROJECT, _VIDEO, path=clip, name="clip.mp4")
    with pytest.raises(SplitNotFrozenError):
        _save_label(
            workspace,
            1,
            review_status=ReviewStatus.reviewed,
            boxes=[],
            annotator="operator",
        )
