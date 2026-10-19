"""CVAT batch zip. Splits stay on the frozen manifest. Predictions are not ground truth."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import io
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import pytest
import yaml
from typer.testing import CliRunner

from excavation_lab import storage
from excavation_lab.cli import app
from excavation_lab.datasets import cvat_export as export_mod
from excavation_lab.datasets import splits as split_mod
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import add_class, list_catalogs
from excavation_lab.datasets.cvat_export import (
    MANIFEST_COLUMNS,
    PredictionBox,
    add_zip_member,
    export_cvat,
)
from excavation_lab.datasets.geometry import xyxy_to_xywh
from excavation_lab.datasets.sampler import SampleResult, sample_frames
from excavation_lab.datasets.splits import (
    freeze_split,
    frozen_manifest_path,
    write_blocks,
)
from excavation_lab.errors import LabError
from excavation_lab.media.reader import FrameReader
from excavation_lab.schemas import (
    DetectionClassName,
    SplitCounts,
    SplitName,
    labels_path,
    repo_root,
)

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
_BLOCKS = (
    RoleBlock(start_s=0, end_s=60, role="train"),
    RoleBlock(start_s=72, end_s=92, role="validation"),
    RoleBlock(start_s=104, end_s=124, role="test"),
)
_SPEC_COLUMNS = [
    "image_file",
    "video_id",
    "source_frame_index",
    "source_pts",
    "time_base",
    "source_time_s",
    "episode_id",
    "split",
    "split_group_id",
    "image_sha256",
    "width",
    "height",
]
_NETWORK = {"requests", "httpx", "urllib", "socket", "aiohttp", "streamlit"}


@dataclass
class Prepared:
    workspace: Path
    clip: Path
    project_dir: Path
    manifest_path: Path
    split_manifest: Path
    sample: SampleResult


def _calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    called: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            called.add(func.id)
        elif isinstance(func, ast.Attribute):
            called.add(func.attr)
    return called


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _forbid(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("propose_split")


@pytest.fixture(autouse=True)
def _guard_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(split_mod, "propose_split", _forbid)
    monkeypatch.setattr(split_mod, "freeze_split", _forbid)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Prepared:
    """Synthetic clip. Not accuracy evidence for pit video."""
    root = tmp_path_factory.mktemp("cvat")
    clip = root / "clip.mp4"
    write_synthetic_mp4(clip, 126, 2)
    workspace = root / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    storage.replace_extra_roots(workspace, "pilot", [clip.parent])
    storage.insert_video(workspace, "pilot", "vid", path=clip, name=clip.name)
    freeze_split(workspace, "pilot", write_blocks(workspace, "pilot", "vid", list(_BLOCKS)))
    results = [
        sample_frames(workspace, "pilot", video_id="vid", split=name, count=2)
        for name in ("train", "validation", "test")
    ]
    for result in results:
        assert len(result.rows) == 2
    project_dir = Path(storage.project_directory(workspace, "pilot"))
    manifest_path = project_dir / "samples" / "vid" / "sample_manifest.json"
    sample = _merge_samples(results, manifest_path)
    frozen = split_mod.require_frozen(workspace, "pilot", "vid")
    assert [block.role for block in frozen.blocks] == ["train", "validation", "test"]
    return Prepared(
        workspace=workspace,
        clip=clip,
        project_dir=project_dir,
        manifest_path=manifest_path,
        split_manifest=frozen_manifest_path(workspace, "pilot", "vid"),
        sample=sample,
    )


def _merge_samples(results: list[SampleResult], manifest_path: Path) -> SampleResult:
    rows = [row for result in results for row in result.rows]
    counts = SplitCounts(
        train=sum(row.split is SplitName.train for row in rows),
        validation=sum(row.split is SplitName.validation for row in rows),
        test=sum(row.split is SplitName.test for row in rows),
    )
    merged = SampleResult(
        project_id=results[0].project_id,
        video_id=results[0].video_id,
        split_content_hash=results[0].split_content_hash,
        plan=None,
        spacing_s=results[0].spacing_s,
        requested=counts,
        written=counts,
        shortfall=SplitCounts(train=0, validation=0, test=0),
        rows=rows,
        manifest_path=str(manifest_path),
        images_dir=results[0].images_dir,
    )
    manifest_path.write_text(
        json.dumps(merged.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return merged


def _zip_files(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def _uncompressed(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(info.file_size for info in archive.infolist())


def _csv_rows(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8")
    assert not text.startswith("\ufeff")
    reader = csv.DictReader(io.StringIO(text))
    assert reader.fieldnames == _SPEC_COLUMNS
    return list(reader)


def test_manifest_columns_match_spec() -> None:
    assert list(MANIFEST_COLUMNS) == _SPEC_COLUMNS


def test_exporter_reads_frozen_split_and_does_not_propose() -> None:
    source = Path(export_mod.__file__)
    called = _calls(source)
    text = source.read_text(encoding="utf-8")
    assert "require_frozen" in called
    assert "require_project_catalog" in called
    assert "propose_split" not in called
    assert "freeze_split" not in called
    assert "sample_frames" not in called
    assert "class_box_lines" not in text
    assert "ui.copy" not in text
    assert "contact_sheet" not in text
    assert _imported_roots(source).isdisjoint(_NETWORK)
    assert "shell" not in called


def test_geometry_converts_xyxy_to_xywh() -> None:
    assert xyxy_to_xywh((10, 20, 40, 55)) == (10.0, 20.0, 30.0, 35.0)
    with pytest.raises(ValueError):
        xyxy_to_xywh((0, 0, 0, 1))
    tree = ast.parse(Path(xyxy_to_xywh.__code__.co_filename).read_text(encoding="utf-8"))
    names = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert "xyxy_to_xywh" in names
    assert "xywh_to_xyxy" in names


@pytest.mark.parametrize(
    "name",
    [
        "../secret.jpg",
        "/etc/passwd",
        "images/../../passwd",
        "images/../secret.jpg",
        r"..\secret.jpg",
        "C:/Windows/note.txt",
    ],
)
def test_zip_member_rejects_parent_and_absolute(tmp_path: Path, name: str) -> None:
    path = tmp_path / "batch.zip"
    with zipfile.ZipFile(path, "w") as archive:
        with pytest.raises(LabError) as caught:
            add_zip_member(archive, name, b"nope")
        assert caught.value.code == "zip_member"
        add_zip_member(archive, "images/ok.jpg", b"ok")
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ["images/ok.jpg"]


def test_train_batch_uses_cache_and_frozen_split(prepared: Prepared) -> None:
    before = prepared.split_manifest.read_bytes()
    result = export_cvat(prepared.workspace, "pilot", "vid__train")
    assert prepared.split_manifest.read_bytes() == before
    assert not any(prepared.project_dir.joinpath("dataset_versions").iterdir())
    assert result.predictions_included is False
    assert result.message == ""
    assert result.zip_path.endswith("annotation_batch.zip")
    assert "dataset_versions" not in Path(result.zip_path).parts
    files = _zip_files(Path(result.zip_path))
    assert "predictions.coco.json" not in files
    assert "_annotations.coco.json" not in files
    assert all(PurePosixPath(name).name != "_annotations.coco.json" for name in files)
    for name in files:
        path = PurePosixPath(name)
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert not name.startswith("/")
    rows = _csv_rows(files["frames_manifest.csv"])
    assert len(rows) == 2
    cache = prepared.project_dir / "samples" / "vid" / "images"
    clocks = {
        frame.source_frame_index: frame
        for frame in FrameReader(prepared.clip, video_id="vid").iterate(0, 60)
        if frame.source_frame_index is not None
    }
    test_indexes = {
        row.source_frame_index
        for row in prepared.sample.rows
        if 104 <= row.source_time_s < 124
    }
    exported_indexes = {int(row["source_frame_index"]) for row in rows}
    assert test_indexes
    assert exported_indexes.isdisjoint(test_indexes)
    for row in rows:
        index = int(row["source_frame_index"])
        assert row["split"] == "train"
        assert row["episode_id"] == ""
        assert 0 <= float(row["source_time_s"]) < 60
        assert row["split_group_id"] == ""
        assert row["video_id"] == "vid"
        assert row["image_file"] == f"vid_{index}.jpg"
        assert row["width"] == "64"
        assert row["height"] == "48"
        member = f"images/{row['image_file']}"
        payload = files[member]
        assert payload == (cache / row["image_file"]).read_bytes()
        assert row["image_sha256"] == hashlib.sha256(payload).hexdigest()
        clock = clocks[index]
        assert int(row["source_pts"]) == clock.source_pts
        assert row["time_base"] == clock.time_base
        assert float(row["source_time_s"]) == pytest.approx(clock.source_time_s, abs=1e-5)
    template = (repo_root() / "templates" / "annotation_instructions.md").read_bytes()
    assert files["annotation_instructions.md"] == template
    assert "Ковш" in template.decode("utf-8")
    assert files["labels.yaml"] == labels_path().read_bytes()
    batch = json.loads(files["batch.json"])
    assert batch["kind"] == "annotation_batch"
    assert batch["uncompressed_bytes"] == _uncompressed(Path(result.zip_path))
    assert batch["uncompressed_bytes"] == result.uncompressed_bytes
    assert batch["predictions_included"] is False
    assert batch["split"] == "train"
    export_cvat(prepared.workspace, "pilot", "vid__train")
    with storage.connect(prepared.project_dir / "project.sqlite") as connection:
        stored = connection.execute(
            "SELECT status, path FROM annotation_batches WHERE batch_id = ?",
            ("vid__train",),
        ).fetchall()
    assert len(stored) == 1
    assert stored[0]["status"] == "exported"
    assert str(stored[0]["path"]).endswith("annotation_batch.zip")


def test_cache_miss_encodes_from_the_reader(
    prepared: Prepared,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_row = next(row for row in prepared.sample.rows if row.split is SplitName.train)
    cache_path = prepared.project_dir / "samples" / "vid" / "images" / sample_row.image_file
    original = cache_path.read_bytes()
    other = next(
        row.image_file
        for row in prepared.sample.rows
        if row.split is SplitName.train and row.image_file != sample_row.image_file
    )
    other_bytes = (cache_path.parent / other).read_bytes()
    calls = {"n": 0}
    real = cv2.imencode

    def _count(*args: object, **kwargs: object):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(cv2, "imencode", _count)
    cache_path.unlink()
    try:
        result = export_cvat(prepared.workspace, "pilot", "vid__train")
        files = _zip_files(Path(result.zip_path))
        shipped = files[f"images/{sample_row.image_file}"]
        assert shipped[:2] == b"\xff\xd8"
        rows = _csv_rows(files["frames_manifest.csv"])
        matched = next(row for row in rows if row["image_file"] == sample_row.image_file)
        assert matched["image_sha256"] == hashlib.sha256(shipped).hexdigest()
        assert files[f"images/{other}"] == other_bytes
        assert calls["n"] == 1
    finally:
        cache_path.write_bytes(original)


def _spy_iterate(
    monkeypatch: pytest.MonkeyPatch,
    yielded: list[int] | None = None,
) -> list[tuple[float, float | None]]:
    """Record each ``iterate`` interval and, in ``yielded``, how many frames it produced."""
    calls: list[tuple[float, float | None]] = []
    original = FrameReader.iterate

    def wrapped(self: FrameReader, start_s: float = 0.0, end_s: float | None = None):
        calls.append((start_s, end_s))
        if yielded is not None:
            yielded.append(0)
            slot = len(yielded) - 1
        for frame in original(self, start_s, end_s):
            if yielded is not None:
                yielded[slot] += 1
            yield frame

    monkeypatch.setattr(FrameReader, "iterate", wrapped)
    return calls


def test_rows_with_pts_and_a_cached_jpeg_export_without_decoding(
    prepared: Prepared,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert all(row.source_pts is not None for row in prepared.sample.rows)
    calls = _spy_iterate(monkeypatch)
    forbidden = {"n": 0}

    def _no_reader(*_args: object, **_kwargs: object) -> None:
        forbidden["n"] += 1
        raise AssertionError("FrameReader opened for a fully cached v2 batch")

    monkeypatch.setattr(export_mod, "FrameReader", _no_reader)
    result = export_cvat(prepared.workspace, "pilot", "vid__train")
    assert calls == []
    assert forbidden["n"] == 0
    rows = _csv_rows(_zip_files(Path(result.zip_path))["frames_manifest.csv"])
    by_index = {row.source_frame_index: row for row in prepared.sample.rows}
    assert len(rows) == 2
    for row in rows:
        source = by_index[int(row["source_frame_index"])]
        assert int(row["source_pts"]) == source.source_pts
        assert row["time_base"] == source.time_base
        assert float(row["source_time_s"]) == pytest.approx(source.source_time_s, abs=1e-9)


def test_cache_miss_seeks_one_frame_instead_of_the_block(
    prepared: Prepared,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_row = next(row for row in prepared.sample.rows if row.split is SplitName.train)
    cache_path = prepared.project_dir / "samples" / "vid" / "images" / sample_row.image_file
    original = cache_path.read_bytes()
    cache_path.unlink()
    yielded: list[int] = []
    calls = _spy_iterate(monkeypatch, yielded)
    try:
        export_cvat(prepared.workspace, "pilot", "vid__train")
    finally:
        cache_path.write_bytes(original)
    assert len(calls) == 1
    ((start_s, end_s),) = calls
    assert start_s == pytest.approx(sample_row.source_time_s)
    assert end_s is not None
    assert end_s <= 60.0
    # The block window is not decoded: the read stops at the wanted frame.
    assert yielded == [1]


def test_legacy_manifest_without_pts_still_exports_by_seeking(
    prepared: Prepared,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = prepared.manifest_path.read_bytes()
    document = json.loads(original)
    document.pop("sample_schema_version", None)
    document.pop("decode_failed", None)
    document.pop("tick_pass", None)
    document.pop("tick_pass_reason", None)
    for row in document["rows"]:
        row.pop("source_pts", None)
        row.pop("time_base", None)
    prepared.manifest_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    yielded: list[int] = []
    calls = _spy_iterate(monkeypatch, yielded)
    try:
        result = export_cvat(prepared.workspace, "pilot", "vid__train")
    finally:
        prepared.manifest_path.write_bytes(original)
    legacy_rows = sorted(
        (row for row in prepared.sample.rows if row.split is SplitName.train),
        key=lambda row: row.source_time_s,
    )
    assert len(calls) == len(legacy_rows) == 2
    # Each legacy row costs one seek and one decoded frame, never the block.
    assert yielded == [1, 1]
    for (start_s, end_s), row in zip(sorted(calls), legacy_rows, strict=True):
        assert start_s == pytest.approx(row.source_time_s)
        assert end_s is not None and end_s <= 60.0
    rows = _csv_rows(_zip_files(Path(result.zip_path))["frames_manifest.csv"])
    clocks = {row.source_frame_index: row for row in legacy_rows}
    cache = prepared.project_dir / "samples" / "vid" / "images"
    for row in rows:
        index = int(row["source_frame_index"])
        assert int(row["source_pts"]) == clocks[index].source_pts
        assert row["time_base"] == clocks[index].time_base
        assert _zip_files(Path(result.zip_path))[f"images/{row['image_file']}"] == (
            cache / row["image_file"]
        ).read_bytes()


def test_frozen_split_wins_over_sample_row(prepared: Prepared) -> None:
    original = prepared.manifest_path.read_bytes()
    document = json.loads(original)
    moved = next(row for row in document["rows"] if row["split"] == "train")
    moved_index = moved["source_frame_index"]
    moved["split"] = "validation"
    for key in ("requested", "written"):
        document[key]["train"] -= 1
        document[key]["validation"] += 1
    prepared.manifest_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        train = export_cvat(prepared.workspace, "pilot", "vid__train")
        validation = export_cvat(
            prepared.workspace,
            "pilot",
            "vid__validation",
            include_predictions=True,
        )
        train_rows = _csv_rows(_zip_files(Path(train.zip_path))["frames_manifest.csv"])
        val_rows = _csv_rows(_zip_files(Path(validation.zip_path))["frames_manifest.csv"])
        train_indexes = {int(row["source_frame_index"]) for row in train_rows}
        val_indexes = {int(row["source_frame_index"]) for row in val_rows}
        assert moved_index in train_indexes
        assert moved_index not in val_indexes
        copied = next(row for row in train_rows if int(row["source_frame_index"]) == moved_index)
        assert copied["split"] == "train"
        assert copied["split_group_id"] == ""
        assert validation.predictions_included is False
        assert "predictions.coco.json" not in _zip_files(Path(validation.zip_path))
    finally:
        prepared.manifest_path.write_bytes(original)


def test_train_predictions_are_not_ground_truth(prepared: Prepared) -> None:
    index = next(
        row.source_frame_index for row in prepared.sample.rows if row.split is SplitName.train
    )
    box = PredictionBox(
        source_frame_index=index,
        class_name=DetectionClassName.bucket,
        x1=2,
        y1=4,
        x2=12,
        y2=14,
        score=0.5,
    )
    result = export_cvat(
        prepared.workspace,
        "pilot",
        "vid__train",
        include_predictions=True,
        predictions=[box],
    )
    assert result.predictions_included is True
    assert "not_ground_truth" in result.message
    assert "без предсказаний" not in result.message
    document = json.loads(_zip_files(Path(result.zip_path))["predictions.coco.json"])
    assert "not_ground_truth" in document["info"]["description"]
    assert document["categories"][0] == {"id": 0, "name": "excavator_body"}
    assert all(item["name"] != "background" for item in document["categories"])
    assert document["annotations"][0]["category_id"] == 1
    assert document["annotations"][0]["bbox"] == list(xyxy_to_xywh((2, 4, 12, 14)))
    assert document["annotations"][0]["iscrowd"] == 0
    assert document["annotations"][0]["score"] == 0.5
    assert "_annotations.coco.json" not in _zip_files(Path(result.zip_path))


def test_prediction_for_a_missing_frame_is_refused(prepared: Prepared) -> None:
    box = PredictionBox(
        source_frame_index=10**9,
        class_name=DetectionClassName.bucket,
        x1=1,
        y1=1,
        x2=2,
        y2=2,
    )
    with pytest.raises(LabError) as caught:
        export_cvat(
            prepared.workspace,
            "pilot",
            "vid__train",
            include_predictions=True,
            predictions=[box],
        )
    assert caught.value.code == "prediction_frame"


@pytest.mark.parametrize("split", ["validation", "test"])
def test_held_out_omits_predictions_even_when_asked(prepared: Prepared, split: str) -> None:
    box = PredictionBox(
        source_frame_index=0,
        class_name=DetectionClassName.haul_truck,
        x1=1,
        y1=1,
        x2=8,
        y2=8,
        score=0.137,
    )
    before = prepared.split_manifest.read_bytes()
    result = export_cvat(
        prepared.workspace,
        "pilot",
        f"vid__{split}",
        include_predictions=True,
        predictions=[box],
    )
    assert prepared.split_manifest.read_bytes() == before
    assert result.predictions_included is False
    assert "без предсказаний" in result.message
    assert "not_ground_truth" not in result.message
    files = _zip_files(Path(result.zip_path))
    assert "predictions.coco.json" not in files
    rows = _csv_rows(files["frames_manifest.csv"])
    assert rows
    assert {row["split"] for row in rows} == {split}


def test_cli_train_says_predictions_are_not_ground_truth(prepared: Prepared) -> None:
    result = _RUNNER.invoke(
        app,
        [
            "dataset",
            "export-cvat",
            "--project",
            "pilot",
            "--batch",
            "vid__train",
            "--include-predictions",
            "--workspace",
            str(prepared.workspace),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["predictions_included"] is True
    assert "not_ground_truth" in payload["data"]["message"]
    document = json.loads(
        _zip_files(Path(payload["data"]["zip_path"]))["predictions.coco.json"]
    )
    assert "not_ground_truth" in document["info"]["description"]
    assert document["annotations"] == []


def test_cli_validation_says_predictions_are_omitted(prepared: Prepared) -> None:
    result = _RUNNER.invoke(
        app,
        [
            "dataset",
            "export-cvat",
            "--project",
            "pilot",
            "--batch",
            "vid__validation",
            "--include-predictions",
            "--workspace",
            str(prepared.workspace),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "без предсказаний" in result.output
    assert "not_ground_truth" not in result.output
    zip_path = Path(result.output.strip().splitlines()[-1])
    assert "predictions.coco.json" not in _zip_files(zip_path)


def test_manifest_leaves_a_haul_episode_empty(prepared: Prepared) -> None:
    original = prepared.manifest_path.read_bytes()
    document = json.loads(original)
    document["rows"][0]["episode_id"] = "haul-9"
    prepared.manifest_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        result = export_cvat(prepared.workspace, "pilot", "vid__train")
        payload = _zip_files(Path(result.zip_path))["frames_manifest.csv"]
        rows = _csv_rows(payload)
        assert rows
        assert {row["episode_id"] for row in rows} == {""}
        assert "haul-9" not in payload.decode("utf-8")
        assert {row["split"] for row in rows} == {"train"}
    finally:
        prepared.manifest_path.write_bytes(original)


def test_added_class_is_in_the_next_zip_without_rewriting_boxes(prepared: Prepared) -> None:
    catalog_path = prepared.workspace / "catalogs" / "excavation" / "catalog.yaml"
    original_catalog = catalog_path.read_bytes()
    phrase = "Видимая лопата: зубья и кромка."
    snap = prepared.project_dir / "dataset_versions" / "old-snap" / "train"
    snap.mkdir(parents=True)
    coco_path = snap / "_annotations.coco.json"
    coco_path.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [9, 8, 7, 6],
                        "iscrowd": 0,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    labeled = prepared.project_dir / "annotation_batches" / "labeled"
    labeled.mkdir(parents=True)
    labels_path = labeled / "imported_labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "image_file": "frame.jpg",
                        "boxes": [
                            {
                                "class_name": "bucket",
                                "x1": 1,
                                "y1": 2,
                                "x2": 5,
                                "y2": 6,
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    before_coco = coco_path.read_bytes()
    before_labels = labels_path.read_bytes()
    index = next(
        row.source_frame_index for row in prepared.sample.rows if row.split is SplitName.train
    )
    try:
        add_class(prepared.workspace, "excavation", "shovel", "Лопата", phrase)
        result = export_cvat(
            prepared.workspace,
            "pilot",
            "vid__train",
            include_predictions=True,
            predictions=[
                PredictionBox(
                    source_frame_index=index,
                    class_name=DetectionClassName.bucket,
                    x1=2,
                    y1=4,
                    x2=12,
                    y2=14,
                    score=0.5,
                )
            ],
        )
        files = _zip_files(Path(result.zip_path))
        seed = (repo_root() / "configs" / "labels.yaml").read_bytes()
        assert files["labels.yaml"] != seed
        document = yaml.safe_load(files["labels.yaml"].decode("utf-8"))
        classes = document["classes"]
        assert [item["name"] for item in classes] == [
            "excavator_body",
            "bucket",
            "haul_truck",
            "truck_bed",
            "shovel",
        ]
        shovel = classes[4]
        assert shovel["id"] == 4
        assert shovel["title"] == "Лопата"
        assert shovel["box"] == phrase
        assert classes[1]["box"].startswith("Видимая металлическая часть ковша")
        instructions = files["annotation_instructions.md"].decode("utf-8")
        assert f"| 4 | `shovel` | Лопата | {phrase} |" in instructions
        assert "`iscrowd=1` не используется" in instructions
        assert "review_status=reviewed" in instructions
        assert "visibility=partial" in instructions
        assert "полный ковш" in instructions
        assert "только четыре" not in instructions
        assert "Четыре класса" not in instructions
        assert "четыре класса" not in instructions
        assert "not_ground_truth" in result.message
        predictions = json.loads(files["predictions.coco.json"])
        assert "not_ground_truth" in predictions["info"]["description"]
        assert [item["name"] for item in predictions["categories"]] == [
            "excavator_body",
            "bucket",
            "haul_truck",
            "truck_bed",
            "shovel",
        ]
        assert predictions["annotations"][0]["category_id"] == 1
        assert all(item["category_id"] != 4 for item in predictions["annotations"])
        cache = prepared.project_dir / "samples" / "vid" / "images"
        sample_row = next(row for row in prepared.sample.rows if row.source_frame_index == index)
        assert files[f"images/{sample_row.image_file}"] == (cache / sample_row.image_file).read_bytes()
        assert coco_path.read_bytes() == before_coco
        assert labels_path.read_bytes() == before_labels
        assert b"shovel" not in before_coco
    finally:
        catalog_path.write_bytes(original_catalog)
        shutil.rmtree(prepared.project_dir / "dataset_versions" / "old-snap")
        shutil.rmtree(labeled)


def test_cli_rejects_a_bad_batch_id(tmp_path: Path) -> None:
    result = _RUNNER.invoke(
        app,
        [
            "dataset",
            "export-cvat",
            "--project",
            "pilot",
            "--batch",
            "nope",
            "--workspace",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
