"""CVAT batch zip. Splits stay on the frozen manifest. Predictions are not ground truth."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import pytest
from typer.testing import CliRunner

from excavation_lab import storage
from excavation_lab.cli import app
from excavation_lab.datasets import cvat_export as export_mod
from excavation_lab.datasets import splits as split_mod
from excavation_lab.datasets.cvat_export import (
    MANIFEST_COLUMNS,
    PredictionBox,
    add_zip_member,
    export_cvat,
)
from excavation_lab.datasets.episodes import EpisodeSpec, add_episode
from excavation_lab.datasets.geometry import xyxy_to_xywh
from excavation_lab.datasets.sampler import SampleResult, sample_frames
from excavation_lab.datasets.splits import (
    freeze_split,
    frozen_manifest_path,
    propose_split,
    write_proposal,
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
_EPISODES = (
    EpisodeSpec("train-ep", "vid", "g-train", 0, 60, "погрузка"),
    EpisodeSpec("val-ep", "vid", "g-val", 72, 92, "погрузка"),
    EpisodeSpec("test-ep", "vid", "g-test", 104, 124, "погрузка"),
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
    storage.create_project("pilot", workspace)
    storage.replace_extra_roots(workspace, "pilot", [clip.parent])
    storage.insert_video(workspace, "pilot", "vid", path=clip, name=clip.name)
    for episode in _EPISODES:
        add_episode(workspace, "pilot", episode)
    proposal = propose_split("vid", list(_EPISODES))
    proposal_path = write_proposal(workspace, "pilot", proposal)
    freeze_split(workspace, "pilot", proposal_path)
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
    by_id = {episode.episode_id: episode.split for episode in frozen.episodes}
    assert by_id == {
        "train-ep": SplitName.train,
        "val-ep": SplitName.validation,
        "test-ep": SplitName.test,
    }
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
    assert "require_frozen" in called
    assert "propose_split" not in called
    assert "freeze_split" not in called
    assert "sample_frames" not in called
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
    frozen = split_mod.require_frozen(prepared.workspace, "pilot", "vid")
    episodes = {episode.episode_id: episode for episode in frozen.episodes}
    cache = prepared.project_dir / "samples" / "vid" / "images"
    clocks = {
        frame.source_frame_index: frame
        for frame in FrameReader(prepared.clip, video_id="vid").iterate(0, 60)
        if frame.source_frame_index is not None
    }
    test_indexes = {
        row.source_frame_index
        for row in prepared.sample.rows
        if row.episode_id == "test-ep"
    }
    exported_indexes = {int(row["source_frame_index"]) for row in rows}
    assert test_indexes
    assert exported_indexes.isdisjoint(test_indexes)
    for row in rows:
        episode = episodes[row["episode_id"]]
        index = int(row["source_frame_index"])
        assert row["split"] == episode.split.value == "train"
        assert row["split_group_id"] == episode.split_group_id == "g-train"
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
    sample_row = next(row for row in prepared.sample.rows if row.episode_id == "train-ep")
    cache_path = prepared.project_dir / "samples" / "vid" / "images" / sample_row.image_file
    original = cache_path.read_bytes()
    other = next(
        row.image_file
        for row in prepared.sample.rows
        if row.episode_id == "train-ep" and row.image_file != sample_row.image_file
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


def test_frozen_split_wins_over_sample_row(prepared: Prepared) -> None:
    original = prepared.manifest_path.read_bytes()
    document = json.loads(original)
    moved = next(row for row in document["rows"] if row["episode_id"] == "train-ep")
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
        assert copied["split_group_id"] == "g-train"
        assert validation.predictions_included is False
        assert "predictions.coco.json" not in _zip_files(Path(validation.zip_path))
    finally:
        prepared.manifest_path.write_bytes(original)


def test_train_predictions_are_not_ground_truth(prepared: Prepared) -> None:
    index = next(
        row.source_frame_index for row in prepared.sample.rows if row.episode_id == "train-ep"
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
