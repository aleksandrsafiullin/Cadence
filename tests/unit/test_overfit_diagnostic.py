"""Overfit diagnostic: reviewed train frames only, fake trainer, no weights download."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from excavation_lab import jobs, storage
from excavation_lab.cli import app
from excavation_lab.datasets.episodes import EpisodeSpec
from excavation_lab.datasets.labels import LabelRecord
from excavation_lab.datasets.splits import (
    freeze_split,
    propose_split,
    require_frozen,
    write_proposal,
)
from excavation_lab.models import train as train_mod
from excavation_lab.models.train import OVERFIT_MAX_FRAMES, OVERFIT_OUTPUT_NAME
from excavation_lab.models.train_config import (
    METRIC_ROLE_DIAGNOSTIC_OVERFIT,
    resolve_train_profile,
    write_resolved_train_config,
)
from excavation_lab.schemas import (
    AnnotationBox,
    DetectionClassName,
    JobStatus,
    ReviewStatus,
    Visibility,
    load_pilot,
)

_RUNNER = CliRunner()
_PROJECT = "pilot"
_VIDEO = "vid"
_DATASET_ID = "ds-overfit"
_MARKER = "DO_NOT_OPEN_TEST_SPLIT"


@dataclass(frozen=True)
class _Built:
    workspace: Path
    dataset: Path
    train_names: list[str]


class _DummyProcess:
    def __init__(self) -> None:
        self.pid = os.getpid()

    def poll(self) -> None:
        return None


def _dummy_spawn(*_args: object, **_kwargs: object) -> _DummyProcess:
    return _DummyProcess()


def _cyrillic(text: str) -> bool:
    return any("а" <= char.lower() <= "я" or char.lower() == "ё" for char in text)


def _dump(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _snapshot(root: Path) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        found[str(path.relative_to(root))] = path.read_bytes()
    return found


def _freeze(workspace: Path) -> None:
    storage.insert_video(workspace, _PROJECT, _VIDEO, name="pilot")
    proposal = propose_split(
        _VIDEO,
        [
            EpisodeSpec("train-ep", _VIDEO, "g-train", 0, 600, "погрузка"),
            EpisodeSpec("val-ep", _VIDEO, "g-val", 620, 820, "погрузка"),
            EpisodeSpec("test-ep", _VIDEO, "g-test", 840, 1040, "погрузка"),
        ],
    )
    manifest = write_proposal(workspace, _PROJECT, proposal)
    freeze_split(workspace, _PROJECT, manifest)
    frozen = require_frozen(workspace, _PROJECT, _VIDEO)
    assigned = {episode.episode_id: episode.split.value for episode in frozen.episodes}
    assert assigned == {
        "train-ep": "train",
        "val-ep": "validation",
        "test-ep": "test",
    }


def _record(index: int, review: str) -> LabelRecord:
    boxes: list[AnnotationBox] = []
    if review == "reviewed":
        boxes.append(
            AnnotationBox(
                class_name=DetectionClassName.bucket,
                class_id=1,
                visibility=Visibility.clear,
                x1=1,
                y1=1,
                x2=4,
                y2=4,
            )
        )
    return LabelRecord(
        video_id=_VIDEO,
        source_frame_index=index,
        review_status=ReviewStatus(review),
        boxes=boxes,
    )


def _build(
    tmp_path: Path,
    *,
    train_count: int,
    leak: str | None = None,
    extras_in_test: int = 0,
    duplicate_first: bool = False,
) -> _Built:
    workspace = tmp_path / "workspace"
    storage.create_project(_PROJECT, workspace)
    _freeze(workspace)
    dataset = workspace / "projects" / _PROJECT / "dataset_versions" / _DATASET_ID
    batch_images = workspace / "projects" / _PROJECT / "annotation_batches" / "vid__train" / "images"
    batch_images.mkdir(parents=True)
    frames: list[dict[str, object]] = []
    listed: dict[str, list[dict[str, object]]] = {"train": [], "valid": [], "test": []}
    next_id = 1

    def add(
        name: str,
        *,
        index: int,
        time_s: float,
        episode_id: str,
        group: str,
        review: str,
        directory: str,
        in_coco: bool,
        stored_split: str,
    ) -> None:
        nonlocal next_id
        payload = f"synthetic-{name}".encode()
        (batch_images / name).write_bytes(payload)
        destination = dataset / directory
        destination.mkdir(parents=True, exist_ok=True)
        (destination / name).write_bytes(payload)
        frames.append(
            {
                "image_file": name,
                "video_id": _VIDEO,
                "source_frame_index": index,
                "source_time_s": time_s,
                "episode_id": episode_id,
                "split": stored_split,
                "split_group_id": group,
                "image_sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "width": 8,
                "height": 8,
                "coco_subset": "",
                "flagged": ["unreviewed"] if review == "unreviewed" else [],
                "record": _record(index, review).model_dump(mode="json"),
            }
        )
        if not in_coco:
            return
        listed[directory].append(
            {"id": next_id, "file_name": name, "width": 8, "height": 8}
        )
        next_id += 1

    train_names = [f"tr_{index:03d}.jpg" for index in range(train_count)]
    for index, name in enumerate(train_names):
        add(
            name,
            index=index,
            time_s=float(index + 1),
            episode_id="train-ep",
            group="g-train",
            review="reviewed",
            directory="train",
            in_coco=True,
            stored_split="train",
        )
    add(
        "ambient_val.jpg",
        index=6000,
        time_s=650,
        episode_id="val-ep",
        group="g-val",
        review="reviewed",
        directory="valid",
        in_coco=True,
        stored_split="validation",
    )
    add(
        "ambient_test.jpg",
        index=6001,
        time_s=860,
        episode_id="test-ep",
        group="g-test",
        review="reviewed",
        directory="test",
        in_coco=True,
        stored_split="test",
    )
    if leak is None:
        add(
            "ambient_unreviewed.jpg",
            index=6002,
            time_s=550,
            episode_id="train-ep",
            group="g-train",
            review="unreviewed",
            directory="train",
            in_coco=False,
            stored_split="train",
        )
    if leak == "validation":
        add(
            "leak_validation.jpg",
            index=5000,
            time_s=700,
            episode_id="val-ep",
            group="g-val",
            review="reviewed",
            directory="train",
            in_coco=True,
            stored_split="train",
        )
    elif leak == "test":
        add(
            "leak_test.jpg",
            index=5001,
            time_s=900,
            episode_id="test-ep",
            group="g-test",
            review="reviewed",
            directory="train",
            in_coco=True,
            stored_split="train",
        )
    elif leak == "unreviewed":
        add(
            "leak_unreviewed.jpg",
            index=5002,
            time_s=500,
            episode_id="train-ep",
            group="g-train",
            review="unreviewed",
            directory="train",
            in_coco=True,
            stored_split="train",
        )
    for index in range(extras_in_test):
        add(
            f"pool_{index:03d}.jpg",
            index=7000 + index,
            time_s=float(100 + index),
            episode_id="train-ep",
            group="g-train",
            review="reviewed",
            directory="test",
            in_coco=True,
            stored_split="train",
        )
    if duplicate_first and listed["train"]:
        repeated = dict(listed["train"][0])
        repeated["id"] = next_id
        listed["train"].append(repeated)
    for directory, images in listed.items():
        _dump(
            dataset / directory / "_annotations.coco.json",
            {
                "categories": [{"id": 1, "name": "bucket", "supercategory": "none"}],
                "images": images,
                "annotations": [],
                "marker": _MARKER if directory == "test" else "kept",
            },
        )
    (dataset / "test" / "sentinel.txt").write_text("leave-me", encoding="utf-8")
    _dump(
        batch_images.parent / "imported_labels.json",
        {"schema_version": 1, "frames": frames},
    )
    with storage.connect(storage.project_sqlite(workspace, _PROJECT)) as connection:
        connection.execute(
            """
            INSERT INTO dataset_versions (
                dataset_id, project_id, content_hash, path, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                _DATASET_ID,
                _PROJECT,
                "cd" * 32,
                str(dataset),
                datetime.now(UTC).isoformat(),
            ),
        )
    return _Built(workspace, dataset, train_names)


def _invoke(workspace: Path, profile: str):
    return _RUNNER.invoke(
        app,
        [
            "train",
            "--project",
            _PROJECT,
            "--dataset",
            _DATASET_ID,
            "--profile",
            profile,
            "--workspace",
            str(workspace),
            "--json",
        ],
    )


def _cancel_active(workspace: Path) -> None:
    if not jobs.lab_database(workspace).is_file():
        return
    for record in jobs.reopen(workspace):
        if record.status in {
            JobStatus.queued,
            JobStatus.running,
            JobStatus.cancel_requested,
        }:
            jobs.request_cancel(workspace, _PROJECT, record.job_id, kill_after_s=0.5)


def _wait_calls(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "trainer_calls.json"
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list) and loaded:
                return loaded
        time.sleep(0.02)
    raise AssertionError(f"trainer double wrote no call\n{_logs(run_dir)}")


def _logs(run_dir: Path) -> str:
    chunks = []
    for name in ("worker.log", "train.log"):
        path = run_dir / name
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def _assert_not_quality(document: object) -> None:
    if isinstance(document, dict):
        assert document.get("quality_claim") is not True
        assert document.get("split") not in {"test", "testing"}
        for value in document.values():
            _assert_not_quality(value)
    elif isinstance(document, list):
        for item in document:
            _assert_not_quality(item)


def test_overfit_profile_is_diagnostic_and_reuses_smoke(tmp_path: Path) -> None:
    source = Path(train_mod.__file__).read_text(encoding="utf-8")
    assert "propose_split" not in source
    pilot = load_pilot()
    assert set(pilot.quality_profiles()) == {"smoke", "pilot-small"}
    assert "overfit" not in pilot.quality_profiles()
    assert pilot.profiles.overfit.diagnostic is True
    assert pilot.profiles.smoke.diagnostic is False
    assert pilot.train_profile("overfit").run_test is False

    smoke = resolve_train_profile("smoke")
    overfit = resolve_train_profile("overfit")
    assert overfit.diagnostic is True
    assert overfit.profile == "overfit"
    assert overfit.constructor == smoke.constructor
    assert overfit.constructor["variant"] == "nano"
    assert overfit.constructor["resolution"] == 384
    assert overfit.train_kwargs["amp_dtype"] is None
    assert overfit.train_kwargs["run_test"] is False
    assert overfit.train_kwargs["epochs"] == 8
    assert overfit.train_kwargs["epochs"] < resolve_train_profile("pilot-small").train_kwargs["epochs"]
    for key, value in smoke.train_kwargs.items():
        if key == "epochs":
            continue
        assert overfit.train_kwargs[key] == value
    assert "diagnostic" not in overfit.train_kwargs
    assert "metric_role" not in overfit.train_kwargs
    assert "quality_claim" not in overfit.train_kwargs

    path = tmp_path / "resolved_train_config.json"
    write_resolved_train_config(path, overfit)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["diagnostic"] is True
    assert document["metric_role"] == METRIC_ROLE_DIAGNOSTIC_OVERFIT
    assert document["train_kwargs"]["epochs"] == 8
    assert document["run_test"] is False
    _assert_not_quality(document)

    smoke_path = tmp_path / "smoke.json"
    write_resolved_train_config(smoke_path, smoke)
    smoke_document = json.loads(smoke_path.read_text(encoding="utf-8"))
    assert "diagnostic" not in smoke_document
    assert "metric_role" not in smoke_document
    assert smoke_document.get("quality_claim") is not True


@pytest.mark.parametrize("train_count", [20, 25, 45])
def test_overfit_runs_only_reviewed_train_frames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    train_count: int,
) -> None:
    import rfdetr
    from rfdetr import variants

    class _Spy:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("constructed an RF-DETR model")

    monkeypatch.setattr(rfdetr, "RFDETRNano", _Spy)
    monkeypatch.setattr(rfdetr, "RFDETRSmall", _Spy)
    monkeypatch.setattr(variants, "RFDETRNano", _Spy)
    monkeypatch.setattr(variants, "RFDETRSmall", _Spy)
    monkeypatch.setattr(jobs, "_spawn", _dummy_spawn)
    seen: list[str] = []
    real_require = train_mod.require_frozen

    def _wrapped(workspace: Path, project_id: str, video_id: str):
        seen.append(video_id)
        return real_require(workspace, project_id, video_id)

    monkeypatch.setattr(train_mod, "require_frozen", _wrapped)
    built = _build(tmp_path, train_count=train_count)
    before = _snapshot(built.dataset)
    expected = built.train_names[:OVERFIT_MAX_FRAMES]
    recorded: dict[str, object] = {}

    def trainer(**kwargs: object) -> dict[str, object]:
        view = Path(str(kwargs["dataset_dir"]))
        assert not (view / "test").exists()
        assert view.resolve() != built.dataset.resolve()
        jpgs: set[Path] = set()
        for directory in ("train", "valid"):
            folder = view / directory
            names = []
            for path in folder.iterdir():
                if path.suffix != ".jpg":
                    continue
                assert path.is_symlink()
                jpgs.add(path.resolve())
                names.append(path.name)
                with path.open("rb") as handle:
                    handle.read(1)
            document = json.loads((folder / "_annotations.coco.json").read_text(encoding="utf-8"))
            assert _MARKER not in json.dumps(document)
            assert [row["file_name"] for row in document["images"]] == expected
            assert sorted(names) == sorted(expected)
        test_root = (built.dataset / "test").resolve()
        valid_root = (built.dataset / "valid").resolve()
        train_root = (built.dataset / "train").resolve()
        assert {path.name for path in jpgs} == set(expected)
        for path in jpgs:
            assert path.parent == train_root
            assert test_root not in path.parents
            assert valid_root not in path.parents
        assert (train_root / "ambient_unreviewed.jpg").is_file()
        assert train_root / "ambient_unreviewed.jpg" not in jpgs
        recorded["kwargs"] = dict(kwargs)
        output = Path(str(kwargs["output_dir"]))
        (output / "weights.pth").write_bytes(b"checkpoint")
        return {
            "val/mAP_50_95": 0.42,
            "test/mAP_50_95": 0.99,
            "val/AP/bucket": 0.17,
            "test/AP/bucket": 0.88,
            "per_class": [{"name": "bucket", "ap": 0.88, "split": "test"}],
        }

    started = _invoke(built.workspace, "overfit")
    assert started.exit_code == 0, started.output
    job_id = json.loads(started.stdout)["data"]["job_id"]
    assert train_mod.run_train_job(built.workspace, job_id, trainer=trainer) == 0
    assert seen
    assert set(seen) == {_VIDEO}
    kwargs = recorded["kwargs"]
    assert isinstance(kwargs, dict)
    resolved = resolve_train_profile("overfit")
    assert kwargs["epochs"] == resolved.train_kwargs["epochs"] == 8
    assert kwargs["run_test"] is False
    assert kwargs["amp_dtype"] is None
    assert kwargs["batch_size"] == 1
    assert "gradient_checkpointing" not in kwargs
    run_dir = jobs.run_directory(built.workspace, _PROJECT, job_id)
    for name in (OVERFIT_OUTPUT_NAME, "validation_metrics.json"):
        text = (run_dir / name).read_text(encoding="utf-8")
        document = json.loads(text)
        assert document["metric_role"] == METRIC_ROLE_DIAGNOSTIC_OVERFIT
        assert document["diagnostic"] is True
        assert document["box_map_50_95"] == 0.42
        assert document["bucket_ap"] == 0.17
        assert document["selection_unstable"] is True
        assert document["validation_images"] == len(expected)
        assert "0.99" not in text
        assert "0.88" not in text
        _assert_not_quality(document)
    output = json.loads((run_dir / OVERFIT_OUTPUT_NAME).read_text(encoding="utf-8"))
    assert output["profile"] == "overfit"
    assert output["frame_count"] == len(expected)
    assert output["frames"] == expected
    resolved_text = (run_dir / "resolved_train_config.json").read_text(encoding="utf-8")
    resolved_document = json.loads(resolved_text)
    assert resolved_document["diagnostic"] is True
    assert resolved_document["metric_role"] == METRIC_ROLE_DIAGNOSTIC_OVERFIT
    assert resolved_document["train_kwargs"]["epochs"] == 8
    _assert_not_quality(resolved_document)
    assert _MARKER.encode() not in resolved_text.encode()
    assert (built.dataset / "test" / "sentinel.txt").read_text(encoding="utf-8") == "leave-me"
    assert _snapshot(built.dataset) == before
    if train_count > OVERFIT_MAX_FRAMES:
        assert (built.dataset / "train" / f"tr_{OVERFIT_MAX_FRAMES:03d}.jpg").is_file()
        assert f"tr_{OVERFIT_MAX_FRAMES:03d}.jpg" not in output["frames"]


@pytest.mark.parametrize(
    ("leak", "needle"),
    [
        ("validation", "validation"),
        ("test", "test"),
        ("unreviewed", "не проверен"),
    ],
)
def test_overfit_rejects_frames_outside_reviewed_train(
    tmp_path: Path,
    leak: str,
    needle: str,
) -> None:
    built = _build(tmp_path, train_count=20, leak=leak)
    before = _snapshot(built.dataset)
    result = _invoke(built.workspace, "overfit")
    assert result.exit_code != 0, result.output
    body = json.loads(result.stdout)
    assert body["ok"] is False
    assert body["error"]["code"] == "overfit_frame_rejected"
    assert needle in body["error"]["message"]
    assert _cyrillic(body["error"]["message"])
    assert _snapshot(built.dataset) == before
    assert list(built.workspace.rglob("*.pth")) == []
    runs = built.workspace / "projects" / _PROJECT / "runs"
    assert not runs.exists() or list(runs.iterdir()) == []
    assert (built.dataset / "test" / "sentinel.txt").read_text(encoding="utf-8") == "leave-me"


def test_overfit_refuses_to_invent_frames_below_20(tmp_path: Path) -> None:
    built = _build(tmp_path, train_count=19, extras_in_test=5, duplicate_first=True)
    before = _snapshot(built.dataset)
    result = _invoke(built.workspace, "overfit")
    assert result.exit_code != 0, result.output
    body = json.loads(result.stdout)
    assert body["ok"] is False
    assert body["error"]["code"] == "not_enough_train_frames"
    assert "19" in body["error"]["message"]
    assert "не копируются" in body["error"]["message"]
    assert _cyrillic(body["error"]["message"])
    assert _snapshot(built.dataset) == before
    train_names = {path.name for path in (built.dataset / "train").iterdir()}
    assert "tr_018.jpg" in train_names
    assert "tr_019.jpg" not in train_names
    assert not any(name.startswith("pool_") for name in train_names)
    assert (built.dataset / "test" / "pool_000.jpg").is_file()
    assert (built.dataset / "test" / "sentinel.txt").read_text(encoding="utf-8") == "leave-me"
    assert list(built.workspace.rglob("*.pth")) == []
    runs = built.workspace / "projects" / _PROJECT / "runs"
    assert not runs.exists() or list(runs.iterdir()) == []


def test_overfit_second_job_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(train_mod.FAKE_TRAINER_ENV, "hold")
    built = _build(tmp_path, train_count=20)
    try:
        started = _invoke(built.workspace, "overfit")
        assert started.exit_code == 0, started.output
        body = json.loads(started.stdout)
        job_id = body["data"]["job_id"]
        run_dir = jobs.run_directory(built.workspace, _PROJECT, job_id)
        calls = _wait_calls(run_dir)
        opened = calls[0]["opened"]
        assert isinstance(opened, list)
        test_root = (built.dataset / "test").resolve()
        assert not any(
            Path(str(path)) == test_root or test_root in Path(str(path)).parents for path in opened
        )
        second = _invoke(built.workspace, "overfit")
        assert second.exit_code != 0, second.output
        failure = json.loads(second.stdout)
        assert failure["error"]["code"] == "job_already_running"
        assert len(json.loads((run_dir / "trainer_calls.json").read_text(encoding="utf-8"))) == 1
        assert (built.dataset / "test" / "sentinel.txt").read_text(encoding="utf-8") == "leave-me"
    finally:
        _cancel_active(built.workspace)
