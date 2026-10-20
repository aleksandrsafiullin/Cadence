"""Five-stage stepper. Synthetic rows, no detector and no parquet."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import cv2
import numpy as np

from excavation_lab import jobs, storage
from excavation_lab.actions.catalog import add_action
from excavation_lab.actions.labels import add_interval
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.label_store import save_frame_label
from excavation_lab.datasets.labels import BoxDraft, ReviewStatus
from excavation_lab.datasets.sampler import SampleResult, SampleRow
from excavation_lab.datasets.splits import freeze_split, require_frozen, write_blocks
from excavation_lab.errors import JobAlreadyRunningError
from excavation_lab.schemas import SplitCounts
from excavation_lab.ui.copy import catalog
from excavation_lab.ui.data_page import snapshot_gate
from excavation_lab.ui.workflow import project_stages

_PROJECT = "pilot"
_VIDEO = "clip"
_CYRILLIC = "\u0400"


def test_workflow_module_stays_off_the_page_server() -> None:
    source = Path("src/excavation_lab/ui/workflow.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("starlette" in name or "pyarrow" in name for name in imported)


def test_missing_project_waits_on_every_stage() -> None:
    book = catalog("ru")
    english = catalog("en")
    stages = project_stages(Path("/tmp/does-not-matter"), None, book)
    assert [stage.state for stage in stages] == ["waiting"] * 5
    assert {stage.caption for stage in stages} == {book.need_project}
    assert [stage.path for stage in stages] == [
        "/video",
        "/data",
        "/training",
        "/analysis",
        "/actions",
    ]
    english_stages = project_stages(Path("/tmp/does-not-matter"), "", english)
    _assert_latin(english_stages)
    assert {stage.caption for stage in english_stages} == {english.need_project}


def test_stages_advance_one_current_at_each_milestone(tmp_path: Path) -> None:
    book = catalog("ru")
    english = catalog("en")
    assert book.job_already_running == JobAlreadyRunningError().message
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project(_PROJECT, workspace, catalog_id="excavation")

    stages = project_stages(workspace, _PROJECT, book)
    _assert_current(stages, "video")
    assert stages[0].caption == book.add_local_video

    storage.insert_video(workspace, _PROJECT, _VIDEO, name="clip")
    stages = project_stages(workspace, _PROJECT, book)
    _assert_current(stages, "video")

    _probe(workspace)
    stages = project_stages(workspace, _PROJECT, book)
    _assert_current(stages, "data")
    assert stages[1].caption == book.snapshot_locked
    _assert_latin(project_stages(workspace, _PROJECT, english))

    proposal = write_blocks(
        workspace,
        _PROJECT,
        _VIDEO,
        [
            RoleBlock(start_s=0, end_s=10, role="train"),
            RoleBlock(start_s=20, end_s=30, role="validation"),
        ],
    )
    freeze_split(workspace, _PROJECT, proposal)
    stages = project_stages(workspace, _PROJECT, book)
    _assert_current(stages, "data")
    assert stages[1].caption == book.contact_empty

    _plant_samples(workspace)
    stages = project_stages(workspace, _PROJECT, book)
    _assert_current(stages, "data")
    assert stages[1].caption == snapshot_gate(workspace, _PROJECT, book).reason
    assert "проверенных кадров" in stages[1].caption

    _review(workspace, 0)
    _review(workspace, 1)
    stages = project_stages(workspace, _PROJECT, book)
    _assert_current(stages, "data")
    assert stages[1].caption == book.snapshot_not_built
    _assert_latin(project_stages(workspace, _PROJECT, english))

    _plant_snapshot(workspace)
    stages = project_stages(workspace, _PROJECT, book)
    _assert_current(stages, "training")
    assert stages[2].caption == book.train_jobs_empty

    _insert_job(workspace, "job-train", "train", "rfdetr", "running")
    stages = project_stages(workspace, _PROJECT, book)
    _assert_current(stages, "training")
    assert stages[2].caption == book.job_already_running
    _delete_job(workspace, "job-train")

    _plant_bundle(workspace)
    _insert_run(workspace, "run-base", "job-base", "baseline_coco", "completed")
    stages = project_stages(workspace, _PROJECT, book)
    _assert_current(stages, "analysis")
    assert stages[3].caption == book.analysis_empty

    _insert_run(workspace, "run-custom", "job-custom", "custom", "completed")
    stages = project_stages(workspace, _PROJECT, book)
    _assert_current(stages, "actions")
    assert stages[4].caption == book.action_step_need_catalog

    add_action(workspace, _PROJECT, "digging", "Dig")
    stages = project_stages(workspace, _PROJECT, book)
    _assert_current(stages, "actions")
    assert stages[4].caption == book.action_coverage_hint

    add_interval(workspace, _PROJECT, _VIDEO, 1, 3, "digging")
    add_interval(workspace, _PROJECT, _VIDEO, 21, 24, "digging")
    stages = project_stages(workspace, _PROJECT, book)
    _assert_current(stages, "actions")
    assert stages[4].caption == book.action_no_model

    storage.insert_action_model(
        workspace,
        _PROJECT,
        action_model_id="act1",
        detector_model_id="m1",
        source_run_ids=["run-custom"],
        path=storage.project_directory(workspace, _PROJECT) / "action_models" / "act1",
    )
    stages = project_stages(workspace, _PROJECT, book)
    _assert_current(stages, "actions")
    assert stages[4].caption == book.action_no_inference

    storage.insert_action_inference(
        workspace,
        _PROJECT,
        job_id="job-infer",
        source_run_id="run-custom",
        action_model_id="act1",
        path=storage.project_directory(workspace, _PROJECT) / "runs" / "job-infer",
    )
    stages = project_stages(workspace, _PROJECT, book)
    assert [stage.state for stage in stages] == ["done"] * 5
    assert all(stage.caption == "" for stage in stages)
    _assert_latin(project_stages(workspace, _PROJECT, english))


def _assert_current(stages: tuple[object, ...], stage_id: str) -> None:
    current = [stage for stage in stages if stage.state == "current"]  # type: ignore[attr-defined]
    assert len(current) == 1
    assert current[0].stage_id == stage_id  # type: ignore[attr-defined]
    seen = False
    for stage in stages:
        if stage.stage_id == stage_id:  # type: ignore[attr-defined]
            seen = True
            continue
        if not seen:
            assert stage.state == "done"  # type: ignore[attr-defined]
        else:
            assert stage.state == "waiting"  # type: ignore[attr-defined]


def _assert_latin(stages: tuple[object, ...]) -> None:
    for stage in stages:
        for text in (stage.title, stage.caption, stage.status):  # type: ignore[attr-defined]
            assert not any(_CYRILLIC <= char <= "\u04ff" for char in text)


def _probe(workspace: Path) -> None:
    with storage.connect(storage.project_sqlite(workspace, _PROJECT)) as connection:
        connection.execute(
            """
            UPDATE videos
            SET duration_s = ?, width = ?, height = ?, fps_nominal = ?
            WHERE video_id = ?
            """,
            (40.0, 32, 24, 25.0, _VIDEO),
        )


def _jpeg(tone: int) -> bytes:
    image = np.full((24, 32, 3), tone, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    return encoded.tobytes()


def _plant_samples(workspace: Path) -> None:
    frozen = require_frozen(workspace, _PROJECT, _VIDEO)
    root = storage.project_directory(workspace, _PROJECT) / "samples" / _VIDEO
    images = root / "images"
    images.mkdir(parents=True)
    specs = ((0, 1.0, _jpeg(20), "train"), (1, 21.0, _jpeg(40), "validation"))
    rows: list[SampleRow] = []
    for index, time_s, payload, split in specs:
        name = f"{_VIDEO}_{index}.jpg"
        (images / name).write_bytes(payload)
        rows.append(
            SampleRow(
                video_id=_VIDEO,
                source_frame_index=index,
                source_time_s=time_s,
                episode_id="",
                split=split,
                image_sha256=hashlib.sha256(payload).hexdigest(),
                image_file=name,
                strategy="spaced",
                source_pts=index * 1000,
                time_base="1/1000",
            )
        )
    counts = SplitCounts(train=1, validation=1, test=0)
    manifest = root / "sample_manifest.json"
    body = SampleResult(
        project_id=_PROJECT,
        video_id=_VIDEO,
        split_content_hash=frozen.content_hash,
        spacing_s=1.0,
        requested=counts,
        written=counts,
        shortfall=SplitCounts(train=0, validation=0, test=0),
        rows=rows,
        manifest_path=str(manifest),
        images_dir=str(images),
    )
    manifest.write_text(body.model_dump_json(), encoding="utf-8")


def _review(workspace: Path, index: int) -> None:
    save_frame_label(
        workspace,
        _PROJECT,
        _VIDEO,
        index,
        review_status=ReviewStatus.reviewed,
        boxes=[BoxDraft(category_name="bucket", x1=1, y1=1, x2=8, y2=8)],
        annotator="alex",
    )


def _plant_snapshot(workspace: Path) -> None:
    project = storage.project_directory(workspace, _PROJECT)
    snapshot = project / "dataset_versions" / "ds1"
    snapshot.mkdir(parents=True)
    (snapshot / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
    with storage.connect(storage.project_sqlite(workspace, _PROJECT)) as connection:
        connection.execute(
            """
            INSERT INTO dataset_versions (
                dataset_id, project_id, content_hash, path, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("ds1", _PROJECT, "cd" * 32, str(snapshot), "2020-01-01T00:00:00+00:00"),
        )


def _plant_bundle(workspace: Path) -> None:
    project = storage.project_directory(workspace, _PROJECT)
    directory = project / "models" / "m1"
    directory.mkdir(parents=True)
    (directory / "weights.pth").write_bytes(b"weights")
    with storage.connect(storage.project_sqlite(workspace, _PROJECT)) as connection:
        connection.execute(
            """
            INSERT INTO model_versions (
                model_id, project_id, dataset_id, variant, path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("m1", _PROJECT, "ds1", "nano", str(directory), "2020-01-02T00:00:00+00:00"),
        )


def _insert_run(
    workspace: Path,
    run_id: str,
    job_id: str,
    origin: str,
    status: str,
) -> None:
    with storage.connect(storage.project_sqlite(workspace, _PROJECT)) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                job_id, project_id, status, pid, partial,
                analyzed_end_s, error_code, created_at
            ) VALUES (?, ?, ?, NULL, 0, NULL, NULL, ?)
            """,
            (job_id, _PROJECT, status, "2020-01-03T00:00:00+00:00"),
        )
    storage.insert_analysis_run(
        workspace,
        _PROJECT,
        run_id=run_id,
        job_id=job_id,
        video_id=_VIDEO,
        view_id=None,
        model_id="m1",
        model_origin=origin,
        analysis_fps=5.0,
        source_fps=None,
        export_fps=None,
        start_s=0.0,
        end_s=10.0,
        decode_ms=None,
        encode_ms=None,
    )
    _insert_job(workspace, job_id, "analyze", "analyze", status)


def _insert_job(
    workspace: Path,
    job_id: str,
    kind: str,
    worker: str,
    status: str,
) -> None:
    run_dir = jobs.run_directory(workspace, _PROJECT, job_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with jobs._lab(workspace) as connection:
        connection.execute(
            """
            INSERT INTO lab_jobs (
                job_id, project_id, kind, worker, status, pid, partial,
                analyzed_end_s, error_code, orphaned, run_dir,
                video_duration_s, step_s, sleep_s, created_at
            ) VALUES (
                ?, ?, ?, ?, ?, NULL, 0, NULL, NULL, 0, ?,
                40, 0.2, 0.05, ?
            )
            """,
            (
                job_id,
                _PROJECT,
                kind,
                worker,
                status,
                str(run_dir),
                "2020-01-03T00:00:00+00:00",
            ),
        )


def _delete_job(workspace: Path, job_id: str) -> None:
    with jobs._lab(workspace) as connection:
        connection.execute("DELETE FROM lab_jobs WHERE job_id = ?", (job_id,))
        connection.execute("DELETE FROM workspace_lock WHERE job_id = ?", (job_id,))


def test_readonly_job_lookup_does_not_create_lab_sqlite(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project(_PROJECT, workspace, catalog_id="excavation")
    project_stages(workspace, _PROJECT, catalog("en"))
    assert not (workspace / "lab.sqlite").exists()
