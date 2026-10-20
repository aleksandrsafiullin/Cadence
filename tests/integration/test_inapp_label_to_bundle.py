"""In-app labels → snapshot → training page → worker → bundle on Analysis.

The MP4 and the boxes are synthetic. They check the wiring. They are not
a quality claim for pit video or for the detector.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import re
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from excavation_lab import jobs, storage
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import list_catalogs, read_catalog
from excavation_lab.datasets.coco_import import import_coco
from excavation_lab.datasets.cvat_export import MANIFEST_COLUMNS
from excavation_lab.datasets.label_store import sample_rows_with_role, save_frame_label
from excavation_lab.datasets.labels import BoxDraft
from excavation_lab.datasets.sampler import SampleRow, sample_frames
from excavation_lab.datasets.snapshot import snapshot_dataset
from excavation_lab.datasets.splits import freeze_split, write_blocks
from excavation_lab.errors import JobAlreadyRunningError
from excavation_lab.jobs import PROGRESS_FILENAME, get_job, run_directory
from excavation_lab.media.probe import add_video
from excavation_lab.models.bundle import load_bundle
from excavation_lab.models.train import FAKE_TRAINER_ENV
from excavation_lab.schemas import JobStatus, ReviewStatus, SplitName
from excavation_lab.ui import copy as ui_copy
from excavation_lab.ui import training_page
from excavation_lab.ui.analysis_page import list_bundles
from excavation_lab.web.app import create_app
from excavation_lab.web.preferences import write_project_file

_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SPEC = importlib.util.spec_from_file_location(
    "excavation_lab_synthetic_inapp_train",
    _SYNTHETIC_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_mp4 = _SYNTHETIC.write_synthetic_mp4

_BLOCKS = (
    RoleBlock(start_s=0, end_s=8, role="train"),
    RoleBlock(start_s=18, end_s=24, role="validation"),
    RoleBlock(start_s=34, end_s=35.5, role="test"),
)
_ACTIVE = frozenset(
    {JobStatus.queued, JobStatus.running, JobStatus.cancel_requested}
)
_HEADER = "epoch,train_loss,epoch_duration_s\n"
_ROWS = _HEADER + "1,1.2,4\n2,0.8,5\n"


def test_inapp_snapshot_trains_to_a_bundle_on_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fake trainer through the worker. The page refuses a second start."""
    workspace, dataset_id, dataset_dir, order = _labelled_snapshot(tmp_path)
    monkeypatch.setenv(FAKE_TRAINER_ENV, "hold")
    try:
        with TestClient(create_app(workspace)) as client:
            idle = client.get("/training")
            assert idle.status_code == 200
            assert dataset_id in _option_values(idle.text, "train-dataset")
            assert _selected(idle.text, "train-profile")[0] == "smoke"
            assert _disabled(idle.text, "train-start") is False
            assert _node_text(idle.text, "train-eta") == ui_copy.NOT_CALCULATED
            assert not _chart_markup(idle.text)

            holding = _post(client, "/training/start", _fields(dataset_id))
            hold = _only_active(workspace)
            _wait_progress(workspace, hold.job_id, JobStatus.running.value)
            assert get_job(workspace, "pilot", hold.job_id).status == JobStatus.running
            assert _disabled(holding, "train-start") is True
            refused = _post(client, "/training/start", _fields(dataset_id))
            assert JobAlreadyRunningError().message in refused
            assert "job_already_running" in refused
            assert _only_active(workspace).job_id == hold.job_id
            running = client.get(f"/jobs/{hold.job_id}/progress")
            assert running.status_code == 200
            running_body = running.json()
            assert running_body["status"] == JobStatus.running.value
            assert running_body["progress"]["job_id"] == hold.job_id
            assert running_body["progress"]["run_test"] is False
            assert running_body["chart"] is None

            cancelled = _post(client, "/training/cancel", {"job_id": hold.job_id})
            assert get_job(workspace, "pilot", hold.job_id).status == JobStatus.cancelled
            assert _disabled(cancelled, "train-start") is False

            monkeypatch.setenv(FAKE_TRAINER_ENV, "reload")
            started = _post(client, "/training/start", _fields(dataset_id))
            assert ui_copy.catalog("ru").train_queued in started
            trained = _finished_job(workspace, hold.job_id)
            assert trained.status == JobStatus.completed, _logs(workspace, trained.job_id)
            run_dir = run_directory(workspace, "pilot", trained.job_id)
            _wait_progress(workspace, trained.job_id, JobStatus.completed.value)
            progress = json.loads((run_dir / PROGRESS_FILENAME).read_text(encoding="utf-8"))
            assert progress["status"] == JobStatus.completed.value
            assert progress["run_test"] is False
            assert progress["profile"] == "smoke"

            calls = json.loads((run_dir / "trainer_calls.json").read_text(encoding="utf-8"))
            assert calls[0]["kwargs"]["run_test"] is False
            assert calls[0]["model_constructed"] is False
            _assert_test_unopened(calls[0]["opened"], dataset_dir)
            resolved = json.loads(
                (run_dir / "resolved_train_config.json").read_text(encoding="utf-8")
            )
            assert resolved["run_test"] is False
            log = (run_dir / "train.log").read_text(encoding="utf-8")
            assert "run_test=false" in log
            assert "profile=smoke" in log

            model_id, bundle = _bundle_row(workspace)
            assert (bundle / "weights.pth").is_file()
            assert model_id in list_bundles(workspace, "pilot")
            metrics_text = (run_dir / "validation_metrics.json").read_text(encoding="utf-8")
            metrics = json.loads(metrics_text)
            assert metrics["box_map_50_95"] is None
            assert metrics["bucket_ap"] is None
            assert metrics["synthetic"] is True
            assert "0.5" not in metrics_text
            assert "0.25" not in metrics_text
            card = (bundle / "model_card.md").read_text(encoding="utf-8")
            assert ui_copy.catalog("ru").synthetic_card_use in card
            assert ui_copy.NOT_CALCULATED in card
            assert "0.5" not in card
            published_config = json.loads(
                (bundle / "model_config.json").read_text(encoding="utf-8")
            )
            assert published_config["origin"] == "custom"
            assert published_config["synthetic"] is True
            assert published_config["metric_role"] == metrics["metric_role"]
            mapping = json.loads((bundle / "class_mapping.json").read_text(encoding="utf-8"))
            assert mapping["class_names"] == order
            for index, name in enumerate(order):
                assert mapping["by_name"][name]["internal_id"] == index
            published = json.loads(
                (bundle / "resolved_train_config.json").read_text(encoding="utf-8")
            )
            assert published["run_test"] is False

            def _refuse(*_args: object, **_kwargs: object) -> None:
                raise AssertionError("RF-DETR constructor")

            import rfdetr

            monkeypatch.setattr(rfdetr, "RFDETRNano", _refuse)
            monkeypatch.setattr(rfdetr, "RFDETRSmall", _refuse)
            adapter = load_bundle(bundle)
            try:
                assert list(adapter.metadata().class_names) == order
            finally:
                adapter.close()

            (run_dir / "metrics_history.csv").write_text(_HEADER, encoding="utf-8")
            header_page = client.get("/training")
            assert header_page.status_code == 200
            assert f'<p class="lab-empty">{ui_copy.catalog("ru").training_empty}</p>' in (
                header_page.text
            )
            assert _node_text(header_page.text, "train-eta") == ui_copy.NOT_CALCULATED
            header_progress = client.get(f"/jobs/{trained.job_id}/progress").json()
            assert header_progress["chart"] is None
            assert (
                training_page.should_draw_chart(
                    training_page.training_log_view(_HEADER, planned_epochs=2),
                    None,
                )
                is False
            )

            (run_dir / "metrics_history.csv").write_text(_ROWS, encoding="utf-8")
            drawn = client.get("/training")
            assert f'<p class="lab-empty">{ui_copy.catalog("ru").training_empty}</p>' not in (
                drawn.text
            )
            assert _node_text(drawn.text, "train-eta") == ui_copy.NOT_CALCULATED
            assert f'data-progress-url="jobs/{trained.job_id}/progress"' in drawn.text
            assert 'data-status="completed"' in drawn.text
            measured = training_page.training_log_view(_ROWS, planned_epochs=2)
            assert measured.eta_s is None
            assert training_page.should_draw_chart(measured, None) is True
            chart_body = client.get(f"/jobs/{trained.job_id}/progress").json()
            assert chart_body["status"] == JobStatus.completed.value
            assert chart_body["chart"] is not None
            chart_text = json.dumps(chart_body["chart"])
            assert "1.2" in chart_text
            assert "0.8" in chart_text

            analysis = client.get("/analysis?mode=custom")
            assert analysis.status_code == 200
            assert model_id in analysis.text
            assert ui_copy.catalog("ru").bundle_synthetic in analysis.text
            assert 'id="analysis-bundle"' in analysis.text
    finally:
        _stop(workspace)


def test_cvat_import_keeps_the_in_app_negative(tmp_path: Path) -> None:
    """A later CVAT frame on the same split does not drop the in-app negative."""
    workspace, _dataset_id, _dataset_dir, order = _labelled_snapshot(tmp_path)
    listed = sample_rows_with_role(workspace, "pilot", "vid")
    train = [item.row for item in listed if item.role is SplitName.train]
    assert len(train) >= 3
    negative = train[1]
    imported = train[2]
    zip_path = workspace / "incoming.zip"
    _write_frame_zip(workspace, zip_path, imported)
    result = import_coco(workspace, "pilot", "vid__train", zip_path)
    assert result.image_count == 1
    snapshot = snapshot_dataset(workspace, "pilot")
    root = Path(snapshot.path)
    coco = json.loads((root / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    names = {image["file_name"] for image in coco["images"]}
    assert negative.image_file in names
    assert imported.image_file in names
    assert train[0].image_file in names
    supervision = json.loads(
        (root / "train" / "class_supervision.json").read_text(encoding="utf-8")
    )
    side = supervision["images"][negative.image_file]
    assert side["positives"] == []
    assert side["unsupervised"] == []
    assert set(side["negatives"]) == set(order)
    mapping = json.loads((root / "class_mapping.json").read_text(encoding="utf-8"))
    ranked = sorted(mapping["by_name"].items(), key=lambda item: item[1]["internal_id"])
    assert [name for name, _spec in ranked] == order


def _write_frame_zip(workspace: Path, zip_path: Path, row: SampleRow) -> None:
    image_file = row.image_file
    payload = (
        storage.project_directory(workspace, "pilot") / "samples" / "vid" / "images" / image_file
    ).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == row.image_sha256
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    height, width = int(image.shape[0]), int(image.shape[1])
    assert row.source_pts is not None and row.time_base is not None
    manifest = {
        "image_file": image_file,
        "video_id": "vid",
        "source_frame_index": str(row.source_frame_index),
        "source_pts": str(row.source_pts),
        "time_base": row.time_base,
        "source_time_s": f"{row.source_time_s:.6f}",
        "episode_id": row.episode_id,
        "split": "train",
        "split_group_id": "",
        "image_sha256": row.image_sha256,
        "width": str(width),
        "height": str(height),
    }
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerow(manifest)
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
                "bbox": [4, 4, 10, 8],
                "area": 80,
                "iscrowd": 0,
            }
        ],
    }
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("frames_manifest.csv", buffer.getvalue())
        archive.writestr(
            "annotations/instances_default.json",
            json.dumps(document, ensure_ascii=False) + "\n",
        )
        archive.writestr(f"images/{image_file}", payload)


def _labelled_snapshot(tmp_path: Path) -> tuple[Path, str, Path, list[str]]:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    write_project_file(workspace, "pilot")
    clip = workspace / "clip.mp4"
    write_synthetic_mp4(clip, 36, 5)
    video = add_video(workspace, "pilot", clip, video_id="vid")
    video_id = str(video["video_id"])
    freeze_split(
        workspace,
        "pilot",
        write_blocks(workspace, "pilot", video_id, list(_BLOCKS)),
    )
    sample_frames(workspace, "pilot", plan="content")
    listed = sample_rows_with_role(workspace, "pilot", video_id)
    by_role = {
        SplitName.train: [item.row for item in listed if item.role is SplitName.train],
        SplitName.validation: [
            item.row for item in listed if item.role is SplitName.validation
        ],
        SplitName.test: [item.row for item in listed if item.role is SplitName.test],
    }
    assert len(by_role[SplitName.train]) >= 2
    assert len(by_role[SplitName.validation]) >= 1
    assert len(by_role[SplitName.test]) >= 1
    box = BoxDraft(
        category_name="excavator_body",
        x1=8,
        y1=8,
        x2=40,
        y2=30,
    )
    save_frame_label(
        workspace,
        "pilot",
        video_id,
        by_role[SplitName.train][0].source_frame_index,
        review_status=ReviewStatus.reviewed,
        boxes=[box],
        annotator="operator",
    )
    empty = save_frame_label(
        workspace,
        "pilot",
        video_id,
        by_role[SplitName.train][1].source_frame_index,
        review_status=ReviewStatus.reviewed,
        boxes=[],
        annotator="operator",
    )
    assert empty.label_state.value == "reviewed_empty"
    save_frame_label(
        workspace,
        "pilot",
        video_id,
        by_role[SplitName.validation][0].source_frame_index,
        review_status=ReviewStatus.reviewed,
        boxes=[box],
        annotator="operator",
    )
    save_frame_label(
        workspace,
        "pilot",
        video_id,
        by_role[SplitName.test][0].source_frame_index,
        review_status=ReviewStatus.reviewed,
        boxes=[box],
        annotator="operator",
    )
    snapshot = snapshot_dataset(workspace, "pilot")
    dataset_dir = Path(snapshot.path)
    assert (dataset_dir / "train").is_dir()
    assert (dataset_dir / "valid").is_dir()
    assert (dataset_dir / "test").is_dir()
    assert list((dataset_dir / "test").glob("*.jpg"))
    catalog = read_catalog(workspace, "excavation")
    order = [item.name for item in sorted(catalog.classes, key=lambda item: item.id)]
    return workspace, snapshot.dataset_id, dataset_dir, order


def _assert_test_unopened(opened: object, dataset_dir: Path) -> None:
    assert isinstance(opened, list) and opened
    test_root = (dataset_dir / "test").resolve()
    for item in opened:
        assert isinstance(item, str)
        path = Path(item)
        assert "test" not in path.parts
        resolved = path.resolve()
        assert resolved != test_root
        assert test_root not in resolved.parents


def _bundle_row(workspace: Path) -> tuple[str, Path]:
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        rows = connection.execute(
            """
            SELECT model_id, path FROM model_versions
            WHERE project_id = ?
            """,
            ("pilot",),
        ).fetchall()
    assert len(rows) == 1
    model_id = rows[0]["model_id"]
    raw = rows[0]["path"]
    assert isinstance(model_id, str) and isinstance(raw, str)
    return model_id, Path(raw)


def _only_active(workspace: Path):
    found = [job for job in jobs.reopen(workspace) if job.status in _ACTIVE]
    assert len(found) == 1, [(job.job_id, job.status.value) for job in jobs.reopen(workspace)]
    return found[0]


def _finished_job(workspace: Path, skipped: str):
    deadline = time.monotonic() + 90
    latest = None
    while time.monotonic() < deadline:
        for job in jobs.reopen(workspace):
            if job.job_id == skipped:
                continue
            latest = job
            if job.status == JobStatus.completed:
                return job
            if job.status == JobStatus.failed:
                raise AssertionError(_logs(workspace, job.job_id))
        time.sleep(0.05)
    raise AssertionError(_logs(workspace, None if latest is None else latest.job_id))


def _wait_progress(workspace: Path, job_id: str, status: str) -> None:
    path = run_directory(workspace, "pilot", job_id) / PROGRESS_FILENAME
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded.get("status") == status:
                return
        time.sleep(0.05)
    raise AssertionError(_logs(workspace, job_id))


def _logs(workspace: Path, job_id: str | None) -> str:
    if job_id is None:
        return "train job did not finish"
    run_dir = run_directory(workspace, "pilot", job_id)
    parts = [f"job {job_id}"]
    for name in ("train.log", "worker.log"):
        path = run_dir / name
        if path.is_file():
            parts.append(f"{name}:\n{path.read_text(encoding='utf-8')[-4000:]}")
    return "\n".join(parts)


def _stop(workspace: Path) -> None:
    if not jobs.lab_database(workspace).is_file():
        return
    for job in jobs.reopen(workspace):
        if job.status in _ACTIVE:
            jobs.request_cancel(workspace, "pilot", job.job_id, kill_after_s=1)


def _fields(dataset_id: str) -> dict[str, str]:
    values = training_page.profile_field_values("smoke")
    return {
        "dataset_id": dataset_id,
        "profile": "smoke",
        "variant": str(values["variant"]),
        "device": "auto",
        "resolution": str(values["resolution"]),
        "epochs": str(values["epochs"]),
        "batch_size": str(values["batch_size"]),
        "gradient_accumulation": str(values["gradient_accumulation"]),
        "learning_rate": str(values["learning_rate"]),
        "encoder_learning_rate": str(values["encoder_learning_rate"]),
        "workers": str(values["workers"]),
        "seed": str(values["seed"]),
    }


def _post(client: TestClient, path: str, data: dict[str, str]) -> str:
    response = client.post(path, data=data, follow_redirects=False)
    assert response.status_code == 303, response.text
    page = client.get("/training")
    assert page.status_code == 200
    return page.text


def _option_values(html: str, element_id: str) -> list[str]:
    body = _select_body(html, element_id)
    return re.findall(r'<option\b[^>]*value="([^"]*)"', body)


def _selected(html: str, element_id: str) -> tuple[str, str]:
    body = _select_body(html, element_id)
    for attrs, label in re.findall(r"<option\b([^>]*)>([^<]*)</option>", body):
        if re.search(r"\sselected(?:\s|>|$)", attrs):
            value = re.search(r'value="([^"]*)"', attrs)
            assert value is not None
            return value.group(1), label
    raise AssertionError(element_id)


def _select_body(html: str, element_id: str) -> str:
    match = re.search(
        rf'<select\b[^>]*id="{element_id}"[^>]*>(.*?)</select>',
        html,
        re.DOTALL,
    )
    assert match is not None, element_id
    return match.group(1)


def _node_text(html: str, element_id: str) -> str:
    match = re.search(rf'id="{element_id}"[^>]*>([^<]*)</', html)
    assert match is not None, element_id
    return match.group(1)


def _disabled(html: str, element_id: str) -> bool:
    match = re.search(rf'<button\b[^>]*id="{element_id}"[^>]*>', html)
    assert match is not None, element_id
    return re.search(r"\sdisabled(?:\s|=|>)", match.group(0)) is not None


def _chart_markup(html: str) -> bool:
    lowered = html.lower()
    return "lab-chart" in html or 'id="loss-chart"' in html or "<svg" in lowered
