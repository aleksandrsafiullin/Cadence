"""Optional real smoke train. Not collected by default.

``pytest`` addopts are ``-m 'not manual'``. Do not run this from the unit suite.
A passing run changes weights and writes a loadable ``weights.pth``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from excavation_lab.datasets.catalog import list_catalogs

pytestmark = pytest.mark.manual


def test_smoke_train_writes_a_loadable_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fine-tune Nano on a tiny synthetic set. Downloads official weights."""
    import time
    from datetime import UTC, datetime

    from PIL import Image
    from typer.testing import CliRunner

    from excavation_lab import jobs, storage
    from excavation_lab.cli import app
    from excavation_lab.models.train import FAKE_TRAINER_ENV, MANUAL_SMOKE_ENV
    from excavation_lab.schemas import JobStatus

    monkeypatch.delenv(FAKE_TRAINER_ENV, raising=False)
    monkeypatch.setenv(MANUAL_SMOKE_ENV, "1")
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    dataset = workspace / "dataset"
    for split, count in (("train", 2), ("valid", 1), ("test", 1)):
        directory = dataset / split
        directory.mkdir(parents=True)
        images = []
        for index in range(count):
            file_name = f"{split}_{index}.jpg"
            Image.new("RGB", (64, 48), color=(20, 40, 60)).save(directory / file_name)
            images.append({"id": index + 1, "file_name": file_name, "width": 64, "height": 48})
        (directory / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "images": images,
                    "annotations": [],
                    "categories": [
                        {"id": 1, "name": "excavator_body"},
                        {"id": 2, "name": "bucket"},
                        {"id": 3, "name": "haul_truck"},
                        {"id": 4, "name": "truck_bed"},
                    ],
                }
            ),
            encoding="utf-8",
        )
    dataset_id = "manualds"
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            """
            INSERT INTO dataset_versions (
                dataset_id, project_id, content_hash, path, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (dataset_id, "pilot", "cd" * 32, str(dataset), datetime.now(UTC).isoformat()),
        )
    runner = CliRunner()
    started = runner.invoke(
        app,
        [
            "train",
            "--project",
            "pilot",
            "--dataset",
            dataset_id,
            "--profile",
            "smoke",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert started.exit_code == 0, started.output
    job_id = json.loads(started.stdout)["data"]["job_id"]
    deadline = time.monotonic() + 45 * 60
    record = jobs.get_job(workspace, "pilot", job_id)
    while record.status not in {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}:
        if time.monotonic() >= deadline:
            jobs.request_cancel(workspace, "pilot", job_id)
            raise AssertionError("smoke train did not finish")
        time.sleep(1)
        record = jobs.get_job(workspace, "pilot", job_id)
    assert record.status == JobStatus.completed, record.error_code
    weights = jobs.run_directory(workspace, "pilot", job_id) / "weights.pth"
    assert weights.is_file()
    assert weights.stat().st_size > 0
    import torch

    torch.load(weights, map_location="cpu", weights_only=False)
