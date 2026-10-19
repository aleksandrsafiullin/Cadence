"""Unsupervised columns are zeroed before the label loss is reduced.

Tiny tensors. The fake trainer does not call ``model.train``.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from excavation_lab import jobs, storage
from excavation_lab.datasets.catalog import (
    add_class,
    list_catalogs,
    require_project_catalog,
)
from excavation_lab.errors import LabError
from excavation_lab.models import train as train_mod
from excavation_lab.models.supervision import (
    apply_logit_mask,
    catalog_class_names,
    load_supervision,
    observe_unsupervised,
    unsupervised_class_indexes,
    wrap_loss_labels,
)

_FOUR = ["excavator_body", "bucket", "haul_truck", "truck_bed"]
_NAMES = [*_FOUR, "shovel"]
_HASH = "ab" * 32
_QUERIES = 3
_BUCKET = 1
_SHOVEL = 4


def test_before_shovel_column_is_zero_and_after_shovel_stays_negative(tmp_path: Path) -> None:
    split = tmp_path / "train"
    _write_supervision_split(split)
    table = load_supervision(split / "class_supervision.json")
    assert table is not None
    assert _NAMES.index("bucket") == _BUCKET
    assert _NAMES.index("shovel") == _SHOVEL
    assert len(_NAMES) == 5

    before = unsupervised_class_indexes(table, _NAMES, image_id=1, file_name="before.jpg")
    by_id = unsupervised_class_indexes(table, _NAMES, image_id=1, file_name=None)
    by_file = unsupervised_class_indexes(table, _NAMES, image_id=None, file_name="before.jpg")
    assert before == by_id == by_file == (_SHOVEL,)
    pos = torch.full((_QUERIES, len(_NAMES) + 1), 0.4)
    neg = torch.full((_QUERIES, len(_NAMES) + 1), 0.7)
    apply_logit_mask(pos, neg, before)
    assert torch.count_nonzero(pos[:, _SHOVEL]) == 0
    assert torch.count_nonzero(neg[:, _SHOVEL]) == 0
    assert torch.equal(pos[:, _BUCKET], torch.full((_QUERIES,), 0.4))
    assert torch.equal(neg[:, _BUCKET], torch.full((_QUERIES,), 0.7))
    assert torch.equal(pos[:, len(_NAMES)], torch.full((_QUERIES,), 0.4))
    assert torch.equal(neg[:, len(_NAMES)], torch.full((_QUERIES,), 0.7))
    assert pos.shape[0] == _QUERIES

    after = unsupervised_class_indexes(table, _NAMES, image_id=2, file_name="after.jpg")
    assert after == ()
    after_pos = torch.full((_QUERIES, len(_NAMES) + 1), 0.4)
    after_neg = torch.full((_QUERIES, len(_NAMES) + 1), 0.7)
    apply_logit_mask(after_pos, after_neg, after)
    assert torch.equal(after_pos[:, _SHOVEL], torch.full((_QUERIES,), 0.4))
    assert torch.equal(after_neg[:, _SHOVEL], torch.full((_QUERIES,), 0.7))
    assert torch.equal(after_pos[:, _BUCKET], torch.full((_QUERIES,), 0.4))


def test_wrapper_zeros_shovel_inside_loss_labels_and_restores(tmp_path: Path) -> None:
    from rfdetr.models.criterion import SetCriterion

    split = tmp_path / "train"
    _write_supervision_split(split)
    (tmp_path / "valid").mkdir()
    original = SetCriterion.loss_labels
    criterion = SetCriterion.__new__(SetCriterion)
    criterion.ia_bce_loss = True
    criterion.focal_alpha = 0.25
    criterion.training = True
    logits = torch.zeros(1, _QUERIES, len(_NAMES) + 1, requires_grad=True)
    boxes = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1], [0.7, 0.3, 0.1, 0.1]]],
    )
    outputs = {"pred_logits": logits, "pred_boxes": boxes}
    targets = [
        {
            "labels": torch.tensor([_BUCKET]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "image_id": torch.tensor([1]),
        }
    ]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    num_boxes = torch.tensor(1.0)
    with wrap_loss_labels(tmp_path, _NAMES):
        assert SetCriterion.loss_labels is not original
        loss = SetCriterion.loss_labels(
            criterion,
            outputs,
            targets,
            indices,
            num_boxes,
            log=False,
        )
    assert SetCriterion.loss_labels is original
    loss["loss_ce"].backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[0, :, _SHOVEL]) == 0
    assert torch.count_nonzero(logits.grad[0, :, _BUCKET]) > 0
    assert logits.shape[1] == _QUERIES

    try:
        with wrap_loss_labels(tmp_path, _NAMES):
            raise RuntimeError("stop")
    except RuntimeError:
        pass
    assert SetCriterion.loss_labels is original


def test_fake_trainer_records_unsupervised_shovel_and_pretrain_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    add_class(workspace, "excavation", "shovel", "Лопата", "Видимая лопата.")
    names = catalog_class_names(require_project_catalog(workspace, "pilot"))
    assert names == _NAMES
    dataset = workspace / "dataset"
    _write_supervision_split(dataset / "train")
    _write_split(dataset / "valid", ["valid.jpg"], annotations=[])
    (dataset / "test").mkdir()
    (dataset / "test" / "sentinel.txt").write_text("leave-me", encoding="utf-8")
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            """
            INSERT INTO dataset_versions (
                dataset_id, project_id, content_hash, path, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("ds1", "pilot", _HASH, str(dataset), datetime.now(UTC).isoformat()),
        )
    cache = workspace / "cache" / "weights" / "rf-detr-nano.pth"
    _save_checkpoint(cache)
    monkeypatch.setenv(train_mod.FAKE_TRAINER_ENV, "fail")
    monkeypatch.setattr(jobs, "_spawn", _dummy_spawn)
    started = train_mod.enqueue_train(workspace, "pilot", "ds1", "smoke")
    assert train_mod.run_train_job(workspace, started.job_id) == 1
    run_dir = jobs.run_directory(workspace, "pilot", started.job_id)
    calls = json.loads((run_dir / "trainer_calls.json").read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert calls[0]["model_constructed"] is False
    kwargs = calls[0]["kwargs"]
    assert kwargs["run_test"] is False
    assert kwargs["grad_accum_steps"] == 1
    assert "num_queries" not in kwargs
    recorded = Path(kwargs["pretrain_weights"])
    assert recorded.name == "class_head.pth"
    assert recorded.parent == run_dir.resolve()
    seen = calls[0]["supervision_seen"]
    before = next(row for row in seen if row["file_name"] == "before.jpg")
    after = next(row for row in seen if row["file_name"] == "after.jpg")
    assert before["image_id"] == 1
    assert before["unsupervised"] == ["shovel"]
    assert after["image_id"] == 2
    assert after["unsupervised"] == []
    opened = " ".join(calls[0]["opened"])
    assert "sentinel.txt" not in opened
    assert observe_unsupervised(dataset, names)[0]["unsupervised"] == ["shovel"]


def test_background_index_is_not_masked() -> None:
    pos = torch.ones(2, 6)
    neg = torch.ones(2, 6)
    with pytest.raises(LabError) as caught:
        apply_logit_mask(pos, neg, [5])
    assert caught.value.code == "class_supervision"
    assert torch.equal(pos, torch.ones(2, 6))
    assert torch.equal(neg, torch.ones(2, 6))


def _write_supervision_split(directory: Path) -> None:
    before = {
        "file_name": "before.jpg",
        "image_id": 1,
        "negatives": ["excavator_body", "haul_truck", "truck_bed"],
        "positives": ["bucket"],
        "unsupervised": ["shovel"],
    }
    after = {
        "file_name": "after.jpg",
        "image_id": 2,
        "negatives": ["shovel", "excavator_body", "haul_truck", "truck_bed"],
        "positives": ["bucket"],
        "unsupervised": [],
    }
    images = {
        "1": before,
        "before.jpg": before,
        "2": after,
        "after.jpg": after,
    }
    annotations = [
        _box(1, 1, _BUCKET),
        _box(2, 2, _BUCKET),
    ]
    _write_split(
        directory,
        ["before.jpg", "after.jpg"],
        annotations=annotations,
        supervision={"schema_version": 1, "images": images},
    )


def _box(annotation_id: int, image_id: int, category_id: int) -> dict[str, object]:
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [1, 1, 2, 2],
        "iscrowd": 0,
    }


def _write_split(
    directory: Path,
    names: list[str],
    *,
    annotations: list[dict[str, object]],
    supervision: dict[str, object] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    images = []
    for index, name in enumerate(names, start=1):
        (directory / name).write_bytes(b"\xff\xd8\xff\xd9")
        images.append({"id": index, "file_name": name, "width": 8, "height": 8})
    (directory / "_annotations.coco.json").write_text(
        json.dumps({"annotations": annotations, "categories": [], "images": images}),
        encoding="utf-8",
    )
    if supervision is not None:
        (directory / "class_supervision.json").write_text(
            json.dumps(supervision),
            encoding="utf-8",
        )


def _save_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weight = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    bias = torch.arange(5, dtype=torch.float32)
    torch.save(
        {
            "model": {
                "class_embed.weight": weight,
                "class_embed.bias": bias,
                "bbox_embed.layers.0.weight": torch.zeros(2, 2),
            },
            "args": {"class_names": list(_FOUR), "num_classes": 4},
        },
        path,
    )


class _DummyProcess:
    def __init__(self) -> None:
        self.pid = os.getpid()

    def poll(self) -> None:
        return None


def _dummy_spawn(*_args: object, **_kwargs: object) -> _DummyProcess:
    return _DummyProcess()
