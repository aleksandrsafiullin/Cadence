"""Workspace catalog files. The workspace is tmp_path. No pilot video."""

from __future__ import annotations

import inspect
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from excavation_lab import schemas
from excavation_lab.datasets import catalog
from excavation_lab.errors import LabError


def test_first_list_and_read_seed_excavation_outside_projects(tmp_path: Path) -> None:
    assert catalog.list_catalogs(tmp_path) == ["excavation"]
    path = tmp_path / "catalogs" / "excavation" / "catalog.yaml"
    assert path.is_file()
    assert path.relative_to(tmp_path).as_posix() == "catalogs/excavation/catalog.yaml"
    assert not (tmp_path / "projects").exists()

    labels = schemas.load_labels()
    loaded = catalog.read_catalog(tmp_path, "excavation")
    assert [(item.id, item.name, item.title, item.box) for item in loaded.classes] == [
        (item.id, item.name.value, item.title, item.box) for item in labels.classes
    ]
    stamps = [item.added_at for item in loaded.classes]
    assert len(set(stamps)) == 1
    assert stamps[0].tzinfo is not None
    assert stamps[0].utcoffset() == timedelta(0)
    assert isinstance(stamps[0], datetime)

    blob = path.read_bytes()
    again = catalog.read_catalog(tmp_path, "excavation")
    catalog.list_catalogs(tmp_path)
    assert path.read_bytes() == blob
    assert [(item.id, item.name, item.title, item.box, item.added_at) for item in again.classes] == [
        (item.id, item.name, item.title, item.box, item.added_at) for item in loaded.classes
    ]
    assert not (tmp_path / "projects").exists()


def test_catalog_id_token_and_reserved_excavation(tmp_path: Path) -> None:
    with pytest.raises(LabError) as caught:
        catalog.read_catalog(tmp_path, "Pit")
    assert caught.value.code == "invalid_catalog_id"
    assert list(tmp_path.iterdir()) == []

    catalog.list_catalogs(tmp_path)
    long_id = "a" * 64
    copied = catalog.copy_catalog(tmp_path, "excavation", long_id)
    assert copied.classes[0].name == "excavator_body"
    assert (tmp_path / "catalogs" / long_id / "catalog.yaml").is_file()

    for catalog_id in ("a" * 65, "1pit", "pit-a", "", "pit a", "ковш"):
        with pytest.raises(LabError) as caught:
            catalog.copy_catalog(tmp_path, "excavation", catalog_id)
        assert caught.value.code == "invalid_catalog_id"

    seed = (tmp_path / "catalogs" / "excavation" / "catalog.yaml").read_bytes()
    with pytest.raises(LabError) as caught:
        catalog.copy_catalog(tmp_path, long_id, "excavation")
    assert caught.value.code == "reserved_catalog_id"
    assert (tmp_path / "catalogs" / "excavation" / "catalog.yaml").read_bytes() == seed
    assert not (tmp_path / "projects").exists()


def test_copy_keeps_added_at_and_ignores_later_edits_of_the_copy(tmp_path: Path) -> None:
    original = catalog.read_catalog(tmp_path, "excavation")
    source = tmp_path / "catalogs" / "excavation" / "catalog.yaml"
    blob = source.read_bytes()
    time.sleep(0.05)
    copied = catalog.copy_catalog(tmp_path, "excavation", "pit")
    assert source.read_bytes() == blob
    assert [(item.id, item.name, item.title, item.box, item.added_at) for item in copied.classes] == [
        (item.id, item.name, item.title, item.box, item.added_at) for item in original.classes
    ]
    assert catalog.list_catalogs(tmp_path) == ["excavation", "pit"]

    catalog.add_class(tmp_path, "pit", "shovel", "Лопата", "Видимая лопата.")
    assert source.read_bytes() == blob
    pit = catalog.read_catalog(tmp_path, "pit")
    assert [item.name for item in pit.classes] == [
        "excavator_body",
        "bucket",
        "haul_truck",
        "truck_bed",
        "shovel",
    ]
    assert [item.name for item in catalog.read_catalog(tmp_path, "excavation").classes] == [
        "excavator_body",
        "bucket",
        "haul_truck",
        "truck_bed",
    ]
    assert not (tmp_path / "projects").exists()
    assert not (tmp_path / "projects" / "pit" / "catalog.yaml").exists()


def test_add_uses_the_smallest_unused_id_and_keeps_old_rows(tmp_path: Path) -> None:
    early = "2026-03-01T00:00:00+00:00"
    later = "2026-03-01T00:00:01+00:00"
    path = _write_catalog(
        tmp_path,
        "pit",
        [
            _row(0, "shovel", added_at=early),
            _row(2, "dozer", added_at=later),
        ],
    )
    blob = path.read_bytes()
    loaded = catalog.read_catalog(tmp_path, "pit")
    assert loaded.classes[0].added_at < loaded.classes[1].added_at
    assert loaded.classes[1].added_at.isoformat().startswith(loaded.classes[0].added_at.isoformat()[:18])
    assert path.read_bytes() == blob

    added = catalog.add_class(tmp_path, "pit", "loader", "Погрузчик", "Видимый погрузчик.")
    assert [(item.id, item.name) for item in added.classes] == [
        (0, "shovel"),
        (2, "dozer"),
        (1, "loader"),
    ]
    assert added.classes[0].added_at == loaded.classes[0].added_at
    assert added.classes[1].added_at == loaded.classes[1].added_at
    assert added.classes[2].added_at > added.classes[0].added_at
    assert added.classes[2].added_at.utcoffset() == timedelta(0)

    _write_catalog(
        tmp_path,
        "quarry",
        [_row(2, "shovel"), _row(5, "dozer")],
    )
    quarry = catalog.add_class(tmp_path, "quarry", "loader", "Погрузчик", "Видимый погрузчик.")
    assert [(item.id, item.name) for item in quarry.classes] == [
        (2, "shovel"),
        (5, "dozer"),
        (0, "loader"),
    ]


def test_refused_add_does_not_change_the_file(tmp_path: Path) -> None:
    catalog.list_catalogs(tmp_path)
    path = tmp_path / "catalogs" / "excavation" / "catalog.yaml"
    catalog.add_class(tmp_path, "excavation", "shovel", "Лопата", "Видимая лопата.")
    blob = path.read_bytes()
    with pytest.raises(LabError) as caught:
        catalog.add_class(tmp_path, "excavation", "shovel", "Другая", "Другая рамка.")
    assert caught.value.code == "duplicate_class_name"
    assert path.read_bytes() == blob

    with pytest.raises(LabError) as caught:
        catalog.add_class(tmp_path, "excavation", "ковш", "Ковш", "Рамка.")
    assert caught.value.code == "invalid_class_name"
    assert path.read_bytes() == blob

    with pytest.raises(LabError) as caught:
        catalog.add_class(tmp_path, "excavation", "loader", "", "Рамка.")
    assert caught.value.code == "catalog_invalid"
    assert path.read_bytes() == blob
    assert not list(path.parent.glob("*.partial"))

    for index in range(11):
        catalog.add_class(
            tmp_path,
            "excavation",
            f"c{index}",
            f"Класс {index}",
            f"Рамка {index}.",
        )
    blob = path.read_bytes()
    with pytest.raises(LabError) as caught:
        catalog.add_class(tmp_path, "excavation", "c11", "Лишний", "Рамка.")
    assert caught.value.code == "catalog_full"
    assert path.read_bytes() == blob
    loaded = catalog.read_catalog(tmp_path, "excavation")
    assert len(loaded.classes) == 16
    assert [item.id for item in loaded.classes[:5]] == [0, 1, 2, 3, 4]


def test_duplicate_ids_are_not_rewritten(tmp_path: Path) -> None:
    path = _write_catalog(
        tmp_path,
        "pit",
        [_row(0, "shovel"), _row(0, "dozer")],
    )
    blob = path.read_bytes()
    with pytest.raises(LabError) as caught:
        catalog.read_catalog(tmp_path, "pit")
    assert "duplicate class id" in caught.value.message
    assert path.read_bytes() == blob
    with pytest.raises(LabError):
        catalog.add_class(tmp_path, "pit", "loader", "Погрузчик", "Видимый погрузчик.")
    assert path.read_bytes() == blob


def test_edit_changes_only_title_and_box(tmp_path: Path) -> None:
    loaded = catalog.read_catalog(tmp_path, "excavation")
    path = tmp_path / "catalogs" / "excavation" / "catalog.yaml"
    stamp = loaded.classes[1].added_at
    box = loaded.classes[1].box
    edited = catalog.edit_class(
        tmp_path,
        "excavation",
        "bucket",
        title="Ковш экскаватора",
    )
    row = edited.classes[1]
    assert row.title == "Ковш экскаватора"
    assert row.box == box
    assert row.id == 1
    assert row.name == "bucket"
    assert row.added_at == stamp
    assert [(item.id, item.name, item.added_at) for item in edited.classes] == [
        (item.id, item.name, item.added_at) for item in loaded.classes
    ]

    both = catalog.edit_class(
        tmp_path,
        "excavation",
        "bucket",
        title="Ковш",
        box="Только видимый металл.",
    )
    assert both.classes[1].title == "Ковш"
    assert both.classes[1].box == "Только видимый металл."
    assert both.classes[1].added_at == stamp
    assert both.classes[1].id == 1
    assert both.classes[1].name == "bucket"

    blob = path.read_bytes()
    with pytest.raises(LabError) as caught:
        catalog.edit_class(tmp_path, "excavation", "bucket", title="")
    assert caught.value.code == "catalog_invalid"
    assert path.read_bytes() == blob
    with pytest.raises(LabError) as caught:
        catalog.edit_class(tmp_path, "excavation", "bucket")
    assert caught.value.code == "invalid_edit"
    assert path.read_bytes() == blob


def test_rename_and_delete_refuse_in_use_without_a_flag(tmp_path: Path) -> None:
    for function in (catalog.rename_class, catalog.delete_class, catalog.add_class, catalog.edit_class):
        assert "in_use" not in inspect.signature(function).parameters

    catalog.list_catalogs(tmp_path)
    path = tmp_path / "catalogs" / "excavation" / "catalog.yaml"
    _write_labels(tmp_path, "demo", "bucket")
    blob = path.read_bytes()
    with pytest.raises(LabError) as caught:
        catalog.rename_class(tmp_path, "excavation", "bucket", "cab")
    assert caught.value.code == "class_in_use"
    assert path.read_bytes() == blob
    with pytest.raises(LabError) as caught:
        catalog.delete_class(tmp_path, "excavation", "bucket")
    assert caught.value.code == "class_in_use"
    assert path.read_bytes() == blob

    _write_mapping(tmp_path, "model-1", "haul_truck")
    blob = path.read_bytes()
    with pytest.raises(LabError) as caught:
        catalog.rename_class(tmp_path, "excavation", "haul_truck", "truck")
    assert caught.value.code == "class_in_use"
    assert path.read_bytes() == blob

    catalog.add_class(tmp_path, "excavation", "shovel", "Лопата", "Видимая лопата.")
    blob = path.read_bytes()
    with pytest.raises(LabError) as caught:
        catalog.delete_class(tmp_path, "excavation", "shovel")
    assert caught.value.code == "class_delete_refused"
    assert path.read_bytes() == blob
    assert any(
        item.name == "shovel" for item in catalog.read_catalog(tmp_path, "excavation").classes
    )

    renamed = catalog.rename_class(tmp_path, "excavation", "shovel", "spade")
    row = next(item for item in renamed.classes if item.name == "spade")
    assert row.id == 4
    assert row.title == "Лопата"
    assert row.box == "Видимая лопата."
    assert "shovel" not in {item.name for item in renamed.classes}


def test_other_catalog_boxes_do_not_block_and_missing_column_does(tmp_path: Path) -> None:
    before = catalog.read_catalog(tmp_path, "excavation")
    path = tmp_path / "catalogs" / "excavation" / "catalog.yaml"
    _write_labels(tmp_path, "other", "bucket")
    _set_catalog_id(tmp_path / "projects" / "other", "pit")
    renamed = catalog.rename_class(tmp_path, "excavation", "bucket", "cab")
    assert renamed.classes[1].name == "cab"
    assert renamed.classes[1].id == 1
    assert renamed.classes[1].title == before.classes[1].title
    assert renamed.classes[1].added_at == before.classes[1].added_at

    _write_labels(tmp_path, "mine", "truck_bed")
    blob = path.read_bytes()
    with pytest.raises(LabError) as caught:
        catalog.rename_class(tmp_path, "excavation", "truck_bed", "bed")
    assert caught.value.code == "class_in_use"
    assert path.read_bytes() == blob


def test_create_project_catalog_id_skips_a_different_catalog(tmp_path: Path) -> None:
    from excavation_lab.storage import create_project

    catalog.list_catalogs(tmp_path)
    catalog.copy_catalog(tmp_path, "excavation", "pit")
    create_project("mine", tmp_path, catalog_id="excavation")
    create_project("other", tmp_path, catalog_id="pit")
    _write_labels(tmp_path, "other", "bucket")
    renamed = catalog.rename_class(tmp_path, "excavation", "bucket", "scoop")
    assert any(item.name == "scoop" for item in renamed.classes)
    _write_labels(tmp_path, "mine", "haul_truck")
    with pytest.raises(LabError) as caught:
        catalog.rename_class(tmp_path, "excavation", "haul_truck", "hauler")
    assert caught.value.code == "class_in_use"


def test_add_and_edit_do_not_use_the_in_use_scan(tmp_path: Path) -> None:
    loaded = catalog.read_catalog(tmp_path, "excavation")
    stamp = loaded.classes[1].added_at
    _write_labels(tmp_path, "demo", "bucket", "shovel")
    _write_mapping(tmp_path, "model-1", "shovel")
    edited = catalog.edit_class(tmp_path, "excavation", "bucket", title="Ковш в работе")
    assert edited.classes[1].title == "Ковш в работе"
    assert edited.classes[1].added_at == stamp
    assert edited.classes[1].name == "bucket"
    added = catalog.add_class(tmp_path, "excavation", "shovel", "Лопата", "Видимая лопата.")
    assert added.classes[-1].name == "shovel"
    assert added.classes[-1].id == 4


def test_catalog_module_does_not_create_projects(tmp_path: Path) -> None:
    source = Path(catalog.__file__).read_text(encoding="utf-8")
    assert "create_project" not in source
    catalog.copy_catalog(tmp_path, "excavation", "pit")
    assert not (tmp_path / "projects").exists()


def _write_catalog(root: Path, catalog_id: str, classes: list[dict[str, object]]) -> Path:
    raw = schemas.read_yaml_mapping(schemas.labels_path())
    raw["classes"] = classes
    path = root / "catalogs" / catalog_id / "catalog.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _row(
    class_id: int,
    name: str,
    *,
    title: str = "Класс",
    box: str = "Видимая часть.",
    added_at: str = "2026-01-01T00:00:00+00:00",
) -> dict[str, object]:
    return {
        "id": class_id,
        "name": name,
        "title": title,
        "box": box,
        "added_at": added_at,
    }


def _write_labels(root: Path, project: str, *class_names: str) -> None:
    batch = root / "projects" / project / "annotation_batches" / "batch-1"
    batch.mkdir(parents=True, exist_ok=True)
    document = {
        "frames": [
            {
                "record": {
                    "boxes": [{"class_name": class_name} for class_name in class_names],
                }
            }
        ]
    }
    (batch / "imported_labels.json").write_text(
        json.dumps(document),
        encoding="utf-8",
    )


def _write_mapping(root: Path, model_id: str, class_name: str) -> None:
    model = root / "weights" / model_id
    model.mkdir(parents=True)
    document = {
        "schema_version": 1,
        "by_name": {
            class_name: {"internal_id": 1, "coco_category_id": 1},
        },
    }
    (model / "class_mapping.json").write_text(json.dumps(document), encoding="utf-8")


def _set_catalog_id(project_dir: Path, catalog_id: str) -> None:
    connection = sqlite3.connect(project_dir / "project.sqlite")
    connection.execute("CREATE TABLE projects (project_id TEXT, catalog_id TEXT)")
    connection.execute(
        "INSERT INTO projects (project_id, catalog_id) VALUES ('p', ?)",
        (catalog_id,),
    )
    connection.commit()
    connection.close()
