"""Model bundle files, checksums, and the Russian card. No training."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from excavation_lab import storage
from excavation_lab.datasets.catalog import catalog_from_seed, list_catalogs
from excavation_lab.datasets.coco_import import build_class_mapping
from excavation_lab.datasets.splits import SAME_SCENE_MESSAGE
from excavation_lab.errors import LabError
from excavation_lab.models import bundle as bundle_mod
from excavation_lab.models.bundle import (
    BUNDLE_FILENAMES,
    CLASS_NAMES,
    NOT_CALCULATED,
    OTHER_CAMERA_ABSENT,
    RESUME_ABSENT,
    BundleChecksumError,
    ModelBundleSpec,
    publish_train_bundle,
    verify_bundle_checksums,
    write_model_bundle,
)
from excavation_lab.models.train_config import (
    resolve_train_profile,
    write_resolved_train_config,
)

_REPO = Path(__file__).resolve().parents[2]
_HASH = "ab" * 32
_CLASS_ORDER = ["excavator_body", "bucket", "haul_truck", "truck_bed"]


def _spec(**overrides: object) -> ModelBundleSpec:
    payload: dict[str, object] = {
        "variant": "nano",
        "resolution": 384,
        "dataset_hash": _HASH,
        "library_versions": {
            "rfdetr": "1.11.2",
            "torch": "2.14.1",
            "torchvision": "0.29.1",
        },
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
        "environment_report": {"schema_version": 1, "inference_device": "cpu"},
        "validation_metrics": {
            "schema_version": 1,
            "selection_metric": "val/mAP_50_95",
            "box_map_50_95": 0.5,
            "bucket_ap": 0.25,
            "selection_unstable": True,
            "validation_images": 1,
        },
        "weights": b"not-a-real-checkpoint",
        "view_id": "face-shovel",
        "equipment_family": "hydraulic_shovel",
        "split_independence": "episode_split",
        "excluded_conditions": "Ночная съёмка в датасет не входила.",
        "other_camera_tested": False,
    }
    payload.update(overrides)
    return ModelBundleSpec(**payload)  # type: ignore[arg-type]


def _write(tmp_path: Path, name: str, **overrides: object) -> Path:
    return write_model_bundle(tmp_path / name, _spec(**overrides))


def _card(bundle: Path) -> str:
    return (bundle / "model_card.md").read_text(encoding="utf-8")


def _json(bundle: Path, name: str) -> dict[str, object]:
    document = json.loads((bundle / name).read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _assert_english_keys(node: object) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            assert isinstance(key, str)
            assert key.isascii()
            _assert_english_keys(value)
    elif isinstance(node, list):
        for item in node:
            _assert_english_keys(item)


def test_fake_weight_blob_checks_checksum_and_class_order(tmp_path: Path) -> None:
    mapping = build_class_mapping(catalog_from_seed())
    mapping["class_names"] = list(reversed(CLASS_NAMES))
    bundle = _write(tmp_path, "m1", class_mapping=mapping)
    for name in (*BUNDLE_FILENAMES, "checksums.json"):
        assert (bundle / name).is_file()
    assert not (bundle / "training_checkpoint.ckpt").exists()

    config = _json(bundle, "model_config.json")
    assert config["origin"] == "custom"
    assert config["variant"] == "nano"
    assert config["resolution"] == 384
    assert config["dataset_hash"] == _HASH
    assert config["class_names"] == _CLASS_ORDER
    assert config["class_names"].index("bucket") == 1  # type: ignore[union-attr]
    assert config["resume_checkpoint"] is False
    versions = config["library_versions"]
    assert isinstance(versions, dict)
    assert versions["rfdetr"] == "1.11.2"
    assert set(versions) == {"rfdetr", "torch", "torchvision"}

    stored_mapping = _json(bundle, "class_mapping.json")
    assert stored_mapping["class_names"] == _CLASS_ORDER
    by_name = stored_mapping["by_name"]
    assert isinstance(by_name, dict)
    assert by_name["excavator_body"]["internal_id"] == 0  # type: ignore[index]
    assert by_name["bucket"]["internal_id"] == 1  # type: ignore[index]
    assert by_name["haul_truck"]["internal_id"] == 2  # type: ignore[index]
    assert by_name["truck_bed"]["internal_id"] == 3  # type: ignore[index]

    recorded = _json(bundle, "checksums.json")
    assert recorded["algorithm"] == "sha256"
    files = recorded["files"]
    assert isinstance(files, dict)
    on_disk = {path.name for path in bundle.iterdir() if path.is_file()}
    assert "checksums.json" not in files
    assert set(files) == on_disk - {"checksums.json"}
    assert files["weights.pth"] == hashlib.sha256(b"not-a-real-checkpoint").hexdigest()
    verify_bundle_checksums(bundle)

    card = _card(bundle)
    assert "Область применения" in card
    assert "view_id: face-shovel" in card
    assert "equipment_family: hydraulic_shovel" in card
    assert "episode_split" in card
    assert "same_scene" in card
    assert "Ночная съёмка в датасет не входила." in card
    assert "- box mAP50:95: 0.5" in card
    assert "- AP ковша: 0.25" in card
    assert "validation_metrics.json" in card
    assert OTHER_CAMERA_ABSENT in card
    assert RESUME_ABSENT in card
    assert "- История метрик: не рассчитано" in card
    assert (bundle / "metrics_history.csv").read_text(encoding="utf-8") == "epoch\n"
    assert any("а" <= char.lower() <= "я" for char in card)
    for name in ("model_config.json", "class_mapping.json", "checksums.json"):
        _assert_english_keys(_json(bundle, name))


def test_flipped_byte_and_extra_file_fail_verification(tmp_path: Path) -> None:
    flipped = _write(tmp_path, "flip")
    verify_bundle_checksums(flipped)
    target = flipped / "weights.pth"
    blob = bytearray(target.read_bytes())
    blob[0] ^= 0x01
    target.write_bytes(blob)
    with pytest.raises(BundleChecksumError) as caught:
        verify_bundle_checksums(flipped)
    assert caught.value.code == "checksum_mismatch"
    assert "weights.pth" in caught.value.message

    extra = _write(tmp_path, "extra")
    (extra / "note.txt").write_text("x", encoding="utf-8")
    with pytest.raises(BundleChecksumError) as extra_caught:
        verify_bundle_checksums(extra)
    assert extra_caught.value.code == "checksum_mismatch"
    assert "note.txt" in extra_caught.value.message


def test_empty_metrics_stay_unspecified_and_a_log_is_copied(tmp_path: Path) -> None:
    empty = _write(
        tmp_path,
        "empty",
        validation_metrics={
            "schema_version": 1,
            "box_map_50_95": None,
            "bucket_ap": None,
        },
    )
    card = _card(empty)
    assert "- box mAP50:95: не рассчитано" in card
    assert "- AP ковша: не рассчитано" in card
    assert "- История метрик: не рассчитано" in card
    metrics = _json(empty, "validation_metrics.json")
    assert metrics["box_map_50_95"] is None
    assert metrics["bucket_ap"] is None
    raw = (empty / "validation_metrics.json").read_text(encoding="utf-8")
    assert '"box_map_50_95": null' in raw
    assert '"box_map_50_95": 0' not in raw
    assert '"bucket_ap": 0' not in raw

    log = "epoch,val/mAP_50_95\n1,0.5\n"
    copied = _write(
        tmp_path,
        "logged",
        metrics_log=log.encode("utf-8"),
        validation_metrics={
            "schema_version": 1,
            "box_map_50_95": None,
            "bucket_ap": 0.25,
        },
    )
    assert (copied / "metrics_history.csv").read_text(encoding="utf-8") == log
    logged_card = _card(copied)
    assert "- История метрик: в журнале есть записи (1)" in logged_card
    assert "- box mAP50:95: не рассчитано" in logged_card
    assert "- AP ковша: 0.25" in logged_card
    assert "- box mAP50:95: 0.5" not in logged_card

    header_only = _write(tmp_path, "header", metrics_log=b"epoch,val/mAP_50_95\n")
    assert (header_only / "metrics_history.csv").read_text(encoding="utf-8") == (
        "epoch,val/mAP_50_95\n"
    )
    assert "- История метрик: не рассчитано" in _card(header_only)


def test_resume_file_is_copied_only_from_bytes(tmp_path: Path) -> None:
    absent = _write(tmp_path, "absent")
    assert not (absent / "training_checkpoint.ckpt").exists()
    assert _json(absent, "model_config.json")["resume_checkpoint"] is False
    assert RESUME_ABSENT in _card(absent)
    assert "training_checkpoint.ckpt" not in _json(absent, "checksums.json")["files"]  # type: ignore[index]

    present = _write(tmp_path, "present", training_checkpoint=b"library-ckpt")
    assert (present / "training_checkpoint.ckpt").read_bytes() == b"library-ckpt"
    assert _json(present, "model_config.json")["resume_checkpoint"] is True
    files = _json(present, "checksums.json")["files"]
    assert isinstance(files, dict)
    assert "training_checkpoint.ckpt" in files
    card = _card(present)
    assert "Загрузка этого файла здесь не проверяется." in card
    assert RESUME_ABSENT not in card
    verify_bundle_checksums(present)

    with pytest.raises(LabError) as caught:
        _write(tmp_path, "blank", training_checkpoint=b"")
    assert caught.value.code == "resume_checkpoint"
    assert not (tmp_path / "blank").exists()


def test_swapped_class_ids_are_rejected(tmp_path: Path) -> None:
    mapping = build_class_mapping(catalog_from_seed())
    mapping["by_name"]["bucket"]["internal_id"] = 0
    mapping["by_name"]["excavator_body"]["internal_id"] = 1
    with pytest.raises(LabError) as caught:
        _write(tmp_path, "swapped", class_mapping=mapping)
    assert caught.value.code == "class_order"
    assert not (tmp_path / "swapped").exists()


def test_same_scene_and_other_camera_lines(tmp_path: Path) -> None:
    scene = _write(tmp_path, "scene", split_independence="same_scene", excluded_conditions=None)
    card = _card(scene)
    assert "Независимость разбиения: same_scene." in card
    assert SAME_SCENE_MESSAGE in card
    assert "episode_split" in card
    assert "Исключённые условия" in card
    assert NOT_CALCULATED in card

    checked = _write(tmp_path, "camera", other_camera_tested=True)
    checked_card = _card(checked)
    assert "Другая камера проверялась" in checked_card
    assert OTHER_CAMERA_ABSENT not in checked_card


def test_overfit_is_not_written_as_a_quality_card(tmp_path: Path) -> None:
    with pytest.raises(LabError) as claimed:
        _write(
            tmp_path,
            "claimed",
            resolved_train_config={
                "profile": "overfit",
                "diagnostic": True,
                "metric_role": "diagnostic_overfit",
                "quality_claim": True,
            },
            validation_metrics={
                "box_map_50_95": 0.42,
                "bucket_ap": 0.17,
                "metric_role": "diagnostic_overfit",
                "diagnostic": True,
                "quality_claim": True,
                "test/mAP_50_95": 0.99,
            },
        )
    assert claimed.value.code == "quality_claim"
    assert not (tmp_path / "claimed").exists()

    bundle = _write(
        tmp_path,
        "diagnostic",
        resolved_train_config={
            "profile": "overfit",
            "diagnostic": True,
            "metric_role": "diagnostic_overfit",
            "run_test": False,
        },
        validation_metrics={
            "box_map_50_95": 0.42,
            "bucket_ap": 0.17,
            "metric_role": "diagnostic_overfit",
            "diagnostic": True,
            "selection_unstable": True,
            "validation_images": 20,
        },
    )
    card = _card(bundle)
    stored = _json(bundle, "validation_metrics.json")
    assert stored["metric_role"] == "diagnostic_overfit"
    assert stored.get("quality_claim") is not True
    assert "test/mAP_50_95" not in stored
    assert "0.42" in card
    assert "0.17" in card
    assert "0.99" not in card
    assert "diagnostic_overfit" in card
    assert "Оценка качества на test в эту карточку не входит." in card
    assert "quality_claim" not in card
    resolved = _json(bundle, "resolved_train_config.json")
    assert resolved["metric_role"] == "diagnostic_overfit"
    assert resolved.get("quality_claim") is not True


def test_model_card_template_covers_the_required_sections() -> None:
    template = (_REPO / "docs" / "MODEL_CARD_TEMPLATE.md").read_text(encoding="utf-8")
    for needle in (
        "Область применения",
        "view_id",
        "equipment_family",
        "episode_split",
        "same_scene",
        "Исключённые условия",
        "validation_metrics.json",
        "не рассчитано",
        "diagnostic_overfit",
        OTHER_CAMERA_ABSENT,
        "не является успешной точкой возобновления",
    ):
        assert needle in template


def test_publish_writes_workspace_weights_and_skips_pth_as_resume(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    dataset = workspace / "dataset"
    dataset.mkdir()
    (dataset / "split_manifest.json").write_text(
        json.dumps({"schema_version": 1, "independence": "same_scene"}) + "\n",
        encoding="utf-8",
    )
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            """
            INSERT INTO dataset_versions (
                dataset_id, project_id, content_hash, path, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("ds1", "pilot", _HASH, str(dataset), datetime.now(UTC).isoformat()),
        )
        connection.execute(
            """
            INSERT INTO views (
                project_id, view_id, site_id, camera_id, equipment_family,
                measured, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "pilot",
                "face-shovel",
                None,
                None,
                "hydraulic_shovel",
                0,
                datetime.now(UTC).isoformat(),
            ),
        )
    run_dir = workspace / "run"
    run_dir.mkdir()
    resolved = resolve_train_profile("smoke")
    write_resolved_train_config(run_dir / "resolved_train_config.json", resolved)
    (run_dir / "validation_metrics.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "box_map_50_95": 0.5,
                "bucket_ap": 0.25,
                "selection_unstable": True,
                "validation_images": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "weights.pth").write_bytes(b"inference-bytes")
    (run_dir / "checkpoint_best_total.pth").write_bytes(b"stripped-pth")
    (run_dir / "last.ckpt").write_bytes(b"library-ckpt")
    (run_dir / "checkpoint_2.ckpt").write_bytes(b"epoch-2")

    bundle = publish_train_bundle(
        workspace,
        "pilot",
        "abc123",
        dataset_id="ds1",
        dataset_dir=dataset,
        run_dir=run_dir,
        profile="smoke",
        resolved=resolved,
    )
    assert bundle == (workspace / "weights" / "smoke-abc123").resolve()
    assert (bundle / "class_mapping.json").is_file()
    assert not (workspace / "projects" / "pilot" / "models" / "smoke-abc123").exists()
    assert (bundle / "weights.pth").read_bytes() == b"inference-bytes"
    assert (bundle / "training_checkpoint.ckpt").read_bytes() == b"library-ckpt"
    assert _json(bundle, "model_config.json")["class_names"] == _CLASS_ORDER
    assert _json(bundle, "model_config.json")["dataset_hash"] == _HASH
    assert _json(bundle, "model_config.json")["resume_checkpoint"] is True
    verify_bundle_checksums(bundle)
    card = _card(bundle)
    assert "view_id: face-shovel" in card
    assert "equipment_family: hydraulic_shovel" in card
    assert "Независимость разбиения: same_scene." in card
    assert SAME_SCENE_MESSAGE in card
    assert OTHER_CAMERA_ABSENT in card
    assert "- box mAP50:95: 0.5" in card
    assert "Загрузка этого файла здесь не проверяется." in card
    with storage.connect(storage.project_sqlite(workspace, "pilot")) as connection:
        row = connection.execute(
            "SELECT variant, path, dataset_id FROM model_versions WHERE model_id = ?",
            ("smoke-abc123",),
        ).fetchone()
    assert row["variant"] == "nano"
    assert row["dataset_id"] == "ds1"
    assert row["path"] == str(bundle.resolve())


def test_publish_refuses_to_copy_official_coco_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    official = tmp_path / "cache" / "weights" / "rf-detr-nano.pth"
    official.parent.mkdir(parents=True)
    official.write_bytes(b"coco-baseline")
    monkeypatch.setattr(bundle_mod, "official_nano_weights_path", lambda workspace=None: official)
    monkeypatch.setattr(bundle_mod, "weights_cache_dir", lambda workspace=None: official.parent)
    run_dir = workspace / "run"
    run_dir.mkdir()
    (run_dir / "weights.pth").symlink_to(official)
    with pytest.raises(LabError) as caught:
        publish_train_bundle(
            workspace,
            "pilot",
            "abc123",
            dataset_id="ds1",
            dataset_dir=run_dir,
            run_dir=run_dir,
            profile="smoke",
            resolved=resolve_train_profile("smoke"),
        )
    assert caught.value.code == "baseline_weights"
    assert official.read_bytes() == b"coco-baseline"
    assert list((workspace / "projects" / "pilot" / "models").iterdir()) == []
    assert not (workspace / "weights").exists()


def test_synthetic_wiring_card_drops_numeric_map(tmp_path: Path) -> None:
    from excavation_lab.models.train_config import METRIC_ROLE_SYNTHETIC_WIRING
    from excavation_lab.ui import copy as ui_copy

    bundle = _write(
        tmp_path,
        "wire",
        validation_metrics={
            "schema_version": 1,
            "selection_metric": "val/mAP_50_95",
            "box_map_50_95": 0.5,
            "bucket_ap": 0.25,
            "selection_unstable": True,
            "validation_images": 1,
            "synthetic": True,
            "metric_role": METRIC_ROLE_SYNTHETIC_WIRING,
        },
    )
    card = _card(bundle)
    russian = ui_copy.catalog("ru")
    assert russian.synthetic_card_use in card
    assert russian.synthetic_card_role in card
    assert NOT_CALCULATED in card
    assert "0.5" not in card
    assert "0.25" not in card
    assert "Дообученный детектор" not in card
    config = _json(bundle, "model_config.json")
    assert config["origin"] == "custom"
    assert config["synthetic"] is True
    assert config["metric_role"] == METRIC_ROLE_SYNTHETIC_WIRING
    stored = _json(bundle, "validation_metrics.json")
    assert stored["box_map_50_95"] is None
    assert stored["bucket_ap"] is None
    assert stored["synthetic"] is True
    plain = _json(_write(tmp_path, "measured"), "model_config.json")
    assert "synthetic" not in plain
    assert ui_copy.catalog("en").synthetic_card_use != russian.synthetic_card_use
