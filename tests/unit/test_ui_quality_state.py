"""Quality page: measured metrics, hash mismatch, train-only labeling."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
from starlette.testclient import TestClient

from excavation_lab import storage
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.splits import freeze_split, write_blocks
from excavation_lab.errors import DatasetHashMismatchError, LabError
from excavation_lab.models.bundle import CLASS_NAMES
from excavation_lab.models.quality import (
    CLASS_METRIC_KEYS,
    load_quality_report,
    metric_text,
)
from excavation_lab.models.quality import (
    compare as compare_runs,
)
from excavation_lab.ui import copy as ui_copy
from excavation_lab.ui import quality_page
from excavation_lab.web.app import create_app
from excavation_lab.web.preferences import write_project_file

_HASH_A = "ab" * 32
_HASH_B = "cd" * 32
_BLOCKS = (
    RoleBlock(start_s=0, end_s=60, role="train"),
    RoleBlock(start_s=72, end_s=92, role="validation"),
    RoleBlock(start_s=104, end_s=124, role="test"),
)
_CLASS_ROWS = {
    "excavator_body": {
        "ap50": 0.31,
        "ap50_95": 0.16,
        "precision": 0.33,
        "recall": 0.34,
        "recall_clear": 0.35,
        "recall_partial": 0.17,
        "gt_count": 4,
    },
    "bucket": {
        "ap50": 0.42,
        "ap50_95": 0.21,
        "precision": 0.41,
        "recall": 0.43,
        "recall_clear": 0.51,
        "recall_partial": 0.22,
        "gt_count": 3,
    },
    "haul_truck": {
        "ap50": 0.53,
        "ap50_95": 0.26,
        "precision": 0.54,
        "recall": 0.55,
        "recall_clear": 0.56,
        "recall_partial": 0.28,
        "gt_count": 5,
    },
    "truck_bed": {
        "ap50": 0.64,
        "ap50_95": 0.32,
        "precision": 0.65,
        "recall": 0.66,
        "recall_clear": 0.67,
        "recall_partial": 0.29,
        "gt_count": 6,
    },
}
_METRIC = re.compile(
    r'<div class="lab-metric">\s*<span>([^<]*)</span>\s*<b>([^<]*)</b>\s*</div>'
)


def test_page_does_not_import_a_detector_or_draw_boxes() -> None:
    source = Path(quality_page.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules: list[str] = []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module.split(".", 1)[0])
            names.extend(alias.name for alias in node.names)
    assert "rfdetr" not in modules
    assert "starlette" not in modules
    assert "DetectorAdapter" not in names
    assert "enqueue_train" not in names
    assert "enqueue_analyze" not in names
    assert "shell=True" not in source
    assert "st.image" not in source
    assert "st.altair_chart" not in source
    assert "тоннаж" not in source
    assert "т/ч" not in source
    assert "xyxy" not in source
    web = Path(quality_page.__file__).resolve().parents[1] / "web"
    routes = (web / "quality_routes.py").read_text(encoding="utf-8")
    template = (web / "templates" / "quality.html").read_text(encoding="utf-8")
    for text in (routes, template):
        assert "rfdetr" not in text
        assert "import streamlit" not in text
        assert "enqueue_train" not in text
        assert "enqueue_analyze" not in text
        assert "DetectorAdapter" not in text
        assert "altair" not in text.lower()
        assert "lab-chart" not in text
        assert "<svg" not in text.lower()
        assert "<canvas" not in text.lower()


def test_chart_omits_missing_scores_and_refuses_all_zeros() -> None:
    missing = {
        "status": "not_computed",
        "classes": {"bucket": {"ap50": ui_copy.NOT_CALCULATED}},
    }
    assert quality_page.ap50_chart_frame(missing) is None
    zeros = {
        "status": "computed",
        "classes": {"bucket": {"ap50": 0}, "haul_truck": {"ap50": 0.0}},
    }
    assert quality_page.ap50_chart_frame(zeros) is None
    partial = {
        "status": "computed",
        "classes": {
            "bucket": {"ap50": 0.42},
            "haul_truck": {"ap50": ui_copy.NOT_CALCULATED},
        },
    }
    frame = quality_page.ap50_chart_frame(partial)
    assert frame is not None
    assert [row["AP50"] for row in frame] == [0.42]
    assert quality_page.send_enabled("train") is True
    assert quality_page.send_enabled("validation") is False
    assert quality_page.send_enabled("test") is False
    assert quality_page.send_enabled("unknown") is False
    assert quality_page.send_enabled(None) is False


def test_measured_metrics_keep_idf1_unevaluated(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    _write_json(workspace, "run-measured", "metrics.json", _measured_document())
    report = load_quality_report(workspace, "pilot", "run-measured")
    with TestClient(create_app(workspace)) as client:
        response = client.get("/quality")
    assert response.status_code == 200
    text = response.text
    metrics = _metric_map(text)
    for name in CLASS_NAMES:
        for key in CLASS_METRIC_KEYS:
            label = f"{ui_copy.CLASS_LABELS[name]} — {ui_copy.METRIC_LABELS[key]}"
            assert metrics[label] == metric_text(report["classes"][name][key])
    assert metrics[ui_copy.METRIC_LABELS["reviewed_frame_count"]] == metric_text(
        report["reviewed_frame_count"]
    )
    assert metrics[ui_copy.METRIC_LABELS["gt_object_count"]] == metric_text(
        report["gt_object_count"]
    )
    for key in ("recall_clear", "recall_partial"):
        label = f"{ui_copy.QUALITY_VISIBILITY} — {ui_copy.METRIC_LABELS[key]}"
        assert metrics[label] == metric_text(report[key])
    assert report["idf1"] == "not_evaluated"
    assert report["hota"] == "not_evaluated"
    assert metrics[ui_copy.METRIC_LABELS["idf1"]] == ui_copy.NOT_CALCULATED
    assert metrics[ui_copy.METRIC_LABELS["hota"]] == ui_copy.NOT_CALCULATED
    assert ui_copy.NOT_EVALUATED_CAPTION in text
    assert ui_copy.NOT_CALCULATED in text
    assert "0.99" not in text
    assert "0.88" not in text
    assert ui_copy.TARGET_CAPTION in text
    assert text.index(ui_copy.TARGET_CAPTION) < text.index(metric_text(0.91))
    floor = f"{ui_copy.METRIC_LABELS['score_floor']}: {metric_text(report['score_floor'])}."
    threshold = (
        f"{ui_copy.METRIC_LABELS['operating_threshold']}: "
        f"{metric_text(report['operating_threshold'])}."
    )
    assert floor in text
    assert threshold in text
    assert "<img" not in text.lower()
    assert ui_copy.SEND_TO_LABELING not in text
    assert "lab-chart" not in text
    assert "<svg" not in text.lower()
    chart = quality_page.ap50_chart_frame(report)
    assert chart is not None
    assert 0.99 not in [row["AP50"] for row in chart]
    assert any(row["AP50"] != 0 for row in chart)


def test_compare_shows_hash_mismatch_without_a_delta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _project(tmp_path)
    low = {name: {"ap50": 0.25} for name in CLASS_NAMES}
    high = {name: {"ap50": 0.55} for name in CLASS_NAMES}
    _write_json(workspace, "run-a", "metrics.json", {"dataset_hash": _HASH_A, "classes": low})
    _write_json(workspace, "run-b", "metrics.json", {"dataset_hash": _HASH_B, "classes": high})
    orphan = storage.project_directory(workspace, "pilot") / "runs" / "orphan"
    orphan.mkdir()
    (orphan / "matches.json").write_text("{}\n", encoding="utf-8")
    calls: list[tuple[object, ...]] = []
    real = quality_page.compare

    def wrapped(*args: object, **kwargs: object) -> object:
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(quality_page, "compare", wrapped)
    with TestClient(create_app(workspace)) as client:
        first = client.get("/quality")
        assert first.status_code == 200
        assert _select_values(first.text, "quality_run") == ["run-a", "run-b"]
        assert _selected(first.text, "quality_run_a") == "run-a"
        assert _selected(first.text, "quality_run_b") == "run-b"
        assert calls == []
        posted = client.post(
            "/quality/compare",
            data={"quality_run_a": "run-a", "quality_run_b": "run-b"},
            follow_redirects=False,
        )
        assert posted.status_code == 303
        assert urlparse(posted.headers["location"]).path == "/quality"
        page = client.get("/quality")
    assert page.status_code == 200
    text = page.text
    expected = DatasetHashMismatchError().message
    assert expected == (
        "Сравнение прогонов невозможно: dataset_hash не совпадает. "
        "Разница метрик не выводится."
    )
    assert expected in text
    assert expected in _alerts(text)
    assert ui_copy.QUALITY_DELTA not in text
    assert "0.55" not in text
    assert "0.3" not in text
    assert not any(label.startswith(ui_copy.QUALITY_DELTA) for label, _value in _metrics(text))
    assert _select_values(text, "quality_run") == ["run-a", "run-b"]
    assert _selected(text, "quality_run_a") == "run-a"
    assert _selected(text, "quality_run_b") == "run-b"
    assert calls
    assert calls[0][1:] == ("pilot", "run-a", "run-b")
    assert len(calls) == 1
    assert "<img" not in text.lower()


def test_compare_same_hash_shows_a_delta(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    low = {name: {"ap50": 0.25} for name in CLASS_NAMES}
    high = {name: {"ap50": 0.55} for name in CLASS_NAMES}
    _write_json(workspace, "run-a", "metrics.json", {"dataset_hash": _HASH_A, "classes": low})
    _write_json(workspace, "run-b", "metrics.json", {"dataset_hash": _HASH_A, "classes": high})
    result = compare_runs(workspace, "pilot", "run-a", "run-b")
    book = ui_copy.catalog("ru")
    label = f"{book.quality_delta} — {book.class_labels['bucket']} — {book.metric_labels['ap50']}"
    expected = quality_page._shown_metric(result["delta"]["bucket"]["ap50"], book)
    with TestClient(create_app(workspace)) as client:
        posted = client.post(
            "/quality/compare",
            data={"quality_run_a": "run-a", "quality_run_b": "run-b"},
            follow_redirects=False,
        )
        assert posted.status_code == 303
        page = client.get("/quality")
    assert page.status_code == 200
    assert ui_copy.QUALITY_DELTA in page.text
    assert _metric_map(page.text)[label] == expected
    assert expected != book.not_calculated


def test_stale_eval_id_uses_a_real_run(tmp_path: Path) -> None:
    workspace = _project(tmp_path)
    for eval_id, digest in (("run-a", _HASH_A), ("run-b", _HASH_B)):
        _write_json(
            workspace,
            eval_id,
            "metrics.json",
            {"dataset_hash": digest, "classes": {"bucket": {"ap50": 0.25}}},
        )
    with TestClient(create_app(workspace)) as client:
        response = client.get("/quality", params={"quality_run": "gone"})
    assert response.status_code == 200
    assert 'name="quality_run"' in response.text
    assert ui_copy.QUALITY_RUN in response.text
    assert _selected(response.text, "quality_run") in {"run-a", "run-b"}
    assert "gone" not in _select_values(response.text, "quality_run")


def test_test_row_stays_disabled_and_train_row_queues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace, manifest = _error_project(tmp_path)
    before = manifest.read_bytes()
    report = load_quality_report(workspace, "pilot", "run-errors")
    pairs = [(row["frame_id"], row["split"], row["kind"]) for row in report["errors"]]
    assert pairs == [("vid:1", "train", "false_negative"), ("vid:3", "test", "false_positive")]
    train_key = quality_page.send_button_key(0, "false_negative", "vid:1")
    test_key = quality_page.send_button_key(1, "false_positive", "vid:3")
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    real = quality_page.send_to_labeling

    def wrapped(*args: object, **kwargs: object) -> object:
        calls.append((args, dict(kwargs)))
        return real(*args, **kwargs)

    monkeypatch.setattr(quality_page, "send_to_labeling", wrapped)
    with TestClient(create_app(workspace)) as client:
        page = client.get("/quality")
        assert page.status_code == 200
        assert _disabled(page.text, test_key)
        assert not _disabled(page.text, train_key)
        assert _batch_names(workspace) == []
        assert calls == []
        assert "Разбиение: train." in page.text
        assert "Разбиение: test." in page.text
        assert "xyxy" not in page.text
        forced = client.post(
            "/quality/label",
            data={"quality_run": "run-errors", "frame_id": "vid:3", test_key: "1"},
            follow_redirects=False,
        )
        assert forced.status_code == 303
        assert urlparse(forced.headers["location"]).path == "/quality"
        assert calls == []
        assert _batch_names(workspace) == []
        posted = client.post(
            "/quality/label",
            data={"quality_run": "run-errors", "frame_id": "vid:1", train_key: "1"},
            follow_redirects=False,
        )
        assert posted.status_code == 303
        shown = client.get("/quality")
    assert len(calls) == 1
    assert calls[0][0][1:] == ("pilot", "vid:1")
    assert calls[0][1]["eval_id"] == "run-errors"
    assert _batch_names(workspace) == ["quality__vid__1"]
    assert _disabled(shown.text, test_key)
    assert manifest.read_bytes() == before
    assert ui_copy.sent_to_labeling_message("vid:1") in shown.text


def test_labeling_error_is_the_lab_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace, _manifest = _error_project(tmp_path)
    message = ui_copy.not_train_frame_message("vid:1", "test")

    def boom(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise LabError("not_train_frame", message)

    monkeypatch.setattr(quality_page, "send_to_labeling", boom)
    train_key = quality_page.send_button_key(0, "false_negative", "vid:1")
    test_key = quality_page.send_button_key(1, "false_positive", "vid:3")
    with TestClient(create_app(workspace)) as client:
        page = client.get("/quality")
        assert _disabled(page.text, test_key)
        posted = client.post(
            "/quality/label",
            data={"quality_run": "run-errors", "frame_id": "vid:1", train_key: "1"},
            follow_redirects=False,
        )
        assert posted.status_code == 303
        shown = client.get("/quality")
    assert message in _alerts(shown.text)
    assert _batch_names(workspace) == []
    assert _disabled(shown.text, test_key)


def _measured_document() -> dict[str, object]:
    return {
        "dataset_hash": _HASH_A,
        "split": "validation",
        "reviewed_frame_count": 12,
        "gt_object_count": 7,
        "recall_clear": 0.61,
        "recall_partial": 0.27,
        "score_floor": 0.001,
        "operating_threshold": 0.15,
        "idf1": 0.99,
        "hota": 0.88,
        "classes": {name: dict(row) for name, row in _CLASS_ROWS.items()},
        "targets": {
            "haul_truck": {
                "precision": {"target": 0.91, "value": 0.54, "status": "target"},
                "recall": {"target": 0.92, "value": 0.55, "status": "target"},
            },
            "bucket": {
                "precision": {"target": 0.86, "value": 0.41, "status": "target"},
                "recall": {"target": 0.87, "value": 0.43, "status": "target"},
            },
        },
    }


def _error_project(tmp_path: Path) -> tuple[Path, Path]:
    workspace = _project(tmp_path)
    manifest = _freeze(workspace)
    _write_sample(workspace)
    _write_json(
        workspace,
        "run-errors",
        "metrics.json",
        {"dataset_hash": _HASH_A, "classes": {"bucket": {"ap50": 0.42}}},
    )
    _write_json(
        workspace,
        "run-errors",
        "matches.json",
        {
            "video_id": "vid",
            "false_positives": [
                {
                    "class_name": "bucket",
                    "source_frame_index": 3,
                    "source_time_s": 110.0,
                    "split": "train",
                    "xyxy": [1, 2, 3, 4],
                }
            ],
            "false_negatives": [
                {
                    "class_name": "haul_truck",
                    "source_frame_index": 1,
                    "source_time_s": 10.0,
                    "split": "test",
                    "xyxy": [0, 0, 8, 8],
                }
            ],
        },
    )
    return workspace, manifest


def _project(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project("pilot", workspace, catalog_id="excavation")
    write_project_file(workspace, "pilot")
    return workspace


def _write_json(workspace: Path, eval_id: str, name: str, document: object) -> None:
    directory = storage.project_directory(workspace, "pilot") / "runs" / eval_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(
        json.dumps(document, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _freeze(workspace: Path) -> Path:
    storage.insert_video(workspace, "pilot", "vid", name="pilot")
    manifest = write_blocks(workspace, "pilot", "vid", list(_BLOCKS))
    _frozen, path = freeze_split(workspace, "pilot", manifest)
    return path


def _write_sample(workspace: Path) -> None:
    """Sample split column lies. The frozen episode is the authority."""
    directory = storage.project_directory(workspace, "pilot") / "samples" / "vid"
    directory.mkdir(parents=True)
    document = {
        "rows": [
            {
                "video_id": "vid",
                "source_frame_index": 1,
                "source_time_s": 10.0,
                "episode_id": "train-ep",
                "split": "test",
            },
            {
                "video_id": "vid",
                "source_frame_index": 3,
                "source_time_s": 110.0,
                "episode_id": "test-ep",
                "split": "train",
            },
        ]
    }
    (directory / "sample_manifest.json").write_text(
        json.dumps(document) + "\n",
        encoding="utf-8",
    )


def _batch_names(workspace: Path) -> list[str]:
    root = storage.project_directory(workspace, "pilot") / "annotation_batches"
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir())


def _metrics(html: str) -> list[tuple[str, str]]:
    return [(label, value) for label, value in _METRIC.findall(html)]


def _metric_map(html: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for label, value in _metrics(html):
        assert label not in found, label
        found[label] = value
    return found


def _select_body(html: str, name: str) -> str:
    for attrs, body in re.findall(r"<select\b([^>]*)>(.*?)</select>", html, re.DOTALL):
        if re.search(rf'\bname="{re.escape(name)}"', attrs):
            return body
    raise AssertionError(name)


def _select_values(html: str, name: str) -> list[str]:
    found: list[str] = []
    for attrs, label in re.findall(r"<option\b([^>]*)>([^<]*)</option>", _select_body(html, name)):
        value = re.search(r'value="([^"]*)"', attrs)
        found.append(value.group(1) if value else label.strip())
    return found


def _selected(html: str, name: str) -> str:
    chosen: list[str] = []
    for attrs, label in re.findall(r"<option\b([^>]*)>([^<]*)</option>", _select_body(html, name)):
        if re.search(r"\sselected(?:\s|>|$)", attrs):
            value = re.search(r'value="([^"]*)"', attrs)
            chosen.append(value.group(1) if value else label.strip())
    assert len(chosen) == 1, (name, chosen)
    return chosen[0]


def _button_attrs(html: str, name: str) -> str:
    for attrs, _text in re.findall(r"<button\b([^>]*)>([^<]*)</button>", html):
        if f'name="{name}"' in attrs:
            return attrs
    raise AssertionError(name)


def _disabled(html: str, name: str) -> bool:
    return re.search(r"\bdisabled\b", _button_attrs(html, name)) is not None


def _alerts(html: str) -> list[str]:
    return re.findall(r'<p class="lab-alert" role="alert">([^<]*)</p>', html)
