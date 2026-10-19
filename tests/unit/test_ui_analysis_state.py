"""Analysis page: one MP4, one diagnostic frame, no invented metrics."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from PIL import Image
from starlette.testclient import TestClient

from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.errors import LabError
from excavation_lab.hardware import official_nano_weights_path
from excavation_lab.jobs import reopen, request_cancel, start_job
from excavation_lab.media.probe import add_video
from excavation_lab.schemas import JobStatus, load_pilot
from excavation_lab.storage import (
    connect,
    create_project,
    insert_analysis_run,
    list_videos,
    project_directory,
    project_sqlite,
)
from excavation_lab.tracking.analyze import FAKE_DETECTOR_ENV, REQUEST_NAME
from excavation_lab.ui import analysis_page, video_page
from excavation_lab.ui import copy as ui_copy
from excavation_lab.ui.video_page import job_status_line
from excavation_lab.web import analysis_routes
from excavation_lab.web.app import create_app

_SYNTHETIC_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic.py"
_SYNTHETIC_SPEC = importlib.util.spec_from_file_location(
    "excavation_lab_synthetic_mp4_analysis_page",
    _SYNTHETIC_PATH,
)
assert _SYNTHETIC_SPEC is not None and _SYNTHETIC_SPEC.loader is not None
_SYNTHETIC = importlib.util.module_from_spec(_SYNTHETIC_SPEC)
_SYNTHETIC_SPEC.loader.exec_module(_SYNTHETIC)
write_synthetic_mp4 = _SYNTHETIC.write_synthetic_mp4

_PAGE_SOURCE = Path(analysis_page.__file__).read_text(encoding="utf-8")
_WEB_SOURCE = Path(analysis_routes.__file__).read_text(encoding="utf-8")
_TEMPLATE = (
    Path(analysis_routes.__file__).resolve().parent / "templates" / "analysis.html"
).read_text(encoding="utf-8")
_SCRIPT = (
    Path(analysis_routes.__file__).resolve().parent / "static" / "lab.js"
).read_text(encoding="utf-8")
_ZERO_TOKEN = re.compile(r"(?<![\d.])0(?:[.,]0+)?%?(?![\d.])")
_ACTIVE = {
    JobStatus.queued,
    JobStatus.running,
    JobStatus.cancel_requested,
}
_FILL = (210, 210, 210)
_BACKGROUND = (16, 16, 16)


def _thresholds() -> dict[str, str]:
    draft = analysis_page.threshold_draft_from_pilot()
    return {
        "high_confidence": format(draft.high_confidence, "g"),
        "activation": format(draft.activation, "g"),
        "lost_buffer_s": format(draft.lost_buffer_s, "g"),
        "min_observations": str(draft.min_observations),
        "display_threshold": format(draft.display_threshold, "g"),
    }


def _start_data(video_id: str, **overrides: str) -> dict[str, str]:
    data = {
        "video_id": video_id,
        "mode": "baseline",
        "analysis_fps": format(load_pilot().analysis_fps, "g"),
        **_thresholds(),
    }
    data.update(overrides)
    return data


def _post(
    client: TestClient,
    path: str,
    data: dict[str, str | list[str]],
) -> str:
    response = client.post(path, data=data, follow_redirects=False)
    assert response.status_code == 303, response.text
    location = urlparse(response.headers["location"])
    assert location.path == "/analysis"
    assert location.query == ""
    page = client.get("/analysis")
    assert page.status_code == 200
    return page.text


def _input_value(html: str, element_id: str) -> str:
    match = re.search(rf'<input\b[^>]*id="{element_id}"[^>]*>', html)
    assert match is not None, element_id
    found = re.search(r'value="([^"]*)"', match.group(0))
    assert found is not None, element_id
    return found.group(1)


def _selected(html: str, element_id: str) -> tuple[str, str]:
    match = re.search(
        rf'<select\b[^>]*id="{element_id}"[^>]*>(.*?)</select>',
        html,
        re.DOTALL,
    )
    assert match is not None, element_id
    for attrs, label in re.findall(r"<option\b([^>]*)>([^<]*)</option>", match.group(1)):
        if re.search(r"\sselected(?:\s|>|$)", attrs):
            value = re.search(r'value="([^"]*)"', attrs)
            assert value is not None, attrs
            return value.group(1), label.strip()
    raise AssertionError(element_id)


def _disabled(html: str, element_id: str) -> bool:
    match = re.search(rf'<button\b[^>]*id="{element_id}"[^>]*>', html)
    assert match is not None, element_id
    return re.search(r"\sdisabled(?:\s|=|>)", match.group(0)) is not None


def _form_html(html: str, form_id: str) -> str:
    match = re.search(rf'<form\b[^>]*id="{form_id}"[^>]*>(.*?)</form>', html, re.DOTALL)
    assert match is not None, form_id
    return match.group(1)


def _table_html(html: str, table_id: str) -> str:
    match = re.search(rf'<table\b[^>]*id="{table_id}"[^>]*>(.*?)</table>', html, re.DOTALL)
    assert match is not None, table_id
    return match.group(1)


def _video_src(html: str) -> str | None:
    match = re.search(r"<video\b([^>]*)>", html)
    if match is None:
        return None
    src = re.search(r'src="([^"]*)"', match.group(1))
    if src is None:
        return None
    return src.group(1)


def _stop_jobs(workspace: Path, project_id: str) -> None:
    for job in reopen(workspace):
        if job.status in _ACTIVE:
            request_cancel(workspace, project_id, job.job_id, kill_after_s=0.4)


def _requests(workspace: Path, project_id: str) -> list[Path]:
    runs = project_directory(workspace, project_id) / "runs"
    if not runs.is_dir():
        return []
    return sorted(runs.glob(f"*/{REQUEST_NAME}"))


def _clip(tmp_path: Path) -> str:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    clip = tmp_path / "clip.mp4"
    write_synthetic_mp4(clip, 0.4, 10)
    added = add_video(tmp_path, "pilot", clip)
    return added["video_id"]


def _shift_clock(workspace: Path, video_id: str, offset_s: float, duration_s: float) -> None:
    with connect(project_sqlite(workspace, "pilot")) as connection:
        connection.execute(
            """
            UPDATE videos
            SET start_offset_s = ?, duration_s = ?
            WHERE video_id = ?
            """,
            (offset_s, duration_s, video_id),
        )


def _weights(workspace: Path) -> Path:
    path = official_nano_weights_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"cached-name-only")
    return path


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _rect(name: str, x0: float, x1: float) -> list[dict[str, object]]:
    return [
        {"zone": name, "x": x0, "y": 0.0},
        {"zone": name, "x": x1, "y": 0.0},
        {"zone": name, "x": x1, "y": 1.0},
        {"zone": name, "x": x0, "y": 1.0},
    ]


def test_page_calls_services_and_does_not_load_weights() -> None:
    assert "st.fragment" not in _PAGE_SOURCE
    assert "st.video(" not in _PAGE_SOURCE
    assert "import streamlit" not in _PAGE_SOURCE
    assert "enqueue_analyze" in _PAGE_SOURCE
    assert "enqueue_video_export" in _PAGE_SOURCE
    assert "shell=True" not in _PAGE_SOURCE
    assert "open_detector" not in _PAGE_SOURCE
    assert "open_analysis_detector" not in _PAGE_SOURCE
    assert "load_bundle" not in _PAGE_SOURCE
    assert "RFDETR" not in _PAGE_SOURCE
    assert "encode_annotated_video(" not in _PAGE_SOURCE
    assert "drawable_canvas" not in _PAGE_SOURCE
    assert "streamlit_drawable_canvas" not in _PAGE_SOURCE
    assert "starlette" not in _PAGE_SOURCE
    assert "import streamlit" not in _WEB_SOURCE
    assert "from streamlit" not in _WEB_SOURCE
    assert "cv2" not in _WEB_SOURCE
    assert "charts" not in _WEB_SOURCE
    assert "open_detector" not in _WEB_SOURCE
    assert "load_bundle" not in _WEB_SOURCE
    assert "encode_rgb_jpeg" in _WEB_SOURCE
    assert _TEMPLATE.lower().count("<video") == 1
    assert "<canvas" not in _TEMPLATE.lower()
    assert "chart.js" not in _TEMPLATE
    assert "drawable_canvas" not in _TEMPLATE
    assert "data-job-line" in _TEMPLATE
    assert "data-progress-url" in _TEMPLATE
    assert 'src="/static/lab.js"' in _TEMPLATE
    assert "data-job-line" in _SCRIPT
    assert "body.line" in _SCRIPT
    assert ".src" not in _SCRIPT
    assert "<video" not in _SCRIPT


def test_empty_header_is_not_calculated_and_draws_no_boxes() -> None:
    header = analysis_page.analysis_header(None)
    assert header.source_time == ui_copy.NOT_CALCULATED
    assert header.model_version == ui_copy.NOT_CALCULATED
    assert header.observed == ui_copy.NOT_CALCULATED
    text = "\n".join(analysis_page.header_lines(header))
    assert text.count(ui_copy.NOT_CALCULATED) == 3
    assert _ZERO_TOKEN.search(text) is None
    review = analysis_page.review_frame([], [], 1.0)
    assert review.detections == ()
    assert review.predicted == ()


def test_empty_project_screen_has_no_metrics_or_boxes(tmp_path: Path) -> None:
    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/analysis")
    assert response.status_code == 200
    text = response.text
    assert text.count(ui_copy.NOT_CALCULATED) >= 3
    assert "Время источника" in text
    assert "Версия модели" in text
    assert "Доля наблюдаемых кадров" in text
    assert ui_copy.ANALYSIS_EMPTY in text
    assert ui_copy.ANALYSIS_NO_VIDEO in text
    assert _ZERO_TOKEN.search(text) is None, text
    assert "<video" not in text.lower()
    assert "<img" not in text.lower()
    assert "<canvas" not in text.lower()
    main = re.search(r"<main\b[^>]*>(.*)</main>", text, re.DOTALL)
    assert main is not None
    assert "<table" not in main.group(1).lower()
    assert "lab-chart" not in text
    assert "excavator_body" not in text
    assert "Экскаватор" not in text


def test_predicted_box_stays_out_of_the_detection_table() -> None:
    detections = [
        {
            "class_name": "bucket",
            "score": 0.91,
            "x1": 12.25,
            "y1": 1.5,
            "x2": 20.5,
            "y2": 8.5,
            "track_id_nullable": 4,
            "observation_state": "observed",
            "source_time_s": 0.1,
        },
        {
            "class_name": "bucket",
            "score": 0.2,
            "x1": 41.5,
            "y1": 2.5,
            "x2": 48.5,
            "y2": 9.5,
            "track_id_nullable": 4,
            "observation_state": "predicted",
            "source_time_s": 0.1,
        },
    ]
    tracks = [
        {
            "class_name": "bucket",
            "state": "predicted",
            "x1_nullable": 41.5,
            "y1_nullable": 2.5,
            "x2_nullable": 48.5,
            "y2_nullable": 9.5,
            "is_observed": False,
            "source_time_s": 0.1,
        },
        {
            "class_name": "haul_truck",
            "state": "observed",
            "x1_nullable": 3.5,
            "y1_nullable": 1.5,
            "x2_nullable": 9.5,
            "y2_nullable": 6.5,
            "is_observed": True,
            "source_time_s": 0.1,
        },
    ]
    review = analysis_page.review_frame(detections, tracks, 0.1)
    assert [row["x1"] for row in review.detections] == [12.25]
    assert all(row["observation_state"] != "predicted" for row in review.detections)
    assert 41.5 not in [row["x1"] for row in review.detections]
    assert len(review.predicted) == 1
    assert review.predicted[0]["state"] == "predicted"
    assert review.predicted[0]["x1_nullable"] == 41.5
    assert ui_copy.PREDICTED_HEADING == "predicted"


def test_observed_share_names_the_scope_and_skips_unanalyzed_frames() -> None:
    frames = [
        {
            "analyzed": True,
            "source_time_s": 0.1,
            "bucket_observation_state": "observed",
            "model_version": "bundle-a",
        },
        {
            "analyzed": True,
            "source_time_s": 0.2,
            "bucket_observation_state": "missing",
            "model_version": "bundle-a",
        },
        {
            "analyzed": False,
            "source_time_s": 0.3,
            "bucket_observation_state": "missing",
            "model_version": "bundle-a",
        },
    ]
    detections = [{"observation_state": "observed", "source_time_s": 0.1}]
    bucket = analysis_page.observed_share(frames, detections)
    assert bucket.text == "1 из 2, ковш"
    assert bucket.scope == "bucket"
    assert "3" not in bucket.text

    none = [
        {**row, "bucket_observation_state": "missing"}
        for row in frames
        if row["analyzed"]
    ]
    empty = analysis_page.observed_share(none, [])
    assert empty.text == "нет наблюдаемых кадров из 2, ковш"
    assert "0" not in empty.text

    classes = [
        {"analyzed": True, "source_time_s": 1.0, "bucket_observation_state": None},
        {"analyzed": True, "source_time_s": 2.0, "bucket_observation_state": None},
        {"analyzed": False, "source_time_s": 3.0, "bucket_observation_state": None},
    ]
    class_detections = [{"observation_state": "observed", "source_time_s": 1.0}]
    mixed = analysis_page.observed_share(classes, class_detections)
    assert mixed.text == "1 из 2, все классы"
    assert mixed.scope == "all"

    broken = [{"analyzed": True, "source_time_s": None, "bucket_observation_state": "observed"}]
    skipped = [{"analyzed": False, "source_time_s": 1.0, "bucket_observation_state": "observed"}]
    for rows in (broken, skipped, []):
        share = analysis_page.observed_share(rows, detections)
        assert share.text == ui_copy.NOT_CALCULATED
        assert share.text != "0"
        assert _ZERO_TOKEN.search(share.text) is None


def test_interval_defaults_follow_the_source_clock() -> None:
    late = {"start_offset_s": 300, "duration_s": 10}
    assert analysis_page.interval_defaults(late) == (300.0, 310.0)
    assert analysis_page.interval_defaults(late) == video_page.interval_bounds(late)
    assert analysis_page.interval_defaults({"start_offset_s": None, "duration_s": 10}) == (
        None,
        None,
    )
    assert analysis_page.interval_defaults(late) != (0.0, 10.0)


def test_threshold_edit_does_not_rewrite_finished_parquet(tmp_path: Path) -> None:
    finished = tmp_path / "run"
    table = finished / "detections.parquet"
    _write_rows(table, [{"x1": 12.25, "observation_state": "observed"}])
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    draft = replace(analysis_page.threshold_draft_from_pilot(), display_threshold=0.8)
    assert analysis_page.note_threshold_edit(draft, finished_run=finished) == (
        ui_copy.THRESHOLD_NEW_RUN
    )
    assert "новый запуск" in ui_copy.THRESHOLD_NEW_RUN
    bad = replace(draft, high_confidence=0.8, activation=0.1)
    message = analysis_page.note_threshold_edit(bad, finished_run=finished)
    assert "Завершённый прогон не изменён" in message
    assert hashlib.sha256(table.read_bytes()).hexdigest() == digest
    assert not (finished / REQUEST_NAME).exists()


def test_start_stays_disabled_while_a_job_holds_the_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FAKE_DETECTOR_ENV, "1")
    video_id = _clip(tmp_path)
    _weights(tmp_path)
    job = start_job(
        tmp_path,
        "pilot",
        kind="analyze",
        worker="sleep",
        video_duration_s=30,
        step_s=1,
        sleep_s=2,
    )
    calls = {"enqueue": 0}

    def _refuse(*_args: object, **_kwargs: object) -> None:
        calls["enqueue"] += 1
        raise AssertionError("enqueue")

    monkeypatch.setattr(analysis_page, "enqueue_analyze", _refuse)
    try:
        with TestClient(create_app(tmp_path)) as client:
            page = client.get("/analysis")
            assert page.status_code == 200
            text = page.text
            assert _disabled(text, "analysis-start-button")
            assert _disabled(text, "analysis-export")
            assert "уже выполняется задание" in text
            assert f'data-progress-url="jobs/{job.job_id}/progress"' in text
            assert 'data-job-line' in text
            assert "<video" not in text.lower()
            assert "<canvas" not in text.lower()
            assert "lab-chart" not in text
            progress = client.get(f"/jobs/{job.job_id}/progress")
            assert progress.status_code == 200
            payload = progress.json()
            assert payload["chart"] is None
            assert payload["line"] == job_status_line(reopen(tmp_path)[0])
            assert payload["line"] in text
            again = client.get("/analysis")
            assert _disabled(again.text, "analysis-start-button")
            posted = _post(client, "/analysis/start", _start_data(video_id, start_s="0", end_s="0.2"))
            assert "уже выполняется задание" in posted
        assert len(reopen(tmp_path)) == 1
        outcome = analysis_page.request_analysis(
            tmp_path,
            "pilot",
            video_id,
            "baseline_coco",
            5,
            0.0,
            0.2,
        )
        assert outcome.job_id is None
        assert outcome.error is not None
        assert "уже выполняется задание" in outcome.error
        assert calls["enqueue"] == 0
        assert _requests(tmp_path, "pilot") == []
    finally:
        _stop_jobs(tmp_path, "pilot")
        assert job.job_id


def test_start_enqueues_the_source_interval_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FAKE_DETECTOR_ENV, "1")
    video_id = _clip(tmp_path)
    _shift_clock(tmp_path, video_id, 300.0, 10.0)
    weights = _weights(tmp_path)
    try:
        with TestClient(create_app(tmp_path)) as client:
            page = client.get("/analysis")
            assert page.status_code == 200
            text = page.text
            assert float(_input_value(text, "analysis-start")) == 300.0
            assert float(_input_value(text, "analysis-end")) == 310.0
            assert float(_input_value(text, "analysis-fps")) == load_pilot().analysis_fps
            assert ui_copy.ANALYSIS_FPS_HELP in text
            assert ui_copy.ANALYSIS_PLAYER_HELP in text
            assert ui_copy.BASELINE_CAPTION in text
            assert ui_copy.THRESHOLD_NEW_RUN in text
            form = _form_html(text, "analysis-start-form")
            assert 'name="boxes"' not in form
            assert form.find('id="analysis-start-button"') < form.find('id="analysis-threshold"')
            _post(
                client,
                "/analysis/start",
                _start_data(video_id, start_s="300", end_s="310"),
            )
            requests = _requests(tmp_path, "pilot")
            assert len(requests) == 1
            body = json.loads(requests[0].read_text(encoding="utf-8"))
            assert body["start_s"] == 300.0
            assert body["end_s"] == 310.0
            assert body["analysis_fps"] == 5
            assert body["model_id"] == "baseline_coco"
            assert "toggles" not in body
            assert "display_threshold" in body
            assert "bytetrack" in body
            refreshed = client.get("/analysis")
            assert refreshed.status_code == 200
            assert len(_requests(tmp_path, "pilot")) == 1
            assert weights.read_bytes() == b"cached-name-only"
    finally:
        _stop_jobs(tmp_path, "pilot")


def test_missing_offset_does_not_become_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_id = _clip(tmp_path)
    with connect(project_sqlite(tmp_path, "pilot")) as connection:
        connection.execute(
            "UPDATE videos SET start_offset_s = NULL WHERE video_id = ?",
            (video_id,),
        )
    record = list_videos(tmp_path, "pilot")[0]
    assert analysis_page.interval_defaults(record)[0] is None
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _spy(*args: object, **kwargs: object) -> analysis_page.AnalysisAttempt:
        captured.append((args, kwargs))
        return analysis_page.AnalysisAttempt()

    monkeypatch.setattr(analysis_page, "request_analysis", _spy)
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/analysis")
        assert page.status_code == 200
        assert 'id="analysis-start"' not in page.text
        assert 'name="start_s"' not in page.text
        assert ui_copy.ANALYSIS_INTERVAL_UNKNOWN in page.text
        _post(client, "/analysis/start", _start_data(video_id))
    assert len(captured) == 1
    assert captured[0][0][5] is None
    assert captured[0][0][6] is None
    assert _requests(tmp_path, "pilot") == []


def test_measured_zero_offset_stays_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_id = _clip(tmp_path)
    _shift_clock(tmp_path, video_id, 0.0, 10.0)
    _weights(tmp_path)
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _spy(*args: object, **kwargs: object) -> analysis_page.AnalysisAttempt:
        captured.append((args, kwargs))
        return analysis_page.AnalysisAttempt(notice="held")

    monkeypatch.setattr(analysis_page, "request_analysis", _spy)
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/analysis")
        assert page.status_code == 200
        assert _input_value(page.text, "analysis-start") == "0"
        assert _input_value(page.text, "analysis-end") == "10"
        _post(
            client,
            "/analysis/start",
            _start_data(video_id, start_s="0", end_s="10", analysis_fps="4"),
        )
    assert len(captured) == 1
    args, kwargs = captured[0]
    assert args[3] == "baseline_coco"
    assert args[4] == 4.0
    assert args[5] == 0.0
    assert args[6] == 10.0
    assert isinstance(kwargs["thresholds"], analysis_page.ThresholdDraft)
    assert _requests(tmp_path, "pilot") == []


def test_missing_weights_show_the_notice_and_do_not_enqueue(tmp_path: Path) -> None:
    video_id = _clip(tmp_path)
    assert not official_nano_weights_path(tmp_path).is_file()
    try:
        with TestClient(create_app(tmp_path)) as client:
            page = client.get("/analysis")
            assert page.status_code == 200
            assert ui_copy.BASELINE_CAPTION in page.text
            assert "Официальные веса" in page.text
            assert "локальный кэш" in page.text
            posted = _post(
                client,
                "/analysis/start",
                _start_data(video_id, start_s="0", end_s="0.2"),
            )
            assert "Официальные веса" in posted
        assert _requests(tmp_path, "pilot") == []
        outcome = analysis_page.request_analysis(
            tmp_path,
            "pilot",
            video_id,
            "baseline_coco",
            5,
            0.0,
            0.2,
        )
        assert outcome.job_id is None
        assert outcome.notice is not None
        assert "Официальные веса" in outcome.notice
    finally:
        _stop_jobs(tmp_path, "pilot")


def test_custom_without_a_bundle_does_not_enqueue(tmp_path: Path) -> None:
    video_id = _clip(tmp_path)
    _weights(tmp_path)
    try:
        with TestClient(create_app(tmp_path)) as client:
            page = client.get("/analysis", params={"mode": "custom"})
            assert page.status_code == 200
            assert ui_copy.CUSTOM_NEEDS_BUNDLE in page.text
            assert 'id="analysis-bundle"' not in page.text
            posted = _post(
                client,
                "/analysis/start",
                _start_data(video_id, mode="custom", start_s="0", end_s="0.2"),
            )
            assert ui_copy.CUSTOM_NEEDS_BUNDLE in posted
        assert _requests(tmp_path, "pilot") == []
        outcome = analysis_page.request_analysis(
            tmp_path,
            "pilot",
            video_id,
            None,
            5,
            0.0,
            0.2,
        )
        assert outcome.job_id is None
        assert outcome.error == ui_copy.CUSTOM_NEEDS_BUNDLE
    finally:
        _stop_jobs(tmp_path, "pilot")


def test_bundle_list_shows_the_synthetic_flag_when_the_config_has_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FAKE_DETECTOR_ENV, "1")
    _clip(tmp_path)
    bundle = tmp_path / "weights" / "wire-model"
    bundle.mkdir(parents=True)
    (bundle / "weights.pth").write_bytes(b"not-loaded")
    (bundle / "model_config.json").write_text(
        json.dumps({"origin": "custom", "synthetic": True, "metric_role": "synthetic_wiring"}),
        encoding="utf-8",
    )
    plain = tmp_path / "weights" / "pit-model"
    plain.mkdir()
    (plain / "weights.pth").write_bytes(b"not-loaded")
    stamped = datetime.now(UTC).isoformat()
    with connect(project_sqlite(tmp_path, "pilot")) as connection:
        connection.execute(
            """
            INSERT INTO model_versions (
                model_id, project_id, dataset_id, variant, path, created_at
            ) VALUES ('pit-model', 'pilot', NULL, 'nano', ?, ?)
            """,
            (str(plain.resolve()), stamped),
        )
        connection.execute(
            """
            INSERT INTO model_versions (
                model_id, project_id, dataset_id, variant, path, created_at
            ) VALUES ('wire-model', 'pilot', NULL, 'nano', ?, ?)
            """,
            (str(bundle.resolve()), stamped),
        )
    russian = ui_copy.catalog("ru")
    english = ui_copy.catalog("en")
    options = {
        item.model_id: item
        for item in analysis_page.list_bundle_options(tmp_path, "pilot")
    }
    assert options["wire-model"].synthetic is True
    assert options["pit-model"].synthetic is False
    assert analysis_page.bundle_option_label(options["wire-model"], russian).endswith(
        russian.bundle_synthetic
    )
    assert analysis_page.bundle_option_label(options["wire-model"], english).endswith(
        english.bundle_synthetic
    )
    assert analysis_page.bundle_option_label(options["pit-model"], english) == "pit-model"
    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/analysis", params={"mode": "custom"})
    assert page.status_code == 200
    assert f"wire-model — {russian.bundle_synthetic}" in page.text
    assert f"pit-model — {russian.bundle_synthetic}" not in page.text
    assert 'value="wire-model"' in page.text


def test_custom_stores_the_bundle_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FAKE_DETECTOR_ENV, "1")
    video_id = _clip(tmp_path)
    bundle = tmp_path / "weights" / "pit-model"
    bundle.mkdir(parents=True)
    (bundle / "weights.pth").write_bytes(b"not-loaded")
    with connect(project_sqlite(tmp_path, "pilot")) as connection:
        connection.execute(
            """
            INSERT INTO model_versions (
                model_id, project_id, dataset_id, variant, path, created_at
            ) VALUES ('pit-model', 'pilot', NULL, 'nano', ?, ?)
            """,
            (str(bundle.resolve()), datetime.now(UTC).isoformat()),
        )
    try:
        with TestClient(create_app(tmp_path)) as client:
            page = client.get("/analysis", params={"mode": "custom"})
            assert page.status_code == 200
            selected = _selected(page.text, "analysis-bundle")
            assert selected[0] == "pit-model"
            assert selected[1] == "pit-model"
            _post(
                client,
                "/analysis/start",
                _start_data(
                    video_id,
                    mode="custom",
                    bundle="pit-model",
                    start_s="0",
                    end_s="0.2",
                ),
            )
        requests = _requests(tmp_path, "pilot")
        assert len(requests) == 1
        body = json.loads(requests[0].read_text(encoding="utf-8"))
        assert body["model_id"] == "pit-model"
        assert (bundle / "weights.pth").read_bytes() == b"not-loaded"
    finally:
        _stop_jobs(tmp_path, "pilot")


def test_playback_is_one_mp4_and_a_separate_predicted_list(tmp_path: Path) -> None:
    video_id = _clip(tmp_path)
    run = project_directory(tmp_path, "pilot") / "runs" / "demo"
    frames = [
        {
            "analyzed": True,
            "source_time_s": 0.1,
            "bucket_observation_state": "observed",
            "model_version": "bundle-a",
        },
        {
            "analyzed": True,
            "source_time_s": 0.2,
            "bucket_observation_state": "missing",
            "model_version": "bundle-a",
        },
        {
            "analyzed": False,
            "source_time_s": 0.3,
            "bucket_observation_state": "missing",
            "model_version": "bundle-a",
        },
    ]
    detections = [
        {
            "class_name": "bucket",
            "score": 0.91,
            "x1": 12.25,
            "y1": 1.5,
            "x2": 20.5,
            "y2": 8.5,
            "track_id_nullable": 4,
            "observation_state": "observed",
            "source_time_s": 0.1,
        }
    ]
    tracks = [
        {
            "class_name": "bucket",
            "state": "predicted",
            "x1_nullable": 41.5,
            "y1_nullable": 2.5,
            "x2_nullable": 48.5,
            "y2_nullable": 9.5,
            "is_observed": False,
            "source_time_s": 0.1,
        }
    ]
    _write_rows(run / "frame_status.parquet", frames)
    _write_rows(run / "detections.parquet", detections)
    _write_rows(run / "track_states.parquet", tracks)
    (run / "run.json").write_text(
        json.dumps({"video_id": video_id, "model_origin": "baseline_coco"}),
        encoding="utf-8",
    )
    export = project_directory(tmp_path, "pilot") / "exports" / "demo" / "annotated.mp4"
    export.parent.mkdir(parents=True)
    export.write_bytes((tmp_path / "clip.mp4").read_bytes())
    digest = hashlib.sha256((run / "detections.parquet").read_bytes()).hexdigest()
    snapshot = analysis_page.load_snapshot(run)
    assert snapshot is not None
    review = analysis_page.review_frame(
        snapshot.detection_rows,
        snapshot.track_rows,
        0.1,
    )
    assert [row["x1"] for row in review.detections] == [12.25]
    assert review.predicted[0]["state"] == "predicted"
    header = analysis_page.analysis_header(snapshot, 0.1)
    assert header.model_version == "bundle-a"
    assert header.observed == "1 из 2, ковш"
    assert header.source_time == "0.1 с"

    with TestClient(create_app(tmp_path)) as client:
        page = client.get("/analysis")
        assert page.status_code == 200
        text = page.text
        assert text.lower().count("<video") == 1
        assert text.lower().count("<img") <= 2
        assert "<canvas" not in text.lower()
        assert "lab-chart" not in text
        assert _video_src(text) == "/runs/pilot/demo/video"
        assert "predicted" in text
        assert "bundle-a" in text
        assert "1 из 2, ковш" in text
        assert ui_copy.BASELINE_CAPTION in text
        detection_text = _table_html(text, "analysis-detections")
        predicted_text = _table_html(text, "analysis-predicted")
        assert "12.25" in detection_text
        assert "41.5" not in detection_text
        assert "41.5" in predicted_text
        posted = _post(
            client,
            "/analysis/threshold",
            _start_data(video_id, display_threshold="0.8", run_id="demo"),
        )
        assert ui_copy.THRESHOLD_NEW_RUN in posted
        assert hashlib.sha256((run / "detections.parquet").read_bytes()).hexdigest() == digest
        assert _requests(tmp_path, "pilot") == []

        def _boom(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("frames")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(analysis_page, "read_source_frame", _boom)
        monkeypatch.setattr("excavation_lab.media.reader.FrameReader.iterate", _boom)
        try:
            ranged = client.get("/runs/pilot/demo/video", headers={"Range": "bytes=0-7"})
        finally:
            monkeypatch.undo()
        assert ranged.status_code == 206
        assert ranged.headers["content-type"].startswith("video/mp4")
        assert ranged.content == export.read_bytes()[:8]
        blocked = client.get("/runs/pilot/demo/video", params={"path": str(tmp_path / "clip.mp4")})
        assert blocked.status_code == 404


def test_zone_preview_draws_vertices_and_keeps_unit_coordinates(tmp_path: Path) -> None:
    right = analysis_page.zone_preview_rgb(_rect("dig_zone", 0.5, 1.0), width=40, height=20)
    left = analysis_page.zone_preview_rgb(_rect("dig_zone", 0.0, 0.5), width=40, height=20)
    assert right is not None and left is not None
    assert tuple(right[10, 30]) == _FILL
    assert tuple(right[10, 8]) == _BACKGROUND
    assert tuple(left[10, 8]) == _FILL
    assert tuple(left[10, 30]) == _BACKGROUND
    rows = _rect("dig_zone", 0.7, 1.0)
    drawn = analysis_page.zone_preview_rgb(rows, width=40, height=20)
    assert drawn is not None
    assert rows[0]["x"] == 0.7

    list_catalogs(tmp_path)
    create_project("pilot", tmp_path, catalog_id="excavation")
    saved = analysis_page.save_zone_table(tmp_path, "pilot", "pit_a", rows)
    assert saved.parent == project_directory(tmp_path, "pilot") / "views"
    loaded = yaml.safe_load(saved.read_text(encoding="utf-8"))
    assert loaded["dig_zone"][0]["x"] == 0.7
    assert all(0.0 <= point["x"] <= 1.0 for point in loaded["dig_zone"])
    repo_view = Path(__file__).resolve().parents[2] / "configs" / "views" / "pit_a.yaml"
    assert not repo_view.exists()

    bad = _rect("dig_zone", 0.0, 1.0)
    bad[1] = {"zone": "dig_zone", "x": 1.2, "y": 0.0}
    assert analysis_page.zone_preview_rgb(bad, width=40, height=20) is None
    with pytest.raises(LabError, match="вне доли кадра"):
        analysis_page.save_zone_table(tmp_path, "pilot", "pit_b", bad)
    assert not (project_directory(tmp_path, "pilot") / "views" / "pit_b.yaml").exists()


def test_editor_delta_keeps_the_edited_vertex_on_save_and_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_id = _clip(tmp_path)
    video = list_videos(tmp_path, "pilot")[0]
    width = int(video["width"])
    height = int(video["height"])
    view_id = "pit_delta"
    run_id = "run-zones"
    stale = 0.2
    edited = {
        "zone": ["dig_zone", "dig_zone", "dig_zone"],
        "x": ["0.35", "0.8", "0.5"],
        "y": ["0.1", "0.1", "0.8"],
    }

    class _Idle:
        pid = 2_147_483_647

    monkeypatch.setattr("excavation_lab.jobs._spawn", lambda *_args, **_kwargs: _Idle())
    try:
        run = project_directory(tmp_path, "pilot") / "runs" / run_id
        _write_rows(
            run / "frame_status.parquet",
            [{"analyzed": True, "source_time_s": 0.1, "model_version": "bundle-a"}],
        )
        _write_rows(
            run / "detections.parquet",
            [{"x1": 12.25, "observation_state": "observed", "source_time_s": 0.1}],
        )
        (run / "run.json").write_text(
            json.dumps({"video_id": video_id, "model_origin": "custom"}),
            encoding="utf-8",
        )
        digest = hashlib.sha256((run / "detections.parquet").read_bytes()).hexdigest()
        with connect(project_sqlite(tmp_path, "pilot")) as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, project_id, status, pid, partial,
                    analyzed_end_s, error_code, created_at
                ) VALUES (?, ?, 'completed', NULL, 0, NULL, NULL, ?)
                """,
                ("job-zones", "pilot", "2026-10-10T00:00:00+00:00"),
            )
        insert_analysis_run(
            tmp_path,
            "pilot",
            run_id=run_id,
            job_id="job-zones",
            video_id=video_id,
            view_id=view_id,
            model_id="model-1",
            model_origin="custom",
            analysis_fps=5.0,
            source_fps=None,
            export_fps=None,
            start_s=0.0,
            end_s=None,
            decode_ms=None,
            encode_ms=None,
        )
        fields: dict[str, str | list[str]] = {
            "video_id": video_id,
            "view_id": view_id,
            "run_id": run_id,
            **edited,
            "boxes": "1",
            "class_names": "1",
            "scores": "1",
            "track_ids": "1",
            "bucket_trail": "1",
            "zones": "1",
        }
        with TestClient(create_app(tmp_path)) as client:
            saved_page = _post(client, "/analysis/zones", fields)
            assert ui_copy.ZONE_SAVED in saved_page
            saved = project_directory(tmp_path, "pilot") / "views" / f"{view_id}.yaml"
            loaded = yaml.safe_load(saved.read_text(encoding="utf-8"))
            assert loaded["dig_zone"] != []
            assert loaded["dig_zone"][0] == {"x": 0.35, "y": 0.1}
            assert stale not in [point["x"] for point in loaded["dig_zone"]]
            assert all(0.0 <= point["x"] <= 1.0 for point in loaded["dig_zone"])
            assert all(0.0 <= point["y"] <= 1.0 for point in loaded["dig_zone"])
            _post(client, "/analysis/export", fields)
            requests = list(
                (project_directory(tmp_path, "pilot") / "runs").glob("*/export_request.json")
            )
            assert len(requests) == 1
            body = json.loads(requests[0].read_text(encoding="utf-8"))
            assert body["toggles"]["zones"] is True
            assert "boxes" not in body or body["toggles"]["boxes"] is True
            zones = [zone for zone in body["zones"] if zone["name"] == "dig_zone"]
            assert len(zones) == 1
            points = [tuple(point) for point in zones[0]["points"]]
            assert (0.35 * width, 0.1 * height) in points
            assert (stale * width, 0.1 * height) not in points
            assert hashlib.sha256((run / "detections.parquet").read_bytes()).hexdigest() == digest
            assert not list((project_directory(tmp_path, "pilot") / "exports").glob("**/*.mp4"))
            bad: dict[str, str | list[str]] = {
                "video_id": video_id,
                "view_id": "pit_b",
                "zone": ["dig_zone", "dig_zone", "dig_zone"],
                "x": ["0", "1.2", "0.5"],
                "y": ["0", "0", "1"],
            }
            rejected = _post(client, "/analysis/zones", bad)
            assert "вне доли кадра" in rejected
        assert not (project_directory(tmp_path, "pilot") / "views" / "pit_b.yaml").exists()
        repo_view = Path(__file__).resolve().parents[2] / "configs" / "views" / f"{view_id}.yaml"
        assert not repo_view.exists()
    finally:
        _stop_jobs(tmp_path, "pilot")


def test_frame_route_encodes_rgb_and_rejects_a_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_id = _clip(tmp_path)

    def _red(_path: Path, _video_id: str, _source_time_s: float) -> np.ndarray:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[..., 0] = 255
        return image

    monkeypatch.setattr(analysis_page, "read_source_frame", _red)
    with TestClient(create_app(tmp_path)) as client:
        refused = client.get(
            "/analysis/frame",
            params={"video_id": video_id, "source_time_s": "0.1", "path": str(tmp_path / "clip.mp4")},
        )
        assert refused.status_code == 400
        missing = client.get("/analysis/frame", params={"path": "/tmp/clip.mp4"})
        assert missing.status_code == 400
        response = client.get(
            "/analysis/frame",
            params={"video_id": video_id, "source_time_s": "0.1"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")
    pixel = Image.open(BytesIO(response.content)).convert("RGB").getpixel((4, 4))
    assert isinstance(pixel, tuple)
    assert pixel[0] > 200
    assert pixel[2] < 40
