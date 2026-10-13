"""Loss and AP50 specs reach the page as Altair JSON. The script is not Vega."""

from __future__ import annotations

import json
import re
from pathlib import Path

from starlette.testclient import TestClient

from excavation_lab import jobs as job_module
from excavation_lab.storage import create_project, project_directory
from excavation_lab.ui import training_page
from excavation_lab.ui.charts import ap50_chart, loss_chart
from excavation_lab.ui.tokens import theme_custom_properties
from excavation_lab.web.app import create_app

_EPOCHS = (1, 2, 3)
_LOSSES = (0.137, 0.864, 0.421)
_HEX = re.compile(r"#[0-9A-Fa-f]{3,8}\b")
_STATIC = Path(__file__).resolve().parents[2] / "src" / "excavation_lab" / "web" / "static"
_LOG = "".join(
    f"{epoch},{loss}\n" for epoch, loss in zip(_EPOCHS, _LOSSES, strict=True)
)
_HEADER = "epoch,train_loss\n" + _LOG


def test_spec_fields_are_epoch_loss_and_class_ap50() -> None:
    loss = loss_chart(_EPOCHS, _LOSSES, "dark").to_dict()
    bars = ap50_chart(({"класс": "Ковш", "AP50": 0.42},), "light").to_dict()
    assert {tuple(row) for row in loss["data"]["values"]} == {("epoch", "loss")}
    assert [row["epoch"] for row in loss["data"]["values"]] == list(_EPOCHS)
    assert [row["loss"] for row in loss["data"]["values"]] == list(_LOSSES)
    assert {tuple(row) for row in bars["data"]["values"]} == {("класс", "AP50")}
    assert bars["data"]["values"][0]["класс"] == "Ковш"
    assert bars["data"]["values"][0]["AP50"] == 0.42
    script = (_STATIC / "chart.js").read_text(encoding="utf-8")
    assert "data.values" in script
    assert "row.epoch" in script
    assert "row.loss" in script
    assert 'row["класс"]' in script
    assert "row.AP50" in script
    assert 'mark.type === "line"' in script
    assert 'mark.type === "bar"' in script
    assert "fetch(" not in script
    assert "vega" not in script.lower()
    assert _HEX.search(script) is None


def test_progress_chart_is_the_loss_spec(tmp_path: Path) -> None:
    create_project("pilot", tmp_path)
    view = training_page.training_log_view(_HEADER, planned_epochs=50)
    assert training_page.should_draw_chart(view, None) is True
    job_id = "chartjob01"
    run_dir = _insert_job(tmp_path, job_id)
    (run_dir / "train_request.json").write_text('{"profile":"smoke"}\n', encoding="utf-8")
    (run_dir / "metrics.csv").write_text(_HEADER, encoding="utf-8")
    with TestClient(create_app(tmp_path)) as client:
        response = client.get(f"/jobs/{job_id}/progress")
    assert response.status_code == 200
    chart = response.json()["chart"]
    observed = theme_custom_properties("dark")["--pdm-series-observed"]
    assert chart == loss_chart(_EPOCHS, _LOSSES, "dark").to_dict()
    assert chart["mark"]["color"] == observed
    assert [row["epoch"] for row in chart["data"]["values"]] == list(_EPOCHS)
    assert [row["loss"] for row in chart["data"]["values"]] == list(_LOSSES)


def test_header_only_progress_has_no_chart(tmp_path: Path) -> None:
    create_project("pilot", tmp_path)
    job_id = "charthead01"
    run_dir = _insert_job(tmp_path, job_id)
    (run_dir / "train_request.json").write_text('{"profile":"smoke"}\n', encoding="utf-8")
    (run_dir / "metrics.csv").write_text("epoch\n", encoding="utf-8")
    view = training_page.training_log_view("epoch\n", planned_epochs=50)
    assert training_page.should_draw_chart(view, None) is False
    with TestClient(create_app(tmp_path)) as client:
        body = client.get(f"/jobs/{job_id}/progress").json()
    assert body["chart"] is None


def test_quality_without_scores_omits_the_spec(tmp_path: Path) -> None:
    create_project("pilot", tmp_path)
    _metrics(tmp_path, "run-empty", {"classes": {"bucket": {}}})
    with TestClient(create_app(tmp_path)) as client:
        missing = client.get("/quality")
    assert missing.status_code == 200
    assert 'id="ap50-chart"' not in missing.text
    assert "vega-lite" not in missing.text
    _metrics(
        tmp_path,
        "run-partial",
        {"classes": {"bucket": {"ap50": 0.42}, "haul_truck": {}}},
    )
    with TestClient(create_app(tmp_path)) as client:
        partial = client.get("/quality", params={"quality_run": "run-partial"})
    assert partial.status_code == 200
    spec = _embedded(partial.text)
    assert [row["AP50"] for row in spec["data"]["values"]] == [0.42]
    assert spec["mark"]["color"] == theme_custom_properties("dark")["--pdm-accent"]
    assert spec["data"]["values"][0]["класс"]
    assert "haul" not in json.dumps(spec["data"]["values"])


def test_events_and_analysis_responses_have_no_chart_spec(tmp_path: Path) -> None:
    create_project("pilot", tmp_path)
    job_id = "analyzejob1"
    _insert_job(tmp_path, job_id, kind="analyze")
    with TestClient(create_app(tmp_path)) as client:
        events = client.get("/events")
        analysis = client.get("/analysis")
        progress = client.get(f"/jobs/{job_id}/progress")
    assert events.status_code == 200
    assert analysis.status_code == 200
    assert progress.status_code == 200
    for text in (events.text, analysis.text, progress.text):
        assert 'id="ap50-chart"' not in text
        assert "vega-lite" not in text
        assert "$schema" not in text
    assert progress.json()["chart"] is None


def _embedded(html: str) -> dict[str, object]:
    match = re.search(
        r'<script type="application/json" id="ap50-chart">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    loaded = json.loads(match.group(1))
    assert isinstance(loaded, dict)
    return loaded


def _metrics(workspace: Path, eval_id: str, document: dict[str, object]) -> None:
    directory = project_directory(workspace, "pilot") / "runs" / eval_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(
        json.dumps(document),
        encoding="utf-8",
    )


def _insert_job(workspace: Path, job_id: str, *, kind: str = "train") -> Path:
    run_dir = job_module.run_directory(workspace, "pilot", job_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with job_module._lab(workspace) as connection:
        connection.execute(
            """
            INSERT INTO lab_jobs (
                job_id, project_id, kind, worker, status, pid, partial,
                analyzed_end_s, error_code, orphaned, run_dir,
                video_duration_s, step_s, sleep_s, created_at
            ) VALUES (
                ?, ?, ?, 'sleep', 'completed', NULL, 0, NULL, NULL, 0, ?, 10, 1, 0.05, ?
            )
            """,
            (
                job_id,
                "pilot",
                kind,
                str(run_dir),
                "2026-01-01T00:00:00+00:00",
            ),
        )
    return run_dir
