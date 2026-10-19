"""Action inference: unknown for a mask, a low score, and a short blip."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from excavation_lab import jobs, storage
from excavation_lab.actions.features import FeatureSpec, build_features
from excavation_lab.actions.infer import (
    INTERVAL_COLUMNS,
    ActionBundle,
    enqueue_action_infer,
    export_action_intervals,
    load_action_bundle,
    merge_intervals,
    predict_frames,
    run_action_infer_job,
    write_action_results,
)
from excavation_lab.actions.model import ActionNet, Standardizer
from excavation_lab.cli import app
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.splits import canonical_json_bytes
from excavation_lab.errors import JobAlreadyRunningError, LabError
from excavation_lab.jobs import get_job, request_cancel, run_directory, start_job
from excavation_lab.schemas import JobStatus
from excavation_lab.tracking.export_tables import (
    DETECTIONS_COLUMNS,
    DETECTIONS_NAME,
    FRAME_STATUS_COLUMNS,
    FRAME_STATUS_NAME,
    TRACK_STATE_COLUMNS,
    TRACK_STATES_NAME,
)

_RUNNER = CliRunner()
_PROJECT = "mvp"
_VIDEO = "vid"
_DETECTOR = "det_custom"
_MODEL = "act1"
_FPS = 5.0


class _DummyProcess:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.pid = os.getpid()

    def poll(self) -> None:
        return None


def _dummy_spawn(*_args: object, **_kwargs: object) -> _DummyProcess:
    return _DummyProcess()


def _spec(fps: float = _FPS) -> FeatureSpec:
    return FeatureSpec(class_names=("mark",), analysis_fps=fps)


def _checksums(directory: Path) -> None:
    files: dict[str, str] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.name == "checksums.json" or not path.is_file():
            continue
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"algorithm": "sha256", "files": files, "schema_version": 1}
    (directory / "checksums.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_bundle(
    directory: Path,
    *,
    fps: float = _FPS,
    detector_id: str = _DETECTOR,
    model_id: str = _MODEL,
    class_order: tuple[str, ...] = ("digging", "waiting"),
    catalog_order: tuple[str, ...] | None = None,
    feature_hash: str | None = None,
    outputs: int | None = None,
) -> FeatureSpec:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    spec = _spec(fps)
    n_features = len(spec.columns)
    n_classes = len(class_order) if outputs is None else outputs
    module = ActionNet(n_features, n_classes, 1)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()
        module.conv1.weight[0, 0, 1] = 1
        module.conv2.weight[0, 0, 1] = 1
        if n_classes >= 1:
            module.fc.weight[0, 0] = -20
            module.fc.bias[0] = 10
        if n_classes >= 2:
            module.fc.weight[1, 0] = 20
            module.fc.bias[1] = -10
    torch.save(module.state_dict(), directory / "weights.pt")
    standardizer = Standardizer(np.zeros(n_features), np.ones(n_features))
    (directory / "standardizer.json").write_text(
        json.dumps(standardizer.to_json(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    spec_document = spec.model_dump(mode="json")
    (directory / "feature_spec.json").write_text(
        json.dumps(spec_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(canonical_json_bytes(spec_document)).hexdigest()
    names = catalog_order if catalog_order is not None else class_order
    window_frames = max(3, round(0.6 * fps))
    document = {
        "action_catalog": [
            {"id": index, "name": name, "title": name} for index, name in enumerate(names)
        ],
        "action_model_id": model_id,
        "analysis_fps": fps,
        "class_order": list(class_order),
        "config": {"min_confidence": 0.5, "min_duration_s": 1.0},
        "detector_model_id": detector_id,
        "feature_spec": spec_document,
        "feature_spec_hash": digest if feature_hash is None else feature_hash,
        "standardizer": standardizer.to_json(),
        "window_frames": window_frames,
        "window_s": 0.6,
    }
    (directory / "action_bundle.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _checksums(directory)
    return spec


def _feature_frame(spec: FeatureSpec, signal: list[float], mask: list[int]) -> pd.DataFrame:
    rows = []
    for index, (value, flag) in enumerate(zip(signal, mask, strict=True)):
        row = dict.fromkeys(spec.columns, 0.0)
        row["mark_present"] = value
        row["frame_mask"] = flag
        row["run_id"] = "run_a"
        row["run_index"] = 0
        row["run_start"] = index == 0
        row["source_frame_index"] = index
        row["source_time_s"] = index * 0.2
        rows.append(row)
    return pd.DataFrame(rows)


def _write_tables(run_dir: Path, times: list[float], masked_index: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, stamp in enumerate(times):
        row = dict.fromkeys(FRAME_STATUS_COLUMNS)
        row.update(
            {
                "run_id": "run_a",
                "video_id": _VIDEO,
                "source_frame_index": index,
                "source_time_s": stamp,
                "decode_status": "ok",
                "analyzed": index != masked_index,
                "width": 64,
                "height": 48,
                "num_detections": 0,
            }
        )
        rows.append(row)
    pd.DataFrame(rows, columns=list(FRAME_STATUS_COLUMNS)).to_parquet(run_dir / FRAME_STATUS_NAME)
    pd.DataFrame(columns=list(DETECTIONS_COLUMNS)).to_parquet(run_dir / DETECTIONS_NAME)
    pd.DataFrame(columns=list(TRACK_STATE_COLUMNS)).to_parquet(run_dir / TRACK_STATES_NAME)


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(jobs, "_spawn", _dummy_spawn)
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project(_PROJECT, workspace, catalog_id="excavation")
    storage.insert_video(workspace, _PROJECT, _VIDEO, name="clip")
    return workspace


def _finish_run(
    workspace: Path,
    *,
    run_id: str = "run_a",
    model_id: str = _DETECTOR,
    origin: str = "custom",
    analysis_fps: float = _FPS,
    times: list[float] | None = None,
    masked_index: int = 10,
) -> str:
    record = start_job(
        workspace,
        _PROJECT,
        kind="analyze",
        worker="sleep",
        video_duration_s=10.0,
        step_s=1.0,
    )
    stamps = times if times is not None else [index * 0.2 for index in range(20)] + [90.0]
    _write_tables(run_directory(workspace, _PROJECT, record.job_id), stamps, masked_index)
    jobs.mark_completed(workspace, _PROJECT, record.job_id, [run_directory(workspace, _PROJECT, record.job_id) / FRAME_STATUS_NAME])
    storage.insert_analysis_run(
        workspace,
        _PROJECT,
        run_id=run_id,
        job_id=record.job_id,
        video_id=_VIDEO,
        view_id=None,
        model_id=model_id,
        model_origin=origin,
        analysis_fps=analysis_fps,
        source_fps=None,
        export_fps=None,
        start_s=0.0,
        end_s=100.0,
        decode_ms=None,
        encode_ms=None,
    )
    return run_id


def _register(workspace: Path, bundle: Path, *, detector_id: str = _DETECTOR) -> None:
    storage.insert_action_model(
        workspace,
        _PROJECT,
        action_model_id=_MODEL,
        detector_model_id=detector_id,
        source_run_ids=["run_a"],
        path=bundle,
    )


def test_mask_is_unknown_and_low_confidence_is_not_a_class(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    spec = _write_bundle(bundle_dir)
    bundle = load_action_bundle(bundle_dir)
    mask = [1] * 40
    for index in range(12, 18):
        mask[index] = 0
    predicted = predict_frames(bundle, _feature_frame(spec, [0.0] * 40, mask))
    assert predicted.iloc[12]["action_name"] == "unknown"
    assert math.isnan(predicted.iloc[12]["confidence"])
    assert math.isnan(predicted.iloc[12]["p_digging"])
    assert predicted.iloc[8]["action_name"] == "digging"
    assert predicted.iloc[8]["confidence"] > 0.9

    flat = predict_frames(bundle, _feature_frame(spec, [0.5] * 40, [1] * 40))
    middle = flat.iloc[20]
    assert middle["action_name"] in {"digging", "waiting"}
    assert middle["confidence"] == pytest.approx(0.5, abs=1e-3)
    spans = merge_intervals(flat, min_confidence=0.6, min_duration_s=1.0)
    assert spans
    assert {span.action_name for span in spans} == {"unknown"}
    paths = write_action_results(
        tmp_path / "low",
        flat,
        spans,
        source_run_id="run_a",
        action_model_id=_MODEL,
        detector_model_id=_DETECTOR,
        min_confidence=0.6,
        min_duration_s=1.0,
    )
    stored = pd.read_parquet(paths[0])
    assert stored.iloc[20]["action"] == "unknown"
    assert bool(stored.iloc[20]["unknown"])
    assert stored.iloc[20]["confidence"] == pytest.approx(0.5, abs=1e-3)


def test_short_blip_becomes_unknown_and_csv_roundtrips(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    spec = _write_bundle(bundle_dir)
    bundle = load_action_bundle(bundle_dir)
    signal = [0.0] * 60
    for index in (25, 26, 27):
        signal[index] = 1.0
    predicted = predict_frames(bundle, _feature_frame(spec, signal, [1] * 60))
    assert predicted.iloc[26]["action_name"] == "waiting"
    assert predicted.iloc[26]["confidence"] > 0.9
    spans = merge_intervals(
        predicted,
        min_confidence=0.6,
        min_duration_s=1.0,
        analysis_fps=_FPS,
    )
    assert spans[-1].end_s == pytest.approx(59 * 0.2 + 1.0 / _FPS)
    for earlier, later in itertools.pairwise(spans):
        assert earlier.end_s > earlier.start_s
        assert earlier.end_s <= later.start_s + 1e-9
        assert later.start_s >= earlier.end_s - 1e-9
    blips = [
        span
        for span in spans
        if span.action_name == "unknown" and span.mean_confidence is not None and span.mean_confidence > 0.9
    ]
    assert len(blips) == 1
    assert blips[0].duration_s == pytest.approx(0.6, abs=1e-6)
    assert blips[0].frames == 3
    assert "waiting" not in {span.action_name for span in spans}
    assert "digging" in {span.action_name for span in spans}
    digging = [span for span in spans if span.action_name == "digging"]
    assert any(span.duration_s > 4.0 for span in digging)
    paths = write_action_results(
        tmp_path / "out",
        predicted,
        spans,
        source_run_id="run_a",
        action_model_id=_MODEL,
        detector_model_id=_DETECTOR,
        min_confidence=0.6,
        min_duration_s=1.0,
    )
    text = paths[1].read_text(encoding="utf-8")
    assert "0,6" not in text
    assert text.startswith("start_s,end_s,duration_s,action_name,mean_confidence,frames\n")
    loaded = pd.read_csv(paths[1])
    assert list(loaded.columns) == list(INTERVAL_COLUMNS)
    row = loaded.loc[loaded["frames"] == 3].iloc[0]
    assert row["action_name"] == "unknown"
    assert row["duration_s"] == pytest.approx(0.6, abs=1e-6)
    assert row["mean_confidence"] > 0.9
    sidecar = json.loads(paths[2].read_text(encoding="utf-8"))
    assert sidecar["source_run_id"] == "run_a"
    assert sidecar["unknown_share"] > 0.0
    assert b"\xef\xbb\xbf" not in paths[2].read_bytes()


def test_gap_stays_unlabeled() -> None:
    frame = pd.DataFrame(
        {
            "source_time_s": [0.0, 0.2, 0.4, 5.0, 5.2],
            "action_name": ["digging"] * 5,
            "confidence": [0.95] * 5,
        }
    )
    frame.attrs["analysis_fps"] = _FPS
    spans = merge_intervals(frame, min_confidence=0.5, min_duration_s=0.3)
    assert len(spans) == 2
    assert spans[0].action_name == "digging"
    assert spans[1].action_name == "digging"
    assert spans[0].end_s == pytest.approx(0.6)
    assert spans[1].start_s == pytest.approx(5.0)
    assert spans[0].end_s < spans[1].start_s
    assert all(not (span.start_s <= 1.0 < span.end_s) for span in spans)
    assert spans[1].end_s == pytest.approx(5.2 + 1.0 / _FPS)


def test_bundle_refuses_hash_class_order_and_checksum(tmp_path: Path) -> None:
    broken_hash = tmp_path / "hash"
    _write_bundle(broken_hash, feature_hash="0" * 64)
    with pytest.raises(LabError) as hashed:
        load_action_bundle(broken_hash)
    assert hashed.value.code == "feature_spec_mismatch"

    swapped = tmp_path / "order"
    _write_bundle(swapped, class_order=("waiting", "digging"), catalog_order=("digging", "waiting"))
    with pytest.raises(LabError) as ordered:
        load_action_bundle(swapped)
    assert ordered.value.code == "class_order"

    tampered = tmp_path / "sum"
    _write_bundle(tampered)
    weights = tampered / "weights.pt"
    weights.write_bytes(weights.read_bytes() + b"x")
    with pytest.raises(LabError) as checksum:
        load_action_bundle(tampered)
    assert checksum.value.code == "action_bundle_checksum"


def test_job_keeps_test_frames_and_refuses_detector_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _project(tmp_path, monkeypatch)
    _finish_run(workspace)
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    _register(workspace, bundle)
    other = tmp_path / "other"
    _write_bundle(other, detector_id="other_det", model_id="act_other")
    storage.insert_action_model(
        workspace,
        _PROJECT,
        action_model_id="act_other",
        detector_model_id="other_det",
        source_run_ids=["run_a"],
        path=other,
    )
    with pytest.raises(LabError) as mismatch:
        enqueue_action_infer(workspace, _PROJECT, "run_a", "act_other")
    assert mismatch.value.code == "action_detector_mismatch"

    slow = tmp_path / "slow"
    _write_bundle(slow, fps=2.0, model_id="act_slow")
    storage.insert_action_model(
        workspace,
        _PROJECT,
        action_model_id="act_slow",
        detector_model_id=_DETECTOR,
        source_run_ids=["run_a"],
        path=slow,
    )
    with pytest.raises(LabError) as fps:
        enqueue_action_infer(workspace, _PROJECT, "run_a", "act_slow")
    assert fps.value.code == "action_fps_mismatch"

    with pytest.raises(LabError) as many:
        enqueue_action_infer(workspace, _PROJECT, ["run_a", "run_b"], _MODEL)  # type: ignore[arg-type]
    assert many.value.code == "invalid_action_infer"

    record = enqueue_action_infer(workspace, _PROJECT, "run_a", _MODEL)
    with pytest.raises(LabError) as early:
        jobs.mark_completed(workspace, _PROJECT, record.job_id)
    assert early.value.code == "missing_artifact"
    assert get_job(workspace, _PROJECT, record.job_id).status is JobStatus.running

    seen: dict[str, object] = {}
    real = build_features

    def spy(tables: object, **kwargs: object) -> pd.DataFrame:
        seen["keep_ranges"] = kwargs.get("keep_ranges", "missing")
        seen["run_index"] = kwargs.get("run_index", "missing")
        seen["run_id"] = kwargs.get("run_id")
        return real(tables, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("excavation_lab.actions.infer.build_features", spy)
    assert run_action_infer_job(workspace, record.job_id) == 0
    assert seen == {"keep_ranges": None, "run_index": 0, "run_id": "run_a"}
    finished = get_job(workspace, _PROJECT, record.job_id)
    assert finished.status is JobStatus.completed
    run_dir = run_directory(workspace, _PROJECT, record.job_id)
    frame = pd.read_parquet(run_dir / "actions.parquet")
    table = pd.read_csv(run_dir / "action_intervals.csv")
    assert 90.0 in set(frame["source_time_s"].tolist())
    masked = frame.loc[frame["source_frame_index"] == 10].iloc[0]
    assert masked["action"] == "unknown"
    assert bool(masked["unknown"])
    assert not any(row.start_s <= 10.0 < row.end_s for row in table.itertuples())
    assert "digging" in set(table["action_name"])
    edge = table.loc[np.isclose(table["start_s"], 0.0)].iloc[0]
    assert edge["action_name"] == "unknown"
    assert pd.isna(edge["mean_confidence"])
    listed = storage.list_action_inferences(workspace, _PROJECT)
    assert len(listed) == 1
    assert listed[0]["job_id"] == record.job_id
    assert listed[0]["source_run_id"] == "run_a"
    assert listed[0]["action_model_id"] == _MODEL
    body = json.loads((run_dir / "action_run.json").read_text(encoding="utf-8"))
    assert body["detector_model_id"] == _DETECTOR
    assert body["min_confidence"] == 0.5
    assert body["min_duration_s"] == 1.0


def test_baseline_run_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = _project(tmp_path, monkeypatch)
    _finish_run(workspace, origin="baseline_coco")
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    _register(workspace, bundle)
    with pytest.raises(LabError) as refused:
        enqueue_action_infer(workspace, _PROJECT, "run_a", _MODEL)
    assert refused.value.code == "action_not_custom"


def test_cli_infer_json_and_excel_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = _project(tmp_path, monkeypatch)
    _finish_run(workspace)
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    _register(workspace, bundle)
    result = _RUNNER.invoke(
        app,
        [
            "actions",
            "infer",
            "--project",
            _PROJECT,
            "--run",
            "run_a",
            "--action-model",
            _MODEL,
            "--min-confidence",
            "0.6",
            "--min-duration",
            "1",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    job_id = payload["data"]["job_id"]
    request = json.loads(
        (run_directory(workspace, _PROJECT, job_id) / "action_infer_request.json").read_text(
            encoding="utf-8"
        )
    )
    assert request["run_id"] == "run_a"
    assert request["min_confidence"] == 0.6
    assert request["min_duration_s"] == 1.0
    assert run_action_infer_job(workspace, job_id) == 0
    destination = tmp_path / "intervals.csv"
    exported = export_action_intervals(workspace, job_id, destination, excel=True)
    raw = Path(exported["csv"]).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    loaded = pd.read_csv(destination)
    assert "action_name" in loaded.columns
    assert list(loaded.columns) == list(INTERVAL_COLUMNS)


def test_infer_obeys_the_lock_and_cancel_inserts_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _project(tmp_path, monkeypatch)
    _finish_run(workspace)
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    _register(workspace, bundle)
    models = [row["action_model_id"] for row in storage.list_action_models(workspace, _PROJECT)]
    holder = start_job(
        workspace,
        _PROJECT,
        kind="analyze",
        worker="sleep",
        video_duration_s=10.0,
        step_s=1.0,
    )
    with pytest.raises(JobAlreadyRunningError):
        enqueue_action_infer(workspace, _PROJECT, "run_a", _MODEL)
    request_cancel(workspace, _PROJECT, holder.job_id)
    assert get_job(workspace, _PROJECT, holder.job_id).status is JobStatus.cancelled

    started = threading.Event()
    job_id: dict[str, str] = {}

    def hold(bundle_obj: ActionBundle, frame: pd.DataFrame) -> pd.DataFrame:
        started.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            status = get_job(workspace, _PROJECT, job_id["id"]).status
            if status in {JobStatus.cancel_requested, JobStatus.cancelled}:
                return predict_frames(bundle_obj, frame)
            time.sleep(0.02)
        raise AssertionError("cancel did not land")

    monkeypatch.setattr("excavation_lab.actions.infer.predict_frames", hold)
    record = enqueue_action_infer(workspace, _PROJECT, "run_a", _MODEL)
    job_id["id"] = record.job_id
    outcome: dict[str, int] = {}

    def _run() -> None:
        outcome["code"] = run_action_infer_job(workspace, record.job_id)

    thread = threading.Thread(target=_run)
    thread.start()
    assert started.wait(5.0)
    with pytest.raises(JobAlreadyRunningError):
        enqueue_action_infer(workspace, _PROJECT, "run_a", _MODEL)
    request_cancel(workspace, _PROJECT, record.job_id)
    thread.join(10.0)
    assert not thread.is_alive()
    assert outcome["code"] == 0
    assert get_job(workspace, _PROJECT, record.job_id).status is JobStatus.cancelled
    assert storage.list_action_inferences(workspace, _PROJECT) == []
    assert [
        row["action_model_id"] for row in storage.list_action_models(workspace, _PROJECT)
    ] == models
