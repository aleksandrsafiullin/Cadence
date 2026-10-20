"""Label page view-model and JSON routes. Synthetic JPEGs, not pit-video accuracy."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from starlette.testclient import TestClient

from excavation_lab import storage
from excavation_lab.datasets.blocks import RoleBlock
from excavation_lab.datasets.catalog import list_catalogs
from excavation_lab.datasets.coco_import import load_imported_frames
from excavation_lab.datasets.label_store import save_frame_label
from excavation_lab.datasets.sampler import SampleResult, SampleRow
from excavation_lab.datasets.splits import freeze_split, require_frozen, write_blocks
from excavation_lab.schemas import SplitCounts, SplitName
from excavation_lab.storage import connect, project_sqlite
from excavation_lab.ui import label_page
from excavation_lab.ui.copy import catalog
from excavation_lab.ui.locale import write_locale_file
from excavation_lab.web.app import create_app
from excavation_lab.web.preferences import write_project_file

_PROJECT = "pilot"
_VIDEO = "vid"
_BLOCKS = (
    RoleBlock(start_s=0, end_s=60, role="train"),
    RoleBlock(start_s=72, end_s=92, role="validation"),
    RoleBlock(start_s=104, end_s=124, role="test"),
)


def test_ready_page_and_unknown_frame(tmp_path: Path) -> None:
    book = catalog("ru")
    workspace = _lab(tmp_path)
    with TestClient(create_app(workspace)) as client:
        missing = client.get("/label")
        assert missing.status_code == 200
        assert book.label_loading in missing.text
        assert not (workspace / "ui" / "annotator.txt").exists()
        page = client.get("/label", params={"video": _VIDEO, "frame": "10"})
        assert page.status_code == 200
        assert 'id="box-canvas"' in page.text
        assert 'id="label-state"' in page.text
        assert "/static/box_labeler.js" in page.text
        assert book.label_mark_reviewed in page.text
        assert book.label_unreviewed_hint in page.text
        assert book.label_fit_width_note in page.text
        assert 'value="operator"' in page.text
        assert "Ковш" in page.text
        state = _state(page.text)
        assert state["width"] == 32
        assert state["height"] == 24
        assert state["suggestions_allowed"] is False
        assert state["trainable"] is False
        assert state["is_negative"] is False
        assert state["classes"][1]["name"] == "bucket"
        assert state["classes"][0]["key"] == "1"
        assert state["classes"][0]["title"]
        assert state["classes"][0]["box"]
        assert state["image_url"].startswith("/data/frame?")
        assert "suggestions" not in state
        image = client.get(state["image_url"])
        assert image.status_code == 200
        assert image.headers["content-type"].startswith("image/jpeg")
        assert image.content[:2] == b"\xff\xd8"
        escaped = client.get(
            "/data/frame",
            params={"video": _VIDEO, "name": "../sample_manifest.json"},
        )
        assert escaped.status_code == 404
        unknown = client.get("/label", params={"video": _VIDEO, "frame": "999"})
        assert unknown.status_code == 404
        assert book.label_unknown_frame in unknown.text
        traversal = client.get("/label/frame", params={"video": "../secrets", "frame": "0"})
        assert traversal.status_code == 404
        assert traversal.json()["ok"] is False
        assert traversal.json()["code"] == "video_not_found"
    assert not (workspace / "ui" / "annotator.txt").exists()


def test_post_saves_through_label_store_and_get_reflects_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = _lab(tmp_path)
    calls = {"n": 0}
    real = save_frame_label

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(label_page, "save_frame_label", spy)
    body = {
        "review_status": "reviewed",
        "note": "edge",
        "boxes": [
            {
                "class_name": "bucket",
                "x1": 1,
                "y1": 1,
                "x2": 34,
                "y2": 26,
                "visibility": "partial",
                "truncated": True,
                "instance_local_id": "b1",
                "note": "teeth",
            }
        ],
    }
    with TestClient(create_app(workspace)) as client:
        saved = client.post("/label/frame", params={"video": _VIDEO, "frame": "10"}, json=body)
        assert saved.status_code == 200
        payload = saved.json()
        assert payload["ok"] is True
        assert payload["boxes"][0]["class_name"] == "bucket"
        assert payload["boxes"][0]["x2"] == 32
        assert payload["boxes"][0]["y2"] == 24
        assert payload["boxes"][0]["visibility"] == "partial"
        assert payload["boxes"][0]["truncated"] is True
        assert payload["annotator"] == "operator"
        assert payload["trainable"] is True
        assert payload["is_negative"] is False
        again = client.get("/label/frame", params={"video": _VIDEO, "frame": "10"})
        assert again.status_code == 200
        assert again.json()["boxes"] == payload["boxes"]
        assert again.json()["note"] == "edge"
        listed = client.get("/label/frames", params={"video": _VIDEO})
        assert listed.status_code == 200
        frames = {item["index"]: item for item in listed.json()["frames"]}
        assert frames[10]["box_count"] == 1
        assert frames[10]["review_status"] == "reviewed"
        progress = listed.json()["progress"]
        assert progress["train"] == {
            "reviewed": 1,
            "unreviewed": 1,
            "excluded": 0,
            "total": 2,
        }
        assert progress["validation"]["total"] == 1
        assert progress["validation"]["unreviewed"] == 1
        assert progress["test"]["total"] == 0
    assert calls["n"] == 1
    loaded = load_imported_frames(workspace, _PROJECT)
    stored = {frame.source_frame_index: frame for frame in loaded}
    assert stored[10].record.boxes[0].x2 == 32
    assert stored[10].record.boxes[0].class_name == "bucket"


def test_unreviewed_post_is_not_a_negative(tmp_path: Path) -> None:
    workspace = _lab(tmp_path)
    body = {
        "review_status": "unreviewed",
        "boxes": [{"class_name": "bucket", "x1": 1, "y1": 1, "x2": 8, "y2": 6}],
    }
    with TestClient(create_app(workspace)) as client:
        saved = client.post("/label/frame", params={"video": _VIDEO, "frame": "10"}, json=body)
        assert saved.status_code == 200
        payload = saved.json()
        assert payload["review_status"] == "unreviewed"
        assert payload["label_state"] == "unreviewed"
        assert payload["trainable"] is False
        assert payload["is_negative"] is False
        assert payload["reviewed_at"] is None
    loaded = load_imported_frames(workspace, _PROJECT)
    record = loaded[0].record
    assert record.trainable is False
    assert record.is_negative is False


def test_validation_frame_disallows_suggestions(tmp_path: Path) -> None:
    workspace = _lab(tmp_path)
    _plant_bundle(workspace)
    with TestClient(create_app(workspace)) as client:
        train = client.get("/label/frame", params={"video": _VIDEO, "frame": "10"})
        validation = client.get("/label/frame", params={"video": _VIDEO, "frame": "80"})
    assert train.status_code == 200
    assert validation.status_code == 200
    assert train.json()["split"] == "train"
    assert train.json()["suggestions_allowed"] is True
    assert validation.json()["split"] == "validation"
    assert validation.json()["suggestions_allowed"] is False
    assert "suggestions" not in train.json()
    assert "suggestions" not in validation.json()


def test_annotator_persists_across_requests(tmp_path: Path) -> None:
    workspace = _lab(tmp_path)
    with TestClient(create_app(workspace)) as client:
        first = client.post(
            "/label/annotator",
            data={"annotator": "  mina  ", "video": _VIDEO, "frame": "10"},
        )
        assert first.status_code == 200
        assert 'value="mina"' in first.text
        second = client.get("/label/frame", params={"video": _VIDEO, "frame": "10"})
        assert second.json()["annotator_name"] == "mina"
        saved = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "11"},
            json={"review_status": "reviewed", "boxes": []},
        )
        assert saved.status_code == 200
        assert saved.json()["annotator"] == "mina"
        assert saved.json()["is_negative"] is True
        assert saved.json()["trainable"] is True
    assert (workspace / "ui" / "annotator.txt").read_text(encoding="utf-8").strip() == "mina"


def test_invalid_box_and_unknown_class_do_not_write(tmp_path: Path) -> None:
    book = catalog("en")
    workspace = _lab(tmp_path)
    write_locale_file(workspace, "en")
    with TestClient(create_app(workspace)) as client:
        flat = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "10"},
            json={
                "review_status": "reviewed",
                "boxes": [{"class_name": "bucket", "x1": 0, "y1": 0, "x2": 0, "y2": 10}],
            },
        )
        unknown = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "10"},
            json={
                "review_status": "reviewed",
                "boxes": [{"class_name": "dump", "x1": 1, "y1": 1, "x2": 4, "y2": 4}],
            },
        )
        page = client.get("/label", params={"video": _VIDEO, "frame": "10"})
    assert flat.status_code == 400
    assert flat.json() == {"ok": False, "error": book.label_invalid_box, "code": "invalid_box"}
    assert unknown.status_code == 400
    assert unknown.json()["code"] == "unknown_category"
    assert unknown.json()["error"] == book.label_unknown_class
    assert book.label_mark_reviewed in page.text
    assert catalog("ru").label_mark_reviewed not in page.text
    assert load_imported_frames(workspace, _PROJECT) == []


def test_mostly_outside_box_is_refused_and_slight_overflow_clamps(tmp_path: Path) -> None:
    book = catalog("ru")
    workspace = _lab(tmp_path)
    with TestClient(create_app(workspace)) as client:
        outside = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "10"},
            json={
                "review_status": "reviewed",
                "boxes": [{"class_name": "bucket", "x1": 1, "y1": 1, "x2": 80, "y2": 40}],
            },
        )
        inverted = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "10"},
            json={
                "review_status": "reviewed",
                "boxes": [{"class_name": "bucket", "x1": 10, "y1": 1, "x2": 5, "y2": 8}],
            },
        )
        slight = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "10"},
            json={
                "review_status": "reviewed",
                "boxes": [{"class_name": "bucket", "x1": 1, "y1": 1, "x2": 34, "y2": 26}],
            },
        )
        half = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "11"},
            json={
                "review_status": "reviewed",
                "boxes": [{"class_name": "bucket", "x1": 0, "y1": 0, "x2": 64, "y2": 10}],
            },
        )
    assert outside.status_code == 400
    assert outside.json() == {"ok": False, "error": book.label_invalid_box, "code": "invalid_box"}
    assert inverted.status_code == 400
    assert inverted.json()["code"] == "invalid_box"
    assert slight.status_code == 200
    assert slight.json()["boxes"][0]["x2"] == 32
    assert slight.json()["boxes"][0]["y2"] == 24
    assert half.status_code == 200
    assert half.json()["boxes"][0]["x1"] == 0
    assert half.json()["boxes"][0]["x2"] == 32
    stored = load_imported_frames(workspace, _PROJECT)
    assert {item.source_frame_index for item in stored} == {10, 11}


def test_more_than_200_boxes_is_invalid_label(tmp_path: Path) -> None:
    book = catalog("ru")
    workspace = _lab(tmp_path)
    boxes = [{"class_name": "bucket", "x1": 0, "y1": 0, "x2": 0, "y2": 1}] * 201
    with TestClient(create_app(workspace)) as client:
        response = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "10"},
            json={"review_status": "reviewed", "boxes": boxes},
        )
    assert response.status_code == 400
    assert response.json() == {
        "ok": False,
        "error": book.label_invalid_json,
        "code": "invalid_label",
    }
    assert load_imported_frames(workspace, _PROJECT) == []


def test_frame_post_requires_json_and_annotator_rejects_cross_site(tmp_path: Path) -> None:
    book = catalog("ru")
    english = catalog("en")
    workspace = _lab(tmp_path)
    body = b'{"review_status":"reviewed","boxes":[]}'
    with TestClient(create_app(workspace)) as client:
        plain = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "10"},
            content=body,
            headers={"content-type": "text/plain"},
        )
        charset = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "11"},
            content=body,
            headers={"content-type": "application/json; charset=utf-8"},
        )
        crossed = client.post(
            "/label/annotator",
            data={"annotator": "mina", "video": _VIDEO, "frame": "10"},
            headers={"sec-fetch-site": "cross-site"},
        )
        mixed = client.post(
            "/label/annotator",
            data={"annotator": "mina", "video": _VIDEO, "frame": "10"},
            headers={"sec-fetch-site": "Cross-Site"},
        )
        write_locale_file(workspace, "en")
        english_plain = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "10"},
            content=body,
            headers={"content-type": "text/plain"},
        )
        kept = client.post(
            "/label/annotator",
            data={"annotator": "mina", "video": _VIDEO, "frame": "10"},
            headers={"sec-fetch-site": "same-origin"},
        )
    assert plain.status_code == 415
    assert plain.json() == {
        "ok": False,
        "error": book.label_unsupported_media_type,
        "code": "unsupported_media_type",
    }
    assert english_plain.status_code == 415
    assert english_plain.json()["error"] == english.label_unsupported_media_type
    assert english_plain.json()["code"] == "unsupported_media_type"
    assert charset.status_code == 200
    assert crossed.status_code == 403
    assert crossed.json() == {"ok": False, "error": book.label_cross_site, "code": "cross_site"}
    assert mixed.status_code == 403
    assert mixed.json()["code"] == "cross_site"
    assert kept.status_code == 200
    assert (workspace / "ui" / "annotator.txt").read_text(encoding="utf-8").strip() == "mina"


def test_read_annotator_stops_at_82_bytes(tmp_path: Path) -> None:
    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "annotator.txt").write_bytes(b"mina" + b" " * 78 + b"!" * 5000)
    assert label_page.annotator_name(tmp_path) == "mina"


def test_progress_keeps_excluded_out_of_reviewed(tmp_path: Path) -> None:
    book = catalog("ru")
    english = catalog("en")
    workspace = _lab(tmp_path)
    box = {"class_name": "bucket", "x1": 1, "y1": 1, "x2": 8, "y2": 6}
    with TestClient(create_app(workspace)) as client:
        empty = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "10"},
            json={"review_status": "reviewed", "boxes": []},
        )
        held = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "11"},
            json={"review_status": "unreviewed", "boxes": [box]},
        )
        skipped = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "80"},
            json={"review_status": "excluded", "boxes": [box]},
        )
        listed = client.get("/label/frames", params={"video": _VIDEO})
        page = client.get("/label", params={"video": _VIDEO, "frame": "10"})
        write_locale_file(workspace, "en")
        english_page = client.get("/label", params={"video": _VIDEO, "frame": "11"})
    assert empty.status_code == 200
    assert empty.json()["label_state"] == "reviewed_empty"
    assert empty.json()["is_negative"] is True
    assert empty.json()["review_status"] == "reviewed"
    assert held.status_code == 200
    assert held.json()["review_status"] == "unreviewed"
    assert held.json()["trainable"] is False
    assert held.json()["is_negative"] is False
    assert skipped.status_code == 200
    assert skipped.json()["review_status"] == "excluded"
    counts = label_page.progress(workspace, _PROJECT, _VIDEO)
    assert counts == listed.json()["progress"]
    assert counts["train"] == {
        "reviewed": 1,
        "unreviewed": 1,
        "excluded": 0,
        "total": 2,
    }
    assert counts["validation"] == {
        "reviewed": 0,
        "unreviewed": 0,
        "excluded": 1,
        "total": 1,
    }
    assert counts["test"] == {
        "reviewed": 0,
        "unreviewed": 0,
        "excluded": 0,
        "total": 0,
    }
    assert sum(row["total"] for row in counts.values()) == 3
    assert page.status_code == 200
    assert "10.000" in page.text
    assert book.label_frame_index in page.text
    assert book.label_confirm_empty in page.text
    assert book.label_excluded_hint in page.text
    assert book.label_filter_unreviewed in page.text
    assert book.label_filter_reviewed in page.text
    assert book.label_filter_excluded in page.text
    assert book.label_shortcut_reviewed in page.text
    assert 'id="label-filter-status"' in page.text
    assert 'id="label-filter-split"' in page.text
    assert 'id="label-shortcuts"' in page.text
    assert 'id="label-strip"' in page.text
    assert 'id="label-mark-reviewed"' in page.text
    assert "lab-button-primary" in page.text
    assert "lab-button-secondary" in page.text
    assert f'href="/data?video={_VIDEO}#step-label"' in page.text
    assert 'id="label-reviewed-at"' in page.text
    assert empty.json()["reviewed_at"] in page.text
    assert 'id="label-reviewed-annotator"' in page.text
    assert english_page.status_code == 200
    assert english.label_confirm_empty in english_page.text
    assert english.label_shortcut_reviewed in english_page.text
    assert english.label_filter_unreviewed in english_page.text
    assert book.label_confirm_empty not in english_page.text
    assert "12.000" in english_page.text


def test_unassigned_frames_count_toward_the_displayed_total() -> None:
    book = catalog("ru")
    english = catalog("en")
    frames = [
        label_page.LabelFrame(0, 1.0, "train", "/a", "reviewed", "reviewed", 1),
        label_page.LabelFrame(1, 65.0, None, "/b", "unreviewed", "unreviewed", 0),
        label_page.LabelFrame(2, 66.0, None, "/c", "excluded", "excluded", 0),
    ]
    counts = label_page._progress_from(frames)
    assert counts["train"]["total"] == 1
    assert counts["train"]["reviewed"] == 1
    assert counts["unassigned"] == {
        "reviewed": 0,
        "unreviewed": 1,
        "excluded": 1,
        "total": 2,
    }
    assert sum(row["total"] for row in counts.values()) == len(frames)
    text = label_page.format_progress(counts, book)
    assert book.block_unassigned in text
    assert "из 2" in text
    english_text = label_page.format_progress(counts, english)
    assert english.block_unassigned in english_text
    assert "of 2" in english_text
    assert book.block_unassigned not in english_text


def test_script_json_keeps_markup_inside_the_script_block(tmp_path: Path) -> None:
    note = "</script><!--"
    raw = label_page.script_json({"note": note, "boxes": []})
    assert "</script>" not in raw
    assert "<!--" not in raw
    assert json.loads(raw)["note"] == note
    workspace = _lab(tmp_path)
    with TestClient(create_app(workspace)) as client:
        saved = client.post(
            "/label/frame",
            params={"video": _VIDEO, "frame": "10"},
            json={"review_status": "unreviewed", "note": note, "boxes": []},
        )
        page = client.get("/label", params={"video": _VIDEO, "frame": "10"})
    assert saved.status_code == 200
    assert saved.json()["note"] == note
    marker = '<script type="application/json" id="label-state">'
    start = page.text.index(marker) + len(marker)
    end = page.text.index("</script>", start)
    body = page.text[start:end]
    assert "</script>" not in body
    assert "<!--" not in body
    assert json.loads(body)["note"] == note
    assert 'aria-label="Статус проверки"' in page.text


def test_unfrozen_video_is_refused(tmp_path: Path) -> None:
    book = catalog("ru")
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project(_PROJECT, workspace, catalog_id="excavation")
    write_project_file(workspace, _PROJECT)
    clip = workspace / "clip.mp4"
    clip.write_bytes(b"synthetic-not-a-video")
    storage.insert_video(workspace, _PROJECT, _VIDEO, path=clip, name="clip.mp4")
    with TestClient(create_app(workspace)) as client:
        refused = client.get("/label/frame", params={"video": _VIDEO, "frame": "10"})
    assert refused.status_code == 400
    assert refused.json()["code"] == "split_not_frozen"
    assert refused.json()["error"] == book.label_not_frozen


def _lab(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    list_catalogs(workspace)
    storage.create_project(_PROJECT, workspace, catalog_id="excavation")
    write_project_file(workspace, _PROJECT)
    clip = workspace / "clip.mp4"
    clip.write_bytes(b"synthetic-not-a-video")
    storage.insert_video(workspace, _PROJECT, _VIDEO, path=clip, name="clip.mp4")
    freeze_split(
        workspace,
        _PROJECT,
        write_blocks(workspace, _PROJECT, _VIDEO, list(_BLOCKS)),
    )
    frozen = require_frozen(workspace, _PROJECT, _VIDEO)
    payload = _jpeg(32, 24, 40)
    _plant_sample(
        workspace,
        frozen.content_hash,
        [
            (10, 10.0, payload, SplitName.train),
            (11, 12.0, payload, SplitName.train),
            (80, 80.0, payload, SplitName.validation),
        ],
    )
    return workspace


def _plant_sample(
    workspace: Path,
    content_hash: str,
    specs: list[tuple[int, float, bytes, SplitName]],
) -> None:
    root = storage.project_directory(workspace, _PROJECT) / "samples" / _VIDEO
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows: list[SampleRow] = []
    for index, time_s, payload, split in specs:
        image_file = f"{_VIDEO}_{index}.jpg"
        (images / image_file).write_bytes(payload)
        rows.append(
            SampleRow(
                video_id=_VIDEO,
                source_frame_index=index,
                source_time_s=time_s,
                episode_id="",
                split=split,
                image_sha256=hashlib.sha256(payload).hexdigest(),
                image_file=image_file,
                strategy="spaced",
                source_pts=index * 1000,
                time_base="1/30000",
            )
        )
    manifest = root / "sample_manifest.json"
    counts = {name.value: 0 for name in SplitName}
    for _index, _time_s, _payload, split in specs:
        counts[split.value] += 1
    requested = SplitCounts(
        train=counts["train"],
        validation=counts["validation"],
        test=counts["test"],
    )
    result = SampleResult(
        project_id=_PROJECT,
        video_id=_VIDEO,
        split_content_hash=content_hash,
        spacing_s=2.0,
        requested=requested,
        written=requested,
        shortfall=SplitCounts(train=0, validation=0, test=0),
        rows=rows,
        manifest_path=str(manifest),
        images_dir=str(images),
        sample_schema_version=2,
    )
    manifest.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _plant_bundle(workspace: Path) -> None:
    bundle = workspace / "weights" / "pit-model"
    bundle.mkdir(parents=True)
    (bundle / "weights.pth").write_bytes(b"not-loaded")
    with connect(project_sqlite(workspace, _PROJECT)) as connection:
        connection.execute(
            """
            INSERT INTO model_versions (
                model_id, project_id, dataset_id, variant, path, created_at
            ) VALUES ('pit-model', ?, NULL, 'nano', ?, ?)
            """,
            (_PROJECT, str(bundle.resolve()), datetime.now(UTC).isoformat()),
        )


def _jpeg(width: int, height: int, tone: int) -> bytes:
    image = np.full((height, width, 3), tone, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return encoded.tobytes()


def _state(html: str) -> dict[str, object]:
    marker = '<script type="application/json" id="label-state">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    payload = json.loads(html[start:end])
    assert isinstance(payload, dict)
    return payload
