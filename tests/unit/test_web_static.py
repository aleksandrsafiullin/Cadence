"""Static box-labeler script. Synthetic sampled frames, not pit-video accuracy."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from excavation_lab.ui.copy import catalog
from excavation_lab.web.app import STATIC_DIR, create_app


def _label_helpers():
    path = Path(__file__).with_name("test_label_page.py")
    spec = importlib.util.spec_from_file_location("label_page_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("label page helpers are missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_HOOKS = (
    'getElementById("box-canvas")',
    'getElementById("box-class")',
    'getElementById("box-visibility")',
    'getElementById("box-truncated")',
    'getElementById("box-instance")',
    'getElementById("box-note")',
    'getElementById("box-delete")',
    'getElementById("label-save")',
    'getElementById("label-suggest")',
    "/label/suggest",
    "storedBoxFromSuggestion",
    "setLineDash",
    'getElementById("label-error")',
    'getElementById("label-status")',
    "pointerdown",
    "setPointerCapture",
    "getBoundingClientRect",
    "devicePixelRatio",
    "/label/frame",
    "application/json",
    "Delete",
    "Backspace",
    "Escape",
    "ArrowLeft",
    "ArrowRight",
    "Digit1",
    "KeyR",
    "KeyX",
    "KeyU",
    "KeyV",
    "KeyT",
    "label-filter-status",
    "label-filter-split",
    "label-strip",
    "label-shortcuts",
    "confirm_empty",
    "/label/frames",
    "beforeunload",
    "responseIsStale",
)
_JS_TYPES = {"text/javascript", "application/javascript"}


def test_box_labeler_script_is_served(tmp_path) -> None:
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/static/box_labeler.js")
    assert response.status_code == 200
    media = response.headers["content-type"].split(";", 1)[0].strip().lower()
    assert media in _JS_TYPES
    text = response.text
    assert text == (STATIC_DIR / "box_labeler.js").read_text(encoding="utf-8")
    for hook in _HOOKS:
        assert hook in text


def test_label_page_references_script_and_state(tmp_path) -> None:
    book = catalog("ru")
    template = (STATIC_DIR.parent / "templates" / "label.html").read_text(encoding="utf-8")
    assert 'src="/static/box_labeler.js"' in template
    assert 'id="box-class-hint"' in template
    assert 'id="box-unlabelable-note"' in template
    helpers = _label_helpers()
    workspace = helpers._lab(tmp_path)
    with TestClient(create_app(workspace)) as client:
        page = client.get("/label", params={"video": "vid", "frame": "10"})
    assert page.status_code == 200
    assert 'src="/static/box_labeler.js"' in page.text
    assert book.label_unlabelable_excludes in page.text
    assert book.label_fit_width_note in page.text
    state = helpers._state(page.text)
    assert state["width"] == 32
    assert state["height"] == 24
    assert str(state["image_url"]).startswith("/data/frame?")
    classes = state["classes"]
    assert isinstance(classes, list) and classes
    for item in classes:
        assert item["title"]
        assert item["box"]


_BEHAVIOUR = r"""
const assert = require("assert");
const api = require(process.argv[1]);
const swapped = api.normalizeBox({
  class_name: "bucket",
  x1: 8,
  y1: 6,
  x2: 2,
  y2: 1,
  visibility: "",
  truncated: 1,
  instance_local_id: "",
  note: "",
});
assert.strictEqual(swapped.x1, 2);
assert.strictEqual(swapped.y1, 1);
assert.strictEqual(swapped.x2, 8);
assert.strictEqual(swapped.y2, 6);
assert.strictEqual(swapped.visibility, "clear");
assert.strictEqual(swapped.truncated, true);
assert.strictEqual(swapped.instance_local_id, null);
assert.strictEqual(swapped.note, null);

const accepted = api.storedBoxFromSuggestion({
  class_name: "bucket",
  score: 0.91,
  suggested: true,
  x1: 1,
  y1: 2,
  x2: 4,
  y2: 6,
});
assert.strictEqual(accepted.class_name, "bucket");
assert.strictEqual(accepted.visibility, "clear");
assert.strictEqual(accepted.truncated, false);
assert.strictEqual(accepted.score, undefined);
assert.strictEqual(accepted.suggested, undefined);
assert.strictEqual(accepted.x1, 1);
assert.strictEqual(accepted.y2, 6);

const mid = api.imagePoint(50, 25, {left: 0, top: 0, width: 100, height: 50}, 200, 100);
assert.strictEqual(mid.x, 100);
assert.strictEqual(mid.y, 50);
const outside = api.imagePoint(-10, 999, {left: 0, top: 0, width: 100, height: 50}, 200, 100);
assert.strictEqual(outside.x, 0);
assert.strictEqual(outside.y, 100);

assert.strictEqual(api.needsEmptyConfirm("reviewed", 0, false), true);
assert.strictEqual(api.needsEmptyConfirm("reviewed", 0, true), false);
assert.strictEqual(api.needsEmptyConfirm("unreviewed", 0, false), false);
assert.strictEqual(api.needsEmptyConfirm("reviewed", 2, false), false);

const grown = {x1: 10, y1: 10, x2: 20, y2: 20};
api.applyResize(grown, {x1: 10, y1: 10, x2: 20, y2: 20}, "e", 90, 20, 40, 40);
assert.strictEqual(grown.x1, 10);
assert.strictEqual(grown.x2, 40);
const shrunk = {x1: 10, y1: 10, x2: 30, y2: 20};
api.applyResize(shrunk, {x1: 10, y1: 10, x2: 30, y2: 20}, "w", 100, 10, 40, 40);
assert.strictEqual(shrunk.x2 - shrunk.x1, 2);

assert.strictEqual(api.responseIsStale(1, 2, 0, 0, 0, 0), true);
assert.strictEqual(api.responseIsStale(1, 1, 0, 3, 0, 0), true);
assert.strictEqual(api.responseIsStale(1, 1, 0, 0, 4, 5), true);
assert.strictEqual(api.responseIsStale(1, 1, 0, 0, 4, 4), false);
"""


def test_box_labeler_pure_functions() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available; box labeler behaviour check skipped")
    completed = subprocess.run(
        [node, "-e", _BEHAVIOUR, str(STATIC_DIR / "box_labeler.js")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_box_labeler_syntax() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available; JS syntax check skipped")
    completed = subprocess.run(
        [node, "--check", str(STATIC_DIR / "box_labeler.js")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
