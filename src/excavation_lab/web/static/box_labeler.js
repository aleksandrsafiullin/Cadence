(function () {
  function clamp(value, limit) {
    return Math.max(0, Math.min(limit, value));
  }

  function roundPx(value) {
    return Math.round(value * 100) / 100;
  }

  function normalizeBox(box) {
    var x1 = Number(box.x1);
    var y1 = Number(box.y1);
    var x2 = Number(box.x2);
    var y2 = Number(box.y2);
    var swap;
    if (x2 < x1) {
      swap = x1;
      x1 = x2;
      x2 = swap;
    }
    if (y2 < y1) {
      swap = y1;
      y1 = y2;
      y2 = swap;
    }
    return {
      class_name: box.class_name,
      visibility: box.visibility || "clear",
      truncated: Boolean(box.truncated),
      instance_local_id: box.instance_local_id || null,
      note: box.note || null,
      x1: x1,
      y1: y1,
      x2: x2,
      y2: y2,
    };
  }

  function storedBoxFromSuggestion(box) {
    var normalized = normalizeBox(box);
    return {
      class_name: normalized.class_name,
      visibility: "clear",
      truncated: false,
      instance_local_id: null,
      note: null,
      x1: normalized.x1,
      y1: normalized.y1,
      x2: normalized.x2,
      y2: normalized.y2,
    };
  }

  // rect is the canvas border box in CSS pixels. Stored boxes stay in image pixels.
  function imagePoint(clientX, clientY, rect, width, height) {
    var x = rect.width ? ((clientX - rect.left) / rect.width) * width : 0;
    var y = rect.height ? ((clientY - rect.top) / rect.height) * height : 0;
    return { x: clamp(x, width), y: clamp(y, height) };
  }

  function needsEmptyConfirm(review, boxCount, storedReviewedEmpty) {
    return review === "reviewed" && boxCount === 0 && !storedReviewedEmpty;
  }

  function moves(handle, edge) {
    if (edge === "n") {
      return handle === "n" || handle === "nw" || handle === "ne";
    }
    if (edge === "s") {
      return handle === "s" || handle === "sw" || handle === "se";
    }
    if (edge === "w") {
      return handle === "w" || handle === "nw" || handle === "sw";
    }
    return handle === "e" || handle === "ne" || handle === "se";
  }

  function applyResize(box, origin, handle, x, y, imageWidth, imageHeight) {
    var x1 = origin.x1;
    var y1 = origin.y1;
    var x2 = origin.x2;
    var y2 = origin.y2;
    if (moves(handle, "w")) {
      x1 = clamp(x, origin.x2 - 2);
    }
    if (moves(handle, "e")) {
      x2 = clamp(Math.max(x, origin.x1 + 2), imageWidth);
    }
    if (moves(handle, "n")) {
      y1 = clamp(y, origin.y2 - 2);
    }
    if (moves(handle, "s")) {
      y2 = clamp(Math.max(y, origin.y1 + 2), imageHeight);
    }
    box.x1 = roundPx(x1);
    box.y1 = roundPx(y1);
    box.x2 = roundPx(x2);
    box.y2 = roundPx(y2);
  }

  function responseIsStale(token, currentToken, startedGeneration, generationNow, startedIndex, indexNow) {
    return (
      token !== currentToken ||
      startedGeneration !== generationNow ||
      startedIndex !== indexNow
    );
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      normalizeBox: normalizeBox,
      storedBoxFromSuggestion: storedBoxFromSuggestion,
      imagePoint: imagePoint,
      needsEmptyConfirm: needsEmptyConfirm,
      applyResize: applyResize,
      responseIsStale: responseIsStale,
    };
  }
  if (typeof document === "undefined") {
    return;
  }

  var node = document.getElementById("label-state");
  if (!node) {
    return;
  }
  var state;
  try {
    state = JSON.parse(node.textContent);
  } catch (error) {
    return;
  }
  var canvas = document.getElementById("box-canvas");
  if (!canvas || !state || state.ok === false || !state.width || !state.height) {
    return;
  }
  var image = document.getElementById("label-frame");
  var classSelect = document.getElementById("box-class");
  var visibilityField = document.getElementById("box-visibility");
  var truncatedInput = document.getElementById("box-truncated");
  var instanceInput = document.getElementById("box-instance");
  var noteInput = document.getElementById("box-note");
  var frameNote = document.getElementById("label-frame-note");
  var deleteButton = document.getElementById("box-delete");
  var saveButton = document.getElementById("label-save");
  var suggestButton = document.getElementById("label-suggest");
  var suggestBundle = document.getElementById("label-suggest-bundle");
  var suggestReason = document.getElementById("label-suggest-reason");
  var suggestList = document.getElementById("label-suggestions");
  var suggestAcceptAll = document.getElementById("label-suggest-accept-all");
  var suggestRejectAll = document.getElementById("label-suggest-reject-all");
  var errorNode = document.getElementById("label-error");
  var statusNode = document.getElementById("label-status");
  var hintNode = document.getElementById("box-class-hint");
  var unlabelableNote = document.getElementById("box-unlabelable-note");
  var classes = state.classes || [];
  var boxes = (state.boxes || []).map(normalizeBox);
  var suggestions = [];
  var suggestWait = false;
  var selected = -1;
  var drag = null;
  var editGeneration = 0;
  var saving = false;
  var handles = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
  var copy = readCopy();
  var filterStatus = document.getElementById("label-filter-status");
  var filterSplit = document.getElementById("label-filter-split");
  var strip = document.getElementById("label-strip");
  var shortcuts = document.getElementById("label-shortcuts");
  var prevLink = document.getElementById("label-prev");
  var nextLink = document.getElementById("label-next");
  var progressNode = document.getElementById("label-progress");
  var frames = [];
  var savedGeneration = 0;
  var saveWait = null;
  var markFlight = null;
  var loadToken = 0;
  var pendingLoads = 0;
  var storedReviewedEmpty = state.review_status === "reviewed" && boxes.length === 0;
  var visibilityOrder = ["clear", "partial", "unlabelable"];

  canvas.tabIndex = 0;

  function readCopy() {
    var node = document.getElementById("label-copy");
    if (!node) {
      return {};
    }
    try {
      return JSON.parse(node.textContent) || {};
    } catch (error) {
      return {};
    }
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function classByName(name) {
    var index;
    for (index = 0; index < classes.length; index += 1) {
      if (classes[index].name === name) {
        return classes[index];
      }
    }
    return null;
  }

  function classTitle(name) {
    var item = classByName(name);
    if (item && item.title) {
      return item.title;
    }
    return name || "";
  }

  function classColor(name) {
    var series = [
      "--pdm-series-1",
      "--pdm-series-2",
      "--pdm-series-3",
      "--pdm-series-4",
      "--pdm-series-5",
      "--pdm-series-6",
    ];
    var item = classByName(name);
    var id = item && typeof item.id === "number" ? item.id : 0;
    var slot = ((id % series.length) + series.length) % series.length;
    return cssVar(series[slot]) || cssVar("--pdm-accent");
  }

  function copyBox(box) {
    return normalizeBox(box);
  }

  function clean(value) {
    if (value == null) {
      return null;
    }
    var text = String(value).trim();
    return text ? text : null;
  }

  function visibilityValue() {
    if (!visibilityField) {
      return "clear";
    }
    var checked = visibilityField.querySelector('input[name="box-visibility"]:checked');
    return checked ? checked.value : "clear";
  }

  function setVisibility(value) {
    if (!visibilityField) {
      return;
    }
    visibilityField.querySelectorAll('input[name="box-visibility"]').forEach(function (input) {
      input.checked = input.value === value;
    });
  }

  function updateHints() {
    var item = classSelect ? classByName(classSelect.value) : null;
    if (hintNode) {
      hintNode.textContent = item && item.box ? item.box : "";
    }
    if (unlabelableNote) {
      unlabelableNote.hidden = visibilityValue() !== "unlabelable";
    }
  }

  function syncForm() {
    var box = selected >= 0 ? boxes[selected] : null;
    if (box) {
      if (classSelect) {
        classSelect.value = box.class_name;
      }
      setVisibility(box.visibility || "clear");
      if (truncatedInput) {
        truncatedInput.checked = Boolean(box.truncated);
      }
      if (instanceInput) {
        instanceInput.value = box.instance_local_id || "";
      }
      if (noteInput) {
        noteInput.value = box.note || "";
      }
    } else {
      if (instanceInput) {
        instanceInput.value = "";
      }
      if (noteInput) {
        noteInput.value = "";
      }
    }
    updateHints();
  }

  function writeSelected() {
    if (selected < 0 || !boxes[selected]) {
      return;
    }
    var box = boxes[selected];
    if (classSelect) {
      box.class_name = classSelect.value;
    }
    box.visibility = visibilityValue();
    box.truncated = Boolean(truncatedInput && truncatedInput.checked);
    box.instance_local_id = instanceInput ? clean(instanceInput.value) : box.instance_local_id;
    box.note = noteInput ? clean(noteInput.value) : box.note;
  }

  function onFormEdit() {
    updateHints();
    if (selected < 0) {
      return;
    }
    editGeneration += 1;
    if (statusNode) {
      statusNode.textContent = "";
    }
    writeSelected();
    draw();
  }

  function eventPoint(event) {
    return imagePoint(
      event.clientX,
      event.clientY,
      canvas.getBoundingClientRect(),
      state.width,
      state.height,
    );
  }

  function viewScale() {
    var rect = canvas.getBoundingClientRect();
    return {
      x: rect.width ? state.width / rect.width : 1,
      y: rect.height ? state.height / rect.height : 1,
    };
  }

  function handlePoint(box, name) {
    var cx = (box.x1 + box.x2) / 2;
    var cy = (box.y1 + box.y2) / 2;
    if (name === "nw") {
      return [box.x1, box.y1];
    }
    if (name === "n") {
      return [cx, box.y1];
    }
    if (name === "ne") {
      return [box.x2, box.y1];
    }
    if (name === "e") {
      return [box.x2, cy];
    }
    if (name === "se") {
      return [box.x2, box.y2];
    }
    if (name === "s") {
      return [cx, box.y2];
    }
    if (name === "sw") {
      return [box.x1, box.y2];
    }
    return [box.x1, cy];
  }

  function hitHandle(box, x, y) {
    var scale = viewScale();
    var reachX = 8 * scale.x;
    var reachY = 8 * scale.y;
    var index;
    for (index = 0; index < handles.length; index += 1) {
      var point = handlePoint(box, handles[index]);
      if (Math.abs(x - point[0]) <= reachX / 2 && Math.abs(y - point[1]) <= reachY / 2) {
        return handles[index];
      }
    }
    return null;
  }

  function hitBox(x, y) {
    var best = -1;
    var bestArea = Infinity;
    boxes.forEach(function (box, index) {
      if (x < box.x1 || y < box.y1 || x > box.x2 || y > box.y2) {
        return;
      }
      var area = (box.x2 - box.x1) * (box.y2 - box.y1);
      if (area < bestArea) {
        best = index;
        bestArea = area;
      }
    });
    return best;
  }

  function applyMove(box, origin, x, y) {
    var width = origin.x2 - origin.x1;
    var height = origin.y2 - origin.y1;
    var x1 = origin.x1 + (x - drag.startX);
    var y1 = origin.y1 + (y - drag.startY);
    x1 = clamp(x1, state.width - width);
    y1 = clamp(y1, state.height - height);
    box.x1 = roundPx(x1);
    box.y1 = roundPx(y1);
    box.x2 = roundPx(x1 + width);
    box.y2 = roundPx(y1 + height);
  }

  function commitDraw() {
    var x1 = Math.min(drag.x1, drag.x2);
    var y1 = Math.min(drag.y1, drag.y2);
    var x2 = Math.max(drag.x1, drag.x2);
    var y2 = Math.max(drag.y1, drag.y2);
    if (x2 - x1 < 2 || y2 - y1 < 2) {
      return;
    }
    boxes.push({
      class_name: classSelect ? classSelect.value : "",
      visibility: visibilityValue(),
      truncated: Boolean(truncatedInput && truncatedInput.checked),
      instance_local_id: instanceInput ? clean(instanceInput.value) : null,
      note: noteInput ? clean(noteInput.value) : null,
      x1: roundPx(x1),
      y1: roundPx(y1),
      x2: roundPx(x2),
      y2: roundPx(y2),
    });
    selected = boxes.length - 1;
    editGeneration += 1;
  }

  function onPointerDown(event) {
    if (event.button !== 0 || drag) {
      return;
    }
    event.preventDefault();
    canvas.focus({ preventScroll: true });
    try {
      canvas.setPointerCapture(event.pointerId);
    } catch (error) {
      // Untrusted pointer events have no active pointer id. Canvas listeners still run.
    }
    var point = eventPoint(event);
    var index = hitBox(point.x, point.y);
    var handle = null;
    if (index === selected && selected >= 0) {
      handle = hitHandle(boxes[selected], point.x, point.y);
    }
    if (!handle && index < 0 && selected >= 0) {
      handle = hitHandle(boxes[selected], point.x, point.y);
    }
    if (handle) {
      drag = {
        mode: "resize",
        pointerId: event.pointerId,
        handle: handle,
        origin: copyBox(boxes[selected]),
      };
      return;
    }
    if (index >= 0) {
      selected = index;
      drag = {
        mode: "move",
        pointerId: event.pointerId,
        startX: point.x,
        startY: point.y,
        origin: copyBox(boxes[index]),
      };
      syncForm();
      draw();
      return;
    }
    selected = -1;
    drag = {
      mode: "draw",
      pointerId: event.pointerId,
      x1: point.x,
      y1: point.y,
      x2: point.x,
      y2: point.y,
    };
    syncForm();
    draw();
  }

  function onPointerMove(event) {
    var point = eventPoint(event);
    if (!drag || event.pointerId !== drag.pointerId) {
      var cursor = "crosshair";
      var over = hitBox(point.x, point.y);
      if (selected >= 0) {
        var handle = hitHandle(boxes[selected], point.x, point.y);
        if (handle && (over < 0 || over === selected)) {
          cursor = cursorFor(handle);
        } else if (over >= 0) {
          cursor = "move";
        }
      } else if (over >= 0) {
        cursor = "move";
      }
      canvas.style.cursor = cursor;
      return;
    }
    if (drag.mode === "draw") {
      drag.x2 = point.x;
      drag.y2 = point.y;
    } else if (drag.mode === "move") {
      applyMove(boxes[selected], drag.origin, point.x, point.y);
      editGeneration += 1;
    } else if (drag.mode === "resize") {
      applyResize(
        boxes[selected],
        drag.origin,
        drag.handle,
        point.x,
        point.y,
        state.width,
        state.height,
      );
      editGeneration += 1;
    }
    draw();
  }

  function cursorFor(handle) {
    if (handle === "n" || handle === "s") {
      return "ns-resize";
    }
    if (handle === "e" || handle === "w") {
      return "ew-resize";
    }
    if (handle === "ne" || handle === "sw") {
      return "nesw-resize";
    }
    return "nwse-resize";
  }

  function finishPointer(event, commit) {
    if (!drag || event.pointerId !== drag.pointerId) {
      return;
    }
    if (commit && drag.mode === "draw") {
      commitDraw();
    }
    drag = null;
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    syncForm();
    draw();
  }

  function removeSelected() {
    if (selected < 0 || selected >= boxes.length) {
      return;
    }
    boxes.splice(selected, 1);
    selected = -1;
    editGeneration += 1;
    if (statusNode) {
      statusNode.textContent = "";
    }
    syncForm();
    draw();
  }

  function typingTarget(target) {
    if (!target || !target.tagName) {
      return false;
    }
    var tag = target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
  }

  function onKey(event) {
    if (event.key === "Escape" && typingTarget(event.target)) {
      event.target.blur();
      event.preventDefault();
      return;
    }
    if (typingTarget(event.target)) {
      return;
    }
    if (event.metaKey || event.ctrlKey || event.altKey) {
      return;
    }
    if (event.key === "Escape") {
      selected = -1;
      syncForm();
      draw();
      return;
    }
    if (event.key === "?" || (event.code === "Slash" && event.shiftKey)) {
      event.preventDefault();
      showShortcuts();
      return;
    }
    if (event.shiftKey) {
      return;
    }
    var classDigit = {
      Digit1: "1",
      Digit2: "2",
      Digit3: "3",
      Digit4: "4",
      Digit5: "5",
      Digit6: "6",
      Digit7: "7",
      Digit8: "8",
      Digit9: "9",
    }[event.code];
    if (!classDigit && event.key >= "1" && event.key <= "9") {
      classDigit = event.key;
    }
    if (classDigit) {
      event.preventDefault();
      selectClass(classDigit);
      return;
    }
    if (event.repeat) {
      return;
    }
    if (event.code === "KeyV" || event.key === "v" || event.key === "V") {
      event.preventDefault();
      cycleVisibility();
      return;
    }
    if (event.code === "KeyT" || event.key === "t" || event.key === "T") {
      event.preventDefault();
      toggleTruncated();
      return;
    }
    if (event.code === "KeyR" || event.key === "r" || event.key === "R") {
      event.preventDefault();
      mark("reviewed", true);
      return;
    }
    if (event.code === "KeyX" || event.key === "x" || event.key === "X") {
      event.preventDefault();
      mark("excluded", true);
      return;
    }
    if (event.code === "KeyU" || event.key === "u" || event.key === "U") {
      event.preventDefault();
      mark("unreviewed", false);
      return;
    }
    if (event.code === "ArrowLeft" || event.key === "ArrowLeft") {
      event.preventDefault();
      go(neighbor(-1));
      return;
    }
    if (event.code === "ArrowRight" || event.key === "ArrowRight") {
      event.preventDefault();
      go(neighbor(1));
      return;
    }
    if ((event.key === "Delete" || event.key === "Backspace") && selected >= 0) {
      event.preventDefault();
      removeSelected();
    }
  }

  function selectClass(key) {
    var index;
    var item = null;
    for (index = 0; index < classes.length; index += 1) {
      if (classes[index].key === key) {
        item = classes[index];
        break;
      }
    }
    if (!item || !classSelect) {
      return;
    }
    classSelect.value = item.name;
    if (selected >= 0 && boxes[selected]) {
      boxes[selected].class_name = item.name;
      editGeneration += 1;
      if (statusNode) {
        statusNode.textContent = "";
      }
    }
    updateHints();
    draw();
  }

  function cycleVisibility() {
    var current = selected >= 0 && boxes[selected] ? boxes[selected].visibility : visibilityValue();
    var at = visibilityOrder.indexOf(current);
    var next = visibilityOrder[(at + 1) % visibilityOrder.length];
    setVisibility(next);
    if (selected >= 0 && boxes[selected]) {
      boxes[selected].visibility = next;
      editGeneration += 1;
      if (statusNode) {
        statusNode.textContent = "";
      }
      draw();
    }
    updateHints();
  }

  function toggleTruncated() {
    var next = !(truncatedInput && truncatedInput.checked);
    if (truncatedInput) {
      truncatedInput.checked = next;
    }
    if (selected >= 0 && boxes[selected]) {
      boxes[selected].truncated = next;
      editGeneration += 1;
      if (statusNode) {
        statusNode.textContent = "";
      }
      draw();
    }
  }

  function showShortcuts() {
    if (!shortcuts) {
      return;
    }
    shortcuts.hidden = false;
    shortcuts.scrollIntoView({ block: "nearest" });
    shortcuts.focus({ preventScroll: true });
  }

  function paintOrder() {
    var order = boxes.map(function (box, index) {
      return { index: index, area: (box.x2 - box.x1) * (box.y2 - box.y1) };
    });
    order.sort(function (left, right) {
      return right.area - left.area;
    });
    if (selected >= 0) {
      order = order.filter(function (item) {
        return item.index !== selected;
      });
      order.push({ index: selected, area: 0 });
    }
    return order;
  }

  function prepare() {
    var dpr = window.devicePixelRatio || 1;
    var nextWidth = Math.max(1, Math.round(state.width * dpr));
    var nextHeight = Math.max(1, Math.round(state.height * dpr));
    if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
      canvas.width = nextWidth;
      canvas.height = nextHeight;
    }
    var ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function strokeBox(ctx, box, color, scale, active) {
    var width = box.x2 - box.x1;
    var height = box.y2 - box.y1;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = (active ? 3 : 2) * scale.x;
    ctx.setLineDash(box.visibility === "partial" ? [6 * scale.x, 4 * scale.x] : []);
    ctx.strokeRect(box.x1, box.y1, width, height);
    if (active) {
      ctx.setLineDash([]);
      ctx.strokeStyle = cssVar("--pdm-text");
      ctx.lineWidth = scale.x;
      ctx.strokeRect(
        box.x1 - 2 * scale.x,
        box.y1 - 2 * scale.y,
        width + 4 * scale.x,
        height + 4 * scale.y,
      );
    }
    ctx.restore();
  }

  function drawLabel(ctx, box, title, color, scale) {
    var family = cssVar("--pdm-font") || "sans-serif";
    var fontPx = 13 * scale.x;
    ctx.save();
    ctx.font = "600 " + fontPx + "px " + family;
    ctx.textBaseline = "top";
    var padX = 4 * scale.x;
    var padY = 2 * scale.y;
    var textWidth = ctx.measureText(title).width;
    var top = box.y1 - fontPx - padY * 2 - 2 * scale.y;
    if (top < 0) {
      top = box.y1 + 2 * scale.y;
    }
    ctx.fillStyle = cssVar("--pdm-surface");
    ctx.globalAlpha = 0.92;
    ctx.fillRect(box.x1, top, textWidth + padX * 2, fontPx + padY * 2);
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.fillText(title, box.x1 + padX, top + padY);
    ctx.restore();
  }

  function drawHandles(ctx, box, color, scale) {
    var reachX = 8 * scale.x;
    var reachY = 8 * scale.y;
    ctx.save();
    ctx.fillStyle = cssVar("--pdm-surface");
    ctx.strokeStyle = color;
    ctx.lineWidth = scale.x;
    handles.forEach(function (name) {
      var point = handlePoint(box, name);
      ctx.fillRect(point[0] - reachX / 2, point[1] - reachY / 2, reachX, reachY);
      ctx.strokeRect(point[0] - reachX / 2, point[1] - reachY / 2, reachX, reachY);
    });
    ctx.restore();
  }

  function drawRubber(ctx, scale) {
    var x1 = Math.min(drag.x1, drag.x2);
    var y1 = Math.min(drag.y1, drag.y2);
    var x2 = Math.max(drag.x1, drag.x2);
    var y2 = Math.max(drag.y1, drag.y2);
    var color = classColor(classSelect ? classSelect.value : "");
    strokeBox(
      ctx,
      { x1: x1, y1: y1, x2: x2, y2: y2, visibility: visibilityValue() },
      color,
      scale,
      true,
    );
  }

  function draw() {
    var ctx = prepare();
    if (image && image.complete && image.naturalWidth) {
      ctx.drawImage(image, 0, 0, state.width, state.height);
    }
    var scale = viewScale();
    paintOrder().forEach(function (item) {
      var box = boxes[item.index];
      var color = classColor(box.class_name);
      var active = item.index === selected;
      strokeBox(ctx, box, color, scale, active);
      drawLabel(ctx, box, classTitle(box.class_name), color, scale);
      if (active) {
        drawHandles(ctx, box, color, scale);
      }
    });
    if (drag && drag.mode === "draw") {
      drawRubber(ctx, scale);
    }
    suggestions.forEach(function (box) {
      strokeSuggestion(ctx, box, scale);
      drawLabel(ctx, box, suggestionTitle(box), suggestionColor(), scale);
    });
  }

  function suggestionColor() {
    return cssVar("--pdm-warning") || "#b86e00";
  }

  function suggestionTitle(box) {
    var title = classTitle(box.class_name);
    var score = Number(box.score);
    if (!Number.isFinite(score)) {
      return title;
    }
    return title + " " + score.toFixed(2);
  }

  function strokeSuggestion(ctx, box, scale) {
    var width = box.x2 - box.x1;
    var height = box.y2 - box.y1;
    ctx.save();
    ctx.strokeStyle = suggestionColor();
    ctx.lineWidth = 2 * scale.x;
    ctx.setLineDash([10 * scale.x, 6 * scale.x]);
    ctx.strokeRect(box.x1, box.y1, width, height);
    ctx.restore();
  }

  function showError(text) {
    if (statusNode) {
      statusNode.textContent = "";
    }
    if (!errorNode) {
      return;
    }
    errorNode.hidden = !text;
    errorNode.textContent = text || "";
  }

  function showSaved() {
    if (errorNode) {
      errorNode.hidden = true;
      errorNode.textContent = "";
    }
    if (statusNode) {
      statusNode.textContent = statusNode.getAttribute("data-saved") || "";
    }
  }

  function failedText() {
    if (errorNode && errorNode.getAttribute("data-failed")) {
      return errorNode.getAttribute("data-failed");
    }
    return "";
  }

  function payloadBoxes() {
    return boxes.filter(function (box) {
      return !box.suggested;
    }).map(function (box) {
      return {
        class_name: box.class_name,
        visibility: box.visibility || "clear",
        truncated: Boolean(box.truncated),
        instance_local_id: box.instance_local_id || null,
        note: box.note || null,
        x1: box.x1,
        y1: box.y1,
        x2: box.x2,
        y2: box.y2,
      };
    });
  }

  function updateSuggestControls(payload) {
    var allowed = Boolean(payload && payload.suggestions_allowed);
    var reason = payload && payload.suggestion_reason ? payload.suggestion_reason : "";
    if (suggestButton) {
      suggestButton.disabled = !allowed || suggestWait;
      if (reason) {
        suggestButton.setAttribute("title", reason);
      } else {
        suggestButton.removeAttribute("title");
      }
    }
    if (suggestBundle) {
      suggestBundle.disabled = !allowed;
      if (!allowed && reason) {
        suggestBundle.setAttribute("title", reason);
      } else {
        suggestBundle.removeAttribute("title");
      }
    }
    if (suggestReason) {
      suggestReason.hidden = !reason;
      suggestReason.textContent = reason;
    }
  }

  function suggestionButtons() {
    var pending = suggestions.length > 0;
    var empty = suggestAcceptAll ? suggestAcceptAll.getAttribute("data-empty") || "" : "";
    if (suggestAcceptAll) {
      suggestAcceptAll.disabled = !pending;
      if (pending) {
        suggestAcceptAll.removeAttribute("title");
      } else if (empty) {
        suggestAcceptAll.setAttribute("title", empty);
      }
    }
    if (suggestRejectAll) {
      suggestRejectAll.disabled = !pending;
      if (pending) {
        suggestRejectAll.removeAttribute("title");
      } else if (empty) {
        suggestRejectAll.setAttribute("title", empty);
      }
    }
    var note = document.getElementById("label-suggest-none");
    if (note) {
      note.hidden = pending;
    }
  }

  function renderSuggestions() {
    suggestionButtons();
    if (!suggestList) {
      return;
    }
    suggestList.replaceChildren();
    suggestions.forEach(function (box, index) {
      var item = document.createElement("li");
      var label = document.createElement("span");
      label.textContent = suggestionTitle(box);
      var accept = document.createElement("button");
      accept.type = "button";
      accept.className = "lab-button";
      accept.textContent = copy.suggest_accept || "";
      accept.addEventListener("click", function () {
        acceptSuggestion(index);
      });
      item.appendChild(label);
      item.appendChild(accept);
      suggestList.appendChild(item);
    });
  }

  function acceptSuggestion(index) {
    var box = suggestions[index];
    if (!box) {
      return;
    }
    suggestions.splice(index, 1);
    boxes.push(storedBoxFromSuggestion(box));
    selected = boxes.length - 1;
    editGeneration += 1;
    if (statusNode) {
      statusNode.textContent = "";
    }
    renderSuggestions();
    syncForm();
    draw();
  }

  function acceptEverySuggestion() {
    if (!suggestions.length) {
      return;
    }
    suggestions.forEach(function (box) {
      boxes.push(storedBoxFromSuggestion(box));
    });
    suggestions = [];
    selected = boxes.length - 1;
    editGeneration += 1;
    if (statusNode) {
      statusNode.textContent = "";
    }
    renderSuggestions();
    syncForm();
    draw();
  }

  function rejectEverySuggestion() {
    if (!suggestions.length) {
      return;
    }
    suggestions = [];
    renderSuggestions();
    draw();
  }

  function suggestUrl() {
    return (
      "/label/suggest?video=" +
      encodeURIComponent(state.video_id) +
      "&frame=" +
      encodeURIComponent(String(state.index))
    );
  }

  function requestSuggestions() {
    if (!suggestButton || suggestButton.disabled || suggestWait) {
      return;
    }
    var modelId = suggestBundle ? suggestBundle.value : "";
    var frameIndex = state.index;
    suggestWait = true;
    suggestButton.disabled = true;
    fetch(suggestUrl(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ model_id: modelId }),
    })
      .then(function (response) {
        return response.json().then(
          function (payload) {
            return { ok: response.ok && payload && payload.ok, payload: payload };
          },
          function () {
            return { ok: false, payload: null };
          },
        );
      })
      .then(function (result) {
        suggestWait = false;
        if (state.index !== frameIndex) {
          updateSuggestControls(state);
          return;
        }
        if (!result.ok) {
          var code = result.payload && result.payload.code;
          if (code === "job_running" || code === "not_train_frame" || code === "bundle_missing") {
            state.suggestions_allowed = false;
            state.suggestion_block = code;
            state.suggestion_reason = (result.payload && result.payload.error) || "";
          }
          updateSuggestControls(state);
          showError((result.payload && result.payload.error) || failedText());
          return;
        }
        suggestions = (result.payload.boxes || []).map(function (box) {
          var stored = normalizeBox(box);
          stored.score = Number(box.score);
          stored.suggested = true;
          return stored;
        });
        renderSuggestions();
        draw();
        updateSuggestControls(state);
        if (statusNode && result.payload.dropped) {
          var template = copy.suggest_dropped || "";
          statusNode.textContent = template.replace("{count}", String(result.payload.dropped));
        }
      })
      .catch(function () {
        suggestWait = false;
        updateSuggestControls(state);
        showError(failedText());
      });
  }

  function reflectReview(status) {
    document.querySelectorAll(".lab-label-status [data-review]").forEach(function (button) {
      if (button.getAttribute("data-review") === status) {
        button.setAttribute("aria-pressed", "true");
      } else {
        button.removeAttribute("aria-pressed");
      }
    });
  }

  function applyPayload(payload) {
    if (!payload || !Array.isArray(payload.boxes)) {
      return;
    }
    boxes = payload.boxes.map(normalizeBox);
    if (typeof payload.review_status === "string") {
      state.review_status = payload.review_status;
      reflectReview(payload.review_status);
    }
    if (frameNote && Object.prototype.hasOwnProperty.call(payload, "note")) {
      frameNote.value = payload.note || "";
    }
    if (selected >= boxes.length) {
      selected = -1;
    }
    storedReviewedEmpty = state.review_status === "reviewed" && boxes.length === 0;
    reflectHeader(payload);
    syncForm();
    draw();
  }

  function dirty() {
    return editGeneration !== savedGeneration;
  }

  function confirmEmpty() {
    var text = copy.confirm_empty || "";
    if (!text) {
      return false;
    }
    return window.confirm(text);
  }

  function save(reviewOverride) {
    if (saveWait) {
      return saveWait.then(function (ok) {
        if (!ok) {
          return false;
        }
        return save(reviewOverride);
      });
    }
    saving = true;
    var generation = editGeneration;
    var review = reviewOverride || state.review_status;
    var frameIndex = state.index;
    var body = {
      review_status: review,
      boxes: payloadBoxes(),
      note: frameNote ? clean(frameNote.value) : null,
    };
    var promise = fetch(frameUrl(frameIndex), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(body),
    })
      .then(function (response) {
        return response.json().then(
          function (payload) {
            return { ok: response.ok && payload && payload.ok, payload: payload };
          },
          function () {
            return { ok: false, payload: null };
          },
        );
      })
      .then(function (result) {
        saving = false;
        saveWait = null;
        if (state.index !== frameIndex) {
          return false;
        }
        if (!result.ok) {
          showError((result.payload && result.payload.error) || failedText());
          return false;
        }
        rememberFrame(result.payload);
        if (result.payload.progress) {
          updateProgress(result.payload.progress);
        }
        if (generation === editGeneration) {
          applyPayload(result.payload);
          savedGeneration = editGeneration;
        }
        showSaved();
        renderStrip();
        paintNav();
        return true;
      })
      .catch(function () {
        saving = false;
        saveWait = null;
        showError(failedText());
        return false;
      });
    saveWait = promise;
    return promise;
  }

  function flush() {
    if (!dirty()) {
      return Promise.resolve(true);
    }
    var generation = editGeneration;
    return save(state.review_status).then(function (ok) {
      if (!ok) {
        return false;
      }
      if (editGeneration !== generation) {
        return flush();
      }
      return true;
    });
  }

  function mark(status, advance) {
    if (markFlight || pendingLoads) {
      return;
    }
    if (needsEmptyConfirm(status, boxes.length, storedReviewedEmpty) && !confirmEmpty()) {
      return;
    }
    var startedIndex = state.index;
    var previousStatus = state.review_status;
    var previousGeneration = editGeneration;
    state.review_status = status;
    reflectReview(status);
    editGeneration += 1;
    if (statusNode) {
      statusNode.textContent = "";
    }
    var flight = save(status).then(function (ok) {
      if (state.index !== startedIndex) {
        return;
      }
      if (!ok) {
        state.review_status = previousStatus;
        reflectReview(previousStatus);
        if (editGeneration === previousGeneration + 1) {
          editGeneration = previousGeneration;
        }
        return;
      }
      if (!advance) {
        return;
      }
      var leave = function () {
        if (state.index !== startedIndex) {
          return;
        }
        var next = neighbor(1);
        if (next != null) {
          loadFrame(next, false);
        }
      };
      if (dirty()) {
        return flush().then(function (again) {
          if (again) {
            leave();
          }
        });
      }
      leave();
    });
    markFlight = flight;
    flight.then(
      function () {
        if (markFlight === flight) {
          markFlight = null;
        }
      },
      function () {
        if (markFlight === flight) {
          markFlight = null;
        }
      },
    );
  }

  function go(index) {
    if (index == null || index === state.index) {
      return;
    }
    if (needsEmptyConfirm(state.review_status, boxes.length, storedReviewedEmpty) && dirty() && !confirmEmpty()) {
      return;
    }
    flush().then(function (ok) {
      if (!ok) {
        return;
      }
      loadFrame(index, false);
    });
  }

  function frameUrl(index) {
    return (
      "/label/frame?video=" +
      encodeURIComponent(state.video_id) +
      "&frame=" +
      encodeURIComponent(String(index))
    );
  }

  function pageUrl(index) {
    return (
      "/label?video=" +
      encodeURIComponent(state.video_id) +
      "&frame=" +
      encodeURIComponent(String(index))
    );
  }

  function loadFrame(index, replace) {
    loadToken += 1;
    pendingLoads += 1;
    var token = loadToken;
    var startedGeneration = editGeneration;
    var startedIndex = state.index;
    return fetch(frameUrl(index), { headers: { Accept: "application/json" } })
      .then(function (response) {
        return response.json().then(
          function (payload) {
            return { ok: response.ok && payload && payload.ok, payload: payload };
          },
          function () {
            return { ok: false, payload: null };
          },
        );
      })
      .then(function (result) {
        if (
          responseIsStale(
            token,
            loadToken,
            startedGeneration,
            editGeneration,
            startedIndex,
            state.index,
          )
        ) {
          return false;
        }
        if (!result.ok) {
          showError((result.payload && result.payload.error) || failedText());
          return false;
        }
        adoptFrame(result.payload, replace);
        return true;
      })
      .catch(function () {
        if (
          responseIsStale(
            token,
            loadToken,
            startedGeneration,
            editGeneration,
            startedIndex,
            state.index,
          )
        ) {
          return false;
        }
        showError(failedText());
        return false;
      })
      .then(function (ok) {
        pendingLoads = Math.max(0, pendingLoads - 1);
        return ok;
      });
  }

  function adoptFrame(payload, replace) {
    drag = null;
    selected = -1;
    state = payload;
    boxes = (payload.boxes || []).map(normalizeBox);
    suggestions = [];
    editGeneration = 0;
    savedGeneration = 0;
    storedReviewedEmpty = state.review_status === "reviewed" && boxes.length === 0;
    if (frameNote) {
      frameNote.value = payload.note || "";
    }
    reflectReview(state.review_status);
    reflectHeader(payload);
    if (payload.progress) {
      updateProgress(payload.progress);
    }
    showImage(payload.image_url);
    if (window.history && history.pushState) {
      var historyUrl = pageUrl(payload.index);
      if (replace) {
        history.replaceState({ frame: payload.index }, "", historyUrl);
      } else if (window.location.pathname + window.location.search !== historyUrl) {
        history.pushState({ frame: payload.index }, "", historyUrl);
      }
    }
    syncForm();
    renderStrip();
    paintNav();
    renderSuggestions();
    updateSuggestControls(payload);
    showError("");
    if (statusNode) {
      statusNode.textContent = "";
    }
  }

  function showImage(url) {
    if (!image) {
      draw();
      return;
    }
    image.alt = document.getElementById("label-video")
      ? document.getElementById("label-video").textContent
      : "";
    image.width = state.width;
    image.height = state.height;
    if (image.getAttribute("src") === url && image.complete && image.naturalWidth) {
      draw();
      return;
    }
    image.onload = function () {
      draw();
    };
    image.src = url;
  }

  function reflectHeader(payload) {
    var timeNode = document.getElementById("label-time");
    var indexNode = document.getElementById("label-index");
    var badge = document.querySelector("#label-split .lab-label-split");
    var meta = document.getElementById("label-reviewed-meta");
    var whenNode = document.getElementById("label-reviewed-at");
    var whoNode = document.getElementById("label-reviewed-annotator");
    var seconds = copy.seconds || "";
    if (timeNode && typeof payload.source_time_s === "number") {
      timeNode.textContent = payload.source_time_s.toFixed(3) + (seconds ? " " + seconds : "");
    }
    if (indexNode) {
      indexNode.textContent = (copy.frame_index || "") + " " + payload.index;
    }
    if (badge) {
      badge.textContent = payload.split_label || "";
      badge.setAttribute("data-split", payload.split || "");
    }
    if (!meta) {
      return;
    }
    if (payload.reviewed_at) {
      meta.hidden = false;
      if (whenNode) {
        whenNode.textContent = payload.reviewed_at;
        whenNode.setAttribute("datetime", payload.reviewed_at);
      }
      if (whoNode) {
        whoNode.textContent = payload.annotator || "";
      }
    } else {
      meta.hidden = true;
    }
  }

  function updateProgress(counts) {
    if (!progressNode || !copy.progress || !counts) {
      return;
    }
    var names = ["train", "validation", "test", "unassigned"];
    var parts = names.map(function (split) {
      var row = counts[split] || { reviewed: 0, unreviewed: 0, excluded: 0, total: 0 };
      var title = copy.splits && copy.splits[split] ? copy.splits[split] : split;
      return copy.progress.replace(/\{(\w+)\}/g, function (_match, key) {
        if (key === "split") {
          return title;
        }
        return row[key];
      });
    });
    progressNode.textContent = parts.join(" ");
  }

  function filteredFrames() {
    var status = filterStatus ? filterStatus.value : "all";
    var split = filterSplit ? filterSplit.value : "all";
    return frames.filter(function (frame) {
      if (status !== "all" && frame.review_status !== status) {
        return false;
      }
      if (split !== "all" && frame.split !== split) {
        return false;
      }
      return true;
    });
  }

  function hrefIndex(node) {
    if (!node || node.getAttribute("aria-disabled") === "true") {
      return null;
    }
    var href = node.getAttribute("href") || "";
    var raw = new URLSearchParams(href.split("?")[1] || "").get("frame");
    if (raw == null || !/^\d+$/.test(raw)) {
      return null;
    }
    return Number(raw);
  }

  function neighbor(direction) {
    if (!frames.length) {
      return hrefIndex(direction < 0 ? prevLink : nextLink);
    }
    var seq = filteredFrames();
    var current = state.index;
    var position = -1;
    var index;
    for (index = 0; index < seq.length; index += 1) {
      if (seq[index].index === current) {
        position = index;
        break;
      }
    }
    if (position >= 0) {
      var nextPos = position + direction;
      if (nextPos < 0 || nextPos >= seq.length) {
        return null;
      }
      return seq[nextPos].index;
    }
    if (direction > 0) {
      for (index = 0; index < seq.length; index += 1) {
        if (seq[index].index > current) {
          return seq[index].index;
        }
      }
      return null;
    }
    for (index = seq.length - 1; index >= 0; index -= 1) {
      if (seq[index].index < current) {
        return seq[index].index;
      }
    }
    return null;
  }

  function paintNav() {
    if (!frames.length) {
      return;
    }
    setNav(prevLink, neighbor(-1));
    setNav(nextLink, neighbor(1));
  }

  function setNav(node, index) {
    if (!node) {
      return;
    }
    if (index == null) {
      node.setAttribute("aria-disabled", "true");
      node.removeAttribute("href");
      return;
    }
    node.removeAttribute("aria-disabled");
    node.setAttribute("href", pageUrl(index));
  }

  function renderStrip() {
    if (!strip || !frames.length) {
      return;
    }
    var listed = filteredFrames();
    strip.replaceChildren();
    listed.forEach(function (frame) {
      var item = document.createElement("li");
      var button = document.createElement("button");
      button.type = "button";
      button.textContent = String(frame.index);
      button.setAttribute("data-index", String(frame.index));
      button.setAttribute("data-review", frame.review_status || "unreviewed");
      button.setAttribute("data-split", frame.split || "");
      if (frame.index === state.index) {
        button.setAttribute("aria-current", "true");
      }
      item.appendChild(button);
      strip.appendChild(item);
    });
  }

  function rememberFrame(payload) {
    var index;
    for (index = 0; index < frames.length; index += 1) {
      if (frames[index].index === payload.index) {
        frames[index].review_status = payload.review_status;
        frames[index].label_state = payload.label_state;
        frames[index].box_count = payload.box_count;
        return;
      }
    }
  }

  function loadFrames() {
    return fetch(
      "/label/frames?video=" + encodeURIComponent(state.video_id),
      { headers: { Accept: "application/json" } },
    )
      .then(function (response) {
        return response.json();
      })
      .then(function (payload) {
        if (!payload || !payload.ok || !Array.isArray(payload.frames)) {
          return;
        }
        frames = payload.frames;
        if (payload.progress) {
          updateProgress(payload.progress);
        }
        renderStrip();
        paintNav();
      })
      .catch(function () {
        return null;
      });
  }

  function onNavClick(event) {
    var node = event.currentTarget;
    if (!node || node.getAttribute("aria-disabled") === "true") {
      event.preventDefault();
      return;
    }
    event.preventDefault();
    var href = node.getAttribute("href") || "";
    var query = href.split("?")[1] || "";
    var params = new URLSearchParams(query);
    var raw = params.get("frame");
    if (raw == null || !/^\d+$/.test(raw)) {
      return;
    }
    go(Number(raw));
  }

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", function (event) {
    finishPointer(event, true);
  });
  canvas.addEventListener("pointercancel", function (event) {
    finishPointer(event, false);
  });
  if (classSelect) {
    classSelect.addEventListener("change", onFormEdit);
  }
  if (visibilityField) {
    visibilityField.addEventListener("change", onFormEdit);
  }
  if (truncatedInput) {
    truncatedInput.addEventListener("change", onFormEdit);
  }
  if (instanceInput) {
    instanceInput.addEventListener("input", onFormEdit);
  }
  if (noteInput) {
    noteInput.addEventListener("input", onFormEdit);
  }
  if (frameNote) {
    frameNote.addEventListener("input", function () {
      editGeneration += 1;
      if (statusNode) {
        statusNode.textContent = "";
      }
    });
  }
  if (deleteButton) {
    deleteButton.addEventListener("click", removeSelected);
  }
  if (saveButton) {
    saveButton.addEventListener("click", function () {
      save(state.review_status);
    });
  }
  if (suggestButton) {
    suggestButton.addEventListener("click", requestSuggestions);
  }
  if (suggestAcceptAll) {
    suggestAcceptAll.addEventListener("click", acceptEverySuggestion);
  }
  if (suggestRejectAll) {
    suggestRejectAll.addEventListener("click", rejectEverySuggestion);
  }
  document.querySelectorAll(".lab-label-status [data-review]").forEach(function (button) {
    button.addEventListener("click", function () {
      var status = button.getAttribute("data-review");
      if (status === "reviewed" || status === "excluded" || status === "unreviewed") {
        mark(status, false);
      }
    });
  });
  if (prevLink) {
    prevLink.addEventListener("click", onNavClick);
  }
  if (nextLink) {
    nextLink.addEventListener("click", onNavClick);
  }
  if (filterStatus) {
    filterStatus.addEventListener("change", function () {
      renderStrip();
      paintNav();
    });
  }
  if (filterSplit) {
    filterSplit.addEventListener("change", function () {
      renderStrip();
      paintNav();
    });
  }
  if (strip) {
    strip.addEventListener("click", function (event) {
      var button = event.target.closest("button");
      if (!button || !strip.contains(button)) {
        return;
      }
      var raw = button.getAttribute("data-index");
      if (raw == null || !/^\d+$/.test(raw)) {
        return;
      }
      go(Number(raw));
    });
  }
  window.addEventListener("popstate", function () {
    var params = new URLSearchParams(window.location.search);
    var raw = params.get("frame");
    if (raw == null || !/^\d+$/.test(raw)) {
      return;
    }
    var index = Number(raw);
    if (index === state.index) {
      return;
    }
    flush().then(function (ok) {
      if (!ok) {
        if (window.history && history.pushState) {
          history.pushState({ frame: state.index }, "", pageUrl(state.index));
        }
        return;
      }
      loadFrame(index, true);
    });
  });
  document.addEventListener("keydown", onKey);
  window.addEventListener("beforeunload", function (event) {
    if (!dirty()) {
      return;
    }
    event.preventDefault();
    event.returnValue = "";
  });
  var backLink = document.getElementById("label-back");
  if (backLink) {
    backLink.addEventListener("click", function (event) {
      if (
        event.button !== 0 ||
        event.metaKey ||
        event.ctrlKey ||
        event.shiftKey ||
        event.altKey
      ) {
        return;
      }
      event.preventDefault();
      flush().then(function (ok) {
        if (!ok) {
          return;
        }
        window.location.assign(backLink.href);
      });
    });
  }
  document
    .querySelectorAll(
      'form[action="/label/annotator"], form[action="/project"], form[action="/theme"], form[action="/locale"]',
    )
    .forEach(function (form) {
      form.addEventListener("submit", function (event) {
        if (form.getAttribute("data-labeler-resume") === "1") {
          form.removeAttribute("data-labeler-resume");
          return;
        }
        event.preventDefault();
        var submitter = event.submitter;
        flush().then(function (ok) {
          if (!ok) {
            return;
          }
          form.setAttribute("data-labeler-resume", "1");
          if (typeof form.requestSubmit === "function") {
            form.requestSubmit(submitter || undefined);
          } else {
            form.submit();
          }
        });
      });
    });
  loadFrames();
  renderSuggestions();
  updateSuggestControls(state);
  if (window.ResizeObserver && canvas.parentElement) {
    new ResizeObserver(function () {
      draw();
    }).observe(canvas.parentElement);
  }
  window.addEventListener("resize", function () {
    draw();
  });
  if (image && !(image.complete && image.naturalWidth)) {
    image.addEventListener("load", function () {
      draw();
    });
  }
  syncForm();
  draw();
})();
