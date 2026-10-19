(function () {
  var node = document.getElementById("block-editor-state");
  if (!node) {
    return;
  }
  var state;
  try {
    state = JSON.parse(node.textContent);
  } catch (error) {
    return;
  }
  var video = document.querySelector("#block-stage video");
  var failed = document.getElementById("playback-failed");
  if (video && failed) {
    video.addEventListener("error", function () {
      failed.hidden = false;
    });
  }
  var track = document.getElementById("block-track");
  var errorNode = document.getElementById("block-editor-error");
  var roleInput = document.getElementById("block-role");
  var startInput = document.getElementById("block-start");
  var endInput = document.getElementById("block-end");
  var addButton = document.getElementById("block-add");
  var saveButton = document.getElementById("block-save");
  if (!track || state.duration_s == null || state.offset_s == null) {
    return;
  }

  // Player seconds. The save route converts each edge onto the source clock.
  var duration = state.duration_s;
  var offset = state.offset_s;
  var labels = {};
  state.roles.forEach(function (role) {
    labels[role.value] = role.label;
  });
  var blocks = (state.blocks || []).map(function (block) {
    return {
      start_s: block.start_s - offset,
      end_s: block.end_s - offset,
      role: block.role,
    };
  });
  var selected = blocks.length ? 0 : -1;
  var drag = null;

  function showError(text) {
    if (!errorNode) {
      return;
    }
    errorNode.hidden = !text;
    errorNode.textContent = text || "";
  }

  function clamp(value) {
    return Math.max(0, Math.min(duration, value));
  }

  function overlaps(items) {
    var ordered = items.slice().sort(function (left, right) {
      return left.start_s - right.start_s || left.end_s - right.end_s;
    });
    var index;
    for (index = 1; index < ordered.length; index += 1) {
      if (ordered[index].start_s < ordered[index - 1].end_s) {
        return true;
      }
    }
    return false;
  }

  function fillInputs(block) {
    if (!block || !roleInput || !startInput || !endInput) {
      return;
    }
    roleInput.value = block.role;
    startInput.value = String(block.start_s);
    endInput.value = String(block.end_s);
  }

  function render() {
    track.replaceChildren();
    var indexed = blocks.map(function (block, index) {
      return {
        start_s: block.start_s,
        end_s: block.end_s,
        role: block.role,
        index: index,
      };
    });
    indexed.sort(function (left, right) {
      return left.start_s - right.start_s || left.end_s - right.end_s;
    });
    var pieces = [];
    var cursor = 0;
    indexed.forEach(function (block) {
      var start = Math.max(block.start_s, 0);
      var end = Math.min(block.end_s, duration);
      if (end <= start) {
        return;
      }
      if (start > cursor) {
        pieces.push({ start: cursor, end: start, unassigned: true });
      }
      pieces.push({
        start: start,
        end: end,
        unassigned: false,
        role: block.role,
        index: block.index,
      });
      cursor = end;
    });
    if (cursor < duration) {
      pieces.push({ start: cursor, end: duration, unassigned: true });
    }
    pieces.forEach(function (piece) {
      var span = document.createElement("span");
      span.className =
        "lab-timeline-block" + (piece.unassigned ? " is-unassigned" : "");
      if (!piece.unassigned && piece.index === selected) {
        span.classList.add("is-selected");
      }
      span.style.left = (piece.start / duration) * 100 + "%";
      span.style.width = ((piece.end - piece.start) / duration) * 100 + "%";
      span.textContent = piece.unassigned
        ? state.unassigned_label
        : labels[piece.role];
      if (!piece.unassigned && !state.frozen) {
        span.addEventListener("pointerdown", function (event) {
          if (event.target.classList.contains("lab-timeline-handle")) {
            return;
          }
          selected = piece.index;
          fillInputs(blocks[selected]);
          render();
        });
        addHandle(span, piece.index, "start");
        addHandle(span, piece.index, "end");
      }
      track.appendChild(span);
    });
  }

  function addHandle(parent, index, edge) {
    var handle = document.createElement("span");
    handle.className = "lab-timeline-handle";
    if (edge === "start") {
      handle.style.left = "0";
    } else {
      handle.style.right = "0";
    }
    handle.addEventListener("pointerdown", function (event) {
      event.preventDefault();
      event.stopPropagation();
      selected = index;
      drag = { index: index, edge: edge };
      fillInputs(blocks[index]);
    });
    parent.appendChild(handle);
  }

  window.addEventListener("pointermove", function (event) {
    if (!drag) {
      return;
    }
    var rect = track.getBoundingClientRect();
    var ratio = rect.width ? (event.clientX - rect.left) / rect.width : 0;
    var seconds = clamp(ratio * duration);
    var block = blocks[drag.index];
    if (drag.edge === "start") {
      block.start_s = Math.min(seconds, block.end_s);
    } else {
      block.end_s = Math.max(seconds, block.start_s);
    }
    fillInputs(block);
    render();
  });

  window.addEventListener("pointerup", function () {
    drag = null;
  });

  function readNumber(input) {
    var value = Number(input.value);
    return Number.isFinite(value) ? value : null;
  }

  function addBlock() {
    var start = readNumber(startInput);
    var end = readNumber(endInput);
    if (start === null || end === null || end <= start) {
      showError(state.save_failed);
      return;
    }
    var next = blocks.concat([
      { start_s: start, end_s: end, role: roleInput.value },
    ]);
    if (overlaps(next)) {
      showError(state.overlap_error);
      return;
    }
    blocks = next;
    selected = blocks.length - 1;
    showError("");
    render();
  }

  function applySelected() {
    if (selected < 0) {
      return;
    }
    var start = readNumber(startInput);
    var end = readNumber(endInput);
    if (start === null || end === null) {
      return;
    }
    blocks[selected].start_s = start;
    blocks[selected].end_s = end;
    blocks[selected].role = roleInput.value;
  }

  function save() {
    applySelected();
    if (overlaps(blocks)) {
      showError(state.overlap_error);
      render();
      return;
    }
    fetch("/data/blocks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_id: state.video_id,
        blocks: blocks.map(function (block) {
          return {
            start_s: block.start_s,
            end_s: block.end_s,
            role: block.role,
          };
        }),
      }),
    })
      .then(function (response) {
        return response.json().then(function (body) {
          return { ok: response.ok && body.ok, body: body };
        });
      })
      .then(function (result) {
        if (!result.ok) {
          showError((result.body && result.body.error) || state.save_failed);
          return;
        }
        window.location.reload();
      })
      .catch(function () {
        showError(state.save_failed);
      });
  }

  if (!state.frozen && addButton) {
    addButton.addEventListener("click", addBlock);
  }
  if (!state.frozen && saveButton) {
    saveButton.addEventListener("click", save);
  }
  if (selected >= 0) {
    fillInputs(blocks[selected]);
  }
  render();
})();
