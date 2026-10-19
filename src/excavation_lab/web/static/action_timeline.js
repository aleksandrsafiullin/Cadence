(function () {
  var catalogError = document.getElementById("action-catalog-error");

  function showCatalogError(text) {
    if (!catalogError) {
      return;
    }
    catalogError.hidden = !text;
    catalogError.textContent = text || "";
  }

  function postJson(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(function (response) {
      return response.json().then(function (payload) {
        return {
          ok: response.ok && payload.ok,
          status: response.status,
          body: payload,
        };
      });
    });
  }

  function catalogFailed(result) {
    var text = (result.body && result.body.error) || "";
    showCatalogError(text);
  }

  var seedButton = document.getElementById("action-seed");
  if (seedButton) {
    seedButton.addEventListener("click", function () {
      postJson("/actions/seed", {})
        .then(function (result) {
          if (!result.ok) {
            catalogFailed(result);
            return;
          }
          window.location.reload();
        })
        .catch(function () {
          showCatalogError("");
        });
    });
  }

  var addClass = document.getElementById("action-add-class");
  if (addClass) {
    addClass.addEventListener("click", function () {
      var name = document.getElementById("action-new-name");
      var title = document.getElementById("action-new-title");
      postJson("/actions/class", {
        name: name ? name.value : "",
        title: title ? title.value : "",
      })
        .then(function (result) {
          if (!result.ok) {
            catalogFailed(result);
            return;
          }
          window.location.reload();
        })
        .catch(function () {
          showCatalogError("");
        });
    });
  }

  Array.prototype.forEach.call(
    document.querySelectorAll(".action-title-save"),
    function (button) {
      button.addEventListener("click", function () {
        var name = button.getAttribute("data-name");
        var input = document.querySelector(
          '.action-title-edit[data-name="' + name + '"]'
        );
        postJson("/actions/class", {
          name: name,
          title: input ? input.value : "",
          edit: true,
        })
          .then(function (result) {
            if (!result.ok) {
              catalogFailed(result);
              return;
            }
            window.location.reload();
          })
          .catch(function () {
            showCatalogError("");
          });
      });
    }
  );

  var node = document.getElementById("action-timeline-state");
  if (!node) {
    return;
  }
  var state;
  try {
    state = JSON.parse(node.textContent);
  } catch (error) {
    return;
  }
  var video = document.getElementById("action-player");
  var failed = document.getElementById("action-playback-failed");
  if (video && failed) {
    video.addEventListener("error", function () {
      failed.hidden = false;
    });
  }
  var track = document.getElementById("action-track");
  var errorNode = document.getElementById("action-error");
  var reloadButton = document.getElementById("action-reload");
  var actionInput = document.getElementById("action-select");
  var startInput = document.getElementById("action-start");
  var endInput = document.getElementById("action-end");
  var tableBody = document.querySelector("#action-table tbody");
  var coverage = document.getElementById("action-coverage");
  if (!track || state.duration_s == null || state.offset_s == null) {
    return;
  }

  var duration = state.duration_s;
  var offset = state.offset_s;
  var strings = state.strings || {};
  var blocks = state.blocks || [];
  var intervals = (state.intervals || []).map(function (item) {
    return {
      label_id: item.label_id,
      player_start_s: item.player_start_s,
      player_end_s: item.player_end_s,
      start_s: item.start_s,
      end_s: item.end_s,
      action_name: item.action_name,
      title: item.title,
      annotator: item.annotator,
      created_at: item.created_at,
    };
  });
  var selected = intervals.length ? 0 : -1;
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

  function secondsAt(event) {
    var rect = track.getBoundingClientRect();
    var ratio = rect.width ? (event.clientX - rect.left) / rect.width : 0;
    return clamp(ratio * duration);
  }

  function titleFor(name) {
    var found = (state.classes || []).filter(function (item) {
      return item.name === name;
    })[0];
    return found ? found.title : name;
  }

  function overlaps(items) {
    var ordered = items.slice().sort(function (left, right) {
      return left.player_start_s - right.player_start_s;
    });
    var index;
    for (index = 1; index < ordered.length; index += 1) {
      if (ordered[index].player_start_s < ordered[index - 1].player_end_s) {
        return true;
      }
    }
    return false;
  }

  function placement(start, end) {
    var containing = blocks.filter(function (block) {
      return (
        block.player_start_s != null &&
        block.player_start_s <= start &&
        end <= block.player_end_s
      );
    });
    if (containing.length === 1) {
      return { ok: true, role: containing[0].role, label: containing[0].label };
    }
    var overlapping = blocks.filter(function (block) {
      return (
        block.player_start_s != null &&
        block.player_start_s < end &&
        start < block.player_end_s
      );
    });
    if (!overlapping.length) {
      return { ok: false, error: strings.outside };
    }
    return { ok: false, error: strings.crosses };
  }

  function snapDisplay(source) {
    if (!state.frame_step_s) {
      return source;
    }
    var step = state.frame_step_s;
    return Math.round(source / step) * step;
  }

  function showPlayer(input, playerSeconds, sourceSeconds, snap) {
    if (!input) {
      return;
    }
    input.dataset.player = String(playerSeconds);
    input.dataset.dirty = "";
    var source =
      sourceSeconds == null ? playerSeconds + offset : sourceSeconds;
    if (snap) {
      source = snapDisplay(source);
    }
    input.value = Number(source).toFixed(3);
  }

  function readPlayer(input) {
    if (!input) {
      return null;
    }
    if (input.dataset.dirty === "1") {
      var typed = Number(input.value);
      if (!Number.isFinite(typed)) {
        return null;
      }
      return typed - offset;
    }
    var raw = Number(input.dataset.player);
    if (Number.isFinite(raw)) {
      return raw;
    }
    var shown = Number(input.value);
    if (!Number.isFinite(shown)) {
      return null;
    }
    return shown - offset;
  }

  function fillInputs(item) {
    if (!item) {
      return;
    }
    if (actionInput) {
      actionInput.value = item.action_name;
    }
    showPlayer(startInput, item.player_start_s, item.start_s, false);
    showPlayer(endInput, item.player_end_s, item.end_s, false);
  }

  function roleLabel(role) {
    if (!role) {
      var missing = strings.not_calculated || "";
      return state.frozen ? missing : missing + " · " + (strings.draft || "");
    }
    var block = blocks.filter(function (item) {
      return item.role === role;
    })[0];
    var label = block ? block.label : role;
    if (!state.frozen) {
      return label + " · " + (strings.draft || "");
    }
    return label;
  }

  function render() {
    track.replaceChildren();
    intervals.forEach(function (item, index) {
      var start = item.player_start_s;
      var end = item.player_end_s;
      if (start == null || end == null || end <= start) {
        return;
      }
      var span = document.createElement("span");
      span.className = "lab-timeline-block";
      if (index === selected) {
        span.classList.add("is-selected");
      }
      span.style.left = (Math.max(start, 0) / duration) * 100 + "%";
      span.style.width = (Math.max(end - start, 0) / duration) * 100 + "%";
      span.textContent = item.title || titleFor(item.action_name);
      span.addEventListener("pointerdown", function (event) {
        if (event.target.classList.contains("lab-timeline-handle")) {
          return;
        }
        selected = index;
        fillInputs(intervals[selected]);
        render();
      });
      addHandle(span, index, "start");
      addHandle(span, index, "end");
      track.appendChild(span);
    });
    if (drag && drag.create) {
      var ghost = document.createElement("span");
      ghost.className = "lab-timeline-block is-ghost";
      var ghostStart = Math.min(drag.start, drag.end);
      var ghostEnd = Math.max(drag.start, drag.end);
      ghost.style.left = (ghostStart / duration) * 100 + "%";
      ghost.style.width = ((ghostEnd - ghostStart) / duration) * 100 + "%";
      track.appendChild(ghost);
    }
    renderTable();
    renderCoverage();
  }

  function addHandle(parent, index, edge) {
    var handle = document.createElement("span");
    handle.className = "lab-timeline-handle";
    handle.style[edge === "start" ? "left" : "right"] = "0";
    handle.addEventListener("pointerdown", function (event) {
      event.preventDefault();
      event.stopPropagation();
      selected = index;
      drag = { index: index, edge: edge };
      fillInputs(intervals[index]);
      if (track.setPointerCapture) {
        track.setPointerCapture(event.pointerId);
      }
    });
    parent.appendChild(handle);
  }

  function renderTable() {
    if (!tableBody) {
      return;
    }
    tableBody.replaceChildren();
    intervals.forEach(function (item) {
      var sourceStart =
        item.start_s != null ? item.start_s : item.player_start_s + offset;
      var sourceEnd =
        item.end_s != null ? item.end_s : item.player_end_s + offset;
      var place = placement(item.player_start_s, item.player_end_s);
      var row = document.createElement("tr");
      [sourceStart, sourceEnd, sourceEnd - sourceStart].forEach(function (value) {
        var cell = document.createElement("td");
        cell.textContent = Number(value).toFixed(3);
        row.appendChild(cell);
      });
      var actionCell = document.createElement("td");
      actionCell.textContent = item.title || titleFor(item.action_name);
      row.appendChild(actionCell);
      var roleCell = document.createElement("td");
      roleCell.textContent = place.ok ? roleLabel(place.role) : roleLabel(null);
      row.appendChild(roleCell);
      tableBody.appendChild(row);
    });
  }

  function renderCoverage() {
    if (!coverage) {
      return;
    }
    var tally = {};
    (state.classes || []).forEach(function (item) {
      tally[item.name] = { title: item.title, train: 0, validation: 0 };
    });
    intervals.forEach(function (item) {
      if (!tally[item.action_name]) {
        tally[item.action_name] = {
          title: item.title || item.action_name,
          train: 0,
          validation: 0,
        };
      }
      var place = placement(item.player_start_s, item.player_end_s);
      if (place.ok && (place.role === "train" || place.role === "validation")) {
        tally[item.action_name][place.role] += 1;
      }
    });
    coverage.replaceChildren();
    (state.classes || []).forEach(function (item) {
      var counts = tally[item.name];
      var missing = counts.train === 0 || counts.validation === 0;
      var line = document.createElement("li");
      if (missing) {
        line.className = "lab-coverage-missing";
      }
      line.textContent =
        item.name +
        " — " +
        counts.title +
        ": " +
        strings.train +
        " " +
        counts.train +
        ", " +
        strings.validation +
        " " +
        counts.validation +
        (missing ? " — " + strings.missing : "");
      coverage.appendChild(line);
    });
  }

  track.addEventListener("pointerdown", function (event) {
    if (event.target !== track) {
      return;
    }
    var seconds = secondsAt(event);
    drag = { create: true, start: seconds, end: seconds };
    if (track.setPointerCapture) {
      track.setPointerCapture(event.pointerId);
    }
    render();
  });

  window.addEventListener("pointermove", function (event) {
    if (!drag) {
      return;
    }
    var seconds = secondsAt(event);
    if (drag.create) {
      drag.end = seconds;
      render();
      return;
    }
    var item = intervals[drag.index];
    if (!item) {
      return;
    }
    if (drag.edge === "start") {
      item.player_start_s = Math.min(seconds, item.player_end_s);
      item.start_s = null;
    } else {
      item.player_end_s = Math.max(seconds, item.player_start_s);
      item.end_s = null;
    }
    fillInputs(item);
    render();
  });

  window.addEventListener("pointerup", function () {
    if (!drag) {
      return;
    }
    if (drag.create) {
      var start = Math.min(drag.start, drag.end);
      var end = Math.max(drag.start, drag.end);
      drag = null;
      if (end - start > 0.05) {
        addInterval(start, end, actionInput ? actionInput.value : "");
      } else {
        render();
      }
      return;
    }
    drag = null;
  });

  function addInterval(playerStart, playerEnd, actionName) {
    if (!(playerEnd > playerStart)) {
      showError(strings.invalid);
      return;
    }
    var next = intervals.concat([
      {
        player_start_s: playerStart,
        player_end_s: playerEnd,
        start_s: null,
        end_s: null,
        action_name: actionName,
        title: titleFor(actionName),
      },
    ]);
    if (overlaps(next)) {
      showError(strings.overlap);
      return;
    }
    var place = placement(playerStart, playerEnd);
    if (!place.ok) {
      showError(place.error);
      return;
    }
    intervals = next;
    selected = intervals.length - 1;
    showError("");
    fillInputs(intervals[selected]);
    render();
  }

  function setEdge(input) {
    if (!video || !input) {
      return;
    }
    showPlayer(input, video.currentTime, null, true);
  }

  function applySelectedFromInputs() {
    if (selected < 0) {
      return;
    }
    var item = intervals[selected];
    var start = readPlayer(startInput);
    var end = readPlayer(endInput);
    if (start == null || end == null) {
      return;
    }
    item.player_start_s = start;
    item.player_end_s = end;
    item.start_s = null;
    item.end_s = null;
    if (actionInput) {
      item.action_name = actionInput.value;
      item.title = titleFor(item.action_name);
    }
  }

  function addFromInputs() {
    var start = readPlayer(startInput);
    var end = readPlayer(endInput);
    addInterval(start, end, actionInput ? actionInput.value : "");
  }

  function removeSelected() {
    if (selected < 0) {
      return;
    }
    intervals.splice(selected, 1);
    selected = intervals.length ? Math.min(selected, intervals.length - 1) : -1;
    showError("");
    if (selected >= 0) {
      fillInputs(intervals[selected]);
    }
    render();
  }

  function save() {
    applySelectedFromInputs();
    if (overlaps(intervals)) {
      showError(strings.overlap);
      render();
      return;
    }
    var index;
    for (index = 0; index < intervals.length; index += 1) {
      var item = intervals[index];
      if (!(item.player_end_s > item.player_start_s)) {
        showError(strings.invalid);
        return;
      }
      var place = placement(item.player_start_s, item.player_end_s);
      if (!place.ok) {
        showError(place.error);
        return;
      }
    }
    var query = "/actions/intervals?video=" + encodeURIComponent(state.video_id);
    postJson(query, {
      revision: state.revision,
      intervals: intervals.map(function (item) {
        var row = {
          player_start_s: item.player_start_s,
          player_end_s: item.player_end_s,
          action_name: item.action_name,
        };
        if (item.label_id) {
          row.label_id = item.label_id;
        }
        if (item.annotator) {
          row.annotator = item.annotator;
        }
        if (item.created_at && item.start_s != null) {
          row.created_at = item.created_at;
        }
        return row;
      }),
    })
      .then(function (result) {
        if (!result.ok) {
          showError((result.body && result.body.error) || strings.save_failed);
          if (reloadButton && result.body && result.body.code === "action_labels_stale") {
            reloadButton.hidden = false;
          }
          return;
        }
        window.location.reload();
      })
      .catch(function () {
        showError(strings.save_failed);
      });
  }

  function typingTarget(target) {
    if (!target || !target.tagName) {
      return false;
    }
    var tag = target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
  }

  function stepFrame(direction) {
    if (!video || !state.frame_step_s) {
      return;
    }
    var next = video.currentTime + direction * state.frame_step_s;
    video.currentTime = clamp(next);
  }

  window.addEventListener("keydown", function (event) {
    if (typingTarget(event.target)) {
      return;
    }
    if (event.key === "i") {
      setEdge(startInput);
    } else if (event.key === "o") {
      setEdge(endInput);
    } else if (event.key === "Enter") {
      event.preventDefault();
      addFromInputs();
    } else if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      removeSelected();
    } else if (event.key === "," || event.key === ".") {
      event.preventDefault();
      stepFrame(event.key === "," ? -1 : 1);
    } else if (event.key >= "1" && event.key <= "9" && actionInput) {
      var index = Number(event.key) - 1;
      if (actionInput.options[index]) {
        actionInput.selectedIndex = index;
        if (selected >= 0) {
          intervals[selected].action_name = actionInput.value;
          intervals[selected].title = titleFor(actionInput.value);
          render();
        }
      }
    }
  });

  if (startInput) {
    startInput.addEventListener("input", function () {
      startInput.dataset.dirty = "1";
    });
  }
  if (endInput) {
    endInput.addEventListener("input", function () {
      endInput.dataset.dirty = "1";
    });
  }
  var setStart = document.getElementById("action-set-start");
  var setEnd = document.getElementById("action-set-end");
  var addButton = document.getElementById("action-add");
  var deleteButton = document.getElementById("action-delete");
  var saveButton = document.getElementById("action-save");
  if (setStart) {
    setStart.addEventListener("click", function () {
      setEdge(startInput);
    });
  }
  if (setEnd) {
    setEnd.addEventListener("click", function () {
      setEdge(endInput);
    });
  }
  if (addButton) {
    addButton.addEventListener("click", addFromInputs);
  }
  if (deleteButton) {
    deleteButton.addEventListener("click", removeSelected);
  }
  if (saveButton) {
    saveButton.addEventListener("click", save);
  }
  if (reloadButton) {
    reloadButton.addEventListener("click", function () {
      window.location.reload();
    });
  }
  if (selected >= 0) {
    fillInputs(intervals[selected]);
  }
  render();
})();

(function () {
  var list = document.getElementById("action-run-list");
  var box = document.getElementById("action-readiness");
  var reason = document.getElementById("action-train-reason");
  var start = document.getElementById("action-train-start");

  function paintReadiness(payload) {
    if (!box || !payload) {
      return;
    }
    var reasons = Array.isArray(payload.reasons) ? payload.reasons : [];
    box.textContent = reasons.join("\n");
    if (!start || start.getAttribute("data-blocked") === "1") {
      return;
    }
    var blocked = payload.ok !== true;
    start.disabled = blocked;
    if (reason) {
      reason.hidden = !blocked;
      reason.textContent = blocked ? reasons[0] || "" : "";
    }
  }

  if (list) {
    list.addEventListener("change", function () {
      var ids = [];
      Array.prototype.forEach.call(list.querySelectorAll("input:checked"), function (node) {
        ids.push(node.value);
      });
      window
        .fetch("/actions/ready?runs=" + encodeURIComponent(ids.join(",")), {
          headers: { Accept: "application/json" },
        })
        .then(function (response) {
          return response.json();
        })
        .then(paintReadiness)
        .catch(function () {});
    });
  }

  var model = document.getElementById("action-model");
  if (model) {
    model.addEventListener("change", function () {
      var params = new URLSearchParams(window.location.search);
      params.set("model", model.value);
      window.location.assign("/actions?" + params.toString());
    });
  }

  function chartColor() {
    var value = window
      .getComputedStyle(document.documentElement)
      .getPropertyValue("--pdm-series-observed");
    return value ? value.trim() : "";
  }

  function drawHistory(rows) {
    var host = document.getElementById("action-f1-chart");
    var empty = document.getElementById("action-chart-empty");
    if (!host || typeof window.labDrawChart !== "function") {
      return;
    }
    if (!rows || !rows.length) {
      window.labDrawChart(host, null);
      if (empty) {
        empty.hidden = false;
      }
      return;
    }
    var color = chartColor();
    if (!color) {
      return;
    }
    window.labDrawChart(host, {
      mark: { type: "line", color: color, strokeWidth: 2 },
      data: {
        values: rows.map(function (row) {
          return { epoch: row.epoch, loss: row.val_macro_f1 };
        }),
      },
    });
    if (empty) {
      empty.hidden = true;
    }
  }

  var seed = document.getElementById("action-chart-seed");
  if (seed) {
    try {
      drawHistory(JSON.parse(seed.textContent || "[]"));
    } catch (error) {
      drawHistory([]);
    }
  }

  var poll = document.getElementById("action-job-poll");
  if (!poll) {
    return;
  }
  var jobId = poll.getAttribute("data-job-id") || "";
  var active = { queued: true, running: true, cancel_requested: true };
  var terminal = { completed: true, failed: true, cancelled: true };
  var status = poll.getAttribute("data-status") || "";
  if (!jobId || !active[status]) {
    return;
  }
  var epochNode = document.getElementById("action-epoch");
  var timer = window.setInterval(function () {
    window
      .fetch("/actions/job?id=" + encodeURIComponent(jobId), {
        headers: { Accept: "application/json" },
      })
      .then(function (response) {
        if (!response.ok) {
          return null;
        }
        return response.json();
      })
      .then(function (body) {
        if (!body) {
          return;
        }
        if (epochNode && body.epoch !== null && body.epoch !== undefined) {
          var label = epochNode.getAttribute("data-label") || "";
          var total =
            body.epochs === null || body.epochs === undefined ? "" : " / " + body.epochs;
          epochNode.textContent = label + ": " + body.epoch + total;
        }
        if (Array.isArray(body.history) && body.history.length) {
          drawHistory(body.history);
        }
        if (terminal[body.status]) {
          window.clearInterval(timer);
          window.location.reload();
        }
      })
      .catch(function () {});
  }, 2000);
})();
