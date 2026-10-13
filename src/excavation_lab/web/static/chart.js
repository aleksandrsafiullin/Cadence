(function () {
  var SVG = (function () {
    var holder = document.createElement("div");
    holder.innerHTML = "<svg></svg>";
    var node = holder.firstChild;
    return node && node.namespaceURI ? node.namespaceURI : "";
  })();

  function rowsOf(spec) {
    if (!spec || !spec.data || !Array.isArray(spec.data.values)) {
      return null;
    }
    return spec.data.values;
  }

  function markOf(spec) {
    if (!spec || spec.mark == null) {
      return null;
    }
    if (typeof spec.mark === "string") {
      return { type: spec.mark };
    }
    if (typeof spec.mark === "object") {
      return spec.mark;
    }
    return null;
  }

  function padding(spec) {
    var raw = spec && spec.padding ? spec.padding : {};
    function side(name, fallback) {
      return typeof raw[name] === "number" ? raw[name] : fallback;
    }
    return {
      left: side("left", 56),
      right: side("right", 16),
      bottom: side("bottom", 48),
      top: side("top", 16),
    };
  }

  function finite(value) {
    return typeof value === "number" && isFinite(value) ? value : null;
  }

  function extent(values) {
    var min = values[0];
    var max = values[0];
    for (var i = 1; i < values.length; i += 1) {
      if (values[i] < min) {
        min = values[i];
      }
      if (values[i] > max) {
        max = values[i];
      }
    }
    if (min === max) {
      min -= 1;
      max += 1;
    }
    return { min: min, max: max };
  }

  function sheet(spec, mark) {
    var pad = padding(spec);
    var width = 640;
    var height = typeof spec.height === "number" ? spec.height : 300;
    var svg = document.createElementNS(SVG, "svg");
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.setAttribute("role", "img");
    if (typeof spec.background === "string") {
      svg.style.background = spec.background;
    }
    var font = spec.config && spec.config.font;
    if (typeof font === "string") {
      svg.style.font = font;
    }
    return { svg: svg, pad: pad, width: width, height: height };
  }

  function drawLine(rows, mark, spec) {
    var points = [];
    for (var i = 0; i < rows.length; i += 1) {
      var row = rows[i];
      if (!row) {
        continue;
      }
      var epoch = finite(row.epoch);
      var loss = finite(row.loss);
      if (epoch === null || loss === null) {
        continue;
      }
      points.push({ epoch: epoch, loss: loss });
    }
    if (!points.length || typeof mark.color !== "string") {
      return null;
    }
    var box = sheet(spec, mark);
    var x = extent(points.map(function (point) { return point.epoch; }));
    var y = extent(points.map(function (point) { return point.loss; }));
    var innerW = box.width - box.pad.left - box.pad.right;
    var innerH = box.height - box.pad.top - box.pad.bottom;
    var coords = points.map(function (point) {
      var px = box.pad.left + ((point.epoch - x.min) / (x.max - x.min)) * innerW;
      var py = box.pad.top + (1 - (point.loss - y.min) / (y.max - y.min)) * innerH;
      return px + "," + py;
    });
    var line = document.createElementNS(SVG, "polyline");
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", mark.color);
    line.setAttribute(
      "stroke-width",
      typeof mark.strokeWidth === "number" ? String(mark.strokeWidth) : "2"
    );
    line.setAttribute("points", coords.join(" "));
    box.svg.appendChild(line);
    return box.svg;
  }

  function drawBars(rows, mark, spec) {
    var bars = [];
    for (var i = 0; i < rows.length; i += 1) {
      var row = rows[i];
      if (!row) {
        continue;
      }
      var label = row["класс"];
      var score = finite(row.AP50);
      if (typeof label !== "string" || !label || score === null) {
        continue;
      }
      bars.push({ label: label, score: score });
    }
    if (!bars.length || typeof mark.color !== "string") {
      return null;
    }
    var box = sheet(spec, mark);
    var y = extent(bars.map(function (bar) { return bar.score; }));
    var innerW = box.width - box.pad.left - box.pad.right;
    var innerH = box.height - box.pad.top - box.pad.bottom;
    var slot = innerW / bars.length;
    var gap = Math.min(12, slot * 0.2);
    bars.forEach(function (bar, index) {
      var top = box.pad.top + (1 - (bar.score - y.min) / (y.max - y.min)) * innerH;
      var bottom = box.pad.top + innerH;
      var rect = document.createElementNS(SVG, "rect");
      rect.setAttribute("x", String(box.pad.left + index * slot + gap / 2));
      rect.setAttribute("y", String(top));
      rect.setAttribute("width", String(Math.max(slot - gap, 1)));
      rect.setAttribute("height", String(Math.max(bottom - top, 1)));
      rect.setAttribute("fill", mark.color);
      box.svg.appendChild(rect);
    });
    return box.svg;
  }

  function draw(spec) {
    var mark = markOf(spec);
    var rows = rowsOf(spec);
    if (!SVG || !mark || typeof mark.type !== "string" || !rows) {
      return null;
    }
    if (mark.type === "line") {
      return drawLine(rows, mark, spec);
    }
    if (mark.type === "bar") {
      return drawBars(rows, mark, spec);
    }
    return null;
  }

  function lossHost() {
    var host = document.getElementById("loss-chart");
    if (host) {
      return host;
    }
    var root = document.querySelector(".lab-data");
    if (!root) {
      return null;
    }
    host = document.createElement("div");
    host.id = "loss-chart";
    var anchor = document.getElementById("train-eta");
    var block = anchor;
    while (block && block.className !== "lab-indicators") {
      block = block.parentNode;
    }
    if (block && block.parentNode) {
      block.parentNode.insertBefore(host, block.nextSibling);
    } else {
      root.appendChild(host);
    }
    return host;
  }

  function replaceSvg(host, spec) {
    if (!host) {
      return;
    }
    var picture = draw(spec);
    var current = host.querySelector("svg");
    if (current) {
      current.remove();
    }
    if (picture) {
      host.appendChild(picture);
    }
  }

  window.labReplaceLossChart = function (spec) {
    replaceSvg(lossHost(), spec);
  };

  var embedded = document.getElementById("ap50-chart");
  if (!embedded || !embedded.parentNode) {
    return;
  }
  var parsed = null;
  try {
    parsed = JSON.parse(embedded.textContent || "");
  } catch (error) {
    return;
  }
  var host = document.createElement("div");
  host.id = "ap50-chart-host";
  embedded.parentNode.insertBefore(host, embedded.nextSibling);
  replaceSvg(host, parsed);
})();
