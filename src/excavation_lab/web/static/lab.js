(function () {
  var profilesNode = document.getElementById("train-profiles");
  var profileSelect = document.getElementById("train-profile");
  if (profilesNode && profileSelect) {
    var profiles = JSON.parse(profilesNode.textContent);
    var help = document.getElementById("train-profile-help");
    profileSelect.addEventListener("change", function () {
      var chosen = profiles[profileSelect.value];
      if (!chosen) {
        return;
      }
      var variant = document.getElementById("train-variant");
      if (variant) {
        variant.value = chosen.variant;
      }
      [
        ["train-resolution", "resolution"],
        ["train-epochs", "epochs"],
        ["train-batch", "batch_size"],
        ["train-accumulation", "gradient_accumulation"],
        ["train-lr", "learning_rate"],
        ["train-encoder-lr", "encoder_learning_rate"],
        ["train-workers", "workers"],
        ["train-seed", "seed"],
      ].forEach(function (pair) {
        var input = document.getElementById(pair[0]);
        if (input && chosen[pair[1]] !== undefined && chosen[pair[1]] !== null) {
          input.value = String(chosen[pair[1]]);
        }
      });
      var selected = profileSelect.options[profileSelect.selectedIndex];
      if (help && selected) {
        help.textContent = selected.textContent;
      }
    });
  }

  var modeSelect = document.getElementById("analysis-mode");
  if (modeSelect) {
    modeSelect.addEventListener("change", function () {
      var params = new URLSearchParams(window.location.search);
      params.set("mode", modeSelect.value);
      window.location.assign("/analysis?" + params.toString());
    });
  }

  var node = document.querySelector("[data-job-poll]");
  if (!node) {
    return;
  }
  var active = { queued: true, running: true, cancel_requested: true };
  var terminal = { completed: true, failed: true, cancelled: true };
  var status = node.getAttribute("data-status") || "";
  if (!active[status]) {
    return;
  }
  var url = node.getAttribute("data-progress-url") || "";
  if (!url || url.indexOf("://") !== -1) {
    return;
  }
  var timer = window.setInterval(function () {
    window
      .fetch(url, { headers: { Accept: "application/json" } })
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
        var line = body.line;
        var target = node.querySelector("[data-job-line]");
        if (typeof line === "string" && target) {
          target.textContent = line;
        }
        if (body.chart != null && typeof window.labReplaceLossChart === "function") {
          window.labReplaceLossChart(body.chart);
        }
        if (terminal[body.status]) {
          window.clearInterval(timer);
        }
      })
      .catch(function () {});
  }, 2000);
})();
