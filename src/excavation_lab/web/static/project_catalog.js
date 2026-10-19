(function () {
  var button = document.getElementById("add-class-row");
  var body = document.getElementById("catalog-class-rows");
  if (button && body) {
    button.addEventListener("click", function () {
      var row = body.querySelector("tr");
      if (!row) {
        return;
      }
      var next = row.cloneNode(true);
      next.querySelectorAll("input").forEach(function (input) {
        input.value = "";
      });
      body.appendChild(next);
    });
  }

  var modes = document.querySelectorAll('input[name="catalog_mode"]');
  var existing = document.getElementById("catalog-existing-fields");
  var fresh = document.getElementById("catalog-new-fields");
  if (!modes.length || !existing || !fresh) {
    return;
  }
  function sync() {
    var selected = document.querySelector('input[name="catalog_mode"]:checked');
    var isNew = selected && selected.value === "new";
    existing.hidden = Boolean(isNew);
    fresh.hidden = !isNew;
  }
  modes.forEach(function (mode) {
    mode.addEventListener("change", sync);
  });
  sync();
})();
