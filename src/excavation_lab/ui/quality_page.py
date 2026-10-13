"""«Качество». Measured detection metrics, or «не рассчитано».

Calls ``load_quality_report``, ``compare``, and ``send_to_labeling``.
Does not import RF-DETR, construct a detector, or enqueue a train or
analyze job. IDF1 and HOTA stay ``not evaluated``. A missing score is
not drawn as zero. Match boxes are not drawn.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from excavation_lab.errors import LabError
from excavation_lab.models.quality import (
    CLASS_METRIC_KEYS,
    CLASS_NAMES,
    compare,
    load_quality_report,
    metric_text,
    send_to_labeling,
)
from excavation_lab.storage import project_directory
from excavation_lab.ui import copy as ui_copy
from excavation_lab.ui.charts import ap50_chart
from excavation_lab.ui.theme import resolve_theme

_DELTA_KEYS = ("ap50", "ap50_95", "precision", "recall")
_TARGET_NAMES = ("haul_truck", "bucket")
_TARGET_METRICS = ("precision", "recall")
_ARG = "\u0001"
_ARG_SPLIT = "\u0002"

# The page server and tests call these through this module.
__all__ = [
    "compare",
    "load_quality_report",
    "send_to_labeling",
]


def list_metric_runs(workspace: Path, project_id: str) -> list[str]:
    """Eval ids that have ``runs/<id>/metrics.json``. No invented id."""
    directory = project_directory(workspace, project_id) / "runs"
    if directory.is_symlink() or not directory.is_dir():
        return []
    found: list[str] = []
    for child in sorted(directory.iterdir(), key=lambda item: item.name):
        if not _safe_run_name(child.name) or child.is_symlink() or not child.is_dir():
            continue
        metrics = child / "metrics.json"
        if metrics.is_symlink() or not metrics.is_file():
            continue
        found.append(child.name)
    return found


def ap50_chart_spec(
    report: Mapping[str, Any],
    workspace: Path,
    text: ui_copy.Catalog | None = None,
) -> dict[str, Any] | None:
    """Altair spec when the measured series has a bar. A missing score is omitted."""
    frame = ap50_chart_frame(report, text)
    if frame is not None:
        spec = ap50_chart(frame, resolve_theme(workspace, session={})).to_dict()
    else:
        spec = None
    return spec


def ap50_chart_frame(
    report: Mapping[str, Any],
    text: ui_copy.Catalog | None = None,
) -> list[dict[str, object]] | None:
    """AP50 bars for a computed report.

    Missing scores are omitted. They are not plotted as zero. A series that
    is only zeros is not returned. No catalog argument stays Russian.
    Class ids stay the report keys. Rows stay keyed by ``класс``.
    """
    catalog = _resolved(text)
    if report.get("status") != "computed":
        return None
    classes = report.get("classes")
    if not isinstance(classes, Mapping):
        return None
    rows: list[dict[str, object]] = []
    for name, row in classes.items():
        if not isinstance(row, Mapping):
            continue
        number = _finite(row.get("ap50"))
        if number is None:
            continue
        rows.append(
            {
                "класс": catalog.class_labels.get(str(name), str(name)),
                "AP50": number,
            }
        )
    if not rows or all(item["AP50"] == 0 for item in rows):
        return None
    return rows


def send_enabled(split: object) -> bool:
    """Train rows only. Validation, test, unknown, and a missing split stay off."""
    return split == "train"


def send_button_key(index: int, kind: object, frame_id: object) -> str:
    kind_text = kind if isinstance(kind, str) and kind else "error"
    frame_text = frame_id if isinstance(frame_id, str) and frame_id else "none"
    raw = f"{kind_text}_{frame_text}_{index + 1}"
    safe = "".join(char if char.isalnum() else "_" for char in raw)
    return f"quality_send_{safe}"


def _unknown_report() -> dict[str, Any]:
    blank_target = {
        "target": ui_copy.NOT_CALCULATED,
        "value": ui_copy.NOT_CALCULATED,
        "status": "target",
    }
    return {
        "status": "not_computed",
        "reviewed_frame_count": ui_copy.NOT_CALCULATED,
        "gt_object_count": ui_copy.NOT_CALCULATED,
        "recall_clear": ui_copy.NOT_CALCULATED,
        "recall_partial": ui_copy.NOT_CALCULATED,
        "score_floor": ui_copy.NOT_CALCULATED,
        "operating_threshold": ui_copy.NOT_CALCULATED,
        "classes": {
            name: {key: ui_copy.NOT_CALCULATED for key in CLASS_METRIC_KEYS}
            for name in CLASS_NAMES
        },
        "targets": {
            name: {metric: dict(blank_target) for metric in _TARGET_METRICS}
            for name in _TARGET_NAMES
        },
        "target_caption": ui_copy.TARGET_CAPTION,
        "errors": [],
        "errors_status": "unavailable",
        "errors_message": ui_copy.ERRORS_UNAVAILABLE,
    }


def _class_rows(report: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    classes = report.get("classes")
    if not isinstance(classes, Mapping):
        classes = {}
    names = list(classes.keys()) or list(CLASS_NAMES)
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for name in names:
        row = classes.get(name)
        if not isinstance(row, Mapping):
            row = {}
        rows.append((str(name), row))
    return rows


def _target_pair(spec: object) -> tuple[object, object]:
    if not isinstance(spec, Mapping):
        return None, None
    return spec.get("target"), spec.get("value")


def _shown_metric(value: object, catalog: ui_copy.Catalog) -> str:
    """Numbers stay ``format(value, "g")``. Nulls use the active catalog.

    The Russian token is recognized by equality. It is not replaced inside
    a longer service sentence, and it is not written back onto the report.
    """
    if value is None or value == ui_copy.NOT_CALCULATED:
        return catalog.not_calculated
    rendered = metric_text(value)
    if rendered == ui_copy.NOT_CALCULATED:
        return catalog.not_calculated
    return rendered


def _errors_caption(report: Mapping[str, Any], catalog: ui_copy.Catalog) -> str | None:
    """Known error sentences come from the catalog, not the stored Russian text."""
    status = report.get("errors_status")
    if status == "unavailable":
        return catalog.errors_unavailable
    if status == "available":
        return catalog.errors_from_matches
    message = report.get("errors_message")
    if message == ui_copy.ERRORS_UNAVAILABLE:
        return catalog.errors_unavailable
    if message == ui_copy.ERRORS_FROM_MATCHES:
        return catalog.errors_from_matches
    if isinstance(message, str) and message:
        return message
    return None


def _localized_error(exc: LabError, catalog: ui_copy.Catalog) -> str:
    """Catalog text when this code's Russian sentence is a catalog function.

    The code stays the service code. Unrelated service text stays ``exc.message``.
    """
    translated = _catalog_function_text(exc.code, exc.message, catalog)
    if translated is None:
        return exc.message
    return translated


def _catalog_function_text(code: str, message: str, catalog: ui_copy.Catalog) -> str | None:
    if code == "not_train_frame":
        pair = _not_train_args(message)
        if pair is None:
            return None
        return catalog.not_train_frame_message(pair[0], pair[1])
    if code == "frame_not_found":
        frame_id = _single_arg(message, "frame_not_found_message")
        if frame_id is not None:
            return catalog.frame_not_found_message(frame_id)
        frame_id = _single_arg(message, "frame_ambiguous_message")
        if frame_id is not None:
            return catalog.frame_ambiguous_message(frame_id)
        return None
    if code == "eval_not_found":
        eval_id = _eval_arg(message)
        if eval_id is None:
            return None
        return catalog.eval_not_found_message(eval_id)
    if code == "episode_not_found":
        episode_id = _single_arg(message, "episode_ambiguous_message")
        if episode_id is None:
            return None
        return catalog.episode_ambiguous_message(episode_id)
    return None


def _not_train_args(message: str) -> tuple[str, str] | None:
    """Recover frame and split only when the whole sentence is the Russian function."""
    ru = ui_copy.catalog("ru")
    template = ru.not_train_frame_message(_ARG, _ARG_SPLIT)
    prefix, rest = template.split(_ARG, 1)
    middle, suffix = rest.split(_ARG_SPLIT, 1)
    body = _between(message, prefix, suffix)
    if body is None or middle not in body:
        return None
    frame_id, sep, split = body.partition(middle)
    if not sep or not frame_id or not split or middle in frame_id or middle in split:
        return None
    if message != ru.not_train_frame_message(frame_id, split):
        return None
    return frame_id, split


def _single_arg(message: str, name: str) -> str | None:
    """Recover one argument when the whole sentence is that Russian function."""
    ru = ui_copy.catalog("ru")
    method = getattr(ru, name)
    prefix, suffix = method(_ARG).split(_ARG, 1)
    inner = _between(message, prefix, suffix)
    if inner is None or _ARG in inner:
        return None
    if message != method(inner):
        return None
    return inner


def _eval_arg(message: str) -> str | None:
    ru = ui_copy.catalog("ru")
    if message == ru.eval_not_found_message(""):
        return ""
    prefix, suffix = ru.eval_not_found_message(_ARG).split(_ARG, 1)
    shown = _between(message, prefix, suffix)
    if shown is None or _ARG in shown:
        return None
    if message != ru.eval_not_found_message(shown):
        return None
    return shown


def _between(message: str, prefix: str, suffix: str) -> str | None:
    if not message.startswith(prefix) or not message.endswith(suffix):
        return None
    if len(message) <= len(prefix) + len(suffix):
        return None
    inner = message[len(prefix) : len(message) - len(suffix)]
    if not inner:
        return None
    return inner


def _resolved(text: ui_copy.Catalog | None) -> ui_copy.Catalog:
    if text is not None:
        return text
    return ui_copy.catalog("ru")


def _class_label(name: object, catalog: ui_copy.Catalog) -> str:
    if not isinstance(name, str) or not name:
        return catalog.not_calculated
    return catalog.class_labels.get(name, name)


def _kind_label(kind: object, catalog: ui_copy.Catalog) -> str:
    if not isinstance(kind, str) or not kind:
        return catalog.not_calculated
    return catalog.error_kind_labels.get(kind, kind)


def _split_text(split: object, catalog: ui_copy.Catalog) -> str:
    if isinstance(split, str) and split:
        return split
    return catalog.not_calculated


def _frame_text(frame_id: object, catalog: ui_copy.Catalog) -> str:
    if isinstance(frame_id, str) and frame_id:
        return frame_id
    return catalog.not_calculated


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _safe_run_name(name: str) -> bool:
    if not name or name in {".", ".."}:
        return False
    return "/" not in name and "\\" not in name and "\x00" not in name


def report_fields(report: Mapping[str, Any], catalog: ui_copy.Catalog) -> dict[str, Any]:
    """Rows the quality template prints. Scores go through ``_shown_metric``."""
    summary = [
        (
            catalog.metric_labels["reviewed_frame_count"],
            _shown_metric(report.get("reviewed_frame_count"), catalog),
        ),
        (
            catalog.metric_labels["gt_object_count"],
            _shown_metric(report.get("gt_object_count"), catalog),
        ),
    ]
    visibility = [
        (
            f"{catalog.quality_visibility} — {catalog.metric_labels['recall_clear']}",
            _shown_metric(report.get("recall_clear"), catalog),
        ),
        (
            f"{catalog.quality_visibility} — {catalog.metric_labels['recall_partial']}",
            _shown_metric(report.get("recall_partial"), catalog),
        ),
    ]
    tracking_caption = (
        f"{catalog.metric_labels['idf1']}, {catalog.metric_labels['hota']}: "
        f"{catalog.not_evaluated_caption}. {catalog.not_calculated}."
    )
    tracking = [
        (catalog.metric_labels["idf1"], catalog.not_calculated),
        (catalog.metric_labels["hota"], catalog.not_calculated),
    ]
    classes: list[tuple[str, str]] = []
    for name, row in _class_rows(report):
        label = _class_label(name, catalog)
        for key in CLASS_METRIC_KEYS:
            classes.append(
                (
                    f"{label} — {catalog.metric_labels[key]}",
                    _shown_metric(row.get(key), catalog),
                )
            )
    message = _errors_caption(report, catalog)
    errors = report.get("errors")
    rows = [row for row in errors if isinstance(row, Mapping)] if isinstance(errors, list) else []
    error_rows: list[dict[str, object]] = []
    if rows or message is not None:
        for index, row in enumerate(rows):
            kind = row.get("kind")
            frame_id = row.get("frame_id")
            split = row.get("split")
            enabled = send_enabled(split) and isinstance(frame_id, str) and bool(frame_id)
            error_rows.append(
                {
                    "line": (
                        f"{catalog.error_kind}: {_kind_label(kind, catalog)}. "
                        f"{catalog.error_frame}: {_frame_text(frame_id, catalog)}. "
                        f"{catalog.error_class}: {_class_label(row.get('class_name'), catalog)}. "
                        f"{catalog.error_split}: {_split_text(split, catalog)}."
                    ),
                    "enabled": enabled,
                    "key": send_button_key(index, kind, frame_id),
                    "frame_id": frame_id if isinstance(frame_id, str) else "",
                }
            )
    return {
        "summary_metrics": _pairs(summary),
        "visibility_metrics": _pairs(visibility),
        "tracking_caption": tracking_caption,
        "tracking_metrics": _pairs(tracking),
        "class_metrics": _pairs(classes),
        "target_lines": _target_lines(report, catalog),
        "setting_lines": [
            (
                f"{catalog.metric_labels['score_floor']}: "
                f"{_shown_metric(report.get('score_floor'), catalog)}."
            ),
            (
                f"{catalog.metric_labels['operating_threshold']}: "
                f"{_shown_metric(report.get('operating_threshold'), catalog)}."
            ),
        ],
        "show_errors": bool(rows) or message is not None,
        "errors_message": message,
        "error_rows": error_rows,
    }


def delta_fields(result: Mapping[str, Any], catalog: ui_copy.Catalog) -> list[dict[str, str]]:
    """Class deltas. A missing number stays the catalog phrase, not zero."""
    delta = result.get("delta")
    if not isinstance(delta, Mapping):
        return []
    found: list[dict[str, str]] = []
    for name, row in delta.items():
        if not isinstance(row, Mapping):
            continue
        label = _class_label(name, catalog)
        for key in _DELTA_KEYS:
            found.append(
                {
                    "label": f"{catalog.quality_delta} — {label} — {catalog.metric_labels[key]}",
                    "value": _shown_metric(row.get(key), catalog),
                }
            )
    return found


def _target_lines(report: Mapping[str, Any], catalog: ui_copy.Catalog) -> list[str]:
    targets = report.get("targets")
    if not isinstance(targets, Mapping):
        return []
    lines: list[str] = []
    for name, block in targets.items():
        if not isinstance(block, Mapping):
            continue
        label = _class_label(name, catalog)
        for metric_name in _TARGET_METRICS:
            target, value = _target_pair(block.get(metric_name))
            lines.append(
                f"{label}. {catalog.metric_labels[metric_name]}: "
                f"{_shown_metric(target, catalog)}. {_shown_metric(value, catalog)}."
            )
    return lines


def _pairs(items: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"label": label, "value": value} for label, value in items]
