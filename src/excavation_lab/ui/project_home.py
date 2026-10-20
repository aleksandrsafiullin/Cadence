"""Display rows for the project list. No page server and no SQL."""

from __future__ import annotations

from pathlib import Path

from excavation_lab import storage
from excavation_lab.ui.copy import Catalog

_TEXT_FIELDS = ("name", "catalog_id", "created_at")
_COUNT_FIELDS = (
    "video_count",
    "episode_count",
    "dataset_count",
    "model_count",
)


def display_rows(workspace: Path, book: Catalog) -> list[dict[str, str]]:
    """Eight string fields per project. ``None`` becomes «не рассчитано»."""
    rows: list[dict[str, str]] = []
    for item in storage.list_project_summaries(workspace):
        row = {"project_id": item["project_id"]}
        for key in _TEXT_FIELDS:
            row[key] = _text(item[key], book)
        for key in _COUNT_FIELDS:
            row[key] = _count(item[key], book)
        rows.append(row)
    return rows


def _text(value: str | None, book: Catalog) -> str:
    if value is None:
        return book.not_calculated
    return value


def _count(value: int | None, book: Catalog) -> str:
    if value is None:
        return book.not_calculated
    return format(value, "d")
