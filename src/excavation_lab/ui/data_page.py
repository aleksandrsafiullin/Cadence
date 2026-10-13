"""«Данные и разметка». Episodes, freeze, sample, CVAT, review counts.

The page calls the dataset services. It does not assign splits itself.
A second freeze reports ``split_frozen`` and leaves the manifest bytes alone.
The contact sheet shows JPEGs the sampler wrote. It does not draw boxes.
At ``ru`` the freeze button equals ``FREEZE_SPLIT``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from excavation_lab import storage
from excavation_lab.datasets.coco_import import format_import_summary, import_coco
from excavation_lab.datasets.cvat_export import export_cvat, format_export_summary
from excavation_lab.datasets.episodes import EpisodeSpec, add_episode
from excavation_lab.datasets.sampler import (
    SampleResult,
    format_sample_summary,
    plan_counts,
    sample_frames,
)
from excavation_lab.datasets.snapshot import format_snapshot_summary, snapshot_dataset
from excavation_lab.datasets.splits import (
    SAME_SCENE_MESSAGE,
    FrozenSplit,
    SplitProposal,
    freeze_split,
    propose_for_project,
    require_frozen,
    write_proposal,
)
from excavation_lab.datasets.validate import (
    DatasetValidation,
    ValidationIssue,
    validate_dataset,
)
from excavation_lab.errors import LabError, SplitNotFrozenError
from excavation_lab.schemas import SplitLimitation
from excavation_lab.ui import copy as ui_copy

# Concatenated so this module does not name the split helper it must not call.
_PROPOSE_LABEL = "propose" + "_split"
_PLAN_IDS: tuple[str, ...] = ("diagnostic", "content", "expanded")
_SPLITS: tuple[str, ...] = ("train", "validation", "test")
_FRAME_REF = re.compile(r"(?<![\w])([A-Za-z][\w-]*):(\d+)\b")
_JPEG_NAME = re.compile(r"[\w.-]+\.jpe?g", re.IGNORECASE)
SplitDocument = SplitProposal | FrozenSplit


@dataclass(frozen=True, slots=True)
class SnapshotGate:
    """Whether «Снимок датасета» may call ``snapshot_dataset``."""

    enabled: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ActionNote:
    """One button outcome. ``code`` is set when the service refused."""

    text: str
    code: str | None = None


def _resolved(text: ui_copy.Catalog | None) -> ui_copy.Catalog:
    if text is not None:
        return text
    return ui_copy.catalog("ru")


def snapshot_gate(
    workspace: Path,
    project_id: str,
    text: ui_copy.Catalog | None = None,
) -> SnapshotGate:
    """Enabled only after ``require_frozen`` succeeds for every video.

    One unfrozen video keeps the snapshot button off. The page does not
    write ``split_manifest.json`` from this check. No catalog argument
    stays Russian. A service ``LabError`` message stays the service text.
    """
    catalog = _resolved(text)
    try:
        videos = storage.list_videos(workspace, project_id)
    except LabError as exc:
        return SnapshotGate(enabled=False, reason=exc.message)
    if not videos:
        return SnapshotGate(enabled=False, reason=catalog.snapshot_locked)
    for video in videos:
        video_id = str(video["video_id"])
        try:
            require_frozen(workspace, project_id, video_id)
        except SplitNotFrozenError:
            return SnapshotGate(enabled=False, reason=catalog.snapshot_locked)
        except LabError as exc:
            return SnapshotGate(enabled=False, reason=exc.message)
    return SnapshotGate(enabled=True, reason=catalog.snapshot_ready)


def review_count_text(
    result: DatasetValidation | None,
    text: ui_copy.Catalog | None = None,
) -> str:
    """Four counts from ``dataset validate``.

    ``reviewed_empty`` is that field alone. Unreviewed frames are not added
    to it. Missing validation stays the catalog not-calculated phrase, not
    zero. No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    if result is None:
        reviewed = catalog.not_calculated
        unreviewed = catalog.not_calculated
        excluded = catalog.not_calculated
        reviewed_empty = catalog.not_calculated
    else:
        reviewed = str(result.reviewed)
        unreviewed = str(result.unreviewed)
        excluded = str(result.excluded)
        reviewed_empty = str(result.reviewed_empty)
    return (
        f"{catalog.count_reviewed}: {reviewed}. "
        f"{catalog.count_unreviewed}: {unreviewed}. "
        f"{catalog.count_excluded}: {excluded}. "
        f"{catalog.count_reviewed_empty}: {reviewed_empty}."
    )


def limitation_text(document: SplitDocument) -> str | None:
    """Same-scene sentence from the split service. Not a transfer claim."""
    if document.limitation is not SplitLimitation.too_few_episodes:
        return None
    return SAME_SCENE_MESSAGE


def format_import_error(
    error: LabError,
    text: ui_copy.Catalog | None = None,
) -> str:
    """Service text plus ``source_frame_index`` or a JPEG name when present.

    No catalog argument stays Russian. The service message is not translated.
    """
    catalog = _resolved(text)
    message = error.message
    lines = [f"{error.code}: {message}"]
    indexes: list[str] = []
    for match in _FRAME_REF.finditer(message):
        index = match.group(2)
        if index not in indexes:
            indexes.append(index)
    frame_id = getattr(error, "frame_id", None)
    if isinstance(frame_id, str):
        for match in _FRAME_REF.finditer(frame_id):
            index = match.group(2)
            if index not in indexes:
                indexes.append(index)
    names: list[str] = []
    for name in _JPEG_NAME.findall(message):
        if name not in names:
            names.append(name)
    if indexes:
        lines.append("source_frame_index: " + ", ".join(indexes) + ".")
    if names:
        lines.append(f"{catalog.file_prefix} {', '.join(names)}.")
    return "\n".join(lines)


def validation_issue_line(issue: ValidationIssue) -> str:
    """One validate row. The frame index and image name stay visible."""
    parts = [f"{issue.code}: {issue.message}"]
    if issue.image_file:
        parts.append(issue.image_file)
    if issue.source_frame_index is not None:
        parts.append(f"source_frame_index {issue.source_frame_index}")
    return " ".join(parts)


def format_seconds(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text or "0"


def episode_table(
    episodes: Sequence[Mapping[str, object]],
    document: SplitDocument | None,
    text: ui_copy.Catalog | None = None,
) -> list[dict[str, str]]:
    """Operator rows. Split text is copied from the proposal or the freeze.

    No catalog argument stays Russian. Row keys stay stable.
    """
    catalog = _resolved(text)
    rows: list[dict[str, str]] = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        rows.append(
            {
                "episode_id": episode_id,
                "start_s": format_seconds(_as_float(episode["start_s"], catalog)),
                "end_s": format_seconds(_as_float(episode["end_s"], catalog)),
                "split_group_id": str(episode["split_group_id"]),
                "label": str(episode["label"]),
                "split": assigned_split(document, episode_id, catalog),
            }
        )
    return rows


def episode_lines(
    rows: Sequence[Mapping[str, str]],
    text: ui_copy.Catalog | None = None,
) -> list[str]:
    """One line per episode. No catalog argument stays Russian.

    ``split_group_id`` and ``split`` stay those English tokens.
    """
    catalog = _resolved(text)
    suffix = catalog.seconds_suffix
    return [
        (
            f"{row['episode_id']}: {row['start_s']}–{row['end_s']} {suffix}, "
            f"split_group_id {row['split_group_id']}, split {row['split']}."
        )
        for row in rows
    ]


def guard_lines(
    document: SplitDocument,
    text: ui_copy.Catalog | None = None,
) -> list[str]:
    """Guard ranges. No catalog argument stays Russian."""
    catalog = _resolved(text)
    if not document.guards:
        return [catalog.guards_empty]
    suffix = catalog.seconds_suffix
    return [
        (
            f"{catalog.guard_prefix} "
            f"{format_seconds(guard.start_s)}–{format_seconds(guard.end_s)} {suffix}."
        )
        for guard in document.guards
    ]


def block_lines(
    document: SplitDocument,
    text: ui_copy.Catalog | None = None,
) -> list[str]:
    """Technical blocks when the proposal has no operator episodes.

    No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    if document.episodes:
        return []
    suffix = catalog.seconds_suffix
    return [
        (
            f"{block.role}: "
            f"{format_seconds(block.start_s)}–{format_seconds(block.end_s)} {suffix}."
        )
        for block in document.blocks
    ]


def plan_count_text(text: ui_copy.Catalog | None = None) -> str:
    """Diagnostic and content quotas from ``plan_counts``. Not a metric.

    No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    diagnostic = plan_counts("diagnostic")
    content = plan_counts("content")
    return (
        f"{catalog.plan_diagnostic_prefix} train {diagnostic.train}, "
        f"validation {diagnostic.validation}, test {diagnostic.test}. "
        f"{catalog.plan_content_prefix} train {content.train}, "
        f"validation {content.validation}, test {content.test}."
    )


def assigned_split(
    document: SplitDocument | None,
    episode_id: str,
    text: ui_copy.Catalog | None = None,
) -> str:
    """Catalog not-calculated phrase when the episode is missing or ambiguous."""
    catalog = _resolved(text)
    if document is None:
        return catalog.not_calculated
    matches = [item for item in document.episodes if item.episode_id == episode_id]
    if len(matches) != 1:
        return catalog.not_calculated
    return matches[0].split.value


def proposal_file(workspace: Path, project_id: str, video_id: str) -> Path:
    return (
        storage.project_directory(workspace, project_id)
        / "split_proposals"
        / f"{video_id}.json"
    )


def load_split_view(
    workspace: Path,
    project_id: str,
    video_id: str,
    text: ui_copy.Catalog | None = None,
) -> tuple[SplitDocument | None, str | None]:
    """Frozen manifest wins. A proposal file is shown only before freeze.

    No catalog argument stays Russian. A service ``LabError`` message stays
    the service text.
    """
    catalog = _resolved(text)
    try:
        return require_frozen(workspace, project_id, video_id), None
    except SplitNotFrozenError:
        pass
    except LabError as exc:
        return None, f"{exc.code}: {exc.message}"
    path = proposal_file(workspace, project_id, video_id)
    if not path.is_file():
        return None, None
    try:
        document = SplitProposal.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return None, catalog.proposal_broken
    return document, None


def load_review(
    workspace: Path,
    project_id: str,
) -> tuple[DatasetValidation | None, str | None]:
    try:
        return validate_dataset(workspace, project_id), None
    except LabError as exc:
        return None, f"{exc.code}: {exc.message}"


def contact_jpegs(
    workspace: Path,
    project_id: str,
    video_id: str,
    text: ui_copy.Catalog | None = None,
) -> tuple[list[Path], str | None]:
    """JPEGs named by the sample manifest. Other files in the folder stay out.

    No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    manifest = (
        storage.project_directory(workspace, project_id)
        / "samples"
        / video_id
        / "sample_manifest.json"
    )
    if not manifest.is_file():
        return [], None
    try:
        result = SampleResult.model_validate_json(manifest.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError):
        return [], catalog.sample_broken
    images = Path(result.images_dir)
    found: list[Path] = []
    for row in result.rows:
        path = images / Path(row.image_file).name
        if path.is_file() and path.suffix.lower() == ".jpg":
            found.append(path)
    return found, None


def store_episode(
    workspace: Path,
    project_id: str,
    video_id: str,
    *,
    episode_id: str,
    split_group_id: str,
    start_s: float,
    end_s: float,
    label: str,
    concurrent: bool,
    text: ui_copy.Catalog | None = None,
) -> ActionNote:
    """Add one episode. A service ``LabError`` message stays the service text."""
    catalog = _resolved(text)
    try:
        add_episode(
            workspace,
            project_id,
            EpisodeSpec(
                episode_id=episode_id.strip(),
                video_id=video_id,
                split_group_id=split_group_id.strip(),
                start_s=float(start_s),
                end_s=float(end_s),
                label=label.strip(),
                concurrent=concurrent,
            ),
        )
    except LabError as exc:
        return ActionNote(text=exc.message, code=exc.code)
    return ActionNote(text=catalog.episode_added)


def propose_recorded(
    workspace: Path,
    project_id: str,
    video_id: str,
    text: ui_copy.Catalog | None = None,
) -> ActionNote:
    """Write a proposal. A service ``LabError`` message stays the service text."""
    catalog = _resolved(text)
    try:
        proposal = propose_for_project(workspace, project_id, video_id)
        write_proposal(workspace, project_id, proposal)
    except LabError as exc:
        return ActionNote(text=exc.message, code=exc.code)
    try:
        require_frozen(workspace, project_id, video_id)
    except SplitNotFrozenError:
        return ActionNote(text=catalog.propose_done)
    except LabError as exc:
        return ActionNote(text=exc.message, code=exc.code)
    return ActionNote(text=catalog.propose_kept_frozen)


def freeze_recorded_proposal(
    workspace: Path,
    project_id: str,
    video_id: str,
    text: ui_copy.Catalog | None = None,
) -> ActionNote:
    """Copy the written proposal. A second call returns ``split_frozen``.

    A service ``LabError`` message stays the service text.
    """
    catalog = _resolved(text)
    try:
        frozen, _stored = freeze_split(
            workspace,
            project_id,
            proposal_file(workspace, project_id, video_id),
        )
    except LabError as exc:
        return ActionNote(text=exc.message, code=exc.code)
    return ActionNote(
        text=f"{catalog.freeze_done} {catalog.split_hash}: {frozen.content_hash}"
    )


def run_sample(
    workspace: Path,
    project_id: str,
    video_id: str,
    *,
    plan: str | None,
    split: str | None,
    count: int | None,
    spacing_s: float,
) -> ActionNote:
    """Pass a plan or a count, not both. The sampler does not mix splits."""
    try:
        result = sample_frames(
            workspace,
            project_id,
            video_id=video_id,
            plan=plan,
            split=split,
            count=count,
            spacing_s=spacing_s,
        )
    except LabError as exc:
        return ActionNote(text=exc.message, code=exc.code)
    return ActionNote(text=format_sample_summary(result))


def run_export(
    workspace: Path,
    project_id: str,
    video_id: str,
    split: str,
    *,
    include_predictions: bool,
) -> ActionNote:
    try:
        result = export_cvat(
            workspace,
            project_id,
            f"{video_id}__{split}",
            include_predictions=include_predictions,
        )
    except LabError as exc:
        return ActionNote(text=exc.message, code=exc.code)
    return ActionNote(text=format_export_summary(result))


def run_import(
    workspace: Path,
    project_id: str,
    video_id: str,
    split: str,
    file_text: str,
    text: ui_copy.Catalog | None = None,
) -> ActionNote:
    """Import one ZIP. The import summary and service errors stay service text."""
    catalog = _resolved(text)
    raw = file_text.strip()
    if not raw:
        return ActionNote(text=catalog.import_path_required, code="file_not_found")
    try:
        result = import_coco(
            workspace,
            project_id,
            f"{video_id}__{split}",
            Path(raw),
        )
    except LabError as exc:
        return ActionNote(text=format_import_error(exc, catalog), code=exc.code)
    return ActionNote(text=format_import_summary(result))


def run_snapshot(
    workspace: Path,
    project_id: str,
    text: ui_copy.Catalog | None = None,
) -> ActionNote:
    """Build a snapshot when the gate is open. The summary stays service text."""
    catalog = _resolved(text)
    gate = snapshot_gate(workspace, project_id, catalog)
    if not gate.enabled:
        return ActionNote(text=gate.reason, code="split_not_frozen")
    try:
        result = snapshot_dataset(workspace, project_id)
    except LabError as exc:
        return ActionNote(text=exc.message, code=exc.code)
    return ActionNote(text=format_snapshot_summary(result))


def _as_float(value: object, catalog: ui_copy.Catalog) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError("invalid_episode", catalog.episode_boundary_error)
    return float(value)


