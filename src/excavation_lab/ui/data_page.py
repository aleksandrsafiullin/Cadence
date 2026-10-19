"""«Данные и разметка». Episodes, freeze, sample, CVAT, review counts.

The page calls the dataset services. It does not assign splits itself.
A second freeze reports ``split_frozen`` and leaves the manifest bytes alone.
The contact sheet shows JPEGs the sampler wrote. It does not draw boxes.
At ``ru`` the freeze button equals ``FREEZE_SPLIT``.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from excavation_lab import storage
from excavation_lab.datasets.blocks import GUARD_GAP_S, RoleBlock, guard_gap_warnings
from excavation_lab.datasets.catalog import add_class, require_project_catalog
from excavation_lab.datasets.coco_import import (
    format_import_summary,
    import_coco,
    load_imported_frames,
)
from excavation_lab.datasets.cvat_export import export_cvat, format_export_summary
from excavation_lab.datasets.episodes import (
    EpisodeSpec,
    add_episode,
    replace_episodes,
    specs_from_payload,
)
from excavation_lab.datasets.label_store import batch_directory
from excavation_lab.datasets.sampler import (
    SampleResult,
    format_sample_summary,
    plan_counts,
    sample_frames,
)
from excavation_lab.datasets.snapshot import format_snapshot_summary, snapshot_dataset
from excavation_lab.datasets.splits import (
    FrozenSplit,
    SplitProposal,
    freeze_split,
    propose_for_project,
    require_frozen,
    write_blocks,
    write_proposal,
)
from excavation_lab.datasets.validate import (
    DatasetValidation,
    ValidationIssue,
    validate_dataset,
)
from excavation_lab.errors import LabError, SplitNotFrozenError, TimebaseError
from excavation_lab.media.encoder import read_preview_time_map
from excavation_lab.media.timebase import (
    TimeBase,
    require_monotonic,
    seconds_to_ticks,
    source_time_s,
)
from excavation_lab.media.timebase import (
    require_half_open_interval as require_timebase_interval,
)
from excavation_lab.schemas import ReviewStatus, SplitIndependence
from excavation_lab.ui import copy as ui_copy

# Concatenated so this module does not name the split helper it must not call.
_PROPOSE_LABEL = "propose" + "_split"
_PLAN_IDS: tuple[str, ...] = ("diagnostic", "content", "expanded")
_SPLITS: tuple[str, ...] = ("train", "validation", "test")
_FRAME_REF = re.compile(r"(?<![\w])([A-Za-z][\w-]*):(\d+)\b")
_JPEG_NAME = re.compile(r"[\w.-]+\.jpe?g", re.IGNORECASE)
_CLOCK_UNKNOWN = "Длительность или смещение начала: не рассчитано."
_PREVIEW_UNKNOWN = "Связь preview с исходным временем: не рассчитано."
_PREVIEW_EDGE_S = 1e-6
_PREVIEW_TIME = "Метка времени preview не является конечным числом."
SplitDocument = SplitProposal | FrozenSplit


@dataclass(frozen=True, slots=True)
class SnapshotGate:
    """Whether «Снимок датасета» may call ``snapshot_dataset``.

    ``review_ready`` is the project-wide label condition and does not
    include the freeze check. ``refusal`` is the code ``run_snapshot``
    returns while the button is off.
    """

    enabled: bool
    reason: str
    review_ready: bool = False
    refusal: str = "split_not_frozen"


@dataclass(frozen=True, slots=True)
class ActionNote:
    """One button outcome. ``code`` is set when the service refused."""

    text: str
    code: str | None = None


@dataclass(frozen=True, slots=True)
class LabelClass:
    """One catalog row as stored. ``name`` is the data id."""

    name: str
    title: str
    box: str


def _resolved(text: ui_copy.Catalog | None) -> ui_copy.Catalog:
    if text is not None:
        return text
    return ui_copy.catalog("ru")


def snapshot_gate(
    workspace: Path,
    project_id: str,
    text: ui_copy.Catalog | None = None,
) -> SnapshotGate:
    """Enabled when every video is frozen and both blocks have a reviewed frame.

    Train and validation each need one trainable reviewed frame in the label
    store, counted across the project. One unfrozen video keeps the button
    off. A video with samples and no annotation batch adds nothing and does
    not block the others. Unreviewed frames stay out of the count. No catalog
    argument stays Russian. A service ``LabError`` message stays the service text.
    """
    catalog = _resolved(text)
    try:
        videos = storage.list_videos(workspace, project_id)
    except LabError as exc:
        return SnapshotGate(enabled=False, reason=exc.message)
    train_n, valid_n = reviewed_trainable_counts(workspace, project_id)
    ready = train_n >= 1 and valid_n >= 1
    if not videos:
        return SnapshotGate(
            enabled=False,
            reason=catalog.snapshot_locked,
            review_ready=ready,
        )
    for video in videos:
        video_id = str(video["video_id"])
        try:
            require_frozen(workspace, project_id, video_id)
        except SplitNotFrozenError:
            return SnapshotGate(
                enabled=False,
                reason=catalog.snapshot_locked,
                review_ready=ready,
            )
        except LabError as exc:
            return SnapshotGate(
                enabled=False,
                reason=exc.message,
                review_ready=ready,
            )
    counts = {"train": train_n, "validation": valid_n}
    if train_n < 1:
        return SnapshotGate(
            enabled=False,
            reason=catalog.snapshot_no_train.format(**counts),
            review_ready=False,
            refusal="dataset_not_ready",
        )
    if valid_n < 1:
        return SnapshotGate(
            enabled=False,
            reason=catalog.snapshot_no_validation.format(**counts),
            review_ready=False,
            refusal="dataset_not_ready",
        )
    return SnapshotGate(enabled=True, reason=catalog.snapshot_ready, review_ready=True)


def first_unfrozen_video_id(workspace: Path, project_id: str) -> str | None:
    """First video that still blocks the project snapshot. None when the gate is open."""
    try:
        videos = storage.list_videos(workspace, project_id)
    except LabError:
        return None
    for video in videos:
        video_id = str(video["video_id"])
        try:
            require_frozen(workspace, project_id, video_id)
        except SplitNotFrozenError:
            return video_id
        except LabError:
            return video_id
    return None


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
    """Same-scene sentence from the split service."""
    if document.independence is not SplitIndependence.same_scene:
        return None
    if not document.message:
        return None
    return document.message


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
    ordered = sorted(
        episodes,
        key=lambda episode: (_finite_float(episode["start_s"]), str(episode["episode_id"])),
    )
    rows: list[dict[str, str]] = []
    for index, episode in enumerate(ordered, start=1):
        episode_id = str(episode["episode_id"])
        assigned = assigned_split(document, episode_id, catalog)
        rows.append(
            {
                "number": str(index),
                "episode_id": episode_id,
                "start_s": format_seconds(_as_float(episode["start_s"], catalog)),
                "end_s": format_seconds(_as_float(episode["end_s"], catalog)),
                "split_group_id": str(episode["split_group_id"]),
                "label": str(episode["label"]),
                "split": assigned,
                "split_label": split_label(assigned, catalog),
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

    A gap between two stored blocks is listed as unassigned with its length,
    so the operator sees the guard interval that no split samples.
    No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    if document.episodes:
        return []
    suffix = catalog.seconds_suffix
    names = {
        "train": catalog.split_name_train,
        "validation": catalog.split_name_validation,
        "test": catalog.split_name_test,
    }
    ordered = sorted(document.blocks, key=lambda item: (item.start_s, item.end_s, item.role))
    lines: list[str] = []
    cursor: float | None = None
    for block in ordered:
        if cursor is not None and block.start_s > cursor:
            lines.append(
                f"{catalog.block_unassigned}: "
                f"{format_seconds(cursor)}–{format_seconds(block.start_s)} {suffix} "
                f"({format_seconds(block.start_s - cursor)} {suffix})."
            )
        lines.append(
            f"{names[block.role]}: "
            f"{format_seconds(block.start_s)}–{format_seconds(block.end_s)} {suffix}."
        )
        cursor = block.end_s if cursor is None else max(cursor, block.end_s)
    return lines


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


def split_label(token: str, text: ui_copy.Catalog | None = None) -> str:
    """Catalog name beside a split token. An unknown token stays not-calculated."""
    catalog = _resolved(text)
    names = {
        "train": catalog.split_name_train,
        "validation": catalog.split_name_validation,
        "test": catalog.split_name_test,
    }
    return names.get(token, catalog.not_calculated)


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
        recovered = _blocks_kept_from_old_file(path, video_id)
        if recovered is not None:
            return recovered, None
        if _haul_episode_file(path):
            return None, catalog.proposal_legacy
        return None, catalog.proposal_broken
    return document, None


def stored_episode_labels(
    workspace: Path,
    project_id: str,
    video_id: str,
    text: ui_copy.Catalog | None = None,
) -> list[str]:
    """Labels still stored in an older episode file. This does not assign a role."""
    catalog = _resolved(text)
    parsed = _json_object(proposal_file(workspace, project_id, video_id))
    if parsed is None:
        return []
    episodes = parsed.get("episodes")
    if not isinstance(episodes, list):
        return []
    lines: list[str] = []
    for item in episodes:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        start = item.get("start_s")
        end = item.get("end_s")
        if not isinstance(label, str) or not label.strip():
            continue
        if isinstance(start, bool) or isinstance(end, bool):
            continue
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        lines.append(
            f"{label.strip()}: "
            f"{format_seconds(float(start))}–{format_seconds(float(end))} "
            f"{catalog.seconds_suffix}."
        )
    return lines


def _blocks_kept_from_old_file(path: Path, video_id: str) -> SplitProposal | None:
    """Open role blocks from an older episode file. The file is not rewritten."""
    parsed = _json_object(path)
    if parsed is None or parsed.get("video_id") != video_id:
        return None
    raw_blocks = parsed.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        return None
    try:
        blocks = [RoleBlock.model_validate(item) for item in raw_blocks]
        return SplitProposal(video_id=video_id, blocks=blocks)
    except (ValidationError, ValueError):
        return None


def _haul_episode_file(path: Path) -> bool:
    """True when the file is an old episode split, not a damaged block file."""
    parsed = _json_object(path)
    if parsed is None:
        return False
    episodes = parsed.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        return False
    blocks = parsed.get("blocks")
    return blocks is None or blocks == []


def _json_object(path: Path) -> dict[str, object] | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


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


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    """One step on «Данные и разметка». ``state`` is current, done, or waiting."""

    step_id: str
    anchor: str
    state: str
    title: str
    instruction: str
    status: str
    waiting_reason: str


_WORKFLOW_IDS = ("split", "sample", "label", "snapshot")


def video_review_counts(
    workspace: Path,
    project_id: str,
    video_id: str,
) -> tuple[int | None, int | None]:
    """Reviewed and unreviewed counts for ``video_id`` only.

    ``None`` means this video has no imported frames. That is not a zero.
    Excluded frames are not reviewed. Another video's frames are ignored.
    """
    try:
        frames = load_imported_frames(workspace, project_id)
    except LabError:
        return None, None
    reviewed = 0
    unreviewed = 0
    seen = False
    for frame in frames:
        if frame.video_id != video_id:
            continue
        seen = True
        status = frame.record.review_status
        if status is ReviewStatus.reviewed:
            reviewed += 1
        elif status is ReviewStatus.unreviewed:
            unreviewed += 1
    if not seen:
        return None, None
    return reviewed, unreviewed


@dataclass(frozen=True, slots=True)
class SplitStoreCounts:
    """Review counts for one split. ``known`` is false when that batch file is absent.

    An absent batch is not a zero. ``reviewed_trainable`` counts
    ``trainable`` frames whose review status is reviewed, including
    reviewed-empty. Excluded frames stay in ``excluded``.
    """

    known: bool
    reviewed: int
    unreviewed: int
    excluded: int
    reviewed_trainable: int


def reviewed_trainable_counts(workspace: Path, project_id: str) -> tuple[int, int]:
    """Project-wide trainable reviewed frames in train and validation.

    A missing batch contributes nothing. The pair is ``(train, validation)``.
    """
    try:
        frames = load_imported_frames(workspace, project_id)
    except LabError:
        return 0, 0
    train = 0
    validation = 0
    for frame in frames:
        if not _is_reviewed_trainable(frame):
            continue
        split = _frame_split(frame)
        if split == "train":
            train += 1
        elif split == "validation":
            validation += 1
    return train, validation


def store_counts(
    workspace: Path,
    project_id: str,
    video_id: str | None = None,
) -> dict[str, SplitStoreCounts]:
    """Reviewed, unreviewed, and excluded per split from the label store.

    ``video_id`` None is the whole project. A split with no
    ``imported_labels.json`` is unknown, not zero.
    """
    try:
        frames = load_imported_frames(workspace, project_id)
    except LabError:
        frames = []
    counted: dict[str, SplitStoreCounts] = {}
    for split in _SPLITS:
        known = _split_batch_known(workspace, project_id, video_id, split)
        reviewed = 0
        unreviewed = 0
        excluded = 0
        trainable = 0
        if known:
            for frame in frames:
                if video_id is not None and getattr(frame, "video_id", None) != video_id:
                    continue
                if _frame_split(frame) != split:
                    continue
                record = getattr(frame, "record", None)
                bucket = _status_bucket(record)
                if bucket == "reviewed":
                    reviewed += 1
                    if getattr(record, "trainable", False) is True:
                        trainable += 1
                elif bucket == "unreviewed":
                    unreviewed += 1
                elif bucket == "excluded":
                    excluded += 1
        counted[split] = SplitStoreCounts(
            known=known,
            reviewed=reviewed,
            unreviewed=unreviewed,
            excluded=excluded,
            reviewed_trainable=trainable,
        )
    return counted


def format_store_counts(
    counts: Mapping[str, SplitStoreCounts],
    text: ui_copy.Catalog | None = None,
) -> str:
    """One sentence per split. An unknown split stays «не рассчитано», not 0.

    No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    parts: list[str] = []
    for split in _SPLITS:
        row = counts[split]
        title = split_label(split, catalog)
        if not row.known:
            parts.append(f"{title}: {catalog.not_calculated}.")
            continue
        parts.append(
            f"{title}: {catalog.count_reviewed} {row.reviewed}, "
            f"{catalog.count_unreviewed} {row.unreviewed}, "
            f"{catalog.count_excluded} {row.excluded}."
        )
    return " ".join(parts)


def label_step_caption(
    workspace: Path,
    project_id: str,
    video_id: str,
    text: ui_copy.Catalog | None = None,
) -> str:
    """Reviewed / unreviewed / excluded for this video from ``label_page.progress``.

    No sample manifest yields an empty caption. The caller keeps
    «не рассчитано» for that case. Once the manifest lists frames, a zero
    is a real zero.
    """
    catalog = _resolved(text)
    from excavation_lab.ui.label_page import format_progress, progress

    try:
        counts = progress(workspace, project_id, video_id)
    except LabError:
        return ""
    total = sum(int(row["total"]) for row in counts.values())
    if total <= 0:
        return ""
    return format_progress(counts, catalog)


def _frame_split(frame: object) -> str | None:
    raw = getattr(frame, "split", None)
    if raw is None:
        return None
    value = getattr(raw, "value", raw)
    if value in _SPLITS:
        return str(value)
    return None


def _is_reviewed_trainable(frame: object) -> bool:
    record = getattr(frame, "record", None)
    if record is None:
        return False
    if getattr(record, "review_status", None) is not ReviewStatus.reviewed:
        return False
    return getattr(record, "trainable", False) is True


def _status_bucket(record: object) -> str | None:
    if record is None:
        return None
    status = getattr(record, "review_status", None)
    if status is ReviewStatus.reviewed:
        return "reviewed"
    if status is ReviewStatus.unreviewed:
        return "unreviewed"
    if status is ReviewStatus.excluded:
        return "excluded"
    return None


def _split_batch_known(
    workspace: Path,
    project_id: str,
    video_id: str | None,
    split: str,
) -> bool:
    """True when ``imported_labels.json`` exists for this split.

    A project-wide check is any video's batch. The directory is not created.
    """
    try:
        if video_id is not None:
            path = batch_directory(workspace, project_id, video_id, split)
            return (path / "imported_labels.json").is_file()
        root = storage.project_directory(workspace, project_id) / "annotation_batches"
    except LabError:
        return False
    if not root.is_dir():
        return False
    suffix = f"__{split}"
    for child in root.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        if child.name.endswith(suffix) and (child / "imported_labels.json").is_file():
            return True
    return False


def snapshot_written(workspace: Path, project_id: str) -> bool:
    """True when the latest ``dataset_versions`` row has a real snapshot directory.

    The lookup matches ``validate._latest_snapshot``. A missing row or directory
    is not an error and does not invent a dataset id.
    """
    try:
        project = storage.project_directory(workspace, project_id).resolve()
        db_path = storage.project_sqlite(workspace, project_id)
    except LabError:
        return False
    with storage.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT path FROM dataset_versions
            WHERE project_id = ?
            ORDER BY created_at DESC, dataset_id DESC
            LIMIT 1
            """,
            (project_id,),
        ).fetchone()
    path_text = None if row is None else row["path"]
    if not isinstance(path_text, str) or not path_text:
        return False
    snapshot = Path(path_text).resolve()
    versions = (project / "dataset_versions").resolve()
    if not snapshot.is_dir() or not snapshot.is_relative_to(versions):
        return False
    manifest = (snapshot / "dataset_manifest.json").resolve()
    return manifest.is_file() and manifest.is_relative_to(snapshot)


def workflow_steps(
    *,
    episode_count: int,
    split_state: str,
    jpeg_count: int,
    review_ready: bool,
    gate: SnapshotGate,
    snapshot_exists: bool,
    label_caption: str = "",
    text: ui_copy.Catalog | None = None,
) -> tuple[WorkflowStep, ...]:
    """Four steps for the selected video.

    The first step is the block timeline. It stays current until freeze.
    ``episode_count`` is unused: episodes are not a step. The label step is
    done when the project has a trainable reviewed frame in train and in
    validation. Unreviewed frames stay sampled and do not block that step.
    The editor always has exactly one current step. Snapshot stays that step
    once the earlier steps are done, so the highlight does not disappear.
    Its status word is «готово» only after a snapshot directory exists.
    ``label_caption`` is this video's progress. It is empty when no sample
    manifest exists, so the step does not invent a zero.
    """
    del episode_count
    catalog = _resolved(text)
    frozen = split_state == "frozen"
    has_jpegs = jpeg_count > 0
    labeled = review_ready
    raw = [
        "done" if frozen else "current",
        "waiting" if not frozen else ("done" if has_jpegs else "current"),
        "waiting" if not has_jpegs and not labeled else ("done" if labeled else "current"),
        "current" if gate.enabled else "waiting",
    ]
    states: list[str] = []
    seen_current = False
    for state in raw:
        if state == "current" and seen_current:
            states.append("waiting")
            continue
        if state == "current":
            seen_current = True
        states.append(state)
    if not seen_current:
        for index, state in enumerate(states):
            if state != "done":
                states[index] = "current"
                break
    titles = (
        catalog.split_heading,
        catalog.sample_heading,
        catalog.step_label_title,
        catalog.step_snapshot_title,
    )
    instructions = (
        catalog.step_split_text,
        catalog.step_sample_text,
        catalog.step_label_text,
        catalog.step_snapshot_text,
    )
    status_words = {
        "current": catalog.step_now,
        "done": catalog.step_done,
        "waiting": catalog.step_wait,
    }
    own_reason = {
        "sample": catalog.step_wait_frozen,
        "label": catalog.step_wait_frames,
    }
    earlier_done = all(state == "done" for state in states[:3])
    steps: list[WorkflowStep] = []
    for index, step_id in enumerate(_WORKFLOW_IDS):
        state = states[index]
        status = status_words[state]
        reason = ""
        if step_id == "snapshot" and snapshot_exists and earlier_done:
            status = catalog.step_done
            reason = catalog.step_snapshot_written
        elif step_id == "snapshot" and not gate.enabled:
            reason = gate.reason
        elif step_id == "snapshot" and state == "waiting":
            reason = catalog.step_snapshot_open
        elif step_id == "label" and label_caption and state != "waiting":
            reason = label_caption
        elif state == "waiting":
            reason = own_reason[step_id]
        steps.append(
            WorkflowStep(
                step_id=step_id,
                anchor=f"step-{step_id}",
                state=state,
                title=titles[index],
                instruction=instructions[index],
                status=status,
                waiting_reason=reason,
            )
        )
    return tuple(steps)


@dataclass(frozen=True, slots=True)
class EpisodeWrite:
    """Outcome of replacing one video's episodes. ``episodes`` is empty on refusal."""

    note: ActionNote
    episodes: tuple[dict[str, object], ...]


def replace_stored_episodes(
    workspace: Path,
    project_id: str,
    video_id: str,
    episodes: object,
    text: ui_copy.Catalog | None = None,
) -> EpisodeWrite:
    """Replace the whole set. A service ``LabError`` message stays the service text."""
    catalog = _resolved(text)
    try:
        specs = specs_from_payload(video_id, episodes)
        stored = replace_episodes(workspace, project_id, video_id, specs)
    except LabError as exc:
        return EpisodeWrite(ActionNote(text=exc.message, code=exc.code), ())
    rows = tuple(
        {
            "episode_id": item.episode_id,
            "split_group_id": item.split_group_id,
            "start_s": item.start_s,
            "end_s": item.end_s,
            "label": item.label,
            "concurrent": item.concurrent,
        }
        for item in stored
    )
    return EpisodeWrite(ActionNote(text=catalog.episodes_saved), rows)


def cutter_payload(
    workspace: Path,
    video: Mapping[str, object],
    episodes: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, str]],
    *,
    project_id: str,
    frozen: bool,
    text: ui_copy.Catalog | None = None,
) -> dict[str, object]:
    """JSON the timeline script reads. Duration, offset, and fps stay null when unknown.

    ``reserved_ids`` are episode ids on other videos. This video's ids are
    omitted so a replace can reuse them. A stored offset of 0 stays 0.
    """
    catalog = _resolved(text)
    duration = _known_positive(video.get("duration_s"))
    offset = _known_offset(video.get("start_offset_s"))
    fps = _known_positive(video.get("fps_average"))
    if fps is None:
        fps = _known_positive(video.get("fps_nominal"))
    by_id = {row["episode_id"]: row for row in rows}
    video_id = str(video["video_id"])
    reserved = storage.other_episode_ids(workspace, project_id, video_id)
    stored: list[dict[str, object]] = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        row = by_id.get(episode_id)
        stored.append(
            {
                "episode_id": episode_id,
                "split_group_id": str(episode["split_group_id"]),
                "start_s": _as_float(episode["start_s"], catalog),
                "end_s": _as_float(episode["end_s"], catalog),
                "label": str(episode["label"]),
                "concurrent": episode["concurrent"] is True,
                "split": row["split"] if row else catalog.not_calculated,
                "split_label": row["split_label"] if row else catalog.not_calculated,
            }
        )
    return {
        "video_id": video_id,
        "media_url": f"/media/{project_id}/{video_id}",
        "duration_s": duration,
        "offset_s": offset,
        "fps": fps,
        "reserved_ids": reserved,
        "frozen": frozen,
        "default_label": catalog.default_episode_label,
        "not_calculated": catalog.not_calculated,
        "seconds_suffix": catalog.seconds_suffix,
        "episode_word": catalog.column_episode,
        "save_failed": catalog.episodes_save_failed,
        "episodes": stored,
    }


def frame_step_text(fps: float | None) -> str | None:
    """One frame in seconds. Unknown fps stays ``None``, never a stand-in rate."""
    if fps is None:
        return None
    return _trimmed_seconds(1.0 / fps)


def duration_attr(duration_s: float | None) -> str | None:
    """Template attribute for a known container length. Missing stays ``None``."""
    if duration_s is None:
        return None
    return _trimmed_seconds(duration_s)


def offset_attr(offset_s: float | None) -> str | None:
    """Template attribute for a known source offset. A stored 0 stays ``0``."""
    if offset_s is None:
        return None
    if offset_s == 0:
        return "0"
    return _trimmed_seconds(offset_s)


def source_interval_from_player(
    video: Mapping[str, object],
    player_start_s: float,
    player_end_s: float,
    *,
    preview_map: Path | None = None,
) -> tuple[float, float]:
    """Player times become source seconds before a block is stored.

    ``player_start_s`` and ``player_end_s`` are the player clock. Zero is the
    start of that file, not the source clock. The source-file player reads
    ``start_offset_s`` and ``time_base`` from the probed row and calls
    :func:`source_time_s`. A missing offset or time base is «не рассчитано»,
    not zero, and not ``frame_index / fps``.

    A preview player passes ``preview_map`` (``preview_time_map.csv``). Each
    edge becomes that file's ``source_time_s``. A missing map does not treat
    preview 0 as source 0 and does not add the offset instead.

    The returned pair is half-open source seconds. ``datasets.blocks`` stores
    that pair as given.
    """
    if preview_map is not None:
        start_s = _source_from_preview(preview_map, player_start_s)
        end_s = _source_from_preview(preview_map, player_end_s)
    else:
        start_s = _source_from_clip(video, player_start_s)
        end_s = _source_from_clip(video, player_end_s)
    require_timebase_interval(start_s, end_s)
    require_monotonic(start_s, end_s)
    storage.require_half_open_interval(start_s, end_s)
    return start_s, end_s


@dataclass(frozen=True, slots=True)
class BlockWrite:
    """Outcome of replacing one video's role blocks. ``blocks`` is empty on refusal."""

    note: ActionNote
    blocks: tuple[dict[str, object], ...]


def save_role_blocks(
    workspace: Path,
    project_id: str,
    video_id: str,
    blocks: object,
    text: ui_copy.Catalog | None = None,
) -> BlockWrite:
    """Store player intervals as source-clock blocks.

    Each edge goes through :func:`source_interval_from_player` before
    ``write_blocks``. Overlap uses the active locale's sentence. A frozen
    video keeps the service refusal and does not rewrite the file.
    """
    catalog = _resolved(text)
    if not isinstance(blocks, list):
        return BlockWrite(
            ActionNote(text=catalog.blocks_save_failed, code="invalid_blocks"),
            (),
        )
    try:
        video = storage.get_video(workspace, project_id, video_id)
        converted: list[RoleBlock] = []
        for item in blocks:
            if not isinstance(item, dict):
                return BlockWrite(
                    ActionNote(text=catalog.blocks_save_failed, code="invalid_blocks"),
                    (),
                )
            role = item.get("role")
            start = item.get("start_s")
            end = item.get("end_s")
            if (
                not isinstance(role, str)
                or role not in _SPLITS
                or isinstance(start, bool)
                or isinstance(end, bool)
            ):
                return BlockWrite(
                    ActionNote(text=catalog.blocks_save_failed, code="invalid_blocks"),
                    (),
                )
            if not isinstance(start, int | float) or not isinstance(end, int | float):
                return BlockWrite(
                    ActionNote(text=catalog.blocks_save_failed, code="invalid_blocks"),
                    (),
                )
            source_start, source_end = source_interval_from_player(
                video,
                float(start),
                float(end),
            )
            converted.append(
                RoleBlock(start_s=source_start, end_s=source_end, role=role)
            )
        path = write_blocks(workspace, project_id, video_id, converted)
        stored = SplitProposal.model_validate_json(path.read_text(encoding="utf-8"))
    except LabError as exc:
        if exc.code == "block_overlap":
            return BlockWrite(
                ActionNote(text=catalog.block_overlap, code=exc.code),
                (),
            )
        return BlockWrite(ActionNote(text=exc.message, code=exc.code), ())
    rows = tuple(
        {"start_s": item.start_s, "end_s": item.end_s, "role": item.role}
        for item in stored.blocks
    )
    return BlockWrite(ActionNote(text=catalog.blocks_saved), rows)


def block_editor_payload(
    video: Mapping[str, object],
    document: SplitDocument | None,
    *,
    frozen: bool,
    text: ui_copy.Catalog | None = None,
) -> dict[str, object]:
    """JSON the block timeline reads. Duration and offset stay null when unknown."""
    catalog = _resolved(text)
    fps = _known_positive(video.get("fps_average"))
    if fps is None:
        fps = _known_positive(video.get("fps_nominal"))
    stored: list[dict[str, object]] = []
    if document is not None:
        stored = [
            {"start_s": item.start_s, "end_s": item.end_s, "role": item.role}
            for item in document.blocks
        ]
    return {
        "video_id": str(video["video_id"]),
        "media_url": f"/media/{video['project_id']}/{video['video_id']}"
        if video.get("project_id")
        else "",
        "duration_s": _known_positive(video.get("duration_s")),
        "offset_s": _known_offset(video.get("start_offset_s")),
        "fps": fps,
        "frozen": frozen,
        "not_calculated": catalog.not_calculated,
        "unassigned_label": catalog.block_unassigned,
        "overlap_error": catalog.block_overlap,
        "save_failed": catalog.blocks_save_failed,
        "roles": [
            {"value": name, "label": split_label(name, catalog)} for name in _SPLITS
        ],
        "blocks": stored,
    }


def timeline_pieces(
    blocks: Sequence[RoleBlock],
    offset_s: float | None,
    duration_s: float | None,
    text: ui_copy.Catalog | None = None,
) -> list[dict[str, object]]:
    """Player-window pieces. Gaps are unassigned, not the training role.

    Missing duration or offset draws nothing. A stored 0 offset is a real clock.
    """
    catalog = _resolved(text)
    if offset_s is None or duration_s is None or duration_s <= 0:
        return []
    window_start = offset_s
    window_end = offset_s + duration_s
    ordered = sorted(blocks, key=lambda item: (item.start_s, item.end_s, item.role))
    pieces: list[dict[str, object]] = []
    cursor = window_start
    for block in ordered:
        start = max(block.start_s, window_start)
        end = min(block.end_s, window_end)
        if end <= window_start or start >= window_end or end <= start:
            continue
        if start > cursor:
            pieces.append(
                _piece(
                    cursor,
                    start,
                    window_start,
                    duration_s,
                    catalog.block_unassigned,
                    True,
                )
            )
        pieces.append(
            _piece(
                start,
                end,
                window_start,
                duration_s,
                split_label(block.role, catalog),
                False,
            )
        )
        cursor = end
    if cursor < window_end:
        pieces.append(
            _piece(cursor, window_end, window_start, duration_s, catalog.block_unassigned, True)
        )
    return pieces


def _piece(
    start: float,
    end: float,
    window_start: float,
    duration_s: float,
    label: str,
    unassigned: bool,
) -> dict[str, object]:
    left = (start - window_start) / duration_s * 100.0
    width = (end - start) / duration_s * 100.0
    return {
        "label": label,
        "left": _percent(left),
        "width": _percent(width),
        "unassigned": unassigned,
    }


def timeline_number_marks(
    episodes: Sequence[Mapping[str, object]],
    offset_s: float | None,
    duration_s: float | None,
) -> list[dict[str, str]]:
    """Left-to-right labels under the ruler. Missing clock draws no marks."""
    if offset_s is None or duration_s is None or duration_s <= 0:
        return []
    ordered = sorted(
        episodes,
        key=lambda episode: (float(episode["start_s"]), str(episode["episode_id"])),
    )
    marks: list[dict[str, str]] = []
    for index, episode in enumerate(ordered, start=1):
        start = float(episode["start_s"])
        end = float(episode["end_s"])
        center = ((start + end) / 2.0 - offset_s) / duration_s * 100.0
        marks.append({"number": str(index), "left": _percent(center)})
    return marks


def _percent(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        return "0"
    return text


def _trimmed_seconds(value: float) -> str | None:
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    if not text or text == "0":
        return None
    return text


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
    """Refuse an automatic split. A service ``LabError`` message stays the service text."""
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
    lines = [f"{catalog.freeze_done} {catalog.split_hash}: {frozen.content_hash}"]
    lines.extend(guard_gap_lines(frozen.blocks, catalog))
    return ActionNote(text="\n".join(lines))


def guard_gap_lines(
    blocks: Sequence[RoleBlock],
    text: ui_copy.Catalog | None = None,
) -> list[str]:
    """One line per adjacent pair of different roles closer than the stated guard.

    The freeze is not refused; the operator sees the pair and the actual gap.
    No catalog argument stays Russian.
    """
    catalog = _resolved(text)
    return [
        catalog.block_guard_gap.format(
            earlier=split_label(warning.earlier.role, catalog),
            earlier_end=format_seconds(warning.earlier.end_s),
            later=split_label(warning.later.role, catalog),
            later_start=format_seconds(warning.later.start_s),
            gap=format_seconds(warning.gap_s),
            minimum=format_seconds(GUARD_GAP_S),
        )
        for warning in guard_gap_warnings(blocks)
    ]


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


def labeling_classes(
    workspace: Path,
    project_id: str,
) -> tuple[str | None, tuple[LabelClass, ...]]:
    """Stored catalog rows for the labeling step.

    ``None`` means this project has no catalog. Copy-catalog box phrases
    are not a substitute.
    """
    catalog_id = _project_catalog_id(workspace, project_id)
    if catalog_id is None:
        return None, ()
    try:
        stored = require_project_catalog(workspace, project_id)
    except LabError:
        return None, ()
    rows = tuple(
        LabelClass(name=item.name, title=item.title, box=item.box)
        for item in sorted(stored.classes, key=lambda item: (item.id, item.name))
    )
    return catalog_id, rows


def add_catalog_class(
    workspace: Path,
    project_id: str,
    name: str,
    title: str,
    box: str,
    text: ui_copy.Catalog | None = None,
) -> ActionNote:
    """Append one class on the project catalog. Frame boxes stay as stored."""
    catalog = _resolved(text)
    catalog_id = _project_catalog_id(workspace, project_id)
    if catalog_id is None:
        return ActionNote(text=catalog.catalog_required, code="catalog_missing")
    try:
        add_class(workspace, catalog_id, name, title, box)
    except LabError as exc:
        return ActionNote(text=exc.message, code=exc.code)
    return ActionNote(text=catalog.class_added)


def _project_catalog_id(workspace: Path, project_id: str) -> str | None:
    try:
        db_path = storage.project_sqlite(workspace, project_id)
    except LabError:
        return None
    if not db_path.is_file() or db_path.is_symlink():
        return None
    try:
        with storage.connect(db_path) as connection:
            row = connection.execute("SELECT catalog_id FROM projects LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        return None
    value = None if row is None else row["catalog_id"]
    if not isinstance(value, str) or not value.strip():
        return None
    return value


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
        return ActionNote(text=gate.reason, code=gate.refusal or "split_not_frozen")
    try:
        result = snapshot_dataset(workspace, project_id)
    except LabError as exc:
        return ActionNote(text=exc.message, code=exc.code)
    return ActionNote(text=format_snapshot_summary(result))


def _known_positive(value: object) -> float | None:
    """A finite value above zero. Zero, missing, and non-numbers stay unknown."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _known_offset(value: object) -> float | None:
    """Finite ``start_offset_s`` at or above 0, including a stored 0.

    Same rule as ``video_page._known_offset``. A missing offset is not 0.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _source_from_clip(video: Mapping[str, object], player_s: float) -> float:
    """Elapsed player seconds on the probed source clock.

    Ticks come from the row's ``time_base``. ``source_time_s`` adds
    ``start_offset_s``. A missing measurement is not 0.
    """
    offset = _known_offset(video.get("start_offset_s"))
    if offset is None:
        raise LabError("source_clock_unknown", _CLOCK_UNKNOWN)
    raw_base = video.get("time_base")
    if not isinstance(raw_base, str) or not raw_base.strip():
        raise LabError("source_clock_unknown", _CLOCK_UNKNOWN)
    time_base = TimeBase.parse(raw_base)
    ticks = seconds_to_ticks(player_s, time_base)
    return source_time_s(ticks, time_base, offset)


def _source_from_preview(preview_map: Path, player_s: float) -> float:
    """One preview instant from ``preview_time_map.csv``.

    The stored ``source_time_s`` is already on the source clock. This does
    not add ``start_offset_s`` again and does not substitute 0.
    """
    if not preview_map.is_file():
        raise LabError("source_clock_unknown", _PREVIEW_UNKNOWN)
    rows = read_preview_time_map(preview_map)
    if not rows:
        raise LabError("source_clock_unknown", _PREVIEW_UNKNOWN)
    if isinstance(player_s, bool) or not isinstance(player_s, (int, float)):
        raise TimebaseError(_PREVIEW_TIME)
    preview_s = float(player_s)
    if not math.isfinite(preview_s):
        raise TimebaseError(_PREVIEW_TIME)
    for row in rows:
        if abs(row.preview_time_s - preview_s) <= _PREVIEW_EDGE_S:
            return row.source_time_s
    earlier = [row for row in rows if row.preview_time_s < preview_s]
    later = [row for row in rows if row.preview_time_s > preview_s]
    if not earlier or not later:
        raise LabError("source_clock_unknown", _PREVIEW_UNKNOWN)
    left = earlier[-1]
    right = later[0]
    require_monotonic(left.source_time_s, right.source_time_s)
    span = right.preview_time_s - left.preview_time_s
    if span <= 0:
        raise LabError("source_clock_unknown", _PREVIEW_UNKNOWN)
    share = (preview_s - left.preview_time_s) / span
    return left.source_time_s + share * (right.source_time_s - left.source_time_s)


def _finite_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return number


def _as_float(value: object, catalog: ui_copy.Catalog) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabError("invalid_episode", catalog.episode_boundary_error)
    return float(value)


