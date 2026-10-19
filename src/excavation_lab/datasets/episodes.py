"""Operator episodes for one video.

Intervals are half-open ``[start_s, end_s)``. Overlap in one video is rejected
unless the episodes share a ``split_group_id`` the operator marked concurrent.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from excavation_lab.errors import LabError
from excavation_lab.schemas import Episode
from excavation_lab.storage import (
    get_split_manifest,
    get_video,
    insert_episode,
    list_episodes,
    replace_video_episodes,
    require_half_open_interval,
)

_BAD_EPISODE = (
    "Эпизод задан некорректно: нужны идентификаторы, метка и полуоткрытый интервал."
)
_CLOCK_UNKNOWN = "Длительность или смещение начала: не рассчитано."
_OUTSIDE_SOURCE = "Эпизод выходит за полуоткрытое окно источника."
_WINDOW_SLACK = 1e-6


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """One operator episode. ``label`` is the human label, not a split name."""

    episode_id: str
    video_id: str
    split_group_id: str
    start_s: float
    end_s: float
    label: str
    concurrent: bool = False


def validate_episodes(
    episodes: Sequence[EpisodeSpec],
    *,
    video_id: str | None = None,
) -> list[EpisodeSpec]:
    """Normalize episodes and reject illegal overlaps.

    Touching endpoints do not overlap. Episodes of two videos are checked
    separately. A group is concurrent only when every member is marked.
    """
    normalized = [_normalize(episode) for episode in episodes]
    if video_id is not None and any(episode.video_id != video_id for episode in normalized):
        raise LabError("video_mismatch", "Эпизод относится к другому видео.")
    ids = [episode.episode_id for episode in normalized]
    if len(ids) != len(set(ids)):
        raise LabError("duplicate_episode", "Идентификатор эпизода повторяется.")
    by_video: dict[str, list[EpisodeSpec]] = defaultdict(list)
    for episode in normalized:
        by_video[episode.video_id].append(episode)
    for rows in by_video.values():
        _reject_illegal_overlaps(rows)
    return sorted(normalized, key=lambda episode: (episode.start_s, episode.episode_id))


def replace_episodes(
    workspace: Path,
    project_id: str,
    video_id: str,
    episodes: Sequence[EpisodeSpec],
) -> list[EpisodeSpec]:
    """Replace one video's episodes after ``validate_episodes``.

    The delete and insert run in one SQLite transaction. A frozen split
    manifest refuses the write. A missing source clock, or an episode
    outside ``[start_offset_s, start_offset_s + duration_s)``, refuses the
    write and leaves the previous rows. Rows for any other video stay.
    """
    video = get_video(workspace, project_id, video_id)
    _reject_frozen_split(workspace, project_id, video_id)
    normalized = validate_episodes(episodes, video_id=video_id)
    _reject_outside_source_window(video, normalized)
    replace_video_episodes(
        workspace,
        project_id,
        video_id,
        [
            {
                "episode_id": episode.episode_id,
                "split_group_id": episode.split_group_id,
                "start_s": episode.start_s,
                "end_s": episode.end_s,
                "label": episode.label,
                "concurrent": episode.concurrent,
            }
            for episode in normalized
        ],
    )
    return load_episode_specs(workspace, project_id, video_id)


def specs_from_payload(video_id: str, episodes: object) -> list[EpisodeSpec]:
    """Build specs from the timeline JSON. Unknown shapes use ``invalid_episode``."""
    if not isinstance(episodes, list):
        raise LabError("invalid_episode", _BAD_EPISODE)
    return [_spec_from_payload(video_id, item) for item in episodes]


def add_episode(workspace: Path, project_id: str, episode: EpisodeSpec) -> EpisodeSpec:
    """Validate ``episode`` against the video's rows, then store it.

    A frozen split refuses the insert. This path does not map playback
    time onto the source window; the caller passes a source interval.
    """
    draft = _normalize(episode)
    get_video(workspace, project_id, draft.video_id)
    _reject_frozen_split(workspace, project_id, draft.video_id)
    existing = load_episode_specs(workspace, project_id, draft.video_id)
    validate_episodes([*existing, draft], video_id=draft.video_id)
    insert_episode(
        workspace,
        project_id,
        episode_id=draft.episode_id,
        video_id=draft.video_id,
        split_group_id=draft.split_group_id,
        start_s=draft.start_s,
        end_s=draft.end_s,
        label=draft.label,
        concurrent=draft.concurrent,
    )
    return draft


def load_episode_specs(workspace: Path, project_id: str, video_id: str) -> list[EpisodeSpec]:
    """Load the operator table for one video."""
    return [
        EpisodeSpec(
            episode_id=row["episode_id"],
            video_id=row["video_id"],
            split_group_id=row["split_group_id"],
            start_s=row["start_s"],
            end_s=row["end_s"],
            label=row["label"],
            concurrent=row["concurrent"],
        )
        for row in list_episodes(workspace, project_id, video_id)
    ]


def _reject_outside_source_window(
    video: Mapping[str, object],
    episodes: Sequence[EpisodeSpec],
) -> None:
    """Keep a timeline replace inside the probed source window.

    A missing offset is not 0. A stored 0 is a real start. ``duration_s``
    is the container length, so offset 300 and duration 10 run to 310.
    """
    offset = _known_offset(video.get("start_offset_s"))
    duration = _known_positive(video.get("duration_s"))
    if offset is None or duration is None:
        raise LabError("source_clock_unknown", _CLOCK_UNKNOWN)
    end = offset + duration
    for episode in episodes:
        if episode.start_s < offset - _WINDOW_SLACK or episode.end_s > end + _WINDOW_SLACK:
            raise LabError("episode_outside_source", _OUTSIDE_SOURCE)


def _known_offset(value: object) -> float | None:
    """Finite ``start_offset_s`` at or above 0, including a stored 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _known_positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _reject_frozen_split(workspace: Path, project_id: str, video_id: str) -> None:
    recorded = get_split_manifest(workspace, project_id, video_id)
    # Lazy import: splits loads this module.
    from excavation_lab.datasets.splits import _ALREADY_FROZEN, frozen_manifest_path

    path = frozen_manifest_path(workspace, project_id, video_id)
    if recorded is not None or path.is_file() or path.is_symlink():
        raise LabError("split_frozen", _ALREADY_FROZEN)


def _spec_from_payload(video_id: str, item: object) -> EpisodeSpec:
    if not isinstance(item, Mapping):
        raise LabError("invalid_episode", _BAD_EPISODE)
    episode_id = item.get("episode_id")
    split_group_id = item.get("split_group_id")
    label = item.get("label")
    start_s = item.get("start_s")
    end_s = item.get("end_s")
    concurrent = item.get("concurrent")
    if (
        not isinstance(episode_id, str)
        or not isinstance(split_group_id, str)
        or not isinstance(label, str)
    ):
        raise LabError("invalid_episode", _BAD_EPISODE)
    if (
        isinstance(start_s, bool)
        or isinstance(end_s, bool)
        or not isinstance(start_s, (int, float))
        or not isinstance(end_s, (int, float))
    ):
        raise LabError("invalid_episode", _BAD_EPISODE)
    if concurrent is not True and concurrent is not False:
        raise LabError(
            "invalid_episode",
            "Признак одновременности должен быть логическим.",
        )
    return EpisodeSpec(
        episode_id=episode_id,
        video_id=video_id,
        split_group_id=split_group_id,
        start_s=float(start_s),
        end_s=float(end_s),
        label=label,
        concurrent=concurrent,
    )


def _normalize(episode: EpisodeSpec) -> EpisodeSpec:
    if episode.concurrent is not True and episode.concurrent is not False:
        raise LabError(
            "invalid_episode",
            "Признак одновременности должен быть логическим.",
        )
    require_half_open_interval(episode.start_s, episode.end_s)
    if episode.start_s < 0:
        raise LabError(
            "invalid_interval",
            "Интервал должен быть полуоткрытым: end_s больше start_s.",
        )
    try:
        stored = Episode(
            episode_id=episode.episode_id,
            video_id=episode.video_id,
            split_group_id=episode.split_group_id,
            start_s=episode.start_s,
            end_s=episode.end_s,
            label=episode.label,
        )
    except ValidationError as exc:
        raise LabError("invalid_episode", _BAD_EPISODE) from exc
    return EpisodeSpec(
        episode_id=stored.episode_id,
        video_id=stored.video_id,
        split_group_id=stored.split_group_id,
        start_s=stored.start_s,
        end_s=stored.end_s,
        label=stored.label,
        concurrent=episode.concurrent,
    )


def _reject_illegal_overlaps(episodes: Sequence[EpisodeSpec]) -> None:
    concurrent_groups = _concurrent_groups(episodes)
    for index, left in enumerate(episodes):
        for right in episodes[index + 1 :]:
            if not _overlaps(left, right):
                continue
            same_group = left.split_group_id == right.split_group_id
            if same_group and left.split_group_id in concurrent_groups:
                continue
            raise LabError("episode_overlap", _overlap_message(left, right))


def _concurrent_groups(episodes: Sequence[EpisodeSpec]) -> set[str]:
    members: dict[str, list[EpisodeSpec]] = defaultdict(list)
    for episode in episodes:
        members[episode.split_group_id].append(episode)
    marked: set[str] = set()
    for group_id, rows in members.items():
        flags = {row.concurrent for row in rows}
        if flags == {True}:
            marked.add(group_id)
        elif flags != {False}:
            raise LabError(
                "inconsistent_concurrency",
                "В одной группе все эпизоды должны быть отмечены одновременными или нет.",
            )
    return marked


def _overlaps(left: EpisodeSpec, right: EpisodeSpec) -> bool:
    # Half-open [start, end). Touching at the endpoint is not an overlap.
    return left.start_s < right.end_s and right.start_s < left.end_s


def _overlap_message(left: EpisodeSpec, right: EpisodeSpec) -> str:
    if left.split_group_id == right.split_group_id:
        return (
            f"Эпизоды «{left.episode_id}» и «{right.episode_id}» пересекаются. "
            "Отметьте их split_group_id как одновременную или уберите пересечение."
        )
    return (
        f"Эпизоды «{left.episode_id}» и «{right.episode_id}» пересекаются. "
        "Оставьте их в одной split_group_id и отметьте группу как одновременную."
    )
