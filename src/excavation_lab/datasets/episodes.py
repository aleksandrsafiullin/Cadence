"""Operator episodes for one video.

Intervals are half-open ``[start_s, end_s)``. Overlap in one video is rejected
unless the episodes share a ``split_group_id`` the operator marked concurrent.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from excavation_lab.errors import LabError
from excavation_lab.schemas import Episode
from excavation_lab.storage import (
    get_video,
    insert_episode,
    list_episodes,
    require_half_open_interval,
)

_BAD_EPISODE = (
    "Эпизод задан некорректно: нужны идентификаторы, метка и полуоткрытый интервал."
)


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


def add_episode(workspace: Path, project_id: str, episode: EpisodeSpec) -> EpisodeSpec:
    """Validate ``episode`` against the video's rows, then store it."""
    draft = _normalize(episode)
    get_video(workspace, project_id, draft.video_id)
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
