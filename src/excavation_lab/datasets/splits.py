"""Propose a train/validation/test split without slicing an episode.

Groups are walked in time order. A group that crosses a duration cut moves
entirely into one split. A gap shorter than ``guard_gap_s`` is not a guard:
the later group moves into the earlier split. Frames in a real guard are
``excluded_guard``.

``freeze_split`` copies a proposal to ``split_manifest.json``. It does not
assign episodes again.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, model_validator

from excavation_lab.datasets.episodes import (
    EpisodeSpec,
    load_episode_specs,
    validate_episodes,
)
from excavation_lab.errors import LabError, SplitHashMismatchError, SplitNotFrozenError
from excavation_lab.schemas import (
    SplitFractions,
    SplitIndependence,
    SplitLimitation,
    SplitName,
    StrictModel,
    VersionedModel,
    load_pilot,
)
from excavation_lab.storage import (
    get_split_manifest,
    get_video,
    insert_split_manifest,
    project_directory,
    resolve_project_path,
)

SAME_SCENE_MESSAGE = (
    "Мало эпизодов для независимого разбиения. "
    "Это проверка той же сцены, не переносимость."
)

BlockRole = Literal["train", "validation", "test", "excluded_guard"]
_RANK = {SplitName.train: 0, SplitName.validation: 1, SplitName.test: 2}
_MIN_GUARD_S = 10.0
_MANIFEST_NOT_FOUND = "Файл предложения разбиения не найден."
_PROPOSAL_BROKEN = "Предложение разбиения не является JSON-объектом."
_PROPOSAL_INVALID = (
    "Предложение разбиения некорректно и не может быть зафиксировано."
)
_FROZEN_BROKEN = "Файл split_manifest.json повреждён."
_ALREADY_FROZEN = (
    "Разбиение этого видео уже зафиксировано. "
    "Повторная фиксация в этой версии не выполняется."
)
_VIDEO_MISMATCH = "Файл разбиения относится к другому видео."
_FROZEN_SCHEMA = "Файл split_manifest.json не соответствует схеме разбиения."


class ProposalEpisode(StrictModel):
    episode_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    split_group_id: str = Field(min_length=1)
    start_s: float = Field(ge=0)
    end_s: float
    label: str = Field(min_length=1)
    split: SplitName
    concurrent: bool = False

    @model_validator(mode="after")
    def half_open(self) -> ProposalEpisode:
        if self.end_s <= self.start_s:
            raise ValueError("episode interval must be half-open with end_s > start_s")
        return self


class GuardInterval(StrictModel):
    start_s: float = Field(ge=0)
    end_s: float
    role: Literal["excluded_guard"] = "excluded_guard"

    @model_validator(mode="after")
    def half_open(self) -> GuardInterval:
        if self.end_s <= self.start_s:
            raise ValueError("guard interval must be half-open")
        return self


class TimelineBlock(StrictModel):
    start_s: float = Field(ge=0)
    end_s: float
    role: BlockRole

    @model_validator(mode="after")
    def half_open(self) -> TimelineBlock:
        if self.end_s <= self.start_s:
            raise ValueError("block interval must be half-open")
        return self


class SplitProposal(VersionedModel):
    """Unfrozen proposal. ``freeze_split`` writes ``split_manifest.json``."""

    video_id: str = Field(min_length=1)
    frozen: Literal[False] = False
    independence: SplitIndependence
    limitation: SplitLimitation | None = None
    message: str | None = None
    guard_gap_s: float = Field(ge=_MIN_GUARD_S)
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    duration_s: float | None = None
    episodes: list[ProposalEpisode]
    guards: list[GuardInterval]
    blocks: list[TimelineBlock]

    @model_validator(mode="after")
    def proposal_rules(self) -> SplitProposal:
        _check_independence(self)
        _check_group_atomicity(self)
        _check_guard_gaps(self)
        if not self.episodes:
            _check_technical_blocks(self)
        return self

    def split_for(self, episode_id: str) -> SplitName:
        matches = [episode for episode in self.episodes if episode.episode_id == episode_id]
        if len(matches) != 1:
            raise LabError("episode_not_found", "Эпизод не найден в предложении.")
        return matches[0].split

    def assignment_at(self, time_s: float) -> str | None:
        """Split, ``excluded_guard``, or ``None`` when the instant is unlabeled."""
        splits = {
            episode.split.value
            for episode in self.episodes
            if episode.start_s <= time_s < episode.end_s
        }
        if len(splits) > 1:
            raise LabError("episode_overlap", "В одной точке времени разные split.")
        if len(splits) == 1:
            return next(iter(splits))
        for guard in self.guards:
            if guard.start_s <= time_s < guard.end_s:
                return guard.role
        for block in self.blocks:
            if block.start_s <= time_s < block.end_s:
                return block.role
        return None


class FrozenSplit(VersionedModel):
    """Immutable copy of a proposal plus ``content_hash``."""

    video_id: str = Field(min_length=1)
    frozen: Literal[True] = True
    independence: SplitIndependence
    limitation: SplitLimitation | None = None
    message: str | None = None
    guard_gap_s: float = Field(ge=_MIN_GUARD_S)
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    duration_s: float | None = None
    episodes: list[ProposalEpisode]
    guards: list[GuardInterval]
    blocks: list[TimelineBlock]
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def propose_split(
    video_id: str,
    episodes: Sequence[EpisodeSpec],
    *,
    duration_s: float | None = None,
    guard_gap_s: float | None = None,
    fractions: SplitFractions | None = None,
) -> SplitProposal:
    """Assign whole groups. Do not permute frames before the cut.

    Fractions apply to the sum of episode durations. With no episodes the
    video duration is split into three sequential blocks and the same
    same-scene limitation. That result is not ``episode_split``.
    """
    pilot = load_pilot()
    gap = pilot.guard_gap_s if guard_gap_s is None else guard_gap_s
    parts = pilot.split_fractions if fractions is None else fractions
    _require_guard(gap)
    specs = validate_episodes(list(episodes), video_id=video_id)
    recorded = _optional_duration(duration_s)
    independence, limitation, message = _independence(specs)
    if not specs:
        if recorded is None:
            raise LabError(
                "duration_missing",
                "У видео нет длительности. Сначала выполните probe, "
                "затем предложите разбиение.",
            )
        guards, blocks = _technical_partition(recorded, gap, parts)
        episodes_out: list[ProposalEpisode] = []
    else:
        assigned = _assign_groups(specs, parts)
        _close_short_gaps(assigned, gap)
        episodes_out = _proposal_episodes(assigned)
        guards = _guards_between(episodes_out)
        blocks = _episode_blocks(episodes_out, guards)
    return SplitProposal(
        video_id=video_id,
        frozen=False,
        independence=independence,
        limitation=limitation,
        message=message,
        guard_gap_s=gap,
        train_fraction=parts.train,
        validation_fraction=parts.validation,
        test_fraction=parts.test,
        duration_s=recorded,
        episodes=episodes_out,
        guards=guards,
        blocks=blocks,
    )


def propose_for_project(workspace: Path, project_id: str, video_id: str) -> SplitProposal:
    """Propose from the operator table. Does not freeze."""
    video = get_video(workspace, project_id, video_id)
    specs = load_episode_specs(workspace, project_id, video_id)
    return propose_split(video_id, specs, duration_s=video["duration_s"])


def write_proposal(workspace: Path, project_id: str, proposal: SplitProposal) -> Path:
    """Write the proposal JSON. This is not ``split_manifest.json``."""
    leaf = _safe_video_id(proposal.video_id)
    directory = project_directory(workspace, project_id) / "split_proposals"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{leaf}.json"
    payload = proposal.model_dump(mode="json")
    payload["frozen"] = False
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """UTF-8 JSON with sorted keys and compact separators.

    These bytes have no trailing newline. ``content_hash`` is the SHA-256 of
    this encoding of the manifest object without the ``content_hash`` field.
    """
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LabError(
            "invalid_manifest",
            "Разбиение не сериализуется в канонический JSON.",
        ) from exc
    return text.encode("utf-8")


def frozen_manifest_path(workspace: Path, project_id: str, video_id: str) -> Path:
    """``split_manifests/<video_id>/split_manifest.json`` in the project store."""
    leaf = _safe_video_id(video_id)
    return (
        project_directory(workspace, project_id)
        / "split_manifests"
        / leaf
        / "split_manifest.json"
    )


def freeze_split(
    workspace: Path,
    project_id: str,
    manifest: Path,
) -> tuple[FrozenSplit, Path]:
    """Copy ``manifest`` into the project store as ``split_manifest.json``.

    Episodes stay on the splits already written in the proposal. A second
    freeze of the same video fails with ``split_frozen``.
    """
    resolved = resolve_project_path(workspace, project_id, manifest)
    if not resolved.is_file():
        raise LabError("manifest_not_found", _MANIFEST_NOT_FOUND)
    proposal = _load_proposal(resolved)
    get_video(workspace, project_id, proposal.video_id)
    destination = frozen_manifest_path(workspace, project_id, proposal.video_id)
    recorded = get_split_manifest(workspace, project_id, proposal.video_id)
    if destination.exists() or destination.is_symlink() or recorded is not None:
        raise LabError("split_frozen", _ALREADY_FROZEN)
    body = proposal.model_dump(mode="json")
    body["frozen"] = True
    body.pop("content_hash", None)
    digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    document = dict(body)
    document["content_hash"] = digest
    frozen = FrozenSplit.model_validate(document)
    _write_exact(destination, canonical_json_bytes(document))
    try:
        insert_split_manifest(
            workspace,
            project_id,
            video_id=proposal.video_id,
            content_hash=digest,
            path=destination,
        )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return frozen, destination


def require_frozen(workspace: Path, project_id: str, video_id: str) -> FrozenSplit:
    """Return the frozen split. Missing ``split_manifest.json`` is an error."""
    get_video(workspace, project_id, video_id)
    path = frozen_manifest_path(workspace, project_id, video_id)
    if not path.is_file():
        raise SplitNotFrozenError(video_id)
    return load_frozen(workspace, project_id, video_id)


def load_frozen(workspace: Path, project_id: str, video_id: str) -> FrozenSplit:
    """Read a frozen split and reject a body that no longer matches its hash."""
    get_video(workspace, project_id, video_id)
    path = frozen_manifest_path(workspace, project_id, video_id)
    if not path.is_file():
        raise SplitNotFrozenError(video_id)
    document = _read_json_object(path, broken=_FROZEN_BROKEN)
    stored = document.get("content_hash")
    body = {key: value for key, value in document.items() if key != "content_hash"}
    digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if not isinstance(stored, str) or stored != digest:
        raise SplitHashMismatchError()
    recorded = get_split_manifest(workspace, project_id, video_id)
    if recorded is not None and recorded.get("content_hash") != digest:
        raise SplitHashMismatchError()
    if document.get("video_id") != video_id:
        raise LabError("video_mismatch", _VIDEO_MISMATCH)
    try:
        return FrozenSplit.model_validate(document)
    except ValidationError as exc:
        raise LabError("invalid_manifest", _FROZEN_SCHEMA) from exc


def _load_proposal(path: Path) -> SplitProposal:
    document = _read_json_object(path, broken=_PROPOSAL_BROKEN)
    try:
        return SplitProposal.model_validate(document)
    except ValidationError as exc:
        raise LabError("invalid_manifest", _PROPOSAL_INVALID) from exc


def _read_json_object(path: Path, *, broken: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LabError("invalid_manifest", broken) from exc
    if raw.endswith(b"\n"):
        raw = raw[:-1]
    if raw.endswith(b"\r"):
        raw = raw[:-1]
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LabError("invalid_manifest", broken) from exc
    if not isinstance(parsed, dict):
        raise LabError("invalid_manifest", broken)
    return parsed


def _write_exact(path: Path, payload: bytes) -> None:
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".split-manifest-",
        suffix=".tmp",
        dir=directory,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class _Group:
    split_group_id: str
    episodes: tuple[EpisodeSpec, ...]

    @property
    def duration_s(self) -> float:
        return sum(episode.end_s - episode.start_s for episode in self.episodes)

    @property
    def start_s(self) -> float:
        return min(episode.start_s for episode in self.episodes)


def _assign_groups(
    specs: Sequence[EpisodeSpec],
    fractions: SplitFractions,
) -> list[tuple[_Group, SplitName]]:
    groups = _groups_in_time_order(specs)
    total = sum(group.duration_s for group in groups)
    train_cut = fractions.train * total
    val_cut = (fractions.train + fractions.validation) * total
    assigned: list[tuple[_Group, SplitName]] = []
    cursor = 0.0
    for group in groups:
        start = cursor
        end = cursor + group.duration_s
        cursor = end
        assigned.append((group, _split_for_span(start, end, train_cut, val_cut)))
    return assigned


def _close_short_gaps(
    assigned: list[tuple[_Group, SplitName]],
    guard_gap_s: float,
) -> None:
    """Move a whole later group so a short gap is not called a guard."""
    limit = len(assigned) * len(assigned) + 1
    for _ in range(limit):
        pair = _first_short_pair(assigned, guard_gap_s)
        if pair is None:
            return
        left_index, right_index = pair
        left, left_split = assigned[left_index]
        right, right_split = assigned[right_index]
        earlier = left_split if _RANK[left_split] <= _RANK[right_split] else right_split
        assigned[left_index] = (left, earlier)
        assigned[right_index] = (right, earlier)
    raise LabError(
        "guard_too_small",
        "Не удалось развести части защитным интервалом.",
    )


def _first_short_pair(
    assigned: Sequence[tuple[_Group, SplitName]],
    guard_gap_s: float,
) -> tuple[int, int] | None:
    for left_index, (left, left_split) in enumerate(assigned):
        for right_index in range(left_index + 1, len(assigned)):
            right, right_split = assigned[right_index]
            if left_split == right_split:
                continue
            if _closer_than_guard(left, right, guard_gap_s):
                return left_index, right_index
    return None


def _closer_than_guard(left: _Group, right: _Group, guard_gap_s: float) -> bool:
    return _closest(left, right) + 1e-9 < guard_gap_s


def _closest(left: _Group, right: _Group) -> float:
    return min(
        _separation(a.start_s, a.end_s, b.start_s, b.end_s)
        for a in left.episodes
        for b in right.episodes
    )


def _half_open_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return start_a < end_b and start_b < end_a


def _separation(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    if end_a <= start_b:
        return start_b - end_a
    if end_b <= start_a:
        return start_a - end_b
    return 0.0


def _split_for_span(start: float, end: float, train_cut: float, val_cut: float) -> SplitName:
    """Majority of the duration span. A tie stays with the earlier split."""
    bands = (
        (SplitName.train, 0.0, train_cut),
        (SplitName.validation, train_cut, val_cut),
        (SplitName.test, val_cut, math.inf),
    )
    best = SplitName.train
    best_length = -1.0
    for name, band_start, band_end in bands:
        length = _overlap(start, end, band_start, band_end)
        if length > best_length:
            best = name
            best_length = length
    return best


def _overlap(start: float, end: float, band_start: float, band_end: float) -> float:
    return max(0.0, min(end, band_end) - max(start, band_start))


def _groups_in_time_order(specs: Sequence[EpisodeSpec]) -> list[_Group]:
    members: dict[str, list[EpisodeSpec]] = {}
    for spec in specs:
        members.setdefault(spec.split_group_id, []).append(spec)
    groups = [
        _Group(
            group_id,
            tuple(sorted(rows, key=lambda episode: (episode.start_s, episode.episode_id))),
        )
        for group_id, rows in members.items()
    ]
    groups.sort(key=lambda group: (group.start_s, group.split_group_id))
    return groups


def _proposal_episodes(
    assigned: Sequence[tuple[_Group, SplitName]],
) -> list[ProposalEpisode]:
    rows: list[ProposalEpisode] = []
    for group, split in assigned:
        for spec in group.episodes:
            rows.append(
                ProposalEpisode(
                    episode_id=spec.episode_id,
                    video_id=spec.video_id,
                    split_group_id=spec.split_group_id,
                    start_s=spec.start_s,
                    end_s=spec.end_s,
                    label=spec.label,
                    split=split,
                    concurrent=spec.concurrent,
                )
            )
    rows.sort(key=lambda episode: (episode.start_s, episode.episode_id))
    return rows


@dataclass
class _Span:
    start_s: float
    end_s: float
    split: SplitName


def _guards_between(episodes: Sequence[ProposalEpisode]) -> list[GuardInterval]:
    """The whole uncovered gap between different splits, not a 10 s slice of it."""
    if len(episodes) < 2:
        return []
    ordered = sorted(episodes, key=lambda episode: (episode.start_s, episode.end_s))
    merged: list[_Span] = []
    for episode in ordered:
        if not merged or episode.start_s > merged[-1].end_s:
            merged.append(_Span(episode.start_s, episode.end_s, episode.split))
            continue
        if merged[-1].split != episode.split:
            raise LabError(
                "episode_overlap",
                "Пересекающиеся эпизоды попали в разные части разбиения.",
            )
        merged[-1].end_s = max(merged[-1].end_s, episode.end_s)
    guards: list[GuardInterval] = []
    for left, right in pairwise(merged):
        if right.start_s <= left.end_s or left.split == right.split:
            continue
        guards.append(GuardInterval(start_s=left.end_s, end_s=right.start_s))
    return guards


def _episode_blocks(
    episodes: Sequence[ProposalEpisode],
    guards: Sequence[GuardInterval],
) -> list[TimelineBlock]:
    blocks = [
        TimelineBlock(start_s=episode.start_s, end_s=episode.end_s, role=episode.split.value)
        for episode in episodes
    ]
    blocks.extend(
        TimelineBlock(start_s=guard.start_s, end_s=guard.end_s, role="excluded_guard")
        for guard in guards
    )
    blocks.sort(key=lambda block: (block.start_s, block.end_s, block.role))
    return blocks


def _technical_partition(
    duration_s: float,
    guard_gap_s: float,
    fractions: SplitFractions,
) -> tuple[list[GuardInterval], list[TimelineBlock]]:
    assignable = duration_s - 2.0 * guard_gap_s
    if assignable <= 0:
        raise LabError(
            "duration_too_short",
            "Видео короче двух защитных интервалов. "
            "Три последовательных блока не помещаются.",
        )
    train_end = fractions.train * assignable
    guard1_end = train_end + guard_gap_s
    val_end = guard1_end + fractions.validation * assignable
    guard2_end = val_end + guard_gap_s
    if not guard2_end < duration_s:
        raise LabError(
            "duration_too_short",
            "Видео короче двух защитных интервалов. "
            "Три последовательных блока не помещаются.",
        )
    guards = [
        GuardInterval(start_s=train_end, end_s=guard1_end),
        GuardInterval(start_s=val_end, end_s=guard2_end),
    ]
    blocks = [
        TimelineBlock(start_s=0.0, end_s=train_end, role="train"),
        TimelineBlock(start_s=train_end, end_s=guard1_end, role="excluded_guard"),
        TimelineBlock(start_s=guard1_end, end_s=val_end, role="validation"),
        TimelineBlock(start_s=val_end, end_s=guard2_end, role="excluded_guard"),
        TimelineBlock(start_s=guard2_end, end_s=duration_s, role="test"),
    ]
    return guards, blocks


def _independence(
    specs: Sequence[EpisodeSpec],
) -> tuple[SplitIndependence, SplitLimitation | None, str | None]:
    groups = {spec.split_group_id for spec in specs}
    if len(specs) >= 3 and len(groups) >= 3:
        return SplitIndependence.episode_split, None, None
    return SplitIndependence.same_scene, SplitLimitation.too_few_episodes, SAME_SCENE_MESSAGE


def _check_independence(proposal: SplitProposal) -> None:
    groups = {episode.split_group_id for episode in proposal.episodes}
    enough = len(proposal.episodes) >= 3 and len(groups) >= 3
    if enough:
        if proposal.independence is not SplitIndependence.episode_split:
            raise ValueError("three groups are an episode split")
        if proposal.limitation is not None or proposal.message is not None:
            raise ValueError("episode_split has no same-scene limitation")
        return
    if proposal.independence is not SplitIndependence.same_scene:
        raise ValueError("too few episodes is same_scene")
    if proposal.limitation is not SplitLimitation.too_few_episodes:
        raise ValueError("same_scene proposal requires too_few_episodes")
    if proposal.message != SAME_SCENE_MESSAGE:
        raise ValueError("same_scene proposal requires the limitation message")


def _check_group_atomicity(proposal: SplitProposal) -> None:
    ids = [episode.episode_id for episode in proposal.episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("episode sliced into multiple rows")
    found: dict[str, SplitName] = {}
    for episode in proposal.episodes:
        previous = found.get(episode.split_group_id)
        if previous is not None and previous != episode.split:
            raise ValueError("split_group_id has two splits")
        found[episode.split_group_id] = episode.split


def _check_guard_gaps(proposal: SplitProposal) -> None:
    for guard in proposal.guards:
        if guard.end_s - guard.start_s + 1e-9 < proposal.guard_gap_s:
            raise ValueError("guard shorter than guard_gap_s")
        for episode in proposal.episodes:
            if _half_open_overlap(
                guard.start_s,
                guard.end_s,
                episode.start_s,
                episode.end_s,
            ):
                raise ValueError("guard overlaps an episode")
    episodes = proposal.episodes
    for index, left in enumerate(episodes):
        for right in episodes[index + 1 :]:
            if left.split == right.split:
                continue
            gap = _separation(left.start_s, left.end_s, right.start_s, right.end_s)
            if gap + 1e-9 < proposal.guard_gap_s:
                raise ValueError("different splits are closer than guard_gap_s")


def _check_technical_blocks(proposal: SplitProposal) -> None:
    if proposal.duration_s is None:
        raise ValueError("technical split requires duration_s")
    roles = [block.role for block in proposal.blocks]
    if roles != ["train", "excluded_guard", "validation", "excluded_guard", "test"]:
        raise ValueError("technical split must be three sequential blocks")
    if abs(proposal.blocks[0].start_s) > 1e-9:
        raise ValueError("technical split must start at 0")
    if abs(proposal.blocks[-1].end_s - proposal.duration_s) > 1e-6:
        raise ValueError("technical split must cover the video")
    for prev, nxt in pairwise(proposal.blocks):
        if abs(prev.end_s - nxt.start_s) > 1e-6:
            raise ValueError("technical blocks must meet")
    if proposal.independence is SplitIndependence.episode_split:
        raise ValueError("a video with no episodes is not an episode split")


def _require_guard(guard_gap_s: float) -> None:
    if (
        isinstance(guard_gap_s, bool)
        or not isinstance(guard_gap_s, (int, float))
        or not math.isfinite(guard_gap_s)
        or guard_gap_s < _MIN_GUARD_S
    ):
        raise LabError(
            "invalid_guard_gap",
            "Защитный интервал меньше 10 с. Возьмите guard_gap_s из конфигурации.",
        )


def _optional_duration(duration_s: float | None) -> float | None:
    if duration_s is None:
        return None
    if (
        isinstance(duration_s, bool)
        or not isinstance(duration_s, (int, float))
        or not math.isfinite(duration_s)
        or duration_s <= 0
    ):
        return None
    return float(duration_s)


def _safe_video_id(video_id: str) -> str:
    if (
        not video_id
        or video_id in {".", ".."}
        or "/" in video_id
        or "\\" in video_id
        or "\x00" in video_id
    ):
        raise LabError("invalid_video_id", "Недопустимый идентификатор видео.")
    return video_id
