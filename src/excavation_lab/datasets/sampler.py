"""Sample frames from a frozen split.

The selector reads ``require_frozen`` and does not build a new split.
Validation and test use an even stride across episodes. A perceptual hash
may mark a near-duplicate; that row is still written.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Literal, Protocol, Self

import cv2
import numpy as np
from pydantic import Field, model_validator

from excavation_lab.datasets.splits import FrozenSplit, require_frozen
from excavation_lab.errors import LabError, MediaNotFoundError, TimebaseError
from excavation_lab.media.reader import FrameReader, ReadFrame
from excavation_lab.schemas import (
    DecodeStatus,
    SplitCounts,
    SplitName,
    StrictModel,
    VersionedModel,
    load_pilot,
)
from excavation_lab.storage import connect, get_video, project_directory, project_sqlite

StrategyName = Literal["spaced", "representative", "active"]
PlanName = Literal["diagnostic", "content", "expanded"]

_SPACING_TOLERANCE_S = 1e-3
_NEAR_DUPLICATE_HAMMING = 8
_JPEG_QUALITY = 95
_NEAR_WARNING = (
    "Кадр похож на уже отобранный. Он оставлен: похожий хеш не удаляет кадр."
)
_SPACING_MESSAGE = (
    "Кадры одного эпизода ближе, чем sample_spacing_s. "
    "Такой отбор допустим только с непустой selection_reason."
)
_REJECTED_STRATEGIES = {
    "active": "Стратегия active в этом отборе не выполняется.",
    "easiest": "Отбор самых удобных кадров не выполняется.",
    "hardest": "Отбор самых сложных кадров не выполняется.",
}
_ACTIVE_REASONS = frozenset(
    {
        "low_confidence",
        "bucket_missing_in_working_area",
        "coordinate_jump",
        "ambiguous_bed",
        "false_positive",
    }
)


class _SpacingRow(Protocol):
    episode_id: str
    source_time_s: float
    source_frame_index: int
    selection_reason: str


class CloserPick(StrictModel):
    """A train frame closer than ``sample_spacing_s``. The reason is required."""

    source_frame_index: int = Field(ge=0)
    selection_reason: str = Field(min_length=1)


class SampleRow(StrictModel):
    video_id: str = Field(min_length=1)
    source_frame_index: int = Field(ge=0)
    source_time_s: float = Field(ge=0)
    episode_id: str
    split: SplitName
    selection_reason: str = ""
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_file: str = Field(min_length=1)
    strategy: StrategyName
    model_version: str | None = None
    near_duplicate_warning: str | None = None
    near_duplicate_of: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def jpeg_name_and_warning(self) -> Self:
        expected = f"{self.video_id}_{self.source_frame_index}.jpg"
        if self.image_file != expected or "/" in self.image_file or "\\" in self.image_file:
            raise ValueError("jpeg name must be <video_id>_<source_frame_index>.jpg")
        warned = self.near_duplicate_warning is not None
        linked = self.near_duplicate_of is not None
        if warned != linked:
            raise ValueError("near-duplicate warning and reference are set together")
        if self.near_duplicate_of == self.source_frame_index:
            raise ValueError("a frame is not a near-duplicate of itself")
        if warned and not self.near_duplicate_warning.strip():
            raise ValueError("near-duplicate warning must not be empty")
        return self

    @model_validator(mode="after")
    def active_is_train_with_a_reason(self) -> Self:
        if self.strategy != "active":
            return self
        if self.split is not SplitName.train:
            raise ValueError("active sample is train only")
        if self.selection_reason not in _ACTIVE_REASONS:
            raise ValueError("active sample needs a known selection_reason")
        if self.model_version is None or not self.model_version.strip():
            raise ValueError("active sample needs model_version")
        return self


class SampleResult(VersionedModel):
    project_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1)
    split_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan: PlanName | None = None
    spacing_s: float = Field(gt=0)
    requested: SplitCounts
    written: SplitCounts
    shortfall: SplitCounts
    rows: list[SampleRow]
    manifest_path: str = Field(min_length=1)
    images_dir: str = Field(min_length=1)
    reason_reports: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def counts_match_rows(self) -> Self:
        seen: set[int] = set()
        tallies = {name.value: 0 for name in SplitName}
        for row in self.rows:
            if row.video_id != self.video_id:
                raise ValueError("row video_id does not match the sample")
            if row.source_frame_index in seen:
                raise ValueError("duplicate source_frame_index")
            seen.add(row.source_frame_index)
            tallies[row.split.value] += 1
            if getattr(self.requested, row.split.value) == 0:
                raise ValueError("row belongs to a split that was not requested")
        for name in ("train", "validation", "test"):
            requested = getattr(self.requested, name)
            written = getattr(self.written, name)
            shortfall = getattr(self.shortfall, name)
            if tallies[name] != written:
                raise ValueError("written count does not match rows")
            if written + shortfall != requested:
                raise ValueError("shortfall must be requested minus written")
        return self


@dataclass(frozen=True, slots=True)
class FrameTick:
    video_id: str
    source_frame_index: int
    source_time_s: float
    episode_id: str
    split: SplitName


@dataclass(frozen=True, slots=True)
class SelectedTick:
    video_id: str
    source_frame_index: int
    source_time_s: float
    episode_id: str
    split: SplitName
    selection_reason: str
    strategy: StrategyName
    model_version: str | None = None


@dataclass(frozen=True, slots=True)
class _Window:
    episode_id: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class _Encoded:
    selected: SelectedTick
    image_sha256: str
    image_file: str
    phash: int


@dataclass(frozen=True, slots=True)
class _Request:
    plan: PlanName | None
    counts: Mapping[SplitName, int]
    strategies: Mapping[SplitName, StrategyName]
    spacing_s: float
    extras: tuple[CloserPick, ...]


def plan_counts(plan: str) -> SplitCounts:
    """Counts from ``configs/pilot.yaml``. A zero test count stays zero."""
    pilot = load_pilot()
    if plan == "diagnostic":
        return pilot.sample_plans.diagnostic
    if plan == "content":
        return pilot.sample_plans.content
    if plan == "expanded":
        return pilot.sample_plans.expanded
    raise LabError(
        "unknown_plan",
        "Неизвестный план отбора. Допустимы diagnostic, content и expanded.",
    )


def require_sample_spacing(rows: Sequence[_SpacingRow], spacing_s: float) -> None:
    """A pair closer than ``spacing_s`` needs a reason on one of the two frames.

    The marked frame is the rare pick. It may sit between ordinary frames that
    are still ``spacing_s`` apart from each other.
    """
    _check_spacing_value(spacing_s)
    grouped: dict[str, list[_SpacingRow]] = defaultdict(list)
    for row in rows:
        grouped[row.episode_id].append(row)
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda row: (row.source_time_s, row.source_frame_index),
        )
        for earlier, later in pairwise(ordered):
            if not _is_closer(earlier.source_time_s, later.source_time_s, spacing_s):
                continue
            if earlier.selection_reason.strip() or later.selection_reason.strip():
                continue
            raise LabError("sample_spacing", _SPACING_MESSAGE)


def select_ticks(
    ticks: Sequence[FrameTick],
    *,
    count: int,
    spacing_s: float,
    strategy: StrategyName,
    split: str,
    extras: Sequence[CloserPick] = (),
) -> list[SelectedTick]:
    """Pick indexes. ``representative`` strides episodes. It does not rank difficulty."""
    _check_count(count)
    _check_spacing_value(spacing_s)
    if strategy not in ("spaced", "representative"):
        raise LabError(
            "unknown_strategy",
            "Неизвестная стратегия отбора. Допустимы spaced и representative.",
        )
    try:
        expected = SplitName(split)
    except ValueError as exc:
        raise LabError(
            "unknown_split",
            "Неизвестная часть выборки. Допустимы train, validation и test.",
        ) from exc
    if any(tick.split != expected for tick in ticks):
        raise LabError(
            "sample_split",
            "В отбор попал кадр другой части. Он не записывается.",
        )
    if count == 0:
        return []
    pool = _dedupe(ticks)
    if strategy == "representative":
        if extras:
            raise LabError(
                "sample_args",
                "Дополнительный кадр можно добавить только в train при отборе с интервалом.",
            )
        selected = _representative(pool, count)
    else:
        selected = _spaced(pool, count, spacing_s)
        selected = _apply_extras(selected, extras, pool, count, expected.value)
    selected.sort(key=lambda item: (item.source_time_s, item.source_frame_index))
    require_sample_spacing(selected, spacing_s)
    return selected


def sample_frames(
    workspace: Path,
    project_id: str,
    *,
    video_id: str | None = None,
    split: str | None = None,
    count: int | None = None,
    plan: str | None = None,
    strategy: str | None = None,
    spacing_s: float | None = None,
    extras: Sequence[CloserPick] | None = None,
    model: str | None = None,
) -> SampleResult:
    """Write JPEGs for one frozen video. A short clip returns ``shortfall``.

    ``strategy="active"`` needs ``model``. Without it the call stops as an
    unknown strategy and does not open the project. With it, active learning
    reads saved detections and still goes through the frozen split.
    """
    if strategy == "active":
        if model is None or not str(model).strip():
            raise LabError("unknown_strategy", _REJECTED_STRATEGIES["active"])
        from excavation_lab.datasets.active_learning import sample_active

        return sample_active(
            workspace,
            project_id,
            model_id=str(model),
            video_id=video_id,
            split=split,
            count=count,
            plan=plan,
            spacing_s=spacing_s,
            extras=extras,
        )
    request = _parse_request(
        split=split,
        count=count,
        plan=plan,
        strategy=strategy,
        spacing_s=spacing_s,
        extras=extras,
    )
    resolved_id = _resolve_video_id(workspace, project_id, video_id)
    frozen = require_frozen(workspace, project_id, resolved_id)
    video = get_video(workspace, project_id, resolved_id)
    media_path = _media_path(video["path"])
    reader = FrameReader(media_path, video_id=resolved_id)
    chosen: list[SelectedTick] = []
    windows_by_split: dict[SplitName, list[_Window]] = {}
    for split_name in (SplitName.train, SplitName.validation, SplitName.test):
        wanted = request.counts[split_name]
        # Zero is the plan. Do not fill test from validation or train.
        if wanted == 0:
            continue
        windows = _windows(frozen, split_name)
        windows_by_split[split_name] = windows
        ticks = _collect_ticks(reader, frozen, split_name, windows)
        picked = select_ticks(
            ticks,
            count=wanted,
            spacing_s=request.spacing_s,
            strategy=request.strategies[split_name],
            split=split_name.value,
            extras=request.extras if split_name is SplitName.train else (),
        )
        if len(picked) > wanted:
            raise LabError(
                "sample_internal",
                "Отбор вернул больше кадров, чем запрошено. Лишние кадры не записываются.",
            )
        chosen.extend(picked)
    _reject_duplicate_indexes(chosen)
    require_sample_spacing(chosen, request.spacing_s)
    manifest_path, images_dir = _artifact_paths(
        project_directory(workspace, project_id),
        resolved_id,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if chosen:
        images_dir.mkdir(parents=True, exist_ok=True)
    encoded = _write_chosen(reader, windows_by_split, chosen, images_dir)
    rows = _rows_with_warnings(encoded)
    requested = _split_counts(request.counts)
    written, shortfall = _written_counts(requested, rows)
    result = SampleResult(
        project_id=project_id,
        video_id=resolved_id,
        split_content_hash=frozen.content_hash,
        plan=request.plan,
        spacing_s=request.spacing_s,
        requested=requested,
        written=written,
        shortfall=shortfall,
        rows=rows,
        manifest_path=str(manifest_path),
        images_dir=str(images_dir),
    )
    manifest_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def format_sample_summary(result: SampleResult) -> str:
    """Russian counts. Shortfall stays visible when the clip is shorter than the plan."""

    def line(name: str, label: str) -> str:
        requested = getattr(result.requested, name)
        written = getattr(result.written, name)
        shortfall = getattr(result.shortfall, name)
        return f"{label}: записано {written} из {requested}, не хватает {shortfall}."

    lines = [
        f"Видео {result.video_id}. Интервал {result.spacing_s:.1f} с.",
        line("train", "train"),
        line("validation", "validation"),
        line("test", "test"),
        "Записаны только прочитанные кадры.",
    ]
    if result.reason_reports.get("bucket_missing_in_working_area") == "zone_missing":
        lines.append("Кадры без полигона рабочей зоны пропущены: zone_missing.")
    return "\n".join(lines)


def _parse_request(
    *,
    split: str | None,
    count: int | None,
    plan: str | None,
    strategy: str | None,
    spacing_s: float | None,
    extras: Sequence[CloserPick] | None,
) -> _Request:
    if strategy in _REJECTED_STRATEGIES:
        raise LabError("unknown_strategy", _REJECTED_STRATEGIES[strategy])
    if strategy is not None and strategy not in ("spaced", "representative"):
        raise LabError(
            "unknown_strategy",
            "Неизвестная стратегия отбора. Допустимы spaced и representative.",
        )
    checked_extras = _checked_extras(extras)
    if plan is not None and (split is not None or count is not None or checked_extras):
        raise LabError(
            "sample_args",
            "План задаёт числа по частям. Не сочетайте план со split, count "
            "или дополнительным кадром.",
        )
    if plan is not None:
        counts_model = plan_counts(plan)
        plan_name: PlanName = plan_counts_name(plan)
        counts = {
            SplitName.train: counts_model.train,
            SplitName.validation: counts_model.validation,
            SplitName.test: counts_model.test,
        }
    else:
        plan_name = None
        if split is None or count is None:
            raise LabError(
                "sample_args",
                "Укажите план либо часть выборки и число кадров.",
            )
        parsed_split = _parse_split(split)
        _check_count(count)
        if count < 1:
            raise LabError(
                "sample_args",
                "Число кадров должно быть целым и больше нуля.",
            )
        counts = {
            SplitName.train: 0,
            SplitName.validation: 0,
            SplitName.test: 0,
        }
        counts[parsed_split] = count
    strategies = {
        name: _strategy_for(name, strategy) for name in counts
    }
    if checked_extras:
        targets = [name for name, wanted in counts.items() if wanted > 0]
        train_spaced = (
            targets == [SplitName.train] and strategies[SplitName.train] == "spaced"
        )
        if not train_spaced:
            raise LabError(
                "sample_args",
                "Дополнительный кадр можно добавить только в train при отборе с интервалом.",
            )
    return _Request(
        plan=plan_name,
        counts=counts,
        strategies=strategies,
        spacing_s=_spacing_or_config(spacing_s),
        extras=checked_extras,
    )


def plan_counts_name(plan: str) -> PlanName:
    if plan in ("diagnostic", "content", "expanded"):
        return plan
    raise LabError(
        "unknown_plan",
        "Неизвестный план отбора. Допустимы diagnostic, content и expanded.",
    )


def _checked_extras(extras: Sequence[CloserPick] | None) -> tuple[CloserPick, ...]:
    if not extras:
        return ()
    checked: list[CloserPick] = []
    for item in extras:
        if not isinstance(item, CloserPick):
            raise LabError("sample_args", "Дополнительный кадр задан некорректно.")
        if not item.selection_reason.strip():
            raise LabError("sample_spacing", _SPACING_MESSAGE)
        checked.append(item)
    return tuple(checked)


def _parse_split(split: str) -> SplitName:
    try:
        return SplitName(split)
    except ValueError as exc:
        raise LabError(
            "unknown_split",
            "Неизвестная часть выборки. Допустимы train, validation и test.",
        ) from exc


def _strategy_for(split: SplitName, requested: str | None) -> StrategyName:
    # Validation and test stay on the stride. They are not easiest or hardest.
    if split is not SplitName.train:
        return "representative"
    if requested == "representative":
        return "representative"
    return "spaced"


def _spacing_or_config(spacing_s: float | None) -> float:
    if spacing_s is None:
        return load_pilot().sample_spacing_s
    _check_spacing_value(spacing_s)
    return float(spacing_s)


def _check_spacing_value(spacing_s: float) -> None:
    if (
        isinstance(spacing_s, bool)
        or not isinstance(spacing_s, (int, float))
        or not math.isfinite(spacing_s)
        or spacing_s <= 0
    ):
        raise LabError(
            "sample_args",
            "Интервал отбора должен быть положительным числом секунд.",
        )


def _check_count(count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise LabError(
            "sample_args",
            "Число кадров должно быть целым и не меньше нуля.",
        )


def _resolve_video_id(workspace: Path, project_id: str, video_id: str | None) -> str:
    if video_id is not None:
        if not isinstance(video_id, str) or not video_id.strip() or "\x00" in video_id:
            raise LabError("invalid_video_id", "Недопустимый идентификатор видео.")
        return video_id
    db_path = project_sqlite(workspace, project_id)
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT video_id FROM videos
            WHERE project_id = ?
            ORDER BY video_id
            """,
            (project_id,),
        ).fetchall()
    ids = [row["video_id"] for row in rows]
    if not ids:
        raise LabError("video_not_found", "В проекте нет видео.")
    if len(ids) > 1:
        raise LabError(
            "video_required",
            "В проекте несколько видео. Укажите --video.",
        )
    return ids[0]


def _media_path(stored: object) -> Path:
    if not isinstance(stored, str) or not stored:
        raise LabError(
            "video_path_missing",
            "У видео нет локального файла. Зарегистрируйте путь к MP4, MOV или MKV.",
        )
    path = Path(stored)
    if not path.is_file():
        raise MediaNotFoundError(str(path))
    return path


def _windows(frozen: FrozenSplit, split: SplitName) -> list[_Window]:
    if frozen.episodes:
        episodes = [
            episode for episode in frozen.episodes if episode.split == split
        ]
        episodes.sort(key=lambda episode: (episode.start_s, episode.episode_id))
        return [
            _Window(episode.episode_id, episode.start_s, episode.end_s)
            for episode in episodes
        ]
    # Technical split: blocks have no operator episode. Spacing stays inside the block.
    windows: list[_Window] = []
    for block in frozen.blocks:
        if block.role != split.value:
            continue
        windows.append(_Window("", block.start_s, block.end_s))
    return windows


def _collect_ticks(
    reader: FrameReader,
    frozen: FrozenSplit,
    split: SplitName,
    windows: Sequence[_Window],
) -> list[FrameTick]:
    """Stream each episode. Frame images are not kept."""
    found: dict[int, FrameTick] = {}
    for window in windows:
        for frame in reader.iterate(window.start_s, window.end_s):
            tick = _tick(frame, window, frozen, split)
            if tick is None:
                continue
            found.setdefault(tick.source_frame_index, tick)
    return sorted(
        found.values(),
        key=lambda tick: (tick.source_time_s, tick.source_frame_index),
    )


def _tick(
    frame: ReadFrame,
    window: _Window,
    frozen: FrozenSplit,
    split: SplitName,
) -> FrameTick | None:
    if frame.decode_status is not DecodeStatus.ok or frame.image is None:
        return None
    index = frame.source_frame_index
    time_s = frame.source_time_s
    if index is None or time_s is None:
        return None
    if not math.isfinite(time_s) or time_s < 0:
        raise TimebaseError(
            "Некорректное время кадра. Отбор не подставляет другой кадр."
        )
    if not (window.start_s <= time_s < window.end_s):
        return None
    if _assignment_at(frozen, time_s) != split.value or _in_guard(frozen, time_s):
        return None
    return FrameTick(
        video_id=frame.video_id,
        source_frame_index=index,
        source_time_s=time_s,
        episode_id=window.episode_id,
        split=split,
    )


def _assignment_at(frozen: FrozenSplit, time_s: float) -> str | None:
    """Split name, ``excluded_guard``, or ``None`` outside the frozen timeline."""
    splits = {
        episode.split.value
        for episode in frozen.episodes
        if episode.start_s <= time_s < episode.end_s
    }
    if len(splits) > 1:
        raise LabError("episode_overlap", "В одной точке времени разные split.")
    if len(splits) == 1:
        return next(iter(splits))
    for guard in frozen.guards:
        if guard.start_s <= time_s < guard.end_s:
            return guard.role
    for block in frozen.blocks:
        if block.start_s <= time_s < block.end_s:
            return block.role
    return None


def _in_guard(frozen: FrozenSplit, time_s: float) -> bool:
    return any(guard.start_s <= time_s < guard.end_s for guard in frozen.guards)


def _write_chosen(
    reader: FrameReader,
    windows_by_split: Mapping[SplitName, Sequence[_Window]],
    chosen: Sequence[SelectedTick],
    images_dir: Path,
) -> list[_Encoded]:
    pending = {item.source_frame_index: item for item in chosen}
    if len(pending) != len(chosen):
        raise LabError(
            "duplicate_sample",
            "Один кадр попал в выборку дважды. Копия не записывается.",
        )
    encoded: list[_Encoded] = []
    done: set[int] = set()
    for split_name in (SplitName.train, SplitName.validation, SplitName.test):
        for window in windows_by_split.get(split_name, []):
            if not _window_needed(window, pending, done):
                continue
            for frame in reader.iterate(window.start_s, window.end_s):
                index = frame.source_frame_index
                if index is None or index not in pending or index in done:
                    continue
                encoded.append(_encode_frame(frame, pending[index], images_dir))
                done.add(index)
    missing = sorted(set(pending) - done)
    if missing:
        raise LabError(
            "frame_missing",
            f"Кадр {missing[0]} не прочитан повторно. Подставной кадр не записывается.",
        )
    encoded.sort(
        key=lambda item: (item.selected.source_time_s, item.selected.source_frame_index)
    )
    return encoded


def _window_needed(
    window: _Window,
    pending: Mapping[int, SelectedTick],
    done: set[int],
) -> bool:
    return any(
        index not in done and window.start_s <= item.source_time_s < window.end_s
        for index, item in pending.items()
    )


def _encode_frame(frame: ReadFrame, selected: SelectedTick, images_dir: Path) -> _Encoded:
    image = frame.image
    if image is None:
        raise LabError(
            "frame_missing",
            f"Кадр {selected.source_frame_index} не декодирован. "
            "Подставной кадр не записывается.",
        )
    name = f"{selected.video_id}_{selected.source_frame_index}.jpg"
    destination = images_dir / name
    phash = _average_hash(image)
    payload = _write_jpeg(destination, image)
    return _Encoded(
        selected=selected,
        image_sha256=hashlib.sha256(payload).hexdigest(),
        image_file=name,
        phash=phash,
    )


def _rows_with_warnings(encoded: Sequence[_Encoded]) -> list[SampleRow]:
    """Attach a warning. Every encoded frame stays in the sample."""
    rows: list[SampleRow] = []
    seen: list[tuple[int, int]] = []
    for item in encoded:
        warning: str | None = None
        matched: int | None = None
        for prev_hash, prev_index in seen:
            if _hamming(item.phash, prev_hash) <= _NEAR_DUPLICATE_HAMMING:
                warning = _NEAR_WARNING
                matched = prev_index
                break
        seen.append((item.phash, item.selected.source_frame_index))
        selected = item.selected
        rows.append(
            SampleRow(
                video_id=selected.video_id,
                source_frame_index=selected.source_frame_index,
                source_time_s=selected.source_time_s,
                episode_id=selected.episode_id,
                split=selected.split,
                selection_reason=selected.selection_reason,
                image_sha256=item.image_sha256,
                image_file=item.image_file,
                strategy=selected.strategy,
                model_version=selected.model_version,
                near_duplicate_warning=warning,
                near_duplicate_of=matched,
            )
        )
    if len(rows) != len(encoded):
        raise LabError(
            "sample_internal",
            "Отбор потерял кадр при проверке похожих кадров.",
        )
    return rows


def _representative(ticks: Sequence[FrameTick], count: int) -> list[SelectedTick]:
    """Even share per episode, then a fixed stride. Not an easy/hard ranking."""
    groups = _by_episode(ticks)
    alloc = allocate_even([len(group) for group in groups], count)
    chosen: list[SelectedTick] = []
    for group, take in zip(groups, alloc, strict=True):
        for position in stride_positions(len(group), take):
            chosen.append(_selected(group[position], "representative", "representative"))
    return chosen


def _spaced(
    ticks: Sequence[FrameTick],
    count: int,
    spacing_s: float,
) -> list[SelectedTick]:
    """Keep ``spacing_s`` inside one episode. Do not pull closer frames to fill count."""
    chosen: list[SelectedTick] = []
    for group in _by_episode(ticks):
        last: float | None = None
        for tick in group:
            if last is not None and _is_closer(last, tick.source_time_s, spacing_s):
                continue
            chosen.append(_selected(tick, "", "spaced"))
            last = tick.source_time_s
    if len(chosen) > count:
        chosen = [chosen[position] for position in stride_positions(len(chosen), count)]
    return chosen


def _apply_extras(
    selected: list[SelectedTick],
    extras: Sequence[CloserPick],
    pool: Sequence[FrameTick],
    count: int,
    split_label: str,
) -> list[SelectedTick]:
    by_index = {tick.source_frame_index: tick for tick in pool}
    chosen_ids = {item.source_frame_index for item in selected}
    rows = list(selected)
    for extra in extras:
        index = extra.source_frame_index
        if index in chosen_ids:
            continue
        tick = by_index.get(index)
        if tick is None:
            raise LabError(
                "frame_not_in_split",
                f"Кадр {index} не входит в зафиксированную часть «{split_label}».",
            )
        if len(rows) >= count:
            raise LabError(
                "sample_quota",
                "Дополнительный кадр не помещается в запрошенное число. "
                "Квота не расширяется копией.",
            )
        if not extra.selection_reason.strip():
            raise LabError("sample_spacing", _SPACING_MESSAGE)
        rows.append(_selected(tick, extra.selection_reason, "spaced"))
        chosen_ids.add(index)
    return rows


def _selected(tick: FrameTick, reason: str, strategy: StrategyName) -> SelectedTick:
    return SelectedTick(
        video_id=tick.video_id,
        source_frame_index=tick.source_frame_index,
        source_time_s=tick.source_time_s,
        episode_id=tick.episode_id,
        split=tick.split,
        selection_reason=reason,
        strategy=strategy,
    )


def allocate_even(sizes: Sequence[int], count: int) -> list[int]:
    """Spread picks across episodes in time order. Remainder goes to earlier episodes."""
    alloc = [0 for _ in sizes]
    if count <= 0 or not sizes:
        return alloc
    share = count // len(sizes)
    extra = count % len(sizes)
    remaining = count
    for index, size in enumerate(sizes):
        want = share + (1 if index < extra else 0)
        take = min(max(size, 0), want)
        alloc[index] = take
        remaining -= take
    if remaining <= 0:
        return alloc
    for index, size in enumerate(sizes):
        if remaining <= 0:
            break
        room = max(size, 0) - alloc[index]
        take = min(room, remaining)
        alloc[index] += take
        remaining -= take
    return alloc


def stride_positions(frame_count: int, pick_count: int) -> list[int]:
    """Deterministic indexes in ``0 .. frame_count-1``.

    One pick is the middle frame. Several picks step from the first frame
    to the last. Collisions walk to the nearest free index.
    """
    if pick_count <= 0 or frame_count <= 0:
        return []
    if pick_count >= frame_count:
        return list(range(frame_count))
    if pick_count == 1:
        return [frame_count // 2]
    used: set[int] = set()
    chosen: list[int] = []
    last = frame_count - 1
    for step in range(pick_count):
        raw = _round_half_up(step * last / (pick_count - 1))
        index = _nearest_free(raw, frame_count, used)
        used.add(index)
        chosen.append(index)
    chosen.sort()
    return chosen


def _by_episode(ticks: Sequence[FrameTick]) -> list[list[FrameTick]]:
    order: list[str] = []
    buckets: dict[str, list[FrameTick]] = {}
    ordered = sorted(
        ticks,
        key=lambda item: (item.source_time_s, item.episode_id, item.source_frame_index),
    )
    for tick in ordered:
        if tick.episode_id not in buckets:
            order.append(tick.episode_id)
            buckets[tick.episode_id] = []
        buckets[tick.episode_id].append(tick)
    return [buckets[episode_id] for episode_id in order]


def _dedupe(ticks: Sequence[FrameTick]) -> list[FrameTick]:
    found: dict[int, FrameTick] = {}
    for tick in ticks:
        current = found.get(tick.source_frame_index)
        if current is None or (tick.source_time_s, tick.episode_id) < (
            current.source_time_s,
            current.episode_id,
        ):
            found[tick.source_frame_index] = tick
    return sorted(
        found.values(),
        key=lambda tick: (tick.source_time_s, tick.source_frame_index),
    )


def _reject_duplicate_indexes(chosen: Sequence[SelectedTick]) -> None:
    indexes = [item.source_frame_index for item in chosen]
    if len(indexes) != len(set(indexes)):
        raise LabError(
            "duplicate_sample",
            "Один кадр попал в выборку дважды. Копия не записывается.",
        )


def _split_counts(counts: Mapping[SplitName, int]) -> SplitCounts:
    return SplitCounts(
        train=counts[SplitName.train],
        validation=counts[SplitName.validation],
        test=counts[SplitName.test],
    )


def _written_counts(
    requested: SplitCounts,
    rows: Sequence[SampleRow],
) -> tuple[SplitCounts, SplitCounts]:
    tallies = {name: 0 for name in SplitName}
    for row in rows:
        tallies[row.split] += 1
    written = SplitCounts(
        train=tallies[SplitName.train],
        validation=tallies[SplitName.validation],
        test=tallies[SplitName.test],
    )
    shortfall = SplitCounts(
        train=requested.train - written.train,
        validation=requested.validation - written.validation,
        test=requested.test - written.test,
    )
    return written, shortfall


def _artifact_paths(project_dir: Path, video_id: str) -> tuple[Path, Path]:
    if (
        not video_id
        or video_id in {".", ".."}
        or "/" in video_id
        or "\\" in video_id
        or "\x00" in video_id
    ):
        raise LabError("invalid_video_id", "Недопустимый идентификатор видео.")
    root = project_dir / "samples" / video_id
    return root / "sample_manifest.json", root / "images"


def _is_closer(earlier: float, later: float, spacing_s: float) -> bool:
    return later - earlier + _SPACING_TOLERANCE_S < spacing_s


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _nearest_free(raw: int, frame_count: int, used: set[int]) -> int:
    bounded = min(max(raw, 0), frame_count - 1)
    if bounded not in used:
        return bounded
    for delta in range(1, frame_count):
        right = bounded + delta
        if right < frame_count and right not in used:
            return right
        left = bounded - delta
        if left >= 0 and left not in used:
            return left
    raise LabError(
        "sample_internal",
        "Не удалось выбрать кадр по шагу. Кадр не копируется.",
    )


def _average_hash(image: np.ndarray) -> int:
    gray = _gray(image)
    small = _block_means(gray)
    bits = small >= float(small.mean())
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    return value


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise LabError("frame_unreadable", "Кадр для JPEG должен быть RGB uint8.")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise LabError("frame_unreadable", "Кадр для JPEG должен быть RGB uint8.")
    return image.astype(np.float32).mean(axis=2)


def _block_means(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    rows = np.linspace(0, height, 9)
    cols = np.linspace(0, width, 9)
    small = np.empty((8, 8), dtype=np.float32)
    for i in range(8):
        y0 = int(rows[i])
        y1 = min(height, max(int(rows[i + 1]), y0 + 1))
        for j in range(8):
            x0 = int(cols[j])
            x1 = min(width, max(int(cols[j + 1]), x0 + 1))
            block = gray[y0:y1, x0:x1]
            small[i, j] = 0.0 if block.size == 0 else float(block.mean())
    return small


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _write_jpeg(path: Path, image: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(np.ascontiguousarray(_require_rgb(image)), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY],
    )
    if not ok:
        raise LabError("frame_unreadable", "Не удалось записать JPEG.")
    payload = encoded.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _require_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise LabError("frame_unreadable", "Кадр для JPEG должен быть RGB uint8.")
    return image
