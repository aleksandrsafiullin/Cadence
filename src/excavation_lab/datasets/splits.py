"""Freeze the operator's role blocks. There is no automatic proposal.

``write_blocks`` stores the unfrozen document. ``freeze_split`` copies it to
``split_manifest.json`` once. A second of one video is ``same_scene``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NoReturn

from pydantic import Field, ValidationError, model_validator

from excavation_lab.datasets.blocks import RoleBlock, validate_role_blocks
from excavation_lab.datasets.blocks import assignment_at as role_at
from excavation_lab.errors import LabError, SplitHashMismatchError, SplitNotFrozenError
from excavation_lab.schemas import SplitIndependence, SplitLimitation, VersionedModel
from excavation_lab.storage import (
    get_split_manifest,
    get_video,
    insert_split_manifest,
    project_directory,
    resolve_project_path,
)

SAME_SCENE_MESSAGE = "Блоки одного видео — та же сцена."
_PROPOSE_REFUSED = (
    "Разбиение задают блоки оператора. "
    "Автоматическое предложение не создаётся и не записывается."
)
_MANIFEST_NOT_FOUND = "Файл предложения разбиения не найден."
_PROPOSAL_BROKEN = "Предложение разбиения не является JSON-объектом."
_PROPOSAL_INVALID = "Предложение разбиения некорректно и не может быть зафиксировано."
_FROZEN_BROKEN = "Файл split_manifest.json повреждён."
_ALREADY_FROZEN = (
    "Разбиение этого видео уже зафиксировано. "
    "Повторная фиксация в этой версии не выполняется."
)
_VIDEO_MISMATCH = "Файл разбиения относится к другому видео."
_FROZEN_SCHEMA = "Файл split_manifest.json не соответствует схеме разбиения."
_FROZEN_LEGACY = (
    "Зафиксированный файл этого ролика — разбиение по заездам. "
    "Блоки из него не читаются. Файл не изменён."
)


class _SplitBody(VersionedModel):
    video_id: str = Field(min_length=1)
    independence: SplitIndependence = SplitIndependence.same_scene
    limitation: SplitLimitation | None = None
    message: str = SAME_SCENE_MESSAGE
    blocks: list[RoleBlock]
    episodes: list[Any] = Field(default_factory=list)
    guards: list[Any] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_scene(self) -> _SplitBody:
        if self.independence is not SplitIndependence.same_scene:
            raise ValueError("blocks of one video are same_scene")
        if self.message != SAME_SCENE_MESSAGE:
            raise ValueError("same_scene message is fixed")
        try:
            ordered = validate_role_blocks(self.blocks)
        except LabError as exc:
            raise ValueError(exc.message) from exc
        self.blocks = ordered
        return self

    def assignment_at(self, time_s: float) -> str | None:
        return role_at(self.blocks, time_s)


class SplitProposal(_SplitBody):
    """Unfrozen operator blocks. Not a generated split."""

    frozen: Literal[False] = False


class FrozenSplit(_SplitBody):
    """Copy of the operator blocks plus ``content_hash``."""

    frozen: Literal[True] = True
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def propose_split(*_args: object, **_kwargs: object) -> NoReturn:
    """Refuse. This does not assign roles and does not write a file."""
    raise LabError("split_propose_refused", _PROPOSE_REFUSED)


def propose_for_project(workspace: Path, project_id: str, video_id: str) -> NoReturn:
    """Refuse. Episodes in the project are not a split and nothing is written."""
    del workspace, project_id, video_id
    raise LabError("split_propose_refused", _PROPOSE_REFUSED)


def write_blocks(
    workspace: Path,
    project_id: str,
    video_id: str,
    blocks: Sequence[RoleBlock],
) -> Path:
    """Replace the unfrozen blocks. A frozen video rejects the write."""
    checked = validate_role_blocks(blocks)
    get_video(workspace, project_id, video_id)
    _reject_if_frozen(workspace, project_id, video_id)
    proposal = SplitProposal(video_id=video_id, blocks=checked)
    return write_proposal(workspace, project_id, proposal)


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

    A second freeze of the same video fails with ``split_frozen``.
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


def current_blocks(workspace: Path, project_id: str, video_id: str) -> list[RoleBlock]:
    """Frozen blocks, or the draft proposal when this video is not frozen.

    Action intervals use this list so a label can be stored before freeze.
    Training checks the frozen manifest on its own.
    """
    get_video(workspace, project_id, video_id)
    recorded = get_split_manifest(workspace, project_id, video_id)
    path = frozen_manifest_path(workspace, project_id, video_id)
    if path.is_file() or recorded is not None:
        return list(require_frozen(workspace, project_id, video_id).blocks)
    leaf = _safe_video_id(video_id)
    proposal = project_directory(workspace, project_id) / "split_proposals" / f"{leaf}.json"
    if not proposal.is_file():
        return []
    loaded = _load_proposal(proposal)
    if loaded.video_id != video_id:
        raise LabError("video_mismatch", _VIDEO_MISMATCH)
    return list(loaded.blocks)


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
        return _frozen_model(document)
    except ValidationError as exc:
        raise LabError("invalid_manifest", _FROZEN_SCHEMA) from exc


def _frozen_model(document: Mapping[str, Any]) -> FrozenSplit:
    """Read blocks from a frozen file. Extra episode fields stay on disk.

    The hash was already checked against the original bytes. A file that
    only fails because of the old episode keys is opened from its ``blocks``.
    """
    try:
        return FrozenSplit.model_validate(document)
    except ValidationError:
        blocks = document.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            episodes = document.get("episodes")
            if isinstance(episodes, list) and episodes:
                raise LabError("invalid_manifest", _FROZEN_LEGACY)
            raise
        narrowed = {
            "schema_version": document.get("schema_version", 1),
            "video_id": document.get("video_id"),
            "frozen": True,
            "independence": SplitIndependence.same_scene.value,
            "message": SAME_SCENE_MESSAGE,
            "blocks": blocks,
            "content_hash": document.get("content_hash"),
        }
        return FrozenSplit.model_validate(narrowed)


def _reject_if_frozen(workspace: Path, project_id: str, video_id: str) -> None:
    destination = frozen_manifest_path(workspace, project_id, video_id)
    recorded = get_split_manifest(workspace, project_id, video_id)
    if destination.exists() or destination.is_symlink() or recorded is not None:
        raise LabError("split_frozen", _ALREADY_FROZEN)


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
