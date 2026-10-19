"""Rewrite a checkpoint so each catalog name keeps its own logit.

The detection head has ``len(catalog) + 1`` rows. Foreground rows follow
catalog ids. The background row is the last row, as in ``lwdetr.py``.
``load_pretrain_weights`` resizes with ``_resize_linear`` only when
``class_embed.bias`` is not already ``num_classes + 1``. A file at that
width is an identity copy: the backbone stays, and rows are not repeated
or truncated.

The written file is the rest of the source checkpoint. A head-only dict
would leave the backbone at its initializer because the loader uses
``strict=False``. This module does not edit the source file and does not
write into the official weights cache.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from excavation_lab.errors import LabError
from excavation_lab.hardware import weights_cache_dir

# ``class_embed`` and each ``enc_out_class_embed.{i}`` linear. Background is
# the last row and is not a catalog name.
_HEAD_KEY = re.compile(
    r"(?:^|\.)(?:class_embed|enc_out_class_embed(?:\.\d+)?)\.(?:weight|bias)$"
)
_FRESH_SEED = 17

if TYPE_CHECKING:
    import torch


def rewrite_class_head(
    source: Path,
    destination: Path,
    catalog_names: Sequence[str],
    *,
    workspace: Path | None = None,
) -> Path:
    """Write ``destination`` with classification rows mapped by name.

    A catalog name missing from the source gets one new row. Rows that do
    match are copied. The source background row is copied to the new last
    row and is not used as a new class. ``destination`` is a new file.
    """
    source_path = Path(source)
    dest_path = Path(destination)
    names = _catalog_names(catalog_names)
    _reject_destination(source_path, dest_path, workspace)
    checkpoint = _load_checkpoint(source_path)
    states = _state_dicts(checkpoint)
    rows = _head_rows(states)
    name_index = _name_index(_checkpoint_names(checkpoint), names, rows)
    for state in states:
        _replace_head(state, names, name_index)
    _store_class_names(checkpoint, names)
    _write_checkpoint(checkpoint, dest_path)
    return dest_path


def _catalog_names(catalog_names: Sequence[str]) -> list[str]:
    if isinstance(catalog_names, (str, bytes)) or not isinstance(catalog_names, Sequence):
        raise LabError("class_head", "Список классов каталога должен быть последовательностью имён.")
    names = list(catalog_names)
    if not names or not all(isinstance(name, str) and name for name in names):
        raise LabError("class_head", "В каталоге нет имён классов.")
    if len(set(names)) != len(names):
        raise LabError("class_head", "В каталоге повторяется имя класса.")
    return names


def _reject_destination(source: Path, destination: Path, workspace: Path | None) -> None:
    try:
        dest = destination.resolve()
    except OSError as exc:
        raise LabError("baseline_weights", "Путь к чекпоинту не читается.") from exc
    if source.exists():
        try:
            if dest == source.resolve():
                raise LabError(
                    "class_head",
                    "Исходный чекпоинт не переписывается на месте.",
                )
        except OSError as exc:
            raise LabError("baseline_weights", "Путь к чекпоинту не читается.") from exc
    if workspace is not None:
        cache = weights_cache_dir(workspace).resolve()
        if dest == cache or cache in dest.parents:
            raise LabError(
                "baseline_weights",
                "Переписанная голова не записывается в кэш официальных весов.",
            )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LabError("class_head", "Исходный чекпоинт не найден.")
    from rfdetr.utilities.io import _safe_torch_load

    try:
        loaded = _safe_torch_load(path, trust=False)
    except Exception as exc:
        raise LabError(
            "class_head",
            "Чекпоинт не открывается безопасной загрузкой.",
        ) from exc
    if not isinstance(loaded, dict):
        raise LabError("class_head", "Чекпоинт не является словарём.")
    return loaded


def _state_dicts(checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            found.append(value)
    if not found:
        raise LabError("class_head", "В чекпоинте нет весов модели.")
    return found


def _head_rows(states: Sequence[Mapping[str, Any]]) -> int:
    import torch

    rows: int | None = None
    saw_weight = False
    saw_bias = False
    for state in states:
        for key, value in state.items():
            if not isinstance(key, str) or _HEAD_KEY.search(key) is None:
                continue
            if not isinstance(value, torch.Tensor) or value.ndim not in {1, 2}:
                raise LabError("class_head", f"Строка головы «{key}» не является матрицей.")
            count = int(value.shape[0])
            if count < 1:
                raise LabError("class_head", "В голове нет строк.")
            if rows is None:
                rows = count
            elif count != rows:
                raise LabError("class_head", "Строки class_embed и enc_out_class_embed различаются.")
            if key.endswith("class_embed.weight"):
                saw_weight = True
            if key.endswith("class_embed.bias"):
                saw_bias = True
    if rows is None or not saw_weight or not saw_bias:
        raise LabError("class_head", "В чекпоинте нет class_embed.weight и class_embed.bias.")
    return rows


def _checkpoint_names(checkpoint: Mapping[str, Any]) -> list[str] | None:
    for container_key in ("args", "model_config"):
        names = _names_in(checkpoint.get(container_key))
        if names is not None:
            return names
    return None


def _names_in(container: object) -> list[str] | None:
    if container is None:
        return None
    if isinstance(container, Mapping):
        raw = container.get("class_names")
    else:
        raw = getattr(container, "class_names", None)
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
        raise LabError("class_head", "class_names в чекпоинте не является списком имён.")
    names: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise LabError("class_head", "class_names в чекпоинте не является списком имён.")
        names.append(item)
    return names


def _name_index(source_names: list[str] | None, catalog_names: Sequence[str], rows: int) -> dict[str, int]:
    """Map a catalog name to a source foreground row.

    No catalog name in the file means an official COCO checkpoint: every
    foreground row stays fresh, and only the last row is the background.
    """
    if not source_names:
        return {}
    catalog = set(catalog_names)
    if not any(name in catalog for name in source_names):
        return {}
    if len(source_names) != rows - 1:
        raise LabError(
            "class_head",
            "Число имён в чекпоинте не совпадает с числом строк головы без фона.",
        )
    if len(set(source_names)) != len(source_names):
        raise LabError("class_head", "В чекпоинте повторяется имя класса.")
    return {name: index for index, name in enumerate(source_names)}


def _replace_head(
    state: dict[str, Any],
    catalog_names: Sequence[str],
    name_index: Mapping[str, int],
) -> None:
    import torch

    for key, value in list(state.items()):
        if not isinstance(key, str) or _HEAD_KEY.search(key) is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise LabError("class_head", f"Строка головы «{key}» не является матрицей.")
        state[key] = _remap_rows(value, catalog_names, name_index)


def _remap_rows(
    source: torch.Tensor,
    catalog_names: Sequence[str],
    name_index: Mapping[str, int],
) -> torch.Tensor:
    dest_rows = len(catalog_names) + 1
    if source.ndim == 2:
        fresh = _fresh_weight(dest_rows, int(source.shape[1]), source.dtype, source.device)
    else:
        fresh = _fresh_bias(dest_rows, source.dtype, source.device)
    copied: set[int] = set()
    for dest_index, name in enumerate(catalog_names):
        source_index = name_index.get(name)
        if source_index is None:
            continue
        fresh[dest_index] = source[source_index]
        copied.add(dest_index)
    fresh[-1] = source[-1]
    copied.add(dest_rows - 1)
    _keep_fresh_rows_distinct(fresh, source, copied)
    return fresh.detach()


def _fresh_weight(rows: int, features: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_FRESH_SEED)
    weight = torch.empty((rows, features), dtype=torch.float32)
    torch.nn.init.kaiming_uniform_(weight, a=5**0.5, generator=generator)
    return weight.to(dtype=dtype, device=device)


def _fresh_bias(rows: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_FRESH_SEED)
    bias = torch.empty((rows,), dtype=torch.float32)
    bound = 1 / (max(rows, 1) ** 0.5)
    bias.uniform_(-bound, bound, generator=generator)
    return bias.to(dtype=dtype, device=device)


def _keep_fresh_rows_distinct(dest: torch.Tensor, source: torch.Tensor, copied: set[int]) -> None:
    for index in range(dest.shape[0]):
        if index in copied:
            continue
        if _matches_source_row(dest[index], source):
            dest[index] = dest[index] + 1


def _matches_source_row(row: torch.Tensor, source: torch.Tensor) -> bool:
    import torch

    return any(torch.equal(row, source[index]) for index in range(source.shape[0]))


def _store_class_names(checkpoint: dict[str, Any], catalog_names: Sequence[str]) -> None:
    names = list(catalog_names)
    args = checkpoint.get("args")
    if isinstance(args, dict):
        args["class_names"] = names
        args["num_classes"] = len(names)
        return
    if args is not None and hasattr(args, "__dict__"):
        args.class_names = names
        if hasattr(args, "num_classes"):
            args.num_classes = len(names)
        return
    checkpoint["args"] = {"class_names": names, "num_classes": len(names)}


def _write_checkpoint(checkpoint: dict[str, Any], destination: Path) -> None:
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".class_head-",
        suffix=".pth",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(name)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
