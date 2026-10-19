"""Zero unsupervised class columns inside ``SetCriterion.loss_labels``.

``loss_labels`` builds ``pos_weights`` and ``neg_weights``, then reduces
them on the next line (``criterion.py`` around the ``loss_ce = neg_weights
* src_logits - F.logsigmoid(...)`` statement). The original method returns
only the reduced loss, so masking after it returns does nothing.

A trace on that statement zeros the unsupervised columns in place, and the
library line then reduces those same tensors. A missing COCO box is not a
column: only ``class_supervision.json`` ``unsupervised`` names are.
Background is the last logit and is never one of those names.
"""

from __future__ import annotations

import inspect
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Any

from excavation_lab.datasets.catalog import StoredCatalog
from excavation_lab.errors import LabError

SUPERVISION_FILE = "class_supervision.json"
# Installed rfdetr 1.11.2 ``SetCriterion.loss_labels`` ia-bce reduction.
# Weights are finished above this line; ``loss_ce.sum()`` follows it.
_REDUCTION_LINE = (
    "loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)"
)

if TYPE_CHECKING:
    import torch


def catalog_class_names(catalog: StoredCatalog) -> list[str]:
    """Catalog id order. Background is not an entry."""
    return [item.name for item in sorted(catalog.classes, key=lambda item: item.id)]


def apply_logit_mask(
    pos_weights: torch.Tensor,
    neg_weights: torch.Tensor,
    class_indexes: Sequence[int],
) -> None:
    """Zero each class column on every query of both weight tensors.

    The last logit is background and is left as the library set it.
    Other columns are left untouched.
    """
    import torch

    if not isinstance(pos_weights, torch.Tensor) or not isinstance(neg_weights, torch.Tensor):
        raise LabError("class_supervision", "Веса меток должны быть тензорами.")
    if pos_weights.shape != neg_weights.shape or pos_weights.ndim < 2:
        raise LabError("class_supervision", "pos_weights и neg_weights должны совпадать по форме.")
    background = pos_weights.shape[-1] - 1
    if background < 1:
        raise LabError("class_supervision", "В голове нет столбца класса и фона.")
    for class_index in class_indexes:
        if isinstance(class_index, bool) or not isinstance(class_index, int):
            raise LabError("class_supervision", "Индекс класса должен быть целым.")
        if class_index < 0 or class_index >= background:
            raise LabError(
                "class_supervision",
                "Фон и чужой индекс не обнуляются. Столбец остаётся как его задала библиотека.",
            )
        pos_weights[..., class_index] = 0
        neg_weights[..., class_index] = 0


@dataclass(frozen=True, slots=True)
class _ImageSupervision:
    image_id: int
    file_name: str
    unsupervised: frozenset[str]


@dataclass(frozen=True, slots=True)
class SupervisionTable:
    """Sidecar rows keyed by image id and by file name. Both keys are one row."""

    by_id: dict[int, _ImageSupervision]
    by_file: dict[str, _ImageSupervision]

    def images(self) -> Iterator[_ImageSupervision]:
        for image_id in sorted(self.by_id):
            yield self.by_id[image_id]

    def unsupervised_names(self, *, image_id: int | None, file_name: str | None) -> frozenset[str]:
        """Names marked unsupervised. A missing box is not consulted."""
        if image_id is not None and file_name is None:
            row = self.by_id.get(image_id)
            if row is not None:
                file_name = row.file_name
        if file_name is not None and image_id is None:
            row = self.by_file.get(file_name)
            if row is not None:
                image_id = row.image_id
        from_id = self.by_id.get(image_id) if image_id is not None else None
        from_file = self.by_file.get(file_name) if file_name else None
        if from_id is None and from_file is None:
            return frozenset()
        if (
            from_id is not None
            and from_file is not None
            and from_id.unsupervised != from_file.unsupervised
        ):
            raise LabError(
                "class_supervision",
                "Запись image id и запись имени файла расходятся. Класс не приклеивается к соседнему.",
            )
        found = from_id if from_id is not None else from_file
        assert found is not None
        return found.unsupervised


def load_supervision(path: Path) -> SupervisionTable | None:
    """Read one split sidecar. A missing file supervises nothing extra."""
    if path.is_symlink() or not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError("class_supervision", "class_supervision.json не читается.") from exc
    if not isinstance(document, dict):
        raise LabError("class_supervision", "class_supervision.json должен быть объектом.")
    images = document.get("images")
    if not isinstance(images, dict) or not images:
        raise LabError("class_supervision", "В class_supervision.json нет images.")
    by_id: dict[int, _ImageSupervision] = {}
    by_file: dict[str, _ImageSupervision] = {}
    for record in images.values():
        parsed = _parse_record(record)
        current = by_id.get(parsed.image_id)
        if current is not None and current != parsed:
            raise LabError(
                "class_supervision",
                f"Кадр {parsed.image_id} записан дважды и записи не совпадают.",
            )
        named = by_file.get(parsed.file_name)
        if named is not None and named != parsed:
            raise LabError(
                "class_supervision",
                f"Файл «{parsed.file_name}» записан дважды и записи не совпадают.",
            )
        by_id[parsed.image_id] = parsed
        by_file[parsed.file_name] = parsed
    return SupervisionTable(by_id=by_id, by_file=by_file)


def unsupervised_class_indexes(
    table: SupervisionTable,
    class_names: Sequence[str],
    *,
    image_id: int | None,
    file_name: str | None,
) -> tuple[int, ...]:
    """Catalog indexes for unsupervised names. Order is ``class_names``, not a sort."""
    names = table.unsupervised_names(image_id=image_id, file_name=file_name)
    unknown = sorted(name for name in names if name not in class_names)
    if unknown:
        raise LabError(
            "class_supervision",
            f"Класс «{unknown[0]}» нет в каталоге. Соседний класс не подставляется.",
        )
    return tuple(index for index, name in enumerate(class_names) if name in names)


def observe_unsupervised(dataset_dir: Path, class_names: Sequence[str]) -> list[dict[str, Any]]:
    """Train-split rows the loss wrap would see. One row per image id."""
    table = load_supervision(Path(dataset_dir) / "train" / SUPERVISION_FILE)
    if table is None:
        return []
    rows: list[dict[str, Any]] = []
    for image in table.images():
        indexes = unsupervised_class_indexes(
            table,
            class_names,
            image_id=image.image_id,
            file_name=image.file_name,
        )
        rows.append(
            {
                "image_id": image.image_id,
                "file_name": image.file_name,
                "unsupervised": [class_names[index] for index in indexes],
            }
        )
    return rows


@contextmanager
def wrap_loss_labels(dataset_dir: Path, class_names: Sequence[str]) -> Iterator[None]:
    """Replace ``SetCriterion.loss_labels`` until the caller returns."""
    from rfdetr.models.criterion import SetCriterion

    original = SetCriterion.loss_labels
    reduction_line = _reduction_lineno(original)
    root = Path(dataset_dir)
    tables = {
        "train": load_supervision(root / "train" / SUPERVISION_FILE),
        "valid": load_supervision(root / "valid" / SUPERVISION_FILE),
    }
    file_names = {
        "train": _coco_file_names(root / "train"),
        "valid": _coco_file_names(root / "valid"),
    }
    names = list(class_names)

    def wrapped(
        self: Any,
        outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        indices: list[Any],
        num_boxes: Any,
        log: bool = True,
        matched_targets: Any = None,
    ) -> dict[str, Any]:
        previous = sys.gettrace()

        def tracer(frame: FrameType, event: str, arg: Any) -> Any:
            if (
                event == "line"
                and frame.f_code is original.__code__
                and frame.f_lineno == reduction_line
            ):
                _mask_ready_weights(frame, self, names, tables, file_names)
            if previous is None:
                return tracer
            delegated = previous(frame, event, arg)
            return tracer if delegated is None else delegated

        sys.settrace(tracer)
        try:
            return original(
                self,
                outputs,
                targets,
                indices,
                num_boxes,
                log=log,
                matched_targets=matched_targets,
            )
        finally:
            sys.settrace(previous)

    SetCriterion.loss_labels = wrapped  # type: ignore[method-assign]
    try:
        yield
    finally:
        SetCriterion.loss_labels = original  # type: ignore[method-assign]


def _mask_ready_weights(
    frame: FrameType,
    criterion: Any,
    class_names: Sequence[str],
    tables: Mapping[str, SupervisionTable | None],
    file_names: Mapping[str, dict[int, str]],
) -> None:
    """Zero columns on the tensors this call just built, before the reduction reads them."""
    import torch

    locals_ = frame.f_locals
    pos_weights = locals_.get("pos_weights")
    neg_weights = locals_.get("neg_weights")
    targets = locals_.get("targets")
    if not isinstance(pos_weights, torch.Tensor) or not isinstance(neg_weights, torch.Tensor):
        return
    if not isinstance(targets, list) or pos_weights.shape[0] != len(targets):
        raise LabError(
            "class_supervision",
            "Число кадров в весах меток не совпадает с targets.",
        )
    split = "train" if bool(getattr(criterion, "training", True)) else "valid"
    table = tables.get(split)
    if table is None:
        return
    names_for_id = file_names.get(split, {})
    for batch_index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise LabError("class_supervision", "target должен быть объектом.")
        image_id = _image_id(target)
        file_name = _file_name(target, image_id, names_for_id)
        columns = unsupervised_class_indexes(
            table,
            class_names,
            image_id=image_id,
            file_name=file_name,
        )
        if not columns:
            continue
        apply_logit_mask(pos_weights[batch_index], neg_weights[batch_index], columns)


def _reduction_lineno(method: Any) -> int:
    try:
        lines, start = inspect.getsourcelines(method)
    except OSError as exc:
        raise LabError(
            "class_supervision",
            "Исходник loss_labels недоступен. Маска до сведения не ставится.",
        ) from exc
    found = [start + offset for offset, line in enumerate(lines) if line.strip() == _REDUCTION_LINE]
    if len(found) != 1:
        raise LabError(
            "class_supervision",
            "В loss_labels нет строки сведения pos_weights и neg_weights.",
        )
    return found[0]


def _parse_record(record: object) -> _ImageSupervision:
    if not isinstance(record, dict):
        raise LabError("class_supervision", "Запись кадра должна быть объектом.")
    image_id = record.get("image_id")
    file_name = record.get("file_name")
    unsupervised = record.get("unsupervised")
    if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id < 1:
        raise LabError("class_supervision", "У кадра нет целого image id.")
    if not isinstance(file_name, str) or not file_name:
        raise LabError("class_supervision", "У кадра нет имени файла.")
    if not isinstance(unsupervised, list) or not all(isinstance(name, str) and name for name in unsupervised):
        raise LabError("class_supervision", "unsupervised должен быть списком имён.")
    return _ImageSupervision(
        image_id=image_id,
        file_name=file_name,
        unsupervised=frozenset(unsupervised),
    )


def _image_id(target: Mapping[str, Any]) -> int | None:
    import torch

    value = target.get("image_id")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        number = int(value.reshape(-1)[0].item())
        return number
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _file_name(target: Mapping[str, Any], image_id: int | None, coco_names: Mapping[int, str]) -> str | None:
    value = target.get("file_name")
    if isinstance(value, str) and value:
        return value
    if image_id is None:
        return None
    return coco_names.get(image_id)


def _coco_file_names(split_dir: Path) -> dict[int, str]:
    """Image id to file name. Annotations are not read."""
    path = split_dir / "_annotations.coco.json"
    if path.is_symlink() or not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError("class_supervision", "Аннотации COCO не читаются.") from exc
    images = document.get("images") if isinstance(document, dict) else None
    if not isinstance(images, list):
        return {}
    found: dict[int, str] = {}
    for image in images:
        if not isinstance(image, dict):
            continue
        image_id = image.get("id")
        file_name = image.get("file_name")
        if isinstance(image_id, bool) or not isinstance(image_id, int):
            continue
        if not isinstance(file_name, str) or not file_name:
            continue
        found[image_id] = file_name
    return found
