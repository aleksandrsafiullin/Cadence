"""Project errors.

``PathNotAllowedError`` is the path that resolves outside an allowed root.
``MediaNotFoundError`` is an allowed path with no file.
``CorruptMediaError`` is a file ffprobe cannot decode.
``FfmpegMissingError`` is missing ``ffprobe`` or ``ffmpeg`` on ``PATH``.
``DiskFullError`` is a write that fails because the disk has no free space.
``map_os_error`` is the only mapper from that OS error to ``disk_full``.
``TimebaseError`` is a PTS, duration, or interval the source clock rejects.
``DuplicateFrameError`` is a second yield of one source frame in a single pass.
``CheckpointMissingError`` is custom mode without a weights file.
``IncompatibleCheckpointError`` is a custom bundle whose class order,
variant, or resolution does not match ``weights.pth``.
``ColorOrderError`` is a channel-reversed buffer offered as RGB.
``JobAlreadyRunningError`` is a second train or analyze while one job is active.
``SplitNotFrozenError`` is a video with no ``split_manifest.json``.
``SplitHashMismatchError`` is a frozen manifest whose body no longer matches.
``UnknownCategoryError`` is a category name outside the four detection classes.
``UnsupportedCrowdError`` is ``iscrowd`` other than 0.
``DatasetHashMismatchError`` is a comparison of two runs whose dataset
hashes differ. It carries no metric delta.
``NotFinalError`` is ``evaluate --split test --final`` without a frozen
manifest, a bundle checksum that verifies, or a resolved config hash.
``TestNotLabeledError`` is a test split with no labels. The report uses
«не рассчитано» and does not borrow validation metrics.
"""

import errno
from collections.abc import Iterator
from contextlib import contextmanager


class LabError(Exception):
    """Failure with an English `code` and a Russian `message`."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class PathNotAllowedError(LabError):
    """Resolved path is outside the workspace or a project's allowed roots."""

    def __init__(self, path: str | None = None) -> None:
        if path:
            message = (
                f"Путь вне разрешённых каталогов: {path}. "
                "Выберите путь внутри рабочего каталога проекта "
                "или добавьте каталог в разрешённые."
            )
        else:
            message = (
                "Путь вне разрешённых каталогов. "
                "Выберите путь внутри рабочего каталога проекта "
                "или добавьте каталог в разрешённые."
            )
        super().__init__("path_not_allowed", message)


class MediaNotFoundError(LabError):
    """The resolved path is allowed, and the file is not there."""

    def __init__(self, path: str | None = None) -> None:
        if path:
            message = (
                f"Файл видео не найден: {path}. "
                "Положите MP4, MOV или MKV по этому пути внутри разрешённого каталога. "
                "Команда не скачивает ролик по ссылке."
            )
        else:
            message = (
                "Файл видео не найден. "
                "Положите MP4, MOV или MKV в разрешённый каталог проекта и укажите путь. "
                "Команда не скачивает ролик по ссылке."
            )
        super().__init__("media_not_found", message)


class CorruptMediaError(LabError):
    """ffprobe cannot read a container duration and a video stream."""

    def __init__(self, path: str | None = None) -> None:
        if path:
            message = (
                f"Файл не декодируется как видео: {path}. "
                "Замените его на целый MP4, MOV или MKV."
            )
        else:
            message = (
                "Файл не декодируется как видео. "
                "Замените его на целый MP4, MOV или MKV."
            )
        super().__init__("corrupt_media", message)


class FfmpegMissingError(LabError):
    """``ffprobe`` or ``ffmpeg`` is not on ``PATH``."""

    def __init__(self) -> None:
        super().__init__(
            "ffmpeg_missing",
            "Не найдены ffprobe или ffmpeg. "
            "Установите FFmpeg и добавьте ffprobe и ffmpeg в PATH.",
        )


class DiskFullError(LabError):
    """A write failed because the filesystem has no free space."""

    def __init__(self, path: str | None = None) -> None:
        if path:
            message = (
                f"Не хватает места на диске при записи: {path}. "
                "Освободите место и повторите запись."
            )
        else:
            message = (
                "Не хватает места на диске. "
                "Освободите место и повторите запись."
            )
        super().__init__("disk_full", message)


@contextmanager
def map_os_error(path: str | None = None) -> Iterator[None]:
    """Map a no-space ``OSError`` to ``DiskFullError``.

    Callers wrap the write. This function is the only place that turns
    ``errno.ENOSPC`` into ``disk_full``. Every other exception propagates.
    """
    try:
        yield
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            raise DiskFullError(path) from exc
        raise


class TimebaseError(LabError):
    """PTS, a duration, or an interval cannot be placed on the source clock."""

    def __init__(self, message: str) -> None:
        if not isinstance(message, str) or not message.strip():
            message = (
                "Некорректная шкала времени. "
                "Проверьте PTS, time base и границы интервала."
            )
        super().__init__("timebase", message)


class DuplicateFrameError(LabError):
    """One ``source_frame_index`` was produced twice in a single pass."""

    def __init__(self, source_frame_index: int) -> None:
        self.source_frame_index = source_frame_index
        super().__init__(
            "duplicate_frame",
            f"Исходный кадр {source_frame_index} уже выдан в этом проходе. "
            "Повторная обработка одного кадра остановлена.",
        )


class CheckpointMissingError(LabError):
    """Custom mode was requested and the weights file is not there.

    Baseline COCO is not a substitute.
    """

    def __init__(self, path: str | None = None) -> None:
        if path:
            message = (
                f"Нет весов обученной модели: {path}. "
                "Положите совместимый bundle с weights.pth и четырьмя классами. "
                "Базовая COCO-модель не подставляется."
            )
        else:
            message = (
                "Нет весов обученной модели. "
                "Укажите путь к совместимому bundle с weights.pth и четырьмя классами. "
                "Базовая COCO-модель не подставляется."
            )
        super().__init__("checkpoint_missing", message)


_MATCHING_BUNDLE_STEP = (
    "Обучите модель заново или выберите bundle с тем же вариантом, "
    "разрешением и порядком классов."
)


class IncompatibleCheckpointError(LabError):
    """Variant, resolution, or class order does not match the bundle.

    The resume checkpoint is not a substitute, and baseline COCO is not either.
    """

    def __init__(self, message: str) -> None:
        if not isinstance(message, str) or not message.strip():
            message = (
                "Чекпоинт не совпадает с bundle: вариант, разрешение "
                "или порядок классов. Базовая COCO-модель не подставляется."
            )
        text = message.strip()
        if _MATCHING_BUNDLE_STEP not in text:
            text = f"{text} {_MATCHING_BUNDLE_STEP}"
        super().__init__("incompatible_checkpoint", text)


class JobAlreadyRunningError(LabError):
    """A queued, running, or cancel-requested job already holds the workspace."""

    def __init__(self) -> None:
        super().__init__(
            "job_already_running",
            "В этом рабочем каталоге уже выполняется задание. "
            "Новое обучение или анализ не запускается, "
            "пока текущее не завершится или не будет отменено.",
        )


class ColorOrderError(LabError):
    """A channel-reversed view was passed where the adapter requires RGB."""

    def __init__(self) -> None:
        super().__init__(
            "color_order",
            "Кадр с отрицательным шагом по каналам не принимается как RGB. "
            "Обменяйте каналы BGR один раз и передайте непрерывный RGB. "
            "Повторный обмен каналов не выполняется.",
        )


class SplitNotFrozenError(LabError):
    """``split_manifest.json`` is not in the project store for this video."""

    def __init__(self, video_id: str) -> None:
        self.video_id = video_id
        super().__init__(
            "split_not_frozen",
            f"Разбиение видео «{video_id}» не зафиксировано. "
            "Сначала зафиксируйте split.",
        )


class SplitHashMismatchError(LabError):
    """The frozen file body no longer matches ``content_hash``."""

    def __init__(self) -> None:
        super().__init__(
            "split_hash_mismatch",
            "Содержимое split_manifest.json не совпадает с content_hash. "
            "Зафиксированное разбиение изменено.",
        )


class UnknownCategoryError(LabError):
    """``category_name`` is not one of the four names in ``configs/labels.yaml``.

    The name is not replaced with a nearby class. ``class_id`` stays unset,
    including when a caller also passed COCO category id 0.
    """

    def __init__(self, category_name: str, frame_id: str | None = None) -> None:
        self.category_name = category_name
        self.frame_id = frame_id
        self.class_id = None
        where = f" Кадр {frame_id}." if frame_id else ""
        super().__init__(
            "unknown_category",
            f"Неизвестная категория «{category_name}». "
            "Имя не заменяется ближайшим классом и не считается фоном." + where,
        )


class UnsupportedCrowdError(LabError):
    """``iscrowd`` is not 0.

    Crowd is not visibility ``partial`` and not an occlusion flag.
    ``visibility`` on the error stays unset.
    """

    def __init__(self, frame_id: str | None = None, iscrowd: int | None = None) -> None:
        self.frame_id = frame_id
        self.iscrowd = iscrowd
        self.visibility = None
        if iscrowd is None:
            head = "Значение iscrowd, отличное от 0, не поддерживается."
        else:
            head = f"Значение iscrowd={iscrowd} не поддерживается."
        where = f" Кадр {frame_id}." if frame_id else ""
        super().__init__(
            "unsupported_crowd",
            f"{head} Оно не записывается как окклюзия и не подменяет неопределённость."
            + where,
        )


class DatasetHashMismatchError(LabError):
    """Two runs do not share ``dataset_hash``. No delta is attached."""

    def __init__(self) -> None:
        super().__init__(
            "dataset_hash_mismatch",
            "Сравнение прогонов невозможно: dataset_hash не совпадает. "
            "Разница метрик не выводится.",
        )


class NotFinalError(LabError):
    """``--final`` is missing a frozen manifest, checksum, or config hash.

    The command does not train and does not rewrite thresholds.
    """

    def __init__(self, missing: str) -> None:
        self.missing = missing
        super().__init__(
            "not_final",
            "Финальная оценка на test не выполняется: "
            f"{missing}. "
            "Обучение не запускается и пороги не меняются.",
        )


class TestNotLabeledError(LabError):
    """The test split has no labels.

    ``report`` holds «не рассчитано» for the test numbers. Validation
    metrics are not copied into it.
    """

    def __init__(self, report: dict[str, object]) -> None:
        self.report = report
        super().__init__(
            "test_not_labeled",
            "Разметка test отсутствует. Числа test — «не рассчитано». "
            "Метрики validation не подставляются.",
        )
