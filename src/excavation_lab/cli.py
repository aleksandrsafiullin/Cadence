"""Typer entrypoint and JSON result writer.

YAML, when this package starts reading it, goes through ``yaml.safe_load`` only.
Subprocesses use an argument array and ``sys.executable``.

Forbidden in ``src``, including when only mentioned in a comment:
import ultralytics, import fastapi, import mlflow, sv.ByteTrack,
shell=True, torch.compile. Starlette and Jinja2 are imported from ``web/`` only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from excavation_lab.analytics import unavailable_error
from excavation_lab.datasets.coco_import import format_import_summary, import_coco
from excavation_lab.datasets.cvat_export import export_cvat, format_export_summary
from excavation_lab.datasets.sampler import format_sample_summary, sample_frames
from excavation_lab.datasets.snapshot import format_snapshot_summary, snapshot_dataset
from excavation_lab.datasets.splits import (
    freeze_split,
    propose_for_project,
    write_proposal,
)
from excavation_lab.datasets.validate import format_validation_summary, validate_dataset
from excavation_lab.errors import LabError, TestNotLabeledError
from excavation_lab.hardware import (
    default_report_path,
    run_doctor,
    russian_summary,
    weights_cache_dir,
    weights_download_notice,
)
from excavation_lab.media.probe import (
    add_video,
    format_probe_line,
    format_video_line,
    reprobe_video,
)
from excavation_lab.models.evaluate import evaluate_project, format_evaluation_summary
from excavation_lab.models.train import enqueue_train
from excavation_lab.storage import create_project, default_workspace_root
from excavation_lab.tracking.analyze import enqueue_analyze, format_analyze_summary
from excavation_lab.tracking.export_tables import (
    export_project_run,
    format_csv_export_summary,
)
from excavation_lab.visualization.annotate import AnnotateToggles
from excavation_lab.visualization.export_video import (
    enqueue_video_export,
    format_video_export_summary,
)

app = typer.Typer(
    name="excavation",
    no_args_is_help=True,
    add_completion=False,
)


# Typer builds a command group only when a callback or a subcommand exists.
@app.callback()
def main() -> None:
    """Локальный прототип видеохронометража экскавации."""


def emit(
    ok: bool,
    data: Any = None,
    error: LabError | Mapping[str, Any] | None = None,
) -> int:
    """Print the CLI JSON result and return exit code 0 or 1.

    Success is ``{"ok": true, "data": ...}``.
    Failure is ``{"ok": false, "error": {"code": "...", "message": "..."}}``.
    When ``data`` is passed with a failure, the report is included too.
    """
    if ok:
        payload: dict[str, Any] = {"ok": True, "data": data}
        exit_code = 0
    else:
        payload = {"ok": False, "error": _error_body(error)}
        if data is not None:
            payload["data"] = data
        exit_code = 1
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return exit_code


project_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Проекты в workspace.",
)
app.add_typer(project_app, name="project")


def _raise_lab_error(error: LabError, *, json_output: bool) -> NoReturn:
    if json_output:
        raise typer.Exit(emit(False, error=error))
    typer.echo(f"{error.code}: {error.message}", err=True)
    raise typer.Exit(1)


@project_app.command("create")
def project_create(
    name: Annotated[str, typer.Option("--name", help="Имя проекта.")],
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Создать проект и каталоги хранения."""
    try:
        record = create_project(name, workspace)
    except LabError as exc:
        failure: LabError | None = exc
        record = None
    else:
        failure = None
    if failure is not None:
        _raise_lab_error(failure, json_output=json_output)
    assert record is not None
    if json_output:
        raise typer.Exit(emit(True, data=record))
    typer.echo(f"Проект «{record['name']}» создан: {record['project_dir']}")


@app.command("doctor")
def doctor(
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            help="Куда записать environment_report.json. "
            "По умолчанию workspace/environment_report.json.",
        ),
    ] = None,
    load_weights: Annotated[
        bool,
        typer.Option(
            "--load-weights",
            help="Загрузить официальные веса Nano, прогнать один кадр и один шаг. "
            "Без флага эти проверки не запускаются и сеть не используется.",
        ),
    ] = False,
    allow_cpu_fallback: Annotated[
        bool,
        typer.Option(
            "--allow-cpu-fallback",
            help="Разрешить шаг обучения на CPU, если CUDA и MPS шаг не завершили. "
            "Без флага CPU не подставляется.",
        ),
    ] = False,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. Кэш весов: workspace/cache/weights/.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Проверить ОС, библиотеки, FFmpeg и устройство вывода."""
    destination = output if output is not None else default_report_path()
    root = _workspace(workspace)
    if load_weights:
        typer.echo(weights_download_notice(weights_cache_dir(root)), err=True)
    try:
        report = run_doctor(
            destination,
            load_weights=load_weights,
            allow_cpu_fallback=allow_cpu_fallback,
            workspace=root,
        )
    except LabError as exc:
        if not json_output:
            typer.echo(f"Отчёт: {destination.resolve()}", err=True)
        _raise_lab_error(exc, json_output=json_output)
    if json_output:
        raise typer.Exit(emit(True, data=report))
    typer.echo(russian_summary(report))


@app.command("ui")
def ui(
    port: Annotated[
        int,
        typer.Option(
            "--port",
            help="Порт. По умолчанию 8501. Слушает только 127.0.0.1.",
        ),
    ] = 8501,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
) -> None:
    """Запустить локальный интерфейс."""
    # Stay inside the command so `doctor --help` does not import Starlette.
    from excavation_lab.web.server import serve

    serve(_workspace(workspace), port=port)


@app.command("train")
def train_command(
    project: Annotated[str, typer.Option("--project", help="Идентификатор проекта.")],
    dataset: Annotated[str, typer.Option("--dataset", help="Идентификатор снимка датасета.")],
    profile: Annotated[
        str,
        typer.Option("--profile", help="Профиль smoke, overfit или pilot-small."),
    ],
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Поставить обучение RF-DETR в очередь. Модель загружает worker."""
    root = _workspace(workspace)
    try:
        record = enqueue_train(root, project, dataset, profile)
    except LabError as exc:
        _raise_lab_error(exc, json_output=json_output)
    if json_output:
        raise typer.Exit(emit(True, data=record.model_dump(mode="json")))
    typer.echo(f"Обучение поставлено в очередь: {record.job_id}")


@app.command("evaluate")
def evaluate_command(
    project: Annotated[str, typer.Option("--project", help="Идентификатор проекта.")],
    model: Annotated[str, typer.Option("--model", help="Идентификатор модели.")],
    split: Annotated[
        str,
        typer.Option("--split", help="validation или test."),
    ],
    final: Annotated[
        bool,
        typer.Option(
            "--final",
            help="Финальная оценка test. Не обучает и не меняет пороги.",
        ),
    ] = False,
    detections: Annotated[
        Path | None,
        typer.Option(
            "--detections",
            help="JSON с рамками детектора. Треки и сглаженные рамки не принимаются. "
            "Без флага кадры читаются из снимка и прогоняются через модель.",
        ),
    ] = None,
    compare: Annotated[
        str | None,
        typer.Option(
            "--compare",
            help="Идентификатор другого прогона. Дельта только при одном dataset_hash.",
        ),
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Посчитать AP и precision/recall по рамкам детектора. Обучение не запускается."""
    root = _workspace(workspace)
    try:
        report = evaluate_project(
            root,
            project,
            model,
            split,
            final=final,
            detections_path=detections,
            compare_to=compare,
        )
    except LabError as exc:
        unlabeled = exc.report if isinstance(exc, TestNotLabeledError) else None
        if json_output:
            raise typer.Exit(emit(False, data=unlabeled, error=exc))
        if unlabeled is not None:
            typer.echo(format_evaluation_summary(unlabeled), err=True)
        typer.echo(f"{exc.code}: {exc.message}", err=True)
        raise typer.Exit(1)
    if json_output:
        raise typer.Exit(emit(True, data=report))
    typer.echo(format_evaluation_summary(report))


@app.command("analyze")
def analyze_command(
    project: Annotated[str, typer.Option("--project", help="Идентификатор проекта.")],
    video: Annotated[str, typer.Option("--video", help="Идентификатор видео.")],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="baseline или идентификатор custom bundle. "
            "Неизвестный id не заменяется базовой моделью.",
        ),
    ],
    fps: Annotated[
        float,
        typer.Option("--fps", help="Частота анализа, кадров в секунду."),
    ],
    start: Annotated[
        float,
        typer.Option("--start", help="Начало интервала, секунды исходного видео."),
    ] = 0.0,
    end: Annotated[
        float | None,
        typer.Option("--end", help="Конец интервала, не включая эту секунду."),
    ] = None,
    view: Annotated[
        str | None,
        typer.Option("--view", help="Идентификатор ракурса. Можно не указывать."),
    ] = None,
    reset_reason: Annotated[
        str | None,
        typer.Option(
            "--reset-reason",
            help="Явный сброс треков при смене сцены. Детектора склеек нет.",
        ),
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Поставить анализ видео в очередь. Детектор загружает worker."""
    root = _workspace(workspace)
    try:
        enqueued = enqueue_analyze(
            root,
            project,
            video,
            model,
            fps,
            start_s=start,
            end_s=end,
            view_id=view,
            reset_reason=reset_reason,
        )
    except LabError as exc:
        _raise_lab_error(exc, json_output=json_output)
    if json_output:
        raise typer.Exit(emit(True, data=enqueued.payload()))
    typer.echo(format_analyze_summary(enqueued))


@app.command("export")
def export_command(
    project: Annotated[str, typer.Option("--project", help="Идентификатор проекта.")],
    run: Annotated[str, typer.Option("--run", help="Идентификатор прогона.")],
    format_name: Annotated[
        str,
        typer.Option(
            "--format",
            help="csv или video. События в версии 0.1 не записываются.",
        ),
    ],
    boxes: Annotated[
        bool,
        typer.Option("--boxes/--no-boxes", help="Рисовать рамки."),
    ] = True,
    class_names: Annotated[
        bool,
        typer.Option("--classes/--no-classes", help="Подписи классов."),
    ] = True,
    scores: Annotated[
        bool,
        typer.Option("--scores/--no-scores", help="Показывать score."),
    ] = True,
    track_ids: Annotated[
        bool,
        typer.Option("--track-ids/--no-track-ids", help="Показывать ID трека."),
    ] = True,
    trail: Annotated[
        bool,
        typer.Option("--trail/--no-trail", help="Траектория ковша."),
    ] = True,
    zones: Annotated[
        bool,
        typer.Option("--zones/--no-zones", help="Зоны. Пустой список ничего не рисует."),
    ] = True,
    export_fps: Annotated[
        float | None,
        typer.Option(
            "--export-fps",
            help="Частота ролика. Не частота исходника и не частота анализа.",
        ),
    ] = None,
    excel: Annotated[
        bool,
        typer.Option("--excel", help="Добавить UTF-8 BOM для Excel."),
    ] = False,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Записать CSV или аннотированное видео. События версии 0.1 не создаются."""
    kind = format_name.strip().lower()
    if kind == "events":
        _raise_lab_error(unavailable_error(), json_output=json_output)
    if kind not in {"csv", "video"}:
        _raise_lab_error(
            LabError(
                "invalid_export_format",
                "Формат экспорта должен быть csv или video.",
            ),
            json_output=json_output,
        )
    root = _workspace(workspace)
    try:
        if kind == "video":
            enqueued = enqueue_video_export(
                root,
                project,
                run,
                export_fps=export_fps,
                toggles=AnnotateToggles(
                    boxes=boxes,
                    class_names=class_names,
                    scores=scores,
                    track_ids=track_ids,
                    bucket_trail=trail,
                    zones=zones,
                ),
            )
            result: Any = enqueued.payload()
        else:
            enqueued = None
            result = export_project_run(root, project, run, excel=excel)
    except LabError as exc:
        _raise_lab_error(exc, json_output=json_output)
    if json_output:
        raise typer.Exit(emit(True, data=result))
    if enqueued is not None:
        typer.echo(format_video_export_summary(enqueued))
        return
    typer.echo(format_csv_export_summary(result))


video_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Локальные видео: регистрация и ffprobe.",
)
app.add_typer(video_app, name="video")


def _workspace(path: Path | None) -> Path:
    if path is None:
        return default_workspace_root()
    return path


@video_app.command("add")
def video_add(
    project: Annotated[str, typer.Option("--project", help="Идентификатор проекта.")],
    path: Annotated[Path, typer.Option("--path", help="Локальный путь к видео.")],
    copy: Annotated[
        bool,
        typer.Option(
            "--copy",
            help="Скопировать файл в каталог media проекта. Без флага путь только регистрируется.",
        ),
    ] = False,
    source_url: Annotated[
        str | None,
        typer.Option(
            "--source-url",
            help="Сохранить URL как происхождение. Файл по ссылке не скачивается.",
        ),
    ] = None,
    video: Annotated[
        str | None,
        typer.Option("--video", help="Идентификатор видео. Если не задан, создаётся новый."),
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Зарегистрировать локальное видео. URL не скачивается."""
    try:
        metadata = add_video(
            _workspace(workspace),
            project,
            path,
            copy=copy,
            source_url=source_url,
            video_id=video,
        )
    except LabError as exc:
        _raise_lab_error(exc, json_output=json_output)
    if json_output:
        raise typer.Exit(emit(True, data=metadata))
    typer.echo(format_video_line(metadata))


@video_app.command("probe")
def video_probe(
    project: Annotated[str, typer.Option("--project", help="Идентификатор проекта.")],
    video: Annotated[str, typer.Option("--video", help="Идентификатор видео.")],
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Повторно снять ffprobe и проверить SHA-256 и размер."""
    try:
        report = reprobe_video(_workspace(workspace), project, video)
    except LabError as exc:
        _raise_lab_error(exc, json_output=json_output)
    if json_output:
        raise typer.Exit(emit(True, data=report))
    typer.echo(format_probe_line(report))


dataset_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Эпизоды, разбиение, отбор кадров, CVAT и снимок датасета.",
)
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("split-propose")
def dataset_split_propose(
    project: Annotated[str, typer.Option("--project", help="Идентификатор проекта.")],
    video: Annotated[str, typer.Option("--video", help="Идентификатор видео.")],
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Предложить разбиение. Файл ещё не зафиксирован."""
    root = _workspace(workspace)
    try:
        proposal = propose_for_project(root, project, video)
        path = write_proposal(root, project, proposal)
    except LabError as exc:
        _raise_lab_error(exc, json_output=json_output)
    payload = proposal.model_dump(mode="json")
    payload["path"] = str(path)
    payload["frozen"] = False
    if json_output:
        raise typer.Exit(emit(True, data=payload))
    if proposal.message:
        typer.echo(proposal.message)
    typer.echo(f"Предложение: {path}")
    typer.echo("Разбиение не зафиксировано.")


@dataset_app.command("split-freeze")
def dataset_split_freeze(
    project: Annotated[str, typer.Option("--project", help="Идентификатор проекта.")],
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="JSON предложения разбиения."),
    ],
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Зафиксировать предложение. Повторная фиксация того же видео не выполняется."""
    root = _workspace(workspace)
    try:
        frozen, path = freeze_split(root, project, manifest)
    except LabError as exc:
        _raise_lab_error(exc, json_output=json_output)
    payload = frozen.model_dump(mode="json")
    payload["path"] = str(path)
    if json_output:
        raise typer.Exit(emit(True, data=payload))
    typer.echo(f"Разбиение зафиксировано: {path}")
    typer.echo(f"Хеш: {frozen.content_hash}")


@dataset_app.command("sample")
def dataset_sample(
    project: Annotated[str, typer.Option("--project", help="Идентификатор проекта.")],
    split: Annotated[
        str | None,
        typer.Option("--split", help="train, validation или test."),
    ] = None,
    count: Annotated[
        int | None,
        typer.Option(
            "--count",
            help="Сколько кадров этой части. Квота не добивается копиями.",
        ),
    ] = None,
    plan: Annotated[
        str | None,
        typer.Option(
            "--plan",
            help="diagnostic, content или expanded из configs/pilot.yaml.",
        ),
    ] = None,
    video: Annotated[
        str | None,
        typer.Option(
            "--video",
            help="Идентификатор видео. Если в проекте одно видео, можно не указывать.",
        ),
    ] = None,
    strategy: Annotated[
        str | None,
        typer.Option(
            "--strategy",
            help="spaced или representative. active дополняет только train и требует --model.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Идентификатор модели для --strategy active. "
            "Без сохранённых детекций рамки не создаются.",
        ),
    ] = None,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Отобрать кадры из зафиксированного разбиения."""
    root = _workspace(workspace)
    try:
        result = sample_frames(
            root,
            project,
            video_id=video,
            split=split,
            count=count,
            plan=plan,
            strategy=strategy,
            model=model,
        )
    except LabError as exc:
        _raise_lab_error(exc, json_output=json_output)
    if json_output:
        raise typer.Exit(emit(True, data=result.model_dump(mode="json")))
    typer.echo(format_sample_summary(result))


@dataset_app.command("export-cvat")
def dataset_export_cvat(
    project: Annotated[str, typer.Option("--project", help="Идентификатор проекта.")],
    batch: Annotated[
        str,
        typer.Option(
            "--batch",
            help="Пакет <video_id>__train, <video_id>__validation или <video_id>__test.",
        ),
    ],
    include_predictions: Annotated[
        bool,
        typer.Option(
            "--include-predictions",
            help="Для train записать predictions.coco.json. "
            "Для validation и test файл не создаётся.",
        ),
    ] = False,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Собрать annotation_batch.zip. Разбиение не пересчитывается."""
    root = _workspace(workspace)
    try:
        result = export_cvat(
            root,
            project,
            batch,
            include_predictions=include_predictions,
        )
    except LabError as exc:
        _raise_lab_error(exc, json_output=json_output)
    if json_output:
        raise typer.Exit(emit(True, data=result.model_dump(mode="json")))
    typer.echo(format_export_summary(result))


@dataset_app.command("import-coco")
def dataset_import_coco(
    project: Annotated[str, typer.Option("--project", help="Идентификатор проекта.")],
    batch: Annotated[
        str,
        typer.Option(
            "--batch",
            help="Пакет <video_id>__train, <video_id>__validation или <video_id>__test.",
        ),
    ],
    file: Annotated[
        Path,
        typer.Option("--file", help="ZIP с COCO, кадрами и frames_manifest.csv."),
    ],
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Импортировать COCO. Split берётся из манифеста, не из subset CVAT."""
    root = _workspace(workspace)
    try:
        result = import_coco(root, project, batch, file)
    except LabError as exc:
        _raise_lab_error(exc, json_output=json_output)
    if json_output:
        raise typer.Exit(emit(True, data=result.model_dump(mode="json")))
    typer.echo(format_import_summary(result))


@dataset_app.command("validate")
def dataset_validate(
    project: Annotated[str, typer.Option("--project", help="Идентификатор проекта.")],
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Проверить снимок: файлы, хеши, рамки, классы и split. Без mAP."""
    root = _workspace(workspace)
    try:
        result = validate_dataset(root, project)
    except LabError as exc:
        _raise_lab_error(exc, json_output=json_output)
    body = result.model_dump(mode="json", exclude_none=True)
    if json_output:
        if result.ok:
            raise typer.Exit(emit(True, data=body))
        raise typer.Exit(emit(False, data=body, error=result.failure_error()))
    text = format_validation_summary(result)
    if result.ok:
        typer.echo(text)
        return
    typer.echo(text, err=True)
    raise typer.Exit(1)


@dataset_app.command("snapshot")
def dataset_snapshot(
    project: Annotated[str, typer.Option("--project", help="Идентификатор проекта.")],
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            help="Корень workspace. По умолчанию каталог workspace в репозитории.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Собрать train/valid/test. Без зафиксированного split команда не выполняется."""
    root = _workspace(workspace)
    try:
        result = snapshot_dataset(root, project)
    except LabError as exc:
        _raise_lab_error(exc, json_output=json_output)
    if json_output:
        raise typer.Exit(emit(True, data=result.model_dump(mode="json")))
    typer.echo(format_snapshot_summary(result))


events_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="События и показатели. В версии 0.1 не рассчитываются.",
)
app.add_typer(events_app, name="events")


def _refuse_events(
    project: str | None,
    run: str | None,
    *,
    json_output: bool,
) -> NoReturn:
    """Refuse before any project or run lookup.

    ``--project`` and ``--run`` stay optional. A missing id is
    ``unavailable``, not a Typer usage error. The values are discarded.
    """
    del project, run
    _raise_lab_error(unavailable_error(), json_output=json_output)


@events_app.command("propose")
def events_propose(
    project: Annotated[
        str | None,
        typer.Option(
            "--project",
            help="Идентификатор проекта. Команда не открывает проект.",
        ),
    ] = None,
    run: Annotated[
        str | None,
        typer.Option(
            "--run",
            help="Идентификатор прогона. Команда не открывает прогон.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Не предлагать события. Версия 0.1 их не рассчитывает."""
    _refuse_events(project, run, json_output=json_output)


@events_app.command("export")
def events_export(
    project: Annotated[
        str | None,
        typer.Option(
            "--project",
            help="Идентификатор проекта. Команда не открывает проект.",
        ),
    ] = None,
    run: Annotated[
        str | None,
        typer.Option(
            "--run",
            help="Идентификатор прогона. Команда не открывает прогон.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Печатать JSON-результат."),
    ] = False,
) -> None:
    """Не записывать events.csv и truck_visits.csv."""
    _refuse_events(project, run, json_output=json_output)


def _error_body(error: LabError | Mapping[str, Any] | None) -> dict[str, str]:
    if isinstance(error, LabError):
        return {"code": error.code, "message": error.message}
    if isinstance(error, Mapping) and "code" in error and "message" in error:
        return {"code": str(error["code"]), "message": str(error["message"])}
    return {"code": "internal_error", "message": "Внутренняя ошибка."}


if __name__ == "__main__":
    app()
