# Excavation Video Lab

Local v0.1 prototype for excavation video chronometry. Spec: `EXCAVATION_VIDEO_LAB_SPEC.md`.

## Install

Python 3.11. From this directory on a Mac:

```bash
uv sync
```

Then:

```bash
uv run excavation doctor
uv run excavation ui
```

The app listens on localhost only (`127.0.0.1`). It is not a published service.

Run other commands through `uv run` (`uv run excavation --help`, `uv run pytest`, `uv run ruff`).

## Pilot video

Put the pilot MP4 on a local path you control. `workspace/` is gitignored and is the place for that file and other local data. This repository does not contain the video, and there is no substitute download.

Provenance of the first pilot only: <https://www.youtube.com/watch?v=kjWsR2lVeek>

That URL records where the clip came from. It is not a download step and it does not replace the local MP4.

## Commands

Paths and identifiers below are placeholders. `train --profile pilot-small` is a command that exists. This README does not record a run of that profile.

```bash
uv run excavation doctor
uv run excavation ui

uv run excavation project create --name pilot
uv run excavation video add --project pilot --path /absolute/path/pilot.mp4
uv run excavation video probe --project pilot --video VIDEO_ID

uv run excavation dataset split-propose --project pilot --video VIDEO_ID
uv run excavation dataset split-freeze --project pilot --manifest SPLIT_JSON
uv run excavation dataset sample --project pilot --split train --count 60
uv run excavation dataset export-cvat --project pilot --batch BATCH_ID
uv run excavation dataset import-coco --project pilot --batch BATCH_ID --file CVAT_ZIP
uv run excavation dataset validate --project pilot
uv run excavation dataset snapshot --project pilot

uv run excavation train --project pilot --dataset DATASET_ID --profile smoke
uv run excavation train --project pilot --dataset DATASET_ID --profile pilot-small
uv run excavation evaluate --project pilot --model MODEL_ID --split validation
uv run excavation evaluate --project pilot --model MODEL_ID --split test --final

uv run excavation analyze --project pilot --video VIDEO_ID --model MODEL_ID --fps 5
uv run excavation export --project pilot --run RUN_ID --format video
uv run excavation export --project pilot --run RUN_ID --format csv
```

`events propose` and `events export` are unavailable in v0.1:

```bash
uv run excavation events propose --project pilot --run RUN_ID
uv run excavation events export --project pilot --run RUN_ID
```

События и показатели в версии 0.1 не рассчитываются.

## Tests

```bash
uv run pytest tests -q
uv run ruff check src tests
```

A green test suite is not acceptance of a trained model.

Synthetic fixtures are labeled and are not accuracy evidence. This repository does not claim detection or tracking quality.

## Licenses

Third-party packages keep their own licenses. Nano and Small were documented as Apache 2.0 by the vendor. That vendor note covers those weights. It does not license this environment as Apache 2.0. A public license for this project is chosen separately.

## Docs

- `docs/ANNOTATION_GUIDE.md` — how to mark the four classes
- `docs/ENVIRONMENT.md` — Python, lockfile, FFmpeg, page server, CVAT
- `docs/DECISIONS.md` — resolver and library facts already checked
- `docs/SCHEMAS.md` — result files and units
- `docs/MODEL_CARD_TEMPLATE.md` — card fields for a real bundle
- `docs/TEST_REPORT.md` — commands and measurements for this handoff
