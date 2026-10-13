# Environment

## Python

Python 3.11. `pyproject.toml` requires `>=3.11,<3.12`. Dependencies are locked in `uv.lock` and installed with `uv sync`. The lockfile is tracked.

## FFmpeg

`ffmpeg` and `ffprobe` must be on `PATH`. Probe, decode, and encode call those binaries. The app does not ship a private FFmpeg build.

## Page server

The UI is a localhost Starlette server. Jinja2 pages, our CSS, and small vanilla JS.

- Bind address: `127.0.0.1`. The app listens on localhost only.
- Upload cap: 8192 MiB. Do not raise it without a reason.
- Telemetry off.
- No auth.

A file already on disk can be added by a local path under `workspace/`. A long file is that path, not a RAM load.

## CVAT

CVAT Community is a separate install. Follow the vendor guide:

<https://docs.cvat.ai/docs/administration/community/basics/installation/>

The app does not start CVAT containers. Export a CVAT task from this lab, mark it in CVAT, then import the COCO zip back.
