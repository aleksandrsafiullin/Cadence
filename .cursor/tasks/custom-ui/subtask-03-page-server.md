# Subtask 3: Page server and CLI entry

## Goal

Boot a Starlette app bound to `127.0.0.1`, with a base template and `uv run excavation ui`. Pages can be empty shells. Video behavior is the next subtask.

## Context

Spec §22 and §23. `app.py` is the Streamlit entry and must keep working until subtask 11. This subtask adds `src/excavation_lab/web/` beside it. `src/excavation_lab/cli.py` is the Typer app (`excavation = excavation_lab.cli:app`). Its module docstring currently says `import streamlit` belongs only in `ui/` or `app.py`.

## Acceptance Criteria

- [ ] `src/excavation_lab/web/app.py` exposes `create_app(workspace: Path) -> Starlette`. `src/excavation_lab/web/server.py` sets `HOST = "127.0.0.1"` and `serve(workspace, port=8501)` calls uvicorn with that host. A host other than `127.0.0.1` raises before the socket opens.
- [ ] `excavation ui` is a Typer command. Default port 8501. The server import sits inside the command function. `excavation doctor --help` does not import Starlette (assert with `sys.modules` in a test that invokes the help path, or by importing `cli` and checking `starlette` was not imported until `serve` runs).
- [ ] `create_app` mounts static files and a Jinja2 environment with autoescape on, loading `src/excavation_lab/web/templates/` and `src/excavation_lab/web/static/` via `Path(__file__)`.
- [ ] `templates/base.html` is a document with `<html lang="ru" data-theme="dark">`, a skip to content, and blocks for sidebar and main. It links `/static/lab_theme.css` and `/static/lab.css`. No CDN URL. `lab.css` contains no hex colors; colors are `var(--pdm-…)`.
- [ ] `GET /` returns 200 and the product title from the Russian catalog (`Excavation Video Lab`) for an empty `tmp_path` workspace. The body has no `<video>` element.
- [ ] `tests/unit/test_ui_empty_states.py` localhost assertions gain a sibling that reads `HOST` and fails if it is `0.0.0.0`. The existing `.streamlit/config.toml` test stays until subtask 11.
- [ ] The auth/telemetry scan in `test_shell_does_not_call_auth_or_telemetry` also walks `src/excavation_lab/web/`. New files contain no `0.0.0.0`, no `st.login`, no `telemetry.streamlit.io`, and no `http://` or `https://` URL.
- [ ] `web/` does not import `fastapi` or `streamlit`. Core modules still do not import Starlette.
- [ ] `cli.py` docstring says Starlette and Jinja2 are imported from `web/` only. It may still mention that Streamlit remains in `ui/` and `app.py` until removal.

## Implementation Notes

Routes in this subtask: `GET /` renders `base.html` with the Russian catalog product title and an empty main block. Do not port the video form yet. Do not delete `app.py`.

`lab_theme.css` is served from `src/excavation_lab/ui/lab_theme.css` (a second static mount or an explicit route). Do not copy the file. Do not edit the design fence.

Jinja: `select_autoescape(["html"])`.

Tests use `starlette.testclient.TestClient` and `create_app(tmp_path)`. No live port in unit tests.

`pyproject.toml` hatch package is already `src/excavation_lab`, so `web` is included. Templates are loaded from disk next to the source, the same way `tokens.py` loads CSS. Editable `uv run` is the supported launch.

Keep the command array style of the rest of the CLI. Do not use `shell=True`.

## Dependencies

Subtask 2.

## Verification

```bash
uv run pytest tests/unit/test_core_imports.py tests/unit/test_forbidden_stack.py tests/unit/test_ui_empty_states.py tests/unit/test_no_streamlit_import.py -q
uv run ruff check src tests
uv run excavation ui --help
```

`--help` must exit 0 and mention the command. Do not leave the server running in the test. A manual check, if you start it, is `http://127.0.0.1:8501/` showing the product title and no video element. Stop the process when done.
