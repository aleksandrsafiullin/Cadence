# Subtask 2: Add the page-server dependencies

## Goal

Add Starlette, Jinja2, and uvicorn to the project, and httpx for tests. Leave Streamlit installed.

## Context

Spec §2 after subtask 1. The server is not written here. `pyproject.toml` today lists `streamlit` and does not list `starlette`, `jinja2`, or `uvicorn`. Altair is not a direct dependency; subtask 10 adds it. Do not remove `streamlit` in this subtask. `tests/unit/test_forbidden_stack.py` already rejects `fastapi`. That rejection stays.

## Acceptance Criteria

- [ ] `uv add starlette jinja2 uvicorn` and `uv add --dev httpx` were actually run. Versions are whatever the resolver wrote into `uv.lock`. Do not hand-edit the lock.
- [ ] `pyproject.toml` still depends on `streamlit`.
- [ ] `fastapi` is not a dependency. `tests/unit/test_forbidden_stack.py` still fails an import of `fastapi`.
- [ ] A new test walks `media`, `datasets`, `models`, `tracking`, `analytics`, `jobs.py`, `storage.py`, `hardware.py`, and `schemas.py` and fails if any of them import `starlette`, `jinja2`, or `streamlit`.
- [ ] `tests/unit/test_no_streamlit_import.py` is unchanged. UI modules and `app.py` may still import Streamlit.
- [ ] No React, Vue, or Node toolchain file is added. No `package.json`.

## Implementation Notes

Put the core-import test in `tests/unit/test_core_imports.py`. Parse files with `ast`, same approach as `test_no_streamlit_import.py`. Do not import the web package here; it does not exist yet.

`httpx` is only for `starlette.testclient.TestClient` in later subtasks. It is a dev dependency.

If `uv add` fails, record the error in `docs/DECISIONS.md` and stop. Do not invent a pin.

## Dependencies

Subtask 1 (the spec already names these libraries). The add can run against the current spec if subtask 1 is in the same branch; do not add FastAPI because the spec forbids it.

## Verification

```bash
uv run pytest tests/unit/test_forbidden_stack.py tests/unit/test_core_imports.py tests/unit/test_no_streamlit_import.py -q
uv run ruff check src tests
```

Do not start a server. Do not remove Streamlit.
