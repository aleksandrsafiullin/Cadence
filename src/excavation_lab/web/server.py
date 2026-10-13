"""Bind the page server to localhost."""

from __future__ import annotations

from pathlib import Path

HOST = "127.0.0.1"


def serve(workspace: Path, port: int = 8501, host: str = HOST) -> None:
    """Run uvicorn on ``127.0.0.1``.

    ``host`` is checked first. Any other value raises before uvicorn imports
    and before the socket opens. The process still binds ``HOST``, not ``host``.
    """
    if host != HOST:
        raise ValueError(f"Интерфейс слушает только {HOST}.")
    import uvicorn

    from excavation_lab.web.app import create_app

    uvicorn.run(create_app(Path(workspace)), host=HOST, port=port)
