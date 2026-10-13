"""macOS file dialog that returns one local video path.

The page server calls this. Probe math does not. Pytest injects a runner
or ``EXCAVATION_CHOOSE_PATH`` and must not open the dialog.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

_ENV_CHOOSE_PATH = "EXCAVATION_CHOOSE_PATH"
_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv"})
# Extensions alone are not enough for `choose file of type`.
_SCRIPT = """\
on run argv
  try
    set picked to choose file of type {"mp4", "mov", "mkv", "public.mpeg-4", "com.apple.quicktime-movie", "org.matroska.mkv"} with prompt (item 1 of argv)
    return POSIX path of picked
  on error number -128
    return ""
  end try
end run
"""


class ChooseFileFailed(Exception):
    """The dialog did not return a usable path."""


def choose_command(prompt: str) -> list[str]:
    """Argument array for one ``osascript`` choose-file dialog."""
    return ["osascript", "-e", _SCRIPT, "--", prompt]


def choose_video_path(*, prompt: str, runner: Callable | None = None) -> Path | None:
    """Return one absolute video path, or ``None`` when the user cancels.

    A non-empty ``EXCAVATION_CHOOSE_PATH`` is returned as ``Path`` when it is
    absolute. That path is not checked for existence or suffix, and ``runner``
    is not called. A relative value raises ``ChooseFileFailed`` without
    calling ``runner``. Unset or ``""`` uses ``runner``, or ``subprocess.run``
    on ``choose_command(prompt)`` when ``runner`` is omitted.
    """
    stub = os.environ.get(_ENV_CHOOSE_PATH)
    if isinstance(stub, str) and stub != "":
        path = Path(stub)
        if not path.is_absolute():
            raise ChooseFileFailed
        return path

    execute = _default_runner if runner is None else runner
    try:
        completed = execute(choose_command(prompt))
    except (OSError, UnicodeError) as exc:
        raise ChooseFileFailed from exc
    if completed.returncode != 0:
        raise ChooseFileFailed
    stdout = completed.stdout
    if not isinstance(stdout, str):
        raise ChooseFileFailed
    text = stdout.strip()
    # Cancel (-128) is exit 0 and an empty return. Non-zero already failed above.
    if text == "":
        return None
    return _existing_video(text)


def _default_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
    )


def _existing_video(text: str) -> Path:
    path = Path(text)
    if (
        not path.is_absolute()
        or path.suffix.lower() not in _VIDEO_SUFFIXES
        or not path.is_file()
    ):
        raise ChooseFileFailed
    return path
