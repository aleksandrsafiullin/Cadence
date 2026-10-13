"""Native file chooser. No test runs osascript."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from excavation_lab.ui import copy as ui_copy
from excavation_lab.web import choose_file
from excavation_lab.web.choose_file import (
    ChooseFileFailed,
    choose_command,
    choose_video_path,
)

_TYPE_LIST = (
    '{"mp4", "mov", "mkv", "public.mpeg-4", '
    '"com.apple.quicktime-movie", "org.matroska.mkv"}'
)


@pytest.fixture(autouse=True)
def _block_osascript(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if the default runner reaches ``subprocess.run``."""
    monkeypatch.delenv("EXCAVATION_CHOOSE_PATH", raising=False)

    def _forbidden(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"osascript must not run: {args!r} {kwargs!r}")

    monkeypatch.setattr(choose_file.subprocess, "run", _forbidden)


def _completed(
    argv: list[str], code: int, stdout: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr="")


def test_choose_command_keeps_the_prompt_after_double_dash() -> None:
    prompt = 'Choose an MP4; rm -rf "$HOME"'
    argv = choose_command(prompt)
    assert argv[0] == "osascript"
    assert argv[1] == "-e"
    assert argv[3] == "--"
    assert argv[4] == prompt
    assert len(argv) == 5
    script = argv[2]
    assert isinstance(script, str)
    assert prompt not in script
    assert script.startswith("on run argv\n")
    assert "on error number -128" in script
    assert 'return ""' in script
    assert "return POSIX path of picked" in script
    assert _TYPE_LIST in script
    assert "with prompt (item 1 of argv)" in script


def test_cancel_returns_none() -> None:
    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(argv, 0, "\n")

    assert choose_video_path(prompt="Выберите MP4, MOV или MKV", runner=runner) is None


def test_empty_stdout_on_nonzero_exit_raises() -> None:
    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(argv, 1, "")

    with pytest.raises(ChooseFileFailed):
        choose_video_path(prompt="p", runner=runner)


def test_nonzero_exit_raises_even_when_stdout_is_a_video(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(argv, 2, f"{clip}\n")

    with pytest.raises(ChooseFileFailed):
        choose_video_path(prompt="p", runner=runner)


@pytest.mark.parametrize("suffix", [".mp4", ".mov", ".mkv", ".MP4", ".MOV", ".MKV"])
def test_existing_video_path_is_returned(tmp_path: Path, suffix: str) -> None:
    clip = tmp_path / f"clip{suffix}"
    clip.write_bytes(b"x")

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(argv, 0, f"{clip}\n")

    found = choose_video_path(prompt="p", runner=runner)
    assert found == clip
    assert isinstance(found, Path)


def test_bad_suffix_raises(tmp_path: Path) -> None:
    clip = tmp_path / "clip.txt"
    clip.write_bytes(b"x")

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(argv, 0, str(clip))

    with pytest.raises(ChooseFileFailed):
        choose_video_path(prompt="p", runner=runner)


def test_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "gone.mkv"

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(argv, 0, str(missing))

    with pytest.raises(ChooseFileFailed):
        choose_video_path(prompt="p", runner=runner)


def test_relative_runner_path_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "clip.mp4").write_bytes(b"x")

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(argv, 0, "clip.mp4\n")

    with pytest.raises(ChooseFileFailed):
        choose_video_path(prompt="p", runner=runner)


def test_directory_raises(tmp_path: Path) -> None:
    folder = tmp_path / "clip.mp4"
    folder.mkdir()

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(argv, 0, str(folder))

    with pytest.raises(ChooseFileFailed):
        choose_video_path(prompt="p", runner=runner)


def test_env_stub_skips_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen = tmp_path / "missing.txt"
    monkeypatch.setenv("EXCAVATION_CHOOSE_PATH", str(chosen))

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(argv)

    found = choose_video_path(prompt="ignored", runner=runner)
    assert found == chosen
    assert isinstance(found, Path)
    assert not chosen.exists()


@pytest.mark.parametrize("value", ["clip.mp4", "videos/a.mkv", "  "])
def test_relative_env_stub_raises_without_the_runner(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "clip.mp4").write_bytes(b"x")
    monkeypatch.setenv("EXCAVATION_CHOOSE_PATH", value)

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(argv)

    with pytest.raises(ChooseFileFailed):
        choose_video_path(prompt="ignored", runner=runner)


def test_blank_env_uses_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXCAVATION_CHOOSE_PATH", "")
    clip = tmp_path / "clip.mkv"
    clip.write_bytes(b"x")
    seen: list[list[str]] = []

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        return _completed(argv, 0, f"{clip}\n")

    prompt = "Choose an MP4, MOV, or MKV"
    assert choose_video_path(prompt=prompt, runner=runner) == clip
    assert seen == [choose_command(prompt)]


def test_unset_env_uses_the_runner(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    called = False

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        return _completed(argv, 0, str(clip))

    assert choose_video_path(prompt="p", runner=runner) == clip
    assert called
    assert "EXCAVATION_CHOOSE_PATH" not in choose_file.os.environ


def test_runner_file_not_found_becomes_choose_file_failed() -> None:
    seen: list[list[str]] = []

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        raise FileNotFoundError(argv[0])

    prompt = "Choose an MP4, MOV, or MKV"
    with pytest.raises(ChooseFileFailed):
        choose_video_path(prompt=prompt, runner=runner)
    assert seen == [choose_command(prompt)]
    assert "EXCAVATION_CHOOSE_PATH" not in choose_file.os.environ


def test_catalog_chooser_copy() -> None:
    ru = ui_copy.catalog("ru")
    en = ui_copy.catalog("en")
    assert ru.choose_file_button == "Выбрать файл"
    assert en.choose_file_button == "Choose file"
    assert ru.choose_file_prompt == "Выберите MP4, MOV или MKV"
    assert en.choose_file_prompt == "Choose an MP4, MOV, or MKV"
    assert ru.choose_failed == "Не удалось открыть выбор файла."
    assert en.choose_failed == "Could not open the file chooser."
    assert ru.add_local_video == (
        "Ролика нет. Выберите локальный MP4, MOV или MKV, который уже лежит на диске. "
        "Образец не встроен, и приложение не скачивает замену."
    )
    assert en.add_local_video == (
        "There is no video. Choose a local MP4, MOV, or MKV that is already on disk. "
        "A sample is not included, and the application does not download a substitute."
    )
    assert ui_copy.CHOOSE_FILE_BUTTON == ru.choose_file_button
    assert ui_copy.CHOOSE_FILE_PROMPT == ru.choose_file_prompt
    assert ui_copy.CHOOSE_FAILED == ru.choose_failed
    for text in (en.choose_file_prompt, en.choose_failed, en.choose_file_button, en.add_local_video):
        assert not any("\u0400" <= char <= "\u04FF" for char in text)


def test_module_does_not_shell_out_or_import_media() -> None:
    source = Path(choose_file.__file__).read_text(encoding="utf-8")
    assert "shell=True" not in source
    tree = ast.parse(source)
    imported: set[str] = set()
    runs: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            if node.module == "excavation_lab":
                imported.update(f"excavation_lab.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "run"
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
            ):
                runs.append(node)
    assert imported.isdisjoint({"starlette", "jinja2", "excavation_lab.media"})
    assert not any(name.startswith("excavation_lab.media.") for name in imported)
    assert len(runs) == 1
    keywords = {keyword.arg: keyword.value for keyword in runs[0].keywords}
    assert "shell" not in keywords
    assert isinstance(keywords["capture_output"], ast.Constant)
    assert keywords["capture_output"].value is True
    assert isinstance(keywords["text"], ast.Constant)
    assert keywords["text"].value is True
    assert isinstance(keywords["check"], ast.Constant)
    assert keywords["check"].value is False
