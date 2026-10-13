# Subtask 1: Native file chooser

## Goal

Add a localhost macOS file chooser that returns one absolute path to an existing `.mp4`, `.mov`, or `.mkv`, and that pytest can stub without opening a dialog.

## Context

Spec §6.1: the UI's primary input is a local file inside the allowed directories. The video page today uses `<input type="file" id="upload-file">`, whose `.value` is a fakepath and which does not register anything until a second submit. This subtask does not change the template. It only adds the callable the next subtask's route will use.

The chooser is a page-server concern. `src/excavation_lab/media/probe.py` keeps probe math and must not import it. `src/excavation_lab/ui/video_page.py` must not gain `subprocess` or Starlette (`test_page_does_not_shell_out_or_build_a_detector`).

## Acceptance Criteria

- [ ] `src/excavation_lab/web/choose_file.py` exposes `choose_command(prompt: str) -> list[str]` and `choose_video_path(*, prompt: str, runner: Callable | None = None) -> Path | None`.
- [ ] `choose_command` returns `["osascript", "-e", <script>, "--", prompt]`. The script is one argument. The prompt is the argument after `--`, not interpolated into a shell string. The script asks for one file of type `mp4` / `mov` / `mkv` (UTI aliases `public.mpeg-4` and `com.apple.quicktime-movie` may be included) and returns its POSIX path.
- [ ] User cancel is AppleScript error number `-128`. The script catches that number, returns `""`, and exits 0. `choose_video_path` then returns `None`.
- [ ] Any other non-zero exit raises `ChooseFileFailed` (defined in this module, no user-facing paragraph). Empty stdout that is not the cancel path also raises `ChooseFileFailed`.
- [ ] A successful stdout is stripped, must be absolute, must name an existing file, and the suffix must be `.mp4`, `.mov`, or `.mkv` (case-insensitive). Otherwise raise `ChooseFileFailed`. Do not copy the file and do not call `add_video`.
- [ ] The default runner is `subprocess.run(argv, capture_output=True, text=True, check=False)` with `argv` from `choose_command`. `shell` is not an argument. The module source does not contain `shell=True`.
- [ ] When `EXCAVATION_CHOOSE_PATH` is set to a non-empty string, the runner is not called. An absolute path is returned as `Path` even if the file is missing or the suffix is wrong (the route in subtask 2 decides). A relative value raises `ChooseFileFailed` and still does not call the runner. Unset or blank falls through to the runner.
- [ ] `tests/unit/test_choose_file.py` covers: argv shape, cancel → `None`, bad suffix / missing file / relative path → `ChooseFileFailed`, env stub does not call a runner that would raise if called, and the default runner is not invoked by any test. No test executes `osascript`.
- [ ] Catalog strings for the dialog prompt and the failure sentence exist in both locales, even though the template does not show them yet: `choose_file_prompt` («Выберите MP4, MOV или MKV» / `Choose an MP4, MOV, or MKV`) and `choose_failed` («Не удалось открыть выбор файла.» / `Could not open the file chooser.`). Add the fields on `Catalog`, both `_RussianCatalog` and `_EnglishCatalog` literals, and the `RU` module aliases next to the other video aliases. English has no Cyrillic. `tests/unit/test_ui_locale.py::_assert_translated` stays green.

## Implementation Notes

`choose_video_path` does not import Starlette, Jinja2, or `excavation_lab.media`. It may raise `ChooseFileFailed` only. The route will map that to `choose_failed`.

Do not read a path from an HTTP body. The environment variable is the browser-pass seam for subtask 3, not a product setting. Do not mention it in the page copy.

A trailing newline from `osascript` is stripped. `POSIX path of` a file has no trailing slash; still `Path` the stripped text and require `path.is_absolute()`.

Suggested script body, passed as the single `-e` argument:

```applescript
on run argv
  try
    set picked to choose file of type {"mp4", "mov", "mkv", "public.mpeg-4", "com.apple.quicktime-movie"} with prompt (item 1 of argv)
    return POSIX path of picked
  on error number -128
    return ""
  end try
end run
```

## Dependencies

None.

## Verification

```bash
uv run pytest tests/unit/test_choose_file.py tests/unit/test_ui_locale.py -q
uv run ruff check src tests
```

Confirm no test process has an open file dialog. Do not set `EXCAVATION_CHOOSE_PATH` in the shell that runs the rest of the suite unless a test sets it itself and clears it.
