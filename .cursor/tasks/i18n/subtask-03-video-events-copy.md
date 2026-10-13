# Subtask 3: Video and Events copy

## Goal

Bind Video and Events to the active catalog. `render` and every `on_click` / `on_change` on those pages resolve one catalog and use it for every user-visible string, including buttons and captions that already live in `copy.py`. At locale `ru` the screen still matches the module-level constants the current tests assert. Events stays unavailable in both languages.

## Context

Spec §15 «Видео» and «События», spec §21 missing measurements. `events_page.py` shows `ui_copy.EVENTS_UNAVAILABLE` and then `unavailable_error().message` (`События и показатели в версии 0.1 не рассчитываются.`). CLI tests lock that `LabError` message. The page must keep calling `unavailable_error()` and must stop using its message as the visible caption.

`video_page.py` still has Russian outside `copy.py`: the extra-directory sentence, `с` suffixes, duplicate-video `LabError`, and `_hardware_text` (`да` / `нет` / `байт` / `найдены`). It also displays `hardware.weights_download_notice`, which tests find via the substring `Официальные веса`. That helper stays Russian for the CLI. The page formats the same warning from the catalog.

## Acceptance Criteria

- [ ] `video_page.render`, `events_page.render`, and their `on_click` / `on_change` callbacks each call the resolver once and pass that catalog into every label, button, caption, help string, and error the page itself shows. That includes existing keys such as `ADD_VIDEO_BUTTON`, `ADD_LOCAL_VIDEO`, `LOCAL_PATH_LABEL`, `EVENTS_UNAVAILABLE`, and `EVENT_INDICATOR_LABELS`. A widget path that still reads `ui_copy.ADD_VIDEO_BUTTON` (or any other module-level sentence) is not done.
- [ ] Helpers that tests call with no catalog argument still return Russian: `job_status_line`, offset and time suffixes (`300 с`, `0.000 с` style), upload rejection constants. `render` does not use that default. It passes the resolved catalog. At `ru`, the rendered string equals the module-level constant.
- [ ] New strings are added to both catalogs in this subtask. Russian text is moved unchanged. English is a full sentence, not a shortened label.
- [ ] The hardware caption is one catalog sentence. The Russian wording is unchanged and includes the device, CUDA, MPS, memory (`Память` / `байт` or «не рассчитано»), and `ffprobe и ffmpeg` (`найдены` / `не найдены`), plus `HARDWARE_PROBES`. It does not call `hardware.russian_summary`. `hardware.py` is unchanged.
- [ ] The on-screen weights warning comes from the catalog and still contains `Официальные веса` in Russian. The cache path stays in the sentence. `weights_download_notice` is still what `doctor` prints.
- [ ] Duplicate-video text raised in `video_page.py` comes from the catalog. Other `exc.message` values from media/storage stay as the service raised them.
- [ ] Events: `unavailable_error()` is still called. The visible warning and caption are catalog strings in the active language. Metrics use the active `not calculated` / «не рассчитано». No buttons, no chart, no zero. `EVENTS_V02_CAPTION` is translated and still mentions `v0.2`.
- [ ] `analytics/__init__.py` message stays the Russian sentence the CLI test asserts.
- [ ] No emoji. `icon=""` on warnings and errors that already pass it.

## Implementation Notes

Files: `src/excavation_lab/ui/video_page.py`, `src/excavation_lab/ui/events_page.py`, `src/excavation_lab/ui/copy.py`.

Resolve at the top of `render` and again inside each callback. Do not close over the module-level `ui_copy` names for widget text. Callbacks that run outside `render` call the resolver themselves so they do not keep a Russian catalog captured at import.

`job_status_line` fragments to the catalog passed in, Russian unchanged: `процесс пропал`, `частичный результат`, and the `с` suffix on `analyzed_end_s`. Status **keys** stay English; labels come from that catalog’s `JOB_STATUS_LABELS`. A missing status is the catalog’s not-calculated phrase, not `0`. The no-argument call stays Russian for tests.

Observed pattern for helpers:

```python
def job_status_line(job: JobRecord, text: Catalog | None = None) -> str:
    catalog = text if text is not None else russian_catalog
```

Existing callers omit `text`. The render path passes `resolve`’s catalog.

Events indicator names are already `EVENT_INDICATOR_LABELS`. Translate that tuple in the English catalog. Do not add Confirm / Reject / Correct time controls. `test_events_unavailable.py` asserts those English control names are absent from the source.

Do not change `tests/unit/test_ui_video_page.py` assertions that look for `Официальные веса`, `200 МБ`, or `по пути` on the Russian constants. If a rendered string moves, the Russian catalog value must still satisfy the assertion at default `ru`.

## Dependencies

Subtask 2.

## Verification

```bash
uv run pytest tests/unit/test_ui_video_page.py tests/unit/test_events_unavailable.py tests/unit/test_ui_empty_states.py -q
uv run ruff check src/excavation_lab/ui/video_page.py src/excavation_lab/ui/events_page.py src/excavation_lab/ui/copy.py
```
