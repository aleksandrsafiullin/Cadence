# Subtask 9: CLI locale (out of scope)

## Goal

Do not implement CLI translation in this slice. Record why, so a later change does not grow a second string system or a half-translated `--help`.

## Context

Spec §23: the `excavation` CLI is Russian today, returns non-zero on error, and has a JSON result mode. `src/excavation_lab/cli.py` has Typer help and echo strings in Russian. More Russian user text lives in `hardware.russian_summary`, `hardware.weights_download_notice`, `analytics.unavailable_error`, and `LabError` messages under `datasets/`, `models/`, `media/`, `jobs.py`, and `storage.py`. The Data page also shows three service formatters that stay Russian in this slice: `format_sample_summary` (`src/excavation_lab/datasets/sampler.py`), `format_import_summary` (`src/excavation_lab/datasets/coco_import.py`), and `format_export_summary` (`src/excavation_lab/datasets/cvat_export.py`). They are operator text, same as dataset `LabError`s. Subtask 4 leaves their call sites in place. Subtask 8 does not treat that Russian success text as a failed switch.

The UI catalogs in `copy.py` are importable without Streamlit, so a future CLI can call `catalog("ru")` or `catalog("en")` for strings that already live there. They do not cover Typer help or those service sentences. Translating only the Typer lines would leave command output in Russian. That is a half-translated CLI. This slice prefers a complete UI.

Default CLI locale stays Russian. Existing CLI tests, including `unavailable_error().message` and the doctor summary, keep passing because this subtask changes nothing.

## Acceptance Criteria

- [ ] No `--lang` flag.
- [ ] No edits to `src/excavation_lab/cli.py` for translation.
- [ ] No edits to `hardware.russian_summary` or `weights_download_notice`.
- [ ] No edits to `format_sample_summary`, `format_import_summary`, or `format_export_summary`.
- [ ] No new string catalog beside `copy.py`.
- [ ] `cli.py` does not import `excavation_lab.ui.locale` (that module imports Streamlit).

## Implementation Notes

Skip this file during implementation. The UI may format the weights warning and the Events caption from `copy.py`. Those are page strings. The functions the CLI prints stay Russian.

A follow-up slice, if it happens, should: default `--lang` to `ru`; accept only `ru` and `en`; resolve without reading the Streamlit session or `ui/locale.txt` (a worker and a terminal are not the browser); reuse `catalog()`; translate service messages and Typer help in one pass rather than help alone. JSON `error.code` stays English. JSON `error.message` may translate only when that follow-up says so, and tests that lock the Russian message must be updated in that slice, not this one.

## Dependencies

None. Not part of the execution order.

## Verification

This subtask is done by not doing it. The UI slice’s check is still:

```bash
uv run pytest tests -q
uv run ruff check src tests
```

No CLI locale test is added here.
