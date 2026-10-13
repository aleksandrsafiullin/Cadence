# Subtask 1: Locale catalogs

## Goal

Turn `src/excavation_lab/ui/copy.py` into two streamlit-free catalogs, `ru` and `en`, with the same keys. Existing module-level Russian names stay the Russian catalog so current tests and `models/quality.py` keep passing.

## Context

Spec §15 says the interface is Russian. This slice keeps that as the default and adds English. `copy.py` is the current catalog. Its module docstring already says compute may import it and it must not import Streamlit. `src/excavation_lab/models/quality.py` imports it and bakes «не рассчитано» plus Russian sentences into reports and `LabError` messages. A worker must keep doing that in Russian even after the UI can switch.

`tests/unit/test_ui_theme.py` locks `PAGE_*` to the six Russian titles and locks `JOB_STATUS_LABELS` keys. `tests/unit/test_quality_queue.py` locks `NOT_CALCULATED == "не рассчитано"`. `tests/unit/test_ui_analysis_state.py` locks `PREDICTED_HEADING == "predicted"`. Do not reword the Russian strings.

## Acceptance Criteria

- [ ] `copy.py` has no Streamlit import and does not read `st.session_state`, the filesystem, or a locale file.
- [ ] `catalog("ru")` and the module-level names (`NOT_CALCULATED`, `PAGE_VIDEO`, `JOB_STATUS_LABELS`, `not_train_frame_message`, and the rest) are the current Russian text.
- [ ] Both catalogs implement `lead(page_id)` for `video`, `data`, `training`, `analysis`, `quality`, `events`. `PAGE_DESCRIPTIONS` stays a module-level alias keyed by the Russian titles, built from `catalog("ru").lead(...)`, for existing tests. Screens use `lead(page_id)` on the active catalog, not that dict.
- [ ] `catalog("en")` has the same fields. English is a full translation of every Russian sentence that contains Cyrillic. It is not a copy of the Russian value.
- [ ] `catalog(None)`, `catalog("fr")`, `catalog("")`, and `catalog("EN")` return the Russian catalog and do not raise.
- [ ] Invariant fields are identical in both catalogs: `Excavation Video Lab`, `Light`, `Dark`, `PROJECT`, `WORKFLOW`, `APPEARANCE`, job-status keys, class ids, `not_evaluated`, `not evaluated`, `predicted`.
- [ ] English page titles are `Video`, `Data and labeling`, `Training`, `Analysis`, `Quality`, `Events`. English missing-measurement text is exactly `not calculated`. English class labels are `excavator body`, `bucket`, `haul truck`, `truck bed`.
- [ ] Switcher literals `Русский` and `English` exist once, shared by both locales, not as two different translations.
- [ ] Locale caption and label exist in both catalogs. Russian label `Язык`. English label `Language`. Russian caption `Меняет только язык интерфейса, не данные, модель и измерения.` English caption `Changes only the interface language, not the data, the model, or the measurements.`
- [ ] `models/quality.py` still imports only `excavation_lab.ui.copy`, calls Russian text (`catalog("ru")` or the aliases), and its docstring says a worker does not follow the browser locale. It does not import `locale`.
- [ ] `docs/DECISIONS.md` has a short dated note: Russian is the default UI locale; English is selectable; the spec’s “interface in Russian” remains the default.
- [ ] Existing pytest stays green. No page module is switched to English in this subtask.

## Implementation Notes

Keep this as one subtask. The key set is already the constants in `copy.py`. The work is the dataclass, the Russian aliases, `lead(page_id)`, the English sentences, and the import-time check, all in that file, with no page wiring. Splitting the English fill would leave the Cyrillic check red and would split the key set. That is past two hours only if pages are edited here. They are not.

Keep one module, `src/excavation_lab/ui/copy.py`. A frozen dataclass (or a Mapping built so a missing English field fails at import) is enough. Not a package, not gettext, not JSON files.

Suggested shape:

```python
def catalog(locale: str | None = None) -> Catalog: ...
```

Module-level functions keep their current signatures and delegate to the Russian catalog. Example: `not_train_frame_message(frame_id, split)` stays Russian. The English catalog has its own method so word order can change. Do not share one format string and only swap «не рассчитано».

`lead(page_id)` is the API both catalogs share. Page ids are `video`, `data`, `training`, `analysis`, `quality`, `events`. An unknown id fails visibly. Module-level `PAGE_DESCRIPTIONS` remains the test-facing alias:

```python
PAGE_DESCRIPTIONS = {
    PAGE_VIDEO: RU.lead("video"),
    # same for the other five Russian titles
}
```

Tests that read `ui_copy.PAGE_DESCRIPTIONS[ui_copy.PAGE_VIDEO]` stay valid. `shell.render_frame` does not look leads up in that dict. Subtask 2 calls `active.lead(page_id)`.

Job-status **labels** translate (`Queued`, `Running`, `Completed`, `Failed`, `Cancel requested`, `Cancelled`). The dict keys do not.

Import-time check: every Russian value that contains Cyrillic has a different English value, and that English value contains no Cyrillic. Invariant Latin strings are equal. The shared option literal `Русский` is not an English catalog sentence.

`quality.py`: one explicit Russian binding is enough, for example `NOT_CALCULATED = ui_copy.catalog("ru").not_calculated`, which must still equal `"не рассчитано"`. Leave the report dict and `LabError` messages in Russian. Do not thread a locale argument through `load_quality_report`.

Do not edit page modules, `theme.py`, `cli.py`, or `charts.py` here.

## Dependencies

None.

## Verification

```bash
uv run pytest tests -q
uv run ruff check src tests
```

Confirm `uv run python -c "from excavation_lab.ui.copy import catalog; assert catalog('en').not_calculated == 'not calculated'; assert catalog('ru').not_calculated == 'не рассчитано'"`.

Importing `excavation_lab.models.quality` must not import Streamlit. Do not import `excavation_lab.ui.locale` from `models/`. The AST checks that prove `copy.py` is Streamlit-free and that `quality.py` does not import the resolver are in subtask 7. `tests/unit/test_no_streamlit_import.py` allows every file under `ui/` and does not prove either.
