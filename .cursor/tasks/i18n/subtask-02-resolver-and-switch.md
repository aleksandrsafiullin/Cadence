# Subtask 2: Resolver and switch

## Goal

Resolve `ru` or `en` from session state and `<workspace>/ui/locale.txt`, and put a language switch in the APPEARANCE group next to Light | Dark. Sidebar chrome uses the active catalog. Default remains Russian, and reading does not create the file.

## Context

Spec §22: localhost only, paths stay inside the workspace, no symlink escape. Theme persistence in `src/excavation_lab/ui/theme.py` is the pattern: read does not create, invalid text does not raise, write is `os.replace` from `on_change` only. `tests/unit/test_theme_persistence.py` uses `segmented_control[0]` for Light | Dark. That index must stay the theme control.

`app.py` builds navigation before the page runs, so page titles have to be resolved there, not only inside `render_frame`. `shell.render_frame` looks up `PAGE_DESCRIPTIONS` by the Russian title string. English titles need a page id.

## Acceptance Criteria

- [ ] New module `src/excavation_lab/ui/locale.py` imports Streamlit. `copy.py` does not import it.
- [ ] Preference path is `<workspace>/ui/locale.txt`. Contents are exactly `ru` or `en` plus a newline. Read does not create `ui/` or the file.
- [ ] Missing file and empty session resolve to `ru` and do not write `lab_locale` for a missing file.
- [ ] A valid file is copied into session `lab_locale` on a fresh session. Session `ru` or `en` wins over the file once set, same order as `lab_theme`.
- [ ] `stopping`, empty, `fr`, `EN`, and any other value resolve to `ru`, do not raise, and are not stored as `lab_locale`. The invalid file is left as it is.
- [ ] Write runs only from the switch `on_change`. Temp file is created in `ui/`, then `os.replace` onto `locale.txt`. After resolve, the file must stay inside the workspace. A symlink that leaves the workspace is refused. A failed write keeps the previous valid locale (else the file, else `ru`) and shows the catalog’s write-failed sentence. It does not raise.
- [ ] Switch sits in the existing APPEARANCE group, immediately after the Light | Dark control. Theme remains `segmented_control[0]`. Options are the literals `Русский` and `English`. Label is `Язык` or `Language` for the active locale, `label_visibility="collapsed"`. One caption under it is the locale sentence from the active catalog. `icon=""` if an error is shown. No emoji.
- [ ] Product title, kickers, and `Light` | `Dark` stay the invariant strings. Theme caption uses the active catalog (`Меняет только оформление.` in Russian).
- [ ] `render_frame` still accepts `ui_copy.PAGE_VIDEO` and the other Russian titles. It maps those, the English titles, and the page ids onto one id, then renders the **active** catalog’s title and `lead(page_id)`. It does not index `PAGE_DESCRIPTIONS`. Unknown text still fails visibly (do not invent a lead).
- [ ] The empty project selectbox stores the stable id `no-project`, not `ui_copy.NO_PROJECT`. `format_func` shows the active catalog’s no-project label. At `ru` that label is still `нет проекта`. If `project_picker` is not in the current options, set it to `options[0]` before the widget is built (the same reset `train_dataset` already uses). A language switch on an empty workspace does not raise. Real project names stay the option values when projects exist.
- [ ] `app.py` and `navigation.py` use the same resolver, so sidebar links and `st.Page` titles match. `st.set_page_config(page_title=...)` stays `Excavation Video Lab`.
- [ ] Tests that render the shell at the default locale still see Russian titles, the Russian theme caption, and Light | Dark in slot 0. No test writes the repo `workspace/ui/locale.txt`.
- [ ] Page bodies may still call `ui_copy` for buttons and captions. Subtasks 3–6 bind each page to the active catalog. Chrome (title, `lead`, project label, theme caption, locale caption) is already on the active catalog in this subtask.

## Implementation Notes

Mirror `theme.py`: `preference_path`, `read_locale_file`, `resolve_locale`, `write_locale_file`, `commit_locale_choice`, `render_locale_switch`. Reuse `PathNotAllowedError` and the same `_require_inside` idea. Strip the file text. Only `ru` and `en` are valid after strip. `RU` is invalid.

Widget key `lab_locale_choice`, distinct from `lab_appearance`. Map `Русский` → `ru`, `English` → `en`. Do not store the endonym in `lab_locale`.

`shell.py` order stays: PROJECT, project picker, WORKFLOW, page links, APPEARANCE, theme switch, locale switch. Pass the workspace into `build_navigation`. Resolve locale before `st.navigation` is built so a rerun after `on_change` picks up the new titles.

`render_frame(ui_copy.PAGE_VIDEO)` at locale `ru` must still show `Видео` and `catalog("ru").lead("video")`, which equals `PAGE_DESCRIPTIONS[PAGE_VIDEO]`. `tests/unit/test_theme_persistence.py` calls `render_frame` directly. `tests/unit/test_ui_empty_states.py` expects that Russian lead and `PRODUCT_TITLE` in the rendered text.

`tests/unit/test_ui_empty_states.py` currently asserts `app.selectbox[0].options == [ui_copy.NO_PROJECT]`. After the stored id is `no-project`, change that assertion so the visible label is still `ui_copy.NO_PROJECT`. If AppTest exposes raw option ids, assert `options == ["no-project"]` and assert the Russian format result equals `ui_copy.NO_PROJECT`. Leave `_assert_honest` and the other empty-state checks in place.

Do not clear other session keys in `on_change`. The sentinel plus the options reset is what keeps `project_picker` from holding `нет проекта` across a switch. Subtask 5 stores `analysis_mode` and `analysis_time` as stable values for the same reason.

`models/` and `cli.py` do not import `locale.py`.

## Dependencies

Subtask 1.

## Verification

```bash
uv run pytest tests/unit/test_theme_persistence.py tests/unit/test_ui_theme.py tests/unit/test_ui_empty_states.py tests/unit/test_no_streamlit_import.py -q
uv run ruff check src tests
```

Full `uv run pytest tests -q` if those pass. Manual check can wait for subtask 8. Do not point a test at the repo `workspace/` directory.
