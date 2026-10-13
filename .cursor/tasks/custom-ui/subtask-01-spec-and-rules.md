# Subtask 1: Spec, rules, agents, decisions

## Goal

Make the spec, project rules, agent prompts, and the launch docs describe the custom page server. Docs only. No application code.

## Context

The user ordered the spec and the rules to follow the UI change. That overrides the current line that names Streamlit as the interface. Spec sections that still say Streamlit: the stack sentence (§0, line 14), the stack table (§2, line 53), the core-import sentence (§3, line 90), the repo tree (§4, lines 110–111), the upload paragraph (§6.1, line 175), the playback paragraph (§15 «Анализ», line 514), the worker paragraph (§16, line 528), readiness item 13 (§19, line 645), the CLI block (§23, line 752), and sources S10 and S16 (lines 841 and 847).

Do not rewrite detector, split, tracking, KPI, or M0–M4 model text.

## Acceptance Criteria

- [ ] §0 stack sentence names Jinja2 pages on a localhost Starlette server. The clause «Не начинать с … React-фронтенда» in the next paragraph stays.
- [ ] §2 interface row is Starlette + Jinja2 + our CSS + small vanilla JS. Purpose stays a local Russian browser UI. FastAPI and React stay in the «do not add» dependency list. Starlette is not added to that list.
- [ ] §3 says the compute core does not import Starlette, Jinja2, or Streamlit. UI and CLI still call the same services.
- [ ] §4 tree drops `.streamlit/config.toml` and `app.py` as the UI entry and adds `src/excavation_lab/web/` (templates and static). The rest of the tree stays.
- [ ] §6.1 no longer says Streamlit owns the upload cap. The cap stays 8192 MiB, documented, not raised without a reason. A long file is still a disk path, not a RAM load. S10 in the source table no longer points at `st.file_uploader` as the mechanism.
- [ ] §15 «Анализ» says the working video is an encoded MP4 via HTTP Range, diagnosis is one frame plus a table, and progress is a later request that reads the atomic progress JSON. The `st.fragment` sentence and the live S16 how-to are gone. Page lists for Видео, Данные, Обучение, Качество, and the v0.2 События behavior stay.
- [ ] §16 says a second click or a second request must not start another GPU/MPS job. Statuses, cancel, `partial=true`, and the PID/lock sentence stay.
- [ ] §19 item 13 says a second click or a second request, not a Streamlit rerun.
- [ ] §23 launch block is `uv run excavation doctor` and `uv run excavation ui`.
- [ ] S16 row no longer cites the Streamlit fragment doc. It states that partial UI refresh is a same-origin GET of the job progress JSON. Later source ids (S17–S20) keep their numbers.
- [ ] `.cursor/rules/project-core.mdc` stack line drops Streamlit and adds Starlette, Jinja2, and vanilla JS. FastAPI and React stay under «Not in this project». Layout names `src/excavation_lab/web/` instead of `app.py` as the UI entry. The verify paragraph says `uv run excavation ui`, not `streamlit run`. The principles sentence that the compute core does not import Streamlit stays, and it also names Starlette and Jinja2.
- [ ] `.cursor/rules/media-and-time.mdc` replaces «document Streamlit's upload cap» with the documented 8192 MiB cap. The rest of that rule stays.
- [ ] `.cursor/rules/streamlit-and-worker.mdc` is deleted. `.cursor/rules/web-ui-and-worker.mdc` replaces it: `alwaysApply: false`, globs `src/excavation_lab/web/**`, `src/excavation_lab/ui/**`, `src/excavation_lab/cli.py`, `src/excavation_lab/jobs.py`, `src/excavation_lab/storage.py`. Worker invariants stay (one job, atomic progress JSON, the six statuses, cancel, orphaned, localhost, no telemetry, path and ZIP checks, no `shell=True`). Playback and progress wording match the spec edits above. «События» stays unavailable.
- [ ] `.cursor/agents/planner.md`, `plan-reviewer.md`, `developer.md`, and `code-reviewer.md` call the UI a Starlette page server. They keep «no FastAPI, no React». Developer launch line is `uv run excavation ui`. The planner path row for UI / worker is `src/excavation_lab/web/`, `src/excavation_lab/ui/`, `jobs.py`, `cli.py`. It does not list `app.py`. The developer package list adds `web/`. Reviewer checks that core modules do not import Starlette, Jinja2, or Streamlit, and that a second request does not spawn a job. Plan-reviewer paths drop `app.py` as an allowed UI entry.
- [ ] `.cursor/agents/ui-designer.md` is updated in the same pass. It still says «Streamlit only» and would reject this epic. Point it at `web-ui-and-worker.mdc` and at Jinja templates. The read step is `src/excavation_lab/web/` and `src/excavation_lab/ui/`, not `app.py`. It still forbids React, FastAPI, and a canvas library. Localhost, telemetry off, and the 8192 MiB upload cap replace the `.streamlit/config.toml` bullet.
- [ ] Search `.cursor/rules/` and `.cursor/agents/` for `streamlit`. The only sentences that stay are the ones that mean the compute core does not import Streamlit (`project-core.mdc` principles, `code-reviewer.md` import bullet). Every other hit is the page server, `uv run excavation ui`, or deleted with `streamlit-and-worker.mdc`.
- [ ] `README.md` index line that calls `docs/ENVIRONMENT.md` the Streamlit doc names the page server instead. The install block and the commands block are `uv run excavation ui`, from the criterion above.
- [ ] `.cursor/skills/subagent-planner-dev-review-orchestration/SKILL.md` stops naming Streamlit as the UI. `.cursor/skills/subagent-dev-review-workflow/SKILL.md` does not say Streamlit; its UI line and the developer prompt path drop `app.py` and name `src/excavation_lab/web/` plus `src/excavation_lab/ui/`.
- [ ] `docs/DECISIONS.md` gains one entry, dated 2026-10-11: Streamlit left because the Video page could not be a real layout; replacement is Starlette + Jinja2 + CSS + small JS; FastAPI, React, and a SPA bundler stay forbidden; services and the worker lock stay.
- [ ] `README.md` launch commands (install block and the commands block) are `uv run excavation ui`. Localhost sentence stays.
- [ ] `docs/ENVIRONMENT.md` «Streamlit» section becomes the page server: bind `127.0.0.1`, upload cap 8192 MiB, telemetry off, no auth. It does not tell the reader to edit `.streamlit/config.toml`.
- [ ] `docs/TEST_REPORT.md` is not edited. `.cursor/tasks/i18n/**`, `.cursor/tasks/lab-design/**`, and the original `.cursor/tasks/master-plan.md` are not edited.

## Implementation Notes

Edit the spec in place. Do not translate it and do not renumber sections.

`project-core.mdc` principles line «Compute core does not import Streamlit» becomes Starlette, Jinja2, and Streamlit. The «one GPU/MPS job» and «localhost only» bullets stay.

The new rule file carries the product sentences from the old rule: Russian copy, English identifiers, honest empty states, no demo player, no invented metrics, no loss curve before logs, no ETA before measured steps, zone editing as a vertex table.

The acceptance list is the file list. It includes `.cursor/rules/media-and-time.mdc`, the planner UI/worker path row, plan-reviewer paths, the developer package list including `web/`, the ui-designer read step, both skill files, and the README index line. Leave model frontmatter alone.

DECISIONS entry matches the existing heading style (`## Title`, a date line, short paragraphs).

## Dependencies

None. This subtask is docs only, so it can land before the server exists. README will name `excavation ui` one subtask before the command exists; subtask 3 implements that command.

## Verification

Read the diff. Search the spec, `README.md`, `docs/ENVIRONMENT.md`, `.cursor/rules/`, `.cursor/agents/`, and the two skill files above for `streamlit`. Remaining hits are only the compute-core import ban, plus `docs/TEST_REPORT.md` which may still contain `streamlit 1.65.0`. `streamlit run` and `st.fragment` are gone. Do not edit `.cursor/tasks/i18n/**` or `.cursor/tasks/lab-design/**`.

No pytest required for the prose edit. Do not start the app.
