---
name: subagent-dev-review-workflow
description: Orchestrates developer and code-reviewer subagents for Excavation Video Lab implementation. Use when the user requests implementation, a feature, a bugfix, or any development task. Developer implements first, code-reviewer validates, then developer applies fixes if needed.
---

# Subagent Dev–Review Workflow (Excavation Video Lab)

Developer implements, code-reviewer validates, developer applies fixes.

## Steps

### 1. Launch developer

Use the Task tool with `subagent_type: "developer"`:

- **description**: 3–5 words
- **prompt**: Task, concrete paths under `src/excavation_lab/` (including `src/excavation_lab/web/` and `src/excavation_lab/ui/`), `tests/`, or `configs/`, plus constraints (episode splits, PTS timebase, RF-DETR only, ByteTrack, one worker, `uv run`)
- Wait for completion

### 2. Launch code-reviewer

`subagent_type: "code-reviewer"`:

- **description**: short summary
- **prompt**: Review the developer's work. Focus on class mapping, train/test isolation, timestamps, tracking buffers, fake metrics, worker/UI, and tests under `tests/unit` or `tests/integration`
- Wait for completion

### 3. Fix loop

If the reviewer reports **Critical** or **Warnings**:

- Launch `developer` again with the fix list
- Launch `code-reviewer` again
- Repeat once more only if critical or warnings remain

Suggestions are optional. Escalate if the same Critical items survive two cycles.

## Checklist

```
- [ ] Developer
- [ ] Code-reviewer
- [ ] Parse critical / warnings / suggestions
- [ ] If critical or warnings: developer with the fix list
- [ ] Code-reviewer again
- [ ] Clean review → done
```

## Example

**User**: "A clip that starts at 300s is timestamped from 0"

1. **developer**: trace PTS and `source_time_s`; add a fragment-start test
2. **code-reviewer**: confirm preview time is mapped, not substituted for source time
3. **developer** (if needed): apply fixes and `uv run pytest tests -q`

## Repo rules

- Package is `src/excavation_lab/`, not `src/pdm`
- Spec invariants: `EXCAVATION_VIDEO_LAB_SPEC.md` §19
- UI: `src/excavation_lab/web/` and `src/excavation_lab/ui/`
- Commands: `uv run pytest tests -q` and `uv run ruff check src tests`
- Do not run `pilot-small` training unless the user asks

## Agents

- `.cursor/agents/developer.md`
- `.cursor/agents/code-reviewer.md`
