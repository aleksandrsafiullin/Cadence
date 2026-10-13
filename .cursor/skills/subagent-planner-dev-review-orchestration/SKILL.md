---
name: subagent-planner-dev-review-orchestration
description: Orchestrates planner, plan-reviewer, developer, and code-reviewer for large Excavation Video Lab tasks. Use for multi-step epics across media, COCO datasets, RF-DETR, tracking, the Starlette page server, or the worker. Sequence: planner → plan-reviewer → per subtask developer → code-reviewer.
---

# Subagent Planner–Plan Review–Dev–Review (Excavation Video Lab)

Planner decomposes the task, plan-reviewer validates, then each subtask is implemented and reviewed.

```
Planner → Plan Reviewer → (revise if needed) → [Per subtask: Developer → Code-reviewer → fix loop] → Done
```

## Phase 1: Planner

`subagent_type: "planner"`

- **prompt**: Full scope, paths under `src/excavation_lab/`, and spec constraints (M0–M4, no test leakage, PTS, ByteTrack, v0.2 gated). Write `master-plan.md` and subtask files in `.cursor/tasks/`
- Wait

## Phase 2: Plan review

`subagent_type: "plan-reviewer"`

- **prompt**: Review `.cursor/tasks/` against the spec. Return Critical, Warnings, Suggestions, and Verdict (`APPROVED` or `REVISE`)

If `REVISE` or any Critical/Warning remains, re-run planner with the feedback, then plan-reviewer again.

**Stop:** verdict `APPROVED` or only Suggestions; the same Critical items after 2 cycles; or more than 10 review cycles → ask the user.

## Phase 3: Subtasks

List `.cursor/tasks/subtask-*.md`, sort by number, read each file before handing it to the developer.

## Phase 4: Per subtask

1. **developer** — full subtask: goal, acceptance criteria, notes, verification
2. **code-reviewer** — acceptance criteria plus leakage, timebase, association, worker/UI, tests
3. Fix loop until no Critical or Warnings remain

Escalate if the same Critical items survive two fix cycles, or one subtask exceeds 10 fix cycles.

## Phase 5: Completion

Summarize what shipped, which subtasks finished, and leftover suggestions. Do not treat smoke metrics as model quality.

## Checklist

```
- [ ] Planner
- [ ] Plan-reviewer
- [ ] Revise until APPROVED
- [ ] Sort subtasks
- [ ] For each: developer → reviewer → fix loop
- [ ] Summarize
```

## Example

**User**: "Freeze episode splits before CVAT export"

1. **planner**: manifest schema, guard interval, export lock, unit test, UI button disabled before freeze
2. **plan-reviewer**: flags a missing "sampler must not pull test" test; planner adds it
3. **developer** / **code-reviewer** per subtask

## Agents

- `.cursor/agents/planner.md`
- `.cursor/agents/plan-reviewer.md`
- `.cursor/agents/developer.md`
- `.cursor/agents/code-reviewer.md`

## Notes

- Commands use `uv run`
- Dataset subtasks: split and COCO round-trip tests
- Model subtasks: `run_test=False` and bundle reload
- Media subtasks: PTS fragment start and preview duration
- UI/worker subtasks: single-job lock and honest empty states
- Follow planner dependency order after approval
