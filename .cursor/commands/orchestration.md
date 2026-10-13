# Orchestration

Execute planner → plan review → dev → review for a **complex task** in Excavation Video Lab.

**Task**: Use the text after `/orchestration` as the task. If none was provided, ask what to orchestrate.

## Workflow

Read `.cursor/skills/subagent-planner-dev-review-orchestration/SKILL.md` and apply it.

Sequence: planner → plan-reviewer → planner (revise if needed) → for each subtask (developer → code-reviewer → fix loop) → done

## Repo constraints

- **Root**: `src/excavation_lab/`
- **Spec**: `EXCAVATION_VIDEO_LAB_SPEC.md` milestones M0–M4
- **Stack**: Python 3.11 · uv · RF-DETR Nano/Small · ByteTrack · PyAV · Streamlit · pytest · Ruff
- Decompose across media/time, episode splits and COCO, train/eval, tracking/export, UI/worker, tests
- v0.2 events stay a separate phase
- Splits, timebase, and association each need a named test from spec §19
- Do not plan Ultralytics, FastAPI/React, hosted inference, or a substitute MP4 download

## Verification baseline

```bash
uv run pytest tests -q
uv run ruff check src tests
```
