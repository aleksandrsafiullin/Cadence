# Implementation

Execute the dev–review workflow for an **easy task or fix** in Excavation Video Lab.

**Task**: Use the text after `/implementation` as the task. If none was provided, ask what to implement.

## Workflow

Read `.cursor/skills/subagent-dev-review-workflow/SKILL.md` and apply it.

Sequence: developer → code-reviewer → developer (fix if needed) → code-reviewer

## Repo constraints

- **Product**: local RF-DETR fine-tune, ByteTrack, bucket/bed association, Russian Streamlit UI
- **Spec**: `EXCAVATION_VIDEO_LAB_SPEC.md`
- **Root**: `src/excavation_lab/`, `app.py`, `configs/`, `tests/`
- **Stack**: Python 3.11 · uv · PyTorch · `rfdetr==1.11.2` · `trackers` · supervision · PyAV · Streamlit · pytest · Ruff
- **Verify**: `uv run pytest tests -q` and `uv run ruff check src tests`
- Do not plan or import `src/pdm`

## Key references

- `EXCAVATION_VIDEO_LAB_SPEC.md`
- `configs/pilot.yaml`, `configs/labels.yaml` once they exist
- `.cursor/rules/project-core.mdc`
