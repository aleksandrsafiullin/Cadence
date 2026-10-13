# Test report

Date: 2026-10-10. Close-out of the synthetic workflow. A green suite is not acceptance of a trained model.

## Commands run

```bash
uv run pytest tests -q
```

Pytest exited 0. Summary line: `516 passed, 3 deselected, 3 warnings in 55.57s`. The three deselected tests are `manual`. Two warnings are Click deprecation notices imported by Typer. The third is a `torch.jit.script` future warning. This run did not download weights and did not train. A sandboxed run of the same command failed the job-lock tests with `PermissionError: ps` and signals that did not arrive. The result above is the rerun outside the sandbox. The job lock was not changed.

```bash
uv run ruff check src tests
```

Ruff exited 0. Output: `All checks passed!`

```bash
uv run excavation events propose --project pilot --run none --json
```

The process exited 1. JSON: `{"ok": false, "error": {"code": "unavailable", "message": "События и показатели в версии 0.1 не рассчитываются."}}`. `error.code` is `unavailable`. No `events.csv` was written.

## Versions

Not reprinted in this close-out. The block below is the earlier print from 2026-10-10.

`uv --version`:

```text
uv 0.12.13 (0ebbd9274 2026-09-10 aarch64-apple-darwin)
```

`uv run python -c "import sys; print(sys.version)"`:

```text
3.11.16 (main, Sep  1 2026, 14:08:22) [Clang 22.1.3 ]
```

`uv run python -c` with `importlib.metadata.version` for the named distributions:

```text
rfdetr 1.11.2
torch 2.14.1
torchvision 0.29.1
trackers 2.6.1
streamlit 1.65.0
supervision 0.30.9
```

## Data used

No pilot MP4. No pit labels. No weight download. The workflow test writes a 64×48 synthetic clip of about 25 s at 2 fps under pytest `tmp_path`, registers the probed duration, and imports one box it wrote. That clip is not accuracy evidence.

## Artifacts

Pytest removes the temporary clip, parquet, preview, and COCO zip. Nothing from that run is kept in the repo. No weights and no dataset snapshot were produced. `events.csv` and `truck_visits.csv` were not written.

## Measured metrics

| Metric | Value |
|---|---|
| AP | не рассчитано |
| precision | не рассчитано |
| recall | не рассчитано |
| per-class AP | не рассчитано |
| per-class precision | не рассчитано |
| per-class recall | не рассчитано |
| bucket AP | не рассчитано |
| clear-object recall | не рассчитано |
| partial-object recall | не рассчитано |
| event precision | не рассчитано |
| event recall | не рассчитано |

## Unchecked items

- no pilot MP4
- no labels
- no GPU training run
- no `pilot-small`
- no smoke
- `excavation doctor` was not run for this report
- `doctor --load-weights` was not run
- Streamlit was started only to confirm the log binds to `127.0.0.1`, then stopped. That is not a quality check
