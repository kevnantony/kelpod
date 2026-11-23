# Hand-over: MVP milestone 1 (plan to closed IFC on 3 sample plans)

Read `KELPOD_CONTEXT.md`, `CLAUDE.md` and `docs/ARCHITECTURE.md` first. Work on branch
`claude/kelpod-mvp-fa1abc`; commit and push that branch after every finished step; never push `main`.

## Definition of done

The pipeline runs end to end on at least 3 synthetic sample plans, writes IFC files that IfcOpenShell
re-opens, the closure metric and runtime are reported honestly, and `uv run pytest` passes.

## Status (keep this up to date; newest first)

- [x] Old LoRA-factory code deleted (it stays in git history).
- [x] Package skeleton, contracts (`src/kelpod/contracts.py`), pipeline, CLI.
- [ ] Research first (rule: check what exists before building anything):
  docs/research/OPEN_SOURCE_SURVEY.md.
- [ ] Modules, one subagent each (briefs in docs/AGENT_BRIEFS.md): clean, segment, ocr, geometry,
  export, evaluation. A first attempt was stopped part-way on 2026-11-23 when work moved to the
  server, so module folders may hold partial, untested WIP: review it, keep what is good.
- [ ] Integration: `kelpod convert` on 3 synthetic plans, `scripts/benchmark.py` report.
- [ ] Minimal FastAPI endpoint (`kelpod.api`), if time allows.
- [ ] Final report in `docs/MVP_REPORT.md`: what works, what doesn't, what needs real plans.

## Compute

The MVP runs on CPU. If a step needs a GPU (e.g. fine-tuning or running a learned segmenter, a
heavier OCR model), use the cluster's GPUs through Slurm from the login host; never run GPU jobs on
the login node itself. Partitions seen on 2026-11-23: `pascal` (GTX 1080 Ti), `volta` (V100),
`ampere` (RTX A6000, A100). Example: `srun -p ampere --gres=gpu:1 --time=02:00:00 uv run python …`
or an `sbatch` script under `scripts/slurm/`. Keep jobs small and release GPUs when done.
