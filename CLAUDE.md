# Kelpod: instructions for coding agents

Read `KELPOD_CONTEXT.md` first. It describes the product (old German floor plans to closed IFC models
for HottCAD), the pipeline, the strategy, the pilot and the course requirements.

- Only `presentation/` and `KELPOD_CONTEXT.md` belong to the current product. The rest of the repo is
  left over from an abandoned idea: do not build on it.
- Python via `uv`, LaTeX via Tectonic. The dev Mac has no NVIDIA GPU.
- Licences: this public repo is the open-source, educational version. Any open-source code, model
  or dataset may be used (GPL/AGPL code and non-commercial datasets such as CubiCasa5k included), as
  long as you follow its terms and add it to `CREDITS.md` with its licence and link. The paid version
  is built separately, so do not reject a tool here only because it is not commercially usable.
- Keep the pilot customer's name out of this public repo.
- Never present targets as results; mark unbuilt ideas as ideas.
- `main` is the current baseline. Merge into `main` only by promotion under `docs/EVAL_PROTOCOL.md`
  (`scripts/promotion_check.sh` must pass), plus docs-only status commits. Experiments live on
  `exp/<name>` branches in their own worktrees. Ask before publishing anything outside this repo.
- Never weaken, bypass or game the evaluation (`docs/EVAL_PROTOCOL.md` §6). Report flaws, fix them on
  an `eval/*` branch.
- Cluster resources: every job must be explainable. The whole pascal cluster (GPUs included) may be
  used when that gets results faster, as long as `docs/EXPERIMENTS.md` says why. For every job,
  CPU or GPU: size the request from measurements, log the justification, the expected and the
  measured CPU/GPU hours, and checkpoint long runs. Never hold a job that waits or polls for input
  (submit it when the input exists), cancel jobs idle for more than 30 minutes, and keep the agent
  job itself small (heavy evaluation and training go to their own jobs).
- Keep the `## Status` block in `README.md` (between the `status:start` / `status:end` markers)
  current: **Done**, **Doing now**, **Next**, with a "Last updated" timestamp (date, time, CET).
  Update it about once an hour while working (also when pausing for a usage limit: say so and when
  work resumes). Commit and push it together with the work. Short plain bullets; measured numbers
  only, marked synthetic or real.
- Commit and push about once an hour, not after every small change.
