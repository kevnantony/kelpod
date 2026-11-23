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
- Ask before pushing to `main` or publishing.
- Keep the `## Status` block in `README.md` (between the `status:start` / `status:end` markers)
  current: **Done**, **Doing now**, **Next**, with a "Last updated" timestamp (date, time, CET).
  Update it whenever a step lands, a new step starts, or the plan changes, and at least every few
  hours while working (also when pausing for a usage limit: say so and when work resumes). Commit and
  push it together with the work. Short plain bullets; measured numbers only, marked synthetic or real.
