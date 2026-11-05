# kelpod / ZLLF — instructions for coding agents

Read `docs/PRD.md` (requirements, IDs FR-xx/NFR-xx) and `docs/DEVELOPMENT_PLAN.md` (step IDs P1.x.y) first.

## Environment
- Python 3.12 via **uv**: `uv run pytest`, `uv run ruff check`, `uv run mypy src/zllf/core`.
- Do **not** edit `pyproject.toml` or `uv.lock`, and do not `uv add`/`pip install` into `.venv`.
  If you need a package that is not installed, try it with `uv run --with <pkg> ...` and list it
  under "dependencies needed" in your report.
- Dev machine: Apple M4 Pro, 24 GB unified memory, **no NVIDIA GPU**. MLX replaces CUDA/vLLM
  locally (`models.backend: mlx`). Several agents share this machine: in tests use the fake
  backend; real models only via the `toy` profile (0.5B) and never the 7B teacher.

## Hard rules
1. **No network calls** at runtime. Models load from the local Hugging Face cache only
   (`HF_HUB_OFFLINE=1` is set on `import zllf`). Never download in library code.
2. **Never log or persist raw document text or redacted values** in logs, events, reports
   (NFR-SEC-06). Log counts, IDs and hashes.
3. Stages are **deterministic** given config + seed: use `ctx.rng(salt)` / `ctx.np_rng(salt)`.
4. Stay inside the directories you own; shared contracts live in `src/zllf/core`,
   `src/zllf/stages/base.py`, `src/zllf/llm/base.py`. Do not change them — if a contract is
   insufficient, work around it and describe the needed change in your report.
5. Every module gets tests under `tests/unit/<area>/` that pass with the **fake** profile
   (`load_config(hardware="fake")`). MLX tests are marked `@pytest.mark.mlx`.

## Contracts (summary)
- Config: `zllf.core.config.load_config(profile="mode_a", hardware="fake"|"toy"|"mac")` → `RunConfig`.
- Run dir: `zllf.core.manifest.RunPaths`; stage output dirs `01_ingest … 06_package`
  (`paths.stage(name)`), persistent scratch `paths.work(name)`.
- Stage: subclass `zllf.stages.base.Stage`, implement `run(ctx, out_dir) -> StageResult`, class
  lives at `zllf.stages.<name>.stage:<Name>Stage`. Write outputs only into `out_dir`.
- Artefact rows: `zllf.core.schemas` (Chunk, Redaction, SynthExample, EvalItem, GateResult, …)
  with `to_parquet/read_parquet/write_jsonl/read_jsonl`.
- Models: `ctx.llm("teacher"|"student"|"judge")`, `ctx.nli()`, `ctx.embedder()` →
  interfaces in `zllf.llm.base`. Fakes in `zllf.llm.fake`.
- Runner: `zllf.pipeline.init_run / run_pipeline / approve`.

### Files each stage writes (in `out_dir`)
| Stage | Files |
|---|---|
| ingest | `documents.parquet` (DocumentRecord), `pages.parquet` (Block), `chunks.parquet` (Chunk, duplicates removed), `ingest_report.json` |
| sanitize | `chunks_clean.parquet` (Chunk), `redactions.parquet` (Redaction), `pii_fingerprints.json` (salted hashes only), `policy.yaml`, `sanitize_report.json`; returns `needs_approval=True` |
| synth | `train.jsonl`, `val.jsonl`, `test.jsonl` (SynthExample, incl. unanswerables), `canaries.json` (list[Canary]), `taxonomy.json`, `synth_report.json` |
| train | `hw_profile.json` (HardwareProfile), `trials.json` (list[TrialResult]), `adapter/` (final LoRA adapter, loadable by the backend), `train_report.json` |
| evaluate | `predictions.jsonl` (EvalPrediction), `results.json`, `gates.json` (list[GateResult]), `eval_report.json` |
| package | `adapter/`, `report.json`, `report.pdf`, `sbom.cdx.json`, `SHA256SUMS`, `SHA256SUMS.sig` |

## Style
ruff (line length 100), type hints everywhere, small pure functions, docstrings on public API.
Match the surrounding code; no new frameworks without a reason in the report.
