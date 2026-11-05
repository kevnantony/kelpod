# Runbook: developing on Apple Silicon (M-series Mac)

The dev machine is an Apple M4 Pro with 24 GB unified memory and no NVIDIA GPU. Locally, model
work runs on **MLX** (`mlx-lm`); see [ADR-0001](../adr/0001-apple-silicon-mlx-local-profile.md).
Containers and CI are Linux/CPU and use the `fake` profile.

## Setup

```bash
uv sync                       # Python 3.12 env from uv.lock (mlx/mlx-lm only on darwin-arm64)
cd web && npm ci && cd ..     # UI
uv run pre-commit install     # hooks: ruff, ruff-format, detect-secrets, large/data-file guard
make lint test                # fake profile, no models needed
```

Do not `pip install` into `.venv`; try extra packages with `uv run --with <pkg> ...`.

## Hardware profiles

| Profile | Backend | Models | Use |
|---|---|---|---|
| `fake` | fake | none | unit tests, CI, containers, egress test |
| `toy` | mlx | Qwen2.5-0.5B-Instruct-4bit (student and teacher), nli-deberta-v3-xsmall, all-MiniLM-L6-v2 | smoke runs, minutes |
| `mac` | mlx | Qwen2.5-1.5B-Instruct-4bit student, Qwen2.5-7B-Instruct-4bit teacher, nli-deberta-v3-base, bge-m3 | demo-quality runs, hours |

Files: `configs/hardware/<name>.yaml`. Several agents/people share the Mac: use `toy` for
anything automated and never load the 7B teacher in tests.

## Download models (the only online step)

```bash
ZLLF_ALLOW_ONLINE=1 uv run --with huggingface_hub python scripts/download_models.py --hardware toy
ZLLF_ALLOW_ONLINE=1 uv run --with huggingface_hub python scripts/download_models.py --hardware toy mac
uv run python scripts/download_models.py --verify            # offline re-hash
```

- Models go into the normal Hugging Face cache (`~/.cache/huggingface/hub`, or `HF_HUB_CACHE`),
  where `zllf` finds them with `HF_HUB_OFFLINE=1`.
- `models/manifest.json` (gitignored) records repo, **pinned commit**, licence (from the model
  card), per-file sha256 and total size. Re-running reuses the pinned commit;
  `--update` moves to the current `main`.
- ONNX/TF/Flax/GGUF files are skipped; `*.bin` is skipped when safetensors exist.
- Sizes: toy ≈ 0.65 GB. mac ≈ 7–8 GB (the 7B 4-bit teacher ≈ 4.3 GB, bge-m3 ≈ 2.3 GB).
  On a slow line, fetch `toy` first.
- spaCy and Docling artefacts are not Hugging Face models: `ZLLF_MODELS_DIR` and
  `ZLLF_DOCLING_ARTIFACTS_PATH` (see `src/zllf/stages/sanitize/nlp.py`,
  `src/zllf/stages/ingest/docling_adapter.py`).

## Run

```bash
make toy-run                                     # zllf run ... --hardware toy
uv run zllf run --corpus tests/fixtures/corpus --hardware fake --auto-approve --runs-dir runs
uv run pytest -m mlx                             # MLX tests (needs the toy models)
uv run zllf serve-api --with-worker              # API + UI on 127.0.0.1:8000
```

Demo corpus (public NRC documents):
`ZLLF_ALLOW_ONLINE=1 uv run python scripts/fetch_demo_corpus.py` → `bench/corpora/data/nrc-fr/`
(see `bench/corpora/README.md`).

## MLX notes

- MLX uses unified memory; the GPU can use roughly 70–75 % of RAM by default (≈ 17 GB on 24 GB).
  The 7B 4-bit teacher (~4.3 GB weights + KV cache) and a 1.5B student fit, but not with a
  browser full of tabs and Docker Desktop holding 8 GB: lower Docker's memory limit while running
  the `mac` profile.
- `mlx-lm` LoRA trains 4-bit models directly (QLoRA-style); `lora.num_layers` in the profile
  controls how many layers get adapters.
- MLX wheels exist only for macOS arm64; `pyproject.toml` marks `mlx`/`mlx-lm` darwin-arm64
  only, so Linux CI and containers skip them.
- First call compiles Metal kernels; time the second run.
- `HF_HUB_OFFLINE=1` is set on `import zllf`. If loading fails with "offline mode", the model
  is not in the cache: run `download_models.py`, never set `ZLLF_ALLOW_ONLINE` for the app.

## Docker on the Mac

```bash
docker build -f deploy/docker/Dockerfile.app -t zllf-app:dev .          # ~3.7 GB, CPU torch
docker build --build-arg WEB_STAGE=web-prebuilt -f deploy/docker/Dockerfile.app -t zllf-app:dev .
docker compose -f deploy/compose/compose.yaml -f deploy/compose/compose.dev.yaml up --build
deploy/egress/run_egress_test.sh --mode docker
```

Containers are Linux arm64: no MLX, no Metal. Use `ZLLF_HARDWARE=fake` there (the compose
default). `WEB_STAGE=web-prebuilt` copies an existing `web/dist` instead of running `npm ci` in
the build. The first build downloads ~1 GB of wheels.

### CPU-only torch

`uv.lock` resolves `torch` from PyPI; on Linux those wheels depend on ~3 GB of `nvidia-*`
CUDA packages. `deploy/scripts/install_cpu_env.sh` (used by the Dockerfile and CI) installs
every locked version but takes `torch`/`torchvision` as `+cpu` builds
(`uv pip install --torch-backend cpu`) and drops the CUDA-only packages. The clean long-term fix
is a `[tool.uv.sources]` entry in `pyproject.toml` routing torch to the PyTorch CPU index on
Linux (owned by the platform track).
