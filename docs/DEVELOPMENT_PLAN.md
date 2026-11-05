# Technical Development Plan — Zero-Leak LoRA Factory (ZLLF)

| | |
|---|---|
| **Repo** | `github.com/kevnantony/kelpod` |
| **Companion** | [`PRD.md`](PRD.md) — requirement IDs (FR-xx, NFR-xx) referenced below |
| **Status** | Draft v0.1 — 2026-11-05 |
| **Team** | Kevin Antony, Arsha Fahima Tajdeen (+ coding agents) |

> **How to use this plan.** Work is organised as **Phase → Step → Sub-step** with IDs like `P1.3.2`. Each step lists the PRD requirements it satisfies, the files it touches and a **Done when** checklist. Turn each step into a GitHub issue (title = ID + name) and each phase into a milestone. Week numbers are *course weeks*; map them to dates once the semester calendar is fixed.

---

## Contents

1. [Architecture](#1-architecture)
2. [Technology stack](#2-technology-stack)
3. [Repository layout](#3-repository-layout)
4. [Environments, hardware and models](#4-environments-hardware-and-models)
5. [Phase 0 — Foundations (weeks 1–2)](#phase-0--foundations-weeks-12)
6. [Phase 1 — Core Offline MVP (weeks 3–6) → R1](#phase-1--core-offline-mvp-weeks-36--r1)
7. [Phase 2 — AutoML & Safety Gates (weeks 7–10) → R2](#phase-2--automl--safety-gates-weeks-710--r2)
8. [Phase 3 — Zero-Code Appliance & Pilot Demo (week 11 → end) → R3](#phase-3--zero-code-appliance--pilot-demo-week-11--end--r3)
9. [Phase 4 — Enterprise (after the semester) → R4](#phase-4--enterprise-after-the-semester--r4)
10. [Testing strategy](#10-testing-strategy)
11. [CI/CD and release process](#11-cicd-and-release-process)
12. [Team workflow and workstreams](#12-team-workflow-and-workstreams)
13. [Decision gates and risk checkpoints](#13-decision-gates-and-risk-checkpoints)
14. [Appendices](#14-appendices)

---

## 1. Architecture

### 1.1 Components

```
┌──────────────────────────────── Appliance (one Linux host, NVIDIA GPU(s)) ────────────────────────────────┐
│                                                                                                            │
│  Browser ──HTTPS──▶ ┌────────────┐   REST/SSE   ┌──────────────┐   job queue (DB)   ┌──────────────────┐ │
│  (local LAN only)   │  web (SPA) │ ───────────▶ │  api         │ ─────────────────▶ │  worker          │ │
│                     │  React+TS  │              │  FastAPI     │ ◀───────────────── │  pipeline runner │ │
│                     └────────────┘              │  auth, CRUD, │   status/events    │  (stage state    │ │
│                                                 │  run control │                    │   machine)       │ │
│                                                 └──────┬───────┘                    └────────┬─────────┘ │
│                                                        │ SQLite (→ Postgres R4)              │           │
│                                                        ▼                                     ▼           │
│                                               ┌─────────────────┐            ┌──────────────────────────┐ │
│                                               │ data volume      │◀──────────│ GPU tasks (subprocesses)  │ │
│                                               │ /data/workspaces │           │  • teacher vLLM (synth)   │ │
│                                               │ /data/runs       │           │  • trainer (PEFT/TRL)     │ │
│                                               │ /data/models     │           │  • evaluator (vLLM)       │ │
│                                               │ /data/registry   │           └──────────────────────────┘ │
│                                               └─────────────────┘                                        │
│                                                        │                                                  │
│                                                        ▼                                                  │
│                                               ┌─────────────────┐   OpenAI-compatible API (LAN)           │
│                                               │ serve: vLLM      │◀──── apps / playground                  │
│                                               │ base + LoRAs     │                                         │
│                                               └─────────────────┘                                         │
│  Network: containers on an internal-only network; host firewall drops all egress (NFR-SEC-01)            │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Component | Responsibility |
|---|---|
| `web` | Single-page app; static build served by `api`. No CDN, no external fonts. |
| `api` | FastAPI: auth, workspaces, corpus upload, run control, progress via Server-Sent Events, reports, registry, deployments. |
| `worker` | Polls the job table, runs the stage state machine, launches GPU work as subprocesses, writes artefacts and events. Exactly one GPU job at a time in R1–R3. |
| `serve` | vLLM server with LoRA enabled, hosting deployed adapters. Stopped while a training job needs the whole GPU (GPU arbiter, §1.4). |
| Data volume | Everything persistent. The containers themselves are stateless and read-only. |

### 1.2 Pipeline state machine

```
CREATED → INGESTING → SANITIZING → AWAITING_APPROVAL ─(user approves)→ SYNTHESIZING
        → PROFILING → TRAINING (HPO trials → final fit) → EVALUATING
        → { GATES_PASSED → PACKAGED → (DEPLOYED) | GATES_FAILED }
Any state → CANCELLING → CANCELLED;  any state → FAILED (with error code);  FAILED/CANCELLED → resume → last good state
```

Rules:
- Each state = one **stage** implementing `run(ctx) -> StageResult`, idempotent (NFR-REL-05).
- A stage writes outputs to a temporary directory, then atomically renames it and updates `manifest.json`. A crash never leaves half-written outputs.
- State transitions are written to the DB in a transaction together with an `audit_event`.
- Resume = restart the first stage whose output manifest is missing or whose input hashes changed.

### 1.3 Run directory (artefact contract)

```
/data/runs/<run_id>/
  manifest.json                  # run config, seeds, input hashes, per-stage status + output hashes
  01_ingest/   documents.parquet  pages.parquet  chunks.parquet  ingest_report.json
  02_sanitize/ chunks_clean.parquet  redactions.parquet (type, span, policy — no raw values)  policy.yaml
  03_synth/    taxonomy.json  candidates.jsonl  judged.jsonl  train.jsonl  val.jsonl  test.jsonl
               unanswerable_{train,test}.jsonl  canaries.json  synth_report.json
  04_train/    hw_profile.json  dryrun.json  study.db (Optuna)  trials/<n>/{config.json,metrics.jsonl,ckpt/}
               final/adapter/ (safetensors + adapter_config.json)
  05_eval/     results.json  per_item/*.jsonl  rag_index/  gates.json
  06_package/  adapter/  report.json  report.pdf  sbom.cdx.json  SHA256SUMS  SHA256SUMS.sig
  logs/        worker.jsonl  stage-*.jsonl
```

Tabular artefacts use Parquet (fast, typed); training data uses JSONL (what HF/TRL expect).

### 1.4 GPU arbiter

On a 1–2 GPU box the teacher, trainer, evaluator and serving server cannot all hold VRAM at once.
- The worker acquires a **GPU lease** (DB row + file lock) before any GPU stage.
- If `serve` holds the GPU and a job needs it, the worker asks `serve` to unload (R1–R3: stop the container; R4: dedicate one GPU to serving).
- Each GPU subprocess is started with `CUDA_VISIBLE_DEVICES` set by the arbiter and is killed on cancel.

### 1.5 Data model (SQLModel / SQLite)

| Table | Key fields |
|---|---|
| `user` | id, username, password_hash (argon2), role (admin/user), created_at |
| `workspace` | id, name, sensitivity_label, base_model_id, created_by |
| `document` | id, workspace_id, path, sha256, mime, pages, status, error |
| `run` | id, workspace_id, mode (A/B/C), profile, state, budget_s, seed, created_at, started_at, finished_at, error_code |
| `stage_run` | id, run_id, stage, state, started_at, finished_at, output_hash, metrics_json |
| `trial` | id, run_id, number, params_json, state, objective, metrics_json |
| `adapter` | id, run_id, base_model_id, path, sha256, gates_passed, created_at |
| `deployment` | id, adapter_id, endpoint_name, state, created_at |
| `gold_item` | id, workspace_id, question, reference_answer, reference_json, answerable |
| `model` | id, role (student/teacher/aux), name, path, sha256, license |
| `audit_event` | id, ts, actor, action, object_type, object_id, details_json |
| `api_key` | id, user_id, hash, created_at, revoked_at |

Migrations with Alembic from day one.

### 1.6 API surface (v1)

| Method & path | Purpose |
|---|---|
| `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/me` | Session auth |
| `GET/POST /api/workspaces`, `GET/PATCH/DELETE /api/workspaces/{id}` | Workspaces |
| `POST /api/workspaces/{id}/documents` (multipart, chunked) ; `POST …/documents:scan` (server folder) ; `GET …/documents` | Corpus |
| `GET /api/workspaces/{id}/documents/{doc}/preview` | Parsed text + redaction spans |
| `POST /api/workspaces/{id}/runs` ; `GET /api/runs/{id}` ; `POST /api/runs/{id}:approve` ; `POST /api/runs/{id}:cancel` ; `POST /api/runs/{id}:resume` | Run control |
| `GET /api/runs/{id}/events` (SSE) | Live progress |
| `GET /api/runs/{id}/trials` ; `GET /api/runs/{id}/scorecard` ; `GET /api/runs/{id}/report.{json,pdf}` | Results |
| `GET/POST /api/workspaces/{id}/gold` (CSV/JSONL import) | Gold set |
| `GET /api/adapters` ; `POST /api/adapters/{id}:deploy` ; `POST /api/deployments/{id}:undeploy` | Registry and serving |
| `POST /api/playground/chat` (proxy to vLLM, adds citations/JSON validation) | Playground |
| `GET /api/system/hardware` ; `GET /api/system/health` ; `GET /api/models` ; `POST /api/models:import` | Admin |

OpenAPI schema is generated by FastAPI and the TypeScript client is generated from it (`openapi-typescript`), so frontend and backend never drift.

### 1.7 Configuration and goal profiles

- `configs/defaults.yaml` — all thresholds from the PRD (τ_entail = 0.92, ε_forget = 3, split ratios, chunk size, …).
- `configs/profiles/mode_a.yaml`, `mode_b.yaml`, `mode_c.yaml` — synthesis task mix, objective weights (w1, w2, w3), gate thresholds.
- `configs/hardware/*.yaml` — toy, 24 GB, 48 GB, 80 GB presets (overridden by the live profiler).
- Loaded into typed **pydantic-settings** models; the fully resolved config is stored in each run's `manifest.json`.

---

## 2. Technology stack

Pin exact versions in `pyproject.toml`/`uv.lock` at Phase 0; the list below names components, not versions.

| Area | Choice | Why | Alternatives considered |
|---|---|---|---|
| Language / packaging | Python 3.11, **uv** (lockfile, `uv pip download` for the offline wheelhouse) | Fast, reproducible, hash-pinned | Poetry, pip-tools |
| API | **FastAPI** + Uvicorn, pydantic v2 | Typed, OpenAPI out of the box | Django, Flask |
| DB | **SQLite** (WAL) via SQLModel + Alembic; Postgres in R4 | Zero-ops on an appliance | Postgres from day 1 |
| Job orchestration | Own state machine + DB job table + worker process | Simple, resumable, no broker | Celery/RQ (needs Redis), Prefect/Dagster (heavier) |
| Document parsing | **Docling** (layout, tables, OCR via EasyOCR/Tesseract/RapidOCR), with offline model artefacts | MIT, layout-aware, works offline | Unstructured, Marker (GPL), PyMuPDF (AGPL) |
| Dedup | **datasketch** MinHash LSH | Standard, fast | text-dedup |
| PII | **Presidio** + spaCy (`en_core_web_lg`, `de_core_news_lg`) + custom regex recognisers | Configurable, offline, DE+EN | GLiNER-PII, own NER |
| Teacher / eval inference | **vLLM** (offline, batched; guided JSON decoding) | Throughput; LoRA serving; OpenAI API | SGLang, TGI |
| NLI judge | DeBERTa-v3-large NLI cross-encoder via `sentence-transformers`/`transformers` | Strong, small, local | LLM-only judge |
| Training | **transformers + PEFT + TRL** (`SFTTrainer`; `DPOTrainer`/`ORPOTrainer` in R4), **bitsandbytes** (NF4), FlashAttention-2; **Unsloth** optional speed-up on single GPU | De-facto standard; QLoRA support | Axolotl (config-driven wrapper), LLaMA-Factory |
| HPO | **Optuna** (`TPESampler`, `SuccessiveHalvingPruner`/`HyperbandPruner`, SQLite storage) single-node; **Ray Tune** + ASHA for multi-GPU in R4 | Simple, resumable, pruning built in | Ray Tune from day 1 (heavier) |
| Retrieval (RAG baseline) | BGE-M3 embeddings + **FAISS** (CPU) | Local, simple | Qdrant, LanceDB |
| Benchmarks | Own harness + bundled subsets (MMLU, GSM8K, IFEval, JSON-schema tasks, HumanEval-mini); optionally **lm-evaluation-harness** with local datasets | Offline, fast | — |
| Metrics | `rouge-score`, embedding cosine, LLM-judge rubric, field-level F1, `jsonschema` | — | — |
| GPU monitoring | `nvidia-ml-py` (NVML) | Temperature, power, memory | DCGM |
| Reports | Jinja2 HTML → PDF via **WeasyPrint**; charts via matplotlib (SVG) | Offline, no browser needed | Typst, ReportLab |
| SBOM | **CycloneDX** (`cyclonedx-python-lib`; ML-BOM components for models/datasets) | Standard, includes ML-BOM | SPDX 3 AI profile |
| Signing | **minisign** or **cosign** with a local Ed25519 key; offline verification | Diode-friendly | GPG |
| Frontend | **React + TypeScript + Vite**, TanStack Query, React Router, Tailwind + shadcn/ui, `openapi-typescript` | Fast to build; static bundle | HTMX + Jinja, Svelte |
| Containers | Docker / OCI images, Docker Compose; base `nvidia/cuda` runtime (Ubuntu) → UBI/STIG in R4 | GPU support, familiar | Podman (supported later) |
| Quality | ruff, mypy (strict on `zllf.core`), pytest, pytest-cov, hypothesis; eslint, prettier, vitest, Playwright | — | — |
| CI | GitHub Actions (CPU); manual/self-hosted runner for GPU jobs | Free for public repo | — |

---

## 3. Repository layout

```
kelpod/
├── README.md                     # what it is, quick start (toy mode), links to docs
├── LICENSE                       # decide in P0.1 (see open question)
├── CLAUDE.md / AGENTS.md         # conventions for coding agents (P0.6)
├── pyproject.toml  uv.lock       # Python workspace
├── Makefile                      # make setup | lint | test | toy-run | bundle | egress-test
├── configs/
│   ├── defaults.yaml
│   ├── profiles/  mode_a.yaml mode_b.yaml mode_c.yaml
│   ├── hardware/  toy.yaml gpu24.yaml gpu48.yaml gpu80.yaml
│   └── pii/       rules_de.yaml rules_en.yaml
├── src/zllf/
│   ├── core/          # config, logging, ids, hashing, manifest, errors, offline guard
│   ├── db/            # models, session, alembic migrations
│   ├── api/           # FastAPI app, routers, auth, SSE
│   ├── worker/        # job loop, state machine, GPU arbiter
│   ├── stages/
│   │   ├── ingest/        # loaders, docling adapter, OCR, chunker, dedup
│   │   ├── sanitize/      # presidio engine, recognisers, policy, preview spans
│   │   ├── synth/         # teacher client, prompts, generators, judge, splits, canaries
│   │   ├── train/         # hw profiler, dryrun, trainer, OOM handler, HPO, replay mixer
│   │   ├── evaluate/      # harness, gates, RAG baseline, benchmarks, judge rubric
│   │   └── package/       # export, sbom, report, signing, registry
│   ├── serve/         # vLLM launcher, adapter loading, playground proxy
│   ├── llm/           # thin client over vLLM (offline engine + OpenAI-compatible HTTP)
│   └── cli.py         # `zllf run|ingest|synth|train|eval|package|serve|verify-egress`
├── web/                # React app
├── bench/
│   ├── anchor/        # forgetting benchmark subsets (licence-checked)
│   ├── replay/        # general instruction replay data (licence-checked)
│   └── corpora/       # *scripts* to fetch public demo corpora (no data committed)
├── deploy/
│   ├── docker/        # Dockerfile.api, Dockerfile.worker, Dockerfile.serve
│   ├── compose/       # compose.yaml (internal network), compose.dev.yaml
│   ├── bundle/        # build_bundle.sh, verify_bundle.sh, install.sh, manifest schema
│   └── egress/        # nftables rules, tcpdump harness, egress test script
├── scripts/            # download_models.py (online build host only), make_synthetic_notes.py, …
├── tests/
│   ├── unit/  integration/  e2e_toy/  gpu/  fixtures/
└── docs/
    ├── PRD.md  DEVELOPMENT_PLAN.md
    ├── adr/           # architecture decision records (0001-…)
    └── runbooks/      # install, upgrade, troubleshooting
```

---

## 4. Environments, hardware and models

### 4.1 Environments

| Env | Hardware | Models | Used for |
|---|---|---|---|
| **toy** | Any laptop incl. Apple Silicon, CPU only | ≈ 0.5B student & teacher; small NLI/embeddings | Daily dev, unit/integration tests, CI. Checks plumbing, not quality. vLLM is replaced by a `transformers` fallback backend in this profile. |
| **gpu-dev** | Linux, 1× 24–48 GB NVIDIA GPU | 7–8B student, 14B AWQ teacher | Stage development, benchmarks |
| **gpu-demo** | Linux, 1–2× 48–80 GB GPU | 8B student, 14–32B AWQ teacher | Reference runs, timings, final demo |
| **air-gapped test** | gpu-demo machine with egress firewall / unplugged | Same as demo | Egress test, install-from-bundle test |

> The current dev Mac has no NVIDIA GPU, so vLLM (CUDA), bitsandbytes and FlashAttention will not run there. Plan on SSH + VS Code Remote / Claude Code on a Linux GPU server for all GPU work.

### 4.2 Model inventory to pre-download (online build host → bundle)

| Role | Toy | GPU default |
|---|---|---|
| Student | Qwen2.5-0.5B-Instruct | Qwen3-8B (+ Llama-3.1-8B-Instruct for the pitch comparison) |
| Teacher | Qwen2.5-0.5B-Instruct | Qwen 14B instruct AWQ (24–48 GB) / 32B AWQ (80 GB) |
| NLI | small NLI cross-encoder | DeBERTa-v3-large NLI |
| Embeddings | small MiniLM | BGE-M3 |
| OCR/layout | Docling artefacts + EasyOCR (en, de) | same |
| NER | spaCy `en_core_web_sm`, `de_core_news_sm` | `*_lg` |

`scripts/download_models.py` fetches each model by **pinned revision (commit hash)**, writes `models/manifest.json` (name, revision, sha256 per file, licence) — the only script allowed network access, and only on the build host.

---

## Phase 0 — Foundations (weeks 1–2)

**Goal:** a repo where both founders (and coding agents) can work productively; GPU access secured; decisions recorded.

### P0.1 Project decisions (ADRs)
- [ ] P0.1.1 Write `docs/adr/0001-record-architecture-decisions.md` (template).
- [ ] P0.1.2 ADR-0002 stack (§2), ADR-0003 orchestration (own state machine, not Celery), ADR-0004 HPO (Optuna single-node), ADR-0005 parsing (Docling), ADR-0006 serving (vLLM).
- [ ] P0.1.3 Decide code licence (e.g., Apache-2.0 vs. "all rights reserved"/BSL for a startup) and repo visibility — PRD open question 6.
- [ ] P0.1.4 Choose primary demo domain and Mode (recommendation: healthcare, Modes A + C) and secondary (energy, Mode A).
- **Done when:** ADRs merged; open questions 1, 2, 4, 6 in the PRD answered or owned.

### P0.2 Repository scaffolding
- [ ] P0.2.1 `pyproject.toml` with uv; `src/zllf` package; ruff + mypy config; `Makefile`.
- [ ] P0.2.2 `pre-commit` hooks: ruff, ruff-format, mypy (core), end-of-file, detect-secrets, large-file guard (block files > 5 MB and `*.pdf`, `*.parquet`, `*.safetensors` outside fixtures).
- [ ] P0.2.3 `.gitignore` for `/data`, `/models`, `*.safetensors`, `.env`, `runs/`.
- [ ] P0.2.4 `web/` scaffold: Vite + React + TS + Tailwind + eslint/prettier + vitest.
- [ ] P0.2.5 README with toy-mode quick start; `CONTRIBUTING.md` (branching, commit style).
- **Done when:** `make setup && make lint && make test` passes on a fresh clone (Mac + Linux).

### P0.3 Core library skeleton (`zllf.core`)
- [ ] P0.3.1 `config.py`: pydantic-settings models for defaults/profiles/hardware; YAML loader; resolved-config dump.
- [ ] P0.3.2 `logging.py`: structlog JSON logs with `run_id`, `stage`; redaction-safe formatter (never logs chunk text by default).
- [ ] P0.3.3 `hashing.py`: streaming SHA-256 for files/dirs; `manifest.py`: read/write/atomic-commit stage outputs.
- [ ] P0.3.4 `offline.py`: **offline guard** — sets all telemetry/offline env vars (Appendix A) at import; optional `socket` monkey-patch in tests that raises on any non-loopback connect.
- [ ] P0.3.5 `errors.py`: error codes (`E_INGEST_UNSUPPORTED`, `E_OOM_UNRECOVERABLE`, …) with user-facing messages.
- **Done when:** unit tests cover config resolution, atomic manifest commit (simulated crash), offline guard blocks a test HTTP call.

### P0.4 GPU access and environment
- [ ] P0.4.1 Secure a Linux GPU machine for the semester (institute cluster / course allocation); document access in `docs/runbooks/gpu-access.md` (no credentials in repo).
- [ ] P0.4.2 Install driver/CUDA check script `scripts/check_gpu.py` (prints GPUs, VRAM, CUDA, BF16/FP8 support).
- [ ] P0.4.3 Smoke test on GPU: load 8B student in 4-bit, one forward/backward pass; run vLLM with the 14B AWQ teacher and generate 10 completions. Record tokens/s → `docs/benchmarks/baseline.md`.
- [ ] P0.4.4 If cluster uses Slurm: write `deploy/slurm/*.sbatch` wrappers; note that Docker may be replaced by Apptainer there (keep the code container-agnostic).
- **Done when:** both founders can run the GPU smoke test; throughput numbers recorded.

### P0.5 Model and data acquisition (online build host)
- [ ] P0.5.1 `scripts/download_models.py` with pinned revisions + manifest + licence field (§4.2).
- [ ] P0.5.2 `bench/corpora/fetch_*.py` for the chosen demo corpus (public docs only) + licence notes in `bench/corpora/README.md`.
- [ ] P0.5.3 `scripts/make_synthetic_notes.py`: generate synthetic patient notes with known, labelled PII and biomarkers (ground truth for FR-03 recall and Mode C extraction). Use templates + teacher model; store the labels.
- [ ] P0.5.4 Build the **reference corpora**: `ref-small` (50 docs, for CI on GPU), `ref-500` (500 PDFs, R1 exit), `ref-1000p` (1,000 pages, timing target).
- **Done when:** all artefacts reproducible from scripts; nothing restricted committed.

### P0.6 Working with coding agents
- [ ] P0.6.1 `CLAUDE.md`/`AGENTS.md`: architecture summary, run directory contract, "never add network calls", test commands, style rules.
- [ ] P0.6.2 Issue template with *Context / Requirement IDs / Done when*; PR template with checklist (tests, offline guard, docs).
- [ ] P0.6.3 Rule: every agent-written PR is reviewed by the other founder; GPU-dependent code must include a toy-profile test.

### P0.7 Project management
- [ ] P0.7.1 GitHub Project board; milestones R1/R2/R3; one issue per step in this plan.
- [ ] P0.7.2 Weekly 30-min planning + demo; decision log in `docs/adr`.

---

## Phase 1 — Core Offline MVP (weeks 3–6) → R1

**Exit criteria (Intermediary Goal 1):** inside an offline Docker container, `ref-500` (500 PDFs) is ingested, synthetic QA is generated with the local vLLM teacher, a QLoRA adapter is trained on one GPU with fixed defaults, evaluated on the test split, and served — **with zero outbound packets and no OOM crash**. A minimal UI (or CLI + simple page) starts the run and shows progress.

### P1.1 Ingestion (FR-01, FR-02 basic)
- [ ] P1.1.1 **Loader registry** keyed by MIME/extension: PDF, DOCX, TXT, MD, HTML, XML, CSV, Parquet, images (TIFF/PNG/JPEG). Unknown types → `skipped` with reason.
- [ ] P1.1.2 **Docling adapter**: convert to a common `ParsedDocument {doc_id, sha256, pages[], blocks[{type: heading|paragraph|list|table|caption, text, page, bbox, level}]}`; configure Docling `artifacts_path` to bundled models (no downloads).
- [ ] P1.1.3 **Scanned-page detection** (no/low text layer → OCR); OCR languages en + de; record OCR flag per page.
- [ ] P1.1.4 **Tables** → Markdown and a structured cell grid (kept for Mode C and for "quote-first" synthesis of table facts).
- [ ] P1.1.5 **CSV/Parquet**: row groups rendered as text records with column headers; cap rows per chunk.
- [ ] P1.1.6 **Normalisation**: Unicode NFKC, de-hyphenation across line breaks, header/footer removal (repeated lines across pages), whitespace cleanup.
- [ ] P1.1.7 **Structural chunker**: section → paragraph → sentence fallback; target 512 tokens (student tokenizer), max 1,024, 10 % overlap only at sentence-fallback splits; keep `heading_path` and page refs in each chunk.
- [ ] P1.1.8 **Dedup**: exact (hash) then MinHash LSH (5-gram shingles, Jaccard ≥ 0.85) on chunks; keep first occurrence; record duplicates.
- [ ] P1.1.9 Parallelism: process pool for CPU parsing; per-document timeout; failures isolated per document.
- [ ] P1.1.10 Outputs + `ingest_report.json` (counts by type, OCR pages, failures, duplicates, tokens).
- [ ] P1.1.11 Tests: fixtures (digital PDF, scanned PDF, DOCX with table, broken PDF); golden-file tests on chunk boundaries; property test that chunk text concatenation covers the document.
- **Done when:** `ref-500` ingests offline; ≥ 99 % of files parsed or skipped with reason; throughput logged.

### P1.2 Sanitisation v1 (FR-03, NFR-SEC-06)
- [ ] P1.2.1 Presidio `AnalyzerEngine` with spaCy en + de NLP engines; language detection per chunk.
- [ ] P1.2.2 Custom recognisers (`configs/pii/rules_*.yaml`): German health-insurance number (KVNR), IBAN, German tax ID, phone/e-mail, dates of birth patterns, patient/case IDs (configurable regex), US SSN, classification headers.
- [ ] P1.2.3 Policy engine: per entity type `keep | mask | drop_chunk`; consistent typed placeholders per document (`[PERSON_1]`) via a per-run salted mapping held **in memory only** (never persisted).
- [ ] P1.2.4 Write `redactions.parquet` with (chunk_id, start, end, entity_type, action, score) — no raw text.
- [ ] P1.2.5 Store a **raw-PII fingerprint set** (salted hashes of redacted values) for the memorisation gate (P2.6.4) — hashes only.
- [ ] P1.2.6 Evaluation script on synthetic-notes ground truth → precision/recall per entity type; target recall ≥ 95 %.
- [ ] P1.2.7 `AWAITING_APPROVAL` state: in R1 the CLI/UI shows a summary (counts per type + 10 sample redacted chunks) and an approve button/command.
- **Done when:** recall ≥ 90 % in R1 (≥ 95 % by R2); log scan test finds no raw PII in any log/report.

### P1.3 LLM runtime layer (`zllf.llm`)
- [ ] P1.3.1 Interface `LLMBackend.generate(batch, sampling, guided_json=None) -> list[Output]`.
- [ ] P1.3.2 `VLLMOfflineBackend` (in-process `vllm.LLM`, for batch synthesis/eval) and `VLLMHTTPBackend` (OpenAI-compatible server, for serving/playground).
- [ ] P1.3.3 `HFBackend` (transformers `generate`) for the toy profile and CPU CI.
- [ ] P1.3.4 Lifecycle: start/stop, VRAM fraction (`gpu_memory_utilization`), `max_model_len`, LoRA loading; ensure process exits and frees VRAM (subprocess per stage).
- [ ] P1.3.5 Prompt templates in `src/zllf/stages/synth/prompts/*.j2`, versioned; template hash recorded in manifest.
- **Done when:** same synthesis code runs on toy (HF) and GPU (vLLM).

### P1.4 Synthetic data v1 (FR-04 basic)
- [ ] P1.4.1 **Quote-first QA generator** (Mode A): for each chunk, teacher extracts 1–5 verbatim key quotes (guided JSON), then writes one question + answer per quote; temperature 0.1; output `{question, answer, quote, chunk_id, doc_id, task: "qa"}`.
- [ ] P1.4.2 **Exact-substring check**: quote must appear in the chunk (after whitespace normalisation); else drop.
- [ ] P1.4.3 **NLI judge**: premise = chunk (or quote ± context window), hypothesis = answer restated as statement; keep if entailment ≥ τ (0.92).
- [ ] P1.4.4 **Light LLM judge** (teacher, separate prompt): "Is the question answerable from the passage alone and is the answer complete? yes/no + reason".
- [ ] P1.4.5 Dedup synthetic questions (embedding cosine ≥ 0.95 → drop).
- [ ] P1.4.6 **Document-level split** 70/15/15 with fixed seed (PRD §11.2).
- [ ] P1.4.7 Chat formatting with the student's chat template; system prompt per mode.
- [ ] P1.4.8 `synth_report.json`: candidates, kept, drop reasons, per-doc coverage, avg lengths.
- [ ] P1.4.9 Human audit tool: `zllf audit-synth --n 100` → CSV for manual labelling; compute grounding score.
- **Done when:** on `ref-500`, ≥ 3,000 kept pairs; human audit grounding ≥ 85 % in R1 (target ≥ 92 % by R2).

### P1.5 Training v1 (fixed defaults)
- [ ] P1.5.1 Trainer wrapper on TRL `SFTTrainer` + PEFT `LoraConfig`; QLoRA (NF4, double quant, BF16 compute) with defaults r = 16, α = 32, dropout 0.05, all-linear targets, lr 2e-4 cosine, 2 epochs, max seq 2,048, packing on.
- [ ] P1.5.2 Loss on assistant tokens only (completion-only masking).
- [ ] P1.5.3 Checkpoints every N steps + time-based (≤ 15 min) with atomic save; resume from last checkpoint (FR-12 basic).
- [ ] P1.5.4 Metrics stream to `trials/<n>/metrics.jsonl` (loss, lr, tokens/s, VRAM peak, GPU temp) — no W&B (offline); worker turns them into SSE events.
- [ ] P1.5.5 Seed everything; record library versions and GPU model.
- [ ] P1.5.6 Toy-profile test: train 20 steps on CPU with 0.5B model.
- **Done when:** 8B QLoRA trains on `ref-500` data on one GPU; adapter < 500 MB; resume after `kill -9` works.

### P1.6 Evaluation v1
- [ ] P1.6.1 Batched inference on the test split for base vs. adapter (vLLM with LoRA).
- [ ] P1.6.2 Metrics: ROUGE-L, embedding similarity, LLM-judge correctness (rubric, temperature 0).
- [ ] P1.6.3 Simple forgetting check: 200-item mixed general set (subset of bundled benchmarks) base vs. adapter.
- [ ] P1.6.4 `results.json` + simple HTML summary.
- **Done when:** adapter vs. base numbers produced automatically at the end of every run.

### P1.7 Package and serve v1 (FR-08 adapter, FR-15 endpoint)
- [ ] P1.7.1 Export adapter dir + `SHA256SUMS`; register in `adapter` table.
- [ ] P1.7.2 `serve` container: vLLM OpenAI server with `--enable-lora`, `--max-loras`, `--max-lora-rank` matching the HPO space; runtime LoRA loading enabled; bound to the internal network/LAN only.
- [ ] P1.7.3 `zllf deploy <adapter_id>` loads the adapter under a stable model name (`ws-<workspace>-v<n>`).
- **Done when:** `curl` chat completion with the adapter works with networking disabled.

### P1.8 Worker, state machine and run CLI (FR-12 stage-level, NFR-REL-05)
- [ ] P1.8.1 DB models + Alembic initial migration (§1.5).
- [ ] P1.8.2 Worker loop: claim next `run` (row lock), execute stages in order, write `stage_run` rows and events.
- [ ] P1.8.3 Cancel: cooperative flag checked between batches + SIGTERM to GPU subprocess; cleanup temp dirs.
- [ ] P1.8.4 Resume: recompute from manifests (§1.2).
- [ ] P1.8.5 `zllf run --workspace … --mode A --corpus ./docs` end-to-end CLI.
- **Done when:** toy e2e test passes in CI; GPU e2e on `ref-500` passes.

### P1.9 Minimal API + UI (FR-10 minimal)
- [ ] P1.9.1 API: workspaces, upload (chunked, resumable), start run, approve, cancel, run status, SSE events.
- [ ] P1.9.2 UI: create workspace → upload → approve redaction summary → start → progress timeline → results table → deploy button.
- [ ] P1.9.3 All assets local (no CDN fonts/scripts); build served by FastAPI.
- **Done when:** a full run can be started and followed from the browser.

### P1.10 Offline container and egress test (NFR-SEC-01, NFR-REL-03)
- [ ] P1.10.1 Dockerfiles (multi-stage): `api`, `worker` (CUDA + torch + vLLM + training libs), `serve` (vLLM). Install from the **local wheelhouse** only (`pip install --no-index --find-links`).
- [ ] P1.10.2 Models mounted read-only from `/data/models`; startup check that every model in the manifest exists and hashes match (fast mode: size + sample hash; full verify on install).
- [ ] P1.10.3 `compose.yaml`: `internal: true` network for api/worker/serve; only the UI port published to the host LAN.
- [ ] P1.10.4 **Egress test** `deploy/egress/run_egress_test.sh`: start stack in a network namespace; nftables rule `log + counter + drop` for all non-loopback egress; `tcpdump` on all interfaces; run toy (CI) or `ref-500` (GPU) pipeline; assert 0 packets and 0 DNS queries; store pcap summary as artefact.
- [ ] P1.10.5 Also run the full GPU pipeline with `docker run --network none` for the worker as a second line of evidence.
- **Done when:** egress test green in CI (toy) and on GPU (`ref-500`) — **R1 exit criterion**.

### P1.11 R1 demo and retrospective
- [ ] P1.11.1 Record a 3-minute demo video (offline proof: show unplugged/blocked network + counter = 0).
- [ ] P1.11.2 Measure timings per stage on `ref-500` and `ref-1000p` → `docs/benchmarks/r1.md`; update the PRD §8.2 budget.
- [ ] P1.11.3 Retro: scope adjustments for Phase 2.

---

## Phase 2 — AutoML & Safety Gates (weeks 7–10) → R2

**Exit criteria (Intermediary Goal 2):** HPO selects hyperparameters automatically within a time budget; replay prevents forgetting; the four gates run and block bad adapters; RAG baseline and gold set are evaluated; a Model Readiness Scorecard PDF and A-SBOM are produced. **Tuned adapter ≥ 12 % better than the fixed-default adapter on the domain hold-out** of the reference corpus (original roadmap exit criterion).

### P2.1 Hardware profiler (FR-06)
- [ ] P2.1.1 NVML/torch probe: GPU count, names, VRAM total/free, compute capability, BF16/FP8 support, NVLink/PCIe topology (`nvidia-smi topo -m` parse), CPU cores, RAM, free disk on `/data`.
- [ ] P2.1.2 **Memory model**: estimate VRAM for (model size, quantisation, rank, target modules, seq len, micro-batch, checkpointing) using a calibrated formula; calibrate with measured peaks from P0.4.3/P1.5.
- [ ] P2.1.3 Decision table → `hw_profile.json`: QLoRA vs. 16-bit LoRA; FlashAttention-2 if supported; initial micro-batch/accumulation to hit target effective batch (e.g., 16); `gpu_memory_utilization` for vLLM stages.
- [ ] P2.1.4 **Dry run**: forward + backward at max sequence length with the largest rank in the search space; measure peak; shrink the search space (drop configs predicted > 90 % VRAM).
- **Done when:** profiles for 24/48/80 GB machines validated; dry run predicts OOM-free settings in 20/20 runs.

### P2.2 OOM-safe trial runner (NFR-REL-01, NFR-REL-02)
- [ ] P2.2.1 Each trial runs in a **subprocess**; the controller survives any trial crash.
- [ ] P2.2.2 On `torch.cuda.OutOfMemoryError` (or CUDA OOM exit code): halve micro-batch, double accumulation, enable gradient checkpointing, (optionally Unsloth kernels), retry up to 3 times; then mark trial `failed_oom` with the config, not the study.
- [ ] P2.2.3 Atomic checkpoints every ≤ 15 min; resume trial from checkpoint after power loss; Optuna storage in SQLite so the study resumes too.
- [ ] P2.2.4 Chaos test: random `kill -9` and simulated OOM during a GPU run → run completes.
- **Done when:** chaos test passes 5/5.

### P2.3 Replay mixing (FR-05)
- [ ] P2.3.1 Bundle a licence-checked general instruction dataset (`bench/replay/`, e.g., 20–50k samples) — read-only, hashed.
- [ ] P2.3.2 Mixer: sample β_replay fraction (by tokens) into the training set; deterministic per seed; record sample IDs.
- [ ] P2.3.3 Corpus-size rule: β_min = clip(0.30 − 0.05·log10(n_domain_samples/500), 0.05, 0.30) (tune empirically).
- **Done when:** with β = 0 vs. tuned β on `ref-small`, forgetting suite shows the expected difference (documented).

### P2.4 HPO controller (FR-07, FR-19)
- [ ] P2.4.1 Optuna study per run: `TPESampler(multivariate=True, seed)`, `SuccessiveHalvingPruner` (min resource = 25 % of steps, reduction factor 3) or Hyperband.
- [ ] P2.4.2 Search space exactly as PRD §8.1; constraints: drop configs failing the VRAM model; conditional sampling (α from α/r ratio).
- [ ] P2.4.3 **Objective** = PRD eq. (3): L_domain_val (val loss) + w2·max(0, L_anchor − L_anchor_base − ε) + w3·Φ_halluc; Φ_halluc measured on a small val set of unanswerables (fabrication rate). Intermediate values reported at each eval step for pruning.
- [ ] P2.4.4 **Fidelity**: trials train on a subset (e.g., 30 % of data, 1 epoch); the best config is retrained at full fidelity for the final adapter.
- [ ] P2.4.5 **Warm start**: enqueue 2–3 known-good configs per mode (from Phase-1 experiments) as first trials.
- [ ] P2.4.6 **Time-budget planner**: measure tokens/s in the dry run → estimate trial cost → choose n_trials and fidelity to fit `T_budget` minus other stages; re-plan after each trial; hard stop before budget.
- [ ] P2.4.7 Leaderboard events to UI (trial, params summary, objective, state).
- [ ] P2.4.8 rsLoRA / LoRA+ as categorical options in the space (R2 stretch).
- [ ] P2.4.9 Experiment: tuned vs. fixed-default adapter on `ref-500` and gold set; ≥ 12 % relative on domain hold-out → `docs/benchmarks/r2-hpo.md`.
- **Done when:** exit criterion met or documented with analysis and a plan.

### P2.5 Synthesis v2 (FR-04 full)
- [ ] P2.5.1 Taxonomy extraction (entities, acronyms + expansions, concepts) with guided JSON; merge across chunks; `taxonomy.json`.
- [ ] P2.5.2 Task generators per mode: summarisation, acronym/definition, report drafting in house style (B); multi-turn troubleshooting dialogues (A, P1); extraction tasks with schema-valid targets (C, prepared for P3.4).
- [ ] P2.5.3 **Unanswerable generator**: questions about entities/attributes absent from the corpus (verified absent by retrieval + NLI), target = calibrated abstention text; split into train and gate-test sets.
- [ ] P2.5.4 **Canaries**: insert k (e.g., 10) random secret strings into synthetic training samples with distinct contexts; store in `canaries.json` (test-only knowledge).
- [ ] P2.5.5 Throughput: vLLM batch sizes, prefix caching for shared prompts; target synthesis ≤ 1.5 h for 1,000 pages.
- [ ] P2.5.6 Re-run human audit; grounding ≥ 92 %.
- **Done when:** grounding ≥ 92 % on the 100-pair audit; synthesis fits budget.

### P2.6 Evaluation harness and four gates (Stage ⑤, FR-13, FR-14)
- [ ] P2.6.1 **Harness**: item = {id, prompt, references, type}; runner batches through vLLM for 4 variants (base, base+RAG, adapter, adapter+RAG); per-item results JSONL.
- [ ] P2.6.2 **RAG baseline**: BGE-M3 embeddings over `chunks_clean`, FAISS index, top-k = 5, fixed prompt with citations.
- [ ] P2.6.3 **G-1 Domain**: synthetic test split + gold set; metrics: ROUGE-L, embedding similarity, LLM-judge correctness (rubric prompt, temperature 0, fixed judge model); bootstrap 95 % CI (1,000 resamples).
- [ ] P2.6.4 **G-2 Forgetting**: bundled subsets (e.g., MMLU 500, GSM8K 200, IFEval 200, JSON-schema 200, HumanEval-mini 50); per-suite delta vs. base; fail if any drop > ε_forget.
- [ ] P2.6.5 **G-3 Hallucination**: held-out unanswerables + adversarial prompts ("According to the manual, what is the procedure for [non-existent code]?"); classify abstain/answer with judge; abstention ≥ 90 %, fabrication ≤ 5 %.
- [ ] P2.6.6 **G-4 Memorisation**: (a) canary prefix-completion with greedy and sampled decoding — fail if any canary is reproduced; (b) exposure metric (rank of canary among random candidates) reported; (c) scan all eval outputs for raw-PII fingerprints (salted hash match on n-grams) — fail on any hit.
- [ ] P2.6.7 Gate engine: thresholds from profile; `gates.json` with pass/fail, value, threshold, evidence links; admin override records justification (FR, audit).
- [ ] P2.6.8 Gold set import (CSV/JSONL) API + validation (FR-14).
- [ ] P2.6.9 Judge validation: founders label 100 judge decisions → agreement rate reported in the scorecard.
- **Done when:** a deliberately bad adapter (high lr, no replay) is blocked by G-2; an adapter trained without unanswerables is flagged by G-3; a canary-overfit adapter is blocked by G-4.

### P2.7 A-SBOM and reports (FR-09)
- [ ] P2.7.1 `report.json` schema (JSON Schema in repo): run config, input doc hashes, redaction stats, synthesis stats, trials, final params, loss curves, gate results, environment (versions, GPU, driver), seeds, timings.
- [ ] P2.7.2 CycloneDX ML-BOM: components for base model, teacher, NLI, embeddings, datasets (replay, anchor, synthetic), software packages (from lockfile), adapter (with hashes); validate with CycloneDX schema.
- [ ] P2.7.3 Scorecard PDF (WeasyPrint): page 1 traffic-light gates + headline metrics vs. base/RAG; page 2 examples; appendix trials/curves/lineage.
- [ ] P2.7.4 `SHA256SUMS` over the package + Ed25519 signature with the appliance key.
- [ ] P2.7.5 `zllf verify-package <dir>` checks hashes and signature.
- **Done when:** report validates; reproducibility test (same seed + inputs) reproduces metrics within ±1 point.

### P2.8 Watchdog and scratch cleanup (NFR-REL-04, NFR-SEC-03)
- [ ] P2.8.1 NVML watchdog thread: temperature, power, throttle reasons, memory; events to UI; lower power limit (requires privilege — document; fallback: pause between steps) when temp > threshold.
- [ ] P2.8.2 Cleanup on completion/cancel: delete trial checkpoints except the winner, tokenisation caches, temp datasets; record what was deleted.

### P2.9 R2 demo and retrospective
- [ ] P2.9.1 Demo: show HPO leaderboard, a blocked bad adapter, scorecard PDF.
- [ ] P2.9.2 Benchmarks `docs/benchmarks/r2.md` (time-to-model on `ref-1000p`, gains, gates).
- [ ] P2.9.3 Retro; freeze R3 scope.

---

## Phase 3 — Zero-Code Appliance & Pilot Demo (week 11 → end) → R3

**Exit criteria (End Goal):** a non-ML user goes from drag-and-drop to a deployed local endpoint in the web UI; installation from a signed offline bundle; validated on the chosen "AI for Good" dataset with ≥ 12 % relative improvement over the base model on the human gold set; business case validated.

### P3.1 Full web UI (FR-10, PRD §7.2)
- [ ] P3.1.1 Design system (shadcn/ui), layout, plain-language copy (PRD §7.3); i18n-ready strings (EN first, DE later).
- [ ] P3.1.2 Login + first-run admin wizard (FR-16).
- [ ] P3.1.3 Workspaces list/create (sensitivity label, base model picker with licence badge).
- [ ] P3.1.4 Corpus page: drag-and-drop files **and folders** (directory upload), resumable chunked upload, server-folder picker (NFS/SMB mount paths configured by admin), file table with statuses.
- [ ] P3.1.5 Redaction preview: document viewer (page image + text with highlighted spans), per-type toggles, custom regex tester, approve.
- [ ] P3.1.6 Goal page: Mode A/B/C cards, JSON-schema editor with templates and validation (Mode C), time budget picker, gold set upload/editor.
- [ ] P3.1.7 Run monitor: stage timeline, ETA, friendly log, GPU gauges, HPO leaderboard, cancel/resume.
- [ ] P3.1.8 Scorecard page: gates, metrics vs. 3 baselines with CIs, example comparisons, report downloads, Deploy (disabled unless gates pass; override flow for admins).
- [ ] P3.1.9 Playground: chat with streaming, citations (Mode A, via retrieval of supporting chunks), JSON validation badge (Mode C), compare with base side by side.
- [ ] P3.1.10 Models & deployments page: adapters, versions, deploy/undeploy, endpoint URL, API keys, copy-paste `curl`/Python example.
- [ ] P3.1.11 Admin: hardware profile, disk usage, users, audit log viewer, installed models, update import.
- [ ] P3.1.12 Playwright e2e tests for the golden path on the toy profile.
- **Done when:** golden path Playwright test green; usability tests (P3.6) pass.

### P3.2 Auth, audit and API keys (FR-16, NFR-SEC-05)
- [ ] P3.2.1 Argon2 password hashing; secure HTTP-only session cookies; CSRF protection; rate-limited login.
- [ ] P3.2.2 Roles: admin / user; workspace-level access.
- [ ] P3.2.3 API keys (hashed) for the inference endpoint via a thin auth proxy in front of vLLM.
- [ ] P3.2.4 Audit log for all state-changing actions.
- [ ] P3.2.5 TLS with a self-signed/internal-CA certificate generated at install.

### P3.3 Multi-adapter serving (FR-21 basic)
- [ ] P3.3.1 vLLM with `--max-loras N`; registry → load/unload via runtime API; model names per workspace.
- [ ] P3.3.2 GPU arbiter integration: serving pauses during training on single-GPU boxes, with a UI banner; on 2-GPU boxes, GPU 1 dedicated to serving.
- [ ] P3.3.3 Load test: 5 adapters, 10 concurrent users → latency report.

### P3.4 Mode C structured extraction (FR-18)
- [ ] P3.4.1 Schema handling: JSON Schema draft 2020-12 subset; templates (biomarkers, medication list, fault report).
- [ ] P3.4.2 Synthesis: teacher extracts target JSON from chunks under guided decoding; validation + NLI check per field value (value must appear/entail from text).
- [ ] P3.4.3 Training with schema in the system prompt; evaluation: field-level precision/recall/F1, exact match, schema-validity rate.
- [ ] P3.4.4 Inference: guided JSON decoding in the playground/API (`response_format` / guided JSON).
- **Done when:** ≥ 98 % schema-valid outputs; F1 improvement over base reported on synthetic-notes ground truth.

### P3.5 Offline bundle and installer (NFR-SEC-02, NFR-OPS-01, FR-17)
- [ ] P3.5.1 `deploy/bundle/build_bundle.sh` (online build host): build images → `docker save`; wheelhouse; models; frontend build; `bundle-manifest.json` (every file + sha256 + licence); split into fixed-size parts (e.g., 4 GB) for media/diodes; sign manifest (Ed25519).
- [ ] P3.5.2 `verify_bundle.sh`: verify signature and all hashes **before** anything is loaded; refuse on mismatch.
- [ ] P3.5.3 `install.sh`: prerequisites check (driver, Docker + NVIDIA container toolkit, disk), `docker load`, copy models, create volumes (LUKS optional), generate TLS cert + admin bootstrap token, apply host firewall egress-drop rules, start stack, run health check + egress self-test.
- [ ] P3.5.4 `upgrade.sh`: verify, load new images, DB migration with backup, rollback on failure.
- [ ] P3.5.5 Model import bundles (signed) for new base/teacher models (FR-17).
- [ ] P3.5.6 Test: fresh Ubuntu GPU machine with network disabled → install → full run → egress test 0 packets.
- **Done when:** install-from-bundle test passes twice from scratch.

### P3.6 Validation on the "AI for Good" dataset
- [ ] P3.6.1 Finalise the demo corpus (P0.1.4) and licences.
- [ ] P3.6.2 Build the **gold set** (100–200 items; ≥ 20 % unanswerable; Mode C items with reference JSON); have the domain advisor verify a sample.
- [ ] P3.6.3 Run full pipeline 3× with different seeds; report mean ± CI for base, base+RAG, adapter, adapter+RAG.
- [ ] P3.6.4 Error analysis: categorise 50 failures (retrieval, knowledge, format, hallucination) → improvements or documented limitations.
- [ ] P3.6.5 Usability test with 3 non-ML participants (task script, time on task, SUS questionnaire, observed issues) → fix top issues.
- **Done when:** ≥ 12 % relative improvement over base on the gold set with CI lower bound > 0, or a clearly reported result with analysis.

### P3.7 Business case validation
- [ ] P3.7.1 ≥ 5 discovery interviews (hospital research IT, clinicians, energy operators); interview guide + notes (no confidential info in the public repo).
- [ ] P3.7.2 Pricing hypothesis + cost model (hardware tiers, licence per GPU server/year, pilot package).
- [ ] P3.7.3 Competitive landscape (cloud fine-tuning platforms, on-prem LLM platforms, consultancies) and differentiation.
- [ ] P3.7.4 Compliance one-pager: data flow, no-egress evidence, GDPR support statement (to be reviewed by a DPO).
- [ ] P3.7.5 Pitch deck + live demo script + backup recorded demo.

### P3.8 Hardening for the final demo
- [ ] P3.8.1 Bug bash; fix all P0 bugs.
- [ ] P3.8.2 Container hardening: non-root, read-only FS, dropped capabilities; image CVE scan (Trivy) report.
- [ ] P3.8.3 Documentation: install runbook, user guide (screenshots), troubleshooting, architecture overview.
- [ ] P3.8.4 Tag `v0.3.0` (R3) with signed bundle and release notes.

---

## Phase 4 — Enterprise (after the semester) → R4

Ordered by expected pilot value; each item becomes its own epic.

1. **P4.1 DPO/ORPO alignment (FR-20).** Synthesise chosen/rejected pairs via factual perturbation and format corruption; TRL `DPOTrainer`/`ORPOTrainer` after SFT; add as optional stage in HPO; must reduce hallucination without gate regressions.
2. **P4.2 Multi-GPU HPO.** Ray Tune + ASHA, one trial per GPU; DDP/FSDP for larger students (32–70B); NCCL timeout tuning + watchdog integration.
3. **P4.3 Multi-LoRA gateway (FR-21 full).** Routing by workspace/API key, quotas, usage metering, adapter hot-swap, canary deployments.
4. **P4.4 Enterprise identity.** RBAC; PKI/smartcard (CAC) via mutual TLS; LDAP/AD integration; SSO only where networks allow.
5. **P4.5 Drift retraining (FR-11).** Watched folders/buckets (S3-compatible MinIO); corpus-shift detection (> 15 %); incremental retraining; champion/challenger shadow evaluation.
6. **P4.6 Security hardening (NFR-SEC-03, NFR-COMP-01).** Encrypted scratch with key destruction on completion; STIG-hardened UBI/RHEL or Ubuntu Pro FIPS images; zero critical CVEs gate in CI; data-diode-ready split archives with forward error correction.
7. **P4.7 Merged/quantised exports (FR-08 full).** Merge LoRA into base; AWQ/GPTQ/GGUF export; re-run gates on the quantised model.
8. **P4.8 Postgres + HA**, backup/restore, metrics (Prometheus, local only), larger tiers (70B teacher, 10k–100k pages SLAs).
9. **P4.9 Certifications & pilots.** ISO 27001-aligned processes, customer security questionnaires, 2 pilot deployments (PRD Phase-3 exit criterion of the original roadmap).

---

## 10. Testing strategy

| Layer | What | Where | When |
|---|---|---|---|
| Unit | Pure functions: chunker, recognisers, policy, splits, metrics, gate logic, memory model, planner | CPU | Every PR |
| Property | Chunk coverage, split disjointness by document, placeholder consistency | CPU (hypothesis) | Every PR |
| Integration | API + DB + worker with fake LLM backend; stage contracts (manifests) | CPU | Every PR |
| **Toy e2e** | Full pipeline with 0.5B models on 5 small docs | CPU, CI | Every PR (≤ 15 min) |
| **Egress (toy)** | Toy e2e inside netns with drop+count firewall | CPU, CI (Linux runner) | Every PR to `main` |
| GPU e2e | `ref-small` full pipeline | GPU server | Nightly/manual, before each release |
| Benchmarks | `ref-500`, `ref-1000p` timings and quality | GPU server | Each milestone |
| Chaos | kill -9, simulated OOM, disk full, power loss | GPU server | R2, R3 |
| Security | secret scan, dependency audit (`pip-audit`), image scan (Trivy), log PII scan | CI | Every PR / release |
| UI | vitest components; Playwright golden path (toy) | CI | Every PR |
| Usability | Moderated sessions | In person | R3 |

Test data rules: only synthetic or licence-cleared fixtures in the repo; PII fixtures are synthetic.

---

## 11. CI/CD and release process

- **Branching:** trunk-based; short-lived feature branches `feat/P1.4-quote-first-qa`; PRs require 1 review + green CI; squash merge; Conventional Commits.
- **CI jobs (GitHub Actions):** `lint` (ruff, mypy, eslint), `test-unit`, `test-integration`, `e2e-toy`, `egress-toy` (Linux runner with sudo for nftables/netns), `web-build`, `security` (detect-secrets, pip-audit). Cache uv and npm.
- **GPU jobs:** `make gpu-e2e` on the GPU server, triggered manually (or via a self-hosted runner if the institute allows it). Results posted to the PR as a comment/artifact.
- **Versioning:** SemVer; `v0.1.0` = R1, `v0.2.0` = R2, `v0.3.0` = R3.
- **Release checklist:** all exit criteria; benchmarks updated; egress test on GPU; bundle built, signed, verified, installed on a clean machine; release notes; PRD/plan updated.

---

## 12. Team workflow and workstreams

Two parallel tracks keep the founders unblocked; swap reviews across tracks.

| Track | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| **Track A — Data & Product** | P1.1 ingest, P1.2 sanitise, P1.4 synthesis, P1.9 UI | P2.5 synthesis v2, P2.6 gold set + RAG, P2.7 reports | P3.1 UI, P3.4 Mode C, P3.6 validation, P3.7 business |
| **Track B — Models & Platform** | P1.3 LLM layer, P1.5 training, P1.6 eval, P1.7 serve, P1.8 worker, P1.10 containers/egress | P2.1 profiler, P2.2 OOM runner, P2.3 replay, P2.4 HPO, P2.6 gates engine, P2.8 watchdog | P3.2 auth, P3.3 serving, P3.5 bundle/installer, P3.8 hardening |

Assign tracks in P0.1. Coding agents take well-specified sub-steps (e.g., a recogniser, a loader, a UI page) with tests; humans own interfaces, experiments and reviews.

**Cadence:** Monday planning (pick issues), daily async update, Friday demo + benchmark log, end-of-phase retro.

---

## 13. Decision gates and risk checkpoints

| When | Check | If it fails |
|---|---|---|
| End of week 2 | GPU access confirmed; 8B QLoRA + 14B AWQ teacher run | Drop to 3–4B student / 7B teacher; push for compute; adjust PRD targets. |
| Week 5 | Synthesis grounding ≥ 85 % on audit | Stricter quote-first prompts, larger teacher, lower τ temporarily; more judge weight. |
| End of Phase 1 | Measured time-to-model extrapolates to ≤ 8 h | Reduce trials/fidelity; smaller teacher; cap synthetic pairs; update PRD. |
| Week 8 | Tuned vs. default ≥ 12 % on domain hold-out | Add warm starts, widen space (rsLoRA/LoRA+), more data diversity; report honestly if not met. |
| Week 9 | Adapter beats base+RAG on gold set | Reposition: "adapter + RAG" as product default; make RAG part of serving. |
| Week 11 | Demo dataset + gold set finalised | Fall back to energy (NRC) corpus which is public-domain. |
| Week 13 | Install-from-bundle works on clean machine | Freeze features; all effort on packaging. |

---

## 14. Appendices

### Appendix A — Offline / telemetry environment (set by `zllf.core.offline` and in all Dockerfiles)

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY=1
HF_HUB_DISABLE_IMPLICIT_TOKEN=1
VLLM_NO_USAGE_STATS=1
VLLM_DO_NOT_TRACK=1
DO_NOT_TRACK=1
RAY_USAGE_STATS_ENABLED=0
WANDB_MODE=disabled
WANDB_DISABLED=true
GRADIO_ANALYTICS_ENABLED=False
TOKENIZERS_PARALLELISM=false
PIP_NO_INDEX=1
```

Verify each against the pinned library versions in P1.10 — names change between releases. The egress test is the real guarantee; env vars are defence in depth.

### Appendix B — Example goal profile (`configs/profiles/mode_a.yaml`)

```yaml
mode: A
display_name: "Deep domain Q&A & troubleshooting"
synthesis:
  tasks: {qa: 0.6, troubleshooting_dialogue: 0.15, summarization: 0.1, unanswerable: 0.15}
  teacher_temperature: 0.1
  quotes_per_chunk: [1, 5]
  entailment_threshold: 0.92
splits: {train: 0.70, val: 0.15, test: 0.15, by: document}
hpo:
  objective_weights: {domain_val: 1.0, forgetting: 2.0, hallucination: 1.5}
  epsilon_forget: 0.03
  min_resource_fraction: 0.25
  reduction_factor: 3
gates:
  domain_rel_improvement_min: 0.12
  forgetting_max_drop_points: 3.0
  abstention_min: 0.90
  fabrication_max: 0.05
  canary_extractions_max: 0
  raw_pii_hits_max: 0
```

### Appendix C — Time-budget planner (sketch)

```
T_budget  = deadline − now
T_fixed   = T_ingest + T_sanitize + T_synth(est. from tokens/s) + T_eval + T_package + slack(5 %)
T_hpo     = T_budget − T_fixed − T_final_fit
cost(trial) ≈ tokens_per_trial / train_tokens_per_s(profile, rank) + eval_overhead
n_trials  = clamp(floor(T_hpo / E[cost(trial) under ASHA]), 4, 32)
if n_trials < 4: reduce fidelity (data fraction) → then reduce synthetic set size → then warn user
re-plan after every trial with measured costs
```

### Appendix D — Definition of Done (every step)
- Code + tests (toy-profile test for GPU code) + docs updated.
- No new network calls; offline guard tests pass.
- Logs contain no raw document text or PII by default.
- Manifest/artefact contract respected; idempotent re-run verified.
- Reviewed by the other founder.
