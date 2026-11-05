# Product Requirements Document — Zero-Leak LoRA Factory (ZLLF)

| | |
|---|---|
| **Product** | Zero-Leak LoRA Factory (ZLLF) — working title; final name TBD (repo: `kelpod`) |
| **Founders / owners** | Kevin Antony, Arsha Fahima Tajdeen |
| **Context** | Frontier Systems 2026/2027 — startup project |
| **Status** | Draft v0.2 — consolidates the original ZLLF PRD draft and the course pitch |
| **Last updated** | 2026-11-05 |
| **Companion document** | [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md) — architecture and step-by-step build plan |

> **Reading guide.** Requirement IDs (`FR-xx`, `NFR-xx`) are stable and are referenced from the development plan. Every requirement carries a priority (**P0** = must ship, **P1** = should ship, **P2** = later) and a target release (**R1–R4**, see §6). Numeric thresholds are *configurable defaults*, not hard-coded constants, unless stated otherwise.

---

## 1. Summary

Hospitals, clinical research centres and critical-infrastructure operators hold large volumes of sensitive, unstructured documents — patient notes, study reports, maintenance logs, fault-code manuals. They need language models that understand their exact terminology and output formats, but they cannot send data to cloud APIs and they do not employ ML engineers.

**ZLLF is a fully offline software appliance that turns a folder of documents into a tested, deployable, domain-specialised LoRA adapter — with no code and no network connection.** A domain expert drops in documents, picks a goal ("answer troubleshooting questions", "extract biomarkers as JSON"), and the next morning receives a scorecard and a one-click local chat/API endpoint. Inside, an autonomous pipeline parses and redacts the documents, uses a local *teacher* model to synthesise grounded training data, tunes and trains LoRA adapters within the available GPU budget, blocks any adapter that hallucinates, forgets general skills or memorises private data, and packages the winner with a full audit trail.

---

## 2. Problem

Organisations in regulated or isolated environments face three structural barriers to adopting modern LLMs:

1. **Zero cloud egress.** GDPR, HIPAA, national health-data law, KRITIS/NIS2 obligations and internal security policy forbid sending data to commercial LLM APIs or cloud fine-tuning platforms.
2. **No ML talent on site.** IT administrators, clinicians and plant engineers cannot curate datasets, configure training runs, tune LoRA hyperparameters or detect catastrophic forgetting. External contractors take months and need data access the organisation cannot grant.
3. **Offline RAG alone is fragile on dense jargon.** Retrieval works when the base model understands the vocabulary. It degrades with non-standard acronyms, codewords, shorthand clinical notation and strict output schemas the base model was never trained on.

**Value hypothesis.** A domain-adapted small model (7–14B parameters) produced *on site*, *automatically*, and *with verifiable safety gates* is (a) more accurate on in-domain questions and structured extraction than the same base model with or without RAG, and (b) deployable by non-ML staff within one night on a 1–2 GPU workstation.

> ⚠️ The hypothesis must be tested against a **RAG baseline**, not only the bare base model — the pitch's own argument is that RAG is insufficient, so the demo must show it (see §11).

---

## 3. Goals and non-goals

### 3.1 Product goals

| ID | Goal | Measured by |
|---|---|---|
| G1 | **Zero data leakage** — no outbound network traffic across ingestion, training and serving. | 0 outbound packets in the egress test (NFR-SEC-01). |
| G2 | **Zero-code** — a non-ML user goes from documents to a deployed endpoint without editing config or code. | Usability test: 3/3 non-ML testers finish the workflow unaided. |
| G3 | **Overnight time-to-model** — < 8 h from 1,000 raw PDF pages to a deployed endpoint on a 1–2 GPU workstation. | Wall-clock on the reference benchmark corpus (§11). |
| G4 | **Measurable accuracy gain** — ≥ 12 % relative improvement over the untuned base model on a human-verified domain test set. | Gold-set evaluation (§11). |
| G5 | **Safe by default** — no adapter is deployable unless it passes all four gates (domain accuracy, forgetting, hallucination, memorisation). | Gate results in the audit report. |
| G6 | **Auditable** — every adapter ships with a reproducible lineage record (inputs, seeds, hyperparameters, metrics). | A-SBOM present and verifiable for 100 % of adapters. |

### 3.2 Course goals (Frontier Systems 2026/27)

| Milestone | Target | Deliverable |
|---|---|---|
| Intermediary Goal 1 — Core Offline MVP | Weeks 3–6 | Offline container ingests 500 sample PDFs, generates synthetic QA via local vLLM, trains a working QLoRA adapter on one GPU with zero outbound calls. |
| Intermediary Goal 2 — AutoML & Safety Gates | Weeks 7–10 | Automated HPO, four-gate evaluation, forgetting prevention, "Model Readiness Scorecard" PDF. |
| End Goal — Full Appliance & Pilot Demo | End of semester | Zero-code web UI (drag-and-drop → one-click deploy), validated on a real "AI for Good" healthcare or sustainability dataset showing ≥ 12 % improvement over the base model, plus a validated B2B business case. |

### 3.3 Non-goals (v1 / semester)

- **Not a clinical decision-support or medical device.** v1 outputs are for information retrieval, drafting and extraction with human review. Anything intended to drive diagnosis or treatment would fall under EU MDR and is out of scope.
- **No full fine-tuning or pre-training** — LoRA/QLoRA adapters only.
- **No multi-node clusters** in the semester — single machine, 1–2 GPUs (multi-GPU HPO is R4).
- **No defence/classified deployments in the semester.** The original draft targeted SCIFs; the beachhead is now healthcare and clean energy (matches the "AI for Good" framing). Defence remains a later market (R4+), and the security architecture is kept compatible with it.
- **No smartcard/CAC/PKI login, DISA STIG hardening or FIPS images** in the semester (R4).
- **No hosted/cloud version** of the product.
- **No training on real patient data during the course.** Use public or synthetic datasets only (§13).

---

## 4. Target market and personas

### 4.1 Segments

| Priority | Segment | Why now |
|---|---|---|
| Beachhead | **Hospitals, university clinics, biobanks, clinical research organisations** (EU, starting with Germany) | Large unstructured corpora; strict GDPR / health-data rules; no ML teams; strong "AI for Good" story. |
| Second | **Energy and critical-infrastructure operators** (grid, renewables, nuclear, water) | Decades of maintenance logs and proprietary fault codes; OT networks are already isolated. |
| Later (R4+) | **Defence, government, sovereign finance** | Highest willingness to pay; needs accreditation, PKI, STIG — too heavy for the semester. |

**Buyer vs. user.** The *buyer* is the CIO/CISO or research IT lead (cares about compliance, audit trail, hardware cost). The *user* is the domain expert (cares about answer quality and simplicity). The *operator* is the local sysadmin (cares about installs, updates and not debugging CUDA).

### 4.2 Personas

| Persona | Environment | Core pain point | Success metric |
|---|---|---|---|
| **Clinical Research Lead** ("Dr. Meyer") | University hospital research IT, isolated network segment | Wants structured biomarker tables from unstructured notes without GDPR exposure; no ML staff. | Strict JSON-schema compliance; verifiable protection against memorisation of patient data. |
| **Plant / OT Engineer** ("Jonas") | Energy operator, air-gapped OT network | Needs an assistant that knows 30 years of scanned maintenance logs and proprietary PLC fault codes. | Exact citations; zero hallucination on safety-critical procedures. |
| **Enclave Sysadmin** ("Priya") | On-prem data centre behind a firewall or data diode | Maintains GPU servers; cannot debug Python/CUDA dependency conflicts. | Zero external calls; deterministic start-up; VRAM limits always respected; signed offline updates. |
| **Domain Analyst** (defence / government, R4+) | Secure facility | Needs an LLM that understands internal codewords without waiting months for a contractor. | Time from document drop to usable endpoint < 12 h. |

---

## 5. Use cases and user stories

The UI asks for a *goal*, never for hyperparameters. Each goal maps to an internal profile (synthesis mix, HPO objective weights, gate thresholds).

| Mode | Goal (as shown in UI) | Optimises for | Example |
|---|---|---|---|
| **A** | Deep domain Q&A & troubleshooting | Factual recall, citation fidelity, abstaining when unsure | "What is the reset procedure after fault code E-4711?" |
| **B** | Nomenclature, style & report drafting | Internal tone, formatting, acronym mastery | "Draft a shift-handover note in our standard format." |
| **C** | Structured extraction | Deterministic JSON output that validates against a user schema | "Extract all biomarkers with value, unit and date from this note." |

**User stories (P0 unless marked).**

- *US-1* As a domain expert, I can create a workspace, choose a pre-loaded base model and drag-and-drop a folder of documents.
- *US-2* As a domain expert, I can preview parsed text and the proposed redactions, and approve or adjust them before anything is used for training.
- *US-3* As a domain expert, I choose one goal (A/B/C). For Mode C I upload or paste a JSON schema (or pick a template).
- *US-4* As a domain expert, I start a job and can close the browser; the job runs to completion unattended and survives a reboot.
- *US-5* As a domain expert, I see live progress per stage with a time estimate, and I can cancel.
- *US-6* As a domain expert, I receive a Model Readiness Scorecard comparing the new adapter with the base model (and the RAG baseline) on my domain, with pass/fail badges.
- *US-7* As a domain expert, I click **Deploy** and get a local chat playground plus an OpenAI-compatible API endpoint.
- *US-8* As a compliance officer, I download a PDF and JSON audit report with document hashes, dataset statistics, trials, metrics and gate decisions.
- *US-9* As a sysadmin, I install the appliance from a signed offline bundle, with no internet access, using one command.
- *US-10* As a domain expert, I can write or import a small "gold" set of my own questions and answers to evaluate against (strongly recommended; see §11).
- *US-11 (P1)* As a domain expert, I can compare two adapters side by side in the playground.
- *US-12 (P2)* As a sysadmin, I can point the appliance at a watched folder so it retrains when the corpus changes significantly.

---

## 6. Release plan

| Release | Timing | Theme | Scope summary |
|---|---|---|---|
| **R1 — Core Offline MVP** | Course weeks 3–6 | Pipeline works end-to-end, offline | Offline container; PDF/DOCX/TXT/MD ingestion with OCR; basic PII masking; teacher-based QA synthesis with judge filter; single-GPU QLoRA with fixed defaults; basic evaluation; minimal UI or CLI; egress test. |
| **R2 — AutoML & Safety Gates** | Course weeks 7–10 | It picks good settings and refuses bad models | Hardware profiler; OOM-safe trial runner; Bayesian HPO with ASHA early stopping under a time budget; replay mixing; four-gate evaluation; RAG baseline; A-SBOM; Scorecard PDF. |
| **R3 — Zero-Code Appliance & Pilot Demo** | Course week 11 → end of semester | A non-ML user can do it all | Full web UI (workspaces, redaction preview, goal selection, progress, scorecard, playground, deploy); signed offline bundle installer; Mode C schema extraction; validated demo on an "AI for Good" dataset; business case. |
| **R4 — Enterprise** | After the semester | Pilots in real enclaves | DPO/ORPO alignment; multi-GPU Ray Tune; multi-LoRA serving gateway; RBAC with PKI/smartcard; drift-triggered retraining; scratch wiping; STIG/FIPS images; data-diode update channel; merged/quantised exports (GGUF/AWQ/GPTQ). |

---

## 7. User experience

### 7.1 Guided workflow (four steps)

1. **Project initialisation.** Log in to the local web console (local accounts in R3; smartcard/PKI in R4). Create a workspace, set a data-sensitivity label (Public / Internal / Confidential / Restricted), choose a pre-loaded base model.
2. **Corpus connection and sanitisation preview.** Upload files or point to a local/network folder (NFS/SMB in R3; S3-compatible MinIO in R4). The system parses everything and shows a side-by-side preview: original rendering ↔ extracted text with redaction highlights. The user approves the redaction policy (accept all / adjust entity types / add custom patterns).
3. **Objective selection.** Choose Mode A, B or C, an optional time budget ("ready by 07:00"), and optionally a gold question set. Advanced settings are hidden behind an "Expert" toggle (P2).
4. **Autonomous execution and scorecard approval.** The job runs unattended. The user gets a Model Readiness Scorecard: per-gate pass/fail badges, domain accuracy vs. base and vs. RAG, general-skill retention, hallucination/abstention rate, memorisation result, and 5–10 example answers side by side. **Deploy** is enabled only if all P0 gates pass; an administrator may override with a written justification that is recorded in the audit log.

### 7.2 Screens (R3)

| Screen | Key elements |
|---|---|
| Login | Local account; first-run setup wizard creates the admin. |
| Workspaces | List, create, status of latest run, deployed adapter. |
| Corpus | Upload/drop zone, folder picker, file table (type, pages, parse status, OCR flag, duplicate flag), parse errors. |
| Redaction preview | Document viewer with highlighted entities; entity-type toggles; custom regex; approve button. |
| Goal | Mode cards A/B/C; schema editor for C; time-budget picker; gold-set upload. |
| Run monitor | Stage timeline (Ingest → Sanitise → Synthesise → Train/HPO → Evaluate → Package), ETA, live logs (user-friendly), GPU utilisation/temperature, trial leaderboard, cancel. |
| Scorecard | Gate badges, metrics vs. baselines, example comparisons, download report (PDF/JSON), Deploy. |
| Playground | Chat with deployed adapter; citations (Mode A); JSON-validity indicator (Mode C); compare with base. |
| Models & deployments | Registered adapters, versions, active deployment, API key management, endpoint URL and curl example. |
| Admin | Hardware profile, storage usage, users, audit log, installed base/teacher models, update import. |

### 7.3 UX principles

- **No ML vocabulary on the default path.** "Training data examples", not "SFT pairs"; "Quality checks", not "gates"; "How much it remembers general skills", not "anchor loss".
- **Always show what the system will do with the data** before it does it (redaction preview, dataset preview with sample synthetic Q&A).
- **Every number in the scorecard links to an explanation** and to the underlying examples.
- **Fail loudly and helpfully.** Every error states what happened, whether data is safe, and what the user can do.

---

## 8. System overview (product view)

ZLLF is a state machine of six stages running inside the isolated perimeter. Every dependency — base model weights, teacher model, OCR and NER models, embedding and NLI models, Python wheels, CUDA libraries, benchmark datasets — is pre-packaged in a signed offline bundle.

```
 Documents ─▶ ① Ingest & Parse ─▶ ② Sanitise ─▶ ③ Synthesise (Teacher + Judge) ─▶ ④ AutoML Train (HPO)
                                     │ user approves                                         │
                                     ▼ redactions                                            ▼
              ⑥ Package & Serve ◀── ⑤ Evaluate (4 gates + baselines) ◀───────────── candidate adapters
                    │
                    ▼  Scorecard + A-SBOM + local chat/API endpoint
```

> The original draft described "six stages" but listed five. This PRD makes sanitisation its own stage (②) because it has a human approval step and its own requirements.

In the pitch these stages are described as *agents* (Ingestion & Privacy, Teacher, Judge, AutoML Trainer, Safety & Eval, Deployment). In the implementation each "agent" is a **deterministic stage worker** that uses local models as tools, with fixed inputs/outputs and checkpoints — not an open-ended autonomous agent loop. This is deliberate: reproducibility and auditability matter more than flexibility in regulated settings.

### 8.1 Stage specifications

**① Ingest & parse.** Accepts PDF (digital and scanned), DOCX, TXT, Markdown, HTML, XML, CSV, Parquet, TIFF/PNG/JPEG. Layout-aware parsing preserves headings, lists and tables. Scanned pages go through local OCR. Output: normalised documents with structure, page references and SHA-256 hashes. Documents are split into chunks along structural boundaries (section → paragraph → sentence fallback) with a configurable token target. Near-duplicates are removed with MinHash LSH.

**② Sanitise.** Local NER plus regex rule packs detect PII and sensitive markers (names, dates of birth, patient/case IDs, insurance numbers, addresses, phone/e-mail, IBAN, German and US national IDs, classification headers, custom patterns). The policy chooses per entity type: *keep*, *mask with typed placeholder* (`[PATIENT_NAME_1]`, consistent within a document), or *drop chunk*. The user approves the policy in the preview. Every redaction decision is logged (type, position, policy; never the original value).

**③ Synthesise.** A local teacher model (larger than the student, quantised) converts chunks into training data:
- *Taxonomy extraction* — glossary of entities, acronyms and concepts (feeds Mode B data and the scorecard's "terms learned" list).
- *Instruction data* — per mode: grounded Q&A with supporting quote and citation (A); rewrite/drafting/acronym tasks (B); extraction tasks with schema-valid targets (C); multi-turn troubleshooting dialogues (A, P1); short reasoning traces (P1).
- *Unanswerable questions* — plausible in-domain questions whose answer is *not* in the corpus, paired with a correct abstention (feeds hallucination training and the hallucination gate).
- *Judge loop* — every pair is checked (i) for exact presence of its supporting quote in the source chunk, (ii) by an NLI cross-encoder for entailment (default τ = 0.92), and (iii) by an LLM judge for answerability and format. Pairs below threshold are discarded and counted.
- *Preference pairs (R4)* — chosen/rejected pairs where the rejected answer contains an injected factual perturbation or format error, for DPO/ORPO.
- *Splits* — train / validation / test (default 70/15/15), split **by source document** so test questions are never about documents seen in training.

**④ AutoML training.** The hardware profiler sets the memory strategy (QLoRA 4-bit NF4 vs. 16-bit LoRA, batch size, gradient accumulation, gradient checkpointing, FlashAttention). The HPO controller searches θ = (r, α, η, M_target, p_drop, λ_wd, β_replay):

| Hyperparameter | Search space (default) |
|---|---|
| LoRA rank r | {4, 8, 16, 32, 64, 128} |
| LoRA alpha α | {8, 16, 32, 64, 128, 256} (or α/r ratio ∈ {0.5, 1, 2, 4}) |
| Peak learning rate η (cosine decay) | [1e-5, 5e-4], log-uniform |
| Target modules M_target | subsets of {q, k, v, o, gate, up, down}; presets: attention-only, MLP-only, all-linear |
| LoRA dropout p_drop | [0.0, 0.2] |
| AdamW weight decay λ_wd | [0.0, 0.1] |
| Replay ratio β_replay | [0.05, 0.30], lower bound scales up as corpus shrinks |

It minimises

```
L_AutoML(θ) = w1 · L_domain_val(θ)
            + w2 · max(0, L_gen_anchor(θ) − L_gen_base − ε_forget)
            + w3 · Φ_halluc(θ)

subject to  VRAM_peak(θ) ≤ 0.90 · VRAM_available
            T_wall(θ)    ≤ T_budget
```

using TPE (Tree-structured Parzen Estimator) sampling with ASHA-style successive halving: trials that are in the bottom fraction at 25 % of their budget are stopped. The weights (w1, w2, w3) come from the goal profile (Mode A weighs Φ_halluc higher; Mode C adds a schema-validity term). A **time-budget planner** divides T_budget across stages and sets the number of trials and the training fidelity accordingly (§8.2).

**⑤ Evaluate — four gates plus baselines.** The candidate is evaluated against the base model and a RAG baseline (base model + local embedding retrieval over the same sanitised chunks):

| Gate | What it checks | Default pass rule |
|---|---|---|
| G-1 Domain accuracy | Held-out synthetic test split *and* the user's gold set if provided: exact match (Mode C fields), ROUGE-L, embedding similarity, LLM-judge correctness | ≥ base + 12 % relative on primary metric; ≥ RAG baseline (warning only in R2, gate in R3) |
| G-2 Forgetting | Bundled offline suite: general knowledge, reasoning, math, instruction following, JSON-schema adherence, coding (small subsets) | No suite drops more than ε_forget (default 3 points absolute) vs. base |
| G-3 Hallucination | Held-out unanswerable domain questions + adversarial prompts | Abstention rate ≥ 90 %; fabrication rate ≤ 5 % |
| G-4 Memorisation | Random canary strings inserted in training data; prefix-completion probes; verbatim-overlap scan of outputs against the raw (pre-redaction) PII set | 0 canaries extracted; 0 raw PII strings emitted |

Mode C additionally requires ≥ 98 % schema-valid outputs.

**⑥ Package and serve.** The winning adapter is exported as SafeTensors (< 500 MB) with its config, tokenizer reference, the audit report (JSON + PDF) and an A-SBOM (CycloneDX ML-BOM). It is registered in the local model registry and, on **Deploy**, hot-loaded into a local vLLM server exposing an OpenAI-compatible API. Merged and quantised exports (GGUF/AWQ/GPTQ) are R4.

### 8.2 Time budget (reference: 1,000 pages, single 48 GB GPU, 8 h)

| Stage | Budget | Notes |
|---|---|---|
| Ingest + OCR + sanitise | 0.5 h | Mostly CPU; OCR on GPU if available. |
| Synthesis + judge | 1.5 h | ~2,000 chunks → ~10,000 candidate pairs → ~6,000 kept. Teacher runs alone on the GPU. |
| HPO + final training | 4.5 h | ~8–12 trials at reduced fidelity (ASHA), then one full-fidelity run of the best configuration. |
| Evaluation (4 gates + baselines) | 1.0 h | Batched inference via vLLM. |
| Packaging + report | 0.25 h | |
| Slack | 0.25 h | |

These are planning estimates and must be validated in R1/R2 (see `DEVELOPMENT_PLAN.md` § benchmarks). If the measured throughput does not fit, the planner reduces trials before reducing training data.

---

## 9. Functional requirements

IDs FR-01 to FR-11 keep the numbering of the original draft. FR-12 onward are new.

| ID | Module | Requirement | Pri | Rel | Acceptance criteria |
|---|---|---|---|---|---|
| **FR-01** | Air-gap ingestion | Ingest local directories containing PDF, DOCX, TXT, CSV, Parquet, Markdown, HTML, XML and images without any DNS resolution or HTTP(S) call. | P0 | R1 | 500-PDF corpus ingested inside a container with `--network none`; egress monitor records 0 packets; unsupported files are listed with a reason, not silently skipped. |
| **FR-02** | Offline OCR | Extract text and table structure from scanned PDFs and images with a bundled OCR pipeline, ≥ 97 % character accuracy on standard technical documents. | P0 | R1 (basic) / R2 (accuracy target) | CER ≤ 3 % on a 20-page labelled scanned test set; tables exported with row/column structure. |
| **FR-03** | PII / sensitive-marker scrubber | Configurable filters to mask or drop names, IDs, dates of birth, contact data, classification headers and user regexes before synthesis; preview and approval step. | P0 | R1 (rules + NER) / R3 (preview UI) | Recall ≥ 95 % on a labelled synthetic-PII test set (DE + EN); no un-approved corpus reaches stage ③. |
| **FR-04** | Synthetic QA engine | Generate instruction pairs (QA, summarisation, extraction, unanswerables) from chunks; drop pairs below the entailment threshold via the judge loop. | P0 | R1 (QA + judge) / R2 (all task types) | Every kept pair has a source chunk ID and a verbatim supporting quote; ≥ 92 % grounding score on a 100-pair human audit. |
| **FR-05** | Replay mixing | Maintain a read-only bundled general-instruction dataset; blend fraction β_replay into every training run. | P0 | R2 | Training manifest records β_replay and replay sample IDs; G-2 passes on the reference corpus. |
| **FR-06** | Hardware profiler | On job start, detect GPUs (VRAM, compute capability, BF16/FP8 support, NVLink/PCIe), RAM and disk; choose QLoRA vs. LoRA, batch size, accumulation, checkpointing, attention kernel; run a max-sequence-length dry run before HPO. | P0 | R2 | Profile stored with the run; dry run completes; no OOM in 20 consecutive reference runs. |
| **FR-07** | Bayesian HPO loop | Multi-fidelity search over r, α, η, M_target, p_drop, λ_wd, β_replay with TPE and ASHA, stopping weak trials within the first 25 % of their budget, under VRAM and wall-clock constraints. | P0 | R2 | Tuned adapter beats the fixed-default adapter by ≥ 12 % relative on the domain hold-out of the reference corpus (Phase-2 exit criterion of the original roadmap). |
| **FR-08** | Adapter export | Export the winner as a standalone LoRA adapter (SafeTensors, < 500 MB); optionally merged + quantised (GGUF / AWQ / GPTQ). | P1 | R1 (adapter) / R4 (merged) | Exported adapter loads in vLLM and in HF PEFT and reproduces eval scores within ±1 point. |
| **FR-09** | A-SBOM lineage | Every run produces an immutable JSON + PDF audit report: SHA-256 of all inputs, redaction stats, synthetic dataset stats, all trials, loss curves, gate results, software/model versions and hashes, seeds. | P0 | R2 (JSON) / R3 (PDF) | Report validates against its JSON Schema; CycloneDX ML-BOM validates; re-running with the same seed and inputs reproduces metrics within tolerance. |
| **FR-10** | Zero-code web UI | Local browser UI: drag-and-drop/folder selection, redaction preview, goal selection, progress, scorecard, deploy, playground. | P0 | R1 (minimal) / R3 (full) | Usability test: 3 non-ML users complete the workflow without help. |
| **FR-11** | Continual drift retraining | Watch local folders; when corpus volume changes by > 15 %, trigger incremental retraining and champion/challenger shadow evaluation. | P2 | R4 | Challenger only promoted if it beats champion on all gates. |
| **FR-12** | Job control & resume | Jobs are resumable after crash/reboot from the last completed stage or checkpoint; users can cancel; cancelled jobs clean up scratch data. | P0 | R1 (stage-level) / R2 (checkpoint-level) | Kill -9 during training → job resumes from the last checkpoint ≤ 15 min old. |
| **FR-13** | RAG baseline | Build a local retrieval index over sanitised chunks and evaluate base+RAG and adapter+RAG alongside the adapter. | P0 | R2 | Scorecard shows all four variants: base, base+RAG, adapter, adapter+RAG. |
| **FR-14** | Gold evaluation set | User can import (CSV/JSONL) or write questions with reference answers; used in G-1 and reported separately from synthetic metrics. | P0 | R2 (import) / R3 (UI editor) | Scorecard shows gold-set metrics with 95 % bootstrap confidence intervals. |
| **FR-15** | Serving & playground | One-click deploy to a local OpenAI-compatible endpoint; chat playground with citations (Mode A) and JSON validation (Mode C); API keys. | P0 | R1 (endpoint) / R3 (playground) | `curl` against the endpoint returns adapter output; playground works with network disabled. |
| **FR-16** | Local authentication | Local user accounts with roles (admin, user); session auth; audit log of logins and actions. | P1 | R3 | Unauthenticated API calls return 401; actions appear in audit log. |
| **FR-17** | Model management | Pre-installed base and teacher models; admin can import additional models from a signed bundle; licence shown per model. | P1 | R3 | Import rejects unsigned/tampered bundles. |
| **FR-18** | Mode C schema extraction | Accept a JSON Schema; synthesise extraction tasks; constrained decoding at inference; validation in eval. | P0 | R3 | ≥ 98 % schema-valid outputs on the test split. |
| **FR-19** | Time-budget planner | Given a deadline or budget, allocate time across stages and scale HPO trials/fidelity; show ETA. | P1 | R2 | Runs finish within budget + 10 % on the reference corpus. |
| **FR-20** | Preference alignment | DPO/ORPO stage using synthesised chosen/rejected pairs. | P2 | R4 | Hallucination rate reduced vs. SFT-only adapter, no gate regressions. |
| **FR-21** | Multi-LoRA serving | Serve multiple adapters (one per workspace) on one shared base model concurrently. | P1 | R3 (basic, vLLM) / R4 (gateway with routing/quotas) | 5 adapters served concurrently on one GPU. |

---

## 10. Non-functional requirements

### 10.1 Security and privacy

| ID | Requirement | Pri | Rel |
|---|---|---|---|
| **NFR-SEC-01 Zero egress** | The stack runs with external network interfaces disabled. Containers run on an internal-only network. All telemetry in dependencies is disabled by environment and config (Hugging Face, PyTorch, vLLM, Ray, Weights & Biases, Gradio, etc.). An automated egress test (packet capture + firewall drop counters) runs in CI and on every release. | P0 | R1 |
| **NFR-SEC-02 Signed offline updates** | Software, models and patches are delivered as deterministic split-archive tarballs verified by Ed25519 signatures offline before import; compatible with one-way data diodes (no handshake needed). | P0 | R3 (signing) / R4 (diode-ready split archives) |
| **NFR-SEC-03 Scratch sanitisation** | On job completion or cancellation, intermediate caches, temporary datasets and checkpoints that are not part of the final artefact are deleted; R4 adds zero-fill/crypto-erase on encrypted scratch volumes. | P1 | R2 (delete) / R4 (wipe) |
| **NFR-SEC-04 Data at rest** | Workspace data stored on an encrypted volume (LUKS) in appliance installs; file permissions restrict access to the service user. | P1 | R3 |
| **NFR-SEC-05 Least privilege** | Containers run as non-root, read-only root filesystem where possible, no added Linux capabilities except GPU access. | P1 | R3 |
| **NFR-SEC-06 No raw PII in logs** | Logs, reports and the audit trail never contain raw redacted values. | P0 | R1 |
| **NFR-SEC-07 Supply chain** | All wheels/images are pinned by hash; SBOM for software; CVE scan at release with zero critical CVEs. | P1 | R3 (scan) / R4 (zero-critical policy) |

### 10.2 Reliability and determinism

| ID | Requirement | Pri | Rel |
|---|---|---|---|
| **NFR-REL-01 OOM immunity** | Dry-run forward/backward pass at max sequence length before HPO. On OOM in a trial: halve micro-batch, double accumulation, enable gradient checkpointing (and Unsloth kernels where supported), retry — without crashing the controller. | P0 | R2 |
| **NFR-REL-02 Power-loss checkpointing** | Atomic checkpoints at least every 15 min; job resumes automatically after restart. | P0 | R2 |
| **NFR-REL-03 Determinism** | Fixed seeds per run, recorded in the A-SBOM; deterministic data splits; container start-up is deterministic (no downloads, no compilation at runtime). | P0 | R1 |
| **NFR-REL-04 Thermal / hang watchdog** | NVML watchdog monitors temperature and power; lowers power limits instead of aborting; extends collective-communication timeouts on multi-GPU. | P1 | R2 (monitor) / R4 (multi-GPU) |
| **NFR-REL-05 Idempotent stages** | Each stage reads inputs and writes outputs to a run directory with a manifest; re-running a completed stage with the same inputs is a no-op. | P0 | R1 |

### 10.3 Performance

| ID | Requirement | Target |
|---|---|---|
| NFR-PERF-01 | Time-to-model, 1,000 pages, 1× 48 GB GPU | < 8 h (stretch: < 6 h) |
| NFR-PERF-02 | Ingestion throughput, digital PDF, CPU | ≥ 5 pages/s |
| NFR-PERF-03 | Serving latency, 8B model + adapter, single user | first token < 1 s; ≥ 30 tokens/s |
| NFR-PERF-04 | UI responsiveness | All pages load < 1 s locally; progress updates ≤ 5 s delay |

### 10.4 Usability, operability, compliance

- **NFR-USE-01** Default path requires no ML knowledge and no terminal use after installation.
- **NFR-OPS-01** One-command install from the offline bundle; one-command upgrade preserving data; health endpoint and disk-space checks.
- **NFR-OPS-02** Structured logs (JSON) with run and stage IDs; UI shows a user-friendly subset.
- **NFR-COMP-01 (R4)** Container images built on hardened (DISA STIG) RHEL/UBI or Ubuntu Pro FIPS 140-3 bases with zero critical CVEs at release.
- **NFR-COMP-02** Documentation supports customers' GDPR processes: data-flow description, record of processing template, statement that no data leaves the machine. (Not legal advice; validate with a data-protection officer during pilots.)
- **NFR-COMP-03** Model and dataset licences are tracked per artefact and shown in the UI and the A-SBOM.

---

## 11. Success metrics and evaluation methodology

### 11.1 Core product metrics

| Metric | Target | How measured |
|---|---|---|
| Zero data leakage | 0 outbound packets | Egress test: run full pipeline in an isolated network namespace with a logging firewall; count packets on all non-loopback interfaces. |
| Time-to-model | < 8 h for 1,000 pages on 1–2 GPUs | Reference corpus run, wall-clock from upload to endpoint ready. |
| Accuracy gain | ≥ 12 % relative over base on gold set | See §11.2. |
| Grounding of synthetic data | ≥ 92 % | Human audit of 100 randomly sampled kept pairs: is the answer fully supported by the cited quote? |
| Reliability | 0 OOM crashes | 20 consecutive reference runs across 2 hardware profiles. |
| Usability | 3/3 testers complete unaided | Moderated test with non-ML users (e.g., medical students, engineers). |

### 11.2 Avoiding circular evaluation

Measuring the adapter only on questions generated by the same teacher that produced its training data rewards learning the teacher's style, not domain knowledge. Therefore:

1. **Gold set.** For each demo domain, create 100–200 question/answer pairs **written or verified by a human** (founders plus a domain advisor), covering factual recall, troubleshooting, abstention (questions whose answer is not in the corpus) and, for Mode C, extraction with reference JSON.
2. **Document-level split.** Gold questions are drawn from documents in the test split only.
3. **Four variants.** Report base, base+RAG, adapter, adapter+RAG.
4. **Scoring.** Mode A/B: LLM-judge correctness with a fixed rubric (local judge model, temperature 0) plus 20 % human spot-check to estimate judge agreement. Mode C: field-level exact match and F1.
5. **Statistics.** 95 % bootstrap confidence intervals; claim the ≥ 12 % improvement only if the lower bound of the relative improvement is > 0 and the point estimate is ≥ 12 %.

### 11.3 Business metrics (course end)

- ≥ 5 discovery interviews with target users/buyers (hospital research IT, clinical researchers, energy operators).
- ≥ 1 letter of intent or pilot conversation.
- Validated pricing hypothesis (see §14).

---

## 12. Hardware tiers

| Tier | Dev / Demo (semester) | Tactical / Edge box | Departmental enclave | Sovereign enterprise rack |
|---|---|---|---|---|
| **Compute** | 1× 24–48 GB GPU (e.g., RTX 4090 / A6000 / L40S) or 1× A100/H100 80 GB | 2× RTX 6000 Ada or 2× L40S (96 GB total) | 4× H100 NVL / A100 (320–384 GB) | 8× H100/H200 SXM (640 GB–1.1 TB) |
| **RAM / storage** | 64–128 GB / 2 TB NVMe | 256 GB / 8 TB NVMe RAID-1 | 512 GB / 30 TB NVMe (FIPS) | 2 TB / 100 TB NVMe cluster |
| **Student models** | 7–9B (QLoRA) | 7–14B | up to 70B (QLoRA) / 32B (16-bit LoRA) | 70B LoRA, 405B QLoRA, vision-LLMs |
| **Teacher** | 14–32B, 4-bit AWQ | 14–32B AWQ | 70B FP8 | 70B–405B ensemble |
| **SLA** | < 8 h / 1,000 pages | < 8 h / 1,000 pages | < 6 h / 10,000 pages | < 12 h / 100,000 pages |

A **CPU-only "toy" profile** with tiny models (≈ 0.5B student and teacher) exists for development and CI. It checks plumbing, not quality.

Development machines without NVIDIA GPUs (e.g., Apple Silicon laptops) run only the toy profile; GPU stages run on a Linux GPU server.

---

## 13. Models, data and licensing

### 13.1 Model inventory (defaults; re-evaluate at R1 kickoff)

| Role | Default candidates | Licence note |
|---|---|---|
| Student base | Qwen3-8B; Llama-3.1-8B-Instruct (comparison model named in the pitch); Mistral-7B-Instruct | Prefer Apache-2.0 models for commercial use. Llama licence requires attribution and "Llama" naming for derivatives. |
| Teacher | Qwen-family 14B–32B instruct, AWQ 4-bit | Check that the licence allows using outputs to train other models. |
| OCR / layout | Docling with its bundled layout/table models and a local OCR engine (EasyOCR/Tesseract/RapidOCR) | MIT / Apache. |
| NER (PII) | Presidio + spaCy models (`en_core_web_lg`, `de_core_news_lg`) + custom recognisers | MIT. |
| NLI judge | DeBERTa-v3-large NLI cross-encoder | MIT. |
| Embeddings (RAG, similarity) | BGE-M3 (multilingual) | MIT. |
| Toy profile | Qwen2.5-0.5B-Instruct or similar | Apache-2.0. |

All models are stored with SHA-256 hashes and licence metadata in the bundle manifest.

### 13.2 Demo datasets ("AI for Good") — public or synthetic only

| Domain | Candidate sources (verify licence before use) |
|---|---|
| Healthcare — guidelines / Q&A (Mode A) | Public clinical guidelines (e.g., WHO guidelines, German AWMF guidelines where licensing allows), PubMed Central Open Access subset. |
| Healthcare — extraction (Mode C) | Synthetic patient notes generated for the project (no real patients); MTSamples-style transcriptions; credentialed datasets (e.g., MIMIC) only with a signed data-use agreement and never committed to the repo. |
| Clean energy / critical infrastructure (Mode A) | Public regulator documents, e.g., US NRC Licensee Event Reports and inspection reports (US-government works), public wind-turbine and grid maintenance manuals where licensing allows. |

**No dataset is committed to the public repository unless its licence allows redistribution.** Large or restricted datasets are fetched by a documented script on the internet-connected build machine and imported into the appliance offline.

---

## 14. Business model (summary)

- **Offer.** (1) Software licence for the offline container bundle, priced per GPU server per year; (2) pre-configured hardware appliance (hardware at cost + margin + licence); (3) paid pilot / onboarding package.
- **Customers.** University hospitals and clinical research institutes; energy and utility operators; later public sector and defence.
- **Why they pay.** Replaces months of contractor work and the need for an ML team; satisfies data-protection and security constraints out of the box; produces an audit trail that compliance teams can sign off.
- **To validate this semester.** Willingness to pay, procurement path (public tenders for hospitals?), required certifications, hardware already available on site.

---

## 15. Risks and mitigations

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R-1 | **Synthetic data poisoning** — teacher misreads a dense table and the student learns false facts. | High | Teacher temperature ≈ 0.1; quote-first generation (extract an exact supporting quote, then write the question); exact substring check; NLI filter (drop < 0.92); human audit of 100 pairs per release. |
| R-2 | **Catastrophic forgetting on small corpora** (e.g., 50 pages). | High | β_replay scales inversely with corpus size; rsLoRA / LoRA+ scaling; early stopping on the anchor benchmark; G-2 gate. |
| R-3 | **GPU thermal throttling / hangs** in non-datacentre rooms. | Medium | NVML watchdog; dynamic power caps; extended NCCL timeouts; checkpoint every 15 min. |
| R-4 | **Circular evaluation** inflates results. | High | Human gold set, document-level splits, RAG baseline, CIs (§11.2). |
| R-5 | **Time budget not met** (HPO too slow for 8 h). | Medium | Time-budget planner; reduced-fidelity trials; warm-start from per-mode priors; measure early in R1. |
| R-6 | **No reliable GPU access** for the team during the semester. | High | Secure an institute/course GPU in week 1; toy profile for daily development; keep jobs resumable. |
| R-7 | **Model/data licensing** blocks commercial use. | Medium | Apache-licensed defaults; licence registry; legal check before first pilot. |
| R-8 | **German-language performance** (teacher/student/NER weaker on German clinical text). | Medium | Choose multilingual models; German PII recognisers; include German documents in the reference corpus. |
| R-9 | **Regulatory scope creep** (MDR, EU AI Act obligations for healthcare uses). | Medium | Position v1 as information/extraction tool with human review; document intended use; seek advice before pilots. |
| R-10 | **Scope too large for one semester.** | High | Strict release scoping (§6); defence features deferred to R4; weekly scope review. |
| R-11 | **Public repository leaks sensitive business or customer material.** | Medium | No real data, credentials or customer names in the repo; consider making the repo private before pilots. |

---

## 16. Open questions

1. **Which GPU(s) are guaranteed for the semester?** (institute cluster, course allocation, own hardware) — determines model sizes and the realistic time-to-model target.
2. **Which "AI for Good" dataset is the end-of-semester demo?** Healthcare (Mode A + C) or clean energy (Mode A)? Recommendation: one primary (healthcare), one secondary.
3. **Who is the domain advisor** who verifies the gold set (clinician / engineer)?
4. **Product name** (kelpod vs. other).
5. **Is the ≥ 12 % target relative or absolute?** This PRD assumes *relative* on the gold set.
6. **Repository visibility** — should the repo stay public?
7. **Course requirements** for demo format, documentation and code licence.

---

## 17. Changes from the original ZLLF draft

- Beachhead market changed from defence/SCIF to healthcare and clean energy; defence moved to R4+.
- Sanitisation made its own stage; the pipeline is now six explicit stages (the draft said six but listed five).
- Added RAG baseline (FR-13), human gold set (FR-14), circular-evaluation safeguards (§11.2) and document-level splits.
- Added job control/resume (FR-12), serving/playground (FR-15), auth (FR-16), model management (FR-17), Mode C extraction (FR-18), time-budget planner (FR-19), DPO (FR-20), multi-LoRA serving (FR-21).
- Releases realigned to the course milestones; DPO, PKI/CAC, STIG and drift retraining deferred to R4.
- Added dev/demo hardware tier and a CPU toy profile; added model/data licensing section.
- Filled the missing persona, hardware-tier and roadmap tables.

---

## Glossary

| Term | Meaning |
|---|---|
| **LoRA / QLoRA** | Low-Rank Adaptation: train small low-rank matrices instead of all model weights. QLoRA does this on a 4-bit-quantised base model to save memory. |
| **Adapter** | The small set of trained LoRA weights that specialises a base model. |
| **Teacher / student** | The larger model that writes training data / the smaller model being adapted. |
| **Judge** | Checks each synthetic example against its source (NLI model + LLM). |
| **HPO, TPE, ASHA** | Hyperparameter optimisation; a Bayesian sampling method; an early-stopping scheme that stops weak trials early. |
| **Replay** | Mixing general-purpose training examples in to prevent forgetting. |
| **Catastrophic forgetting** | Loss of general abilities after narrow fine-tuning. |
| **Canary** | A random secret string planted in training data to test whether the model memorises data. |
| **A-SBOM** | AI Software Bill of Materials — a record of all data, models, code and settings used to produce an adapter. |
| **RAG** | Retrieval-Augmented Generation — retrieve relevant passages and pass them to the model at question time. |
| **Air gap / data diode** | Physical network isolation / hardware that allows data to flow in one direction only. |
