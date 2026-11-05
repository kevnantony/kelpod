# ZLLF HTTP API v1 (contract between `src/zllf/api` and `web/`)

Base path `/api`. JSON everywhere except uploads (multipart) and events (SSE). Auth: session
cookie `zllf_session` set by `POST /api/auth/login`; all other endpoints return `401` without it.
State-changing requests (POST/PUT/PATCH/DELETE, including login, setup and uploads) must send
the header `X-ZLLF-CSRF: 1`, otherwise `403 E_CSRF`; if an `Origin` header is present it must
match `Host`. Creates (workspace, documents, run) return `201`. `Run` and `Scorecard` also carry
`adapter_id: string | null`; scorecard metric keys are `"suite.metric"` plus `.ci_low`/`.ci_high`.
Event `type`s: progress, stage_started, stage_completed, stage_failed, awaiting_approval,
approved, resumed, cancelled, run_completed, train_metrics.
Errors: `{"error": {"code": "E_…", "message": "user-facing text"}}` with a 4xx/5xx status.
Timestamps: ISO-8601 UTC strings. IDs: strings.

## Types

```ts
type User = { id: string; username: string; role: "admin" | "user" };
type Workspace = {
  id: string; name: string; sensitivity: "public" | "internal" | "confidential" | "restricted";
  base_model: string; created_at: string;
  latest_run: RunSummary | null; deployed_adapter_id: string | null;
};
type DocumentInfo = {
  id: string; name: string; mime: string; size_bytes: number; sha256: string;
  status: "uploaded" | "parsed" | "skipped" | "failed"; pages: number | null; error: string | null;
};
type RunState = "queued" | "running" | "awaiting_approval" | "completed" | "failed" | "cancelled";
type StageName = "ingest" | "sanitize" | "synth" | "train" | "evaluate" | "package";
type StageInfo = {
  name: StageName; status: "pending" | "running" | "completed" | "failed" | "awaiting_approval";
  started_at: string | null; finished_at: string | null; metrics: Record<string, unknown>;
  error: string | null;
};
type RunSummary = { id: string; workspace_id: string; mode: "A" | "B" | "C"; state: RunState;
  created_at: string; current_stage: StageName | null; };
type Run = RunSummary & { hardware: string; stages: StageInfo[]; error: string | null;
  eta_seconds: number | null; };
type RedactionSummary = {
  counts_by_type: Record<string, number>; chunks_dropped: number; chunks_total: number;
  samples: { chunk_id: string; text_redacted: string; spans: { start: number; end: number;
    entity_type: string; action: string }[] }[];   // text AFTER masking only
};
type Trial = { number: number; state: string; objective: number | null;
  params: Record<string, unknown>; metrics: Record<string, number>; seconds: number };
type Gate = { gate: string; passed: boolean; value: number; threshold: number;
  details: Record<string, unknown> };
type Scorecard = {
  run_id: string; gates: Gate[]; all_passed: boolean;
  metrics: Record<"base" | "base_rag" | "adapter" | "adapter_rag", Record<string, number>>;
  examples: { prompt: string; base: string; adapter: string; reference: string | null }[];
};
type Adapter = { id: string; run_id: string; workspace_id: string; base_model: string;
  created_at: string; gates_passed: boolean; size_bytes: number; deployed: boolean };
type Deployment = { id: string; adapter_id: string; model_name: string; endpoint: string;
  state: "starting" | "ready" | "stopped" | "failed" };
type Event = { ts: number; run_id: string; type: string; stage?: StageName;
  done?: number; total?: number; message?: string; [k: string]: unknown };
```

## Endpoints

| Method | Path | Body → Response |
|---|---|---|
| POST | `/api/auth/login` | `{username, password}` → `User` (sets cookie) |
| POST | `/api/auth/logout` | → `204` |
| GET | `/api/me` | → `User` |
| GET | `/api/setup` | → `{needs_setup: boolean}` (no auth) |
| POST | `/api/setup` | `{username, password}` → `User` (creates first admin; only when no users exist) |
| GET | `/api/workspaces` | → `Workspace[]` |
| POST | `/api/workspaces` | `{name, sensitivity, base_model}` → `Workspace` |
| GET | `/api/workspaces/{id}` | → `Workspace` |
| DELETE | `/api/workspaces/{id}` | → `204` |
| GET | `/api/workspaces/{id}/documents` | → `DocumentInfo[]` |
| POST | `/api/workspaces/{id}/documents` | multipart `files[]` (field `paths[]` optional, relative paths for folder uploads) → `DocumentInfo[]` |
| DELETE | `/api/workspaces/{id}/documents/{doc_id}` | → `204` |
| GET | `/api/models` | → `{id: string, role: "student" \| "teacher" \| "aux", license: string \| null, installed: boolean}[]` |
| POST | `/api/workspaces/{id}/runs` | `{mode: "A"\|"B"\|"C", hardware?: string, json_schema?: object, budget_seconds?: number}` → `Run` |
| GET | `/api/workspaces/{id}/runs` | → `RunSummary[]` |
| GET | `/api/runs/{id}` | → `Run` |
| GET | `/api/runs/{id}/redactions` | → `RedactionSummary` (available when state = awaiting_approval or later) |
| POST | `/api/runs/{id}/approve` | → `Run` |
| POST | `/api/runs/{id}/cancel` | → `Run` |
| POST | `/api/runs/{id}/resume` | → `Run` |
| GET | `/api/runs/{id}/events` | SSE stream of `Event` (`event: message`, `data: <json>`); replays past events first |
| GET | `/api/runs/{id}/trials` | → `Trial[]` |
| GET | `/api/runs/{id}/scorecard` | → `Scorecard` (404 until evaluate completed) |
| GET | `/api/runs/{id}/report.pdf` · `/report.json` | → file |
| GET/POST | `/api/workspaces/{id}/gold` | → / ← `{question, reference_answer?, reference_json?, answerable}[]` (POST replaces) |
| GET | `/api/adapters` | → `Adapter[]` |
| POST | `/api/adapters/{id}/deploy` | → `Deployment` |
| POST | `/api/deployments/{id}/stop` | → `Deployment` |
| GET | `/api/deployments` | → `Deployment[]` |
| POST | `/api/playground/chat` | `{adapter_id: string \| null, messages: {role, content}[], max_tokens?: number}` → `{output: string, citations: {chunk_id: string, text: string}[], json_valid: boolean \| null}` |
| GET | `/api/system/health` | → `{status: "ok", version: string}` (no auth) |
| GET | `/api/system/hardware` | → `HardwareProfile` (see `zllf.core.schemas`) |

The SPA is served from `/` (static files from `web/dist`), with client-side routing fallback to
`index.html` for non-`/api` paths. In development Vite proxies `/api` to `http://127.0.0.1:8000`.
