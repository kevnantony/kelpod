// Mirrors docs/API.md (ZLLF HTTP API v1). Keep in sync with that file — it is the contract.

export type Role = 'admin' | 'user';
export type User = { id: string; username: string; role: Role };

export type Sensitivity = 'public' | 'internal' | 'confidential' | 'restricted';

export type Workspace = {
  id: string;
  name: string;
  sensitivity: Sensitivity;
  base_model: string;
  created_at: string;
  latest_run: RunSummary | null;
  deployed_adapter_id: string | null;
};

export type DocumentStatus = 'uploaded' | 'parsed' | 'skipped' | 'failed';
export type DocumentInfo = {
  id: string;
  name: string;
  mime: string;
  size_bytes: number;
  sha256: string;
  status: DocumentStatus;
  pages: number | null;
  error: string | null;
};

export type RunState =
  'queued' | 'running' | 'awaiting_approval' | 'completed' | 'failed' | 'cancelled';
export type StageName = 'ingest' | 'sanitize' | 'synth' | 'train' | 'evaluate' | 'package';
export const STAGE_ORDER: readonly StageName[] = [
  'ingest',
  'sanitize',
  'synth',
  'train',
  'evaluate',
  'package',
];

export type StageStatus = 'pending' | 'running' | 'completed' | 'failed' | 'awaiting_approval';
export type StageInfo = {
  name: StageName;
  status: StageStatus;
  started_at: string | null;
  finished_at: string | null;
  metrics: Record<string, unknown>;
  error: string | null;
};

export type Mode = 'A' | 'B' | 'C';
export type RunSummary = {
  id: string;
  workspace_id: string;
  mode: Mode;
  state: RunState;
  created_at: string;
  current_stage: StageName | null;
};
export type Run = RunSummary & {
  hardware: string;
  stages: StageInfo[];
  error: string | null;
  eta_seconds: number | null;
};

export type RedactionSpan = { start: number; end: number; entity_type: string; action: string };
export type RedactionSummary = {
  counts_by_type: Record<string, number>;
  chunks_dropped: number;
  chunks_total: number;
  /** text AFTER masking only — never raw values */
  samples: { chunk_id: string; text_redacted: string; spans: RedactionSpan[] }[];
};

export type Trial = {
  number: number;
  state: string;
  objective: number | null;
  params: Record<string, unknown>;
  metrics: Record<string, number>;
  seconds: number;
};

export type Gate = {
  gate: string;
  passed: boolean;
  value: number;
  threshold: number;
  details: Record<string, unknown>;
};

export type Variant = 'base' | 'base_rag' | 'adapter' | 'adapter_rag';
export type Scorecard = {
  run_id: string;
  gates: Gate[];
  all_passed: boolean;
  metrics: Record<Variant, Record<string, number>>;
  examples: { prompt: string; base: string; adapter: string; reference: string | null }[];
};

export type Adapter = {
  id: string;
  run_id: string;
  workspace_id: string;
  base_model: string;
  created_at: string;
  gates_passed: boolean;
  size_bytes: number;
  deployed: boolean;
};

export type DeploymentState = 'starting' | 'ready' | 'stopped' | 'failed';
export type Deployment = {
  id: string;
  adapter_id: string;
  model_name: string;
  endpoint: string;
  state: DeploymentState;
};

export type RunEvent = {
  /** epoch seconds */
  ts: number;
  run_id: string;
  type: string;
  stage?: StageName;
  done?: number;
  total?: number;
  message?: string;
  [k: string]: unknown;
};

export type ModelInfo = {
  id: string;
  role: 'student' | 'teacher' | 'aux';
  license: string | null;
  installed: boolean;
};

export type CreateWorkspaceBody = { name: string; sensitivity: Sensitivity; base_model: string };
export type StartRunBody = {
  mode: Mode;
  hardware?: string;
  json_schema?: Record<string, unknown>;
  budget_seconds?: number;
};

export type GoldItem = {
  question: string;
  reference_answer?: string;
  reference_json?: Record<string, unknown>;
  answerable: boolean;
};

export type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string };
export type ChatRequest = {
  adapter_id: string | null;
  messages: ChatMessage[];
  max_tokens?: number;
};
export type ChatResponse = {
  output: string;
  citations: { chunk_id: string; text: string }[];
  json_valid: boolean | null;
};

export type Health = { status: 'ok'; version: string };

/** zllf.core.schemas.HardwareProfile */
export type HardwareProfile = {
  platform: string;
  accelerator: 'mps' | 'cuda' | 'cpu';
  device_name: string;
  n_devices: number;
  memory_total_gb: number;
  memory_available_gb: number;
  cpu_cores: number;
  disk_free_gb: number;
  bf16: boolean;
  fp8: boolean;
  backend: string;
  quantization: '4bit' | '8bit' | 'none';
};

export type ApiErrorBody = { error: { code: string; message: string } };
