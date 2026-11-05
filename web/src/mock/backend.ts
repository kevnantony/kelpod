// In-memory fake of the ZLLF API (docs/API.md) for mock mode (VITE_MOCK=1) and tests.
// It is deterministic: the run simulation advances only when `step()` is called (the browser
// installer calls it on a timer; tests call it directly).
import type {
  Adapter,
  ChatRequest,
  ChatResponse,
  Deployment,
  DocumentInfo,
  GoldItem,
  HardwareProfile,
  Mode,
  ModelInfo,
  RedactionSummary,
  Run,
  RunEvent,
  Scorecard,
  Sensitivity,
  StageInfo,
  StageName,
  Trial,
  User,
  Workspace,
} from '../api/types';
import { STAGE_ORDER } from '../api/types';

export type MockResponse = {
  status: number;
  body?: unknown;
  /** set for non-JSON bodies (report files) */
  contentType?: string;
};

type SimStage = { done: number; total: number; ticks: number };
type MockRun = Run & {
  approved: boolean;
  workspaceName: string;
  sim: Record<StageName, SimStage>;
  json_schema?: Record<string, unknown>;
  budget_seconds?: number;
};

type State = {
  seq: number;
  users: (User & { pw: string })[];
  session: string | null;
  workspaces: Omit<Workspace, 'latest_run' | 'deployed_adapter_id'>[];
  documents: Record<string, DocumentInfo[]>;
  gold: Record<string, GoldItem[]>;
  runs: MockRun[];
  events: Record<string, RunEvent[]>;
  trials: Record<string, Trial[]>;
  scorecards: Record<string, Scorecard>;
  redactions: Record<string, RedactionSummary>;
  adapters: Adapter[];
  deployments: Deployment[];
};

export type MockOptions = {
  /** clock in ms (injectable for tests) */
  now?: () => number;
  /** simulated wall-clock seconds per step, used for ETA */
  secondsPerStep?: number;
};

const MODELS: ModelInfo[] = [
  { id: 'Qwen/Qwen2.5-0.5B-Instruct', role: 'student', license: 'Apache-2.0', installed: true },
  { id: 'Qwen/Qwen2.5-7B-Instruct', role: 'student', license: 'Apache-2.0', installed: true },
  {
    id: 'meta-llama/Llama-3.1-8B-Instruct',
    role: 'student',
    license: 'Llama 3.1 Community',
    installed: false,
  },
  { id: 'Qwen/Qwen2.5-14B-Instruct-AWQ', role: 'teacher', license: 'Apache-2.0', installed: true },
  { id: 'BAAI/bge-m3', role: 'aux', license: 'MIT', installed: true },
  { id: 'cross-encoder/nli-deberta-v3-base', role: 'aux', license: 'Apache-2.0', installed: true },
];

const HARDWARE: HardwareProfile = {
  platform: 'darwin-arm64',
  accelerator: 'mps',
  device_name: 'Apple M4 Pro (mock)',
  n_devices: 1,
  memory_total_gb: 24,
  memory_available_gb: 17.5,
  cpu_cores: 14,
  disk_free_gb: 412.3,
  bf16: true,
  fp8: false,
  backend: 'mlx',
  quantization: '4bit',
};

const SUPPORTED: Record<string, string> = {
  pdf: 'application/pdf',
  docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  txt: 'text/plain',
  md: 'text/markdown',
  html: 'text/html',
  htm: 'text/html',
  xml: 'application/xml',
  csv: 'text/csv',
  parquet: 'application/vnd.apache.parquet',
  tif: 'image/tiff',
  tiff: 'image/tiff',
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
};

const N_TRIALS = 6;
const TRIAL_TICKS = 3;

function err(status: number, code: string, message: string): MockResponse {
  return { status, body: { error: { code, message } } };
}

function fakeSha(s: string): string {
  // Not cryptographic — just a stable 64-hex-char identifier for mock documents.
  let out = '';
  let h = 0x811c9dc5;
  for (let round = 0; round < 8; round++) {
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i) + round;
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    out += h.toString(16).padStart(8, '0');
  }
  return out;
}

function pwHash(pw: string): string {
  return fakeSha(`zllf-mock:${pw}`);
}

export class MockBackend {
  private s: State;
  private readonly now: () => number;
  private readonly secondsPerStep: number;
  private listeners = new Map<string, Set<(e: RunEvent) => void>>();

  constructor(opts: MockOptions = {}, initial?: State) {
    this.now = opts.now ?? (() => Date.now());
    this.secondsPerStep = opts.secondsPerStep ?? 0.6;
    this.s = initial ?? {
      seq: 1,
      users: [],
      session: null,
      workspaces: [],
      documents: {},
      gold: {},
      runs: [],
      events: {},
      trials: {},
      scorecards: {},
      redactions: {},
      adapters: [],
      deployments: [],
    };
  }

  static fromJSON(json: string, opts: MockOptions = {}): MockBackend {
    return new MockBackend(opts, JSON.parse(json) as State);
  }

  toJSON(): string {
    return JSON.stringify(this.s);
  }

  // ---------------------------------------------------------------- helpers
  private id(prefix: string): string {
    return `${prefix}_${(this.s.seq++).toString(36).padStart(4, '0')}`;
  }

  private iso(): string {
    return new Date(this.now()).toISOString();
  }

  private user(): User | null {
    const u = this.s.users.find((x) => x.id === this.s.session);
    return u ? { id: u.id, username: u.username, role: u.role } : null;
  }

  private workspace(id: string): Workspace | null {
    const w = this.s.workspaces.find((x) => x.id === id);
    if (!w) return null;
    const runs = this.s.runs.filter((r) => r.workspace_id === id);
    const latest = runs[runs.length - 1];
    const dep = this.s.adapters.find((a) => a.workspace_id === id && a.deployed);
    return {
      ...w,
      latest_run: latest ? this.summary(latest) : null,
      deployed_adapter_id: dep?.id ?? null,
    };
  }

  private summary(r: MockRun) {
    return {
      id: r.id,
      workspace_id: r.workspace_id,
      mode: r.mode,
      state: r.state,
      created_at: r.created_at,
      current_stage: r.current_stage,
    };
  }

  private publicRun(r: MockRun): Run {
    return {
      ...this.summary(r),
      hardware: r.hardware,
      stages: r.stages.map((s) => ({ ...s, metrics: { ...s.metrics } })),
      error: r.error,
      eta_seconds: this.eta(r),
    };
  }

  private eta(r: MockRun): number | null {
    if (r.state !== 'running' && r.state !== 'queued') return null;
    let ticks = 0;
    for (const name of STAGE_ORDER) {
      const st = r.stages.find((s) => s.name === name)!;
      if (st.status === 'completed') continue;
      const sim = r.sim[name];
      ticks += Math.ceil(sim.ticks * (1 - sim.done / Math.max(1, sim.total)));
    }
    return Math.round(ticks * this.secondsPerStep);
  }

  private emit(run: MockRun, type: string, extra: Partial<RunEvent> = {}): void {
    const ev: RunEvent = { ts: this.now() / 1000, run_id: run.id, type, ...extra };
    (this.s.events[run.id] ??= []).push(ev);
    this.listeners.get(run.id)?.forEach((l) => l(ev));
  }

  /** Subscribe to a run's events. Past events are replayed synchronously first. */
  subscribe(runId: string, listener: (e: RunEvent) => void): () => void {
    for (const ev of this.s.events[runId] ?? []) listener(ev);
    let set = this.listeners.get(runId);
    if (!set) this.listeners.set(runId, (set = new Set()));
    set.add(listener);
    return () => set.delete(listener);
  }

  // ---------------------------------------------------------------- simulation

  /** Advance every active run (and document parsing / deployments) by one tick. */
  step(): void {
    for (const docs of Object.values(this.s.documents)) {
      for (const d of docs) if (d.status === 'uploaded') this.parseDoc(d);
    }
    for (const d of this.s.deployments) if (d.state === 'starting') d.state = 'ready';
    for (const r of this.s.runs) this.stepRun(r);
  }

  private parseDoc(d: DocumentInfo): void {
    const ext = d.name.split('.').pop()?.toLowerCase() ?? '';
    if (d.size_bytes === 0) {
      d.status = 'failed';
      d.error = 'The file is empty. Check the original and upload it again.';
    } else if (!(ext in SUPPORTED)) {
      d.status = 'skipped';
      d.error = `“.${ext}” files are not supported. Convert it to PDF, Word or text.`;
    } else {
      d.status = 'parsed';
      d.pages = ['pdf', 'docx', 'tif', 'tiff'].includes(ext)
        ? Math.max(1, Math.round(d.size_bytes / 40_000))
        : 1;
    }
  }

  private stage(r: MockRun, name: StageName): StageInfo {
    return r.stages.find((s) => s.name === name)!;
  }

  private startStage(r: MockRun, name: StageName): void {
    const st = this.stage(r, name);
    st.status = 'running';
    st.started_at = this.iso();
    st.error = null;
    r.current_stage = name;
    this.emit(r, 'stage_started', { stage: name });
  }

  private stepRun(r: MockRun): void {
    if (r.state === 'queued') {
      r.state = 'running';
      this.startStage(r, 'ingest');
      return;
    }
    if (r.state !== 'running' || !r.current_stage) return;
    const name = r.current_stage;
    const st = this.stage(r, name);
    const sim = r.sim[name];
    const inc = Math.max(1, Math.ceil(sim.total / sim.ticks));
    const before = sim.done;
    sim.done = Math.min(sim.total, sim.done + inc);
    if (name === 'train') this.trainProgress(r, before, sim.done);
    this.emit(r, 'progress', {
      stage: name,
      done: sim.done,
      total: sim.total,
      message: this.progressMessage(name, sim.done, sim.total),
    });
    if (sim.done < sim.total) return;

    st.status = 'completed';
    st.finished_at = this.iso();
    st.metrics = this.stageMetrics(r, name);
    this.emit(r, 'stage_completed', { stage: name, metrics: st.metrics });
    if (name === 'evaluate') this.s.scorecards[r.id] = this.makeScorecard(r);

    if (name === 'sanitize' && !r.approved) {
      st.status = 'awaiting_approval';
      r.state = 'awaiting_approval';
      this.emit(r, 'awaiting_approval', { stage: 'sanitize' });
      return;
    }
    const next = STAGE_ORDER[STAGE_ORDER.indexOf(name) + 1];
    if (next) {
      this.startStage(r, next);
      return;
    }
    r.state = 'completed';
    r.current_stage = null;
    const card = this.s.scorecards[r.id];
    this.s.adapters.push({
      id: this.id('adp'),
      run_id: r.id,
      workspace_id: r.workspace_id,
      base_model: this.s.workspaces.find((w) => w.id === r.workspace_id)?.base_model ?? '',
      created_at: this.iso(),
      gates_passed: card?.all_passed ?? false,
      size_bytes: 83_886_080,
      deployed: false,
    });
    this.emit(r, 'run_completed');
  }

  private progressMessage(name: StageName, done: number, total: number): string {
    switch (name) {
      case 'ingest':
        return `Read ${done} of ${total} documents`;
      case 'sanitize':
        return `Checked ${done} of ${total} text sections for personal details`;
      case 'synth':
        return `Wrote ${done} of ${total} training examples`;
      case 'train':
        return `Training progress ${Math.round((100 * done) / total)} %`;
      case 'evaluate':
        return `Ran ${done} of ${total} test questions`;
      case 'package':
        return `Packaging step ${done} of ${total}`;
    }
  }

  private trainProgress(r: MockRun, before: number, after: number): void {
    const trials = (this.s.trials[r.id] ??= []);
    const finishedBefore = Math.floor(before / TRIAL_TICKS);
    const finishedAfter = Math.floor(after / TRIAL_TICKS);
    for (let n = finishedBefore; n < finishedAfter && n < N_TRIALS; n++) {
      const ranks = [8, 16, 32, 16, 64, 32];
      const pruned = n === 2 || n === 4;
      const objective = pruned ? null : Number((0.9 - n * 0.07 + (n % 2) * 0.05).toFixed(4));
      const t: Trial = {
        number: n,
        state: pruned ? 'pruned' : 'complete',
        objective,
        params: {
          rank: ranks[n],
          alpha: (ranks[n] ?? 16) * 2,
          learning_rate: Number((2e-4 / (1 + n * 0.3)).toExponential(2)),
          target_modules: n % 3 === 0 ? 'all' : n % 3 === 1 ? 'attention' : 'mlp',
          dropout: 0.05,
          weight_decay: 0.01,
          replay_ratio: 0.1 + n * 0.02,
        },
        metrics: pruned
          ? { val_loss: 1.4 - n * 0.01 }
          : { val_loss: Number((1.2 - n * 0.08).toFixed(3)), peak_memory_gb: 9 + n * 0.6 },
        seconds: 240 + n * 37,
      };
      trials.push(t);
      this.emit(r, 'trial_completed', {
        stage: 'train',
        trial: n,
        message: pruned
          ? `Settings ${n + 1} of ${N_TRIALS} stopped early — not promising.`
          : `Settings ${n + 1} of ${N_TRIALS} tried — score ${objective}.`,
      });
    }
  }

  private stageMetrics(r: MockRun, name: StageName): Record<string, unknown> {
    const docs = this.parsedDocs(r.workspace_id).length;
    switch (name) {
      case 'ingest':
        return { documents: docs, chunks: r.sim.sanitize.total, duplicates_removed: 2 };
      case 'sanitize': {
        const red = this.s.redactions[r.id];
        return {
          entities_found: red ? Object.values(red.counts_by_type).reduce((a, b) => a + b, 0) : 0,
          chunks_dropped: red?.chunks_dropped ?? 0,
        };
      }
      case 'synth':
        return { examples: r.sim.synth.total, unanswerable: 24, discarded: 17 };
      case 'train':
        return { trials: N_TRIALS, best_trial: 5 };
      case 'evaluate':
        return { test_questions: r.sim.evaluate.total };
      case 'package':
        return { adapter_mb: 80 };
    }
  }

  private parsedDocs(wsId: string): DocumentInfo[] {
    return (this.s.documents[wsId] ?? []).filter((d) => d.status === 'parsed');
  }

  private makeRedactions(): RedactionSummary {
    const samples = [
      'Patient [PATIENT_NAME_1], born [DATE_OF_BIRTH_1], presented with elevated CRP. Contact: [PHONE_1].',
      'Ticket raised by [PERSON_1] ([EMAIL_1]) for case [CASE_ID_1]: fault code E-4711 on line 3.',
      'Invoice to [ORGANIZATION_1], [ADDRESS_1]. Payment to [IBAN_1] within 30 days.',
      'Follow-up with [PERSON_2] scheduled; [PATIENT_NAME_1] to fast before the blood test.',
    ];
    return {
      counts_by_type: {
        PATIENT_NAME: 41,
        PERSON: 27,
        DATE_OF_BIRTH: 12,
        PHONE: 9,
        EMAIL: 14,
        CASE_ID: 22,
        ADDRESS: 6,
        IBAN: 3,
        ORGANIZATION: 18,
      },
      chunks_dropped: 4,
      chunks_total: 0, // filled in by caller
      samples: samples.map((text, i) => {
        const spans: RedactionSummary['samples'][number]['spans'] = [];
        for (const m of text.matchAll(/\[([A-Z_]+?)_\d+\]/g)) {
          spans.push({
            start: m.index,
            end: m.index + m[0].length,
            entity_type: m[1] ?? 'UNKNOWN',
            action: 'mask',
          });
        }
        return {
          chunk_id: `chunk_${String(i * 7 + 3).padStart(5, '0')}`,
          text_redacted: text,
          spans,
        };
      }),
    };
  }

  private makeScorecard(r: MockRun): Scorecard {
    const failing = /fail/i.test(r.workspaceName);
    const gates: Scorecard['gates'] = [
      {
        gate: 'G1_domain',
        passed: !failing,
        value: failing ? 0.05 : 0.31,
        threshold: 0.12,
        details: { metric: 'correctness', relative_improvement: failing ? 0.05 : 0.31 },
      },
      {
        gate: 'G2_forgetting',
        passed: true,
        value: 1.4,
        threshold: 3,
        details: { worst_suite: 'math' },
      },
      {
        gate: 'G3_hallucination',
        passed: true,
        value: 0.93,
        threshold: 0.9,
        details: { fabrication_rate: 0.03 },
      },
      { gate: 'G4_memorization', passed: true, value: 0, threshold: 0, details: { canaries: 8 } },
    ];
    if (r.mode === 'C') {
      gates.push({
        gate: 'G5_schema',
        passed: true,
        value: 0.992,
        threshold: 0.98,
        details: {},
      });
    }
    const m = (acc: number, abst: number) => {
      const row: Record<string, number> = {
        correctness: acc,
        rouge_l: Number((acc * 0.8).toFixed(3)),
        abstention_rate: abst,
      };
      if (r.mode === 'C') row.schema_valid = Math.min(1, acc + 0.3);
      return row;
    };
    const examples =
      r.mode === 'C'
        ? [
            {
              prompt: 'Extract biomarkers from: "CRP 12 mg/L on 2024-03-02, HbA1c 6.1 %."',
              base: 'The note mentions CRP and HbA1c values.',
              adapter:
                '{"biomarkers":[{"name":"CRP","value":12,"unit":"mg/L","date":"2024-03-02"},{"name":"HbA1c","value":6.1,"unit":"%"}]}',
              reference:
                '{"biomarkers":[{"name":"CRP","value":12,"unit":"mg/L","date":"2024-03-02"},{"name":"HbA1c","value":6.1,"unit":"%"}]}',
            },
          ]
        : [
            {
              prompt: 'What is the reset procedure after fault code E-4711?',
              base: 'Usually you can restart the device by turning it off and on again.',
              adapter:
                'Isolate line 3, wait for the hydraulic pressure to fall below 2 bar, then hold RESET for 5 s. (Maintenance manual §4.2)',
              reference: 'Isolate the line, wait until pressure < 2 bar, hold RESET for 5 seconds.',
            },
            {
              prompt: 'Who approves overtime for the night shift?',
              base: 'Your manager usually approves overtime.',
              adapter:
                'The shift supervisor on duty approves it in the handover log (HR policy 7.1).',
              reference: 'The shift supervisor on duty.',
            },
            {
              prompt: 'What is the warranty period for the pump model X-900?',
              base: 'The X-900 has a 2-year warranty.',
              adapter: 'I could not find the warranty period for the X-900 in your documents.',
              reference: null,
            },
          ];
    return {
      run_id: r.id,
      gates,
      all_passed: gates.every((g) => g.passed),
      metrics: {
        base: m(0.41, 0.35),
        base_rag: m(0.58, 0.52),
        adapter: m(failing ? 0.43 : 0.74, 0.93),
        adapter_rag: m(failing ? 0.6 : 0.79, 0.95),
      },
      examples,
    };
  }

  // ---------------------------------------------------------------- request handling

  /** Handle one API request. `path` excludes the `/api` prefix. */
  handle(method: string, path: string, body?: unknown): MockResponse {
    const p = path.split('?')[0]!.replace(/\/+$/, '') || '/';
    const M = method.toUpperCase();

    // --- unauthenticated
    if (M === 'GET' && p === '/system/health')
      return { status: 200, body: { status: 'ok', version: '0.1.0-mock' } };
    if (M === 'GET' && p === '/setup')
      return { status: 200, body: { needs_setup: this.s.users.length === 0 } };
    if (M === 'POST' && p === '/setup') return this.setup(body);
    if (M === 'POST' && p === '/auth/login') return this.login(body);
    if (M === 'POST' && p === '/auth/logout') {
      this.s.session = null;
      return { status: 204 };
    }

    const me = this.user();
    if (!me) return err(401, 'E_AUTH', 'Please sign in to continue.');
    if (M === 'GET' && p === '/me') return { status: 200, body: me };

    let m: RegExpMatchArray | null;
    // --- workspaces
    if (p === '/workspaces') {
      if (M === 'GET')
        return { status: 200, body: this.s.workspaces.map((w) => this.workspace(w.id)) };
      if (M === 'POST') return this.createWorkspace(body);
    }
    if ((m = p.match(/^\/workspaces\/([^/]+)$/))) {
      const ws = this.workspace(m[1]!);
      if (!ws) return this.notFound('workspace');
      if (M === 'GET') return { status: 200, body: ws };
      if (M === 'DELETE') {
        this.s.workspaces = this.s.workspaces.filter((w) => w.id !== ws.id);
        delete this.s.documents[ws.id];
        return { status: 204 };
      }
    }
    if ((m = p.match(/^\/workspaces\/([^/]+)\/documents$/))) {
      if (!this.workspace(m[1]!)) return this.notFound('workspace');
      if (M === 'GET') return { status: 200, body: this.s.documents[m[1]!] ?? [] };
      if (M === 'POST') return this.upload(m[1]!, body);
    }
    if ((m = p.match(/^\/workspaces\/([^/]+)\/documents\/([^/]+)$/)) && M === 'DELETE') {
      const docs = this.s.documents[m[1]!];
      if (!docs?.some((d) => d.id === m![2])) return this.notFound('document');
      this.s.documents[m[1]!] = docs.filter((d) => d.id !== m![2]);
      return { status: 204 };
    }
    if ((m = p.match(/^\/workspaces\/([^/]+)\/runs$/))) {
      const ws = this.workspace(m[1]!);
      if (!ws) return this.notFound('workspace');
      if (M === 'GET')
        return {
          status: 200,
          body: this.s.runs
            .filter((r) => r.workspace_id === ws.id)
            .map((r) => this.summary(r))
            .reverse(),
        };
      if (M === 'POST') return this.startRun(ws, body);
    }
    if ((m = p.match(/^\/workspaces\/([^/]+)\/gold$/))) {
      if (!this.workspace(m[1]!)) return this.notFound('workspace');
      if (M === 'GET') return { status: 200, body: this.s.gold[m[1]!] ?? [] };
      if (M === 'POST') {
        if (!Array.isArray(body))
          return err(422, 'E_VALIDATION', 'The question set must be a list of questions.');
        const items = body as GoldItem[];
        if (items.some((i) => !i || typeof i.question !== 'string' || !i.question.trim()))
          return err(422, 'E_VALIDATION', 'Every entry needs a non-empty question.');
        this.s.gold[m[1]!] = items;
        return { status: 200, body: items };
      }
    }
    if (M === 'GET' && p === '/models') return { status: 200, body: MODELS };

    // --- runs
    if ((m = p.match(/^\/runs\/([^/]+)(?:\/([a-z.]+))?$/))) {
      const r = this.s.runs.find((x) => x.id === m![1]);
      if (!r) return this.notFound('run');
      return this.runAction(M, r, m[2]);
    }

    // --- adapters & deployments
    if (M === 'GET' && p === '/adapters') return { status: 200, body: this.s.adapters };
    if (M === 'GET' && p === '/deployments') return { status: 200, body: this.s.deployments };
    if ((m = p.match(/^\/adapters\/([^/]+)\/deploy$/)) && M === 'POST') return this.deploy(m[1]!);
    if ((m = p.match(/^\/deployments\/([^/]+)\/stop$/)) && M === 'POST') {
      const d = this.s.deployments.find((x) => x.id === m![1]);
      if (!d) return this.notFound('deployment');
      d.state = 'stopped';
      const a = this.s.adapters.find((x) => x.id === d.adapter_id);
      if (a) a.deployed = false;
      return { status: 200, body: d };
    }
    if (M === 'POST' && p === '/playground/chat') return this.chat(body as ChatRequest);
    if (M === 'GET' && p === '/system/hardware') return { status: 200, body: HARDWARE };

    return err(404, 'E_NOT_FOUND', `No such API endpoint: ${M} /api${p}`);
  }

  private notFound(what: string): MockResponse {
    return err(404, 'E_NOT_FOUND', `This ${what} was not found. It may have been deleted.`);
  }

  private creds(body: unknown): { username: string; password: string } | null {
    if (!body || typeof body !== 'object') return null;
    const b = body as Record<string, unknown>;
    if (typeof b.username !== 'string' || typeof b.password !== 'string') return null;
    return { username: b.username.trim(), password: b.password };
  }

  private setup(body: unknown): MockResponse {
    if (this.s.users.length > 0)
      return err(409, 'E_SETUP_DONE', 'Setup is already complete. Sign in instead.');
    const c = this.creds(body);
    if (!c || !c.username) return err(422, 'E_VALIDATION', 'Enter a user name and a password.');
    if (c.password.length < 8)
      return err(422, 'E_WEAK_PASSWORD', 'Choose a password with at least 8 characters.');
    const u = {
      id: this.id('usr'),
      username: c.username,
      role: 'admin' as const,
      pw: pwHash(c.password),
    };
    this.s.users.push(u);
    return { status: 200, body: { id: u.id, username: u.username, role: u.role } };
  }

  private login(body: unknown): MockResponse {
    const c = this.creds(body);
    const u = c && this.s.users.find((x) => x.username === c.username);
    if (!c || !u || u.pw !== pwHash(c.password))
      return err(401, 'E_BAD_CREDENTIALS', 'User name or password is wrong. Try again.');
    this.s.session = u.id;
    return { status: 200, body: { id: u.id, username: u.username, role: u.role } };
  }

  private createWorkspace(body: unknown): MockResponse {
    const b = (body ?? {}) as Record<string, unknown>;
    const name = typeof b.name === 'string' ? b.name.trim() : '';
    const sens = b.sensitivity as Sensitivity;
    if (!name) return err(422, 'E_VALIDATION', 'Give the workspace a name.');
    if (!['public', 'internal', 'confidential', 'restricted'].includes(sens))
      return err(422, 'E_VALIDATION', 'Choose a sensitivity label.');
    const model = MODELS.find((x) => x.id === b.base_model && x.role === 'student');
    if (!model) return err(422, 'E_VALIDATION', 'Choose one of the installed base models.');
    if (!model.installed)
      return err(
        422,
        'E_MODEL_NOT_INSTALLED',
        `${model.id} is not installed on this appliance. Pick another model or ask an administrator to import it.`,
      );
    if (this.s.workspaces.some((w) => w.name.toLowerCase() === name.toLowerCase()))
      return err(
        409,
        'E_CONFLICT',
        'A workspace with this name already exists. Choose another name.',
      );
    const w = {
      id: this.id('ws'),
      name,
      sensitivity: sens,
      base_model: model.id,
      created_at: this.iso(),
    };
    this.s.workspaces.push(w);
    return { status: 201, body: this.workspace(w.id) };
  }

  private upload(wsId: string, body: unknown): MockResponse {
    if (!(body instanceof FormData))
      return err(422, 'E_VALIDATION', 'Upload must be multipart form data.');
    const files = body.getAll('files[]').filter((f): f is File => f instanceof File);
    const paths = body.getAll('paths[]').map(String);
    if (files.length === 0) return err(422, 'E_VALIDATION', 'No files were received. Try again.');
    const docs = (this.s.documents[wsId] ??= []);
    const out: DocumentInfo[] = files.map((f, i) => {
      const name = paths[i] || f.name;
      const ext = name.split('.').pop()?.toLowerCase() ?? '';
      const d: DocumentInfo = {
        id: this.id('doc'),
        name,
        mime: SUPPORTED[ext] ?? (f.type || 'application/octet-stream'),
        size_bytes: f.size,
        sha256: fakeSha(`${name}:${f.size}`),
        status: 'uploaded',
        pages: null,
        error: null,
      };
      const dup = docs.find((x) => x.sha256 === d.sha256);
      if (dup) {
        d.status = 'skipped';
        d.error = `Same content as “${dup.name}” — skipped as a duplicate.`;
      }
      docs.push(d);
      return d;
    });
    return { status: 201, body: out };
  }

  private startRun(ws: Workspace, body: unknown): MockResponse {
    const b = (body ?? {}) as Record<string, unknown>;
    const mode = b.mode as Mode;
    if (!['A', 'B', 'C'].includes(mode))
      return err(422, 'E_VALIDATION', 'Choose a goal (A, B or C).');
    if (mode === 'C' && (typeof b.json_schema !== 'object' || b.json_schema === null))
      return err(
        422,
        'E_SCHEMA_REQUIRED',
        'Structured extraction needs a form (JSON schema). Pick a template or paste one.',
      );
    const active = this.s.runs.find(
      (r) =>
        r.workspace_id === ws.id && ['queued', 'running', 'awaiting_approval'].includes(r.state),
    );
    if (active)
      return err(
        409,
        'E_RUN_ACTIVE',
        'A run is already in progress for this workspace. Wait for it to finish or cancel it first.',
      );
    const docs = this.parsedDocs(ws.id).length;
    if (docs === 0)
      return err(
        409,
        'E_NO_DOCUMENTS',
        'There are no readable documents in this workspace yet. Upload documents and wait until they show as “Ready”.',
      );
    const chunks = docs * 12;
    const r: MockRun = {
      id: this.id('run'),
      workspace_id: ws.id,
      workspaceName: ws.name,
      mode,
      state: 'queued',
      created_at: this.iso(),
      current_stage: null,
      hardware: typeof b.hardware === 'string' ? b.hardware : 'mac',
      stages: STAGE_ORDER.map((name) => ({
        name,
        status: 'pending',
        started_at: null,
        finished_at: null,
        metrics: {},
        error: null,
      })),
      error: null,
      eta_seconds: null,
      approved: false,
      json_schema: b.json_schema as Record<string, unknown> | undefined,
      budget_seconds: typeof b.budget_seconds === 'number' ? b.budget_seconds : undefined,
      sim: {
        ingest: { done: 0, total: docs, ticks: 3 },
        sanitize: { done: 0, total: chunks, ticks: 4 },
        synth: { done: 0, total: chunks * 3, ticks: 6 },
        train: { done: 0, total: N_TRIALS * TRIAL_TICKS, ticks: N_TRIALS * TRIAL_TICKS },
        evaluate: { done: 0, total: 120, ticks: 5 },
        package: { done: 0, total: 3, ticks: 2 },
      },
    };
    const red = this.makeRedactions();
    red.chunks_total = chunks;
    this.s.redactions[r.id] = red;
    this.s.runs.push(r);
    this.emit(r, 'queued', { message: 'Run queued.' });
    return { status: 201, body: this.publicRun(r) };
  }

  private runAction(M: string, r: MockRun, action: string | undefined): MockResponse {
    const ok = () => ({ status: 200, body: this.publicRun(r) });
    if (!action && M === 'GET') return ok();
    if (M === 'GET' && action === 'redactions') {
      const sanitize = this.stage(r, 'sanitize');
      if (!['completed', 'awaiting_approval'].includes(sanitize.status))
        return err(404, 'E_NOT_READY', 'The privacy filter has not finished yet.');
      return { status: 200, body: this.s.redactions[r.id] };
    }
    if (M === 'GET' && action === 'trials') return { status: 200, body: this.s.trials[r.id] ?? [] };
    if (M === 'GET' && action === 'scorecard') {
      const card = this.s.scorecards[r.id];
      if (!card) return err(404, 'E_NOT_READY', 'The quality checks have not finished yet.');
      return { status: 200, body: card };
    }
    if (M === 'GET' && (action === 'report.json' || action === 'report.pdf')) {
      if (r.state !== 'completed')
        return err(404, 'E_NOT_READY', 'The report is created when the run has finished.');
      if (action === 'report.json')
        return {
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(
            {
              run_id: r.id,
              mode: r.mode,
              scorecard: this.s.scorecards[r.id],
              trials: this.s.trials[r.id],
            },
            null,
            2,
          ),
        };
      return { status: 200, contentType: 'application/pdf', body: MOCK_PDF };
    }
    if (M === 'POST' && action === 'approve') {
      if (r.state !== 'awaiting_approval')
        return err(409, 'E_STATE', 'This run is not waiting for approval.');
      r.approved = true;
      this.stage(r, 'sanitize').status = 'completed';
      r.state = 'running';
      this.emit(r, 'approved');
      this.startStage(r, 'synth');
      return ok();
    }
    if (M === 'POST' && action === 'cancel') {
      if (!['queued', 'running', 'awaiting_approval'].includes(r.state))
        return err(409, 'E_STATE', 'Only a run that is in progress can be cancelled.');
      if (r.current_stage) {
        const st = this.stage(r, r.current_stage);
        if (st.status === 'running') {
          st.status = 'failed';
          st.error = 'cancelled';
        }
      }
      r.state = 'cancelled';
      this.emit(r, 'cancelled', { stage: r.current_stage ?? undefined });
      return ok();
    }
    if (M === 'POST' && action === 'resume') {
      if (!['cancelled', 'failed'].includes(r.state))
        return err(409, 'E_STATE', 'Only a cancelled or failed run can be resumed.');
      const first = r.stages.find((s) => s.status !== 'completed');
      if (
        first?.name === 'sanitize' &&
        !r.approved &&
        this.stage(r, 'sanitize').status === 'awaiting_approval'
      ) {
        r.state = 'awaiting_approval';
      } else if (first) {
        r.state = 'running';
        r.error = null;
        this.emit(r, 'resumed', { stage: first.name });
        this.startStage(r, first.name);
      }
      return ok();
    }
    return err(405, 'E_METHOD', `${M} is not allowed here.`);
  }

  private deploy(adapterId: string): MockResponse {
    const a = this.s.adapters.find((x) => x.id === adapterId);
    if (!a) return this.notFound('model');
    if (!a.gates_passed)
      return err(
        409,
        'E_GATES_FAILED',
        'This model did not pass all quality checks, so it cannot be deployed.',
      );
    // One active deployment per workspace: stop the previous one.
    for (const d of this.s.deployments) {
      const other = this.s.adapters.find((x) => x.id === d.adapter_id);
      if (other && other.workspace_id === a.workspace_id && d.state !== 'stopped') {
        d.state = 'stopped';
        other.deployed = false;
      }
    }
    const existing = this.s.deployments.find((d) => d.adapter_id === a.id);
    a.deployed = true;
    if (existing) {
      existing.state = 'starting';
      return { status: 200, body: existing };
    }
    const ws = this.s.workspaces.find((w) => w.id === a.workspace_id);
    const slug = (ws?.name ?? 'model')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '');
    const d: Deployment = {
      id: this.id('dep'),
      adapter_id: a.id,
      model_name: `zllf-${slug}`,
      endpoint: 'http://127.0.0.1:8001/v1',
      state: 'starting',
    };
    this.s.deployments.push(d);
    return { status: 200, body: d };
  }

  private chat(req: ChatRequest): MockResponse {
    if (!req || !Array.isArray(req.messages) || req.messages.length === 0)
      return err(422, 'E_VALIDATION', 'Type a message first.');
    const last = req.messages[req.messages.length - 1]?.content ?? '';
    if (req.adapter_id === null) {
      const res: ChatResponse = {
        output: `(Original model) I don't have specific knowledge of your organisation's documents, but generally speaking: ${last.slice(0, 60)}…`,
        citations: [],
        json_valid: null,
      };
      return { status: 200, body: res };
    }
    const a = this.s.adapters.find((x) => x.id === req.adapter_id);
    const dep = this.s.deployments.find(
      (d) => d.adapter_id === req.adapter_id && d.state === 'ready',
    );
    if (!a || !dep)
      return err(
        409,
        'E_NOT_DEPLOYED',
        'This model is not running. Deploy it on the Models page first.',
      );
    const run = this.s.runs.find((r) => r.id === a.run_id);
    if (run?.mode === 'C') {
      const res: ChatResponse = {
        output: JSON.stringify(
          { biomarkers: [{ name: 'CRP', value: 12, unit: 'mg/L', date: '2024-03-02' }] },
          null,
          2,
        ),
        citations: [],
        json_valid: true,
      };
      return { status: 200, body: res };
    }
    const res: ChatResponse = {
      output:
        'According to the maintenance manual, isolate line 3, wait until the hydraulic pressure is below 2 bar and hold RESET for 5 seconds. If the fault returns, call the on-duty engineer [PERSON_1].',
      citations: [
        {
          chunk_id: 'chunk_00031',
          text: '§4.2 Reset after E-4711: isolate the line and wait until pressure < 2 bar…',
        },
        {
          chunk_id: 'chunk_00032',
          text: 'Hold the RESET button for 5 s. If the fault persists, escalate to…',
        },
      ],
      json_valid: null,
    };
    return { status: 200, body: res };
  }
}

// Smallest valid one-page PDF so "Download PDF" produces a file viewers can open.
const MOCK_PDF = `%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 595 842]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 58>>stream
BT /F1 18 Tf 72 770 Td (ZLLF audit report - mock mode) Tj ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
trailer<</Root 1 0 R>>
%%EOF`;
