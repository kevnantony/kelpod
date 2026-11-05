import { beforeEach, describe, expect, it } from 'vitest';
import type { Adapter, Deployment, Run, RunEvent, Scorecard, Trial, Workspace } from '../api/types';
import { MockBackend } from './backend';

let t = 0;
const now = () => (t += 1000);

function ok<T>(b: MockBackend, method: string, path: string, body?: unknown): T {
  const res = b.handle(method, path, body);
  if (res.status >= 400)
    throw new Error(`${method} ${path} -> ${res.status} ${JSON.stringify(res.body)}`);
  return res.body as T;
}

function uploadForm(names: string[]): FormData {
  const fd = new FormData();
  for (const n of names) {
    fd.append('files[]', new File(['hello world'], n.split('/').pop()!));
    fd.append('paths[]', n);
  }
  return fd;
}

function setupWorkspace(b: MockBackend, name = 'Line 3 manuals'): Workspace {
  ok(b, 'POST', '/setup', { username: 'admin', password: 'password1' });
  ok(b, 'POST', '/auth/login', { username: 'admin', password: 'password1' });
  const ws = ok<Workspace>(b, 'POST', '/workspaces', {
    name,
    sensitivity: 'confidential',
    base_model: 'Qwen/Qwen2.5-0.5B-Instruct',
  });
  ok(
    b,
    'POST',
    `/workspaces/${ws.id}/documents`,
    uploadForm(['docs/a.pdf', 'docs/b.docx', 'docs/x.exe']),
  );
  b.step(); // parse documents
  return ws;
}

function runUntil(b: MockBackend, runId: string, pred: (r: Run) => boolean, max = 200): Run {
  for (let i = 0; i < max; i++) {
    const r = ok<Run>(b, 'GET', `/runs/${runId}`);
    if (pred(r)) return r;
    b.step();
  }
  throw new Error('run did not reach expected state');
}

describe('MockBackend', () => {
  let b: MockBackend;
  beforeEach(() => {
    t = 0;
    b = new MockBackend({ now });
  });

  it('requires setup, then auth', () => {
    expect(ok<{ needs_setup: boolean }>(b, 'GET', '/setup').needs_setup).toBe(true);
    expect(b.handle('GET', '/workspaces').status).toBe(401);
    expect(b.handle('POST', '/setup', { username: 'a', password: 'short' }).status).toBe(422);
    ok(b, 'POST', '/setup', { username: 'a', password: 'longenough' });
    expect(b.handle('POST', '/setup', { username: 'b', password: 'longenough' }).status).toBe(409);
    expect(b.handle('POST', '/auth/login', { username: 'a', password: 'wrong' })).toMatchObject({
      status: 401,
      body: { error: { code: 'E_BAD_CREDENTIALS' } },
    });
    ok(b, 'POST', '/auth/login', { username: 'a', password: 'longenough' });
    expect(ok<{ role: string }>(b, 'GET', '/me').role).toBe('admin');
    expect(b.handle('POST', '/auth/logout').status).toBe(204);
    expect(b.handle('GET', '/me').status).toBe(401);
  });

  it('parses uploaded documents with relative paths', () => {
    const ws = setupWorkspace(b);
    const docs = ok<{ name: string; status: string }[]>(b, 'GET', `/workspaces/${ws.id}/documents`);
    expect(docs.map((d) => [d.name, d.status])).toEqual([
      ['docs/a.pdf', 'parsed'],
      ['docs/b.docx', 'parsed'],
      ['docs/x.exe', 'skipped'],
    ]);
  });

  it('refuses to start without readable documents', () => {
    ok(b, 'POST', '/setup', { username: 'a', password: 'longenough' });
    ok(b, 'POST', '/auth/login', { username: 'a', password: 'longenough' });
    const ws = ok<Workspace>(b, 'POST', '/workspaces', {
      name: 'empty',
      sensitivity: 'public',
      base_model: 'Qwen/Qwen2.5-0.5B-Instruct',
    });
    expect(b.handle('POST', `/workspaces/${ws.id}/runs`, { mode: 'A' })).toMatchObject({
      status: 409,
      body: { error: { code: 'E_NO_DOCUMENTS' } },
    });
  });

  it('simulates a full run: stages, approval pause, trials, scorecard, adapter, deploy, chat', () => {
    const ws = setupWorkspace(b);
    const run = ok<Run>(b, 'POST', `/workspaces/${ws.id}/runs`, {
      mode: 'A',
      budget_seconds: 3600,
    });
    expect(run.state).toBe('queued');
    expect(run.stages.map((s) => s.name)).toEqual([
      'ingest',
      'sanitize',
      'synth',
      'train',
      'evaluate',
      'package',
    ]);
    expect(b.handle('POST', `/workspaces/${ws.id}/runs`, { mode: 'A' }).status).toBe(409);

    const events: RunEvent[] = [];
    b.subscribe(run.id, (e) => events.push(e));

    // Redactions are not available until sanitize finished.
    expect(b.handle('GET', `/runs/${run.id}/redactions`).status).toBe(404);
    const paused = runUntil(b, run.id, (r) => r.state === 'awaiting_approval');
    expect(paused.current_stage).toBe('sanitize');
    expect(paused.stages.find((s) => s.name === 'sanitize')!.status).toBe('awaiting_approval');
    expect(paused.stages.find((s) => s.name === 'synth')!.status).toBe('pending');
    // Stays paused no matter how long we wait.
    for (let i = 0; i < 10; i++) b.step();
    expect(ok<Run>(b, 'GET', `/runs/${run.id}`).state).toBe('awaiting_approval');

    const red = ok<{
      samples: { text_redacted: string; spans: { start: number; end: number }[] }[];
    }>(b, 'GET', `/runs/${run.id}/redactions`);
    const s0 = red.samples[0]!;
    expect(s0.text_redacted.slice(s0.spans[0]!.start, s0.spans[0]!.end)).toMatch(
      /^\[[A-Z_]+_\d+\]$/,
    );
    expect(b.handle('GET', `/runs/${run.id}/scorecard`).status).toBe(404);

    const approved = ok<Run>(b, 'POST', `/runs/${run.id}/approve`);
    expect(approved.state).toBe('running');
    expect(approved.current_stage).toBe('synth');
    expect(approved.eta_seconds).toBeGreaterThan(0);

    const done = runUntil(b, run.id, (r) => r.state === 'completed');
    expect(done.stages.every((s) => s.status === 'completed')).toBe(true);
    expect(done.eta_seconds).toBeNull();

    const types = new Set(events.map((e) => e.type));
    for (const ty of [
      'stage_started',
      'progress',
      'stage_completed',
      'awaiting_approval',
      'trial_completed',
      'run_completed',
    ])
      expect(types.has(ty)).toBe(true);
    expect(events.every((e) => e.run_id === run.id && typeof e.ts === 'number')).toBe(true);

    const trials = ok<Trial[]>(b, 'GET', `/runs/${run.id}/trials`);
    expect(trials).toHaveLength(6);
    expect(trials.some((tr) => tr.state === 'pruned')).toBe(true);

    const card = ok<Scorecard>(b, 'GET', `/runs/${run.id}/scorecard`);
    expect(card.all_passed).toBe(true);
    expect(card.gates.map((g) => g.gate)).toEqual([
      'G1_domain',
      'G2_forgetting',
      'G3_hallucination',
      'G4_memorization',
    ]);
    expect(Object.keys(card.metrics).sort()).toEqual([
      'adapter',
      'adapter_rag',
      'base',
      'base_rag',
    ]);
    expect(b.handle('GET', `/runs/${run.id}/report.pdf`)).toMatchObject({
      status: 200,
      contentType: 'application/pdf',
    });

    const [adapter] = ok<Adapter[]>(b, 'GET', '/adapters');
    expect(adapter).toMatchObject({ run_id: run.id, gates_passed: true, deployed: false });

    const dep = ok<Deployment>(b, 'POST', `/adapters/${adapter!.id}/deploy`);
    expect(dep.state).toBe('starting');
    expect(
      b.handle('POST', '/playground/chat', {
        adapter_id: adapter!.id,
        messages: [{ role: 'user', content: 'hi' }],
      }).status,
    ).toBe(409);
    b.step();
    expect(ok<Deployment[]>(b, 'GET', '/deployments')[0]!.state).toBe('ready');
    expect(ok<Workspace>(b, 'GET', `/workspaces/${ws.id}`).deployed_adapter_id).toBe(adapter!.id);
    const chat = ok<{ output: string; citations: unknown[] }>(b, 'POST', '/playground/chat', {
      adapter_id: adapter!.id,
      messages: [{ role: 'user', content: 'How do I reset E-4711?' }],
    });
    expect(chat.citations.length).toBeGreaterThan(0);

    ok(b, 'POST', `/deployments/${dep.id}/stop`);
    expect(ok<Adapter[]>(b, 'GET', '/adapters')[0]!.deployed).toBe(false);
  });

  it('replays past events to late subscribers', () => {
    const ws = setupWorkspace(b);
    const run = ok<Run>(b, 'POST', `/workspaces/${ws.id}/runs`, { mode: 'B' });
    b.step();
    b.step();
    const replay: string[] = [];
    b.subscribe(run.id, (e) => replay.push(e.type));
    expect(replay.slice(0, 2)).toEqual(['queued', 'stage_started']);
  });

  it('supports cancel and resume', () => {
    const ws = setupWorkspace(b);
    const run = ok<Run>(b, 'POST', `/workspaces/${ws.id}/runs`, { mode: 'A' });
    b.step();
    b.step();
    const cancelled = ok<Run>(b, 'POST', `/runs/${run.id}/cancel`);
    expect(cancelled.state).toBe('cancelled');
    expect(cancelled.stages.find((s) => s.name === 'ingest')).toMatchObject({
      status: 'failed',
      error: 'cancelled',
    });
    b.step();
    expect(ok<Run>(b, 'GET', `/runs/${run.id}`).state).toBe('cancelled');
    expect(b.handle('POST', `/runs/${run.id}/cancel`).status).toBe(409);
    const resumed = ok<Run>(b, 'POST', `/runs/${run.id}/resume`);
    expect(resumed.state).toBe('running');
    expect(runUntil(b, run.id, (r) => r.state === 'awaiting_approval').state).toBe(
      'awaiting_approval',
    );
  });

  it('produces a failing quality check for workspaces named "fail" and blocks deploy', () => {
    const ws = setupWorkspace(b, 'Demo fail case');
    const run = ok<Run>(b, 'POST', `/workspaces/${ws.id}/runs`, { mode: 'A' });
    runUntil(b, run.id, (r) => r.state === 'awaiting_approval');
    ok(b, 'POST', `/runs/${run.id}/approve`);
    runUntil(b, run.id, (r) => r.state === 'completed');
    const card = ok<Scorecard>(b, 'GET', `/runs/${run.id}/scorecard`);
    expect(card.all_passed).toBe(false);
    const [adapter] = ok<Adapter[]>(b, 'GET', '/adapters');
    expect(b.handle('POST', `/adapters/${adapter!.id}/deploy`)).toMatchObject({
      status: 409,
      body: { error: { code: 'E_GATES_FAILED' } },
    });
  });

  it('requires a schema for mode C and adds the schema check', () => {
    const ws = setupWorkspace(b);
    expect(b.handle('POST', `/workspaces/${ws.id}/runs`, { mode: 'C' }).status).toBe(422);
    const run = ok<Run>(b, 'POST', `/workspaces/${ws.id}/runs`, {
      mode: 'C',
      json_schema: { type: 'object', properties: { a: { type: 'string' } } },
    });
    runUntil(b, run.id, (r) => r.state === 'awaiting_approval');
    ok(b, 'POST', `/runs/${run.id}/approve`);
    runUntil(b, run.id, (r) => r.state === 'completed');
    const card = ok<Scorecard>(b, 'GET', `/runs/${run.id}/scorecard`);
    expect(card.gates.map((g) => g.gate)).toContain('G5_schema');
  });

  it('round-trips state through JSON', () => {
    const ws = setupWorkspace(b);
    const copy = MockBackend.fromJSON(b.toJSON(), { now });
    expect(ok<Workspace>(copy, 'GET', `/workspaces/${ws.id}`).name).toBe(ws.name);
  });

  it('stores and replaces the gold set', () => {
    const ws = setupWorkspace(b);
    ok(b, 'POST', `/workspaces/${ws.id}/gold`, [{ question: 'Q?', answerable: true }]);
    expect(ok<unknown[]>(b, 'GET', `/workspaces/${ws.id}/gold`)).toHaveLength(1);
    expect(
      b.handle('POST', `/workspaces/${ws.id}/gold`, [{ question: '', answerable: true }]).status,
    ).toBe(422);
  });
});
