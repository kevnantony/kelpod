import { useEffect, useRef, useState, type FormEvent } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { api } from '../api/client';
import { useAdapters, useDeployments, useWorkspaces } from '../api/hooks';
import type { ChatMessage, ChatResponse } from '../api/types';
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorBox,
  Loading,
  PageHeader,
  SelectField,
  Spinner,
} from '../components/ui';
import { cx } from '../lib/cx';

type Reply =
  | { status: 'pending' }
  | { status: 'done'; res: ChatResponse }
  | { status: 'error'; error: unknown };
type Turn = { prompt: string; adapter: Reply; base: Reply | null };

function ReplyView({ reply, label, accent }: { reply: Reply; label: string; accent?: boolean }) {
  return (
    <div className="min-w-0 rounded-md bg-white p-3 ring-1 ring-slate-200">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p
          className={cx(
            'text-xs font-semibold uppercase',
            accent ? 'text-indigo-700' : 'text-slate-500',
          )}
        >
          {label}
        </p>
        {reply.status === 'done' && reply.res.json_valid !== null && (
          <Badge tone={reply.res.json_valid ? 'success' : 'danger'}>
            {reply.res.json_valid ? 'Valid structured output' : 'Does not match the form'}
          </Badge>
        )}
      </div>
      {reply.status === 'pending' && (
        <div className="mt-2 text-slate-600">
          <Spinner className="h-4 w-4" label="Waiting for the answer…" />
        </div>
      )}
      {reply.status === 'error' && (
        <div className="mt-2">
          <ErrorBox title="No answer" error={reply.error} />
        </div>
      )}
      {reply.status === 'done' && (
        <>
          <p className="mt-1 whitespace-pre-wrap wrap-anywhere text-sm">{reply.res.output}</p>
          {reply.res.citations.length > 0 && (
            <details className="mt-2 text-sm" open>
              <summary className="cursor-pointer font-medium text-slate-700">
                Sources ({reply.res.citations.length})
              </summary>
              <ol className="mt-1 list-decimal space-y-1 pl-5 text-slate-600">
                {reply.res.citations.map((c) => (
                  <li key={c.chunk_id}>
                    <span className="wrap-anywhere">{c.text}</span>{' '}
                    <span className="text-xs text-slate-500">(section {c.chunk_id})</span>
                  </li>
                ))}
              </ol>
            </details>
          )}
        </>
      )}
    </div>
  );
}

function historyFor(turns: Turn[], which: 'adapter' | 'base'): ChatMessage[] {
  const msgs: ChatMessage[] = [];
  for (const t of turns) {
    const r = which === 'adapter' ? t.adapter : t.base;
    if (!r || r.status !== 'done') continue;
    msgs.push({ role: 'user', content: t.prompt }, { role: 'assistant', content: r.res.output });
  }
  return msgs;
}

export function PlaygroundPage() {
  const [params, setParams] = useSearchParams();
  const adapters = useAdapters();
  const deployments = useDeployments();
  const workspaces = useWorkspaces();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState('');
  const [compare, setCompare] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  const live = (deployments.data ?? []).filter(
    (d) => d.state === 'ready' || d.state === 'starting',
  );
  const options = live.map((d) => {
    const a = adapters.data?.find((x) => x.id === d.adapter_id);
    const ws = workspaces.data?.find((w) => w.id === a?.workspace_id);
    return {
      id: d.adapter_id,
      label: `${ws?.name ?? d.model_name} (${d.model_name})`,
      state: d.state,
    };
  });
  const selected = params.get('adapter') || options[0]?.id || '';
  const current = options.find((o) => o.id === selected);
  const busy = turns.some((t) => t.adapter.status === 'pending' || t.base?.status === 'pending');

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: 'nearest' });
  }, [turns]);

  const ask = async (which: 'adapter' | 'base', idx: number, prompt: string, prior: Turn[]) => {
    const messages: ChatMessage[] = [
      ...historyFor(prior, which),
      { role: 'user', content: prompt },
    ];
    let reply: Reply;
    try {
      const res = await api.chat({ adapter_id: which === 'adapter' ? selected : null, messages });
      reply = { status: 'done', res };
    } catch (error) {
      reply = { status: 'error', error };
    }
    setTurns((ts) => ts.map((t, i) => (i === idx ? { ...t, [which]: reply } : t)));
  };

  const send = (e: FormEvent) => {
    e.preventDefault();
    const prompt = input.trim();
    if (!prompt || !selected || busy) return;
    const prior = turns;
    const idx = prior.length;
    setTurns([
      ...prior,
      { prompt, adapter: { status: 'pending' }, base: compare ? { status: 'pending' } : null },
    ]);
    setInput('');
    void ask('adapter', idx, prompt, prior);
    if (compare) void ask('base', idx, prompt, prior);
  };

  const loading = adapters.isPending || deployments.isPending;
  const error = adapters.error ?? deployments.error;

  return (
    <>
      <PageHeader
        title="Try your model"
        description="Ask questions to a deployed model. Everything runs on this appliance."
      />
      {loading ? (
        <Loading label="Loading deployed models…" />
      ) : error ? (
        <ErrorBox
          title="Could not load models"
          error={error}
          onRetry={() => {
            void adapters.refetch();
            void deployments.refetch();
          }}
        />
      ) : options.length === 0 ? (
        <EmptyState
          title="No model is deployed yet"
          action={
            <Link className="font-medium text-indigo-700 underline" to="/models">
              Go to Models to deploy one
            </Link>
          }
        >
          Deploy a model that passed its quality checks, then come back here to chat with it.
        </EmptyState>
      ) : (
        <div className="space-y-4">
          <Card>
            <div className="grid items-end gap-4 sm:grid-cols-2">
              <SelectField
                label="Model"
                value={selected}
                onChange={(e) => {
                  setParams({ adapter: e.target.value });
                  setTurns([]);
                }}
              >
                {options.map((o) => (
                  <option key={o.id} value={o.id}>
                    {o.label}
                    {o.state === 'starting' ? ' — starting…' : ''}
                  </option>
                ))}
              </SelectField>
              <div className="flex flex-wrap items-center gap-4">
                <label className="inline-flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={compare}
                    onChange={(e) => setCompare(e.target.checked)}
                    className="h-4 w-4 rounded border-slate-300 accent-indigo-700"
                  />
                  Compare with the original model
                </label>
                {turns.length > 0 && (
                  <Button variant="ghost" size="sm" onClick={() => setTurns([])} disabled={busy}>
                    Clear conversation
                  </Button>
                )}
              </div>
            </div>
            {current?.state === 'starting' && (
              <p className="mt-2 text-sm text-amber-800">
                This model is still starting. Answers may fail for a few seconds.
              </p>
            )}
          </Card>

          <section aria-label="Conversation" aria-live="polite" className="space-y-4">
            {turns.length === 0 && (
              <p className="text-center text-sm text-slate-600">
                Ask something your documents cover — and something they don’t, to see it admit it.
              </p>
            )}
            {turns.map((t, i) => (
              <div key={i} className="space-y-2">
                <div className="ml-auto max-w-[85%] rounded-lg bg-indigo-700 px-3 py-2 text-sm text-white w-fit">
                  <span className="sr-only">You: </span>
                  <span className="whitespace-pre-wrap wrap-anywhere">{t.prompt}</span>
                </div>
                <div className={cx('grid gap-2', t.base && 'md:grid-cols-2')}>
                  <ReplyView reply={t.adapter} label="Your model" accent />
                  {t.base && <ReplyView reply={t.base} label="Original model" />}
                </div>
              </div>
            ))}
            <div ref={endRef} />
          </section>

          <form
            onSubmit={send}
            className="sticky bottom-2 flex gap-2 rounded-lg bg-white p-2 shadow-md ring-1 ring-slate-200"
          >
            <label htmlFor="prompt" className="sr-only">
              Your message
            </label>
            <textarea
              id="prompt"
              rows={2}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  e.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Type a question… (Enter to send, Shift+Enter for a new line)"
              className="min-w-0 flex-1 resize-none rounded-md px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-600"
            />
            <Button type="submit" disabled={!input.trim() || busy} className="self-end">
              Send
            </Button>
          </form>
        </div>
      )}
    </>
  );
}
