import type { StageInfo } from '../api/types';
import type { StageProgress } from '../lib/progress';
import { STAGE_ORDER } from '../api/types';
import { humanize, STAGE_LABELS } from '../lib/copy';
import { formatDuration } from '../lib/format';
import { Badge, ProgressBar } from './ui';
import { cx } from '../lib/cx';

function StatusIcon({ status }: { status: StageInfo['status'] }) {
  const base = 'flex h-8 w-8 shrink-0 items-center justify-center rounded-full ring-2';
  if (status === 'completed')
    return (
      <span className={cx(base, 'bg-emerald-600 text-white ring-emerald-600')} aria-hidden="true">
        <svg viewBox="0 0 20 20" className="h-4 w-4" fill="currentColor">
          <path d="M16.7 5.3a1 1 0 0 1 0 1.4l-8 8a1 1 0 0 1-1.4 0l-4-4a1 1 0 1 1 1.4-1.4L8 12.6l7.3-7.3a1 1 0 0 1 1.4 0z" />
        </svg>
      </span>
    );
  if (status === 'running')
    return (
      <span className={cx(base, 'bg-white text-indigo-700 ring-indigo-600')} aria-hidden="true">
        <span className="h-3 w-3 animate-pulse rounded-full bg-indigo-600" />
      </span>
    );
  if (status === 'awaiting_approval')
    return (
      <span className={cx(base, 'bg-amber-100 text-amber-800 ring-amber-500')} aria-hidden="true">
        !
      </span>
    );
  if (status === 'failed')
    return (
      <span className={cx(base, 'bg-red-600 text-white ring-red-600')} aria-hidden="true">
        ×
      </span>
    );
  return <span className={cx(base, 'bg-white ring-slate-300')} aria-hidden="true" />;
}

const STATUS_TEXT: Record<StageInfo['status'], string> = {
  pending: 'Not started',
  running: 'In progress',
  completed: 'Done',
  failed: 'Stopped',
  awaiting_approval: 'Waiting for your review',
};

function elapsed(s: StageInfo): number | null {
  if (!s.started_at || !s.finished_at) return null;
  return (Date.parse(s.finished_at) - Date.parse(s.started_at)) / 1000;
}

function MetricChips({ metrics }: { metrics: Record<string, unknown> }) {
  const entries = Object.entries(metrics).filter(
    ([, v]) => typeof v === 'number' || typeof v === 'string',
  );
  if (!entries.length) return null;
  return (
    <ul className="mt-1 flex flex-wrap gap-1">
      {entries.slice(0, 4).map(([k, v]) => (
        <li key={k}>
          <Badge>
            {humanize(k)}: {String(v)}
          </Badge>
        </li>
      ))}
    </ul>
  );
}

export function StageTimeline({
  stages,
  progress,
}: {
  stages: StageInfo[];
  progress: StageProgress;
}) {
  const byName = new Map(stages.map((s) => [s.name, s]));
  return (
    <ol className="space-y-0" aria-label="Run steps">
      {STAGE_ORDER.map((name, i) => {
        const s: StageInfo = byName.get(name) ?? {
          name,
          status: 'pending',
          started_at: null,
          finished_at: null,
          metrics: {},
          error: null,
        };
        const p = progress[name];
        const last = i === STAGE_ORDER.length - 1;
        const dur = elapsed(s);
        return (
          <li
            key={name}
            className="relative flex gap-3 pb-5"
            aria-current={s.status === 'running' ? 'step' : undefined}
          >
            {!last && (
              <span
                aria-hidden="true"
                className={cx(
                  'absolute left-4 top-8 -ml-px h-[calc(100%-2rem)] w-0.5',
                  s.status === 'completed' ? 'bg-emerald-600' : 'bg-slate-200',
                )}
              />
            )}
            <StatusIcon status={s.status} />
            <div className="min-w-0 flex-1 pt-1">
              <div className="flex flex-wrap items-baseline justify-between gap-x-3">
                <h3 className="font-medium text-slate-900">{STAGE_LABELS[name].title}</h3>
                <span className="text-xs text-slate-600">
                  {STATUS_TEXT[s.status]}
                  {dur != null && ` · ${formatDuration(dur)}`}
                </span>
              </div>
              <p className="text-sm text-slate-600">{STAGE_LABELS[name].description}</p>
              {s.status === 'running' && (
                <div className="mt-2">
                  <ProgressBar
                    value={p ? p.done : null}
                    max={p?.total ?? 1}
                    label={`${STAGE_LABELS[name].title} progress`}
                  />
                  {p?.message && <p className="mt-1 text-xs text-slate-600">{p.message}</p>}
                </div>
              )}
              {s.status === 'failed' && s.error && (
                <p className="mt-1 text-sm text-red-700">
                  {s.error === 'cancelled' ? 'Cancelled by a user.' : `Problem: ${s.error}`}
                </p>
              )}
              {s.status === 'completed' && <MetricChips metrics={s.metrics} />}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
