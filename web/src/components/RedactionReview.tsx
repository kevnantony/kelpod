import type { ReactNode } from 'react';
import type { RedactionSummary } from '../api/types';
import { ACTION_LABELS, entityLabel } from '../lib/copy';
import { highlightParts } from '../lib/highlight';
import { Alert, Button, Card, ErrorBox } from './ui';

type Span = RedactionSummary['samples'][number]['spans'][number];

function Highlighted({ text, spans }: { text: string; spans: Span[] }): ReactNode {
  return highlightParts(text, spans).map((p, i) =>
    p.span ? (
      <mark
        key={i}
        className="rounded bg-amber-200 px-0.5 font-mono text-[0.85em] text-amber-950"
        title={`${entityLabel(p.span.entity_type)} — ${ACTION_LABELS[p.span.action] ?? p.span.action}`}
      >
        {p.text}
        <span className="sr-only"> ({entityLabel(p.span.entity_type)} removed)</span>
      </mark>
    ) : (
      <span key={i}>{p.text}</span>
    ),
  );
}

export type RedactionReviewProps = {
  summary: RedactionSummary;
  onApprove: () => void;
  approving?: boolean;
  approveError?: unknown;
  /** false when the run is no longer waiting (already approved / cancelled) */
  canApprove?: boolean;
};

export function RedactionReview({
  summary,
  onApprove,
  approving,
  approveError,
  canApprove = true,
}: RedactionReviewProps) {
  const counts = Object.entries(summary.counts_by_type).sort((a, b) => b[1] - a[1]);
  const total = counts.reduce((s, [, n]) => s + n, 0);
  const kept = summary.chunks_total - summary.chunks_dropped;

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <div className="rounded-lg bg-white p-4 ring-1 ring-slate-200">
          <p className="text-sm text-slate-600">Personal details masked</p>
          <p className="mt-1 text-3xl font-bold tabular-nums" data-testid="total-masked">
            {total.toLocaleString()}
          </p>
        </div>
        <div className="rounded-lg bg-white p-4 ring-1 ring-slate-200">
          <p className="text-sm text-slate-600">Text sections kept</p>
          <p className="mt-1 text-3xl font-bold tabular-nums">
            {kept.toLocaleString()}
            <span className="text-base font-normal text-slate-600">
              {' '}
              of {summary.chunks_total.toLocaleString()}
            </span>
          </p>
        </div>
        <div className="rounded-lg bg-white p-4 ring-1 ring-slate-200">
          <p className="text-sm text-slate-600">Sections removed entirely</p>
          <p className="mt-1 text-3xl font-bold tabular-nums">
            {summary.chunks_dropped.toLocaleString()}
          </p>
          <p className="text-xs text-slate-600">Too sensitive to use even when masked.</p>
        </div>
      </div>

      <Card title="What was found">
        {counts.length === 0 ? (
          <p className="text-sm text-slate-700">
            No personal or sensitive details were found. Check the samples below to confirm.
          </p>
        ) : (
          <ul
            className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3"
            aria-label="Masked details by type"
          >
            {counts.map(([type, n]) => (
              <li
                key={type}
                className="flex items-center justify-between rounded-md bg-slate-50 px-3 py-2 text-sm"
              >
                <span>{entityLabel(type)}</span>
                <span className="font-semibold tabular-nums">{n.toLocaleString()}</span>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card title="Examples after masking">
        <p className="mb-3 text-sm text-slate-600">
          This is exactly the text that will be used for training. Highlighted placeholders replace
          the original details, which are never stored or shown.
        </p>
        {summary.samples.length === 0 ? (
          <p className="text-sm text-slate-600">No examples available.</p>
        ) : (
          <ul className="space-y-3">
            {summary.samples.map((s) => (
              <li
                key={s.chunk_id}
                className="rounded-md bg-slate-50 p-3 text-sm leading-relaxed ring-1 ring-slate-200"
              >
                <p className="whitespace-pre-wrap wrap-anywhere">
                  <Highlighted text={s.text_redacted} spans={s.spans} />
                </p>
                <p className="mt-1 text-xs text-slate-500">Section {s.chunk_id}</p>
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card title="Approve">
        <p className="text-sm text-slate-700">
          If something was missed or masked by mistake, cancel the run and ask an administrator to
          adjust the privacy rules. Approving lets the appliance continue with writing training
          examples and training. Your approval is recorded in the audit report.
        </p>
        {approveError != null && (
          <div className="mt-3">
            <ErrorBox title="Approval was not saved" error={approveError} />
          </div>
        )}
        {!canApprove && (
          <div className="mt-3">
            <Alert tone="info">This run is not waiting for approval any more.</Alert>
          </div>
        )}
        <Button className="mt-4" onClick={onApprove} loading={approving} disabled={!canApprove}>
          Approve and continue
        </Button>
      </Card>
    </div>
  );
}
