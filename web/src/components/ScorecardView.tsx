import type { Gate, Scorecard, Variant } from '../api/types';
import { gateCopy, metricLabel, VARIANT_LABELS } from '../lib/copy';
import { formatMetric } from '../lib/format';
import { PassFail } from './badges';
import { Alert, Button, Card, ErrorBox, TableWrap, td, th } from './ui';
import { cx } from '../lib/cx';

/** Gates where the value must stay at or below the threshold (everything else: at or above). */
const AT_MOST = new Set(['G2_forgetting', 'G4_memorization']);

const VARIANTS: Variant[] = ['base', 'base_rag', 'adapter', 'adapter_rag'];

export function GateRow({ gate }: { gate: Gate }) {
  const c = gateCopy(gate.gate);
  const rule = AT_MOST.has(gate.gate) ? 'at most' : 'at least';
  return (
    <li
      className={cx(
        'rounded-lg p-4 ring-1',
        gate.passed ? 'bg-white ring-slate-200' : 'bg-red-50 ring-red-300',
      )}
      data-testid={`gate-${gate.gate}`}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h3 className="font-semibold text-slate-900">{c.title}</h3>
        <PassFail passed={gate.passed} />
      </div>
      <p className="mt-1 text-sm text-slate-700">
        Result: <strong className="tabular-nums">{c.format(gate.value)}</strong> · needed {rule}{' '}
        <span className="tabular-nums">{c.format(gate.threshold)}</span>
      </p>
      <details className="mt-2 text-sm text-slate-600">
        <summary className="cursor-pointer font-medium text-indigo-700">
          What does this mean?
        </summary>
        <p className="mt-1">{c.explanation}</p>
        {!gate.passed && (
          <p className="mt-1 text-red-800">
            What you can do: add more (or more varied) documents, add your own test questions, or
            allow more time, then start a new run.
          </p>
        )}
      </details>
    </li>
  );
}

export type ScorecardViewProps = {
  card: Scorecard;
  /** adapter produced by this run, if registered */
  adapterId: string | null;
  deployed: boolean;
  onDeploy: () => void;
  deploying?: boolean;
  deployError?: unknown;
  onDownload: (kind: 'pdf' | 'json') => void;
  downloadError?: unknown;
};

export function ScorecardView({
  card,
  adapterId,
  deployed,
  onDeploy,
  deploying,
  deployError,
  onDownload,
  downloadError,
}: ScorecardViewProps) {
  const failed = card.gates.filter((g) => !g.passed);
  const metricKeys = Array.from(
    new Set(VARIANTS.flatMap((v) => Object.keys(card.metrics[v] ?? {}))),
  );
  const canDeploy = card.all_passed && !!adapterId && !deployed;

  return (
    <div className="space-y-6">
      {card.all_passed ? (
        <Alert tone="success" title="All quality checks passed">
          The new model is ready to deploy.
        </Alert>
      ) : (
        <Alert
          tone="danger"
          title={`${failed.length} quality check${failed.length === 1 ? '' : 's'} not passed`}
        >
          This model cannot be deployed. Nothing is wrong with your data — the model simply is not
          good enough yet. See the checks below for what to do.
        </Alert>
      )}

      <Card title="Quality checks">
        <ul className="grid gap-3 md:grid-cols-2">
          {card.gates.map((g) => (
            <GateRow key={g.gate} gate={g} />
          ))}
        </ul>
      </Card>

      <Card title="How it compares">
        <p className="mb-3 text-sm text-slate-600">
          Measured on questions held back from training (and your own test questions, if you added
          them).
        </p>
        <TableWrap label="Comparison with the original model">
          <thead>
            <tr>
              <th className={th}>Measure</th>
              {VARIANTS.map((v) => (
                <th
                  key={v}
                  className={cx(th, v === 'adapter' && 'bg-indigo-50')}
                  title={VARIANT_LABELS[v].hint}
                >
                  {VARIANT_LABELS[v].title}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metricKeys.map((k) => (
              <tr key={k}>
                <th scope="row" className={cx(td, 'font-medium text-slate-800')}>
                  {metricLabel(k)}
                </th>
                {VARIANTS.map((v) => (
                  <td
                    key={v}
                    className={cx(
                      td,
                      'tabular-nums',
                      v === 'adapter' && 'bg-indigo-50 font-semibold',
                    )}
                  >
                    {formatMetric(card.metrics[v]?.[k])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </TableWrap>
      </Card>

      {card.examples.length > 0 && (
        <Card title="Example answers">
          <ul className="space-y-4">
            {card.examples.map((ex, i) => (
              <li key={i} className="rounded-md ring-1 ring-slate-200">
                <p className="border-b border-slate-200 bg-slate-50 px-3 py-2 text-sm font-medium">
                  {ex.prompt}
                </p>
                <div className="grid gap-px bg-slate-200 md:grid-cols-2">
                  <div className="min-w-0 bg-white p-3">
                    <p className="text-xs font-semibold uppercase text-slate-500">Original model</p>
                    <p className="mt-1 whitespace-pre-wrap wrap-anywhere text-sm">{ex.base}</p>
                  </div>
                  <div className="min-w-0 bg-white p-3">
                    <p className="text-xs font-semibold uppercase text-indigo-700">
                      Your new model
                    </p>
                    <p className="mt-1 whitespace-pre-wrap wrap-anywhere text-sm">{ex.adapter}</p>
                  </div>
                </div>
                <p className="border-t border-slate-200 px-3 py-2 text-xs text-slate-600">
                  {ex.reference ? (
                    <>
                      <span className="font-semibold">Expected: </span>
                      {ex.reference}
                    </>
                  ) : (
                    'Your documents do not answer this — the right response is to say so.'
                  )}
                </p>
              </li>
            ))}
          </ul>
        </Card>
      )}

      <Card title="Audit report and deployment">
        <div className="flex flex-wrap gap-2">
          <a
            href="#download-pdf"
            onClick={(e) => {
              e.preventDefault();
              onDownload('pdf');
            }}
            className="rounded-md bg-white px-4 py-2 text-sm font-medium text-slate-800 ring-1 ring-inset ring-slate-300 hover:bg-slate-50"
          >
            Download report (PDF)
          </a>
          <a
            href="#download-json"
            onClick={(e) => {
              e.preventDefault();
              onDownload('json');
            }}
            className="rounded-md bg-white px-4 py-2 text-sm font-medium text-slate-800 ring-1 ring-inset ring-slate-300 hover:bg-slate-50"
          >
            Download report (JSON)
          </a>
        </div>
        {downloadError != null && (
          <div className="mt-3">
            <ErrorBox title="Download failed" error={downloadError} />
          </div>
        )}
        <div className="mt-4 border-t border-slate-200 pt-4">
          <Button
            onClick={onDeploy}
            disabled={!canDeploy}
            loading={deploying}
            aria-describedby="deploy-help"
          >
            {deployed ? 'Deployed' : 'Deploy'}
          </Button>
          <p id="deploy-help" className="mt-2 text-sm text-slate-600">
            {deployed
              ? 'This model is live. Try it in the playground or connect other tools via the Models page.'
              : !card.all_passed
                ? 'Deploy is available only when every quality check has passed.'
                : !adapterId
                  ? 'The model is still being packaged. Deploy becomes available in a moment.'
                  : 'Starts the model on this appliance with a chat playground and an API for other tools.'}
          </p>
          {deployError != null && (
            <div className="mt-3">
              <ErrorBox title="Deployment did not start" error={deployError} />
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
