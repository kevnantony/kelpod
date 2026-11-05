import { useMemo } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { qk, useRun, useRunEvents, useTrials, useWorkspace } from '../api/hooks';
import type { Run } from '../api/types';
import { RunStateBadge } from '../components/badges';
import { StageTimeline } from '../components/StageTimeline';
import { progressFromEvents } from '../lib/progress';
import { TrialTable } from '../components/TrialTable';
import { Alert, Button, Card, ErrorBox, Loading, PageHeader } from '../components/ui';
import { describeEvent, MODES } from '../lib/copy';
import { formatDuration, formatTime } from '../lib/format';

function RunControls({ run }: { run: Run }) {
  const qc = useQueryClient();
  const onRun = (r: Run) => {
    qc.setQueryData(qk.run(r.id), r);
    void qc.invalidateQueries({ queryKey: qk.workspace(r.workspace_id) });
  };
  const cancel = useMutation({ mutationFn: () => api.cancelRun(run.id), onSuccess: onRun });
  const resume = useMutation({ mutationFn: () => api.resumeRun(run.id), onSuccess: onRun });
  const canCancel = ['queued', 'running', 'awaiting_approval'].includes(run.state);
  const canResume = ['cancelled', 'failed'].includes(run.state);
  return (
    <div className="flex flex-col items-end gap-2">
      <div className="flex flex-wrap gap-2">
        {canCancel && (
          <Button variant="secondary" loading={cancel.isPending} onClick={() => cancel.mutate()}>
            Cancel run
          </Button>
        )}
        {canResume && (
          <Button loading={resume.isPending} onClick={() => resume.mutate()}>
            Resume
          </Button>
        )}
      </div>
      {cancel.isError && <ErrorBox title="Could not cancel" error={cancel.error} />}
      {resume.isError && <ErrorBox title="Could not resume" error={resume.error} />}
    </div>
  );
}

function NextStep({ run }: { run: Run }) {
  if (run.state === 'awaiting_approval')
    return (
      <Alert
        tone="warning"
        title="Your review is needed"
        action={
          <Link
            to={`/runs/${run.id}/review`}
            className="inline-block rounded-md bg-indigo-700 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-800"
          >
            Review the privacy filter
          </Link>
        }
      >
        The documents have been read and personal details masked. Nothing is used for training until
        you check and approve the result.
      </Alert>
    );
  if (run.state === 'completed')
    return (
      <Alert
        tone="success"
        title="Your model is ready"
        action={
          <Link
            to={`/runs/${run.id}/scorecard`}
            className="inline-block rounded-md bg-indigo-700 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-800"
          >
            See the quality report
          </Link>
        }
      >
        All steps finished. Check the quality report before deploying it.
      </Alert>
    );
  if (run.state === 'failed')
    return (
      <Alert tone="danger" title="The run stopped with a problem">
        {run.error ?? 'One of the steps could not finish.'} Your documents are safe and nothing left
        this machine. Press <strong>Resume</strong> to continue from the last finished step, or ask
        an administrator to check the system logs if it keeps failing.
      </Alert>
    );
  if (run.state === 'cancelled')
    return (
      <Alert tone="neutral" title="Run cancelled">
        Finished steps are kept. Press <strong>Resume</strong> to continue where it stopped.
      </Alert>
    );
  return null;
}

export function RunPage() {
  const { runId = '' } = useParams();
  const run = useRun(runId);
  const { events, connected } = useRunEvents(runId);
  const progress = useMemo(() => progressFromEvents(events), [events]);
  const trainStarted = !!run.data?.stages.some((s) => s.name === 'train' && s.status !== 'pending');
  const trials = useTrials(runId, trainStarted);
  const ws = useWorkspace(run.data?.workspace_id ?? '');

  // Friendly log: skip noisy progress ticks except the last per stage.
  const log = useMemo(() => {
    const lastProgress = new Map<string, number>();
    events.forEach((e, i) => {
      if (e.type === 'progress') lastProgress.set(e.stage ?? '', i);
    });
    return events
      .filter((e, i) => e.type !== 'progress' || lastProgress.get(e.stage ?? '') === i)
      .slice(-60)
      .reverse();
  }, [events]);

  if (run.isPending) return <Loading label="Loading run…" />;
  if (run.isError)
    return (
      <ErrorBox
        title="Could not load this run"
        error={run.error}
        onRetry={() => void run.refetch()}
      />
    );
  const r = run.data;

  return (
    <>
      <PageHeader
        breadcrumb={
          <>
            <Link to="/" className="hover:underline">
              Workspaces
            </Link>
            {' / '}
            <Link to={`/workspaces/${r.workspace_id}/runs`} className="hover:underline">
              {ws.data?.name ?? 'Workspace'}
            </Link>
          </>
        }
        title={
          <span className="flex flex-wrap items-center gap-3">
            Building your model <RunStateBadge state={r.state} />
          </span>
        }
        description={
          <>
            Goal: {MODES[r.mode].title}
            {r.eta_seconds != null && (
              <>
                {' · '}
                <strong>About {formatDuration(r.eta_seconds)} left</strong>
              </>
            )}
          </>
        }
        actions={<RunControls run={r} />}
      />
      <div className="mb-6">
        <NextStep run={r} />
      </div>
      <div className="grid gap-6 lg:grid-cols-5">
        <Card title="Steps" className="lg:col-span-3">
          <StageTimeline stages={r.stages} progress={progress} />
        </Card>
        <Card
          title="Activity"
          className="lg:col-span-2"
          actions={
            <span className="text-xs text-slate-600" aria-live="polite">
              {connected ? 'Live' : 'Reconnecting…'}
            </span>
          }
        >
          {log.length === 0 ? (
            <p className="text-sm text-slate-600">Waiting for the first update…</p>
          ) : (
            <ol
              className="max-h-[28rem] space-y-2 overflow-y-auto text-sm"
              aria-live="polite"
              aria-relevant="additions"
            >
              {log.map((e, i) => (
                <li key={`${e.ts}-${i}`} className="flex gap-3">
                  <time
                    className="shrink-0 tabular-nums text-xs text-slate-500"
                    dateTime={new Date(e.ts * 1000).toISOString()}
                  >
                    {formatTime(e.ts)}
                  </time>
                  <span className="text-slate-800">{describeEvent(e)}</span>
                </li>
              ))}
            </ol>
          )}
        </Card>
      </div>
      {trainStarted && (
        <Card title="Training settings tried" className="mt-6">
          <p className="mb-3 text-sm text-slate-600">
            The appliance tries several training settings automatically and keeps the best one.
            Unpromising attempts are stopped early to save time.
          </p>
          {trials.isError ? (
            <ErrorBox error={trials.error} onRetry={() => void trials.refetch()} />
          ) : !trials.data?.length ? (
            <p className="text-sm text-slate-600">The first attempt is running…</p>
          ) : (
            <TrialTable trials={trials.data} />
          )}
        </Card>
      )}
    </>
  );
}
