import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { api, downloadFile, isApiError } from '../api/client';
import { qk, useAdapters, useRun, useScorecard } from '../api/hooks';
import { ScorecardView } from '../components/ScorecardView';
import { Alert, EmptyState, ErrorBox, Loading, PageHeader } from '../components/ui';
import { MODES } from '../lib/copy';

export function ScorecardPage() {
  const { runId = '' } = useParams();
  const run = useRun(runId);
  const card = useScorecard(runId);
  const adapters = useAdapters();
  const qc = useQueryClient();
  const adapter = adapters.data?.find((a) => a.run_id === runId) ?? null;

  const deploy = useMutation({
    mutationFn: () => api.deployAdapter(adapter!.id),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.adapters });
      void qc.invalidateQueries({ queryKey: qk.deployments });
      void qc.invalidateQueries({ queryKey: qk.workspaces });
    },
  });
  const download = useMutation({
    mutationFn: (kind: 'pdf' | 'json') =>
      downloadFile(api.reportUrl(runId, kind), `zllf-report-${runId}.${kind}`),
  });

  const header = (
    <PageHeader
      breadcrumb={
        <Link to={`/runs/${runId}`} className="hover:underline">
          Back to run progress
        </Link>
      }
      title="Quality report"
      description={run.data ? `Goal: ${MODES[run.data.mode].title}` : undefined}
    />
  );

  if (card.isPending)
    return (
      <>
        {header}
        <Loading label="Loading the quality report…" />
      </>
    );
  if (card.isError) {
    if (isApiError(card.error) && card.error.status === 404)
      return (
        <>
          {header}
          <EmptyState
            title="The quality checks have not finished yet"
            action={
              <Link className="font-medium text-indigo-700 underline" to={`/runs/${runId}`}>
                Watch progress
              </Link>
            }
          >
            The report appears here as soon as the new model has been tested.
          </EmptyState>
        </>
      );
    return (
      <>
        {header}
        <ErrorBox
          title="Could not load the quality report"
          error={card.error}
          onRetry={() => void card.refetch()}
        />
      </>
    );
  }

  return (
    <>
      {header}
      {deploy.isSuccess && (
        <div className="mb-4">
          <Alert
            tone="success"
            title="Deployment started"
            action={
              <span className="flex flex-wrap gap-4">
                <Link
                  className="font-medium underline"
                  to={`/playground?adapter=${adapter?.id ?? ''}`}
                >
                  Try it in the playground
                </Link>
                <Link className="font-medium underline" to="/models">
                  API details
                </Link>
              </span>
            }
          >
            The model is starting up on this appliance. It will be ready in a few seconds.
          </Alert>
        </div>
      )}
      <ScorecardView
        card={card.data}
        adapterId={adapter?.id ?? null}
        deployed={!!adapter?.deployed}
        onDeploy={() => deploy.mutate()}
        deploying={deploy.isPending}
        deployError={deploy.isError ? deploy.error : null}
        onDownload={(k) => download.mutate(k)}
        downloadError={download.isError ? download.error : null}
      />
    </>
  );
}
