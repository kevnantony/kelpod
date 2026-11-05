import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api, isApiError } from '../api/client';
import { qk, useRedactions, useRun } from '../api/hooks';
import { RedactionReview } from '../components/RedactionReview';
import { EmptyState, ErrorBox, Loading, PageHeader } from '../components/ui';

export function RedactionPage() {
  const { runId = '' } = useParams();
  const run = useRun(runId);
  const qc = useQueryClient();
  const navigate = useNavigate();
  const sanitize = run.data?.stages.find((s) => s.name === 'sanitize');
  const ready = sanitize?.status === 'completed' || sanitize?.status === 'awaiting_approval';
  const red = useRedactions(runId, !!ready);
  const approve = useMutation({
    mutationFn: () => api.approveRun(runId),
    onSuccess: (r) => {
      qc.setQueryData(qk.run(runId), r);
      void qc.invalidateQueries({ queryKey: qk.workspace(r.workspace_id) });
      navigate(`/runs/${runId}`);
    },
  });

  const header = (
    <PageHeader
      breadcrumb={
        <Link to={`/runs/${runId}`} className="hover:underline">
          Back to run progress
        </Link>
      }
      title="Review the privacy filter"
      description="Check what was masked before anything is used for training."
    />
  );

  if (run.isPending) return <Loading label="Loading run…" />;
  if (run.isError)
    return (
      <>
        {header}
        <ErrorBox
          title="Could not load this run"
          error={run.error}
          onRetry={() => void run.refetch()}
        />
      </>
    );
  if (!ready)
    return (
      <>
        {header}
        <EmptyState title="The privacy filter has not finished yet">
          This page fills in once your documents have been read and checked. You will see a notice
          on the run page when it is ready.
        </EmptyState>
      </>
    );

  return (
    <>
      {header}
      {red.isPending ? (
        <Loading label="Loading the privacy summary…" />
      ) : red.isError ? (
        isApiError(red.error) && red.error.status === 404 ? (
          <EmptyState title="Summary not available yet">Try again in a moment.</EmptyState>
        ) : (
          <ErrorBox
            title="Could not load the privacy summary"
            error={red.error}
            onRetry={() => void red.refetch()}
          />
        )
      ) : (
        <RedactionReview
          summary={red.data}
          onApprove={() => approve.mutate()}
          approving={approve.isPending}
          approveError={approve.isError ? approve.error : null}
          canApprove={run.data.state === 'awaiting_approval'}
        />
      )}
    </>
  );
}
