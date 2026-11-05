import { Link, useOutletContext } from 'react-router-dom';
import { useRuns } from '../api/hooks';
import { RunStateBadge } from '../components/badges';
import { Card, EmptyState, ErrorBox, Loading, TableWrap, td, th } from '../components/ui';
import { MODES, STAGE_LABELS } from '../lib/copy';
import { formatDate } from '../lib/format';
import type { WorkspaceOutletContext } from './WorkspaceLayout';

export function RunsPage() {
  const { workspace } = useOutletContext<WorkspaceOutletContext>();
  const runs = useRuns(workspace.id);

  if (runs.isPending) return <Loading label="Loading runs…" />;
  if (runs.isError)
    return (
      <ErrorBox
        title="Could not load runs"
        error={runs.error}
        onRetry={() => void runs.refetch()}
      />
    );
  if (runs.data.length === 0)
    return (
      <EmptyState
        title="No runs yet"
        action={
          <Link
            className="font-medium text-indigo-700 underline"
            to={`/workspaces/${workspace.id}/goal`}
          >
            Choose a goal and start
          </Link>
        }
      >
        Once you start building a model, its progress and quality report appear here.
      </EmptyState>
    );

  return (
    <Card title="Runs">
      <TableWrap label="Runs">
        <thead>
          <tr>
            <th className={th}>Started</th>
            <th className={th}>Goal</th>
            <th className={th}>Status</th>
            <th className={th}>Current step</th>
            <th className={th}>
              <span className="sr-only">Links</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {runs.data.map((r) => (
            <tr key={r.id}>
              <td className={`${td} whitespace-nowrap`}>{formatDate(r.created_at)}</td>
              <td className={td}>{MODES[r.mode].title}</td>
              <td className={td}>
                <RunStateBadge state={r.state} />
              </td>
              <td className={td}>{r.current_stage ? STAGE_LABELS[r.current_stage].title : '—'}</td>
              <td className={`${td} whitespace-nowrap text-right`}>
                <Link className="font-medium text-indigo-700 underline" to={`/runs/${r.id}`}>
                  Progress
                </Link>
                {r.state === 'completed' && (
                  <>
                    {' · '}
                    <Link
                      className="font-medium text-indigo-700 underline"
                      to={`/runs/${r.id}/scorecard`}
                    >
                      Quality report
                    </Link>
                  </>
                )}
                {r.state === 'awaiting_approval' && (
                  <>
                    {' · '}
                    <Link
                      className="font-medium text-indigo-700 underline"
                      to={`/runs/${r.id}/review`}
                    >
                      Review
                    </Link>
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </TableWrap>
    </Card>
  );
}
