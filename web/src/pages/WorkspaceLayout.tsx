import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Link, NavLink, Outlet, useNavigate, useParams } from 'react-router-dom';
import { api } from '../api/client';
import { qk, useWorkspace } from '../api/hooks';
import type { Workspace } from '../api/types';
import { RunStateBadge, SensitivityBadge } from '../components/badges';
import { Alert, Button, ErrorBox, Loading, PageHeader } from '../components/ui';
import { cx } from '../lib/cx';

export type WorkspaceOutletContext = { workspace: Workspace };

const TABS = [
  { to: '', label: '1. Documents', end: true },
  { to: 'goal', label: '2. Goal & start' },
  { to: 'runs', label: '3. Runs & results' },
];

export function WorkspaceLayout() {
  const { wsId = '' } = useParams();
  const ws = useWorkspace(wsId);
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [confirmDelete, setConfirmDelete] = useState(false);
  const del = useMutation({
    mutationFn: () => api.deleteWorkspace(wsId),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.workspaces });
      navigate('/');
    },
  });

  if (ws.isPending) return <Loading label="Loading workspace…" />;
  if (ws.isError)
    return (
      <ErrorBox
        title="Could not open this workspace"
        error={ws.error}
        onRetry={() => void ws.refetch()}
      />
    );

  const w = ws.data;
  const run = w.latest_run;
  const active = run && ['queued', 'running', 'awaiting_approval'].includes(run.state);

  return (
    <>
      <PageHeader
        breadcrumb={
          <Link to="/" className="hover:underline">
            Workspaces
          </Link>
        }
        title={
          <span className="flex flex-wrap items-center gap-3">
            {w.name} <SensitivityBadge value={w.sensitivity} />
          </span>
        }
        description={<span className="text-sm">Starting model: {w.base_model}</span>}
        actions={
          confirmDelete ? (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm text-slate-700">
                Delete workspace and all its documents?
              </span>
              <Button
                variant="danger"
                size="sm"
                loading={del.isPending}
                onClick={() => del.mutate()}
              >
                Yes, delete
              </Button>
              <Button variant="secondary" size="sm" onClick={() => setConfirmDelete(false)}>
                Keep it
              </Button>
            </div>
          ) : (
            <Button variant="ghost" size="sm" onClick={() => setConfirmDelete(true)}>
              Delete workspace
            </Button>
          )
        }
      />
      {del.isError && <ErrorBox title="The workspace was not deleted" error={del.error} />}
      {active && run && (
        <div className="mb-4">
          <Alert
            tone={run.state === 'awaiting_approval' ? 'warning' : 'info'}
            title={
              <span className="flex flex-wrap items-center gap-2">
                A run is in progress <RunStateBadge state={run.state} />
              </span>
            }
            action={
              <Link
                className="font-medium underline"
                to={
                  run.state === 'awaiting_approval' ? `/runs/${run.id}/review` : `/runs/${run.id}`
                }
              >
                {run.state === 'awaiting_approval' ? 'Review the privacy filter' : 'Watch progress'}
              </Link>
            }
          >
            You can close the browser — the run continues on the appliance.
          </Alert>
        </div>
      )}
      <nav
        aria-label="Workspace steps"
        className="mb-6 flex gap-1 overflow-x-auto overflow-y-hidden border-b border-slate-200"
      >
        {TABS.map((t) => (
          <NavLink
            key={t.label}
            to={t.to}
            end={t.end}
            className={({ isActive }) =>
              cx(
                '-mb-px whitespace-nowrap border-b-2 px-3 py-2 text-sm font-medium',
                isActive
                  ? 'border-indigo-700 text-indigo-800'
                  : 'border-transparent text-slate-600 hover:text-slate-900',
              )
            }
          >
            {t.label}
          </NavLink>
        ))}
      </nav>
      <Outlet context={{ workspace: w } satisfies WorkspaceOutletContext} />
    </>
  );
}
