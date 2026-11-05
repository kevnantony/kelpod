import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { api } from '../api/client';
import { qk, useAdapters, useDeployments, useWorkspaces } from '../api/hooks';
import type { Adapter, Deployment } from '../api/types';
import { PassFail } from '../components/badges';
import { CodeBlock, CopyButton } from '../components/CopyButton';
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorBox,
  Loading,
  PageHeader,
  TableWrap,
  td,
  th,
} from '../components/ui';
import { curlExample, pythonExample } from '../lib/endpoint';
import { formatBytes, formatDate } from '../lib/format';

const DEP_STATE: Record<
  Deployment['state'],
  { label: string; tone: 'success' | 'info' | 'neutral' | 'danger' }
> = {
  ready: { label: 'Live', tone: 'success' },
  starting: { label: 'Starting…', tone: 'info' },
  stopped: { label: 'Stopped', tone: 'neutral' },
  failed: { label: 'Failed to start', tone: 'danger' },
};

function AdapterRow({
  a,
  wsName,
  dep,
}: {
  a: Adapter;
  wsName: string;
  dep: Deployment | undefined;
}) {
  const qc = useQueryClient();
  const refresh = () => {
    void qc.invalidateQueries({ queryKey: qk.adapters });
    void qc.invalidateQueries({ queryKey: qk.deployments });
    void qc.invalidateQueries({ queryKey: qk.workspaces });
  };
  const deploy = useMutation({ mutationFn: () => api.deployAdapter(a.id), onSuccess: refresh });
  const stop = useMutation({ mutationFn: () => api.stopDeployment(dep!.id), onSuccess: refresh });
  const live = dep && (dep.state === 'ready' || dep.state === 'starting');
  const err = deploy.error ?? stop.error;
  return (
    <tr>
      <td className={td}>
        <span className="font-medium">{wsName}</span>
        <span className="block text-xs text-slate-500">{a.base_model}</span>
      </td>
      <td className={`${td} whitespace-nowrap`}>{formatDate(a.created_at)}</td>
      <td className={td}>
        <Link
          to={`/runs/${a.run_id}/scorecard`}
          className="inline-block"
          aria-label="Open the quality report"
        >
          <PassFail passed={a.gates_passed} />
        </Link>
      </td>
      <td className={`${td} whitespace-nowrap`}>{formatBytes(a.size_bytes)}</td>
      <td className={td}>
        {dep ? <Badge tone={DEP_STATE[dep.state].tone}>{DEP_STATE[dep.state].label}</Badge> : '—'}
      </td>
      <td className={`${td} text-right`}>
        {live ? (
          <Button
            variant="secondary"
            size="sm"
            loading={stop.isPending}
            onClick={() => stop.mutate()}
          >
            Stop
          </Button>
        ) : (
          <Button
            size="sm"
            loading={deploy.isPending}
            disabled={!a.gates_passed}
            title={
              a.gates_passed
                ? undefined
                : 'Only models that passed every quality check can be deployed'
            }
            onClick={() => deploy.mutate()}
          >
            Deploy
          </Button>
        )}
        {err && <p className="mt-1 text-left text-xs text-red-700">{(err as Error).message}</p>}
      </td>
    </tr>
  );
}

export function ModelsPage() {
  const adapters = useAdapters();
  const deployments = useDeployments();
  const workspaces = useWorkspaces();
  const wsName = (id: string) => workspaces.data?.find((w) => w.id === id)?.name ?? id;
  const live = (deployments.data ?? []).filter((d) => d.state !== 'stopped');

  return (
    <>
      <PageHeader
        title="Models"
        description="Every model built on this appliance, and how other tools can use the live ones."
      />
      <div className="space-y-6">
        <Card title="Built models">
          {adapters.isPending ? (
            <Loading label="Loading models…" />
          ) : adapters.isError ? (
            <ErrorBox
              title="Could not load models"
              error={adapters.error}
              onRetry={() => void adapters.refetch()}
            />
          ) : adapters.data.length === 0 ? (
            <EmptyState
              title="No models built yet"
              action={
                <Link to="/" className="font-medium text-indigo-700 underline">
                  Go to your workspaces
                </Link>
              }
            >
              When a run finishes, its model appears here.
            </EmptyState>
          ) : (
            <TableWrap label="Built models">
              <thead>
                <tr>
                  <th className={th}>Workspace</th>
                  <th className={th}>Built</th>
                  <th className={th}>Quality checks</th>
                  <th className={th}>Size</th>
                  <th className={th}>Status</th>
                  <th className={th}>
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {[...adapters.data].reverse().map((a) => (
                  <AdapterRow
                    key={a.id}
                    a={a}
                    wsName={wsName(a.workspace_id)}
                    dep={deployments.data?.find((d) => d.adapter_id === a.id)}
                  />
                ))}
              </tbody>
            </TableWrap>
          )}
        </Card>

        <Card title="Connect other tools">
          {deployments.isError ? (
            <ErrorBox error={deployments.error} onRetry={() => void deployments.refetch()} />
          ) : live.length === 0 ? (
            <p className="text-sm text-slate-600">
              Deploy a model to get an address other tools can use. It speaks the same language
              (API) as OpenAI, so most existing tools work with it unchanged.
            </p>
          ) : (
            <ul className="space-y-6">
              {live.map((d) => {
                const a = adapters.data?.find((x) => x.id === d.adapter_id);
                return (
                  <li key={d.id} className="space-y-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <h3 className="font-semibold">{a ? wsName(a.workspace_id) : d.model_name}</h3>
                      <Badge tone={DEP_STATE[d.state].tone}>{DEP_STATE[d.state].label}</Badge>
                      <Link
                        to={`/playground?adapter=${d.adapter_id}`}
                        className="text-sm font-medium text-indigo-700 underline"
                      >
                        Try it
                      </Link>
                    </div>
                    <dl className="grid gap-2 text-sm sm:grid-cols-[10rem_1fr]">
                      <dt className="text-slate-600">Address</dt>
                      <dd className="flex flex-wrap items-center gap-2">
                        <code className="break-all rounded bg-slate-100 px-1.5 py-0.5">
                          {d.endpoint}
                        </code>
                        <CopyButton text={d.endpoint} />
                      </dd>
                      <dt className="text-slate-600">Model name</dt>
                      <dd>
                        <code className="break-all rounded bg-slate-100 px-1.5 py-0.5">
                          {d.model_name}
                        </code>
                      </dd>
                    </dl>
                    <CodeBlock label="Command line (curl)" code={curlExample(d)} />
                    <CodeBlock label="Python" code={pythonExample(d)} />
                  </li>
                );
              })}
            </ul>
          )}
        </Card>
      </div>
    </>
  );
}
