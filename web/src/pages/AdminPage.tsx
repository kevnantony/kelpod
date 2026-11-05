import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { qk, useMe, useModels } from '../api/hooks';
import {
  Alert,
  Badge,
  Card,
  ErrorBox,
  Loading,
  PageHeader,
  TableWrap,
  td,
  th,
} from '../components/ui';

const ACCEL: Record<string, string> = {
  mps: 'Apple GPU (Metal)',
  cuda: 'NVIDIA GPU (CUDA)',
  cpu: 'Processor only (no GPU)',
};

const ROLE: Record<string, string> = {
  student: 'Starting model',
  teacher: 'Helper for writing training examples',
  aux: 'Support model (search / checking)',
};

export function AdminPage() {
  const me = useMe();
  const health = useQuery({ queryKey: qk.health, queryFn: api.health, refetchInterval: 15_000 });
  const hw = useQuery({ queryKey: qk.hardware, queryFn: api.hardware });
  const models = useModels();

  return (
    <>
      <PageHeader title="System" description="Health and hardware of this appliance." />
      {me.data?.role !== 'admin' && (
        <div className="mb-4">
          <Alert tone="info">
            This page is meant for administrators. Ask one if something here looks wrong.
          </Alert>
        </div>
      )}
      <div className="grid gap-6 md:grid-cols-2">
        <Card title="Health" className="min-w-0">
          {health.isPending ? (
            <Loading label="Checking…" />
          ) : health.isError ? (
            <ErrorBox
              title="The service is not responding"
              error={health.error}
              onRetry={() => void health.refetch()}
            />
          ) : (
            <dl className="grid grid-cols-[minmax(0,7rem)_minmax(0,1fr)] gap-2 text-sm">
              <dt className="text-slate-600">Status</dt>
              <dd>
                <Badge tone={health.data.status === 'ok' ? 'success' : 'danger'}>
                  {health.data.status === 'ok' ? 'Running normally' : health.data.status}
                </Badge>
              </dd>
              <dt className="text-slate-600">Version</dt>
              <dd className="font-mono">{health.data.version}</dd>
              <dt className="text-slate-600">Network</dt>
              <dd>Offline appliance — no outbound connections</dd>
            </dl>
          )}
        </Card>
        <Card title="Hardware" className="min-w-0">
          {hw.isPending ? (
            <Loading label="Reading hardware…" />
          ) : hw.isError ? (
            <ErrorBox
              title="Could not read the hardware profile"
              error={hw.error}
              onRetry={() => void hw.refetch()}
            />
          ) : (
            <dl className="grid grid-cols-[minmax(0,8rem)_minmax(0,1fr)] gap-2 text-sm">
              <dt className="text-slate-600">Device</dt>
              <dd>
                {hw.data.device_name}
                {hw.data.n_devices > 1 && ` × ${hw.data.n_devices}`}
              </dd>
              <dt className="text-slate-600">Accelerator</dt>
              <dd>{ACCEL[hw.data.accelerator] ?? hw.data.accelerator}</dd>
              <dt className="text-slate-600">Memory</dt>
              <dd>
                {hw.data.memory_available_gb.toFixed(1)} GB free of{' '}
                {hw.data.memory_total_gb.toFixed(1)} GB
              </dd>
              <dt className="text-slate-600">Disk space free</dt>
              <dd>{hw.data.disk_free_gb.toFixed(1)} GB</dd>
              <dt className="text-slate-600">Processor cores</dt>
              <dd>{hw.data.cpu_cores}</dd>
              <dt className="text-slate-600">Platform</dt>
              <dd className="font-mono text-xs">{hw.data.platform}</dd>
              <dt className="text-slate-600">Engine</dt>
              <dd className="font-mono text-xs">
                {hw.data.backend} · {hw.data.quantization}
                {hw.data.bf16 && ' · bf16'}
                {hw.data.fp8 && ' · fp8'}
              </dd>
            </dl>
          )}
        </Card>
        <Card title="Installed models" className="min-w-0 md:col-span-2">
          {models.isPending ? (
            <Loading label="Loading models…" />
          ) : models.isError ? (
            <ErrorBox error={models.error} onRetry={() => void models.refetch()} />
          ) : (
            <TableWrap label="Installed models">
              <thead>
                <tr>
                  <th className={th}>Model</th>
                  <th className={th}>Used as</th>
                  <th className={th}>Licence</th>
                  <th className={th}>Installed</th>
                </tr>
              </thead>
              <tbody>
                {models.data.map((m) => (
                  <tr key={m.id}>
                    <td className={`${td} break-all font-mono text-xs`}>{m.id}</td>
                    <td className={td}>{ROLE[m.role] ?? m.role}</td>
                    <td className={td}>{m.license ?? 'Unknown'}</td>
                    <td className={td}>
                      {m.installed ? (
                        <Badge tone="success">Yes</Badge>
                      ) : (
                        <Badge tone="warning">Not installed</Badge>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </TableWrap>
          )}
        </Card>
      </div>
    </>
  );
}
