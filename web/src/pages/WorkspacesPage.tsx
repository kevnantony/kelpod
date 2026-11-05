import { useState, type FormEvent } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import { qk, useModels, useWorkspaces } from '../api/hooks';
import type { Sensitivity, Workspace } from '../api/types';
import { RunStateBadge, SensitivityBadge } from '../components/badges';
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorBox,
  Loading,
  PageHeader,
  SelectField,
  TextField,
} from '../components/ui';
import { SENSITIVITY, STAGE_LABELS } from '../lib/copy';
import { formatDate } from '../lib/format';

function CreateWorkspaceForm({ onDone }: { onDone: () => void }) {
  const models = useModels();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [sensitivity, setSensitivity] = useState<Sensitivity>('confidential');
  const [baseModel, setBaseModel] = useState('');
  const students = (models.data ?? []).filter((m) => m.role === 'student');
  const firstInstalled = students.find((m) => m.installed)?.id ?? '';
  const chosen = baseModel || firstInstalled;
  const chosenInfo = students.find((m) => m.id === chosen);

  const create = useMutation({
    mutationFn: () => api.createWorkspace({ name: name.trim(), sensitivity, base_model: chosen }),
    onSuccess: (ws) => {
      void qc.invalidateQueries({ queryKey: qk.workspaces });
      onDone();
      navigate(`/workspaces/${ws.id}`);
    },
  });

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !chosen) return;
    create.mutate();
  };

  return (
    <Card title="New workspace" className="mb-6">
      <form onSubmit={submit} className="grid gap-4 md:grid-cols-2">
        <TextField
          label="Name"
          required
          autoFocus
          placeholder="e.g. Maintenance manuals line 3"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <SelectField
          label="How sensitive are the documents?"
          value={sensitivity}
          onChange={(e) => setSensitivity(e.target.value as Sensitivity)}
          hint={SENSITIVITY[sensitivity].description}
        >
          {(Object.keys(SENSITIVITY) as Sensitivity[]).map((s) => (
            <option key={s} value={s}>
              {SENSITIVITY[s].label}
            </option>
          ))}
        </SelectField>
        <div className="md:col-span-2">
          {models.isError ? (
            <ErrorBox
              title="Could not load the list of models"
              error={models.error}
              onRetry={() => void models.refetch()}
            />
          ) : (
            <SelectField
              label="Starting model"
              value={chosen}
              disabled={models.isPending}
              onChange={(e) => setBaseModel(e.target.value)}
              hint={
                chosenInfo ? (
                  <span className="flex flex-wrap items-center gap-2">
                    The model your new one is built on.
                    <Badge tone="neutral">Licence: {chosenInfo.license ?? 'unknown'}</Badge>
                  </span>
                ) : models.isPending ? (
                  'Loading installed models…'
                ) : (
                  'No models are installed. Ask an administrator to import one.'
                )
              }
            >
              {students.map((m) => (
                <option key={m.id} value={m.id} disabled={!m.installed}>
                  {m.id}
                  {m.installed ? '' : ' — not installed'}
                </option>
              ))}
            </SelectField>
          )}
        </div>
        {create.isError && (
          <div className="md:col-span-2">
            <ErrorBox title="The workspace was not created" error={create.error} />
          </div>
        )}
        <div className="flex gap-2 md:col-span-2">
          <Button type="submit" loading={create.isPending} disabled={!name.trim() || !chosen}>
            Create workspace
          </Button>
          <Button variant="secondary" onClick={onDone}>
            Cancel
          </Button>
        </div>
      </form>
    </Card>
  );
}

function WorkspaceCard({ ws }: { ws: Workspace }) {
  const run = ws.latest_run;
  return (
    <li>
      <Link
        to={`/workspaces/${ws.id}`}
        className="block h-full rounded-lg bg-white p-4 shadow-sm ring-1 ring-slate-200 transition hover:ring-indigo-400"
      >
        <div className="flex items-start justify-between gap-2">
          <h2 className="font-semibold text-slate-900 wrap-anywhere">{ws.name}</h2>
          <SensitivityBadge value={ws.sensitivity} />
        </div>
        <p className="mt-1 truncate text-xs text-slate-600" title={ws.base_model}>
          {ws.base_model}
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
          {run ? (
            <>
              <RunStateBadge state={run.state} />
              {run.current_stage && run.state === 'running' && (
                <span className="text-slate-600">{STAGE_LABELS[run.current_stage].title}</span>
              )}
            </>
          ) : (
            <span className="text-slate-600">No runs yet</span>
          )}
          {ws.deployed_adapter_id && <Badge tone="success">Live model</Badge>}
        </div>
        <p className="mt-3 text-xs text-slate-500">Created {formatDate(ws.created_at)}</p>
      </Link>
    </li>
  );
}

export function WorkspacesPage() {
  const ws = useWorkspaces();
  const [creating, setCreating] = useState(false);

  return (
    <>
      <PageHeader
        title="Workspaces"
        description="Each workspace holds one set of documents and the models trained on them."
        actions={!creating && <Button onClick={() => setCreating(true)}>New workspace</Button>}
      />
      {creating && <CreateWorkspaceForm onDone={() => setCreating(false)} />}
      {ws.isPending ? (
        <Loading label="Loading workspaces…" />
      ) : ws.isError ? (
        <ErrorBox
          title="Could not load workspaces"
          error={ws.error}
          onRetry={() => void ws.refetch()}
        />
      ) : ws.data.length === 0 ? (
        !creating && (
          <EmptyState
            title="No workspaces yet"
            action={<Button onClick={() => setCreating(true)}>Create your first workspace</Button>}
          >
            A workspace is where you add documents, pick a goal and get a tested model back — in
            four steps, without writing code.
          </EmptyState>
        )
      ) : (
        <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {ws.data.map((w) => (
            <WorkspaceCard key={w.id} ws={w} />
          ))}
        </ul>
      )}
    </>
  );
}
