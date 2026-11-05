import { useState, type FormEvent } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Link, useNavigate, useOutletContext } from 'react-router-dom';
import { api } from '../api/client';
import { qk, useDocuments } from '../api/hooks';
import type { Mode, StartRunBody } from '../api/types';
import { GoldSetCard } from '../components/GoldSetCard';
import { SchemaEditor } from '../components/SchemaEditor';
import { Alert, Button, Card, ErrorBox, SelectField, TextField } from '../components/ui';
import { cx } from '../lib/cx';
import { MODES } from '../lib/copy';
import { formatDuration } from '../lib/format';
import { validateSchemaText } from '../lib/schemaTemplates';
import { secondsUntil } from '../lib/time';
import type { WorkspaceOutletContext } from './WorkspaceLayout';

type BudgetChoice = 'none' | '7200' | '14400' | '28800' | 'until';

function ModeCards({ value, onChange }: { value: Mode; onChange: (m: Mode) => void }) {
  return (
    <fieldset>
      <legend className="text-lg font-semibold">What should the new model be good at?</legend>
      <div className="mt-3 grid gap-3 md:grid-cols-3">
        {(Object.keys(MODES) as Mode[]).map((m) => {
          const checked = value === m;
          return (
            <label
              key={m}
              className={cx(
                'relative flex cursor-pointer flex-col rounded-lg bg-white p-4 shadow-sm ring-1 transition focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-indigo-600',
                checked ? 'ring-2 ring-indigo-600' : 'ring-slate-200 hover:ring-slate-400',
              )}
            >
              <input
                type="radio"
                name="mode"
                value={m}
                checked={checked}
                onChange={() => onChange(m)}
                className="sr-only"
              />
              <span className="flex items-center justify-between gap-2">
                <span className="font-semibold text-slate-900">{MODES[m].title}</span>
                <span
                  aria-hidden="true"
                  className={cx(
                    'h-4 w-4 shrink-0 rounded-full border-2',
                    checked
                      ? 'border-indigo-600 bg-indigo-600 ring-2 ring-inset ring-white'
                      : 'border-slate-400',
                  )}
                />
              </span>
              <span className="mt-2 text-sm text-slate-600">{MODES[m].summary}</span>
              <span className="mt-3 text-sm italic text-slate-500">{MODES[m].example}</span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

export function GoalPage() {
  const { workspace } = useOutletContext<WorkspaceOutletContext>();
  const docs = useDocuments(workspace.id);
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [mode, setMode] = useState<Mode>('A');
  const [schemaText, setSchemaText] = useState('');
  const [budget, setBudget] = useState<BudgetChoice>('none');
  const [until, setUntil] = useState('07:00');

  const readyDocs = docs.data?.filter((d) => d.status === 'parsed').length ?? 0;
  const schemaCheck = validateSchemaText(schemaText);
  const budgetSeconds =
    budget === 'none'
      ? undefined
      : budget === 'until'
        ? (secondsUntil(until) ?? undefined)
        : Number(budget);
  const active =
    workspace.latest_run &&
    ['queued', 'running', 'awaiting_approval'].includes(workspace.latest_run.state);

  const start = useMutation({
    mutationFn: () => {
      const body: StartRunBody = { mode };
      if (mode === 'C' && schemaCheck.ok) body.json_schema = schemaCheck.schema;
      if (budgetSeconds) body.budget_seconds = budgetSeconds;
      return api.startRun(workspace.id, body);
    },
    onSuccess: (run) => {
      void qc.invalidateQueries({ queryKey: qk.workspace(workspace.id) });
      void qc.invalidateQueries({ queryKey: qk.runs(workspace.id) });
      void qc.invalidateQueries({ queryKey: qk.workspaces });
      navigate(`/runs/${run.id}`);
    },
  });

  const blocker = active
    ? 'A run is already in progress for this workspace.'
    : docs.isSuccess && readyDocs === 0
      ? 'Add documents first and wait until at least one shows as “Ready”.'
      : mode === 'C' && !schemaCheck.ok
        ? 'Fix the form (JSON schema) above before starting.'
        : budget === 'until' && !budgetSeconds
          ? 'Enter a valid time, e.g. 07:00.'
          : null;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!blocker) start.mutate();
  };

  return (
    <form onSubmit={submit} className="space-y-6">
      <ModeCards value={mode} onChange={setMode} />

      {mode === 'C' && (
        <Card title="Which fields should be extracted?">
          <SchemaEditor value={schemaText} onChange={setSchemaText} />
        </Card>
      )}

      <Card title="When do you need it?">
        <div className="grid gap-4 sm:grid-cols-2">
          <SelectField
            label="Time limit"
            value={budget}
            onChange={(e) => setBudget(e.target.value as BudgetChoice)}
            hint="With a shorter limit the appliance tries fewer settings. Quality checks still apply."
          >
            <option value="none">No limit — take the time needed</option>
            <option value="7200">About 2 hours</option>
            <option value="14400">About 4 hours</option>
            <option value="28800">Overnight (about 8 hours)</option>
            <option value="until">Ready by a specific time…</option>
          </SelectField>
          {budget === 'until' && (
            <TextField
              label="Ready by"
              type="time"
              value={until}
              onChange={(e) => setUntil(e.target.value)}
              hint={
                budgetSeconds ? `That is ${formatDuration(budgetSeconds)} from now.` : undefined
              }
            />
          )}
        </div>
      </Card>

      <GoldSetCard workspaceId={workspace.id} />

      <Card title="Start">
        <p className="text-sm text-slate-700">
          {readyDocs} document{readyDocs === 1 ? '' : 's'} ready. The run first reads your documents
          and applies the privacy filter, then <strong>pauses for your approval</strong> before
          anything is used for training. You can close the browser at any time.
        </p>
        {blocker && (
          <div className="mt-3">
            <Alert tone="warning">
              {blocker}{' '}
              {readyDocs === 0 && !active && (
                <Link className="underline" to={`/workspaces/${workspace.id}`}>
                  Go to documents
                </Link>
              )}
              {active && workspace.latest_run && (
                <Link className="underline" to={`/runs/${workspace.latest_run.id}`}>
                  Open the current run
                </Link>
              )}
            </Alert>
          </div>
        )}
        {start.isError && (
          <div className="mt-3">
            <ErrorBox title="The run did not start" error={start.error} />
          </div>
        )}
        <Button type="submit" className="mt-4" loading={start.isPending} disabled={!!blocker}>
          Start building the model
        </Button>
      </Card>
    </form>
  );
}
