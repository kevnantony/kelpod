import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../api/client';
import { qk, useGold } from '../api/hooks';
import type { GoldItem } from '../api/types';
import { parseGold } from '../lib/gold';
import { Alert, Button, Card, ErrorBox, Spinner } from './ui';

/** Optional: the user's own test questions (gold set), uploaded as CSV or JSONL. */
export function GoldSetCard({ workspaceId }: { workspaceId: string }) {
  const gold = useGold(workspaceId);
  const qc = useQueryClient();
  const [parseErrors, setParseErrors] = useState<string[]>([]);
  const save = useMutation({
    mutationFn: (items: GoldItem[]) => api.putGold(workspaceId, items),
    onSuccess: (items) => qc.setQueryData(qk.gold(workspaceId), items),
  });

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    const text = await file.text();
    const { items, errors } = parseGold(text, file.name);
    setParseErrors(errors);
    if (items.length) save.mutate(items);
  };

  const count = gold.data?.length ?? 0;
  const unanswerable = gold.data?.filter((g) => !g.answerable).length ?? 0;

  return (
    <Card title="Your own test questions (recommended)">
      <p className="text-sm text-slate-600">
        A few dozen real questions with the answers you expect make the final quality report much
        more trustworthy. Upload a CSV with columns <code>question</code>,{' '}
        <code>reference_answer</code> and optionally <code>answerable</code> (yes/no), or a JSONL
        file with the same fields.
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <label className="inline-flex cursor-pointer items-center rounded-md bg-white px-4 py-2 text-sm font-medium text-slate-800 ring-1 ring-inset ring-slate-300 hover:bg-slate-50 focus-within:outline-2 focus-within:outline-indigo-600">
          {count ? 'Replace questions…' : 'Upload questions…'}
          <input
            type="file"
            accept=".csv,.jsonl,.json,text/csv,application/json"
            className="sr-only"
            onChange={(e) => {
              void onFile(e.target.files?.[0]);
              e.target.value = '';
            }}
          />
        </label>
        {gold.isPending ? (
          <Spinner className="h-4 w-4" />
        ) : (
          <span className="text-sm text-slate-700" aria-live="polite">
            {count
              ? `${count} question${count === 1 ? '' : 's'} saved${unanswerable ? `, ${unanswerable} without an answer in the documents` : ''}.`
              : 'No test questions yet.'}
          </span>
        )}
        {count > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => save.mutate([])}
            loading={save.isPending}
          >
            Remove all
          </Button>
        )}
      </div>
      {parseErrors.length > 0 && (
        <div className="mt-3">
          <Alert tone="warning" title="Some lines were skipped">
            <ul className="list-disc pl-5">
              {parseErrors.slice(0, 8).map((e, i) => (
                <li key={i}>{e}</li>
              ))}
              {parseErrors.length > 8 && <li>…and {parseErrors.length - 8} more.</li>}
            </ul>
          </Alert>
        </div>
      )}
      {save.isError && (
        <div className="mt-3">
          <ErrorBox title="The questions were not saved" error={save.error} />
        </div>
      )}
      {gold.isError && (
        <div className="mt-3">
          <ErrorBox
            title="Could not load saved questions"
            error={gold.error}
            onRetry={() => void gold.refetch()}
          />
        </div>
      )}
    </Card>
  );
}
