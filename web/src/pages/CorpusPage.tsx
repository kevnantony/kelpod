import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Link, useOutletContext } from 'react-router-dom';
import { api, type UploadItem } from '../api/client';
import { qk, useDocuments } from '../api/hooks';
import type { DocumentInfo } from '../api/types';
import { DocStatusBadge } from '../components/badges';
import { DropZone } from '../components/DropZone';
import {
  Alert,
  Button,
  Card,
  EmptyState,
  ErrorBox,
  Loading,
  ProgressBar,
  TableWrap,
  td,
  th,
} from '../components/ui';
import { formatBytes } from '../lib/format';
import type { WorkspaceOutletContext } from './WorkspaceLayout';

function typeLabel(d: DocumentInfo): string {
  const ext = d.name.includes('.') ? d.name.split('.').pop()!.toUpperCase() : '';
  return ext || d.mime;
}

function DocRow({ wsId, d }: { wsId: string; d: DocumentInfo }) {
  const qc = useQueryClient();
  const del = useMutation({
    mutationFn: () => api.deleteDocument(wsId, d.id),
    onSuccess: () => void qc.invalidateQueries({ queryKey: qk.documents(wsId) }),
  });
  return (
    <tr>
      <td className={`${td} max-w-[18rem]`}>
        <span className="block break-all font-medium text-slate-900">{d.name}</span>
        {d.error && <span className="mt-0.5 block text-xs text-red-700">{d.error}</span>}
        {del.isError && (
          <span className="mt-0.5 block text-xs text-red-700">{(del.error as Error).message}</span>
        )}
      </td>
      <td className={td}>{typeLabel(d)}</td>
      <td className={`${td} whitespace-nowrap`}>{formatBytes(d.size_bytes)}</td>
      <td className={td}>{d.pages ?? '—'}</td>
      <td className={td}>
        <DocStatusBadge status={d.status} />
      </td>
      <td className={`${td} text-right`}>
        <Button
          variant="ghost"
          size="sm"
          aria-label={`Remove ${d.name}`}
          loading={del.isPending}
          onClick={() => del.mutate()}
        >
          Remove
        </Button>
      </td>
    </tr>
  );
}

export function CorpusPage() {
  const { workspace } = useOutletContext<WorkspaceOutletContext>();
  const docs = useDocuments(workspace.id);
  const qc = useQueryClient();
  const [progress, setProgress] = useState<{ loaded: number; total: number; files: number } | null>(
    null,
  );

  const upload = useMutation({
    mutationFn: (items: UploadItem[]) => {
      const total = items.reduce((s, i) => s + i.file.size, 0);
      setProgress({ loaded: 0, total, files: items.length });
      return api.uploadDocuments(workspace.id, items, (loaded, t) =>
        setProgress({ loaded, total: t, files: items.length }),
      );
    },
    onSettled: () => {
      setProgress(null);
      void qc.invalidateQueries({ queryKey: qk.documents(workspace.id) });
    },
  });

  const list = docs.data ?? [];
  const counts = {
    parsed: list.filter((d) => d.status === 'parsed').length,
    pending: list.filter((d) => d.status === 'uploaded').length,
    problems: list.filter((d) => d.status === 'failed' || d.status === 'skipped').length,
  };

  return (
    <div className="space-y-6">
      <section aria-labelledby="add-docs">
        <h2 id="add-docs" className="mb-2 text-lg font-semibold">
          Add documents
        </h2>
        <p className="mb-3 text-sm text-slate-600">
          Files are stored only on this appliance. Before anything is used for training you will see
          what the privacy filter removed and approve it.
        </p>
        <DropZone onFiles={(items) => upload.mutate(items)} disabled={upload.isPending} />
        {progress && (
          <div className="mt-3" aria-live="polite">
            <div className="mb-1 flex justify-between text-sm text-slate-700">
              <span>
                Uploading {progress.files} file{progress.files === 1 ? '' : 's'}…
              </span>
              <span>
                {formatBytes(progress.loaded)} of {formatBytes(progress.total)}
              </span>
            </div>
            <ProgressBar
              value={progress.loaded}
              max={progress.total || 1}
              label="Upload progress"
            />
          </div>
        )}
        {upload.isError && (
          <div className="mt-3">
            <ErrorBox
              title="Some files were not uploaded"
              error={upload.error}
              onRetry={upload.variables ? () => upload.mutate(upload.variables) : undefined}
            />
          </div>
        )}
        {upload.isSuccess && !progress && (
          <div className="mt-3">
            <Alert tone="success">
              {upload.data.length} file{upload.data.length === 1 ? '' : 's'} uploaded. They are
              being read now.
            </Alert>
          </div>
        )}
      </section>

      <Card
        title={`Documents${list.length ? ` (${list.length})` : ''}`}
        actions={
          counts.parsed > 0 && (
            <Link
              to={`/workspaces/${workspace.id}/goal`}
              className="rounded-md bg-indigo-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-800"
            >
              Next: choose a goal
            </Link>
          )
        }
      >
        {docs.isPending ? (
          <Loading label="Loading documents…" />
        ) : docs.isError ? (
          <ErrorBox
            title="Could not load documents"
            error={docs.error}
            onRetry={() => void docs.refetch()}
          />
        ) : list.length === 0 ? (
          <EmptyState title="No documents yet">
            Drop a folder of manuals, reports or notes above to get started.
          </EmptyState>
        ) : (
          <>
            <p className="mb-3 text-sm text-slate-700" aria-live="polite">
              {counts.parsed} ready
              {counts.pending > 0 && ` · ${counts.pending} still being read`}
              {counts.problems > 0 && ` · ${counts.problems} need attention (see notes below)`}
            </p>
            <TableWrap label="Documents">
              <thead>
                <tr>
                  <th className={th}>Name</th>
                  <th className={th}>Type</th>
                  <th className={th}>Size</th>
                  <th className={th}>Pages</th>
                  <th className={th}>Status</th>
                  <th className={th}>
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {list.map((d) => (
                  <DocRow key={d.id} wsId={workspace.id} d={d} />
                ))}
              </tbody>
            </TableWrap>
          </>
        )}
      </Card>
    </div>
  );
}
