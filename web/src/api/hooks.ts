// TanStack Query hooks for the API. Query keys are centralised in `qk`.
import { useEffect, useState } from 'react';
import { useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query';
import { api, isApiError, subscribeRunEvents } from './client';
import type { RunEvent, RunState } from './types';

export const qk = {
  me: ['me'] as const,
  setup: ['setup'] as const,
  workspaces: ['workspaces'] as const,
  workspace: (id: string) => ['workspaces', id] as const,
  documents: (id: string) => ['workspaces', id, 'documents'] as const,
  runs: (id: string) => ['workspaces', id, 'runs'] as const,
  gold: (id: string) => ['workspaces', id, 'gold'] as const,
  models: ['models'] as const,
  run: (id: string) => ['runs', id] as const,
  redactions: (id: string) => ['runs', id, 'redactions'] as const,
  trials: (id: string) => ['runs', id, 'trials'] as const,
  scorecard: (id: string) => ['runs', id, 'scorecard'] as const,
  adapters: ['adapters'] as const,
  deployments: ['deployments'] as const,
  health: ['system', 'health'] as const,
  hardware: ['system', 'hardware'] as const,
};

export const ACTIVE_STATES: RunState[] = ['queued', 'running'];

export const useMe = () =>
  useQuery({ queryKey: qk.me, queryFn: api.me, retry: false, staleTime: 60_000 });
export const useSetupStatus = () =>
  useQuery({ queryKey: qk.setup, queryFn: api.setupStatus, retry: false });
export const useWorkspaces = () =>
  useQuery({ queryKey: qk.workspaces, queryFn: api.listWorkspaces, refetchInterval: 10_000 });
export const useWorkspace = (id: string) =>
  useQuery({ queryKey: qk.workspace(id), queryFn: () => api.getWorkspace(id), enabled: !!id });
export const useModels = () =>
  useQuery({ queryKey: qk.models, queryFn: api.listModels, staleTime: 5 * 60_000 });
export const useRuns = (wsId: string) =>
  useQuery({ queryKey: qk.runs(wsId), queryFn: () => api.listRuns(wsId) });
export const useGold = (wsId: string) =>
  useQuery({ queryKey: qk.gold(wsId), queryFn: () => api.getGold(wsId) });
export const useAdapters = () => useQuery({ queryKey: qk.adapters, queryFn: api.listAdapters });
export const useDeployments = () =>
  useQuery({
    queryKey: qk.deployments,
    queryFn: api.listDeployments,
    // Poll while something is starting up.
    refetchInterval: (q) => (q.state.data?.some((d) => d.state === 'starting') ? 1500 : false),
  });

export function useDocuments(wsId: string) {
  return useQuery({
    queryKey: qk.documents(wsId),
    queryFn: () => api.listDocuments(wsId),
    // Keep polling while documents are still being read.
    refetchInterval: (q) => (q.state.data?.some((d) => d.status === 'uploaded') ? 2000 : false),
  });
}

export function useRun(runId: string) {
  return useQuery({
    queryKey: qk.run(runId),
    queryFn: () => api.getRun(runId),
    // Events drive most updates; polling is a fallback if the event stream drops.
    refetchInterval: (q) =>
      q.state.data && ACTIVE_STATES.includes(q.state.data.state) ? 5000 : false,
  });
}

/** 404 = "not ready yet" for scorecard/redactions: treat as no data, don't retry. */
const notReadyRetry = (count: number, e: unknown) =>
  !(isApiError(e) && (e.status === 404 || e.status === 401)) && count < 2;

export const useRedactions = (runId: string, enabled: boolean) =>
  useQuery({
    queryKey: qk.redactions(runId),
    queryFn: () => api.getRedactions(runId),
    enabled,
    retry: notReadyRetry,
  });
export const useTrials = (runId: string, enabled = true) =>
  useQuery({ queryKey: qk.trials(runId), queryFn: () => api.getTrials(runId), enabled });
export const useScorecard = (runId: string, enabled = true) =>
  useQuery({
    queryKey: qk.scorecard(runId),
    queryFn: () => api.getScorecard(runId),
    enabled,
    retry: notReadyRetry,
  });

function invalidateForEvent(qc: QueryClient, runId: string, ev: RunEvent) {
  if (ev.type === 'progress') return; // progress is rendered from the event itself
  void qc.invalidateQueries({ queryKey: qk.run(runId), exact: true });
  if (ev.type === 'trial_completed' || ev.stage === 'train')
    void qc.invalidateQueries({ queryKey: qk.trials(runId) });
  if (ev.type === 'stage_completed' && ev.stage === 'evaluate')
    void qc.invalidateQueries({ queryKey: qk.scorecard(runId) });
  if (ev.type === 'awaiting_approval')
    void qc.invalidateQueries({ queryKey: qk.redactions(runId) });
  if (ev.type === 'run_completed' || ev.type === 'stage_completed') {
    void qc.invalidateQueries({ queryKey: qk.adapters });
    void qc.invalidateQueries({ queryKey: qk.workspaces });
  }
}

/**
 * Live run events via SSE. Returns the event log (replayed history + live) and connection
 * state. The browser's EventSource reconnects on its own; on reconnect the server replays
 * history, so the log is deduplicated by event content.
 */
export function useRunEvents(runId: string) {
  const qc = useQueryClient();
  const [log, setLog] = useState<{ runId: string; events: RunEvent[] }>({ runId, events: [] });
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const seen = new Set<string>();
    const unsub = subscribeRunEvents(runId, {
      onOpen: () => setConnected(true),
      onError: () => setConnected(false),
      onEvent: (ev) => {
        setConnected(true);
        const key = JSON.stringify(ev);
        if (seen.has(key)) return;
        seen.add(key);
        setLog((prev) =>
          prev.runId === runId ? { runId, events: [...prev.events, ev] } : { runId, events: [ev] },
        );
        invalidateForEvent(qc, runId, ev);
      },
    });
    return unsub;
  }, [runId, qc]);

  return { events: log.runId === runId ? log.events : [], connected };
}
