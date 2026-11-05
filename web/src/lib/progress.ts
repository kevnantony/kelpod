import type { RunEvent, StageName } from '../api/types';

export type StageProgress = Partial<
  Record<StageName, { done: number; total: number; message?: string }>
>;

/** Latest progress event per stage. */
export function progressFromEvents(events: RunEvent[]): StageProgress {
  const out: StageProgress = {};
  for (const ev of events) {
    if (ev.type === 'progress' && ev.stage && ev.done != null && ev.total != null) {
      out[ev.stage] = { done: ev.done, total: ev.total, message: ev.message };
    }
  }
  return out;
}
