import type { Trial } from '../api/types';
import { formatDuration } from '../lib/format';
import { Badge, TableWrap, td, th } from './ui';

const STATE_TEXT: Record<
  string,
  { label: string; tone: 'success' | 'neutral' | 'danger' | 'info' }
> = {
  complete: { label: 'Finished', tone: 'success' },
  pruned: { label: 'Stopped early', tone: 'neutral' },
  failed: { label: 'Failed', tone: 'danger' },
  failed_oom: { label: 'Too big for memory', tone: 'danger' },
  running: { label: 'Running', tone: 'info' },
};

/**
 * Leaderboard of the training settings tried. Lower score is better (it is the combined
 * objective the optimiser minimises).
 */
export function TrialTable({ trials }: { trials: Trial[] }) {
  const ranked = [...trials].sort((a, b) => {
    if (a.objective == null) return 1;
    if (b.objective == null) return -1;
    return a.objective - b.objective;
  });
  const best = ranked[0]?.objective != null ? ranked[0].number : null;
  return (
    <TableWrap label="Training settings tried">
      <thead>
        <tr>
          <th className={th}>Rank</th>
          <th className={th}>Attempt</th>
          <th className={th}>Result</th>
          <th className={th}>
            Score <span className="font-normal text-slate-500">(lower is better)</span>
          </th>
          <th className={th}>Time</th>
        </tr>
      </thead>
      <tbody>
        {ranked.map((t, i) => {
          const st = STATE_TEXT[t.state] ?? { label: t.state, tone: 'neutral' as const };
          return (
            <tr key={t.number} className={t.number === best ? 'bg-emerald-50' : undefined}>
              <td className={td}>{t.objective == null ? '—' : i + 1}</td>
              <td className={td}>
                #{t.number + 1}
                {t.number === best && (
                  <Badge tone="success" className="ml-2">
                    Best
                  </Badge>
                )}
              </td>
              <td className={td}>
                <Badge tone={st.tone}>{st.label}</Badge>
              </td>
              <td className={`${td} tabular-nums`}>
                {t.objective == null ? '—' : t.objective.toFixed(3)}
              </td>
              <td className={td}>{formatDuration(t.seconds)}</td>
            </tr>
          );
        })}
      </tbody>
    </TableWrap>
  );
}
