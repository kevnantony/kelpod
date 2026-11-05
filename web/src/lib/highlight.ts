import type { RedactionSpan } from '../api/types';

type Span = RedactionSpan;

const PLACEHOLDER = /\[[A-Z][A-Z0-9_]*?_\d+\]/g;

/** Split masked text into plain and highlighted parts using the spans (or placeholder pattern). */
export function highlightParts(text: string, spans: Span[]): { text: string; span?: Span }[] {
  let use = spans
    .filter((s) => s.start >= 0 && s.end <= text.length && s.end > s.start)
    .sort((a, b) => a.start - b.start);
  if (use.length === 0) {
    use = [...text.matchAll(PLACEHOLDER)].map((m) => ({
      start: m.index,
      end: m.index + m[0].length,
      entity_type: m[0].slice(1, -1).replace(/_\d+$/, ''),
      action: 'mask',
    }));
  }
  const out: { text: string; span?: Span }[] = [];
  let pos = 0;
  for (const s of use) {
    if (s.start < pos) continue; // overlapping span — skip
    if (s.start > pos) out.push({ text: text.slice(pos, s.start) });
    out.push({ text: text.slice(s.start, s.end), span: s });
    pos = s.end;
  }
  if (pos < text.length) out.push({ text: text.slice(pos) });
  return out;
}
