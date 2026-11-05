// Parse a user's gold question set from CSV or JSONL into the API's GoldItem shape.
import type { GoldItem } from '../api/types';

export type GoldParseResult = { items: GoldItem[]; errors: string[] };

/** RFC-4180-ish CSV parser (quoted fields, escaped quotes, CRLF). */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let inQuotes = false;
  const src = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < src.length; i++) {
    const c = src[i];
    if (inQuotes) {
      if (c === '"') {
        if (src[i + 1] === '"') {
          field += '"';
          i++;
        } else inQuotes = false;
      } else field += c;
    } else if (c === '"') inQuotes = true;
    else if (c === ',') {
      row.push(field);
      field = '';
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && src[i + 1] === '\n') i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
    } else field += c;
  }
  if (field !== '' || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((r) => r.some((f) => f.trim() !== ''));
}

function parseBool(v: unknown): boolean | null {
  if (typeof v === 'boolean') return v;
  if (v === undefined || v === null || v === '') return null;
  const s = String(v).trim().toLowerCase();
  if (['true', 'yes', 'y', '1'].includes(s)) return true;
  if (['false', 'no', 'n', '0'].includes(s)) return false;
  return null;
}

function toItem(raw: Record<string, unknown>, where: string, errors: string[]): GoldItem | null {
  const question = typeof raw.question === 'string' ? raw.question.trim() : '';
  if (!question) {
    errors.push(`${where}: the "question" is empty.`);
    return null;
  }
  const item: GoldItem = { question, answerable: true };
  const ans = raw.reference_answer ?? raw.answer;
  if (typeof ans === 'string' && ans.trim()) item.reference_answer = ans.trim();
  let rj = raw.reference_json;
  if (typeof rj === 'string' && rj.trim()) {
    try {
      rj = JSON.parse(rj);
    } catch {
      errors.push(`${where}: "reference_json" is not valid JSON.`);
      return null;
    }
  }
  if (rj && typeof rj === 'object' && !Array.isArray(rj)) {
    item.reference_json = rj as Record<string, unknown>;
  }
  const a = parseBool(raw.answerable);
  if (a === null) {
    // Default: answerable when an answer is given; a question without an answer is treated as
    // one the documents do not answer.
    item.answerable = Boolean(item.reference_answer || item.reference_json);
  } else item.answerable = a;
  return item;
}

export function parseGold(text: string, filename: string): GoldParseResult {
  const errors: string[] = [];
  const items: GoldItem[] = [];
  const isJsonl = /\.jsonl?$/i.test(filename) || text.trimStart().startsWith('{');
  if (isJsonl) {
    text.split(/\r?\n/).forEach((line, i) => {
      if (!line.trim()) return;
      try {
        const obj: unknown = JSON.parse(line);
        if (!obj || typeof obj !== 'object' || Array.isArray(obj)) {
          errors.push(`Line ${i + 1}: expected an object.`);
          return;
        }
        const it = toItem(obj as Record<string, unknown>, `Line ${i + 1}`, errors);
        if (it) items.push(it);
      } catch {
        errors.push(`Line ${i + 1}: not valid JSON.`);
      }
    });
    return { items, errors };
  }
  const rows = parseCsv(text);
  const header = rows[0]?.map((h) => h.trim().toLowerCase());
  if (!header || !header.includes('question')) {
    return {
      items: [],
      errors: ['The CSV needs a header row with at least a "question" column.'],
    };
  }
  rows.slice(1).forEach((r, i) => {
    const raw: Record<string, unknown> = {};
    header.forEach((h, j) => (raw[h] = r[j] ?? ''));
    const it = toItem(raw, `Row ${i + 2}`, errors);
    if (it) items.push(it);
  });
  return { items, errors };
}
