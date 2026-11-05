import { describe, expect, it } from 'vitest';
import { SCHEMA_TEMPLATES, templateText, validateSchemaText } from './schemaTemplates';

describe('schema templates', () => {
  it('has 2–3+ templates that all validate', () => {
    expect(SCHEMA_TEMPLATES.length).toBeGreaterThanOrEqual(3);
    for (const t of SCHEMA_TEMPLATES) {
      const r = validateSchemaText(templateText(t.id));
      expect(r.ok, t.id).toBe(true);
      if (r.ok) expect(r.fieldCount).toBeGreaterThan(2);
    }
  });

  it('counts nested fields', () => {
    const r = validateSchemaText(templateText('lab_results'));
    // patient_ref, biomarkers + name, value, unit, date
    expect(r).toMatchObject({ ok: true, fieldCount: 6 });
  });
});

describe('validateSchemaText', () => {
  const problems = (text: string) => {
    const r = validateSchemaText(text);
    if (r.ok) throw new Error('expected problems');
    return r.problems;
  };

  it('rejects empty input and invalid JSON with guidance', () => {
    expect(problems('')[0]!.message).toMatch(/template/);
    expect(problems('{"type": "object",')[0]!.message).toMatch(/not valid JSON/);
  });

  it('requires a top-level object with fields', () => {
    expect(problems('[]')[0]!.message).toMatch(/top level/);
    expect(problems('{"type":"string"}')[0]!.message).toMatch(/top level/);
    expect(problems('{"type":"object"}')[0]!.message).toMatch(/at least one field/);
  });

  it('reports unknown types, missing items and bad required with paths', () => {
    const p = problems(
      JSON.stringify({
        type: 'object',
        properties: {
          a: { type: 'text' },
          b: { type: 'array' },
          c: { description: 'no type' },
        },
        required: ['a', 'zzz'],
      }),
    );
    const byPath = Object.fromEntries(p.map((x) => [x.path, x.message]));
    expect(byPath['$.a']).toMatch(/Unknown type "text"/);
    expect(byPath['$.b']).toMatch(/items/);
    expect(byPath['$.c']).toMatch(/Missing "type"/);
    expect(p.some((x) => x.path === '$' && /zzz/.test(x.message))).toBe(true);
  });

  it('accepts type unions and enums', () => {
    const r = validateSchemaText(
      JSON.stringify({
        type: 'object',
        properties: { a: { type: ['string', 'null'] }, b: { enum: ['x', 'y'] } },
      }),
    );
    expect(r.ok).toBe(true);
  });
});
