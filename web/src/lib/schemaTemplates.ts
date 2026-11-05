// Mode C: JSON-schema templates and a small, dependency-free validator for the schema the user
// types. It checks the subset of JSON Schema the extraction pipeline supports and reports problems
// in plain language with the location of each problem.

export type SchemaTemplate = { id: string; title: string; description: string; schema: object };

export const SCHEMA_TEMPLATES: SchemaTemplate[] = [
  {
    id: 'lab_results',
    title: 'Lab results',
    description: 'Biomarkers with value, unit and date from a clinical note.',
    schema: {
      type: 'object',
      properties: {
        patient_ref: { type: 'string', description: 'Masked patient reference' },
        biomarkers: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              value: { type: 'number' },
              unit: { type: 'string' },
              date: { type: 'string', format: 'date' },
            },
            required: ['name', 'value', 'unit'],
          },
        },
      },
      required: ['biomarkers'],
    },
  },
  {
    id: 'incident_report',
    title: 'Incident report',
    description: 'Fault code, affected equipment, severity and actions taken.',
    schema: {
      type: 'object',
      properties: {
        fault_code: { type: 'string' },
        equipment: { type: 'string' },
        severity: { type: 'string', enum: ['low', 'medium', 'high', 'critical'] },
        occurred_at: { type: 'string', format: 'date-time' },
        actions_taken: { type: 'array', items: { type: 'string' } },
        resolved: { type: 'boolean' },
      },
      required: ['fault_code', 'severity', 'resolved'],
    },
  },
  {
    id: 'contract_terms',
    title: 'Contract terms',
    description: 'Parties, dates, amounts and notice period from an agreement.',
    schema: {
      type: 'object',
      properties: {
        parties: { type: 'array', items: { type: 'string' }, minItems: 2 },
        effective_date: { type: 'string', format: 'date' },
        end_date: { type: ['string', 'null'], format: 'date' },
        total_value: { type: 'number', minimum: 0 },
        currency: { type: 'string' },
        notice_period_days: { type: 'integer', minimum: 0 },
      },
      required: ['parties', 'effective_date'],
    },
  },
];

const TYPES = new Set(['object', 'array', 'string', 'number', 'integer', 'boolean', 'null']);

export type SchemaProblem = { path: string; message: string };
export type SchemaCheck =
  | { ok: true; schema: Record<string, unknown>; fieldCount: number }
  | { ok: false; problems: SchemaProblem[] };

function isObj(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function checkNode(node: unknown, path: string, out: SchemaProblem[], depth: number): number {
  if (depth > 8) {
    out.push({ path, message: 'The form is nested too deeply (more than 8 levels).' });
    return 0;
  }
  if (!isObj(node)) {
    out.push({
      path,
      message: 'Each field must be described by an object like {"type": "string"}.',
    });
    return 0;
  }
  const t = node.type;
  const types = Array.isArray(t) ? t : t === undefined ? [] : [t];
  if (types.length === 0 && node.enum === undefined) {
    out.push({ path, message: 'Missing "type" (for example "string", "number" or "object").' });
    return 0;
  }
  for (const ty of types) {
    if (typeof ty !== 'string' || !TYPES.has(ty)) {
      out.push({
        path,
        message: `Unknown type ${JSON.stringify(ty)}. Use one of: ${[...TYPES].join(', ')}.`,
      });
    }
  }
  if (node.enum !== undefined && (!Array.isArray(node.enum) || node.enum.length === 0)) {
    out.push({ path, message: '"enum" must be a non-empty list of allowed values.' });
  }
  let fields = 0;
  if (types.includes('object')) {
    const props = node.properties;
    if (!isObj(props) || Object.keys(props).length === 0) {
      out.push({ path, message: 'An object needs "properties" with at least one field.' });
    } else {
      for (const [k, v] of Object.entries(props)) {
        fields += 1 + checkNode(v, `${path}.${k}`, out, depth + 1);
      }
      if (node.required !== undefined) {
        if (!Array.isArray(node.required) || node.required.some((r) => typeof r !== 'string')) {
          out.push({ path, message: '"required" must be a list of field names.' });
        } else {
          for (const r of node.required as string[]) {
            if (!(r in props)) {
              out.push({
                path,
                message: `"${r}" is listed as required but is not one of the fields.`,
              });
            }
          }
        }
      }
    }
  }
  if (types.includes('array')) {
    if (node.items === undefined) {
      out.push({ path, message: 'A list needs "items" describing what each entry looks like.' });
    } else {
      fields += checkNode(node.items, `${path}[]`, out, depth + 1);
    }
  }
  return fields;
}

/** Parse and check a user-entered schema. */
export function validateSchemaText(text: string): SchemaCheck {
  if (!text.trim()) {
    return { ok: false, problems: [{ path: '$', message: 'Paste a schema or pick a template.' }] };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return {
      ok: false,
      problems: [
        {
          path: '$',
          message: `This is not valid JSON (${msg}). Check for missing commas, quotes or brackets.`,
        },
      ],
    };
  }
  if (!isObj(parsed) || parsed.type !== 'object') {
    return {
      ok: false,
      problems: [
        {
          path: '$',
          message: 'The top level must be an object: {"type": "object", "properties": {…}}.',
        },
      ],
    };
  }
  const problems: SchemaProblem[] = [];
  const fieldCount = checkNode(parsed, '$', problems, 0);
  if (problems.length) return { ok: false, problems };
  return { ok: true, schema: parsed, fieldCount };
}

export function templateText(id: string): string {
  const t = SCHEMA_TEMPLATES.find((x) => x.id === id);
  return t ? JSON.stringify(t.schema, null, 2) : '';
}
