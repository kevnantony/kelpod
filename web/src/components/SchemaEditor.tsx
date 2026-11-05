import { useId } from 'react';
import { SCHEMA_TEMPLATES, templateText, validateSchemaText } from '../lib/schemaTemplates';
import { Alert, Button } from './ui';

/** Mode C: JSON-schema editor with templates and live validation. */
export function SchemaEditor({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const id = useId();
  const check = validateSchemaText(value);
  return (
    <div className="space-y-3">
      <div>
        <p className="text-sm font-medium text-slate-800">Start from a template</p>
        <div className="mt-2 grid gap-2 sm:grid-cols-3">
          {SCHEMA_TEMPLATES.map((t) => (
            <Button
              key={t.id}
              variant="secondary"
              className="h-auto flex-col items-start! text-left"
              onClick={() => onChange(templateText(t.id))}
            >
              <span className="font-semibold">{t.title}</span>
              <span className="text-xs font-normal text-slate-600">{t.description}</span>
            </Button>
          ))}
        </div>
      </div>
      <div>
        <label htmlFor={id} className="block text-sm font-medium text-slate-800">
          The form to fill in (JSON schema)
        </label>
        <textarea
          id={id}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          spellCheck={false}
          rows={14}
          aria-invalid={!check.ok && value.trim() !== ''}
          aria-describedby={`${id}-status`}
          placeholder='{"type": "object", "properties": {"name": {"type": "string"}}}'
          className="mt-1 block w-full rounded-md bg-white px-3 py-2 font-mono text-xs text-slate-900 ring-1 ring-inset ring-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-600"
        />
      </div>
      <div id={`${id}-status`} aria-live="polite">
        {check.ok ? (
          <Alert tone="success">
            Looks good — {check.fieldCount} field{check.fieldCount === 1 ? '' : 's'} will be
            extracted.
          </Alert>
        ) : value.trim() === '' ? (
          <p className="text-sm text-slate-600">Pick a template above or paste your own schema.</p>
        ) : (
          <Alert tone="danger" title="The schema needs fixing">
            <ul className="list-disc pl-5">
              {check.problems.map((p, i) => (
                <li key={i}>
                  <code className="font-mono text-xs">{p.path}</code>: {p.message}
                </li>
              ))}
            </ul>
          </Alert>
        )}
      </div>
    </div>
  );
}
