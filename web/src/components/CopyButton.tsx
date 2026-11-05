import { useState } from 'react';
import { Button } from './ui';

async function copyText(text: string): Promise<boolean> {
  try {
    // Only available in secure contexts (https or localhost).
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through to legacy path
  }
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  let ok: boolean;
  try {
    ok = document.execCommand('copy');
  } catch {
    ok = false;
  }
  ta.remove();
  return ok;
}

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [state, setState] = useState<'idle' | 'ok' | 'fail'>('idle');
  return (
    <Button
      variant="secondary"
      size="sm"
      onClick={async () => {
        setState((await copyText(text)) ? 'ok' : 'fail');
        setTimeout(() => setState('idle'), 2000);
      }}
    >
      <span aria-live="polite">
        {state === 'ok' ? 'Copied' : state === 'fail' ? 'Select and copy manually' : label}
      </span>
    </Button>
  );
}

export function CodeBlock({ code, label }: { code: string; label: string }) {
  return (
    <div className="rounded-md bg-slate-900 text-slate-100">
      <div className="flex items-center justify-between gap-2 border-b border-slate-700 px-3 py-1.5">
        <span className="text-xs text-slate-300">{label}</span>
        <CopyButton text={code} />
      </div>
      <pre className="overflow-x-auto p-3 text-xs leading-relaxed" tabIndex={0}>
        <code>{code}</code>
      </pre>
    </div>
  );
}
