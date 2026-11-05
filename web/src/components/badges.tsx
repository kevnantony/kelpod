import type { DocumentStatus, RunState, Sensitivity } from '../api/types';
import { DOC_STATUS_LABELS, RUN_STATE_LABELS, SENSITIVITY } from '../lib/copy';
import { Badge } from './ui';

const RUN_TONES: Record<RunState, 'neutral' | 'success' | 'warning' | 'danger' | 'info'> = {
  queued: 'neutral',
  running: 'info',
  awaiting_approval: 'warning',
  completed: 'success',
  failed: 'danger',
  cancelled: 'neutral',
};

export function RunStateBadge({ state }: { state: RunState }) {
  return <Badge tone={RUN_TONES[state]}>{RUN_STATE_LABELS[state]}</Badge>;
}

const SENS_TONES: Record<Sensitivity, 'neutral' | 'info' | 'warning' | 'danger'> = {
  public: 'neutral',
  internal: 'info',
  confidential: 'warning',
  restricted: 'danger',
};

export function SensitivityBadge({ value }: { value: Sensitivity }) {
  return (
    <Badge tone={SENS_TONES[value]}>
      <span className="sr-only">Sensitivity: </span>
      {SENSITIVITY[value].label}
    </Badge>
  );
}

const DOC_TONES: Record<DocumentStatus, 'neutral' | 'success' | 'warning' | 'danger'> = {
  uploaded: 'neutral',
  parsed: 'success',
  skipped: 'warning',
  failed: 'danger',
};

export function DocStatusBadge({ status }: { status: DocumentStatus }) {
  return <Badge tone={DOC_TONES[status]}>{DOC_STATUS_LABELS[status]}</Badge>;
}

export function PassFail({ passed }: { passed: boolean }) {
  return passed ? (
    <Badge tone="success">
      <svg viewBox="0 0 20 20" className="h-3.5 w-3.5" aria-hidden="true" fill="currentColor">
        <path d="M16.7 5.3a1 1 0 0 1 0 1.4l-8 8a1 1 0 0 1-1.4 0l-4-4a1 1 0 1 1 1.4-1.4L8 12.6l7.3-7.3a1 1 0 0 1 1.4 0z" />
      </svg>
      Passed
    </Badge>
  ) : (
    <Badge tone="danger">
      <svg viewBox="0 0 20 20" className="h-3.5 w-3.5" aria-hidden="true" fill="currentColor">
        <path d="M5.3 5.3a1 1 0 0 1 1.4 0L10 8.6l3.3-3.3a1 1 0 1 1 1.4 1.4L11.4 10l3.3 3.3a1 1 0 0 1-1.4 1.4L10 11.4l-3.3 3.3a1 1 0 0 1-1.4-1.4L8.6 10 5.3 6.7a1 1 0 0 1 0-1.4z" />
      </svg>
      Not passed
    </Badge>
  );
}
