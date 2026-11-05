// Plain-language copy (PRD §7.3): no ML vocabulary on the default path.
// All user-visible names for stages, quality checks, metrics and states live here so they
// can be translated later (EN first, DE later).
import type { DocumentStatus, Mode, RunState, Sensitivity, StageName, Variant } from '../api/types';

export const STAGE_LABELS: Record<StageName, { title: string; description: string }> = {
  ingest: { title: 'Read documents', description: 'Opening files and extracting their text.' },
  sanitize: {
    title: 'Privacy filter',
    description: 'Finding personal and sensitive details and masking them.',
  },
  synth: {
    title: 'Training examples',
    description: 'Writing practice questions and answers from your documents.',
  },
  train: {
    title: 'Training',
    description: 'Teaching the model, trying several settings and keeping the best.',
  },
  evaluate: {
    title: 'Quality checks',
    description: 'Testing the new model against the original on questions it has never seen.',
  },
  package: {
    title: 'Packaging',
    description: 'Bundling the model with its audit report so it can be deployed.',
  },
};

export const RUN_STATE_LABELS: Record<RunState, string> = {
  queued: 'Waiting to start',
  running: 'In progress',
  awaiting_approval: 'Needs your review',
  completed: 'Finished',
  failed: 'Stopped with a problem',
  cancelled: 'Cancelled',
};

export const DOC_STATUS_LABELS: Record<DocumentStatus, string> = {
  uploaded: 'Waiting to be read',
  parsed: 'Ready',
  skipped: 'Skipped',
  failed: 'Could not read',
};

export const SENSITIVITY: Record<Sensitivity, { label: string; description: string }> = {
  public: { label: 'Public', description: 'Can be shared with anyone.' },
  internal: { label: 'Internal', description: 'For staff of your organisation only.' },
  confidential: {
    label: 'Confidential',
    description: 'Limited to people who need it; contains personal or business-sensitive data.',
  },
  restricted: {
    label: 'Restricted',
    description: 'Highest protection, e.g. patient records or classified material.',
  },
};

export const MODES: Record<Mode, { title: string; summary: string; example: string }> = {
  A: {
    title: 'Answer questions about my documents',
    summary:
      'Expert answers and troubleshooting help, with references to the source, and an honest "I don\'t know" when the documents don\'t say.',
    example: '“What is the reset procedure after fault code E-4711?”',
  },
  B: {
    title: 'Write in our style',
    summary:
      'Learns your terminology, abbreviations, tone and formats, so it can draft reports and notes the way your team writes them.',
    example: '“Draft a shift-handover note in our standard format.”',
  },
  C: {
    title: 'Pull out structured data',
    summary:
      'Reads a document and fills in a fixed form (a JSON schema) every time, so the results can go straight into another system.',
    example: '“Extract all biomarkers with value, unit and date from this note.”',
  },
};

type GateCopy = { title: string; explanation: string; format: (v: number) => string };
const pct = (v: number) => `${(v * 100).toFixed(1)} %`;
const num = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(3));

export const GATES: Record<string, GateCopy> = {
  G1_domain: {
    title: 'Knows your documents',
    explanation:
      'The new model must answer questions about your documents clearly better than the original model, on questions it never saw while learning.',
    format: pct,
  },
  G2_forgetting: {
    title: 'Keeps its general skills',
    explanation:
      'Checks that general abilities (reasoning, maths, following instructions, writing JSON) did not get worse. The value is the largest drop in points on any test.',
    format: (v) => `${v.toFixed(1)} points`,
  },
  G3_hallucination: {
    title: 'Says “I don’t know” when it should',
    explanation:
      'Asks questions your documents do not answer. The model should admit it does not know instead of making something up.',
    format: pct,
  },
  G4_memorization: {
    title: 'Does not leak private data',
    explanation:
      'Hidden test markers and the personal details removed by the privacy filter are probed for. Nothing may come back out of the model.',
    format: (v) => `${num(v)} leaked`,
  },
  G5_schema: {
    title: 'Always fills in the form correctly',
    explanation:
      'Structured answers must match your schema in at least the required share of cases.',
    format: pct,
  },
};

export function gateCopy(gate: string): GateCopy {
  return (
    GATES[gate] ?? {
      title: humanize(gate),
      explanation: 'An additional quality check configured by your administrator.',
      format: num,
    }
  );
}

export const VARIANT_LABELS: Record<Variant, { title: string; hint: string }> = {
  base: { title: 'Original model', hint: 'The base model, untouched.' },
  base_rag: {
    title: 'Original + document search',
    hint: 'The base model, looking up passages from your documents before answering.',
  },
  adapter: { title: 'Your new model', hint: 'The model trained on your documents.' },
  adapter_rag: {
    title: 'New model + document search',
    hint: 'Your new model, also looking up passages before answering.',
  },
};

export const METRIC_LABELS: Record<string, string> = {
  accuracy: 'Correct answers',
  correctness: 'Correct answers',
  rouge_l: 'Wording overlap with reference',
  embedding_similarity: 'Meaning similarity with reference',
  exact_match: 'Exact field matches',
  abstention_rate: 'Admits when it does not know',
  fabrication_rate: 'Made-up answers',
  schema_valid: 'Valid structured output',
  general_skills: 'General skills score',
  citation_fidelity: 'Correct source references',
};

export function metricLabel(key: string): string {
  return METRIC_LABELS[key] ?? humanize(key);
}

export const ENTITY_LABELS: Record<string, string> = {
  PERSON: 'Names',
  PATIENT_NAME: 'Patient names',
  DATE_OF_BIRTH: 'Dates of birth',
  DATE: 'Dates',
  EMAIL: 'E-mail addresses',
  PHONE: 'Phone numbers',
  ADDRESS: 'Postal addresses',
  LOCATION: 'Places',
  IBAN: 'Bank account numbers (IBAN)',
  CREDIT_CARD: 'Card numbers',
  NATIONAL_ID: 'National ID numbers',
  SSN: 'Social security numbers',
  INSURANCE_ID: 'Insurance numbers',
  CASE_ID: 'Case / patient IDs',
  ORGANIZATION: 'Organisations',
  CLASSIFICATION: 'Classification markings',
  CUSTOM: 'Custom patterns',
};

export function entityLabel(t: string): string {
  return ENTITY_LABELS[t] ?? humanize(t);
}

export const ACTION_LABELS: Record<string, string> = {
  mask: 'Masked',
  drop: 'Section removed',
  keep: 'Kept',
};

export function humanize(key: string): string {
  const s = key
    .replace(/^G\d+_/, '')
    .replace(/[_-]+/g, ' ')
    .trim()
    .toLowerCase();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Friendly one-line description of a run event for the activity log. */
export function describeEvent(ev: {
  type: string;
  stage?: StageName;
  done?: number;
  total?: number;
  message?: string;
  [k: string]: unknown;
}): string {
  const stage = ev.stage ? (STAGE_LABELS[ev.stage]?.title ?? humanize(ev.stage)) : null;
  switch (ev.type) {
    case 'stage_started':
      return `${stage ?? 'Step'} started.`;
    case 'stage_completed':
      return `${stage ?? 'Step'} finished.`;
    case 'stage_failed':
      return `${stage ?? 'A step'} stopped with a problem${ev.error ? ` (${String(ev.error)})` : ''}.`;
    case 'awaiting_approval':
      return 'Waiting for you to review the privacy filter before anything is used for training.';
    case 'approved':
      return 'Privacy filter approved. Continuing.';
    case 'cancelled':
      return 'Run cancelled. You can resume it later.';
    case 'resumed':
      return 'Run resumed.';
    case 'trial_completed':
      return ev.message ?? 'Finished trying one set of training settings.';
    case 'run_completed':
      return 'All done — your model is ready for review.';
    case 'progress':
      if (ev.message) return ev.message;
      if (ev.done != null && ev.total != null)
        return `${stage ?? 'Progress'}: ${ev.done} of ${ev.total}`;
      return `${stage ?? 'Progress'} update.`;
    default:
      return ev.message ?? humanize(ev.type);
  }
}
