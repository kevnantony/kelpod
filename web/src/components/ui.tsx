// Small headless-ish UI primitives styled with Tailwind. No external UI kit.
import {
  forwardRef,
  useId,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from 'react';
import { isApiError } from '../api/client';
import { cx } from '../lib/cx';

type Variant = 'primary' | 'secondary' | 'danger' | 'ghost';
const VARIANTS: Record<Variant, string> = {
  primary:
    'bg-indigo-700 text-white hover:bg-indigo-800 disabled:bg-slate-300 disabled:text-slate-600',
  secondary:
    'bg-white text-slate-800 ring-1 ring-inset ring-slate-300 hover:bg-slate-50 disabled:text-slate-400',
  danger: 'bg-red-700 text-white hover:bg-red-800 disabled:bg-slate-300 disabled:text-slate-600',
  ghost: 'text-slate-700 hover:bg-slate-100 disabled:text-slate-400',
};

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  loading?: boolean;
  size?: 'sm' | 'md';
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'primary', loading, size = 'md', className, children, disabled, type, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type ?? 'button'}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-md font-medium transition-colors disabled:cursor-not-allowed',
        size === 'sm' ? 'px-2.5 py-1.5 text-sm' : 'px-4 py-2 text-sm',
        VARIANTS[variant],
        className,
      )}
      {...rest}
    >
      {loading && <Spinner className="h-4 w-4" label="" />}
      {children}
    </button>
  );
});

export function Spinner({ className, label = 'Loading…' }: { className?: string; label?: string }) {
  return (
    <span role={label ? 'status' : undefined} className="inline-flex items-center gap-2">
      <svg
        className={cx('animate-spin text-current', className ?? 'h-5 w-5')}
        viewBox="0 0 24 24"
        fill="none"
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeOpacity="0.25" strokeWidth="4" />
        <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="4" />
      </svg>
      {label && <span className="sr-only">{label}</span>}
    </span>
  );
}

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 py-10 text-slate-600 justify-center">
      <Spinner label="" />
      <span role="status">{label}</span>
    </div>
  );
}

export function Card({
  children,
  className,
  title,
  actions,
  as: As = 'section',
}: {
  children: ReactNode;
  className?: string;
  title?: ReactNode;
  actions?: ReactNode;
  as?: 'section' | 'div' | 'article';
}) {
  return (
    <As className={cx('rounded-lg bg-white shadow-sm ring-1 ring-slate-200', className)}>
      {(title || actions) && (
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 px-4 py-3">
          {title && <h2 className="text-base font-semibold text-slate-900">{title}</h2>}
          {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
        </header>
      )}
      <div className="p-4">{children}</div>
    </As>
  );
}

type Tone = 'neutral' | 'success' | 'warning' | 'danger' | 'info';
const TONES: Record<Tone, string> = {
  neutral: 'bg-slate-100 text-slate-700 ring-slate-300',
  success: 'bg-emerald-50 text-emerald-800 ring-emerald-300',
  warning: 'bg-amber-50 text-amber-900 ring-amber-300',
  danger: 'bg-red-50 text-red-800 ring-red-300',
  info: 'bg-indigo-50 text-indigo-800 ring-indigo-300',
};

export function Badge({
  tone = 'neutral',
  children,
  className,
}: {
  tone?: Tone;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset whitespace-nowrap',
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Alert({
  tone = 'danger',
  title,
  children,
  action,
}: {
  tone?: Tone;
  title?: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div
      role={tone === 'danger' ? 'alert' : 'status'}
      className={cx('rounded-md px-4 py-3 text-sm ring-1 ring-inset', TONES[tone])}
    >
      {title && <p className="font-semibold">{title}</p>}
      {children && <div className={title ? 'mt-1' : undefined}>{children}</div>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

/** Standard error display: what happened + what to do. */
export function ErrorBox({
  error,
  onRetry,
  title = 'Something went wrong',
}: {
  error: unknown;
  onRetry?: () => void;
  title?: string;
}) {
  const message = isApiError(error)
    ? error.message
    : error instanceof Error
      ? error.message
      : 'An unexpected problem occurred.';
  return (
    <Alert
      tone="danger"
      title={title}
      action={
        onRetry && (
          <Button variant="secondary" size="sm" onClick={onRetry}>
            Try again
          </Button>
        )
      }
    >
      {message}
    </Alert>
  );
}

export function EmptyState({
  title,
  children,
  action,
  icon,
}: {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center rounded-lg border-2 border-dashed border-slate-300 px-6 py-10 text-center">
      {icon && <div className="mb-3 text-slate-400">{icon}</div>}
      <h3 className="text-base font-semibold text-slate-900">{title}</h3>
      {children && <div className="mt-1 max-w-md text-sm text-slate-600">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function ProgressBar({
  value,
  max = 1,
  label,
  tone = 'info',
}: {
  value: number | null;
  max?: number;
  label: string;
  tone?: 'info' | 'success' | 'danger';
}) {
  const pct = value == null || max <= 0 ? null : Math.min(100, Math.max(0, (value / max) * 100));
  const color =
    tone === 'success' ? 'bg-emerald-600' : tone === 'danger' ? 'bg-red-600' : 'bg-indigo-600';
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct == null ? undefined : Math.round(pct)}
      className="h-2 w-full overflow-hidden rounded-full bg-slate-200"
    >
      <div
        className={cx('h-full rounded-full transition-all', color, pct == null && 'animate-pulse')}
        style={{ width: pct == null ? '30%' : `${pct}%` }}
      />
    </div>
  );
}

export function PageHeader({
  title,
  description,
  actions,
  breadcrumb,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  breadcrumb?: ReactNode;
}) {
  return (
    <div className="mb-6">
      {breadcrumb && <div className="mb-2 text-sm text-slate-600">{breadcrumb}</div>}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 wrap-anywhere">
            {title}
          </h1>
          {description && <p className="mt-1 max-w-3xl text-slate-600">{description}</p>}
        </div>
        {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
      </div>
    </div>
  );
}

const inputCls =
  'block w-full rounded-md border-0 bg-white px-3 py-2 text-sm text-slate-900 ring-1 ring-inset ring-slate-300 placeholder:text-slate-400 focus:ring-2 focus:ring-indigo-600 focus:outline-none disabled:bg-slate-100';

type FieldProps = { label: string; hint?: ReactNode; error?: string | null };

export function TextField({
  label,
  hint,
  error,
  id,
  ...rest
}: FieldProps & InputHTMLAttributes<HTMLInputElement>) {
  const auto = useId();
  const fid = id ?? auto;
  return (
    <div>
      <label htmlFor={fid} className="block text-sm font-medium text-slate-800">
        {label}
      </label>
      <input
        id={fid}
        aria-invalid={error ? true : undefined}
        aria-describedby={hint || error ? `${fid}-desc` : undefined}
        className={cx(inputCls, 'mt-1', error && 'ring-red-500')}
        {...rest}
      />
      {(hint || error) && (
        <p
          id={`${fid}-desc`}
          className={cx('mt-1 text-xs', error ? 'text-red-700' : 'text-slate-600')}
        >
          {error ?? hint}
        </p>
      )}
    </div>
  );
}

export function SelectField({
  label,
  hint,
  error,
  id,
  children,
  ...rest
}: FieldProps & SelectHTMLAttributes<HTMLSelectElement>) {
  const auto = useId();
  const fid = id ?? auto;
  return (
    <div>
      <label htmlFor={fid} className="block text-sm font-medium text-slate-800">
        {label}
      </label>
      <select
        id={fid}
        aria-describedby={hint || error ? `${fid}-desc` : undefined}
        className={cx(inputCls, 'mt-1')}
        {...rest}
      >
        {children}
      </select>
      {(hint || error) && (
        <p
          id={`${fid}-desc`}
          className={cx('mt-1 text-xs', error ? 'text-red-700' : 'text-slate-600')}
        >
          {error ?? hint}
        </p>
      )}
    </div>
  );
}

export function TextArea({
  label,
  hint,
  error,
  id,
  className,
  ...rest
}: FieldProps & TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const auto = useId();
  const fid = id ?? auto;
  return (
    <div>
      <label htmlFor={fid} className="block text-sm font-medium text-slate-800">
        {label}
      </label>
      <textarea
        id={fid}
        aria-invalid={error ? true : undefined}
        aria-describedby={hint || error ? `${fid}-desc` : undefined}
        className={cx(inputCls, 'mt-1', className)}
        {...rest}
      />
      {(hint || error) && (
        <p
          id={`${fid}-desc`}
          className={cx('mt-1 text-xs', error ? 'text-red-700' : 'text-slate-600')}
        >
          {error ?? hint}
        </p>
      )}
    </div>
  );
}

/** Responsive table wrapper: scrolls horizontally inside the card on narrow screens. */
export function TableWrap({ children, label }: { children: ReactNode; label: string }) {
  return (
    <div
      className="relative -mx-4 overflow-x-auto px-4"
      role="region"
      aria-label={label}
      tabIndex={0}
    >
      <table className="w-full min-w-[32rem] text-left text-sm">{children}</table>
    </div>
  );
}

export const th = 'border-b border-slate-200 px-2 py-2 font-semibold text-slate-700';
export const td = 'border-b border-slate-100 px-2 py-2 align-top';
