import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import type { RedactionSummary } from '../api/types';
import { highlightParts } from '../lib/highlight';
import { RedactionReview } from './RedactionReview';

const text = 'Patient [PATIENT_NAME_1] called from [PHONE_1].';
const summary: RedactionSummary = {
  counts_by_type: { PATIENT_NAME: 12, PHONE: 3, SOMETHING_NEW: 1 },
  chunks_dropped: 2,
  chunks_total: 50,
  samples: [
    {
      chunk_id: 'chunk_1',
      text_redacted: text,
      spans: [
        { start: 8, end: 24, entity_type: 'PATIENT_NAME', action: 'mask' },
        { start: 37, end: 46, entity_type: 'PHONE', action: 'mask' },
      ],
    },
    { chunk_id: 'chunk_2', text_redacted: 'Email [EMAIL_1] only.', spans: [] },
  ],
};

describe('highlightParts', () => {
  it('splits text at spans', () => {
    const parts = highlightParts(text, summary.samples[0]!.spans);
    expect(parts.map((p) => [p.text, !!p.span])).toEqual([
      ['Patient ', false],
      ['[PATIENT_NAME_1]', true],
      [' called from ', false],
      ['[PHONE_1]', true],
      ['.', false],
    ]);
  });

  it('falls back to placeholder detection when spans are missing', () => {
    const parts = highlightParts('Email [EMAIL_1] only.', []);
    expect(parts.find((p) => p.span)).toMatchObject({
      text: '[EMAIL_1]',
      span: { entity_type: 'EMAIL' },
    });
  });

  it('ignores out-of-range and overlapping spans', () => {
    const parts = highlightParts('abcdef', [
      { start: 0, end: 3, entity_type: 'A', action: 'mask' },
      { start: 2, end: 4, entity_type: 'B', action: 'mask' },
      { start: 5, end: 99, entity_type: 'C', action: 'mask' },
    ]);
    expect(parts.map((p) => p.text)).toEqual(['abc', 'def']);
  });
});

describe('RedactionReview', () => {
  it('shows totals and counts by type with plain-language names', () => {
    render(<RedactionReview summary={summary} onApprove={vi.fn()} />);
    expect(screen.getByTestId('total-masked')).toHaveTextContent('16');
    const list = screen.getByRole('list', { name: 'Masked details by type' });
    expect(within(list).getByText('Patient names')).toBeInTheDocument();
    expect(within(list).getByText('Phone numbers')).toBeInTheDocument();
    expect(within(list).getByText('Something new')).toBeInTheDocument();
    expect(screen.getByText('48')).toBeInTheDocument(); // sections kept
  });

  it('highlights placeholders in the samples', () => {
    const { container } = render(<RedactionReview summary={summary} onApprove={vi.fn()} />);
    const marks = Array.from(container.querySelectorAll('mark')).map(
      (m) => m.firstChild?.textContent,
    );
    expect(marks).toEqual(['[PATIENT_NAME_1]', '[PHONE_1]', '[EMAIL_1]']);
  });

  it('calls onApprove', async () => {
    const onApprove = vi.fn();
    render(<RedactionReview summary={summary} onApprove={onApprove} />);
    await userEvent.click(screen.getByRole('button', { name: 'Approve and continue' }));
    expect(onApprove).toHaveBeenCalledTimes(1);
  });

  it('disables approval when the run is no longer waiting', () => {
    render(<RedactionReview summary={summary} onApprove={vi.fn()} canApprove={false} />);
    expect(screen.getByRole('button', { name: 'Approve and continue' })).toBeDisabled();
    expect(screen.getByText(/not waiting for approval/)).toBeInTheDocument();
  });

  it('shows an approval error', () => {
    render(
      <RedactionReview
        summary={summary}
        onApprove={vi.fn()}
        approveError={new Error('Disk full.')}
      />,
    );
    expect(screen.getByRole('alert')).toHaveTextContent('Disk full.');
  });
});
