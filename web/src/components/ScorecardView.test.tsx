import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import type { Scorecard } from '../api/types';
import { ScorecardView, type ScorecardViewProps } from './ScorecardView';

function card(overrides: Partial<Scorecard> = {}): Scorecard {
  return {
    run_id: 'run_1',
    all_passed: true,
    gates: [
      { gate: 'G1_domain', passed: true, value: 0.31, threshold: 0.12, details: {} },
      { gate: 'G2_forgetting', passed: true, value: 1.4, threshold: 3, details: {} },
      { gate: 'G3_hallucination', passed: true, value: 0.93, threshold: 0.9, details: {} },
      { gate: 'G4_memorization', passed: true, value: 0, threshold: 0, details: {} },
    ],
    metrics: {
      base: { correctness: 0.41 },
      base_rag: { correctness: 0.58 },
      adapter: { correctness: 0.74 },
      adapter_rag: { correctness: 0.79 },
    },
    examples: [
      {
        prompt: 'How to reset?',
        base: 'Turn it off.',
        adapter: 'Hold RESET 5 s.',
        reference: 'Hold RESET.',
      },
      { prompt: 'Warranty?', base: '2 years.', adapter: 'Not in the documents.', reference: null },
    ],
    ...overrides,
  };
}

function renderView(props: Partial<ScorecardViewProps> = {}) {
  const onDeploy = vi.fn();
  const onDownload = vi.fn();
  render(
    <ScorecardView
      card={card()}
      adapterId="adp_1"
      deployed={false}
      onDeploy={onDeploy}
      onDownload={onDownload}
      {...props}
    />,
  );
  return { onDeploy, onDownload };
}

describe('ScorecardView', () => {
  it('shows plain-language gate names with pass badges, no jargon', () => {
    renderView();
    expect(screen.getByText('All quality checks passed')).toBeInTheDocument();
    expect(screen.getByText('Knows your documents')).toBeInTheDocument();
    expect(screen.getByText('Keeps its general skills')).toBeInTheDocument();
    expect(screen.getByText('Does not leak private data')).toBeInTheDocument();
    expect(screen.getAllByText('Passed')).toHaveLength(4);
    expect(screen.queryByText(/G1_domain|gate|LoRA|adapter/i)).not.toBeInTheDocument();
    const g1 = screen.getByTestId('gate-G1_domain');
    expect(within(g1).getByText('31.0 %')).toBeInTheDocument();
    expect(within(g1).getByText(/needed at least/)).toBeInTheDocument();
    expect(
      within(screen.getByTestId('gate-G2_forgetting')).getByText(/needed at most/),
    ).toBeInTheDocument();
  });

  it('renders the metrics table with all four variants', () => {
    renderView();
    const table = screen.getByRole('table');
    for (const h of [
      'Original model',
      'Original + document search',
      'Your new model',
      'New model + document search',
    ])
      expect(within(table).getByRole('columnheader', { name: h })).toBeInTheDocument();
    const row = within(table).getByRole('row', { name: /Correct answers/ });
    expect(within(row).getByText('74.0 %')).toBeInTheDocument();
  });

  it('shows example comparisons including unanswerable ones', () => {
    renderView();
    expect(screen.getByText('Hold RESET 5 s.')).toBeInTheDocument();
    expect(screen.getByText(/do not answer this/)).toBeInTheDocument();
  });

  it('enables Deploy when all checks pass and calls onDeploy', async () => {
    const { onDeploy } = renderView();
    const btn = screen.getByRole('button', { name: 'Deploy' });
    expect(btn).toBeEnabled();
    await userEvent.click(btn);
    expect(onDeploy).toHaveBeenCalledTimes(1);
  });

  it('disables Deploy when any check fails and explains why', () => {
    const c = card({ all_passed: false });
    c.gates[2] = {
      gate: 'G3_hallucination',
      passed: false,
      value: 0.7,
      threshold: 0.9,
      details: {},
    };
    renderView({ card: c });
    expect(screen.getByRole('button', { name: 'Deploy' })).toBeDisabled();
    expect(screen.getByText('1 quality check not passed')).toBeInTheDocument();
    expect(screen.getByText('Not passed')).toBeInTheDocument();
    expect(screen.getByText(/only when every quality check has passed/)).toBeInTheDocument();
    expect(screen.getByText(/What you can do/)).toBeInTheDocument();
  });

  it('disables Deploy while no adapter is registered yet, and after deploying', () => {
    const { unmount } = render(
      <ScorecardView
        card={card()}
        adapterId={null}
        deployed={false}
        onDeploy={vi.fn()}
        onDownload={vi.fn()}
      />,
    );
    expect(screen.getByRole('button', { name: 'Deploy' })).toBeDisabled();
    unmount();
    renderView({ deployed: true });
    expect(screen.getByRole('button', { name: 'Deployed' })).toBeDisabled();
  });

  it('offers PDF and JSON report downloads', async () => {
    const { onDownload } = renderView();
    await userEvent.click(screen.getByRole('link', { name: 'Download report (PDF)' }));
    await userEvent.click(screen.getByRole('link', { name: 'Download report (JSON)' }));
    expect(onDownload.mock.calls).toEqual([['pdf'], ['json']]);
  });
});
