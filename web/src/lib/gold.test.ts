import { describe, expect, it } from 'vitest';
import { parseCsv, parseGold } from './gold';
import { secondsUntil } from './time';

describe('parseCsv', () => {
  it('handles quotes, commas, escaped quotes, CRLF and BOM', () => {
    const rows = parseCsv('﻿a,b\r\n"x, y","say ""hi"""\r\n\r\n3,4');
    expect(rows).toEqual([
      ['a', 'b'],
      ['x, y', 'say "hi"'],
      ['3', '4'],
    ]);
  });
});

describe('parseGold', () => {
  it('parses CSV with answerable defaults', () => {
    const { items, errors } = parseGold(
      'question,reference_answer,answerable\nWhat is X?,It is Y,\nWarranty of Z?,,no\n,orphan,\n',
      'gold.csv',
    );
    expect(items).toEqual([
      { question: 'What is X?', reference_answer: 'It is Y', answerable: true },
      { question: 'Warranty of Z?', answerable: false },
    ]);
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatch(/Row 4/);
  });

  it('requires a question column', () => {
    expect(parseGold('q,a\n1,2', 'g.csv').errors[0]).toMatch(/question/);
  });

  it('parses JSONL including reference_json', () => {
    const { items, errors } = parseGold(
      '{"question":"Extract","reference_json":{"a":1}}\nnot json\n{"question":"Q2","answer":"A2","answerable":true}\n',
      'gold.jsonl',
    );
    expect(items).toEqual([
      { question: 'Extract', reference_json: { a: 1 }, answerable: true },
      { question: 'Q2', reference_answer: 'A2', answerable: true },
    ]);
    expect(errors).toEqual(['Line 2: not valid JSON.']);
  });
});

describe('secondsUntil', () => {
  it('computes the next occurrence of a time', () => {
    const now = new Date(2026, 0, 1, 22, 0, 0);
    expect(secondsUntil('07:00', now)).toBe(9 * 3600);
    expect(secondsUntil('23:30', now)).toBe(90 * 60);
    expect(secondsUntil('nope', now)).toBeNull();
  });
});
