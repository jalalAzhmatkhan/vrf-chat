import { describe, expect, it } from 'vitest';
import { extractMarkerElementIds, parseAnswerBlocks, parseInline } from './parseAnswer';

describe('parseInline', () => {
  it('splits plain text, bold, marker, and mono-identifier tokens', () => {
    const nodes = parseInline('Kode error **P8** menandakan {{el:12}} tekanan tinggi pada CN105.');
    expect(nodes).toEqual([
      { type: 'text', value: 'Kode error ' },
      { type: 'bold', children: [{ type: 'mono', value: 'P8' }] },
      { type: 'text', value: ' menandakan ' },
      { type: 'marker', elementId: '12' },
      { type: 'text', value: ' tekanan tinggi pada ' },
      { type: 'mono', value: 'CN105' },
      { type: 'text', value: '.' },
    ]);
  });

  it('detects value+unit identifiers as mono', () => {
    const nodes = parseInline('Tegangan normal adalah 220V pada terminal.');
    expect(nodes).toContainEqual({ type: 'mono', value: '220V' });
  });
});

describe('parseAnswerBlocks', () => {
  it('splits paragraphs on blank lines', () => {
    const blocks = parseAnswerBlocks('Paragraf pertama.\n\nParagraf kedua.');
    expect(blocks).toHaveLength(2);
    expect(blocks[0].type).toBe('paragraph');
    expect(blocks[1].type).toBe('paragraph');
  });

  it('renders "1. " lines as an ordered list', () => {
    const blocks = parseAnswerBlocks('1. Langkah pertama\n2. Langkah kedua');
    expect(blocks).toEqual([
      {
        type: 'ol',
        items: [
          [{ type: 'text', value: 'Langkah pertama' }],
          [{ type: 'text', value: 'Langkah kedua' }],
        ],
      },
    ]);
  });

  it('renders "- "/"* " lines as an unordered list', () => {
    const blocks = parseAnswerBlocks('- Item satu\n- Item dua');
    expect(blocks[0].type).toBe('ul');
    expect(blocks[0].items).toHaveLength(2);
  });

  it('keeps prose and a list as separate blocks within one paragraph chunk', () => {
    const blocks = parseAnswerBlocks('Perhatikan langkah berikut:\n1. Matikan daya\n2. Buka panel');
    expect(blocks.map((b) => b.type)).toEqual(['paragraph', 'ol']);
  });
});

describe('extractMarkerElementIds', () => {
  it('collects every unique element id referenced by a marker', () => {
    const ids = extractMarkerElementIds('lihat {{el:1}} dan {{el:2}} lalu {{el:1}} lagi');
    expect(ids).toEqual(new Set(['1', '2']));
  });

  it('returns an empty set when there are no markers', () => {
    expect(extractMarkerElementIds('teks biasa saja')).toEqual(new Set());
  });
});
