import { describe, expect, it } from 'vitest';
import { getSafeRenderLength } from './markerBuffer';

describe('getSafeRenderLength — §6.3 marker buffering', () => {
  it('returns full length when there is no marker attempt at all', () => {
    expect(getSafeRenderLength('tekan tombol biasa saja.')).toBe(
      'tekan tombol biasa saja.'.length,
    );
  });

  it('returns full length once a marker is fully complete', () => {
    const text = 'tekan tombol {{el:239}} untuk lanjut';
    expect(getSafeRenderLength(text)).toBe(text.length);
  });

  it('holds back an in-progress marker split mid-literal (e.g. "{{el")', () => {
    const text = 'tekan tombol {{el';
    expect(getSafeRenderLength(text)).toBe('tekan tombol '.length);
  });

  it('holds back an in-progress marker split right after the literal, before any digit', () => {
    const text = 'tekan tombol {{el:';
    expect(getSafeRenderLength(text)).toBe('tekan tombol '.length);
  });

  it('holds back at the very first brace, not the second', () => {
    const text = 'abc {{el:123';
    expect(getSafeRenderLength(text)).toBe('abc '.length);
  });

  it('holds back a marker missing only the final closing brace', () => {
    const text = 'lihat {{el:239}';
    expect(getSafeRenderLength(text)).toBe('lihat '.length);
  });

  it('holds back a single opening brace (still could become "{{el:")', () => {
    expect(getSafeRenderLength('teks biasa {')).toBe('teks biasa '.length);
  });

  it('flushes a "{" that turns out not to be a marker attempt', () => {
    const text = 'nilai { tidak lengkap } di sini';
    // "{" followed by " " immediately breaks the "{{el:" prefix match, so
    // the whole string is safe (nothing pending).
    expect(getSafeRenderLength(text)).toBe(text.length);
  });

  it('handles multiple markers, holding back only the final incomplete one', () => {
    const text = 'A {{el:1}} B {{el:2}} C {{el:3';
    expect(getSafeRenderLength(text)).toBe('A {{el:1}} B {{el:2}} C '.length);
  });
});
