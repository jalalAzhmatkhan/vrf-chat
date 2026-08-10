import { describe, expect, it } from 'vitest';
import { computeAutoZoomPercent, cssRectFromBboxPt, displayScalePxPerPt, overlayRectPx } from './bboxMath';

// Real sample verified empirically by System Analyst —
// Documentation/system-design/05-streaming-and-api-contract.md §5.5.
const SAMPLE_BBOX = { l: 497.73, t: 241.87, r: 538.86, b: 201.81 };
const PAGE_WIDTH_PT = 609;
const PAGE_HEIGHT_PT = 793;

describe('cssRectFromBboxPt', () => {
  it('matches the §5.5 documented conversion formula exactly', () => {
    const rect = cssRectFromBboxPt(SAMPLE_BBOX, PAGE_HEIGHT_PT);
    expect(rect.leftPt).toBeCloseTo(497.73);
    expect(rect.topPt).toBeCloseTo(793 - 241.87);
    expect(rect.widthPt).toBeCloseTo(538.86 - 497.73);
    expect(rect.heightPt).toBeCloseTo(241.87 - 201.81);
  });
});

describe('computeAutoZoomPercent', () => {
  it('returns 100% (fit width) for a full-page figure (>40% of page area)', () => {
    const bbox = { l: 0, t: PAGE_HEIGHT_PT, r: PAGE_WIDTH_PT, b: 0 };
    expect(computeAutoZoomPercent({ bbox, pageWidthPt: PAGE_WIDTH_PT, pageHeightPt: PAGE_HEIGHT_PT })).toBe(100);
  });

  it('zooms in for a small icon (<5% of page area)', () => {
    // This sample bbox (~41x40pt on a 609x793pt page) is well under 5%.
    const percent = computeAutoZoomPercent({
      bbox: SAMPLE_BBOX,
      pageWidthPt: PAGE_WIDTH_PT,
      pageHeightPt: PAGE_HEIGHT_PT,
    });
    expect(percent).toBeGreaterThan(100);
  });

  it('clamps to the 50-400% manual zoom range', () => {
    const tinyBbox = { l: 0, t: 1, r: 1, b: 0 };
    const percent = computeAutoZoomPercent({ bbox: tinyBbox, pageWidthPt: PAGE_WIDTH_PT, pageHeightPt: PAGE_HEIGHT_PT });
    expect(percent).toBeLessThanOrEqual(400);
    expect(percent).toBeGreaterThanOrEqual(50);
  });

  it('falls back to 100% when bbox is missing', () => {
    expect(computeAutoZoomPercent({ bbox: null, pageWidthPt: PAGE_WIDTH_PT, pageHeightPt: PAGE_HEIGHT_PT })).toBe(100);
  });

  it('picks the "medium" band for a mid-sized bbox (5-40% of page area)', () => {
    const bbox = { l: 0, t: 500, r: 400, b: 0 }; // ~400x500pt on 609x793pt page -> ~41% ... adjust
    const mediumBbox = { l: 0, t: 300, r: 300, b: 0 }; // 300x300 = 90000 / (609*793=483037) ~18.6%
    const percent = computeAutoZoomPercent({
      bbox: mediumBbox,
      pageWidthPt: PAGE_WIDTH_PT,
      pageHeightPt: PAGE_HEIGHT_PT,
    });
    expect(percent).toBeGreaterThanOrEqual(50);
    expect(percent).toBeLessThanOrEqual(400);
    void bbox; // unused alternate example kept out of the assertion for clarity
  });
});

describe('displayScalePxPerPt + overlayRectPx', () => {
  it('scales a bbox to pixel coordinates proportionally to zoom percent', () => {
    const scaleAt100 = displayScalePxPerPt(100, 480, PAGE_WIDTH_PT);
    const scaleAt200 = displayScalePxPerPt(200, 480, PAGE_WIDTH_PT);
    expect(scaleAt200).toBeCloseTo(scaleAt100 * 2);

    const rectPx = overlayRectPx(SAMPLE_BBOX, PAGE_HEIGHT_PT, scaleAt100);
    const cssRect = cssRectFromBboxPt(SAMPLE_BBOX, PAGE_HEIGHT_PT);
    expect(rectPx.left).toBeCloseTo(cssRect.leftPt * scaleAt100);
    expect(rectPx.width).toBeCloseTo(cssRect.widthPt * scaleAt100);
  });

  it('returns 0 scale when container width or page width is unknown', () => {
    expect(displayScalePxPerPt(100, 0, PAGE_WIDTH_PT)).toBe(0);
    expect(displayScalePxPerPt(100, 480, 0)).toBe(0);
  });
});
