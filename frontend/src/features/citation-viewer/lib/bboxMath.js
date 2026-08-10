/**
 * `bbox` geometry — Documentation/system-design/
 * 05-streaming-and-api-contract.md §5.5 (empirically verified format: PDF
 * points, `CoordOrigin.BOTTOMLEFT`) and
 * Documentation/ui-ux-design/05-citation-viewer.md §4/§5 (overlay
 * conversion + auto-zoom bands).
 */

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

/**
 * Converts a bottom-left-origin PDF-points `bbox` into a top-left-origin
 * CSS rect, still in points (not yet scaled to displayed pixels) — exact
 * formula from §5.5:
 * `css_left=bbox.l, css_top=page_height_pt-bbox.t, css_width=r-l, css_height=t-b`.
 *
 * @param {{l:number,t:number,r:number,b:number}} bbox
 * @param {number} pageHeightPt
 */
export function cssRectFromBboxPt(bbox, pageHeightPt) {
  return {
    leftPt: bbox.l,
    topPt: pageHeightPt - bbox.t,
    widthPt: Math.max(0, bbox.r - bbox.l),
    heightPt: Math.max(0, bbox.t - bbox.b),
  };
}

/**
 * Auto-zoom level (percent), where **100% = fit-to-panel-width** (not
 * native pixel resolution) — `05-citation-viewer.md` §5 explicitly ties
 * the full-page case to "selalu fit 100% lebar panel", so 100% is defined
 * as that baseline; the manual zoom range (50-400%, §6) scales from there.
 * Deliberately independent of the actual container width in pixels (that
 * only matters once converting percent -> displayed px, see
 * `displayScalePxPerPt`) — the ratio bands below only need bbox-vs-page
 * proportions.
 *
 * @param {{ bbox: {l:number,t:number,r:number,b:number}|null, pageWidthPt: number, pageHeightPt: number }} params
 * @returns {number} zoom percent, clamped to [50, 400]
 */
export function computeAutoZoomPercent({ bbox, pageWidthPt, pageHeightPt }) {
  if (!bbox || !pageWidthPt || !pageHeightPt) return 100;
  const widthPt = Math.max(0, bbox.r - bbox.l);
  const heightPt = Math.max(0, bbox.t - bbox.b);
  if (widthPt <= 0 || heightPt <= 0) return 100;

  const areaPt = widthPt * heightPt;
  const pageAreaPt = pageWidthPt * pageHeightPt;
  const ratio = pageAreaPt > 0 ? areaPt / pageAreaPt : 1;

  let percent;
  if (ratio > 0.4) {
    // Full-page figure/diagram: always fit-to-width, zooming in further
    // would cut off context (§5 last bullet).
    percent = 100;
  } else if (ratio >= 0.05) {
    // "Sedang": fit bbox width + ~15% padding on each side.
    percent = (pageWidthPt / (widthPt * 1.3)) * 100;
  } else {
    // "Zoom in": bbox should occupy ~40% of the viewport width.
    percent = ((0.4 * pageWidthPt) / widthPt) * 100;
  }
  return clamp(percent, 50, 400);
}

/**
 * Points -> displayed pixels scale factor for a given zoom percent
 * (100% = fit-to-width baseline, see `computeAutoZoomPercent`).
 * @param {number} zoomPercent
 * @param {number} containerWidthPx
 * @param {number} pageWidthPt
 */
export function displayScalePxPerPt(zoomPercent, containerWidthPx, pageWidthPt) {
  if (!containerWidthPx || !pageWidthPt) return 0;
  const baseScale = containerWidthPx / pageWidthPt;
  return baseScale * (zoomPercent / 100);
}

/**
 * `bbox` (PDF points) -> displayed pixel rect, ready for absolute
 * positioning over the rendered page `<img>`.
 * @param {{l:number,t:number,r:number,b:number}} bbox
 * @param {number} pageHeightPt
 * @param {number} scalePxPerPt
 */
export function overlayRectPx(bbox, pageHeightPt, scalePxPerPt) {
  const rect = cssRectFromBboxPt(bbox, pageHeightPt);
  return {
    left: rect.leftPt * scalePxPerPt,
    top: rect.topPt * scalePxPerPt,
    width: rect.widthPt * scalePxPerPt,
    height: rect.heightPt * scalePxPerPt,
  };
}

export { clamp };
