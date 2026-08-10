import { overlayRectPx } from '../lib/bboxMath';
import styles from './BboxOverlay.module.css';

/**
 * Documentation/ui-ux-design/05-citation-viewer.md §4.
 * `variant='primary'` = the cited element (strong highlight, §4.1);
 * `variant='context'` = every other element on the page (faint outline,
 * §4.2), `variant='context-icon'` = context element of type `icon`
 * (opacity 40% of the context outline, per §4.2 last sentence).
 *
 * @param {{
 *   bbox: {l:number,t:number,r:number,b:number},
 *   pageHeightPt: number,
 *   scalePxPerPt: number,
 *   variant: 'primary'|'context'|'context-icon',
 *   flash?: boolean,
 * }} props
 */
export function BboxOverlay({ bbox, pageHeightPt, scalePxPerPt, variant, flash = false }) {
  if (!bbox || !scalePxPerPt) return null;
  const rect = overlayRectPx(bbox, pageHeightPt, scalePxPerPt);
  const variantClass =
    variant === 'primary' ? styles.primary : variant === 'context-icon' ? styles.contextIcon : styles.context;

  return (
    <div
      className={`${styles.rect} ${variantClass} ${flash ? styles.flash : ''}`}
      style={{ left: rect.left, top: rect.top, width: rect.width, height: rect.height }}
      aria-hidden="true"
    />
  );
}
