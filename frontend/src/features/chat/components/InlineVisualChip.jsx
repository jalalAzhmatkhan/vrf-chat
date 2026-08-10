import { useEffect, useRef, useState } from 'react';
import { getElementCached } from '../../../lib/api/elementsApi';
import styles from './InlineVisualChip.module.css';

const TIER1_TYPES = new Set(['icon']);

function resolveTier(elementType) {
  if (!elementType) return null;
  return TIER1_TYPES.has(elementType) ? 1 : 2;
}

/**
 * `InlineIcon` / `InlineFigureChip` — Documentation/ui-ux-design/
 * 04-chat-ui.md §6.2/§6.5.
 *
 * **Citation-timing note** (documented deviation, see STATUS REPORT): the
 * actual backend (`app/agent/streaming.py`) emits all `citation` SSE events
 * right before `done`, not interleaved with `token` deltas — so a marker
 * flushed early in the stream frequently has no matching entry in
 * `citations[]` yet. Rather than wait (which would either delay the chip
 * indefinitely or force showing a wrongly-sized placeholder for a figure),
 * this component falls back to `GET /api/v1/elements/{element_id}`
 * (cached, deduped — `lib/api/elementsApi.js`) the moment it mounts without
 * a `citation` prop, so it can size itself correctly as early as possible.
 * Once the real `citation` arrives (progressively or at `done`), the prop
 * takes precedence and no further fetch happens.
 *
 * @param {{
 *   elementId: string,
 *   citation?: { element_type: string, image_uri: string|null, visual_description: Record<string, unknown>|null } | null,
 *   onOpenCitation: (elementId: string) => void,
 * }} props
 */
export function InlineVisualChip({ elementId, citation, onOpenCitation }) {
  const [fetched, setFetched] = useState(null);
  const [fetchFailed, setFetchFailed] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);
  const tooltipTimerRef = useRef(null);

  useEffect(() => {
    if (citation) return undefined;
    let cancelled = false;
    getElementCached(elementId)
      .then((element) => {
        if (!cancelled) setFetched(element);
      })
      .catch(() => {
        if (!cancelled) setFetchFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [elementId, citation]);

  useEffect(
    () => () => {
      if (tooltipTimerRef.current) window.clearTimeout(tooltipTimerRef.current);
    },
    [],
  );

  const elementType = citation?.element_type ?? fetched?.element_type ?? null;
  const imageUri = citation?.image_uri ?? fetched?.image_uri ?? null;
  const description =
    citation?.visual_description?.description ?? fetched?.visual_description?.description ?? null;
  // Tier defaults to 1 (smaller reserved footprint) while unresolved.
  const tier = resolveTier(elementType) ?? 1;
  const resolved = Boolean(elementType);

  const altText = description || (tier === 1 ? 'Lihat ikon di manual sumber' : 'Lihat gambar di manual sumber');

  const handleMouseEnter = () => {
    tooltipTimerRef.current = window.setTimeout(() => setShowTooltip(true), 400);
  };
  const handleMouseLeave = () => {
    if (tooltipTimerRef.current) window.clearTimeout(tooltipTimerRef.current);
    setShowTooltip(false);
  };
  const handleActivate = () => onOpenCitation(elementId);

  return (
    <span
      role="button"
      tabIndex={0}
      className={`${styles.chip} ${tier === 1 ? styles.tier1 : styles.tier2}`}
      aria-label={altText}
      onClick={handleActivate}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          handleActivate();
        }
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onFocus={handleMouseEnter}
      onBlur={handleMouseLeave}
    >
      {imageUri && (
        <img
          src={imageUri}
          alt=""
          className={`${styles.img} ${imageLoaded ? styles.imgVisible : styles.imgHidden}`}
          onLoad={() => setImageLoaded(true)}
        />
      )}
      {(!imageUri || !imageLoaded) && (
        <span className={styles.placeholder} aria-hidden="true">
          {fetchFailed ? (
            '⚠'
          ) : (
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
              <rect x="1.5" y="2.5" width="13" height="11" rx="1.5" stroke="currentColor" strokeWidth="1.2" />
              <circle cx="5.5" cy="6.5" r="1.25" stroke="currentColor" strokeWidth="1.1" />
              <path d="M2 12l3.7-3.7a1 1 0 0 1 1.4 0L9 10.2l1.8-1.8a1 1 0 0 1 1.4 0L14.5 11" stroke="currentColor" strokeWidth="1.1" />
            </svg>
          )}
        </span>
      )}
      {showTooltip && resolved && (
        <span className={styles.tooltip} role="tooltip">
          {altText}
        </span>
      )}
    </span>
  );
}
