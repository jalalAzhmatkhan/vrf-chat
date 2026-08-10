import { useEffect, useMemo, useRef, useState } from 'react';
import { getDocumentPage } from '../../lib/api/documentsApi';
import { getElementCached } from '../../lib/api/elementsApi';
import { getCitationTypeMeta } from '../chat/lib/citationTypeMeta';
import { AIPulseLoader } from '../../components/AIPulseLoader/AIPulseLoader';
import { Button } from '../../components/Button/Button';
import { BboxOverlay } from './components/BboxOverlay';
import { ZoomControls } from './components/ZoomControls';
import { CitationPager } from './components/CitationPager';
import { clamp, computeAutoZoomPercent, displayScalePxPerPt } from './lib/bboxMath';
import styles from './CitationViewerPanel.module.css';

const ZOOM_STEP = 25;
const ZOOM_MIN = 50;
const ZOOM_MAX = 400;
const DEFAULT_CONTAINER_WIDTH = 480;

/**
 * Documentation/ui-ux-design/05-citation-viewer.md — overlay drawer
 * (desktop/tablet) / full-screen bottom sheet (mobile), §2. Mounted once by
 * `CitationViewerContext`'s `CitationViewerProvider` — always in the DOM
 * (visibility/position toggled via `isOpen`) so enter/exit transitions can
 * both play via CSS rather than mount/unmount choreography.
 *
 * @param {{
 *   isOpen: boolean,
 *   citations: Array<{document_id:string|null, page:number|null, element_id:string, element_type:string|null, quote:string|null, visual_description:object|null}>,
 *   activeIndex: number,
 *   documentsMap: Map<string, {title:string}>,
 *   onClose: () => void,
 *   onNavigate: (index: number) => void,
 * }} props
 */
export function CitationViewerPanel({ isOpen, citations, activeIndex, documentsMap, onClose, onNavigate }) {
  const citation = citations[activeIndex] ?? null;

  const [resolvedDocPage, setResolvedDocPage] = useState({});
  const [pageData, setPageData] = useState(null);
  const [loadStatus, setLoadStatus] = useState('idle');
  const [zoomPercent, setZoomPercent] = useState(100);
  const [autoZoomRequested, setAutoZoomRequested] = useState(true);
  const [containerWidth, setContainerWidth] = useState(DEFAULT_CONTAINER_WIDTH);
  const [retryNonce, setRetryNonce] = useState(0);

  const containerRef = useRef(null);
  const panelRef = useRef(null);
  const closeButtonRef = useRef(null);
  const lastPageKeyRef = useRef(null);

  const documentId =
    citation?.document_id ?? (citation ? (resolvedDocPage[citation.element_id]?.document_id ?? null) : null);
  const page = citation?.page ?? (citation ? (resolvedDocPage[citation.element_id]?.page ?? null) : null);

  // §3/DIRECT MESSAGE resolution: a marker/citation-row click can precede
  // the `citations[]` payload having this element yet (see
  // CitationViewerContext docstring) — resolve document_id/page via
  // GET /elements/{id} in that case.
  useEffect(() => {
    if (!isOpen || !citation) return undefined;
    if (citation.document_id != null && citation.page != null) return undefined;
    if (resolvedDocPage[citation.element_id]) return undefined;
    let cancelled = false;
    getElementCached(citation.element_id)
      .then((el) => {
        if (cancelled) return;
        setResolvedDocPage((prev) => ({
          ...prev,
          [citation.element_id]: { document_id: el.document_id, page: el.page_number },
        }));
      })
      .catch(() => {
        if (!cancelled) setLoadStatus('error');
      });
    return () => {
      cancelled = true;
    };
  }, [isOpen, citation, resolvedDocPage, retryNonce]);

  // §7: skip re-fetching the page image if it's the same as the previous
  // citation's page.
  useEffect(() => {
    if (!isOpen || documentId == null || page == null) return undefined;
    const pageKey = `${documentId}:${page}`;
    if (pageKey === lastPageKeyRef.current) return undefined;
    let cancelled = false;
    setLoadStatus('loading');
    getDocumentPage(documentId, page)
      .then((data) => {
        if (cancelled) return;
        lastPageKeyRef.current = pageKey;
        setPageData(data);
        setLoadStatus('success');
      })
      .catch(() => {
        if (!cancelled) setLoadStatus('error');
      });
    return () => {
      cancelled = true;
    };
  }, [isOpen, documentId, page, retryNonce]);

  // Re-request auto-zoom (§5) whenever the panel opens or the active
  // citation changes — independent of whether the underlying page fetch
  // above actually re-runs (same-page-different-element case).
  useEffect(() => {
    if (!isOpen) return;
    setAutoZoomRequested(true);
  }, [isOpen, citation?.element_id]);

  const activeElement = useMemo(() => {
    if (!pageData || !citation) return null;
    return pageData.elements.find((el) => String(el.element_id) === String(citation.element_id)) ?? null;
  }, [pageData, citation]);

  useEffect(() => {
    if (!autoZoomRequested || !pageData) return;
    const percent = computeAutoZoomPercent({
      bbox: activeElement?.bbox ?? null,
      pageWidthPt: pageData.page_width_pt,
      pageHeightPt: pageData.page_height_pt,
    });
    setZoomPercent(percent);
    setAutoZoomRequested(false);
  }, [autoZoomRequested, pageData, activeElement]);

  // Measure the scroll container so 100% zoom = fit-to-width (§5/§6).
  useEffect(() => {
    const node = containerRef.current;
    if (!node || typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect?.width;
      if (width) setContainerWidth(width);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [isOpen]);

  const scalePxPerPt = pageData ? displayScalePxPerPt(zoomPercent, containerWidth, pageData.page_width_pt) : 0;

  // Center the viewport on the active bbox — only on citation/page change,
  // not on every manual zoom tick (auto-zoom already handles the "state
  // awal" per §5; after that, user pan/zoom is left alone).
  useEffect(() => {
    const node = containerRef.current;
    if (!node || !pageData || !activeElement?.bbox || !scalePxPerPt) return;
    const widthPt = activeElement.bbox.r - activeElement.bbox.l;
    const heightPt = activeElement.bbox.t - activeElement.bbox.b;
    const leftPt = activeElement.bbox.l;
    const topPt = pageData.page_height_pt - activeElement.bbox.t;
    const centerLeftPx = (leftPt + widthPt / 2) * scalePxPerPt;
    const centerTopPx = (topPt + heightPt / 2) * scalePxPerPt;
    node.scrollLeft = Math.max(0, centerLeftPx - node.clientWidth / 2);
    node.scrollTop = Math.max(0, centerTopPx - node.clientHeight / 2);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageData, activeElement?.element_id]);

  // §10: focus trap, Escape-to-close, initial focus on the close button.
  useEffect(() => {
    if (!isOpen) return undefined;
    closeButtonRef.current?.focus();
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        onClose();
        return;
      }
      if (event.key !== 'Tab') return;
      const focusables = panelRef.current?.querySelectorAll(
        'button:not(:disabled), [href], input, [tabindex]:not([tabindex="-1"])',
      );
      if (!focusables || focusables.length === 0) return;
      const list = Array.from(focusables);
      const first = list[0];
      const last = list[list.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const handleZoomIn = () => setZoomPercent((z) => clamp(z + ZOOM_STEP, ZOOM_MIN, ZOOM_MAX));
  const handleZoomOut = () => setZoomPercent((z) => clamp(z - ZOOM_STEP, ZOOM_MIN, ZOOM_MAX));
  const handleFit = () => setAutoZoomRequested(true);
  const handleWheel = (event) => {
    if (!event.ctrlKey) return;
    event.preventDefault();
    setZoomPercent((z) => clamp(z - event.deltaY * 0.5, ZOOM_MIN, ZOOM_MAX));
  };
  const handleRetry = () => {
    lastPageKeyRef.current = null;
    setLoadStatus('idle');
    setRetryNonce((n) => n + 1);
  };

  const documentTitle = documentId != null ? (documentsMap?.get(String(documentId))?.title ?? `Dokumen ${documentId}`) : '…';
  const typeMeta = citation?.element_type ? getCitationTypeMeta(citation.element_type) : null;
  const description = citation?.visual_description?.description ?? activeElement?.text ?? null;

  const wrapperWidthPx = pageData ? pageData.page_width_pt * scalePxPerPt : 0;
  const wrapperHeightPx = pageData ? pageData.page_height_pt * scalePxPerPt : 0;

  return (
    <>
      {isOpen && <div className={styles.scrim} onClick={onClose} aria-hidden="true" />}
      <div
        ref={panelRef}
        className={`${styles.panel} ${isOpen ? styles.open : ''}`}
        role="dialog"
        aria-modal="false"
        aria-label={documentId != null ? `${documentTitle}, Halaman ${page}` : 'Citation Viewer'}
        aria-hidden={!isOpen}
      >
        <div className={styles.header}>
          <div className={styles.headerText}>
            <span className={styles.docTitle} title={documentTitle}>
              {documentTitle}
            </span>
            {page != null && <span className={`${styles.pageBadge} mono`}>Hal. {page}</span>}
          </div>
          <button ref={closeButtonRef} type="button" className={styles.closeButton} onClick={onClose} aria-label="Tutup">
            ✕
          </button>
        </div>

        <ZoomControls zoomPercent={zoomPercent} onZoomIn={handleZoomIn} onZoomOut={handleZoomOut} onFit={handleFit} />

        <div ref={containerRef} className={`${styles.imageArea} scroll-region`} onWheel={handleWheel}>
          {loadStatus === 'loading' || loadStatus === 'idle' ? (
            <div className={styles.skeleton}>
              <div className={styles.skeletonPage} />
              <AIPulseLoader label={null} size="sm" className={styles.skeletonLoader} />
            </div>
          ) : loadStatus === 'error' ? (
            <div className={styles.errorState}>
              <span className={styles.errorIcon} aria-hidden="true">
                ⚠
              </span>
              <p>Gagal memuat halaman sumber.</p>
              <Button variant="secondary" size="sm" onClick={handleRetry}>
                Coba Lagi
              </Button>
            </div>
          ) : (
            pageData && (
              <div className={styles.pageWrapper} style={{ width: wrapperWidthPx, height: wrapperHeightPx }}>
                {pageData.page_image_uri && (
                  <img
                    src={pageData.page_image_uri}
                    alt={`Halaman ${pageData.page_number}, ${documentTitle}`}
                    className={styles.pageImage}
                    style={{ width: wrapperWidthPx, height: wrapperHeightPx }}
                  />
                )}
                {pageData.elements
                  .filter((el) => String(el.element_id) !== String(citation?.element_id))
                  .map((el) =>
                    el.bbox ? (
                      <BboxOverlay
                        key={el.element_id}
                        bbox={el.bbox}
                        pageHeightPt={pageData.page_height_pt}
                        scalePxPerPt={scalePxPerPt}
                        variant={el.element_type === 'icon' ? 'context-icon' : 'context'}
                      />
                    ) : null,
                  )}
                {activeElement?.bbox && (
                  <BboxOverlay
                    key={citation.element_id}
                    bbox={activeElement.bbox}
                    pageHeightPt={pageData.page_height_pt}
                    scalePxPerPt={scalePxPerPt}
                    variant="primary"
                    flash
                  />
                )}
              </div>
            )
          )}
        </div>

        <div className={styles.footer}>
          {typeMeta && (
            <span className={styles.typeBadge}>
              <span aria-hidden="true">{typeMeta.glyph}</span> {typeMeta.label}
            </span>
          )}
          {description && <p className={styles.description}>{description}</p>}
          {citation?.quote && <blockquote className={styles.quote}>&ldquo;{citation.quote}&rdquo;</blockquote>}
          {loadStatus === 'error' && citation?.quote && (
            <p className={styles.errorFallbackNote}>Kutipan tetap ditampilkan meski gambar gagal dimuat.</p>
          )}
        </div>

        <CitationPager
          current={activeIndex}
          total={citations.length}
          onPrev={() => onNavigate(activeIndex - 1)}
          onNext={() => onNavigate(activeIndex + 1)}
        />
      </div>
    </>
  );
}
