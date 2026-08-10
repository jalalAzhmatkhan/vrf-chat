import { useEffect, useState } from 'react';
import { Button } from '../../../components/Button/Button';
import { getElementCached } from '../../../lib/api/elementsApi';
import { TableCard } from './TableCard';
import styles from './VisualReferenceCard.module.css';

/**
 * Documentation/ui-ux-design/04-chat-ui.md §7 — figure/diagram/table
 * citations that are *not* referenced by an inline `{{el:ID}}` marker.
 *
 * **Table rendering** (`element_type === "table"`): renders a real
 * `<table>` (`TableCard`) from `Citation.content_structured.rows`
 * (`05-streaming-and-api-contract.md` §5.2.1) — this supersedes the earlier
 * FE deviation (image crop / raw mono text fallback) now that the backend
 * exposes structured rows directly on the citation. The old fallback chain
 * is **kept**, not removed, for the case `content_structured` is
 * unavailable: prefer `citation.content_structured` (fast path, no
 * roundtrip), else lazily resolve via `GET /api/v1/elements/{id}` (also
 * covers the truncation case below), else fall back to an `image_uri` crop
 * if the ingestion pipeline produced one, else the raw extracted `text` in
 * a monospaced, scrollable block.
 *
 * **Truncation** (§5.2.1: >200 rows or >50KB → cut to 100 rows +
 * `truncated: true`): shown as an explicit notice, with a "Lihat semua
 * baris" action that fetches the untruncated `content_structured` via
 * `GET /api/v1/elements/{id}` (§6 — that endpoint is contracted to always
 * return the full, non-truncated table) — in addition to (not instead of)
 * "Lihat sumber" which opens the Citation Viewer for visual verification
 * against the source page.
 *
 * @param {{
 *   citation: {document_id:string, page:number, element_id:string, element_type:string, quote:string|null, image_uri:string|null, content_structured:{rows:Array<object>,headers?:string[],truncated?:boolean}|null},
 *   documentsMap: Map<string, {title:string}>,
 *   onOpenCitation: (elementId: string) => void,
 * }} props
 */
export function VisualReferenceCard({ citation, documentsMap, onOpenCitation }) {
  const documentTitle = documentsMap?.get(String(citation.document_id))?.title ?? `Dokumen ${citation.document_id}`;
  const isTable = citation.element_type === 'table';

  const [tableDetail, setTableDetail] = useState(null);
  const [fullContentStructured, setFullContentStructured] = useState(null);
  const [loadingFullTable, setLoadingFullTable] = useState(false);

  const contentStructured = fullContentStructured ?? citation.content_structured ?? tableDetail?.content_structured ?? null;
  const hasRows = Array.isArray(contentStructured?.rows) && contentStructured.rows.length > 0;

  useEffect(() => {
    if (!isTable || contentStructured || citation.image_uri) return undefined;
    let cancelled = false;
    getElementCached(citation.element_id)
      .then((el) => {
        if (!cancelled) setTableDetail(el);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // Only re-run if the resolved data is still missing — `contentStructured`
    // itself is intentionally excluded to avoid re-fetching once resolved.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTable, citation.image_uri, citation.element_id]);

  const handleLoadFullTable = () => {
    setLoadingFullTable(true);
    getElementCached(citation.element_id)
      .then((el) => {
        if (el?.content_structured) setFullContentStructured(el.content_structured);
      })
      .catch(() => {})
      .finally(() => setLoadingFullTable(false));
  };

  const imageUri = citation.image_uri ?? tableDetail?.image_uri ?? null;
  const fallbackText = citation.quote ?? tableDetail?.text ?? null;
  const caption = citation.visual_description?.description ?? (isTable ? 'Tabel' : 'Gambar/skema');
  const isTruncated = Boolean(contentStructured?.truncated) && !fullContentStructured;

  if (isTable && hasRows) {
    // §7 "Table card" layout: caption above the table, table full-width,
    // footer row (source + "Lihat sumber") below — distinct from the
    // figure/diagram layout below (thumbnail + meta side by side).
    return (
      <div className={styles.card}>
        <div className={styles.tableBody}>
          <span className={styles.caption}>{caption}</span>
          <TableCard contentStructured={contentStructured} />
          {isTruncated && (
            <p className={styles.truncatedNotice}>
              ⚠ Tabel ini dipotong — menampilkan sebagian baris.{' '}
              <button
                type="button"
                className={styles.loadFullLink}
                onClick={handleLoadFullTable}
                disabled={loadingFullTable}
              >
                {loadingFullTable ? 'Memuat…' : 'Lihat semua baris'}
              </button>
            </p>
          )}
          <div className={styles.tableFooter}>
            <span className={styles.source}>
              {documentTitle} · <span className="mono">Hal. {citation.page}</span>
            </span>
            <Button variant="tertiary" size="sm" onClick={() => onOpenCitation(citation.element_id)}>
              Lihat sumber
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.card}>
      <div className={styles.body}>
        {imageUri ? (
          <div className={styles.thumbWrap}>
            <img src={imageUri} alt={caption} className={styles.thumb} />
          </div>
        ) : (
          isTable && fallbackText && <pre className={`${styles.textFallback} mono scroll-region`}>{fallbackText}</pre>
        )}
        <div className={styles.meta}>
          <span className={styles.caption}>{caption}</span>
          <span className={styles.source}>
            {documentTitle} · <span className="mono">Hal. {citation.page}</span>
          </span>
          <Button variant="tertiary" size="sm" onClick={() => onOpenCitation(citation.element_id)}>
            {isTable ? 'Lihat sumber' : 'Perbesar'}
          </Button>
        </div>
      </div>
    </div>
  );
}
