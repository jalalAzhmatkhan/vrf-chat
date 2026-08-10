import { useEffect, useState } from 'react';
import { Button } from '../../../components/Button/Button';
import { getElementCached } from '../../../lib/api/elementsApi';
import styles from './VisualReferenceCard.module.css';

/**
 * Documentation/ui-ux-design/04-chat-ui.md §7 — figure/diagram/table
 * citations that are *not* referenced by an inline `{{el:ID}}` marker.
 *
 * **Table gap note** (see STATUS REPORT / DIRECT MESSAGE -> System
 * Analyst): the contract does not expose structured table rows anywhere
 * (`Citation.image_uri`/`visual_description` are scoped to
 * icon/figure/diagram only — §5.2 — and `GET /api/v1/elements/{id}` has no
 * `rows` field either, `05-citation-viewer.md` §8 says the same for the
 * viewer). This renders whatever the element actually has — an
 * `image_uri` crop if the ingestion pipeline produced one, else the raw
 * extracted `text` in a monospaced, scrollable block — rather than
 * fabricating an HTML `<table>` from data that doesn't exist. Visual
 * verification against the highlighted page region (via "Lihat sumber")
 * is the source of truth either way, consistent with
 * `05-citation-viewer.md` §1/§8.
 *
 * @param {{
 *   citation: {document_id:string, page:number, element_id:string, element_type:string, quote:string|null, image_uri:string|null},
 *   documentsMap: Map<string, {title:string}>,
 *   onOpenCitation: (elementId: string) => void,
 * }} props
 */
export function VisualReferenceCard({ citation, documentsMap, onOpenCitation }) {
  const documentTitle = documentsMap?.get(String(citation.document_id))?.title ?? `Dokumen ${citation.document_id}`;
  const isTable = citation.element_type === 'table';

  const [tableDetail, setTableDetail] = useState(null);
  useEffect(() => {
    if (!isTable || citation.image_uri) return undefined;
    let cancelled = false;
    getElementCached(citation.element_id)
      .then((el) => {
        if (!cancelled) setTableDetail(el);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [isTable, citation.image_uri, citation.element_id]);

  const imageUri = citation.image_uri ?? tableDetail?.image_uri ?? null;
  const fallbackText = citation.quote ?? tableDetail?.text ?? null;
  const caption = citation.visual_description?.description ?? (isTable ? 'Tabel' : 'Gambar/skema');

  return (
    <div className={styles.card}>
      <div className={styles.body}>
        {imageUri ? (
          <div className={styles.thumbWrap}>
            <img src={imageUri} alt={caption} className={styles.thumb} />
          </div>
        ) : (
          isTable &&
          fallbackText && (
            <pre className={`${styles.textFallback} mono scroll-region`}>{fallbackText}</pre>
          )
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
