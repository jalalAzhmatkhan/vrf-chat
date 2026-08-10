import { useState } from 'react';
import { ConfidenceBadge } from './ConfidenceBadge';
import { getCitationTypeMeta } from '../lib/citationTypeMeta';
import styles from './CitationList.module.css';

function truncateQuote(quote, max = 80) {
  if (!quote) return null;
  const trimmed = quote.trim();
  return trimmed.length > max ? `${trimmed.slice(0, max).trimEnd()}…` : trimmed;
}

/**
 * Documentation/ui-ux-design/04-chat-ui.md §10.
 *
 * @param {{
 *   citations: Array<{document_id:string, page:number, element_id:string, element_type:string, quote:string|null}>,
 *   documentsMap: Map<string, {title:string}>,
 *   confidence: number,
 *   refused: boolean,
 *   onOpenCitation: (elementId: string) => void,
 * }} props
 */
export function CitationList({ citations, documentsMap, confidence, refused, onOpenCitation }) {
  const [expanded, setExpanded] = useState(citations.length <= 2);

  if (citations.length === 0 && confidence === undefined) return null;

  return (
    <div className={styles.root}>
      <div className={styles.summaryRow}>
        {!refused && <ConfidenceBadge confidence={confidence} />}
        {citations.length > 0 && (
          <button
            type="button"
            className={styles.toggle}
            onClick={() => setExpanded((v) => !v)}
            aria-expanded={expanded}
          >
            <span aria-hidden="true">{expanded ? '▾' : '▸'}</span>
            {refused ? 'Referensi yang diperiksa' : 'Sumber'} ({citations.length})
          </button>
        )}
      </div>
      {expanded && citations.length > 0 && (
        <ul className={styles.list}>
          {citations.map((citation) => {
            const meta = getCitationTypeMeta(citation.element_type);
            const documentTitle = documentsMap?.get(String(citation.document_id))?.title ?? `Dokumen ${citation.document_id}`;
            const quote = truncateQuote(citation.quote);
            return (
              <li key={citation.element_id}>
                <button
                  type="button"
                  className={styles.row}
                  onClick={() => onOpenCitation(citation.element_id)}
                >
                  <span className={styles.glyph} aria-hidden="true">
                    {meta.glyph}
                  </span>
                  <span className={styles.rowBody}>
                    <span className={styles.rowMeta}>
                      {documentTitle} · <span className="mono">Hal. {citation.page}</span> · {meta.label}
                    </span>
                    {quote && <span className={styles.rowQuote}>&ldquo;{quote}&rdquo;</span>}
                  </span>
                  <span className={styles.chevron} aria-hidden="true">
                    ›
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
