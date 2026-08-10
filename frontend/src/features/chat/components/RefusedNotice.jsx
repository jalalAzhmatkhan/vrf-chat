import { RelatedChips } from './RelatedChips';
import { CitationList } from './CitationList';
import styles from './RefusedNotice.module.css';

/**
 * Documentation/ui-ux-design/04-chat-ui.md §11 — deliberately neutral/info
 * color, not danger/warning: this is a correct, intentional "never invent"
 * refusal, not an application failure.
 *
 * @param {{
 *   answer: string,
 *   relatedErrorCodes: string[],
 *   relatedComponents: string[],
 *   citations: Array<object>,
 *   documentsMap: Map<string, {title:string}>,
 *   onOpenCitation: (elementId: string) => void,
 *   onFillComposer: (text: string) => void,
 * }} props
 */
export function RefusedNotice({
  answer,
  relatedErrorCodes,
  relatedComponents,
  citations,
  documentsMap,
  onOpenCitation,
  onFillComposer,
}) {
  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <span className={styles.icon} aria-hidden="true">
          ⌕
        </span>
        <span className={styles.title}>Tidak ditemukan di manual</span>
      </div>
      <p className={styles.body}>{answer}</p>
      <p className={styles.hint}>
        Coba ajukan dengan detail lain (kode error, model unit, atau nama komponen spesifik).
      </p>
      <RelatedChips
        relatedErrorCodes={relatedErrorCodes}
        relatedComponents={relatedComponents}
        onFillComposer={onFillComposer}
      />
      {citations.length > 0 && (
        <CitationList
          citations={citations}
          documentsMap={documentsMap}
          confidence={undefined}
          refused
          onOpenCitation={onOpenCitation}
        />
      )}
    </div>
  );
}
