import styles from './RelatedChips.module.css';

/**
 * Documentation/ui-ux-design/04-chat-ui.md §9. Click-to-prefill is
 * implemented (marked optional in the spec) since it's cheap given the
 * composer prefill plumbing already exists.
 *
 * @param {{
 *   relatedErrorCodes: string[],
 *   relatedComponents: string[],
 *   onFillComposer: (text: string) => void,
 * }} props
 */
export function RelatedChips({ relatedErrorCodes, relatedComponents, onFillComposer }) {
  const hasErrorCodes = relatedErrorCodes && relatedErrorCodes.length > 0;
  const hasComponents = relatedComponents && relatedComponents.length > 0;
  if (!hasErrorCodes && !hasComponents) return null;

  return (
    <div className={styles.root}>
      {hasErrorCodes && (
        <div className={styles.row}>
          <span className={styles.label}>Kode error terkait:</span>
          <div className={styles.chips}>
            {relatedErrorCodes.map((code) => (
              <button
                key={code}
                type="button"
                className={`${styles.chip} mono`}
                onClick={() => onFillComposer(`Apa penyebab lain kode error ${code}?`)}
              >
                {code}
              </button>
            ))}
          </div>
        </div>
      )}
      {hasComponents && (
        <div className={styles.row}>
          <span className={styles.label}>Komponen terkait:</span>
          <div className={styles.chips}>
            {relatedComponents.map((component) => (
              <button
                key={component}
                type="button"
                className={`${styles.chip} mono`}
                onClick={() => onFillComposer(`Jelaskan lebih detail soal komponen ${component}`)}
              >
                {component}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
