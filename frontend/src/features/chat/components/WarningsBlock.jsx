import styles from './WarningsBlock.module.css';

/**
 * Documentation/ui-ux-design/04-chat-ui.md §8. Backend already ships
 * `Warning.severity` (`05-streaming-and-api-contract.md` §5.3), so this
 * implements the "upgrade siap-pakai" two-block split directly rather than
 * the interim single-block fallback the wireframe describes for a
 * not-yet-available field.
 *
 * @param {{ warnings: Array<{message: string, severity: 'safety'|'note'}> }} props
 */
export function WarningsBlock({ warnings }) {
  if (!warnings || warnings.length === 0) return null;
  const safety = warnings.filter((w) => w.severity === 'safety');
  const notes = warnings.filter((w) => w.severity !== 'safety');

  return (
    <div className={styles.root}>
      {safety.length > 0 && (
        <div className={`${styles.block} ${styles.safety}`}>
          <div className={styles.header}>
            <span className={styles.icon} aria-hidden="true">
              ⚠
            </span>
            <span className={styles.title}>Peringatan Keselamatan</span>
          </div>
          <ul className={styles.list}>
            {safety.map((w, i) => (
              <li key={i}>{w.message}</li>
            ))}
          </ul>
        </div>
      )}
      {notes.length > 0 && (
        <div className={`${styles.block} ${styles.note}`}>
          <div className={styles.header}>
            <span className={styles.icon} aria-hidden="true">
              ⚠
            </span>
            <span className={styles.title}>Peringatan</span>
          </div>
          <ul className={styles.list}>
            {notes.map((w, i) => (
              <li key={i}>{w.message}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
