import styles from './CitationPager.module.css';

/** Documentation/ui-ux-design/05-citation-viewer.md §7. */
export function CitationPager({ current, total, onPrev, onNext }) {
  if (total <= 1) return null;
  return (
    <div className={styles.root}>
      <button type="button" className={styles.arrow} onClick={onPrev} disabled={current === 0} aria-label="Sitasi sebelumnya">
        ‹
      </button>
      <span className="mono">
        Sitasi {current + 1}/{total}
      </span>
      <button
        type="button"
        className={styles.arrow}
        onClick={onNext}
        disabled={current === total - 1}
        aria-label="Sitasi berikutnya"
      >
        ›
      </button>
    </div>
  );
}
