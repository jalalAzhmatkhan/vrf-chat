import styles from './ZoomControls.module.css';

/** Documentation/ui-ux-design/05-citation-viewer.md §6. */
export function ZoomControls({ zoomPercent, onZoomIn, onZoomOut, onFit }) {
  return (
    <div className={styles.root}>
      <button type="button" className={styles.button} onClick={onZoomOut} aria-label="Perkecil">
        −
      </button>
      <span className={`${styles.percent} mono`}>{Math.round(zoomPercent)}%</span>
      <button type="button" className={styles.button} onClick={onZoomIn} aria-label="Perbesar">
        +
      </button>
      <button type="button" className={styles.button} onClick={onFit} aria-label="Sesuaikan ke tampilan otomatis">
        ⤢
      </button>
    </div>
  );
}
