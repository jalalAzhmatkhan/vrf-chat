import styles from './ConfidenceBadge.module.css';

/**
 * Documentation/ui-ux-design/04-chat-ui.md §10 — thresholds are an FE
 * heuristic explicitly flagged as provisional pending real eval data
 * (`07-evaluation-system.md`).
 * @param {{ confidence: number }} props
 */
export function ConfidenceBadge({ confidence }) {
  const pct = Math.round((confidence ?? 0) * 100);
  let tier = 'danger';
  let text = 'Keyakinan Rendah';
  if (confidence >= 0.75) {
    tier = 'success';
    text = 'Keyakinan Tinggi';
  } else if (confidence >= 0.4) {
    tier = 'warning';
    text = 'Keyakinan Sedang';
  }
  return (
    <span className={`${styles.badge} ${styles[tier]}`}>
      {text} ({pct}%)
    </span>
  );
}
