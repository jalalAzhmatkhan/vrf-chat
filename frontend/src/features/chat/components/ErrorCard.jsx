import { Button } from '../../../components/Button/Button';
import styles from './ErrorCard.module.css';

/**
 * Documentation/ui-ux-design/04-chat-ui.md §12 — the one place in chat
 * allowed to use "danger red" (distinct from §11 refusal's neutral info
 * color, a deliberate signal per that section).
 *
 * @param {{ code?: string, message: string, onRetry: () => void }} props
 */
export function ErrorCard({ code, message, onRetry }) {
  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <span className={styles.icon} aria-hidden="true">
          ⛔
        </span>
        <span className={styles.title}>Gagal memproses permintaan</span>
      </div>
      <p className={styles.message}>{message}</p>
      {code && <p className={`${styles.code} mono`}>{code}</p>}
      <Button variant="secondary" size="sm" onClick={onRetry}>
        Coba Lagi
      </Button>
    </div>
  );
}
