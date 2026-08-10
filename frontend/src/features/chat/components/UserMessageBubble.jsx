import styles from './UserMessageBubble.module.css';

function formatTime(timestamp) {
  return new Date(timestamp).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
}

/** Documentation/ui-ux-design/04-chat-ui.md §1. */
export function UserMessageBubble({ text, timestamp }) {
  return (
    <div className={styles.wrap}>
      <div className={styles.bubble}>{text}</div>
      <span className={`${styles.timestamp} mono`}>{formatTime(timestamp)}</span>
    </div>
  );
}
