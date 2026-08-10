import { useEffect, useState } from 'react';
import { AIPulseLoader } from '../../../components/AIPulseLoader/AIPulseLoader';
import styles from './AssistantStatusIndicator.module.css';

/**
 * Documentation/ui-ux-design/04-chat-ui.md §3.
 * `stage` enum confirmed in
 * Documentation/system-design/05-streaming-and-api-contract.md §4.1.
 */
const STAGE_LABELS = {
  searching_manual: 'Mencari di manual...',
  building_context: 'Menyusun konteks...',
  generating_answer: 'Menyusun jawaban...',
};

const REASSURANCE_THRESHOLD_SECONDS = 15;

function humanizeStage(stage) {
  if (!stage) return 'Menghubungkan...';
  if (STAGE_LABELS[stage]) return STAGE_LABELS[stage];
  // Fallback for unrecognized stage values (should not normally happen —
  // §4.1 treats the enum as fixed): humanize snake_case -> Title Case.
  return `${stage
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')}...`;
}

function formatElapsed(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

/**
 * @param {{ stage: string|null, requestStartedAt: number }} props
 */
export function AssistantStatusIndicator({ stage, requestStartedAt }) {
  const [elapsedSeconds, setElapsedSeconds] = useState(() =>
    Math.max(0, Math.floor((Date.now() - requestStartedAt) / 1000)),
  );

  useEffect(() => {
    const interval = window.setInterval(() => {
      setElapsedSeconds(Math.max(0, Math.floor((Date.now() - requestStartedAt) / 1000)));
    }, 1000);
    return () => window.clearInterval(interval);
  }, [requestStartedAt]);

  const label = humanizeStage(stage);
  const showReassurance = elapsedSeconds >= REASSURANCE_THRESHOLD_SECONDS;

  return (
    <div className={styles.root} role="status" aria-live="polite">
      <div className={styles.row}>
        <AIPulseLoader label={null} size="lg" />
        <span className={styles.label}>{label}</span>
        <span className={`${styles.timer} mono`}>{formatElapsed(elapsedSeconds)}</span>
      </div>
      {showReassurance && (
        <p className={styles.reassurance}>Manual servis cukup panjang — jawaban sedang disusun.</p>
      )}
    </div>
  );
}
