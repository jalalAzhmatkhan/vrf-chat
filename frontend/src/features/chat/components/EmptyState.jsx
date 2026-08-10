import styles from './EmptyState.module.css';

const EXAMPLES = [
  'Apa penyebab kode error P8?',
  'Bagaimana cara masuk ke mode LOSSNAY?',
  'Berapa nilai resistansi normal untuk sensor TH3?',
  'Bagaimana prosedur mengganti PCB kontrol outdoor unit?',
];

/** Documentation/ui-ux-design/04-chat-ui.md §14. */
export function EmptyState({ onPickExample }) {
  return (
    <div className={`${styles.root} circuit-grid-bg`}>
      <span className={styles.mark} aria-hidden="true">
        ◈
      </span>
      <h2 className={styles.heading}>Tanya apa saja tentang servis VRF/VRV</h2>
      <div className={styles.examples}>
        {EXAMPLES.map((example) => (
          <button key={example} type="button" className={styles.exampleChip} onClick={() => onPickExample(example)}>
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
