import { usePrefersReducedMotion } from '../../hooks/usePrefersReducedMotion';
import styles from './AIPulseLoader.module.css';

/**
 * "AI thinking" / async-in-progress indicator.
 * Spec: Documentation/ui-ux-design/01-design-system.md §5.3 (category 3), §5.4.
 *
 * @param {{ label?: string, size?: 'sm' | 'md', className?: string }} props
 *   `label` defaults to "Memproses..." and is always rendered (visually) when
 *   reduced motion is active, per §5.4 ("ikon diam + label teks
 *   'Memproses...'"). When motion is allowed, `label` is optional — pass
 *   `null` to omit it (e.g. inline in a button where space is tight).
 */
export function AIPulseLoader({ label = 'Memproses...', size = 'md', className = '' }) {
  const prefersReducedMotion = usePrefersReducedMotion();
  const dotSize = size === 'sm' ? styles.dotSm : '';
  const dotClassName = `${styles.dot} ${dotSize} ${prefersReducedMotion ? styles.dotReduced : ''}`;

  return (
    <span className={`${styles.root} ${className}`} role="status" aria-live="polite">
      {/* §5.4: loop stops entirely under reduced motion ("ikon diam") — the
          dots render static via styles.dotReduced, no keyframe animation. */}
      <span className={styles.dots} aria-hidden="true">
        <span className={dotClassName} />
        <span className={dotClassName} />
        <span className={dotClassName} />
      </span>
      {label && <span className={styles.label}>{label}</span>}
    </span>
  );
}
