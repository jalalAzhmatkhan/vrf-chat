import { forwardRef } from 'react';
import { AIPulseLoader } from '../AIPulseLoader/AIPulseLoader';
import styles from './Button.module.css';

/**
 * Button — Documentation/ui-ux-design/01-design-system.md §7.1.
 *
 * @param {{
 *   variant?: 'primary' | 'secondary' | 'tertiary' | 'destructive' | 'icon',
 *   size?: 'sm' | 'md' | 'lg',
 *   loading?: boolean,
 *   children?: React.ReactNode,
 * } & React.ButtonHTMLAttributes<HTMLButtonElement>} props
 */
export const Button = forwardRef(function Button(
  { variant = 'primary', size = 'md', loading = false, disabled, className = '', children, ...rest },
  ref,
) {
  const isIcon = variant === 'icon';
  const variantClass = isIcon ? styles.tertiary : styles[variant] ?? styles.primary;

  return (
    <button
      ref={ref}
      type={rest.type ?? 'button'}
      className={`${styles.base} ${styles[size]} ${variantClass} ${isIcon ? styles.icon : ''} ${className}`}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...rest}
    >
      {loading ? <AIPulseLoader label={null} size="sm" /> : <span className={styles.label}>{children}</span>}
    </button>
  );
});
