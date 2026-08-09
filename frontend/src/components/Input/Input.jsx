import { forwardRef, useId } from 'react';
import styles from './Input.module.css';

/**
 * Input — Documentation/ui-ux-design/01-design-system.md §7.2.
 *
 * @param {{
 *   label?: string,
 *   error?: string,
 *   helperText?: string,
 *   mono?: boolean,
 *   trailing?: React.ReactNode,
 * } & React.InputHTMLAttributes<HTMLInputElement>} props
 */
export const Input = forwardRef(function Input(
  { label, error, helperText, mono = false, trailing, id, className = '', ...rest },
  ref,
) {
  const generatedId = useId();
  const inputId = id ?? generatedId;
  const errorId = error ? `${inputId}-error` : undefined;
  const helperId = helperText ? `${inputId}-helper` : undefined;

  return (
    <div className={`${styles.field} ${error ? styles.hasError : ''} ${className}`}>
      {label && (
        <label className={styles.label} htmlFor={inputId}>
          {label}
        </label>
      )}
      <div className={styles.inputWrapper}>
        <input
          ref={ref}
          id={inputId}
          className={`${styles.input} ${mono ? styles.mono : ''}`}
          style={trailing ? { paddingRight: 44 } : undefined}
          aria-invalid={Boolean(error) || undefined}
          aria-describedby={errorId ?? helperId}
          {...rest}
        />
        {trailing && <div className={styles.trailingSlot}>{trailing}</div>}
      </div>
      {error ? (
        <span id={errorId} className={styles.errorText}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path
              d="M8 1.5 1 14h14L8 1.5Z"
              stroke="currentColor"
              strokeWidth="1.3"
              strokeLinejoin="round"
              fill="none"
            />
            <path d="M8 6.2v3.3M8 11.6v.1" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
          </svg>
          {error}
        </span>
      ) : (
        helperText && (
          <span id={helperId} className={styles.helperText}>
            {helperText}
          </span>
        )
      )}
    </div>
  );
});
