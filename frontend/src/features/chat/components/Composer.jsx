import { useEffect, useRef, useState } from 'react';
import { Button } from '../../../components/Button/Button';
import styles from './Composer.module.css';

const MAX_HEIGHT_PX = 144; // ~6 lines

/**
 * Documentation/ui-ux-design/04-chat-ui.md §13.
 *
 * @param {{ disabled: boolean, value: string, onChange: (v: string) => void, onSubmit: (text: string) => void }} props
 */
export function Composer({ disabled, value, onChange, onSubmit }) {
  const textareaRef = useRef(null);
  const [localValue, setLocalValue] = useState(value ?? '');

  // Allow parents (RelatedChips/EmptyState prefill) to push a value in.
  useEffect(() => {
    if (value !== undefined && value !== localValue) setLocalValue(value);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = 'auto';
    node.style.height = `${Math.min(node.scrollHeight, MAX_HEIGHT_PX)}px`;
  }, [localValue]);

  const handleChange = (event) => {
    setLocalValue(event.target.value);
    onChange?.(event.target.value);
  };

  const submit = () => {
    const trimmed = localValue.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setLocalValue('');
    onChange?.('');
    textareaRef.current?.focus();
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div className={styles.root}>
      <div className={styles.field}>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          placeholder="Tulis pertanyaan tentang servis VRF/VRV..."
          value={localValue}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
        />
        <Button
          variant="primary"
          size="md"
          className={styles.sendButton}
          onClick={submit}
          disabled={disabled || localValue.trim().length === 0}
          aria-label="Kirim pertanyaan"
        >
          <span aria-hidden="true">➤</span>
        </Button>
      </div>
      <span className={styles.helper}>Enter untuk kirim · Shift+Enter baris baru</span>
    </div>
  );
}
