import { useEffect, useRef, useState } from 'react';
import styles from './UserMenu.module.css';

/**
 * Topbar user menu — Documentation/ui-ux-design/03-app-shell-navigation.md §7.
 * No confirmation dialog on logout (deliberate decision, §7).
 */
export function UserMenu({ username, role, onLogout }) {
  const [isOpen, setIsOpen] = useState(false);
  const rootRef = useRef(null);
  const initial = username ? username.charAt(0).toUpperCase() : '?';

  useEffect(() => {
    if (!isOpen) return undefined;
    const handleClickOutside = (event) => {
      if (rootRef.current && !rootRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    const handleEscape = (event) => {
      if (event.key === 'Escape') setIsOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isOpen]);

  return (
    <div className={styles.root} ref={rootRef}>
      <button
        type="button"
        className={styles.trigger}
        onClick={() => setIsOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={isOpen}
      >
        <span className={styles.avatar} aria-hidden="true">
          {initial}
        </span>
        <span className={styles.username}>{username}</span>
        <span aria-hidden="true">▾</span>
      </button>

      {isOpen && (
        <div className={styles.dropdown} role="menu">
          <div className={styles.dropdownHeader}>
            <div className={styles.dropdownName}>{username}</div>
            <span className={styles.roleBadge}>{role}</span>
          </div>
          <div className={styles.divider} />
          <button
            type="button"
            role="menuitem"
            className={styles.dropdownItem}
            onClick={() => {
              setIsOpen(false);
              onLogout();
            }}
          >
            Keluar
          </button>
        </div>
      )}
    </div>
  );
}
