import { UserMenu } from './UserMenu';
import styles from './Topbar.module.css';

export function Topbar({ title, onHamburgerClick, sidebarId, user, onLogout }) {
  return (
    <header className={styles.topbar}>
      <div className={styles.left}>
        <button
          type="button"
          className={styles.hamburger}
          onClick={onHamburgerClick}
          aria-label="Buka menu navigasi"
          aria-controls={sidebarId}
        >
          ≡
        </button>
        <span className={styles.title}>{title}</span>
      </div>
      {user && <UserMenu username={user.username} role={user.role} onLogout={onLogout} />}
    </header>
  );
}
