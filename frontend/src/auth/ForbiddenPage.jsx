import { useNavigate } from 'react-router-dom';
import { Button } from '../components/Button/Button';
import styles from './ForbiddenPage.module.css';

/**
 * §6 of 03-app-shell-navigation.md — rendered inside the shell content area
 * (sidebar/topbar stay as-is), not a standalone error page.
 */
export function ForbiddenPage() {
  const navigate = useNavigate();

  return (
    <div className={styles.root}>
      <span className={styles.icon} aria-hidden="true">
        🔒
      </span>
      <p className={styles.message}>Anda tidak memiliki akses ke halaman ini.</p>
      <Button variant="primary" onClick={() => navigate('/chat')}>
        Kembali ke Chat
      </Button>
    </div>
  );
}
