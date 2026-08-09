import { useEffect, useState } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/useAuth';
import { filterNavByScopes, flattenRouteRequirements } from '../auth/filterNavByScopes';
import { RouteGuard } from '../auth/RouteGuard';
import { ForbiddenPage } from '../auth/ForbiddenPage';
import { AIPulseLoader } from '../components/AIPulseLoader/AIPulseLoader';
import { Button } from '../components/Button/Button';
import { useScrollbarIdle } from '../hooks/useScrollbarIdle';
import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';
import styles from './AppShell.module.css';

const SIDEBAR_COLLAPSED_KEY = 'vrf-chat.sidebarCollapsed';
const SIDEBAR_DOM_ID = 'app-shell-sidebar';

function getPageTitle(pathname) {
  const found = flattenRouteRequirements().find((r) => r.route === pathname);
  return found?.label ?? 'VRF/VRV AI';
}

export function AppShell() {
  const { user, scopes, authMeStatus, apiForbidden, clearApiForbidden, fetchMe, logoutUser, registerOnSessionExpired } =
    useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === 'true',
  );
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
  const contentScrollRef = useScrollbarIdle();

  useEffect(() => {
    registerOnSessionExpired(() => navigate('/login', { replace: true }));
  }, [registerOnSessionExpired, navigate]);

  useEffect(() => {
    // §3.1/F-8: an API-triggered 403 is scoped to "the page/action the user
    // was on when it happened" — clear it on navigation so moving to a
    // different (permitted) page doesn't keep showing the stale "tidak
    // punya akses" state.
    clearApiForbidden();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname]);

  useEffect(() => {
    // Avoid a duplicate fetch if we just came from a successful login
    // (AuthProvider.loginWithCredentials already populated `user`).
    if (authMeStatus === 'idle') {
      fetchMe().catch(() => {
        /* handled via authMeStatus state, see below */
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleSidebarCollapsed = () => {
    setSidebarCollapsed((prev) => {
      const next = !prev;
      localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(next));
      return next;
    });
  };

  if (authMeStatus === 'error-401') {
    // registerOnSessionExpired already triggers the redirect; render nothing
    // meanwhile to avoid a flash of shell chrome.
    return null;
  }

  if (authMeStatus === 'error-network') {
    return (
      <div className={styles.errorPanel}>
        <span className={styles.errorIcon} aria-hidden="true">
          ⚠
        </span>
        <p>
          Gagal memuat sesi.
          <br />
          Periksa koneksi Anda.
        </p>
        <div className={styles.errorActions}>
          <Button variant="primary" onClick={() => fetchMe().catch(() => {})}>
            Coba Lagi
          </Button>
          <Button variant="secondary" onClick={() => navigate('/login', { replace: true })}>
            Ke Login
          </Button>
        </div>
      </div>
    );
  }

  if (authMeStatus !== 'success') {
    return (
      <div className={styles.shell}>
        <div className={styles.skeletonSidebar}>
          <div className={styles.skeletonBrand}>
            <span aria-hidden="true">◈</span>
            <span>VRF/VRV AI</span>
          </div>
          <div className={styles.skeletonBar} style={{ width: '80%' }} />
          <div className={styles.skeletonBar} style={{ width: '60%' }} />
          <div className={styles.skeletonBar} style={{ width: '70%' }} />
        </div>
        <div className={styles.main}>
          <div className={styles.skeletonTopbar}>
            <div className={styles.skeletonBar} style={{ width: 160 }} />
            <div className={styles.skeletonAvatar} />
          </div>
          <div className={styles.skeletonContent}>
            <AIPulseLoader label="Memuat sesi..." />
          </div>
        </div>
      </div>
    );
  }

  const navItems = filterNavByScopes(scopes);

  return (
    <div className={styles.shell}>
      <div className={styles.desktopSidebar}>
        <Sidebar
          id={SIDEBAR_DOM_ID}
          items={navItems}
          collapsed={sidebarCollapsed}
          onToggleCollapse={toggleSidebarCollapsed}
        />
      </div>

      {mobileDrawerOpen && (
        <>
          <div className={styles.drawerBackdrop} onClick={() => setMobileDrawerOpen(false)} />
          <div className={`${styles.drawer} ${styles.open}`}>
            <Sidebar
              items={navItems}
              collapsed={false}
              onToggleCollapse={() => setMobileDrawerOpen(false)}
              onNavigate={() => setMobileDrawerOpen(false)}
            />
          </div>
        </>
      )}

      <div className={styles.main}>
        <Topbar
          title={getPageTitle(location.pathname)}
          sidebarId={SIDEBAR_DOM_ID}
          onHamburgerClick={() => setMobileDrawerOpen(true)}
          user={user}
          onLogout={async () => {
            await logoutUser();
            navigate('/login', { replace: true });
          }}
        />
        <div ref={contentScrollRef} className={`${styles.content} scroll-region`}>
          {apiForbidden ? (
            <ForbiddenPage />
          ) : (
            <RouteGuard>
              <Outlet />
            </RouteGuard>
          )}
        </div>
      </div>
    </div>
  );
}
