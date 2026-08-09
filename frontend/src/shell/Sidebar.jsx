import { NavItem } from './NavItem';
import { useScrollbarIdle } from '../hooks/useScrollbarIdle';
import styles from './Sidebar.module.css';

/**
 * Props: `items` — already scope-filtered by filterNavByScopes() at the
 * AppShell level; Sidebar performs no filtering of its own
 * (03-app-shell-navigation.md §8 "single source of truth"). Top-level
 * groups with `isSectionHeader` (currently only "Admin") render an uppercase
 * divider label above their children, per §2.1 wireframe.
 */
export function Sidebar({ items, collapsed, onToggleCollapse, id, onNavigate }) {
  const scrollRef = useScrollbarIdle();

  return (
    <aside
      ref={scrollRef}
      id={id}
      className={`${styles.sidebar} ${collapsed ? styles.collapsed : ''} scroll-region`}
    >
      <div className={styles.brand}>
        <span className={styles.brandMark} aria-hidden="true">
          ◈
        </span>
        {!collapsed && <span className={styles.brandLabel}>VRF/VRV AI</span>}
      </div>

      <nav className={styles.nav} onClick={onNavigate}>
        {items.map((item) => (
          <NavSection key={item.key} item={item} collapsed={collapsed} />
        ))}
      </nav>

      <div className={styles.footer}>
        <button
          type="button"
          className={styles.collapseToggle}
          onClick={onToggleCollapse}
          aria-label={collapsed ? 'Perluas sidebar' : 'Ciutkan sidebar'}
        >
          <span aria-hidden="true">{collapsed ? '›' : '‹'}</span>
          {!collapsed && <span>Ciutkan</span>}
        </button>
      </div>
    </aside>
  );
}

function NavSection({ item, collapsed }) {
  if (item.isSectionHeader) {
    return (
      <div>
        {!collapsed && <div className={styles.sectionLabel}>{item.label}</div>}
        {(item.children ?? []).map((child) => (
          <NavItem key={child.key} item={child} collapsed={collapsed} />
        ))}
      </div>
    );
  }

  return (
    <div>
      {!collapsed && <div className={styles.sectionLabel}>General</div>}
      <NavItem item={item} collapsed={collapsed} />
    </div>
  );
}
