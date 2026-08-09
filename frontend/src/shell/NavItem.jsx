import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import styles from './NavItem.module.css';

function isRouteActive(pathname, route) {
  return pathname === route || pathname.startsWith(`${route}/`);
}

function containsActiveRoute(node, pathname) {
  if (node.route) return isRouteActive(pathname, node.route);
  return (node.children ?? []).some((child) => containsActiveRoute(child, pathname));
}

/**
 * Renders one leaf link, or (for `isCollapsibleSubNav` groups like RBAC) a
 * toggle + nested links — spec: 03-app-shell-navigation.md §2.1 ("Grup RBAC
 * ... auto-expand jika route aktif ada di dalamnya, selain itu default
 * collapsed").
 */
export function NavItem({ item, collapsed }) {
  const location = useLocation();
  const autoExpand = item.isCollapsibleSubNav ? containsActiveRoute(item, location.pathname) : false;
  const [expanded, setExpanded] = useState(autoExpand);
  const isExpanded = expanded || autoExpand;

  if (item.isCollapsibleSubNav) {
    if (collapsed) {
      // Icon-rail mode: clicking navigates to the last/first child directly
      // rather than expanding in place (§2.2).
      const target = item.children?.[0]?.route ?? '#';
      return (
        <NavLink to={target} className={styles.link} title={item.label}>
          <span className={styles.icon} aria-hidden="true">
            {item.icon}
          </span>
        </NavLink>
      );
    }

    return (
      <div>
        <button
          type="button"
          className={styles.groupToggle}
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={isExpanded}
        >
          <span className={styles.icon} aria-hidden="true">
            {item.icon}
          </span>
          <span className={styles.label}>{item.label}</span>
          <span className={`${styles.chevron} ${isExpanded ? styles.expanded : ''}`} aria-hidden="true">
            ›
          </span>
        </button>
        {isExpanded && (
          <div className={styles.subNav}>
            {(item.children ?? []).map((child) => (
              <NavItem key={child.key} item={child} collapsed={false} />
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <NavLink
      to={item.route}
      className={({ isActive }) => `${styles.link} ${isActive ? styles.active : ''}`}
      title={collapsed ? item.label : undefined}
    >
      {item.icon && (
        <span className={styles.icon} aria-hidden="true">
          {item.icon}
        </span>
      )}
      {!collapsed && <span className={styles.label}>{item.label}</span>}
    </NavLink>
  );
}
