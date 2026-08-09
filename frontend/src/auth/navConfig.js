/**
 * Nav tree definition — single source of truth for both the Sidebar (menu
 * visibility) and RouteGuard (URL access), per
 * Documentation/ui-ux-design/03-app-shell-navigation.md §3, §8.
 *
 * `requiredScopes` + `match`:
 *  - match: 'any' (default) -> user needs at least one of requiredScopes.
 *  - match: 'all' -> user needs every scope listed.
 * Group items (with `children`, no `route`) have no requiredScopes of their
 * own in this config — visibility is derived from whether any child is
 * visible (see filterNavByScopes.js), per §3 "Grup RBAC tampil jika salah
 * satu anaknya tampil".
 */

export const NAV_TREE = [
  {
    key: 'chat',
    label: 'Chat',
    route: '/chat',
    icon: '▣',
    requiredScopes: ['chat:read', 'chat:write'],
    match: 'any',
  },
  {
    key: 'admin',
    label: 'Admin',
    isGroup: true,
    isSectionHeader: true,
    children: [
      {
        key: 'admin-eval-metrics',
        label: 'Evaluation Metrics',
        route: '/admin/metrics',
        icon: '▤',
        requiredScopes: ['admin:eval:read'],
      },
      {
        key: 'admin-review-queue',
        label: 'Manual Review Queue',
        route: '/admin/review-queue',
        icon: '▥',
        requiredScopes: ['admin:review:read'],
      },
      {
        key: 'admin-rbac',
        label: 'RBAC',
        icon: '▼',
        isGroup: true,
        isCollapsibleSubNav: true,
        children: [
          {
            key: 'admin-rbac-permissions',
            label: 'Permission Management',
            route: '/admin/rbac/permissions',
            requiredScopes: ['admin:rbac:read'],
          },
          {
            key: 'admin-rbac-users',
            label: 'User Management',
            route: '/admin/rbac/users',
            requiredScopes: ['admin:rbac:read'],
          },
        ],
      },
    ],
  },
];
