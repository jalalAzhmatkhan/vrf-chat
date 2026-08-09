import { describe, expect, it } from 'vitest';
import { filterNavByScopes, flattenRouteRequirements, hasRequiredScopes } from './filterNavByScopes';

const TREE = [
  { key: 'chat', label: 'Chat', route: '/chat', requiredScopes: ['chat:read', 'chat:write'], match: 'any' },
  {
    key: 'admin',
    label: 'Admin',
    isGroup: true,
    isSectionHeader: true,
    children: [
      { key: 'eval', label: 'Evaluation Metrics', route: '/admin/metrics', requiredScopes: ['admin:eval:read'] },
      {
        key: 'rbac',
        label: 'RBAC',
        isGroup: true,
        isCollapsibleSubNav: true,
        children: [
          { key: 'perm', label: 'Permission Management', route: '/admin/rbac/permissions', requiredScopes: ['admin:rbac:read'] },
          { key: 'users', label: 'User Management', route: '/admin/rbac/users', requiredScopes: ['admin:rbac:read'] },
        ],
      },
    ],
  },
];

describe('filterNavByScopes', () => {
  it('shows only Chat for a plain "user" role (no admin scopes)', () => {
    const result = filterNavByScopes(['chat:read', 'chat:write', 'documents:read'], TREE);
    expect(result).toHaveLength(1);
    expect(result[0].key).toBe('chat');
  });

  it('hides the entire Admin group header when the user has zero admin:* scopes (hide-total, not disabled)', () => {
    const result = filterNavByScopes(['chat:read'], TREE);
    expect(result.some((item) => item.key === 'admin')).toBe(false);
  });

  it('shows Admin group with only the children the user has scope for', () => {
    const result = filterNavByScopes(['chat:read', 'chat:write', 'admin:eval:read'], TREE);
    const admin = result.find((item) => item.key === 'admin');
    expect(admin).toBeDefined();
    expect(admin.children.map((c) => c.key)).toEqual(['eval']);
  });

  it('shows the RBAC sub-group only when at least one of its children is visible', () => {
    const result = filterNavByScopes(['chat:read', 'admin:rbac:read'], TREE);
    const admin = result.find((item) => item.key === 'admin');
    const rbac = admin.children.find((c) => c.key === 'rbac');
    expect(rbac).toBeDefined();
    expect(rbac.children.map((c) => c.key)).toEqual(['perm', 'users']);
  });

  it('shows every item for an admin role with all 10 seeded scopes', () => {
    const allScopes = [
      'chat:read',
      'chat:write',
      'documents:read',
      'documents:write',
      'admin:eval:read',
      'admin:eval:write',
      'admin:review:read',
      'admin:review:write',
      'admin:rbac:read',
      'admin:rbac:write',
    ];
    const result = filterNavByScopes(allScopes, TREE);
    expect(result.map((i) => i.key)).toEqual(['chat', 'admin']);
    const admin = result.find((item) => item.key === 'admin');
    expect(admin.children.map((c) => c.key)).toEqual(['eval', 'rbac']);
  });

  it('returns an empty tree for a user with no scopes at all', () => {
    expect(filterNavByScopes([], TREE)).toEqual([]);
  });
});

describe('hasRequiredScopes', () => {
  it('matches "any" semantics by default', () => {
    expect(hasRequiredScopes({ requiredScopes: ['a', 'b'] }, ['b'])).toBe(true);
    expect(hasRequiredScopes({ requiredScopes: ['a', 'b'] }, ['c'])).toBe(false);
  });

  it('matches "all" semantics when specified', () => {
    expect(hasRequiredScopes({ requiredScopes: ['a', 'b'], match: 'all' }, ['a', 'b'])).toBe(true);
    expect(hasRequiredScopes({ requiredScopes: ['a', 'b'], match: 'all' }, ['a'])).toBe(false);
  });

  it('allows items with no requiredScopes through unconditionally', () => {
    expect(hasRequiredScopes({ requiredScopes: [] }, [])).toBe(true);
  });
});

describe('flattenRouteRequirements', () => {
  it('produces one entry per leaf route, skipping groups', () => {
    const flat = flattenRouteRequirements(TREE);
    expect(flat.map((r) => r.route).sort()).toEqual(
      ['/admin/metrics', '/admin/rbac/permissions', '/admin/rbac/users', '/chat'].sort(),
    );
  });
});
