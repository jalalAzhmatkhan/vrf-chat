import { NAV_TREE } from './navConfig';

/**
 * Single source-of-truth filter util — Documentation/ui-ux-design/
 * 03-app-shell-navigation.md §8: "single source of truth ada di satu util
 * filterNavByScopes(scopes), Sidebar tidak melakukan filtering sendiri".
 *
 * Items/groups the user doesn't have access to are OMITTED entirely (never
 * returned as `disabled`), per §1 "menu admin disembunyikan total".
 *
 * @param {string[]} scopes
 * @param {Array} [tree] defaults to the app's NAV_TREE — overridable for tests.
 * @returns {Array} filtered nav tree
 */
export function filterNavByScopes(scopes, tree = NAV_TREE) {
  return filterItems(tree, new Set(scopes ?? []));
}

function filterItems(items, scopeSet) {
  const result = [];
  for (const item of items) {
    if (item.isGroup) {
      const children = filterItems(item.children ?? [], scopeSet);
      if (children.length > 0) {
        result.push({ ...item, children });
      }
      continue;
    }
    if (hasRequiredScopes(item, scopeSet)) {
      result.push(item);
    }
  }
  return result;
}

export function hasRequiredScopes(item, scopeSet) {
  const required = item.requiredScopes ?? [];
  if (required.length === 0) return true;
  const set = scopeSet instanceof Set ? scopeSet : new Set(scopeSet ?? []);
  return item.match === 'all' ? required.every((s) => set.has(s)) : required.some((s) => set.has(s));
}

/**
 * Flat list of {route, requiredScopes, match} for every leaf item — used by
 * RouteGuard to look up a route's requirement without walking the tree at
 * render time.
 */
export function flattenRouteRequirements(tree = NAV_TREE) {
  const out = [];
  const walk = (nodes) => {
    for (const node of nodes) {
      if (node.isGroup) {
        walk(node.children ?? []);
      } else if (node.route) {
        out.push({
          route: node.route,
          label: node.label,
          requiredScopes: node.requiredScopes ?? [],
          match: node.match ?? 'any',
        });
      }
    }
  };
  walk(tree);
  return out;
}
