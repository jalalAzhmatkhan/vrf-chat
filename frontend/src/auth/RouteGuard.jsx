import { useLocation } from 'react-router-dom';
import { useAuth } from './AuthContext';
import { flattenRouteRequirements, hasRequiredScopes } from './filterNavByScopes';
import { ForbiddenPage } from './ForbiddenPage';

/**
 * Defense-in-depth layer on top of menu filtering (§1 "konsekuensi
 * implementasi") — protects direct URL access to scope-gated routes.
 * Spec: Documentation/ui-ux-design/03-app-shell-navigation.md §1, §6, §8.
 */
export function RouteGuard({ children }) {
  const location = useLocation();
  const { scopes } = useAuth();

  const requirement = flattenRouteRequirements().find((r) => r.route === location.pathname);

  // Routes not present in NAV_TREE (shouldn't normally happen once all
  // routes are registered) are treated as open — the route table itself is
  // the single source of truth per navConfig.js.
  if (!requirement) return children;

  const allowed = hasRequiredScopes(
    { requiredScopes: requirement.requiredScopes, match: requirement.match },
    scopes,
  );

  return allowed ? children : <ForbiddenPage />;
}
