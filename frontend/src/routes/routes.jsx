import { Navigate, Route, Routes } from 'react-router-dom';
import { AppShell } from '../shell/AppShell';
import { LoginPage } from '../pages/LoginPage/LoginPage';
import { PlaceholderPage } from '../pages/PlaceholderPage';
import { ChatPage } from '../features/chat/ChatPage';

/**
 * Route shell for Fase 0 — chat/admin routes are placeholders (real content
 * lands in later phases), but scope-gating (RouteGuard, inside AppShell) is
 * already wired per navConfig.js so the auth/RBAC foundation doesn't need
 * rework later.
 *
 * Note there's no separate "ProtectedRoute" wrapper: AppShell itself calls
 * GET /auth/me on mount, and the shared HTTP client's refresh-on-401 flow
 * (lib/httpClient.js) already redirects to /login whenever that call can't
 * be satisfied (no/expired access token AND no valid refresh cookie) — this
 * covers both "session expired mid-app" and "unauthenticated user navigated
 * straight to /chat" with the same code path.
 */
export function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />

      <Route element={<AppShell />}>
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/admin/metrics" element={<PlaceholderPage title="Evaluation Metrics" />} />
        <Route path="/admin/review-queue" element={<PlaceholderPage title="Manual Review Queue" />} />
        <Route path="/admin/rbac/permissions" element={<PlaceholderPage title="Permission Management" />} />
        <Route path="/admin/rbac/users" element={<PlaceholderPage title="User Management" />} />
      </Route>

      <Route path="/" element={<Navigate to="/chat" replace />} />
      <Route path="*" element={<Navigate to="/chat" replace />} />
    </Routes>
  );
}
